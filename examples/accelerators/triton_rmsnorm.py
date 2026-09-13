"""Forward-only teaching RMSNorm; requires PyTorch + Triton on a supported GPU.

Not GPU-compiled or GPU-executed in the book's CPU validation environment.
The command-line tests are an on-target validation starting point, not a speed
claim. The bounded numerical domain excludes overflowing squares/reductions.
"""
import argparse
import json
import math
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_rows(X, W, Y, sx0, sx1, sw, N: tl.constexpr,
                  EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK).to(tl.int64)
    valid = col < N
    x = tl.load(X + row * sx0 + col * sx1, mask=valid, other=0).to(tl.float32)
    w = tl.load(W + col * sw, mask=valid, other=0).to(tl.float32)
    square_sum = tl.sum(x * x, axis=0)
    y = x * tl.rsqrt(square_sum / N + EPS) * w
    tl.store(Y + row * N + col, y, mask=valid)


def rmsnorm(x, weight, epsilon=1e-6):
    """Fresh contiguous output; positive input strides; no autograd support.

    Teaching limits: width 1..8192, epsilon 1e-12..1e-2, matched FP16/BF16/FP32
    inputs. Finite bounded activations and weights are the caller's contract;
    scanning values here would add device work and synchronization to every call.
    """
    if x.ndim != 2 or weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("expected x[M,N] and weight[N]")
    if x.device.type != "cuda" or weight.device != x.device:
        raise ValueError("inputs must share a supported GPU device")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32) or weight.dtype != x.dtype:
        raise ValueError("matched FP16, BF16, or FP32 inputs required")
    if x.requires_grad or weight.requires_grad:
        raise ValueError("this example implements forward inference only")
    if not 1 <= x.shape[1] <= 8192:
        raise ValueError("teaching kernel supports widths 1..8192")
    if any(stride <= 0 for stride in (*x.stride(), *weight.stride())):
        raise ValueError("positive element strides required")
    if not math.isfinite(epsilon) or not 1e-12 <= epsilon <= 1e-2:
        raise ValueError("teaching epsilon range is 1e-12..1e-2")
    with torch.cuda.device(x.device):
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if x.shape[0]:
            _rmsnorm_rows[(x.shape[0],)](
                x, weight, out, *x.stride(), weight.stride(0), x.shape[1],
                epsilon, triton.next_power_of_2(x.shape[1]), num_warps=4)
    return out


WIDTHS = (1, 3, 127, 128, 129, 4095, 4096, 4097, 8192)
TOLERANCES = {
    torch.float32: (2e-5, 2e-5),
    torch.float16: (2e-3, 2e-3),
    torch.bfloat16: (2e-2, 2e-2),
}


def _dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def planned_fixture(mode):
    """Describe planned cases only; results are emitted by each mode afterward."""
    plans = {
        "forward": {
            "rows": [0, 1, 7],
            "widths": list(WIDTHS),
            "x_strides": ["2*N+3", 2],
            "weight_stride": 2,
            "epsilon": 1e-6,
            "dtypes": [_dtype_name(dtype) for dtype in TOLERANCES],
        },
        "contract": {
            "coverage": "representative wrapper rejections; not exhaustive API validation",
            "categories": [
                "x rank", "weight rank/width", "mixed CPU/GPU device", "unsupported/mismatched dtype",
                "width zero/limit", "x/weight requires_grad", "x/weight zero stride",
                "epsilon range/nonfinite",
            ],
        },
        "stream-ordering": {
            "rows": 1,
            "width": 129,
            "dtype": "float32",
            "producer": "new non-default stream waiting on caller current/origin stream",
            "consumer": "another new non-default stream waiting on origin and producer event",
            "dependency": "origin stream -> producer event -> consumer wait",
        },
    }
    if mode == "all":
        return {name: plans[name] for name in ("forward", "contract", "stream-ordering")}
    return plans[mode]


def provenance(device, mode):
    """Emit facts this process can observe; never infer an unavailable driver."""
    properties = torch.cuda.get_device_properties(device)
    record = {
        "device": properties.name,
        "device_index": device.index,
        "device_capability": f"{properties.major}.{properties.minor}",
        "gcn_arch_name": getattr(properties, "gcnArchName", None),
        "source_revision": os.environ.get("BOOK_SOURCE_REVISION", "unknown"),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_hip_build": torch.version.hip,
        "selected_mode": mode,
        "planned_fixture": planned_fixture(mode),
        "tolerances": {_dtype_name(dtype): {"rtol": rtol, "atol": atol}
                       for dtype, (rtol, atol) in TOLERANCES.items()},
    }
    print("PROVENANCE", json.dumps(record, sort_keys=True, default=str))


def expect_value_error(label, thunk):
    try:
        thunk()
    except ValueError:
        return
    raise AssertionError(f"{label} was accepted")


def run_contract_cases(device):
    """Representative target-only wrapper rejections, not exhaustive API validation."""
    x = torch.ones((1, 3), device=device)
    weight = torch.ones(3, device=device)
    expect_value_error("rank", lambda: rmsnorm(x[0], weight))
    expect_value_error("weight rank", lambda: rmsnorm(x, weight.unsqueeze(0)))
    expect_value_error("width mismatch", lambda: rmsnorm(x, weight[:2]))
    expect_value_error("width zero", lambda: rmsnorm(torch.ones((1, 0), device=device),
                                                       torch.ones(0, device=device)))
    expect_value_error("dtype mismatch", lambda: rmsnorm(x, weight.half()))
    expect_value_error("unsupported x dtype", lambda: rmsnorm(
        torch.ones((1, 3), device=device, dtype=torch.int32),
        torch.ones(3, device=device, dtype=torch.int32)))
    expect_value_error("mixed CPU/GPU device", lambda: rmsnorm(x, torch.ones(3)))
    expect_value_error("width limit", lambda: rmsnorm(torch.ones((1, 8193), device=device),
                                                        torch.ones(8193, device=device)))
    expect_value_error("requires_grad", lambda: rmsnorm(x.detach().requires_grad_(), weight))
    expect_value_error("weight requires_grad", lambda: rmsnorm(x, weight.detach().requires_grad_()))
    expect_value_error("x zero stride", lambda: rmsnorm(
        torch.ones((1, 1), device=device).expand(1, 3), weight))
    expect_value_error("zero stride", lambda: rmsnorm(x, torch.ones(1, device=device).expand(3)))
    for epsilon in (0, 1e-13, 1e-1, math.inf, math.nan):
        expect_value_error(f"epsilon={epsilon}", lambda epsilon=epsilon: rmsnorm(x, weight, epsilon))
    print("PASS representative-contract cases")


def run_stream_ordering_case(device):
    """Exercise producer event -> kernel stream ordering on bounded FP32 input."""
    width = 129
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    source = torch.empty((1, width), device=device, dtype=torch.float32)
    weight = torch.linspace(0.25, 0.75, width, device=device, dtype=torch.float32)
    ready = torch.cuda.Event()
    origin_stream = torch.cuda.current_stream(device)
    producer.wait_stream(origin_stream)
    with torch.cuda.stream(producer):
        source.copy_(torch.linspace(-1, 1, width, device=device).reshape(1, width))
        source.record_stream(producer)
        ready.record(producer)
    consumer.wait_stream(origin_stream)
    consumer.wait_event(ready)
    with torch.cuda.stream(consumer):
        actual = rmsnorm(source, weight)
        source.record_stream(consumer)
        weight.record_stream(consumer)
        actual.record_stream(consumer)
    consumer.synchronize()
    expected = (source * torch.rsqrt((source * source).mean(-1, keepdim=True) + 1e-6)
                * weight)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    print("PASS stream-ordering FP32 width=129 origin=current-stream")


def run_forward_cases(device):
    skipped = []
    attempted = []
    for dtype, (rtol, atol) in TOLERANCES.items():
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            skipped.append(_dtype_name(dtype))
            print("SKIP forward dtype=bf16 reason=runtime-reports-unsupported")
            continue
        attempted.append(_dtype_name(dtype))
        for rows in (0, 1, 7):
            for n in WIDTHS:
                storage = torch.randn((rows, 2*n+3), device=device, dtype=dtype)
                x = storage[:, 1:1+2*n:2]
                weight = torch.randn(2*n, device=device, dtype=dtype)[::2]
                for zero in (False, True):
                    if zero:
                        x.zero_()
                    xf, wf = x.float(), weight.float()
                    expected = (xf * torch.rsqrt((xf*xf).mean(-1, keepdim=True) + 1e-6) * wf).to(dtype)
                    actual = rmsnorm(x, weight)
                    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                    assert actual.is_contiguous()
        torch.cuda.synchronize(device)
        print("PASS bounded-forward dtype=" + _dtype_name(dtype))
    print("RESULT", json.dumps({"attempted_dtypes": attempted, "skipped_dtypes": skipped},
                                sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Target-only Triton RMSNorm checks")
    parser.add_argument("--mode", choices=("forward", "contract", "stream-ordering", "all"),
                        default="forward", help="target-only check group to run")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("A compatible GPU PyTorch/Triton installation is required.")
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    provenance(device, args.mode)
    if args.mode in ("forward", "all"):
        run_forward_cases(device)
    if args.mode in ("contract", "all"):
        run_contract_cases(device)
    if args.mode in ("stream-ordering", "all"):
        run_stream_ordering_case(device)


if __name__ == "__main__":
    main()
