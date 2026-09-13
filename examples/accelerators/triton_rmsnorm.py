"""Forward-only teaching RMSNorm; requires PyTorch + Triton on a supported GPU.

Not GPU-compiled or GPU-executed in the book's CPU validation environment.
The command-line tests are an on-target validation starting point, not a speed
claim. The bounded numerical domain excludes overflowing squares/reductions.
"""
import math

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


def main():
    if not torch.cuda.is_available():
        raise SystemExit("A compatible GPU PyTorch/Triton installation is required.")
    torch.manual_seed(0)
    print("torch", torch.__version__, "triton", triton.__version__)
    print("device", torch.cuda.get_device_name(), "HIP", torch.version.hip)
    # Initial tolerances for these bounded test inputs; not a model quality gate.
    tolerances = {torch.float32: (2e-5, 2e-5), torch.float16: (2e-3, 2e-3),
                  torch.bfloat16: (2e-2, 2e-2)}
    for dtype, (rtol, atol) in tolerances.items():
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print("SKIP BF16: runtime reports unsupported")
            continue
        for rows in (0, 1, 7):
            for n in (1, 3, 127, 128, 129, 4095, 4096, 4097, 8192):
                storage = torch.randn((rows, 2*n+3), device="cuda", dtype=dtype)
                x = storage[:, 1:1+2*n:2]
                weight = torch.randn(2*n, device="cuda", dtype=dtype)[::2]
                for zero in (False, True):
                    if zero:
                        x.zero_()
                    xf, wf = x.float(), weight.float()
                    expected = (xf * torch.rsqrt((xf*xf).mean(-1, keepdim=True) + 1e-6) * wf).to(dtype)
                    actual = rmsnorm(x, weight)
                    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                    assert actual.is_contiguous()
        torch.cuda.synchronize()
        print("PASS bounded forward cases", dtype)


if __name__ == "__main__":
    main()
