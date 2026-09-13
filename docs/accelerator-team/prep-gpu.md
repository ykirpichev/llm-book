# GPU preparation brief: Triton, HIP, and ROCm

Date: September 13, 2026. Reviewer: Astra technical review. Baseline: `1fb029c`.
Scope: preparatory advice for chapter 36, **Accelerator Ecosystems Beyond CUDA
and NVIDIA**. The chapter and campaign log were read. This is not the formal
post-expansion review or a three-hour campaign completion report.

Evidence status: primary documentation checked online; the code below is an
original, reviewed teaching proposal, not GPU-compiled or GPU-executed code.
Only the small numerical calculations in section 4 were evaluated, using host
arithmetic. No manuscript, shared code, or diagrams were changed.

## 1. Recommended teaching contract

Develop one forward-only RMSNorm operation before comparing implementations:

`y[r,c] = x[r,c] * rsqrt(sum_j float32(x[r,j])^2 / N + eps) * w[c]`.

Use two-dimensional `x[M,N]`, one-dimensional `w[N]`, and fresh contiguous
output. Require `N > 0`, finite positive FP32-representable epsilon, supported
dtypes on one device, and valid nonnegative read strides. An especially simple
initial contract rejects zero strides and supports positive strides only.
Input and weight views may be noncontiguous; output never aliases either.
Specify that the implementation has no backward/autograd registration.

Cast **before squaring**, reduce in FP32, divide by **N**, and cast the final
output to its declared storage dtype. FP32 accumulation is not an overflow-proof
norm: finite FP32 inputs can still overflow when squared or summed. State the
bounded-activation domain; do not promise arbitrary-finite-input robustness.
Reject or explicitly define NaN/Inf behavior rather than silently sanitizing it.

Do not call this LayerNorm: RMSNorm does not subtract a mean. Do not imply
bitwise agreement across devices, compiler versions, reduction trees, or
reciprocal-square-root implementations.

## 2. Minimal Triton example proposal

Triton's [LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
is the closest primary starting point for row ownership, FP32 reduction, and
launch structure. The [softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
separately demonstrates independent input/output row strides and power-of-two
tiles with masked tails. Neither tutorial is evidence that this proposed
RMSNorm kernel has been tested or tuned.

```python
import triton
import triton.language as tl

@triton.jit
def rmsnorm_rows(X, W, Y, sx0, sx1, sw,
                 N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK).to(tl.int64)
    valid = col < N
    x = tl.load(X + row * sx0 + col * sx1,
                mask=valid, other=0.0).to(tl.float32)
    w = tl.load(W + col * sw, mask=valid, other=0.0).to(tl.float32)
    square_sum = tl.sum(x * x, axis=0, dtype=tl.float32)
    inv_rms = tl.rsqrt(square_sum / N + EPS)
    result = x * inv_rms * w
    tl.store(Y + row * N + col, result, mask=valid)
```

Host wrapper checklist, preferably implemented once in the optional script:

- Check `x.ndim == 2`, `w.ndim == 1`, matching width and device, dtype in
  FP16/BF16/FP32, the stated stride policy, and the inference-only contract.
- Allocate `torch.empty((M, N), device=x.device, dtype=x.dtype)`. Do not reuse
  the input row stride for this contiguous output.
- Use `BLOCK = triton.next_power_of_2(N)` and an explicit teaching limit such
  as `N <= 8192`; this is an implementation limit, not Triton's universal limit.
- Launch grid `(M,)`, passing element strides `x.stride(0)`, `x.stride(1)`,
  `w.stride(0)`. Return the empty output without launching when `M == 0`.
- Start with `num_warps=4`; label it a configuration to test, not a portable
  optimum. Tune separately by target, row width, row count, and dtype.
- Enter the input tensor's device context before launch. In ROCm PyTorch,
  `torch.cuda` and device type `"cuda"` remain the supported interfaces; a
  literal device type `"hip"` is not the port. See
  [PyTorch HIP semantics](https://docs.pytorch.org/docs/main/notes/hip.html).
- Validate epsilon after conversion to FP32, or use a documented conservative
  range. A positive Python float that rounds to FP32 zero does not protect an
  all-zero row. Avoid a GPU scalar extraction on every launch merely to check it.

The zero-filled invalid lanes contribute nothing to the squared sum. Mask the
weight load and output store too. Integer offsets are promoted before stride
multiplication to avoid accidental 32-bit address-index overflow. The output
pointer's dtype controls the store conversion. These operations are documented
by [`tl.load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html),
[`tl.arange`](https://triton-lang.org/main/python-api/generated/triton.language.arange.html),
[`tl.sum`](https://triton-lang.org/main/python-api/generated/triton.language.sum.html),
and [`tl.rsqrt`](https://triton-lang.org/main/python-api/generated/triton.language.rsqrt.html).

The power-of-two tile is a logical tensor, not a count of launched scalar
threads. Increasing from width 4096 to 4097 doubles this example's tile to 8192;
register pressure and occupancy may change abruptly. A multi-pass or differently
partitioned kernel can be preferable for large widths. General strided reads
establish semantic support, not coalesced access or competitive performance.

## 3. HIP baseline proposal: make synchronization visible

Use a deliberately simple FP32-only baseline with one block per row and exactly
256 threads. This compares the same mathematical operation without burying the
lesson in half-type conversions or target-specific shuffle code. It rereads X
for the output pass; do not describe its byte traffic as identical to the
single-load Triton proposal.

```cpp
#include <hip/hip_runtime.h>
#include <cstdint>

// Teaching contract: blockDim=(256,1,1), one block per valid row;
// n>0, positive element strides, disjoint valid allocations, FP32 x/w/y.
__global__ void rmsnorm_f32(const float* x, const float* w, float* y,
                            int64_t n, int64_t sx0, int64_t sx1,
                            int64_t sw, float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const unsigned tid = threadIdx.x;
    __shared__ float sums[256];
    float local = 0.0f;
    for (int64_t c = tid; c < n; c += 256) {
        const float v = x[row * sx0 + c * sx1];
        local += v * v;
    }
    sums[tid] = local;
    __syncthreads();
    for (unsigned step = 128; step != 0; step >>= 1) {
        if (tid < step) sums[tid] += sums[tid + step];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0] / static_cast<float>(n) + eps);
    for (int64_t c = tid; c < n; c += 256) {
        y[row * n + c] = x[row * sx0 + c * sx1] * inv * w[c * sw];
    }
}
```

No thread exits early: even threads with no elements contribute zero and reach
every block barrier. The reduction is independent of hardware warp/wave width.
Production alternatives include a supported block-reduction primitive and
specialized vector loads; those require their own tests and measurements.

Suggested host launch excerpt, not a complete allocation/cleanup program:

```cpp
// Assume checked nonzero rows, n, strides, epsilon, device, allocation sizes,
// and device grid/block limits. HIP_CHECK throws or terminates on an error.
hipLaunchKernelGGL(rmsnorm_f32, dim3(rows), dim3(256), 0, stream,
                   d_x, d_w, d_y, n, sx0, sx1, sw, eps);
HIP_CHECK(hipGetLastError());       // Check enqueue/launch errors.
HIP_CHECK(hipStreamSynchronize(stream)); // Test boundary, not hot-path policy.
```

Check every allocation/copy/event API as well. A successful launch check does not
prove that asynchronous execution completed. Keep inputs, output, and any host
transfer buffers alive until their consumers finish. Use explicit event ordering
when a different stream produces or consumes a tensor; same-device residency
does not establish a dependency. The [HIP execution model](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/programming_manual.html),
[stream/event guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html),
and [error-handling reference](https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.0/how-to/hip_runtime_api/error_handling.html)
support this workflow. Synchronize at correctness/measurement boundaries, not
after every production operation.

Porting traps worth a compact callout:

- Query warp/wave size instead of substituting every `32` with `64`. HIP's
  ballot/active-mask and `_sync` mask types are unsigned 64-bit even for
  32-lane targets. Active participants must obey the documented mask contract.
- HIP's `_sync` warp builtins became enabled by default in ROCm 7.0. That
  release also changed `warpSize` from the previous compile-time treatment to
  compiler early-folding: do not assume it is valid in every constexpr/template
  use. See [HIP language extensions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html).
- HIP 7.0 changed last-error behavior, removed the implicit `hipFree` wait for
  allocations created with asynchronous allocation APIs, and separated hipRTC
  symbols from the runtime library. Rebuild and audit lifetime/link assumptions
  across that boundary. See [versioned HIP 7.0 changes](https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.1/hip-7-changes.html).
- Inline PTX, matrix instructions, library calls, and layout assumptions need
  an implementation path, not merely renamed syntax. Keep this list separate
  from the baseline kernel's mathematical correctness. The
  [versioned porting guide](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/hip_porting_guide.html)
  also distinguishes NVIDIA PTX/cubin objects from AMD hsaco objects: a
  translated host API does not make a compiled kernel binary portable.

## 4. Small worked examples and acceptance tests

1. **Tail denominator.** For `x=[3,4,0]`, unit weights, and `eps=1e-6`, the sum
   of squares is 25 and the correct output is approximately
   `[1.0392304,1.3856406,0]`. With a four-lane tile, dividing by BLOCK instead
   of N incorrectly gives `[1.1999999,1.5999999,0]`. These are host-arithmetic
   illustrations, not claimed GPU outputs.
2. **Stride trap.** Use an input view such as `storage[:, 1:1+2*N:2]` where
   `storage.shape == (M, 2*N+3)`, plus a strided weight view. Row stride is
   `2*N+3`, column stride is 2, and output row stride is N. A kernel supporting
   only the first stride is still wrong.
3. **Barrier trap.** For N=3 and a 256-thread HIP block, 253 threads load no
   input but must still take part in the reduction barriers. `if (tid >= n)
   return;` is not a valid shortcut here.
4. **Tail/resource sweep.** Test widths 1, 3, 127, 128, 129, 4095, 4096, 4097,
   and 8192, with zero, one, and many rows; padded-row and strided-column views;
   zero inputs; mixed signs; and representative large/small activation scales.
   Test rejected rank/dtype/device/epsilon/width cases as part of the API.
5. **Numerical reference.** Compare against FP64 host arithmetic for mathematical
   error and against an explicit FP32 framework expression for implementation
   agreement. Cast to the requested output dtype at the reference's end. Set
   dtype-aware absolute and relative tolerances from evidence; do not claim one
   universal threshold. Training support additionally needs gradient tests.

## 5. Supported profiling workflow and source/version caveats

Begin with unprofiled correctness and warm latency measurements. Separate JIT
compilation and allocation from the timed kernel. Record shape/stride/dtype,
target architecture, compiler and framework versions, clocks/power policy,
warm-up, repeats, and output checks. Triton's
[`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)
documents warm-up, repetition, and quantile parameters; it is not an end-to-end
serving benchmark.

For current ROCm development, use ROCprofiler-SDK/`rocprofv3` rather than teaching
legacy `rocprof` or `rocprofv2` as the default. AMD explicitly marks the older
stack deprecated in the [legacy profiler notice](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/).
Do not infer a confirmed end-of-support date merely from its forecast language.

Suggested commands to show as an on-target workflow, not as commands run here:

```sh
rocprofv3 --version
rocprofv3 --help
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace -- ./rmsnorm_bench
rocprofv3-avail list --pmc
rocprof-compute profile -n rmsnorm -- ./rmsnorm_bench
```

Then run `rocprof-compute analyze -p` with the actual generated workload/SoC
directory. First use the trace to separate host gaps, copies, and dispatches;
then restrict counter collection to the relevant kernel and supported counters.
Check counter compatibility with `rocprofv3-avail pmc-check` before requesting a
group. These commands are grounded in the
[SDK quick reference](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/quick_guide.html),
[Compute Profiler profile guide](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html),
and [CLI analysis guide](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/analyze/cli.html).

Do not treat counter-collection elapsed time as ordinary application latency:
profiling may replay execution or perturb scheduling. Start with the isolated,
side-effect-free normalization harness. Current Compute Profiler documentation
describes application replay by default and separate single-pass options for
MPI; avoid wrapping a stateful distributed service in a copied single-process
profiling command. Counter availability and roofline metrics depend on the
hardware and profiler release.

The pages served during this check had different component version labels
(including HIP 7.15.0, SDK 1.3.5, and Compute Profiler 3.8.0); these are not a
validated combined installation. Pin documentation to the deployed component
versions and verify commands with installed `--help`. Avoid adopting `develop`
features, beta PC sampling, or experimental kernel replay as universal defaults.
The chapter should say what evidence a reader must collect, not imply these
snippets certify cross-vendor numerical or performance portability.
