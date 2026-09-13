# HIP RMSNorm source companion

`examples/accelerators/hip_rmsnorm.cpp` is an original, source-only teaching
program for the HIP RMSNorm discussion. It has not been compiled or run in
this repository's validation environment; neither source review nor a future
successful build would establish numerical, performance, engine, or model
portability.

## Contract demonstrated

- FP32 forward RMSNorm only: `y = x * rsqrt(mean(x*x) + epsilon) * weight`.
  It has no backward pass, autograd interface, quantization, or in-place path.
- One **256-thread block** owns one nonempty row. Every thread participates in
  all barriers; short rows contribute a zero local sum instead of returning
  early. The divisor is the actual `N`, never a block/tile width.
- Input and weight offsets use signed 64-bit element-stride arithmetic:
  `x[row * sx0 + column * sx1]`, `weight[column * sw]`. Output is newly
  allocated and contiguous: `y[row * N + column]`.
- The fixed host cases use three rows at each of `N=3`, `129`, and `4097`:
  a zero row plus bounded mixed-sign rows, a strided input view
  (`sx0 = 2*N+3`, `sx1 = 2`), and strided weights (`sw = 2`). A double CPU
  reference compares each returned output with initial absolute/relative
  tolerances.
- Allocation, asynchronous copies, kernel-launch status, and the stream
  synchronization are checked. RAII keeps buffers and the non-default stream
  alive through all queued work. It demonstrates ordering within one stream;
  production cross-stream code still needs an explicit event/dependency.

## Build and run on a qualified HIP target

From the repository root, with a ROCm/HIP installation compatible with the
chosen GPU and host toolchain:

```sh
hipcc -O2 -std=c++17 examples/accelerators/hip_rmsnorm.cpp -o hip_rmsnorm
./hip_rmsnorm
```

Record the GPU, ROCm/runtime, compiler, driver, command output, and any
target-specific architecture flags required by that installation. A passing
run only establishes these bounded forward cases on that pinned stack.

## Deliberate limits

The program does not validate all API error paths, empty rows, negative or
zero strides, non-finite values, output overflow, multiple devices, graph
capture, concurrent streams, memory-pool behavior, reductions beyond one
256-thread block, or a serving engine's KV/cache/collective behavior. It
reruns the input load for the output pass and is not a performance baseline.
FP32 squares, partial sums, and final output can overflow outside the bounded
test domain. For a real port, compare against the framework's specified
accumulation/output dtype semantics, add adversarial numerical tests and
launch/error/lifetime coverage, then measure integrated prefill and decode
under the target workload.
