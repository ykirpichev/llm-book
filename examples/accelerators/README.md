# Optional accelerator examples

These examples accompany **Accelerator Ecosystems Beyond CUDA and NVIDIA**.
They are intentionally separate from the CPU test suite. As of September 13,
2026, the repository's validation environment has no supported GPU, TPU, or
Neuron runtime: source review and Python syntax checks do not establish device
compilation, correctness, or performance.

## Triton RMSNorm

`triton_rmsnorm.py` supplies an original, forward-only kernel, a checked host
wrapper, and an on-device smoke-test sweep. It is not a production replacement
for a framework operator. Install compatible PyTorch and Triton builds following
the target's official documentation; do not assume a CPU PyTorch wheel or an
arbitrary pair of releases is sufficient.

Run the bounded forward smoke test from the repository root on a supported target:

```sh
python examples/accelerators/triton_rmsnorm.py
```

Record the printed versions and device, driver/runtime versions, test output,
and any skipped dtype. The script compares bounded random and zero inputs with
an FP32 PyTorch expression and checks strided inputs and tail widths. Its initial
tolerances are not a universal model-level quality criterion. Add adversarial
activation ranges, API rejection tests, concurrent-stream tests, and model
quality evaluation before adoption. No timing results are reported by the script.

The wrapper accepts inference tensors only; it supplies no backward operator.
Inputs must have positive strides and finite values whose intermediate and
output values fit the chosen arithmetic/storage formats. That value-domain
contract is not checked with a synchronizing device scan at each launch.

The `torch.cuda` interface is also used by ROCm PyTorch. This does not prove
that the installed Triton backend supports every GPU or dtype.

### Required target evidence before adoption

The script's `PASS bounded forward cases` output is not completion of this list.
No item below has been recorded as passing on hardware in this repository.

- Run the numerical sweep, document skipped dtypes, and add bounded adversarial
  scales and non-unit learned weights relevant to the model.
- Test rejection of invalid rank, width, device/dtype pairs, epsilon, zero
  strides, and tensors requiring gradients. CPU source parsing does not execute
  these device-wrapper branches.
- Test a non-default stream and an explicit producer-event/consumer-wait case;
  synchronize at a correctness boundary and keep referenced buffers alive.
- Check launch and asynchronous errors on the target. A successful enqueue is
  not completed execution.
- Measure warm operator latency separately from compilation/allocation, then
  run the integrated model quality and serving SLO trial.

## HIP reduction companion

`hip_rmsnorm.cpp` is an original FP32 source-only companion: one 256-thread
block owns each row, all threads participate in the shared reduction, and
the output pass uses the full-row inverse RMS. A complete host test gives
the allocation, copy, stream, launch, error-check, and lifetime context omitted
from the printed excerpt. See its header for the supported input contract.

On a compatible HIP development target, build and run from the repository root:

```sh
hipcc -O2 -std=c++17 examples/accelerators/hip_rmsnorm.cpp -o /tmp/book-hip-rmsnorm
/tmp/book-hip-rmsnorm
```

These commands have not been run here. Successful compilation, device outputs,
and target versions must be recorded before changing the source-only label.
The program is a correctness example, not a tuned kernel or latency benchmark.
Its `std::vector` host buffers are pageable, so calls named `hipMemcpyAsync`
do not establish host/device overlap. It uses traditional `hipMalloc`/`hipFree`,
not the stream-ordered allocation APIs; do not transplant its cleanup policy
into a different allocator without checking lifetime rules.

## CPU planning exercises

These run without accelerator packages:

```sh
python -m examples.accelerator_portability
python -m unittest tests.test_accelerator_portability -v
```

The numerical oracle uses Python floats, not an emulation of FP32 reduction
order. Memory and padding calculations are explicit planning models, not engine
support checks or performance predictions.
