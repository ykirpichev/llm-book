# Optional accelerator examples

These examples accompany **Accelerator Ecosystems Beyond CUDA and NVIDIA**.
They are intentionally separate from the CPU test suite. As of September 13,
2026, the repository's validation environment has no supported GPU, TPU, or
Neuron runtime: source review and Python syntax checks do not establish device
compilation, correctness, or performance.

Use the [accelerator engine acceptance record](../../docs/accelerator-acceptance-template.md)
to compare an eventual candidate with its baseline. The commands below can add
device-operator evidence on a pinned target; they do not fill an engine or
deployment record by themselves.

## Triton RMSNorm

`triton_rmsnorm.py` supplies an original, forward-only kernel, a checked host
wrapper, and an on-device smoke-test sweep. It is not a production replacement
for a framework operator. Install compatible PyTorch and Triton builds following
the target's official documentation; do not assume a CPU PyTorch wheel or an
arbitrary pair of releases is sufficient.

Run an explicitly selected target-only check from the repository root on a
supported target:

```sh
python examples/accelerators/triton_rmsnorm.py --mode forward
python examples/accelerators/triton_rmsnorm.py --mode contract
python examples/accelerators/triton_rmsnorm.py --mode stream-ordering
```

`--mode all` runs all three. Every mode first emits a machine-copyable
`PROVENANCE` record with facts this process can observe: source revision from
`BOOK_SOURCE_REVISION` (or `unknown`), target identity, PyTorch/Triton build
information, the **selected mode**, and its **planned** fixture. Planned cases
are not results: each mode emits a separate `PASS`, `SKIP`, or `RESULT` line.
It never invents a driver version; save the target's driver/environment output
with the result. The forward sweep
compares bounded random and zero inputs with an FP32 PyTorch expression and
checks strided inputs and tail widths. It labels skipped BF16 distinctly from a
pass. Initial tolerances are not a universal model-level quality criterion.

`contract` samples the wrapper's rank, width, device/dtype, gradient, stride,
and epsilon rejection paths; it is representative coverage, not exhaustive API
validation. `stream-ordering` makes a new non-default producer wait on the
current entry stream, records an event, then makes another non-default kernel
stream wait on both dependencies. It keeps the involved tensors alive through
that stream, then synchronizes at a correctness boundary. These are target-only
device-operator checks, not timing, integrated-model, or service claims.

Set `BOOK_SOURCE_REVISION` to the revision being evaluated before invoking a
target mode if the source is not already archived with the captured output.

The wrapper accepts inference tensors only; it supplies no backward operator.
Inputs must have positive strides and finite values whose intermediate and
output values fit the chosen arithmetic/storage formats. That value-domain
contract is not checked with a synchronizing device scan at each launch.

The `torch.cuda` interface is also used by ROCm PyTorch. This does not prove
that the installed Triton backend supports every GPU or dtype.

### Required target evidence before adoption

The script's `PASS bounded-forward dtype=...` output is not completion of this list.
No item below has been recorded as passing on hardware in this repository.

- Save the provenance header, command, driver/environment record, and every
  skipped dtype with the numerical sweep. Add bounded adversarial scales and
  non-unit learned weights relevant to the model.
- Run the `contract` and `stream-ordering` modes on the pinned target. They do
  not replace target error inspection; a successful enqueue is not completed
  execution.
- Measure warm operator latency separately from compilation/allocation, then
  run the integrated model quality and serving SLO trial using the acceptance
  record above.

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
and target versions must be recorded before changing the source-only label. The
program prints a `PROVENANCE` header with the compiled `BOOK_SOURCE_REVISION`
(default `unknown`), HIP header/runtime versions, device properties, and fixed
fixture/tolerances. Record the `hipcc` version and driver/environment output
separately; the program does not infer either.

### Required HIP target evidence before adoption

- Record the `hipcc`/ROCm/device tuple and provenance header; compile and run
  the bounded FP32 fixture, preserving its CPU-comparison result and any error.
- Exercise a producer-event/consumer-wait case separately. The included stream
  orders its own copies, launch, and result copy; it is not cross-stream proof.
- Define and test the production non-finite/input-range/output-overflow policy.
  The fixed fixture is finite and bounded and cannot establish it.
- Run the integrated model quality and service-SLO replay with the acceptance
  record; a correct operator is not a serving-engine result.

The program is a correctness example, not a tuned kernel or latency benchmark.
Its `std::vector` host buffers are pageable, so calls named `hipMemcpyAsync` do
not establish host/device overlap. It uses traditional `hipMalloc`/`hipFree`,
not the stream-ordered allocation APIs; do not transplant its cleanup policy
into a different allocator without checking lifetime rules. It covers fixed
FP32 input on one device, not training, allocator policy, collectives, or
multi-device behavior.

## CPU planning exercises

These run without accelerator packages:

```sh
python -m examples.accelerator_portability
python -m unittest tests.test_accelerator_portability -v
```

The numerical oracle uses Python floats, not an emulation of FP32 reduction
order. Memory and padding calculations are explicit planning models, not engine
support checks or performance predictions.
