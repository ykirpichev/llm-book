# Round 2 — independent technical quality check

Reviewed the complete integrated accelerator chapter and its diff from
`a287725`, the CPU planner and tests, both optional target companions, their
README, and `docs/accelerator-acceptance-template.md`. This is a source and CPU
QC report dated September 13, 2026, not accelerator execution evidence or the
campaign's completion gate. No shared implementation files were edited.

## Verdict

Packages 1–3 pass the technical/source gate. Their new backend, quantization,
DCP, scheduler, and exercise claims are accurate under the stated scope. No
P0/P1 finding. Package 4 needs one P2 reporting correction before its automatic
provenance claim is accepted. Two small documentation precision items follow;
neither changes the kernel algorithm.

## Findings and exact acceptance

### Q1 — P2: target provenance describes the wrong mode's fixture

**Location at first inspection:**
`examples/accelerators/triton_rmsnorm.py:72–91,184`, and
`examples/accelerators/README.md:31–38`.

`provenance(device)` always prints forward-sweep rows `[0,1,7]`, all nine
widths, strided inputs/weights, and all three dtype tolerances. The selected
mode is absent. A `--mode stream-ordering` invocation actually uses contiguous
FP32 `(1,129)` input and a stride-one weight; `contract` performs different
metadata-rejection cases, predominantly at `(1,3)`. A saved provenance header
therefore appears to describe coverage that the selected run never attempted.
The forward-only `RESULT` record does not repair the other modes.

**Correction:** include the selected mode and distinguish its planned cases
from actual pass/skip results. Emit the appropriate fixture for each selected
mode, or clearly name a global catalogue as such and emit per-mode records.
For rejection cases, record the case labels rather than borrowed numerical
tolerances. Preserve `unknown` for unavailable provenance.

**Acceptance:** `contract` and `stream-ordering` output cannot be mistaken for
the forward width/dtype sweep; `all` describes all three groups; attempted,
skipped, and passed cases are distinguishable. Source/syntax review may close
the reporting design, but no generated output is to be presented as a GPU run
without actually executing it on a supported target.

### Q2 — P3: describe rejection coverage as representative, or expand it

**Location:** `examples/accelerators/triton_rmsnorm.py:102–116` and
`examples/accelerators/README.md:40–45` at first inspection.

The current cases do exercise the listed *categories*, but not all declared
rejection branches. Examples not exercised include zero width, invalid input
dtype (as distinct from a weight-dtype mismatch), weight rank/gradient, a GPU
input paired with CPU weights, and NaN epsilon. The current broad README is
defensible as a category summary; a later claim of exhaustive wrapper-contract
coverage would not be.

**Acceptance:** either explicitly call this a representative set and expose
its actual case labels, or add the missing host-validation cases. A second-GPU
device-mismatch case may be conditional with an explicit skip. This review
does not require negative strides that the chosen framework cannot construct,
nor a synchronizing device-value scan that the wrapper intentionally excludes.

### Q3 — P3: restore the shortened HIP launch's omitted local assumptions

**Location:** `manuscript/04_cuda.md:1627–1638` at first inspection.

The shorter source-only label is appropriate, but the prior definition of
`HIP_CHECK` and the explicit live-stream/positive-dimension assumptions were
removed rather than moved. The earlier reduction paragraph retains the exact
256-thread block, nonempty row, FP32-pointer, and positive-stride contract; the
launch should similarly explain its helper without requiring the reader to
open the companion file.

**Acceptance:** one brief adjacent prose clause states that the launch assumes
checked buffers, positive dimensions and a live stream, and that `HIP_CHECK`
reports an error and stops the test. Keep the short provenance label within
three lines of the code fence. No restoration of the long original label is
needed.

## Confirmed technical acceptance

- **DCP arithmetic and scope:** the assistant has 32 query heads and eight KV
  heads. TP16/DCP2 satisfies `TP > Hkv`, `DCP <= TP // Hkv`,
  `(Hq // Hkv) % DCP == 0`, and `TP % DCP == 0`. The prose accurately excludes
  attention DP for the cited path and distinguishes prefill group rules. The
  [Neuron 2.32 DCP design](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/vllm-neuron/docs/design/parallelism/dcp.html)
  supports the paired, interleaved decode mapping. CPU integer checks give
  16 ranks, 268,435,456 bytes/rank (256 MiB), and 4,294,967,296 bytes aggregate
  (4 GiB). The independent-CP helper remains a separate model. There is no
  model-support or physical-device-count guarantee hidden in the calculation.
- **Quantization and connector gates:** the 4 GiB-to-2 GiB calculation is
  explicitly an element-payload change, not total allocated memory or speed.
  Weight artifact, computation dtype, and KV storage are separated. The
  [Neuron feature guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/features-guide.html)
  supports FP8 KV storage followed by BF16 attention computation. The
  [`ROCM_ATTN` implementation](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/rocm_attn/)
  declares its connector capability false; the book correctly dates and
  confines that observation to the inspected backend, not all AMD serving.
- **Scheduler and exercises:** segmented prefill with separate phase batches
  is not confused with lack of continuous batching. The eight-decode replay
  asks for measurements rather than supplying invented timing. Updated
  solutions distinguish cold readiness from warm scheduling and independently
  distinguish TP×CP ranks from DCP-within-TP ranks. The handoff exercise remains
  250 MiB, 13.765625 ms including conversion, and 1.234375 ms remaining budget.
- **Shortened listing labels:** all manuscript provenance-proximity tests
  pass. Triton's wrapper contract remains beside its source; NKI's FP32,
  same-shaped SBUF input/weight assumptions moved into adjacent prose.
  Pallas's blocked coordinates, FP32 128 KiB input tile, and full-row dependency
  remain unchanged. Apart from Q3, no meaningful contract was lost.
- **Triton stream and lifetime review:** for the documented CLI entry on the
  default stream, the producer waits for prior allocation-stream work; its
  event is recorded after production; the consumer waits before launching;
  references and stream records retain tensors through consumer completion.
  The reference calculation follows that completion boundary. This matches
  [PyTorch stream semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)
  and the [stream API](https://docs.pytorch.org/docs/main/generated/torch.cuda.Stream_class.html).
  This is source-level acceptance, not a device scheduling or race test.
- **HIP review:** the original 256-thread reduction, actual-width denominator,
  strided FP32 loads, contiguous stores, finite-output comparison, and checked
  success-path synchronization are preserved. Host vectors outlive device
  buffers; cleanup is explicitly tied to traditional allocation.
  [HIP memory-management documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html)
  supports the stated `hipFree` synchronization behavior. The README correctly
  leaves cross-stream proof as a separate target task and does not infer
  overlap from pageable `hipMemcpyAsync` calls. HIP header/runtime versions are
  distinguished from the separately required compiler/driver record.
- **Acceptance template:** baseline/candidate identity, workload, quality,
  SLO, cold-path, capacity/cost, and raw evidence fields are explicit. Unmeasured
  and skipped target results cannot be filled from CPU validation. It does
  not turn a device-operator pass into an engine/deployment pass.

## Checks actually run

Using the bundled Python runtime:

1. `python3 -m unittest discover -s tests -v`: **103 tests passed**, including
   executable manuscript snippets and status-label checks.
2. `python3 -m examples.accelerator_portability`: completed with the documented
   RMSNorm, padding, KV-placement, and 960-call repayment results.
3. Python AST parsing of `triton_rmsnorm.py`: passed without importing PyTorch
   or Triton. This establishes syntax only.
4. Independent integer/float assertions reproduced the nested-DCP payload,
   one-byte KV payload, RMSNorm padded-divisor example, and handoff budget.

Not run: HIP compilation, Triton JIT compilation, any GPU mode, TPU/Pallas or
Neuron simulation/compilation, target profiling, model quality, serving replay,
or PDF visual inspection. Those are separate gates and are not reported as
passing here.

## Final round-2 closure

Independently re-inspected the fixes after author integration. Q1 is closed:
provenance now names the selected mode, marks its fixture as planned rather
than executed, and keeps pass/skip results separate. The stream description
accurately distinguishes the caller's origin stream from the newly created
producer and consumer. Q2 is closed: coverage is explicitly representative,
with additional width-zero, dtype, weight-rank/gradient, mixed-device,
zero-stride, and NaN-epsilon cases. Q3 is closed: adjacent manuscript prose
restores checked-buffer, positive-dimension, live-stream, and stop-on-error
assumptions without lengthening the provenance label.

The README's PASS label now matches the program. Waiting on the actual caller
current stream also removes the earlier source-review assumption that entry
must be on the default stream. No new lifetime/order problem was found.
The complete CPU suite was rerun: **103 tests passed**. Round 2 passes the
technical/source gate; accelerator compile/run and performance remain untested.
This closes the round, not the timed campaign.
