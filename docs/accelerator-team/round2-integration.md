# Accelerator portability — round 2 reproducibility and integration review

**Scope.** Independent reader-path review from baseline `a287725` of the current chapter, `examples/accelerator_portability.py`, its tests, `examples/accelerators/{README.md,triton_rmsnorm.py,hip_rmsnorm.cpp}`, and the top-level examples index. Accelerator hardware, HIP compilation, Triton compilation, JAX/Pallas execution, and a serving engine were unavailable and were not attempted.

## Result

The CPU path is now genuinely discoverable and runnable: the top-level index links to it, the direct module command works, and the output accurately calls the result modeled physical KV rather than engine memory. The optional paths also have unusually clear source-versus-device labels. A reader cannot mistake the repository's CPU evidence for a GPU/TPU/Neuron result.

No P0 issue was found. The bounded improvements below would make the next target run auditable and turn a device result into a useful deployment decision, rather than adding another hardware dependency or pretending that static checks establish execution.

## Findings and bounded improvements

### P1 — Provide one check-in engine acceptance report, with a baseline decision rule

**Location:** `manuscript/04_cuda.md:1818-1851`, especially the evidence table at `:1844-1849`.

The chapter lists the right ingredients (same model/features, quality, arrival replay, SLO tails, cold readiness, memory, and cost), but it does not give a single report shape that forces a candidate and baseline to use the same denominator. A target team can therefore produce attractive kernel or warm-decode numbers while silently changing a cache format, attention feature, traffic mix, retry accounting, or accepted concurrency.

**Smallest fix:** add one short Markdown template, for example `docs/accelerator-team/engine-acceptance-template.md`, and link it from the chapter. It should require one row per candidate *and baseline* containing:

| Required field group | Minimum record |
| --- | --- |
| Identity | model/checkpoint/tokenizer; engine + backend + compiler/runtime; GPU/TPU/Neuron target; TP/CP/PP; KV dtype/layout/quantization; enabled attention, sampling, and paging features |
| Workload | identical prompt/output cohorts, arrival process, offered load, cancellation/retry policy, warm/cold mix, and run duration |
| Correctness | logits or justified tolerance check; task-quality gate; failures/rejections; cache-ABI conversion status |
| Service | p50/p95/p99 TTFT and token gap by cohort; goodput; accepted concurrency; queue time; cold-ready time; compile/cache misses |
| Capacity/cost | peak memory on the constraining rank, host/network resources, SLO-qualified requests, and cost per SLO-qualified request |
| Decision | pass only if the candidate preserves required features and quality and meets the stated SLO at the same offered load; otherwise record no-go/fallback and the limiting evidence |

This is a fixed reporting contract, not a benchmark claim or an engine support matrix. It is the smallest artifact that connects the chapter's operator and planning examples to a deploy/no-deploy decision.

### P2 — Make target-run provenance automatic, not a reminder readers must reconstruct

**Location:** `examples/accelerators/README.md:17-28,63-76`, `examples/accelerators/triton_rmsnorm.py:57-84`, and `examples/accelerators/hip_rmsnorm.cpp:195-210`.

The README rightly asks the reader to record versions, target, and skips. Triton prints PyTorch/Triton/device/HIP but not a complete runtime/compiler identity; the HIP program prints only pass lines. A pasted `PASS` result is therefore not reproducible enough to compare even two later executions of the same source.

**Smallest fix:** make each target harness emit a compact machine-copyable header before cases: source revision (or a user-supplied revision field), device name/architecture, runtime/driver/compiler identity available through that stack, dtype cases attempted and skipped, shape/stride fixture, epsilon, and tolerances. For HIP, print `hipDeviceProp_t` identity and compile/runtime versions through checked runtime calls where available; for Triton, print PyTorch CUDA/HIP build information alongside the existing versions. Keep the README instruction to save full environment output. This adds no default CPU dependency and does not turn the commands into benchmark evidence.

### P2 — Convert the target-only wrapper checklist into a separately runnable test mode

**Location:** `examples/accelerators/README.md:38-53`; `examples/accelerators/triton_rmsnorm.py:27-54,57-84`.

The current positive sweep is useful: it covers actual widths, tile tails, strides, zero rows, output contiguity, and bounded dtype comparisons. However, the checked wrapper contract remains unexecuted: invalid rank/width/device or dtype combinations, epsilon, zero stride, and `requires_grad` rejection are only prose requirements. The README names the missing non-default-stream producer-event/consumer-wait case, but no reader has a command that proves it was tested.

**Smallest fix:** retain the dependency-free default suite, but add a target-only mode (or a distinct GPU test file invoked in the README) with two labeled groups: `contract` and `stream-ordering`. The first asserts the wrapper's declared rejection paths. The second writes a bounded source tensor on one non-default stream, records an event, waits on the kernel stream, and compares after a correctness-boundary synchronization. Skip with an explicit reason when the installed target cannot support a requested dtype or stream test. Its result remains *device-operator* evidence, not model integration or speed.

### P2 — Publish the HIP target-evidence checklist beside its build command

**Location:** `examples/accelerators/README.md:55-76`.

The Triton companion has a concrete “Required target evidence before adoption” list, while the new HIP companion has only a useful source-limit paragraph. The source itself does more than the printed HIP excerpt (allocation/copy/error checking, bounded CPU comparison, and a non-default stream), so readers need an equally explicit statement of what a successful run does and does not cover.

**Smallest fix:** reuse a compact HIP-specific checklist after the build command: target compilation for the recorded `hipcc`/ROCm/device tuple; bounded fixture output comparison; launch/asynchronous-error observation; separate producer-event case; non-finite/overflow policy; and integrated engine quality/SLO replay. State that the sample uses pageable host vectors, traditional allocation, fixed FP32 input, and one device, so it establishes no copy overlap, allocator policy, multi-device collective behavior, training path, or latency result. This would align the two optional companions without duplicating their source.

## CPU-only metadata/AST tests and optional JAX

Do **not** add an AST/source-string test that tries to certify Triton or HIP semantics. It would mainly assert the spelling of `tl.sum`, `__syncthreads`, or status labels, while a compiler/runtime can still reject the source or give different numerics. The existing CPU oracle tests and the manuscript builder's diagram/status checks are higher-value CPU evidence.

Likewise, do **not** add a JAX/Pallas CPU semantic experiment solely to create another green test. It would duplicate the Python RMSNorm oracle, add a fast-moving optional dependency, and still establish neither TPU layout nor execution. Add it only when the book introduces a complete Pallas body whose indexing/masking semantics need a specifically documented interpreter check; then label it “simulator/interpreter semantics,” pin JAX/jaxlib, and leave it out of the default CPU suite.

## Checks performed

- Bundled Python full suite (`python -m unittest discover -s tests -v`) — **103/103 passed**.
- Reader CPU commands from the optional-accelerators README: `python -m examples.accelerator_portability` and `python -m unittest tests.test_accelerator_portability -v` — **ran successfully; 7/7 accelerator portability tests passed**.
- `git diff --check` — **passed**.

These checks do not compile or execute Triton, HIP, TPU/Pallas, or Neuron code, and no hardware performance claim follows from them.
