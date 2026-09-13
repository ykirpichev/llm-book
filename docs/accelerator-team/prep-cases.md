# Accelerator-portability worked-case brief

Prepared for the chapter-36 expansion campaign. This is a planning artifact,
not a benchmark report: all times, bandwidths, and capacities below are
illustrative inputs to CPU oracles and planners. They must not be presented as
measurements of NVIDIA, AMD, TPU, or Neuron hardware.

## Purpose and boundary

The new **Accelerator Ecosystems Beyond CUDA and NVIDIA** chapter already makes
the right high-level distinction: source portability, semantic portability,
and performance portability are separate claims. The smallest useful addition
is not another vendor survey. It is a set of cases that force a reader to
carry that distinction through one numerical operator, shape specialization,
device placement, state handoff, and a request-level benchmark.

Use the existing illustrative configuration unless a case says otherwise:

| Quantity | Value | Meaning |
| --- | ---: | --- |
| Layers | 32 | decoder layers |
| Hidden width `D` | 4096 | RMSNorm row width and model hidden width |
| KV heads / head dimension | 8 / 128 | grouped-query KV geometry |
| KV element width | 2 bytes | illustrative BF16/FP16 storage ledger |
| Decode batch / context | 8 / 4096 | the chapter's decode shape |
| Logical KV at that shape | 4 GiB | K and V across all layers; before physical overhead |
| Model weights | about 14 GB | two-byte 7B-class illustrative checkpoint |

The cases should appear after the portability introduction or in its worked
examples, not interrupt the CUDA-kernel derivations. Reuse the existing 7B
documentation-assistant context so portability is evaluated as a change to one
service contract rather than a collection of unrelated devices.

## Case 1 — RMSNorm: a source-compatible tile can still be numerically wrong

### Teaching question

An RMSNorm kernel is ported to a new backend and returns plausible values for
`D=4096`. What must be true before that source path is called semantically
portable, especially for a final tile, noncontiguous input, and low-precision
input?

### Setup and derivation

For row `x` with width `D`, learned scale `g`, and a declared positive
`epsilon`, use the mathematical reference contract:

`rms2 = (sum_{i=0}^{D-1} float32(x_i)^2) / D`

`y_i = x_i * rsqrt(rms2 + epsilon) * g_i`.

The target implementation must separately declare its accumulation dtype,
epsilon representation, conversion points, and output dtype. Finite source
values plus a positive host-language epsilon do **not** guarantee finite FP32
intermediates or output: `x_i^2` or its reduction can overflow, a tiny epsilon
can round to zero on conversion, and a finite learned scale can overflow the
final multiplication. Choose one explicit policy: (a) bound the supported
activation/scale/epsilon domain and test that bound, or (b) detect non-finite
intermediates/output and promote, fall back, or return a declared error. A
port may not silently turn these cases into plausible finite values.

A tile whose physical width is `T=256` has sixteen full tiles at `D=4096`; use
`D=4097` to force a 255-lane tail. Masked tail lanes must contribute neither
their squared values nor output stores. A strided view establishes that
logical row order, not a presumed contiguous addressing formula, defines the
result.

Suggested adversarial rows:

1. all zeros (`y` must be zero and finite only when epsilon remains positive
   in the compute dtype; otherwise the implementation must take its declared
   detection/fallback path);
2. alternating `+a` and `-a` (tests squared accumulation without cancellation
   as an excuse);
3. one large finite value plus small values (tests overflow/precision policy);
4. `D=4097` with guard values beyond the valid row (tests tail masking);
5. a noncontiguous row with the same logical values as a contiguous reference;
6. finite values that overflow an FP32 square/reduction, and a finite
   high-scale row whose final multiplication overflows, to exercise the
   declared range or fallback policy.

Do **not** demand bitwise equality across legal reduction orders. The text
should require a declared comparison dtype and tolerances, plus these
invariants: no access or contribution outside `D`; equality to the same
logical row under a noncontiguous view within tolerance; and either finite
output inside the declared numerical domain or the documented
overflow/detection outcome outside it.

### Worked decision

Compilation and a `D=4096` smoke test establish only that one source path
exists. Start with the framework RMSNorm as the target semantic reference.
Retain a custom kernel only if it passes tail/stride/value tests and the
engine-level profile shows a material end-to-end opportunity. If a backend
requires padding to `T`, write the mask and the padded-storage contract down;
do not imply that padding is harmless because `4096` happens to divide today’s
tile.

### CPU oracle contract

Implement a dependency-free `rmsnorm_reference(rows, gamma, epsilon, valid_D,
row_stride)` routine only if the campaign adds a CPU helper.

- Inputs: finite logical rows, scale of length `valid_D`, `epsilon > 0`, and
  explicit logical indexing/stride metadata.
- Output: a higher-precision host-language semantic reference and a `valid_D`
  mask; it is not an FP32 emulator and produces no device timing.
- Required tests: the five rows above; identical logical rows in contiguous
  and strided storage; a negative test proving that an unmasked padded sentinel
  changes the result; and explicit FP32-range/epsilon-cast/final-scale cases.
- Assertions: invalid tail positions are never read/written by the model under
  test; in-domain comparisons use declared `atol + rtol * abs(reference)`
  rather than equality; and out-of-domain cases produce the declared fallback
  or error rather than an unchecked numerical result.

### Diagram semantics

Use a single **RMSNorm tile contract** diagram, not a backend logo diagram:

`logical row and valid-D mask → tile loads → declared-dtype sum-of-squares
reduction → epsilon cast/check → rsqrt → scale/store → finite-result check or
fallback`.

Show the final tile with gray padded lanes whose arrows terminate before the
reduction. Label the semantic boundary above the schedule; below it, optional
backend implementations may have different tile shapes and participant groups.

## Case 2 — Shape buckets: minimize total cost, not padding alone

### Teaching question

A compiled serving path supports prompt buckets 2048 and 4096. Is a
2300-token request better placed in 4096, in a new 3072 bucket, or on an eager
fallback?

### Planner model

For request length `s` assigned to bucket `b >= s`, report separate lower-bound
work multipliers:

- linear-token multiplier: `b / s`;
- dense-attention score-work multiplier: `(b / s)^2`;
- padded tokens: `b - s`.

For `s=2300`, `b=4096`, these are approximately `1.78`, `3.17`, and 1796.
They are accounting ratios, **not** latency predictions. They motivate an
experiment because ragged kernels, chunked prefill, masked tile skipping, and
other work can change realized time.

Choose from candidates by a declared objective such as:

`expected_request_cost = warm_runtime(s,b) + cold_probability(b) * cold_penalty(b)`

plus a separately reported memory/binary/test cost per variant. This avoids
the false conclusion that every smaller bucket is better. A new 3072 variant
may reduce padding but create compile, artifact, graph-memory, warm-up, and
test burden. Eager fallback is a valid candidate, especially for rare shapes.

### Worked decision

Given a representative length histogram, generate candidate boundaries from
observed discontinuities rather than powers of two by habit. Reserve an eager
fallback. Add a bucket only when its frequency and measured warm saving repay
its variant cost and cold-start effect under the deployment SLO. Keep the
expected objective symbolic until real measurements are available.

### CPU planner contract

`assign_bucket(length, buckets, fallback_max_padding_ratio)` returns either a
bucket or `eager`; `bucket_report(requests, candidates, per_bucket_inputs)`
returns:

- assignment and count per bucket/fallback;
- total padded linear positions and padded dense-score positions;
- per-bucket cold starts, compile/load input, artifact bytes, and graph-pool
  reserve as supplied metadata; and
- no device-time estimate unless the caller explicitly provides measured
  per-bucket times.

Required properties: reject `b < s`; assignment is deterministic; changing a
rare shape to eager does not change other requests’ logical lengths; a report
labels all time/cost fields as inputs or measurements, never inferred hardware
facts.

### Diagram semantics

Use a **shape-routing decision tree**: request length → compatibility check →
existing compiled bucket / compile-or-load queue / eager fallback. Place
padding work, executable cache, and readiness deadline on distinct branches.
This prevents a visual claim that a shape bucket is merely a padding choice.

## Case 3 — Per-device shard fit: aggregate capacity is not an allocation

### Teaching question

An alternative accelerator has enough aggregate memory for the 14 GB weights
and 4 GiB logical KV working set. May the service reduce tensor parallelism
from two ranks to one?

### Planner model

For each rank `r`, calculate a peak ledger rather than a fleet total:

`peak_r = weights_r + active_KV_r + reusable_KV_r + workspace_r +
communication_buffers_r + graph_or_compile_reserve_r + safety_reserve_r`.

Require `peak_r <= usable_memory_r` for **every** rank. `usable_memory` is a
measured or configured reserve-aware input, not device nameplate memory.

For the illustrative 8-by-4096 decode shape, logical KV is 4 GiB before page
rounding, duplicated heads, prefix-pool policy, and any allocator reserve. A
one-rank candidate might appear to need `14 GB + 4 GiB`; it still fails if its
peak ledger includes a larger physical KV pool, workspaces, graphs, or safety
reserve. Conversely, a two-rank plan can have more aggregate memory yet fail
one rank through an uneven head, layer, or prefix placement.

### Worked decision

Treat fewer ranks as a hypothesis with two competing effects: it may remove
per-layer collectives, but it may reduce per-request bandwidth/compute and put
all memory pressure on one device. Reject the plan before benchmarking when
any rank exceeds usable memory. For feasible plans, compare the same workload’s
TTFT, token-gap tails, and SLO-constrained admission; “fits” is a gate, not a
performance result.

### CPU planner contract

`validate_rank_plan(plan, usable_memory_by_rank)` accepts explicit per-rank
allocations by category and returns a rank table, maximum utilization, and
named infeasible categories/ranks.

- Reject negative bytes, missing rank ownership, or a plan that supplies only
  aggregate memory.
- Treat replicated allocations as present on every declared replica; shard
  allocations must identify the fraction and logical range they own.
- Required tests: aggregate fit but one-rank overflow; a balanced feasible
  plan; an uneven GQA/KV-head mapping; and an added graph/workspace reserve
  that flips a previously feasible plan.
- The helper must not choose TP/CP degrees or assert performance. It validates
  a proposed physical placement.

### Diagram semantics

Use a **per-rank stacked ledger**, one column per device. Split each column into
weights, active KV, reusable KV, workspaces, communication, and reserve; mark
the usable-memory line. Draw logical ownership arrows above the columns. Do
not draw a single pooled-memory bar for a model-parallel group.

## Case 4 — KV handoff: transfer plus repack must beat the saved queueing

### Teaching question

Can NVIDIA prefill hand a request to AMD decode merely because both can load
the same checkpoint?

### Break-even model

Define an explicit handoff payload `M` after choosing the source and target
logical cache contracts. A lower-bound handoff time is:

`T_handoff >= T_pack + M / beta_source + M / beta_link + M / beta_target +
T_unpack + T_protocol`.

Some implementations overlap stages, so the observed critical path can be
lower than this serial sum but not lower than the longest required dependent
stage. The planner should model overlap only when the handoff implementation
states which buffers and streams make it legal.

For the running decode shape, full logical state is 4 GiB; a 2,000-token
prompt would be about 250 MiB before physical overhead under the same formula.
Neither number is automatically transferable: define model/tokenizer revision,
positions, layer/head ownership, K/V order, dtype, quantization scale and
zero-point metadata, page/block size, and destination reservation before
using `M`. A shared checkpoint is not a cache ABI.

The decision criterion is qualitative until measured inputs exist:

`saved_queueing + saved_phase_interference + capacity_value > handoff_critical_path
 + added_tail_risk + conversion_cost`.

Both sides must also remain inside the TTFT and token-gap contract under
concurrency, cancellation, partial transfer, and failure.

### CPU planner contract

`kv_handoff_plan(source_contract, target_contract, payload, stage_rates,
overlap_edges)` returns either an incompatibility report or a critical-path
lower-bound schedule.

- Contract identity must include model/tokenizer/template, logical token range,
  layer/head partition, K/V ordering, positions, dtype/layout, quantization
  metadata, and page geometry.
- Incompatibilities are errors, not automatic conversions. A conversion step
  must be explicit and have bytes/rate metadata.
- Required tests: same logical layout; a quantization-metadata mismatch;
  changed head ownership requiring repack; destination reservation failure;
  cancellation before ownership transfer; and failure after ownership transfer
  but before source reclamation.
- Report serial stage sum, declared-overlap critical path, source/destination
  peak temporary bytes, and state owner at each commit point. It must not
  label a CPU copy estimate as a network measurement.

### Diagram semantics

Use a **handoff state machine** rather than an arrow between vendor logos:

`source owns → destination reserves → pack/repack → transfer → verify →
destination commits → source releases`.

Annotate the only legal retry points and cancellation behavior. Show that
repacking may need temporary source and destination buffers simultaneously.

## Case 5 — End-to-end acceptance: a faster warm kernel is not a port result

### Teaching question

A target backend reports faster warm RMSNorm and decode microbenchmarks. What
evidence would justify serving the documentation assistant on it?

### Worked evaluation plan

Use one fixed, versioned replay containing short and long prompts, warm and
cold prefixes, cancellation, overload, and the existing GQA decode geometry.
Hold the checkpoint, tokenizer/template, sampling policy, quality evaluation,
and client-visible SLO boundary fixed. Segment results by:

- warm versus cold worker and first use of every executable variant;
- prefill versus decode; prompt/output/context buckets; and cache hit/miss;
- queue delay, compilation/artifact load, host submission, model execution,
  transfer/repack, and stream flush;
- model/version/layout and deployment cohort.

The release gate has three independent rows:

| Claim | Minimum evidence | A disqualifying result |
| --- | --- | --- |
| Semantic portability | CPU/reference comparisons, cache-state and cancellation tests, task-quality guardrails | A tail/stride/cache mismatch, unsupported feature, or quality regression beyond its declared guardrail |
| Warm performance | Operator and engine replay by shape | Conversion, dispatch, collectives, or another phase erases the claimed gain |
| Operational performance | Rolling scale-out and bounded failure replay | Compilation/load/readiness or state recovery causes SLO failure despite fast warm runs |

Report SLO-constrained goodput and cost per successful request, not a single
throughput number. The result may be “do not port now”; that is a valid outcome
when a supported baseline beats a custom path after operational costs.

### CPU planner/fixture contract

`evaluate_port_evidence(events, contract)` validates bookkeeping rather than
pretending to benchmark accelerators.

- Every event carries request ID, worker cohort, cold/warm state, shape,
  model/cache ABI version, phase, timestamps, outcome, and termination reason.
- The fixture calculates boundary-defined TTFT, inter-token gaps, completion,
  rejection, and categorized time. It rejects missing phase boundaries and
  mixes of incompatible artifact/cache versions.
- Required synthetic traces: warm win with no end-to-end win; compile delay
  that breaks scale-out readiness; a KV-handoff repack tail; a cancellation
  releasing reservations; and an equivalent-quality but higher-cost candidate.
- Tests prove accounting and acceptance logic only. They make no hardware
  speed, support, or portability claim.

### Diagram semantics

Use a **request evidence timeline** with lanes for gateway, compiler/artifact
cache, prefill worker, decode worker, and client stream. Distinguish a cold
compile/load span from device execution and queueing. Put semantic checks and
state ownership at transitions, not after the final latency bar.

## Recommended placement and anti-duplication rules

1. Keep the current four portability exercises concise. Expand them with these
   cases only when the worked solution teaches a new decision; do not repeat
   the existing source/semantic/performance definitions.
2. Put the RMSNorm case immediately after the Triton subsection, where it
   makes the portable-operator boundary concrete.
3. Put bucket, rank-fit, and handoff planners in **Port the running model before
   comparing devices**. They instantiate its existing 14 GB / 4 GiB ledger.
4. End with the acceptance matrix/timeline. It turns the chapter’s closing
   “bounded experiment” into a reproducible artifact list rather than adding
   vendor performance claims.
5. If code is added later, keep it in a new CPU-only example/planner module
   with tests. Do not mix it with CUDA excerpts or present it as a compiler,
   runtime, or hardware emulator.

## Source notes

The mathematics above is original illustrative accounting. It does not need a
vendor citation. For changing framework/backend facts, retain source links and
version/date checks in the manuscript:

- [RMSNorm paper](https://arxiv.org/abs/1910.07467) for the root-mean-square
  normalization operation; the case’s FP32 oracle/tolerance is a book contract,
  not a claim that every implementation accumulates identically.
- [Triton installation and compatibility entry point](https://triton-lang.org/main/getting-started/installation.html), which directs readers to the current compatibility material and distinguishes GPU from no-GPU tests.
- [AMD HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html) for translation scope; availability/compatibility must be checked for the deployed release.
- [Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html): Pallas exposes custom kernels for GPU and TPU but has distinct hardware-specific APIs; it also lists compilation-cache and shape-polymorphism facilities that make bucket/cold-start evidence relevant.
- [AWS Neuron vLLM migration guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/getting-started/migration-nxdi-to-vllm-neuron.html) for the moving Neuron/vLLM integration path.

These sources establish interface scope, not support or performance for the
book’s running service. Each such claim needs the exact deployed model,
accelerator, compiler/runtime, and engine version.
