# Round 1 improvement proposals

Role: proposer, September 13, 2026. Read all three round-1 critiques and the
current Chapter 36 (`manuscript/04_cuda.md:1524–1857`), plus the CPU planner and
optional Triton source. This file proposes work; it does not implement it or
complete the three-hour campaign. Line references identify the reviewed draft
and will move during integration.

## Recommendation and constraints

Adopt the five complementary packages below, in order, with a small independent
check after each. Package 1 closes the concrete correctness/source findings;
Packages 2–5 improve the chapter's teaching and operational depth. Do not use a
15–20% compression target as a quota. Reclaim repetition to make room for useful
code and schedules, rather than shrinking the requested treatment.

Preserve the actual Triton kernel, HIP reduction, profiling workflow, numerical
tail example, three new explanatory figures, deployment evidence ladder, and
closing transition. Retain at least one **self-contained executable CPU worked
calculation** in print. Every code fence must keep `Example status:` within its
preceding three lines, as required by `tests/test_manuscript.py`; a global
listing convention cannot replace those labels. Shorten labels without erasing
the distinction between CPU-executed, source-only, and pseudocode material.

Finding keys below: T1–T3 and D1–D3 refer to the technical review; E1–E7 to the
editorial review's priority sections; I1–I4 to the integration review's findings
in order. R1 is the author's additional extreme-range `compile_break_even`
overflow case. These are mapping keys, not revised severity judgments.

## Package 1 — Close the factual and numerical contracts

**Priority:** first; includes I1, the integration review's P1 clarification.
**Closes:** T1, T2, T3, I1, I2, R1; editorial link/version maintenance concerns.

Make a narrowly scoped correctness patch:

- At the first TorchNeuron mention, state that the checked SDK 2.32 native
  PyTorch path is closed beta; link the [native PyTorch overview](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/frameworks/torch/pytorch-native-overview.html).
  Keep this qualification next to the option, not buried in a later callout.
- Replace the broken Neuron Explorer URL with the verified [SDK 2.32 capture
  guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/tools/neuron-explorer/how-to-profile-workload.html).
  Prefer current conceptual links for current claims; retain old links only
  where a historical distinction is actually explained.
- In the CPU RMSNorm oracle, name and range-check the mean-square-plus-epsilon
  quantity before its square root. Reject the reproduced overflowing case
  `([1e154], [1.0], epsilon=1e308)` rather than returning a plausible zero.
  State that intermediate arithmetic must fit the reference's range.
- Give `compile_break_even` an explicit representable-quotient contract and a
  named `ValueError` when the computed repayment count exceeds that range.
  Cover `compile_break_even(1e308, 1e-308, 5e-309)`. Do not return `None` for
  overflow: that already means the candidate does not improve per-call time.
- Make `aggregate_bytes` self-identifying as modeled physical KV allocation,
  excluding weights/workspace/reservations. Prefer renaming the new public
  field to `aggregate_modeled_kv_bytes`, updating all consumers and labels.
- Identify the 4 GiB transfer as the earlier **whole batch of eight 4096-token
  sequences**, not a universal per-request payload. State the handoff unit and
  committed-token assumption. A one-request, 2000-token contrast is 250 MiB
  under this chapter's full-history, uncompressed cache geometry; it is not
  another measurement or a padded allocation estimate.

**Alternative/tradeoff:** scaled numerical algorithms or exact-rational
amortization can accept a wider range, but introduce unrelated numerical
machinery. Explicit rejection is the preferred teaching contract. For the KV
field, a full dataclass docstring and unambiguous printed label would be a
smaller compatible alternative if preserving callers matters.

**Acceptance:** regression tests reject both overflow reproducers with the
documented exception; existing ordinary outputs, zero-compilation and
non-improving-candidate behavior remain unchanged. All renamed-field references
are updated. The 80 ms and serialized 110 ms calculations still agree with
their explicitly named whole-batch payload. The beta qualifier and replacement
link are checked directly against the primary pages. No device evidence is
implied by these fixes.

## Package 2 — Make the reader journey visible, retaining executable reasoning

**Priority:** second, before adding more backend detail.
**Closes:** E1–E6, I4; supports I1/I2 by clarifying the printed calculation.

Add a short roadmap after the portability definitions: operator contract,
state/compilation policy, then deployment evidence. Announce the deliberately
asymmetric comparison: Triton is the worked tiled kernel, HIP exposes block
synchronization, and TPU/NKI expose target-specific scheduling decisions.
Preview the existing ownership/memory/tail/evidence questions before the tour.

Remove the standalone other-options detour, retaining a short SYCL entry/link
near the opening taxonomy. Consolidate generic portability warnings and status
provenance, while keeping mechanism-specific warnings about barriers, layout
legality, and simulator limitations. Keep every required per-listing status
label. Put supported paths first in each backend section, the working-memory
model second, and validation/profiling last; compress Neuron's migration
history into a dated compatibility note without obscuring access restrictions.

Replace the import/assert-only KV listing with about 12–18 lines of standalone
Python that visibly computes the byte formula and compares TP4, TP16, and
TP4×CP2. Show the head-partition/replication and padded-context calculations,
then label logical bytes, modeled bytes/rank, and modeled aggregate KV
separately. Use the figure for the resulting placement, not a second table of
identical printed numbers. Point to the repository helper for input validation
and additional cases. Link the CPU exercise from `examples/README.md` with its
unittest command.

The RMSNorm import/assert snippet and compile import/assert snippet may move
out of print; their numerical reasoning and tests stay. Package 3 replaces,
rather than merely deletes, the repetitive JAX reference. Shorten captions to
the figures' governing point and correct “dividing by three gives [vector]” to
“using N=3 in the mean-square term produces [vector].” Do not expand the lead
or sacrifice the running assistant scenario.

**Alternative/tradeoff:** retain the existing snippets and add mechanism code
beside them. This preserves every line but worsens duplication. Replacing
assertions with visible calculations is preferable to either wholesale code
removal or additive bloat. If the general KV formula becomes too long, keep a
self-contained TP16 worked calculation plus prose contrasts for the others.

**Acceptance:** at least one printed CPU calculation runs independently under
the repository's supported Python and produces the established KV values; it
does not import hidden arithmetic. The chapter still includes actual Triton
and HIP code. All code-status regression checks pass. A reader can identify
the three-stage journey and the distinct purpose of each backend example.
No changed label upgrades source review to compilation or execution.

## Package 3 — Show concrete TPU and NKI tile choices

**Priority:** third; the highest-value requested technical expansion.
**Closes:** D1, D2; improves E3/E4 by replacing false symmetry with real detail.

Replace the JAX formula-only block with a small, explicitly source-only Pallas
`BlockSpec`/index-map fragment for an array whose rows are complete and whose
row count is divisible by eight. Use an eight-row by 4096-column window and
show that one grid step advances eight rows, not eight elements. Explain where
the weight window comes from. The FP32 input tile alone is 128 KiB; output,
weights, temporaries, and additional pipeline buffers need separate accounting.
State the assumed shape and the absence of a complete kernel body/launch.

Use the [BlockSpec rules](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
including their full-dimension exceptions, and the [TPU pipeline model](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html).
Do not promote the arithmetic footprint into a claim of VMEM fit or prescribe
a blanket “all dimensions must be multiples of 8/128” rule.

Beside the NKI schedule, expose the documented `(128, 512)` SBUF tile,
`axis=1`, and `n=512` relationship, showing the required weight shape/broadcast
and explicit computation/output dtype choices. Source it against the selected
SDK's [`rms_norm` contract](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.language.rms_norm.html)
and the [official language source](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_modules/nki/language.html#rms_norm).
Preserve the experimental/API-version qualification. The API fragment is not
a full HBM-to-HBM implementation; shared-HBM returns and SBUF/PSUM distinctions
remain explicit.

Add one small native dependency diagram if space allows: two fragments produce
partial square sums; those sums combine; the inverse RMS becomes available to
both fragments; output scaling then occurs. Label it a logical dependency,
not an available cross-program barrier. Retaining/reloading row values and
choosing supported multi-stage synchronization are implementation decisions.
The point is that a global divisor cannot repair a fragment-local numerator.
This adds useful depth without duplicating the existing padding figure.

**Alternative/tradeoff:** a compact tile/bytes/dependency table instead of the
new diagram is adequate if the figure creates a page break. Full Pallas/NKI
kernels would offer more copyable code but require a much larger legal-layout,
version and launch contract; defer them without a target validation path.

**Acceptance:** every fragment declares complete-row versus row-fragment
ownership, actual reduction width, weight layout and dtype assumptions. Index
maps and byte counts receive independent source/arithmetic review. No listing
is labeled runnable or interpreted here; no framework packages are installed.
Existing input/output/memory-space distinctions survive. Any added diagram is
rendered and checked for the exact reduction dependency, readable labels, and
absence of an implied universal device barrier.

## Package 4 — Connect kernel bodies to launch and validation evidence

**Priority:** fourth; bounded implementation support, not a benchmark project.
**Closes:** I3; strengthens the useful HIP ownership/lifetime material.

Keep the existing HIP reduction in print. Add a short host-boundary fragment
that makes the stated launch contract visible: exactly 256 threads, one block
per nonempty row, checked launch status, and stream synchronization at a test
boundary for asynchronous failures. If a producer uses another stream, show
or explicitly name the event dependency; do not imply that a final synchronize
retroactively orders producer and consumer work. Inputs and output remain alive
until the consumer finishes.

Prefer storing a complete minimal FP32 HIP source example beside the optional
Triton file: kernel signature, output pass, scoped allocation/copy/launch/check
sequence and a documented build command. This is source-only support unless a
compatible compiler/device becomes available; no synthetic benchmark numbers
or successful compilation badge should be added. The printed chapter needs
only the launch mechanism, not all allocation boilerplate.

For Triton, make the README command visibly a bounded forward smoke test, then
add a target-suite checklist or source tests for invalid rank/device/dtype,
epsilon, unsupported width/strides and autograd; bounded adversarial values;
non-default stream/event ordering; and explicit synchronization/error boundaries.
Report skipped BF16 and capture stack/device metadata separately. Optional
device-dependent tests must not be collected as successful CPU validation.

**Alternative/tradeoff:** the minimum closure for I3 is an unchecked required
target checklist next to the command; it honestly describes the adoption gate
with little maintenance. Executable target tests provide greater value and are
recommended where simple, but their presence is not evidence they passed.

**Acceptance:** source review confirms the fixed HIP block size, full barrier
participation, output addressing and buffer lifetimes. Python syntax checks
and CPU tests remain separate from unavailable HIP/Triton compile/runtime
checks. Target harness output cannot reasonably be mistaken for a complete
production qualification. Existing source-reviewed status remains unchanged
until actual target evidence is recorded.

## Package 5 — Turn compilation and exercises into decisions

**Priority:** fifth, followed by integrated technical/editorial/visual QC.
**Closes:** D3, E7 and remaining E5/E6 repetition; reinforces I1.

Replace the compile assertion-only block with a small hypothetical compatibility
manifest or table: identical prompt lengths with different dtype or static
sharding choices may select different artifacts; changed runtime valid length
inside the same bounded storage may reuse one. Explicitly declare which
dimensions the hypothetical integration specializes. Two shape buckets, two
dtypes and two selected static layouts yield eight possible variants under
that assumption, not a universal compiler cache-key rule.

Walk one hot/rare variant through the existing 60-second/960-call serial
break-even model and the compile-lifecycle miss policy. Distinguish prewarming
from amortization: a required rare variant can still deserve prewarming to meet
readiness/SLOs even when local compile cost is not repaid. Include compatible
target/compiler/model configuration and trusted artifact provenance in the
manifest description. Keep the existing [persistent-cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
and [shape-polymorphic export](https://docs.jax.dev/en/latest/export/shape_poly.html)
distinctions; do not imply a single binary handles all concrete shapes.

Keep the cold-start and memory exercises. Replace the recall-oriented first
exercise with an interpreter/simulator-passes-at-4096-and-4097 case asking for
the next compilation, numeric, memory and profiling gates. Make the handoff
exercise require a calculation and decision from explicitly scoped bytes,
effective GiB/s, serialized packing costs and transfer budget; ask what remains
unmeasured and how cancellation/ownership failure changes acceptance. Use the
existing whole-batch numbers or a clearly separate per-request case, never mix
their payload units. Include a short tail/divisor or HIP early-return diagnosis
as a subquestion so the kernel mechanics are assessed too.

**Alternative/tradeoff:** a full executable prewarm optimizer is unnecessary:
the chapter has neither a measured workload distribution nor a compiler cost
model. A manifest plus two decisions exposes the mechanism without inventing
those inputs. If more CPU code is desired, a tiny variant enumeration is safe
but must earn its place beyond the standalone KV calculation in Package 2.

**Acceptance:** the variant count and serial repayment arithmetic are
independently checked; same-length does not mean same-artifact, and reuse of
tracing does not mean reuse of every device executable. Worked solutions derive
at least one answer rather than reciting the evidence table. Final QC reruns
the full supported-runtime test/build/link checks and inspects every changed
chapter page, with special attention to code labels, figure dependencies,
headings and page breaks. The log records selected alternatives, evidence and
remaining device-validation limits. Passing this cycle is not completion of
the campaign before September 14, 2026, 00:40:03 UTC.
