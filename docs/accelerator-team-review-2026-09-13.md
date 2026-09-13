# Accelerator chapter expansion and review campaign

## Execution contract

- User request: substantially deepen the accelerator ecosystem chapter with code,
  sources, diagrams, and worked examples; use three reviewers, proposer, author,
  and quality-checker cycles for at least three hours.
- Start: September 13, 2026, 21:40:03 UTC.
- Earliest completion: September 14, 2026, 00:40:03 UTC.
- Baseline: `1fb029c`, chapter 36, 445 pages, 95 CPU tests.
- Status: active; rounds 1–2 closed, round 3 independent reviewers running. This is not a
  completion report.
- Continuation: thread heartbeat every 15 minutes; automation id
  `accelerator-chapter-review-team` (ACTIVE).
- Scope: this chapter and directly supporting examples, tests, diagrams,
  navigation, and validation records. No remote publication or GPU benchmark
  claims. Preserve all unrelated material.

## Team protocol

The main agent authors the initial expansion. Three independent reviewers then
write separate critiques: technical/source accuracy; prose and pedagogy; code,
numerical contracts, and production integration. The proposer reads all three
and offers 3–5 competing or complementary improvement packages. The author
records selections and integrates them one at a time. A quality checker reviews
the resulting chapter and test/render evidence, sends findings back, and checks
the author's fixes. Repeat with fresh questions through the requested window.

Existing reusable agents: `astra_technical_review`, `sol_prose_review`, and
`terra_pedagogy_review`. They may be reassigned roles between phases. Only the
main author edits manuscript/04_cuda.md and shared figure/build files; reviewers
write separate files under docs/accelerator-team/. Do not overwrite reviews.

## Initial expansion plan

1. Develop RMSNorm as a common operation: scalar semantics, tiled Triton example,
   HIP launch/lifetime discussion, Pallas and NKI interpretation with explicit
   backend assumptions. Avoid presenting four superficially similar snippets
   as tested drop-in replacements.
2. Explain compilation, dynamic shapes, sharding, and benchmarking with small
   calculations and executable CPU planning/oracle references.
3. Add clean native diagrams for the tiled normalization dataflow, compilation
   lifecycle, and topology/state placement where they improve understanding.
4. Expand device/stack-specific development and profiling workflows, including
   memory hierarchy, collectives, and portability failure cases.
5. Verify primary sources, run tests, build, and inspect all changed pages before
   the first formal three-reviewer cycle.

## Cycles and evidence

21:44 UTC: assigned bounded preparatory source/case briefs to all three existing
agents. Main author owns the initial expansion and executable CPU examples.
Formal independent review follows the initial expanded chapter.

21:50 UTC, initial expansion: chapter grew from 7 to 17 pages (250–266); full
book 455 pages. Added original optional Triton kernel/wrapper/on-target tests,
CPU numerical oracle and padding/KV/compile planners, HIP reduction excerpt,
JAX reference, current NKI reading path, and three native vector figures.
Preparatory briefs are prep-gpu.md, prep-tpu-neuron.md, prep-cases.md; the narrow
prep-code-review.md found final-scale overflow in the CPU oracle. Author added
a named error and regression input. No device code execution is claimed.

Initial validation: 103 CPU tests pass; PDF verifier passes, 455 pages, 80 outline
entries, 264 external link annotations, no sparse-page flag. Main author viewed
all chapter pages 250–266 at 95 dpi and the three new diagram proofs at 130 dpi.
Visual inspection caught an ugly shell-command wrap, a misleading single-lane
reduction connector, and a cold execution arrow labelled warm; source fixes
applied. Rebuild/re-render those final fixes during round 1. Initial images are
under tmp/pdfs/accelerator-team/initial-pages and initial-figure-png.

Round 1 is now ready for three independent critiques of the expanded draft.
No formal review cycle has passed yet. Proposer must read all three round-1
reviews before selecting 3–5 packages; main author then integrates and requests
an independent quality check.

21:51 UTC, round-1 reviewers dispatched in parallel (technical, editorial,
integration). Author rebuilt and viewed pages 252, 256, 262 after the initial
visual fixes; all three are clean. Optional Triton Python source passes an AST
syntax parse only; this is not Triton compilation. A network-enabled HTTP check
of all 44 links in manuscript/04_cuda.md found 43 successful responses and one
confirmed 404 at Neuron Explorer capture-profiles.html. Technical reviewer is
resolving that source. The first sandboxed check had DNS failures for all URLs
and was not counted as successful validation.

To be appended as work occurs. Do not mark the campaign complete merely because
one cycle passed or because a turn ends. Read the clock before completion.

## Round 1 author decisions

21:57 UTC: all three independent critiques and the five-package proposal are
complete. Author read all four files. Accept packages 1–5 in order:

1. Correct access/source/range/payload contracts. Use explicit out-of-range
   rejection, not a new scaled-number algorithm; rename the new KV aggregate
   field while it has no established external callers.
2. State the three-stage journey; replace hidden-helper assertions with one
   self-contained CPU KV calculation; retain kernel code and concise per-fence
   evidence labels. No arbitrary compression quota.
3. Replace formula-only JAX with a concrete Pallas block/window fragment and
   add SDK-matched NKI tile arguments. A fragment-reduction figure is conditional
   on adding understanding beyond the three existing figures.
4. Add a source-only HIP launch companion and make target-test coverage explicit.
   Accelerator execution remains unavailable; source tests cannot certify it.
5. Add a concrete compilation-identity decision and stronger applied exercises.

After integration, an independent quality checker must inspect the changes and
author must fix findings. Campaign remains active; this is one cycle only.

22:00 UTC, author integration: packages 1–3 and 5 implemented; package 4's
printed HIP launch boundary and explicit target checklist implemented, with
the separate HIP source companion assigned to Terra as a bounded author task.
The CPU oracle rejects denominator overflow, compile repayment rejects a
nonrepresentable quotient, and aggregate KV naming is explicit. Payload scope,
closed-beta access, and the profiling URL are corrected. The chapter now has a
self-contained CPU calculation, Pallas block coordinates/byte ledger, NKI tile
arguments, fragment-reduction figure, compile-identity table, and five applied
exercises. Removed repeated imported-helper assertion listings while retaining
their tested companion code. All 103 tests pass, including runnable manuscript
code and figure geometry. Sol and Astra are independently checking the revised
editorial and technical result; main author is rebuilding/inspecting visuals.

22:05 UTC, round 1 closed. Technical QC found a positive repayment quotient
underflow and the HIP companion/printed kernel-name mismatch; both were fixed,
regression-checked, and independently closed. The HIP header now states its
fixed FP32 fixture and distinguishes pageable staging/traditional allocation
from overlap or stream-ordered allocation. Editorial QC required a repaired
KV code wrap; author rebuilt and both author and reviewer inspected the clean
130-dpi page 263. Both QC reports append final round-1 passes.

Round-1 evidence: 103/103 CPU tests; rebuilt PDF 458 pages, chapter 250–269,
80 outline entries, 266 external link annotations, no sparse-page flag. All
changed pages were viewed at contact-sheet scale; editorial QC viewed all
20 chapter pages at 110 dpi, and author enlarged Pallas, fragment diagram, NKI,
variant table, worked solutions and corrected KV listing. Four added diagrams
(five total in the chapter) remain original vectors with geometry checks.
45 Part-IV URLs returned successful HTTP responses after the Explorer repair;
the final NKI source alias was then pinned to the independently retrieved SDK
2.32 source. No GPU/HIP/TPU/Neuron compile or execution is recorded. A baseline
HEAD-only repository audit was clean; it was not a scan of uncommitted files.

## Next cycles — resume here

22:13:30 UTC: user explicitly renewed the reviewer/proposer/author/QC loop and
requested at least another 30 minutes, emphasizing natural professional prose.
That minimum runs through 22:43:30 UTC. The prior three-hour campaign remains
active through at least 00:40:03 UTC; this is an added minimum, not an early end.
Round-2 editorial and integration reviews are available; technical review is
still running. Main author read the two available reviews and is awaiting the
third before requesting the proposer. PDF skill is active for render-and-check.

22:20 UTC, round-2 author decisions: all three reviews and the proposer are read.
Accept Packages 1–5 in order. Package 1 removes the duplicate opening taxonomy,
moves RMSNorm qualifications after its example, and keeps the Neuron lesson
continuous. Packages 2–3 replace generic advice with backend/connector/precision
and rank/scheduler decisions; the nested-DCP arithmetic is kept separate from
the independent-CP helper. Update two exercises rather than adding more.
Package 4 is delegated to Terra for the isolated optional companions and a
stable baseline/candidate acceptance template. Package 5 is independent prose,
technical, integration, and rendered-page QC; use reflow before editing art.

Packages 1–4 are now authored. All 103 CPU tests pass. Sol checks editorial flow;
Astra checks technical sources, arithmetic, and the independently authored
target harnesses; Terra checks the chapter's connection to neighboring chapters.
Main author builds and visually inspects every changed page. Accelerator code
is source-only; Python parsing is not GPU compilation or device execution.

22:25 UTC, round 2 closed. Independent technical, editorial/visual, and story
checks all pass after fixes. QC caught mode-mismatched target provenance,
incomplete labels for representative rejection cases, and omitted HIP launch
assumptions; author corrected them and Astra checked closure. Sol's four copy
cuts remove warning repetition while preserving the source constraints. The
abstract CP example and no-PCP nested-DCP scope now connect explicitly to Part V.
The acceptance record is linked from the repository examples index and capstone,
not a machine-local PDF URI or an unpublished remote URL.

Round-2 evidence: all 103 CPU tests pass; 50 Part-IV URLs return successful HTTP
responses; PDF verifier passes at 458 pages, 80 outline entries, 272 external
annotations, no sparse-page flags. Chapter 36 remains pages 250–269, Part V 270.
Sol inspected every final chapter page and part opener. Main author inspected
the initial complete chapter, final changed seams/figures/code/exercises, and
capstone pages 444–445. The fragment and KV figures lost only internal vertical
spacing, retaining all labels and font sizes; the wrapped Python comment is
fixed. Final renders: tmp/pdfs/accelerator-team/round2-final-pages and
round2-capstone. No accelerator compilation, execution, or benchmark occurred.

Round 3 is a broader continuity pass: independent editorial review of all
chapter leads/closings and part transitions; technical review of modern
attention/state/engines/phase sharding; learning/reference-code integration.
The main author checks source evidence and numerical/reference behavior while
awaiting the three critiques, then requests the proposer. Do not mark either
the renewed 30-minute minimum or the original campaign complete yet.

22:06 UTC: round-1 checkpoint committed locally as `a287725`; no push or release.
Post-commit HEAD audit scanned 77 blobs with no selected credential/path flags.
Round-2 reviews dispatched to all three existing agents; expected files are
round2-technical.md (attention/quantization/engine/collective omissions),
round2-editorial.md (bounded flow and pagination polish), and
round2-integration.md (reader reproducibility, tests, deployment evidence).
Main author has not implemented round-2 changes yet. On continuation, collect
all three, assign proposer to produce 3–5 packages, then author/QC/fix as before.

The time contract is still active until at least 00:40:03 UTC. Reuse all three
agents; for each new cycle obtain independent critiques, a proposer with 3–5
packages, author decisions/fixes, and independent QC. Do not re-run round 1 as
if it were unfinished, and do not pad the chapter simply to generate changes.

Round 2 should first address the recorded editorial/layout backlog: Neuron
supported-path/version note before the continuous memory-to-tile lesson;
compress the duplicate opening taxonomy table; normalize HIP/ROCprofiler
status labels; reduce excessive pre-figure whitespace without shrinking text.
Use parallel review questions for (a) technical omissions in the transition
from operator port to real engine/attention/collective support, (b) professional
reader flow and visual pacing, and (c) reproducibility/edge tests of the source
companions and CPU calculations. Preserve hardware-evidence boundaries.

Later cycles should challenge deployment decisions, compatibility manifests,
prefill/decode-specific sharding support and numerical/quantized-kernel limits,
then perform a fresh holistic read and final source/link/render/code audit.
Prefer removing ambiguity and improving worked decisions over a vendor/spec
catalog. Final campaign completion must update README/release-readiness with
current counts, commit scoped changes locally, leave private remote releases
unchanged, deliver the PDF with an honest readiness verdict, and pause the
heartbeat. Do not finalize before the clock reaches the deadline.
