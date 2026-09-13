# Round 2 story and cross-part QC

**Scope.** Read-only navigation and continuity review of the current Part IV
accelerator chapter against the front matter, Part III's inference/service
story, and Part V's collective and distributed-inference chapters. This review
does not assess the optional harness implementation or claim any device run.

## Verdict

The reader journey is substantially coherent. The prior chapter's final CUDA
principle (semantic reference, profile, engine replay) is a strong launch into
the portability chapter. The recurring model remains numerically consistent:
32 layers, width 4096, eight KV heads of dimension 128, two-byte KV elements,
about 14 GB weights, 128 KiB per cached token, the `B=8,S=4096` 4 GiB example,
and the Part III 1.5-second TTFT / 100-ms token-gap objectives. The new
backend/connector/representation gate and the eight-decodes-plus-long-prompt
replay advance the story rather than repeat the earlier scheduling chapters.

No broad reorganization or vendor expansion is warranted. The following three
small navigation/terminology fixes would prevent readers from carrying the
wrong meaning of context parallelism into Part V or losing the new evidence
artifact outside the manuscript.

## Findings

### P1 — Name the abstract `CP=2` capacity exercise as distinct from Part V's phase-specific PCP/DCP

**Location:** `manuscript/04_cuda.md:1757-1793`; compare
`manuscript/05_distributed.md:812-897`.

The chapter correctly contrasts its independent `TP=4, CP=2` placement model
with the inspected nested `TP=16, DCP=2` model. But it initially calls the
first axis simply “context parallelism” and says it partitions token positions.
Part V then gives `PCP` a different operational meaning (prompt query work,
normally an added process-group dimension) and `DCP` a decode-history meaning
(often reusing ranks inside TP). A reader who retains only “CP halves history”
can mistake the capacity exercise for a portable prefill flag or apply its
eight-rank calculation to an engine's DCP configuration.

**Smallest fix:** label the first calculation “an abstract independent
context-axis capacity model” on first use, and append one forward reference:
“Part V distinguishes prefill PCP from decode DCP; this arithmetic is neither
an engine flag nor a schedule.” Retain the existing concrete DCP contrast and
do not add another parallelism table or diagram.

### P2 — Give the reader an explicit route from Chapter 36 to Part V’s relevant section

**Location:** `manuscript/04_cuda.md:1830-1832,1887`; compare
`manuscript/05_distributed.md:766-938`.

“The next part derives the attention decompositions” and the closing reference
to “peers” are conceptually right, but Part V begins with collectives and only
later reaches **Distributed Inference and Stateful Placement**. That later
section contains exactly the material the new chapter tees up: TP decode,
KV ownership, PCP/DCP, handoff, routing, and streaming failure semantics.

**Smallest fix:** replace the first generic handoff with a short pointer such
as “Part V’s *Distributed Inference and Stateful Placement*, after its
collective foundations, derives those decompositions.” The final paragraph can
then keep its concise ownership/failure handoff without restating the same
promise. This improves non-linear reading without duplicating Part V.

### P2 — Make the acceptance template a navigable companion artifact, not only a filesystem path

**Location:** `manuscript/04_cuda.md:1848-1855`,
`docs/accelerator-acceptance-template.md`, and
`manuscript/09_appendices.md:5-69`.

The new template is the practical bridge from the chapter's evidence ladder to
a baseline/candidate decision, but the manuscript currently renders its
repository location as inline code. A PDF-only reader or a learner following
the Part IX capstone has no obvious companion-artifact index that tells them
when to use it.

**Smallest fix:** make the Chapter 36 reference a repository-relative link
(where the build preserves local links) and add one sentence to the Part IX
capstone/lab's “learning order” that points to the acceptance record when the
capstone compares serving backends or accelerator paths. The template should
remain an unfilled form; do not add illustrative vendor results.

## Deliberate non-findings

- Repeating the running SLOs and a compact 4 GiB/250 MiB transfer calculation
  is useful context, not harmful duplication: Part III defines the service
  contract, while Chapter 36 uses it to reject a portability decision.
- The chapter's source/device/engine/deployment evidence ladder complements
  Part III's benchmark protocol and Part V's production-readiness checklist;
  it does not contradict their scope.
- The existing CPU semantic status remains appropriately separate from
  accelerator compilation, numerics, and performance evidence.

## Closure

The author adopted all three bounded navigation fixes. Chapter 36 now calls
the `CP=2` calculation an abstract independent context-axis capacity model and
explicitly distinguishes it from Part V's phase-specific PCP and DCP. Its
handoff now points readers to Part V's **Distributed Inference and Stateful
Placement** section after the collective foundations.

The baseline/candidate acceptance record is now discoverable from the root
README's **Verified teaching examples** section, and the Part IX capstone
points readers to that index when comparing accelerator paths. The route uses
repository documentation links rather than a machine-local PDF URI or an
unpublished remote URL. These changes resolve the findings above without
adding duplicate parallelism material or unsupported benchmark claims.
