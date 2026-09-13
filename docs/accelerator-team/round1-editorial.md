# Round 1 independent editorial and pedagogy review

## Scope and verdict

I reviewed the current worktree of `manuscript/04_cuda.md` from the transition
at lines 1518–1522 through the end of Chapter 36 at line 1857. I also inspected
all 17 initial rendered pages (`tmp/pdfs/accelerator-team/initial-pages`, pages
250–266) and the three standalone figure proofs. I did not read the other
reviewers' formal reviews. I accounted for the stated post-render figure fixes
to the RMSNorm lanes, compile path, and shell wrapping rather than reporting
those already-fixed proof issues.

**Verdict: strong material, not yet editorially ready.** The chapter has a
clear governing idea, a useful running operation, good visual hierarchy, and a
convincing return from kernel mechanics to deployment evidence. The RMSNorm,
KV-placement, and compilation-lifecycle figures all teach real distinctions.
The closing transition to distributed execution is excellent. However, the
chapter currently reads partly like a publication chapter and partly like a
validation dossier: caveats and support disclaimers recur so often that they
obscure the positive model readers should carry away. Several printed code
blocks reproduce arithmetic already visible in prose or a figure without
showing new mechanism. The vendor sequence also becomes a catalogue before the
running assistant returns on page 260.

A focused compression pass—approximately 15–20%, mostly by removing redundant
status language and low-yield snippets—would make this publishable without
reducing rigor.

## Priority 1: restore a visible three-step reader journey

The best latent structure is already present but not announced:

1. **Port the operator contract:** RMSNorm establishes the invariant and shows
   how schedules differ.
2. **Port model state and compilation policy:** the KV and compiled-variant
   examples show that a working operator is not a deployment.
3. **Decide with workload evidence:** the phase and evidence tables define the
   acceptance experiment.

At present, lines 1548–1731 spend roughly nine pages moving through Triton,
HIP, TPU, and Neuron before “Port the running model before comparing devices”
finally reveals the second major stage. The short SYCL/other-options subsection
at lines 1733–1737 further delays that turn.

**Action:** Add one two-sentence roadmap after the three portability tests at
line 1546, using the three stages above. Then cut the standalone “Other options
and the abstraction boundary” subsection. SYCL already appears in the opening
table, and the CPU/appliance paragraph adds scope without advancing the running
decision. If SYCL must remain, reduce lines 1733–1737 to one sentence attached
to the opening table. Let the backend comparison flow directly into “Port the
running model before comparing devices.”

**Why it matters:** Readers then know why they are learning one small kernel,
when the narrative will return to the assistant, and how the backend survey
serves the deployment decision. This change prevents the middle from feeling
like an encyclopedia of vendor warnings.

## Priority 2: consolidate the caveats instead of repeating the thesis

The central warning—successful source translation, compilation, or simulation
does not establish correctness, integration, or speed—is important. It is also
repeated in the opening table and prose (1538–1546), nearly every example-status
line (1558, 1581, 1618, 1648, 1673, 1709, 1753, 1782), both simulator sections
(1691, 1722), the cross-backend table (1724–1731), the compilation discussion
(1772–1797), the evidence table (1823–1828), the validation-status paragraph
(1830), and Exercise 1 (1834/1843).

A mechanical scan of lines 1524–1857 finds about ninety instances of negating
or limiting language such as “not,” “does not,” “cannot,” and “never.” The
result is a defensive rhythm: claim, disclaimer, narrower claim, disclaimer.
By page 258, a reader is working harder to remember what Pallas or NKI *does*
than what each does not prove.

**Action:** Keep the full three-portability distinction at 1546 and the evidence
ladder at 1823–1828. Between them, retain only limitations that are specific to
the mechanism at hand—for example, `interpret=True` not being TPU execution,
the HIP barrier rule, and NKI simulation not modeling engine overlap. Delete
generic repetitions of “not a support guarantee,” “not a performance result,”
and “not equivalent.” Replace the repeated “Example status:” paragraphs with a
single early convention, such as:

> Unless explicitly marked “device-tested,” listings in this chapter are
> semantic references, excerpts, or schedules; the final evidence table states
> what each validation stage establishes.

Then give individual listings short, natural captions only when their status is
unusual. Preserve detailed validation provenance in the repository or end note.

**Why it matters:** Rigor comes from a clear evidence hierarchy, not the number
of times uncertainty is restated. Consolidation will make the remaining target-
specific warnings more salient.

## Priority 3: keep code that reveals a mechanism; move assertion-only snippets out of print

The code is pedagogically uneven.

### Keep

- **Triton body, lines 1583–1596:** This earns its space. It exposes row
  ownership, masks, actual-width division, element strides, FP32 conversion,
  and the output layout in one compact listing.
- **HIP reduction, lines 1620–1636:** This also earns its space because the
  subsequent three-element example makes a barrier invariant visible. It
  creates a genuine comparison with the Triton schedule rather than a second
  syntax sample.
- **NKI schedule, lines 1711–1718:** Keep it as explicitly labeled pseudocode.
  Its value is the HBM–SBUF movement and the unresolved cross-tile reduction,
  not syntactic completeness.
- **Short `rocprofv3` workflow, lines 1650–1655:** Retain after the acknowledged
  line-wrap fix; it turns “profile it” into an actionable first trace.

### Cut from the printed chapter or demote to a repository note

- **CPU oracle assertions, lines 1560–1567:** The figure and line 1556 already
  give the inputs and exact outputs. Importing an unseen helper and asserting
  those same numbers teaches no additional algorithm or API. Keep the helper
  and tests in the repository, with a short prose pointer if desired.
- **JAX reference, lines 1675–1685:** It repeats the RMSNorm expression a second
  time and contains no Pallas construct. Either reduce it to the two dtype
  decisions in prose or explicitly call it the dtype-matched oracle used before
  Pallas. As printed, its heading promises movement toward a Pallas schedule but
  the listing is simply framework RMSNorm.
- **KV-plan assertions, lines 1755–1766:** The prose and the excellent figure
  already expose every number. The imported helper is not inspectable here, so
  the block functions as build evidence rather than instruction.
- **Padding/break-even assertions, lines 1784–1795:** The two equations and their
  interpretations are clearer in lines 1772 and 1780. The block repeats their
  outputs and consumes most of a page without revealing the planner's policy.

**Why it matters:** A serious technical book should print code when the code is
the clearest way to see control, indexing, ownership, or API behavior. Repository
tests are valuable evidence but need not all be typeset as examples. These cuts
would remove about two pages while retaining every concept and all executable
support material.

## Priority 4: make the common-operation comparison deliberately asymmetric

“Carry one operation across the boundary” suggests four comparable
implementations. The chapter actually provides one complete kernel body
(Triton), one partial reduction (HIP), one framework reference plus a prose
schedule (TPU/Pallas), and NKI pseudocode. The labels are honest, but the implied
symmetry creates a small bait-and-switch.

The cross-backend table at lines 1724–1729 is the most efficient comparison in
the section. It focuses on ownership, working state, tails, and evidence—the
right questions—and should do more work.

**Action:** At line 1548, tell readers that Triton is the worked kernel, HIP is a
synchronization contrast, and TPU/NKI are schedule-reading exercises because
unvalidated pseudo-implementations would teach false portability. Move or
preview the four comparison questions from the table before the backend
subsections, then let each subsection answer those questions. Keep the full
table as the synthesis at 1724.

Within the vendor subsections, adopt the same small internal order:

1. supported framework/serving path first;
2. custom-kernel memory and ownership model second;
3. correctness and profiling evidence last.

The TPU section mostly follows this shape. The Neuron section does not: it moves
from hardware/software taxonomy (1693–1697), to a dated serving-stack release
boundary (1699–1701), back to NKI API migration history (1703–1705), then to the
tile schedule and memory model (1707–1722). Move the dated NxD/vLLM release note
to a short compatibility callout after the NKI schedule, or reduce it to one
sentence. Place the HBM/SBUF/PSUM model immediately after the Neuron definition.

**Why it matters:** The reader compares mental models instead of counting which
vendor received runnable code. A common template also makes omissions visible
without requiring repeated disclaimers.

## Priority 5: reduce checklist prose that reads like implementation notes

Several paragraphs are accurate but over-compressed enough to feel copied from
a review checklist rather than written for continuous reading:

- **Lines 1552 and 1569:** overflow, epsilon rounding, two kinds of reference,
  tolerance selection, and task-level quality arrive before the reader has seen
  a device kernel. Move these into one compact “numeric contract” paragraph
  after the Triton example. Keep FP32 accumulation and the actual-width divisor
  in the initial contract; defer the rest.
- **Line 1598:** the unseen host wrapper checks ranks, device, dtype, strides,
  training mode, width, epsilon, allocation, zero rows, warp configuration, and
  the input value domain in one paragraph. Retain the material constraints that
  explain the listing—fresh contiguous output, `N <= 8192`, legal strides—and
  point to wrapper tests for the exhaustive list.
- **Line 1600:** the strided-view construction is useful; the nine-item width
  list is test documentation. In prose, name the categories (“unit, either side
  of powers of two, common 4096, and the 8192 limit”) and leave exact cases in
  the test file. Preserve the excellent 4096-to-4097 observation.
- **Lines 1697–1705:** multiple versioned documentation links and API-migration
  facts interrupt the NKI mental model. Keep the current namespace and one
  migration-guide link in a dated version note; move historical beta behavior
  out of the main paragraph.
- **Lines 1817–1821:** the bounded-experiment prose largely restates the phase
  table and evidence table around it. Keep the assistant's two p99 targets and
  “cost per successful SLO-qualified request,” but let the tables carry the
  enumerations of replay contents and reported fields.

Also revise the ambiguous arithmetic sentence at line 1556. “Dividing by three
gives approximately `[1.039..., 1.385..., 0]`” sounds as though the vector itself
is divided by three. Suggested direction: “Using the actual width `N=3` in the
mean-square term produces …; mistakenly using the padded width four produces
….”

## Priority 6: remove visual and textual duplication around the strongest figures

The figures render clearly, use consistent visual language, and fit comfortably
on the page. No visual redesign is needed for this editorial cycle. The issue is
that captions and adjacent prose sometimes say the same thing three times.

- **RMSNorm, lines 1554–1556 / rendered page 252:** The figure itself says the
  masked lane contributes zero, divide by `N=3`, store three values, and logical
  lanes are not hardware threads. Its long caption repeats all four points, and
  the following paragraph repeats two again. Keep the post-render all-lanes
  collector fix, retain the numerical comparison below, and shorten the caption
  to the one framing point: “Padding changes the tile, not the mathematical row
  width.”
- **KV placement, lines 1747–1768 / pages 260–262:** The figure is the clearest
  part of this sequence. Keep its compact assumptions, but remove the code block
  that repeats the three rows. The prose should explain *why* TP=16 replicates
  heads; the figure should carry the resulting numbers.
- **Compile lifecycle, lines 1772–1797 / pages 262–263:** Keep the corrected
  compile-then-execute path. The figure establishes the hit/miss policy; the
  prose should then explain bucket economics. Removing the assertion block will
  prevent the same calculation appearing in figure, prose, and code.
- **Opening portability diagram plus table, lines 1536–1546 / page 251:** Both
  list the same implementation paths. Prefer the diagram plus the three
  acceptance-test paragraph. If the “what it does not promise” column must
  remain, drop the same vendor names from the diagram caption and reduce the
  table to the two or three confusions readers most often make.

## Priority 7: raise the exercises to the chapter's actual depth

Exercises 2 and 3 are strong: they require attribution and placement reasoning.
Exercises 1 and 4 mostly ask readers to recite prose that immediately precedes
them. Exercise 1 is answered by the three portability definitions near the
opening; Exercise 4 is answered almost verbatim by lines 1811–1813. None asks
readers to apply the chapter's unusually valuable interpreter/simulator boundary
or the RMSNorm tail invariant.

**Action:** Keep Exercises 2 and 3. Replace or sharpen the other two:

- Give a concrete failed test: a kernel passes Pallas `interpret=True` or the
  NKI CPU simulator at widths 4096 and 4097. Ask the reader to design the next
  device-compile, numeric, memory, and profiling gates without claiming speed.
- Give the three-element/four-lane RMSNorm case or the HIP barrier body and ask
  which tempting early-return/divisor change is wrong and why.

If the cross-vendor handoff remains an exercise, require a decision from the
numbers: provide payload, effective link rate, pack/unpack costs, and an SLO
budget, then ask whether the proposal passes and what unmeasured terms remain.
The worked solution can then demonstrate calculation plus systems judgment,
not repeat a checklist.

## Lower-priority copy and maintenance findings

- **Line 1697:** the NKI paragraph sends readers to SDK 2.31 and 2.26 sources,
  while the following paragraphs emphasize SDK 2.32/NKI 0.6. Even when facts
  remain valid, this looks internally inconsistent. Prefer current conceptual
  pages and reserve older links for an explicitly historical note.
- **Line 1722:** the rendered/manuscript Neuron Explorer link uses
  `/en/latest/neuron-explorer/capture-profiles.html`, which is not the current
  path. Use the current AWS page:
  `https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/how-to-profile-workload.html`.
- **Line 1830:** use “labeled” rather than “labelled” if the book's prevailing
  American spelling is retained.
- **Lines 1526–1530:** the lead and scenario are strong, and the prior chapter's
  final transition at 1522 lands cleanly. Do not expand this opening. If space is
  needed for the roadmap, shorten the dated-source disclaimer rather than the
  assistant scenario.
- **Lines 1853–1857:** the final solution and closing paragraph are concise and
  persuasive. Preserve the final sentence; it is the chapter's best transition
  into the next part.

## Recommended edit sequence

1. Declare the three-stage journey and remove the standalone other-options
   detour.
2. Establish one validation-status convention; delete generic repeated caveats.
3. Remove the CPU, JAX, KV-plan, and break-even assertion blocks from print.
4. Reorder/condense the Neuron subsection and make the backend comparison
   deliberately asymmetric.
5. Shorten captions and checklist paragraphs; preserve the three diagrams.
6. Replace two recall exercises with concrete diagnosis/calculation prompts.
7. Re-render pages 250–266 and check that code blocks and headings still break
   naturally after the expected two-page reduction.

No manuscript, example, figure, or build file was edited as part of this review.
