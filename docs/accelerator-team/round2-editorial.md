# Round 2 independent editorial and visual review

**Checkpoint:** `a287725`  
**Role:** professional-book/story/pedagogy reviewer, September 13, 2026  
**Recommendation:** make one bounded structural pass, then re-render. Do not add another backend, example, diagram, or caveat. The chapter's technical depth and overall journey already work; this round should make the positive explanatory line easier to follow and use the reclaimed space to improve pagination.

## Scope

I reviewed `manuscript/04_cuda.md:1524–1883` at the named checkpoint, the round-one editorial backlog, the round-one chapter render on pages 250–269, and the corrected 130 dpi page 263. I did not read other round-two formal reviews. This is a proposal only: I did not edit the manuscript, examples, figures, or shared code.

Preserve the actual Triton and HIP excerpts, the Pallas `BlockSpec`, the NKI API fragment, the visible CPU KV arithmetic, all four new diagrams, the compatibility table, and the five exercises. None is the source of the chapter's remaining editorial friction.

## Priority 1 — Make the opening choose, rather than repeat, its organizing device

**Location:** `manuscript/04_cuda.md:1532–1550`; rendered pages 251–252.

The accelerator-portability diagram and the five-row taxonomy table introduce the same implementation paths in immediate succession. The diagram establishes the hierarchy visually; the surrounding paragraphs already explain the crucial distinctions between device, stack, and kernel language; every backend section then supplies the limitation that the table's third column previews. The table therefore feels like front-loaded reference notes rather than the next step in the argument.

**Proposed revision:** retain the diagram, delete the table at lines 1538–1544, and preserve the three acceptance tests and three-stage roadmap at lines 1546–1548. Keep SYCL as the single linked sentence at line 1550, or attach that sentence to the diagram paragraph; do not recreate a miniature table in prose.

The resulting sequence should be:

1. separate accelerator, stack, and kernel language;
2. see that separation in the portability diagram;
3. define source, semantic, and performance acceptance;
4. preview the operator → state/variants → deployment journey;
5. begin RMSNorm.

**Reader benefit:** the opening moves from concept to method without pausing for a second inventory. It also removes a cluster of negative promises before the reader has seen the mechanisms they qualify.

**Intended layout benefit:** removing the table should recover roughly one-third to one-half page near the front. Re-render before touching figure sizes: this shift is likely to pull the RMSNorm diagram onto page 252 and improve the large gap there, while changing later code/figure breaks globally.

Do not remove the synthesis table at lines 1746–1751/page 261. That later table is earned: after the examples, it compares the questions a porter must answer and helps the reader consolidate four different execution models.

## Priority 2 — Put the supported Neuron route before the uninterrupted tile lesson

**Location:** `manuscript/04_cuda.md:1701–1744`; rendered pages 260–261.

The current order is ecosystem/access → NKI memory hierarchy → NxD/vLLM release history → migration advice → NKI tile schedule. This makes the reader leave the memory model just as HBM, SBUF, and PSUM become concrete, then return to it under the NKI subheading. The material is useful, but its order hides the chapter's strongest Neuron explanation.

**Proposed order:**

1. Retain the opening definition of Trainium, Inferentia, and the Neuron stack at line 1703.
2. Move the current release-boundary and migration content from lines 1707–1709 directly after that definition and compress it to one paragraph.
3. Begin the custom-kernel path with the current NKI/memory paragraph at line 1705, followed immediately by the `NKI: follow the tile between memory spaces` subheading and current-version namespace sentence.
4. Continue without interruption through schedule → SBUF/PSUM/output contract → concrete `(128, 512)` API fragment → simulator/device gates.

**Replacement direction for the compatibility paragraph:** lead affirmatively with the route a new evaluation should inspect, then state the dated boundary once. For example: “For a new serving evaluation under the checked SDK 2.32 documentation, begin with the vLLM Neuron plugin's supported models and targets. The plugin is beta, currently targets Trn2 and Trn3, and no longer depends on NxD Inference; NxD Inference entered maintenance mode with SDK 2.32. Use the migration guide to distinguish a supported model move from custom modeling work, then verify the assistant's attention, cache, quantization, and sampling features.” Retain the existing primary links.

This direction replaces, rather than supplements, the current two paragraphs. Avoid retaining both “Do not infer…” and “should not be presented…” after the supported route has been stated; the dated target sentence carries that boundary more directly.

**Reader benefit:** readers first learn which serving path is current, then receive one continuous code-reading lesson following a tile from HBM to SBUF, through reduction, and back to shared HBM. Version guidance becomes a decision aid rather than an interruption.

**Intended layout benefit:** combining two migration paragraphs into one should remove several lines from page 260 and may keep the NKI pseudocode plus its immediate SBUF explanation together more naturally. Do not shorten the `(128, 512)` example to gain space; it is the section's concrete payoff.

## Priority 3 — Let the RMSNorm example establish the idea before its range limits

**Location:** `manuscript/04_cuda.md:1554–1562`; rendered pages 252–253.

The section currently gives the equation, immediately lists storage/dtype/range/backward qualifications, and only then shows the padding diagram and three-element example. The qualifications are sound, but they delay the small visual result that makes the rest of the accelerator tour intelligible.

**Proposed revision:** keep the first two sentences of line 1556 (separate input/weight/output tensors; FP32 accumulation and output conversion) beside the operator contract. Move the bounded-activation, epsilon-rounding, and training/backward sentences after the diagram and numerical example, folding them into the validation paragraph at line 1562.

The local story becomes equation → implementation dtype contract → padded-tile picture → plausible wrong answer → validation range and tolerances. No technical content needs to be deleted; this is an order change plus light consolidation.

**Reader benefit:** the reader gets a successful mental model before meeting edge-condition bookkeeping. The warnings then answer a natural question—“what else must the oracle cover?”—instead of reading as a pre-emptive disclaimer block.

**Intended layout benefit:** together with removing the opening table, this creates more flexible paragraph/figure pagination on pages 252–253 without splitting the diagram from its worked example.

## Priority 4 — Normalize provenance labels into a short, predictable house style

**Locations:** lines 1574, 1611, 1635, 1652, 1677, 1717, 1732, and 1769.

The required `Example status:` labels should remain before every listing. Their editorial job is provenance, not to repeat the full input contract already explained in adjacent prose. Use three stable openings: `Source-only`, `Explanatory pseudocode`, and `Runnable Python`.

The two clearest fixes are:

- line 1611: `Example status: Source-only HIP kernel excerpt; no compile or device run recorded.`
- line 1652: `Example status: Source-only rocprofv3 recipe; not run here.`

`Hardware-dependent` is less precise than `Source-only`, and `Illustrative on-target` briefly sounds like collected device evidence. The assumptions about 256 threads, pointers, strides, dimensions, and stream lifetime already appear in the introductions and companion source; keep them there.

If the pass normalizes all labels, use similarly compact forms:

- Triton: `Source-only Triton excerpt; companion wrapper exists; no GPU run recorded.`
- HIP launch: `Source-only HIP launch excerpt; no compile or device run recorded.`
- Pallas: `Source-only Pallas window; kernel body and launch omitted.`
- NKI schedule: `Explanatory pseudocode; not executable.`
- NKI API: `Source-only NKI 0.6 fragment for FP32 (128, 512) SBUF tiles; not run.`
- KV: retain `Runnable Python; self-contained KV arithmetic for the three stated layouts.`

**Reader benefit:** a reader can scan one phrase and know the evidence level. Mechanism contracts remain in explanatory prose, where they teach; the status lines stop feeling like test metadata inserted into the book.

**Intended layout benefit:** this saves a few wrapped lines around the dense HIP, Pallas, and NKI listings and reduces the chance that a status line strands a code block on the following page.

## Priority 5 — Reflow after text cuts; modify art only where the gap remains material

**Locations:** rendered pages 252, 255, 262, and 264; following figures on pages 253, 263, and 265.

The current images are legible and no figure should be removed. The issue is rhythm: four pages in one twenty-page chapter end with conspicuously large unused areas because a kept-together code block or figure moves forward. Pages 252 and 264 are the strongest candidates for correction; page 255 is acceptable if preserving the HIP block intact requires it.

Use this order of operations:

1. Apply the opening-table removal, Neuron consolidation, RMSNorm reorder, and shorter labels.
2. Render the entire chapter, not isolated pages, because early cuts will shift every later break.
3. If page 252 is still sparse, keep the RMSNorm diagram and numerical paragraph together; adjust local paragraph spacing rather than separating them.
4. If page 262 still has a large gap before the KV figure, reduce vertical gaps and padding *inside* `kv_port_placement` while preserving its font size and three distinct configurations. Do not move the visible Python calculation out of print.
5. If page 264 remains roughly half empty before `compile_lifecycle`, reduce that figure's top/bottom padding or vertical connector spacing by about 15–20%, then re-render. Preserve the miss/hit branches, explicit admission policy, and caption.
6. Do not split the HIP reduction across pages or shrink code typography to fill page 255. A clean code block is worth some whitespace.

**Acceptance:** no new overflow, split identifier, orphaned status label, or detached caption; all figure text remains comfortably readable at the book's normal viewing size. A page-count reduction is welcome but is not itself the goal.

## Leave the rest alone

The middle and ending now carry the reader well. In particular:

- The Pallas sequence from complete-row windows to the fragment-dependency diagram and `interpret=True` boundary is the chapter's best example of concrete design followed by an evidence boundary.
- The NKI `(128, 512)` fragment is short, current, and specific enough to teach without pretending to be a full kernel.
- The KV figure, source-visible calculation, and admission ledger form a coherent progression from logical bytes to physical placement to deployment capacity.
- The compile-lifecycle figure, eight-variant table, and 960-call arithmetic expose three different decisions—cache identity, readiness, and amortization—and should not be collapsed.
- The five exercises match the section's depth. The final sentence at line 1883 creates a clean handoff from single-device portability to distributed ownership and failure; do not add another summary between it and Part V.

## Round 2 readiness judgment

This is a professional, technically serious chapter that now needs compression and ordering, not expansion. The highest-value bounded revision is: remove the duplicate opening table; state the current Neuron serving path once before beginning an uninterrupted NKI tile lesson; move RMSNorm range qualifications behind the worked example; normalize provenance labels; then re-render and tune only the figures still causing conspicuous gaps.

Those changes should make the prose feel less caveat-led while preserving every substantive mechanism and exercise. They are one editorial/layout cycle, not completion of the wider three-hour campaign.
