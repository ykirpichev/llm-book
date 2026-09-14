# Round 1 editorial quality check after integration

**Role:** independent professional-book/story/pedagogy quality checker, September 13, 2026  
**Verdict:** **conditional pass.** The integrated chapter is substantively ready for the next review round. Packages 1–5 now form a coherent teaching sequence, the requested technical depth remains, and the new material does not read like an indiscriminate backend catalog. One rendered-code defect must be fixed and verified before this round is closed. The remaining flow and pagination items are worthwhile polish, not reasons to remove the new technical examples.

## Scope and evidence

I compared `docs/accelerator-team/round1-proposals.md` with the integrated section in `manuscript/04_cuda.md:1524–1882` and with my earlier `docs/accelerator-team/round1-editorial.md`. I also inspected the regenerated 110 dpi page images for the complete chapter span, `tmp/pdfs/accelerator-team/round1-pages/page-250.png` through `page-269.png`, plus the Part V handoff on page 270. This is therefore a source-and-current-render review, not merely a Markdown review.

The review did not compile or run Triton, HIP, Pallas, NKI, or simulator code and makes no new device-validation claim. I did not edit the manuscript, examples, figures, or shared code.

## Package-by-package result

| Package | Result | Quality judgment |
| --- | --- | --- |
| 1. Factual and numerical contracts | Pass | Access/version qualifications, the scoped handoff arithmetic, and source-only/device-tested distinctions are expressed as contracts rather than buried disclaimers. |
| 2. Reader journey and executable reasoning | Pass | The roadmap at `manuscript/04_cuda.md:1548` gives the chapter a visible operator → state/variants → deployment progression. The self-contained KV calculation at lines 1771–1785 now lets the reader see the arithmetic rather than trusting imported assertions. |
| 3. Concrete TPU and NKI choices | Pass | The Pallas `BlockSpec` at lines 1679–1687, byte ledger at 1691, fragment-dependency figure at 1697, and NKI API fragment at 1734–1740 teach specific ownership and reduction decisions. They retain the crucial boundary between CPU interpretation/simulation and target execution. |
| 4. Launch and validation evidence | Pass | The HIP launch excerpt at lines 1637–1642 closes the earlier gap between a kernel body and its launch/error boundary, while the companion source is clearly kept separate from executed evidence. |
| 5. Variants and exercises | Pass | The eight-variant example and manifest table at lines 1804–1813 turn compilation into a deployment decision. The five exercises now test gates, attribution, placement, quantitative transfer budgeting, and synchronization rather than mostly asking for recall. |

## Required before closing round 1

### 1. Re-render and verify the self-contained KV listing

In the reviewed render, page 263 wraps the `logical_kv` expression from source line 1774 after `element_` and prints the continuation as `> bytes`. It therefore looks like invalid Python even though the source calculation is valid. This is especially damaging because the listing is the chapter's deliberately retained self-contained executable calculation.

The correction already in progress should break the multiplication across parenthesized source lines at semantic boundaries. Acceptance is visual, not just source-level: the regenerated page must show ordinary Python continuation with no renderer-inserted `>` and no split identifier. Recheck the complete code block because a changed line count can move the figure, status label, or following paragraph.

## Recommended editorial polish

### 2. Keep the supported Neuron path ahead of the memory-model lesson

The Neuron sequence at `manuscript/04_cuda.md:1703–1715` currently moves from ecosystem/access, to the NKI memory model, back to the NxD/vLLM release boundary and migration, and then forward again to the NKI schedule. The information is accurate and materially useful, but the chronology interrupts the conceptual motion from HBM/SBUF/PSUM to tile scheduling.

Prefer this order within the existing material:

1. ecosystem, access, and the current supported serving path;
2. a compact dated NxD/vLLM compatibility note;
3. NKI memory spaces, current namespaces, tile schedule, API fragment, and validation gates.

This is a move/condense operation, not a request for more Neuron prose. It would fulfill Package 2's stated “supported path first, working-memory model second, validation last” structure more exactly.

### 3. Use one opening taxonomy device, then reflow the large figures

The accelerator diagram on page 251 and the five-row taxonomy table immediately beneath it both introduce essentially the same implementation paths. The table adds useful negative promises, but at this point those promises are also established in the surrounding prose and backend sections. Keeping both contributes to the visibly loose pagination later in the chapter.

The current render leaves unusually large unused lower areas on pages 252, 255, 262, and 264 before kept-together code or figures move to the following page. Pages 252 and 264 are the most conspicuous. This is not a legibility failure—the figures on pages 253, 263, and 265 are clear—but four such gaps in one twenty-page chapter make the layout feel less finished than the prose.

The least disruptive remedy is to retain the opening diagram and acceptance-test paragraph, remove or sharply compress the duplicate taxonomy table at source lines 1538–1544, and then re-render before making local figure changes. That global reflow may cure several gaps at once. If page 264 remains sparse, reduce vertical whitespace within the compile-lifecycle art or place it closer to its heading; do not shrink code type or delete the substantive KV/compile explanations merely to fill pages.

### 4. Normalize two status labels without weakening provenance

The required per-listing labels are present and useful. Two labels use a different register from the otherwise clear `Source-only` / `Explanatory` / `Runnable Python` scheme:

- line 1611: `Hardware-dependent HIP kernel-body excerpt`
- line 1652: `Illustrative on-target shell workflow`

For a cleaner house style, use `Source-only HIP kernel-body excerpt` and `Source-only ROCprofiler workflow; requires ...; not run ...`. In particular, `on-target` can momentarily sound like recorded target evidence before the next sentence retracts that inference. This is copy polish, not a provenance defect.

## What should not be cut

- Keep the Pallas `BlockSpec`, the 128 KiB tile ledger, and the `interpret=True` boundary. Together they replace the earlier formula-only treatment with a code-reading lesson.
- Keep the NKI `(128, 512)`, `axis=1`, `n=512` fragment and its explicit SBUF assumptions. It is short enough to earn its place and prevents legacy-API hand-waving.
- Keep the HIP launch fragment. Five lines connect lifetime and synchronization prose to an actual launch boundary.
- Keep the standalone KV calculation after repairing its line wrap. It is the strongest bridge from the prior CUDA memory model to cross-device placement.
- Keep all five revised exercises. Exercise 4 derives a numerical decision; Exercise 5 forces the reader to distinguish mathematical width from launch participation. Both materially improve the chapter's assessment depth.

## Reader-journey and render verdict

The chapter now has a professional through-line: define what crosses the portability boundary, carry one small operator across four programming models, port the assistant's state and compilation policy, then demand phase-specific deployment evidence. The deliberate asymmetry is successful. Triton demonstrates a tiled kernel, HIP makes block participation and launch ordering visible, Pallas makes block windows and full-row dependencies visible, and NKI makes memory-space/tile contracts visible. The comparison table on page 261 then synthesizes questions rather than pretending the APIs are equivalent.

The four new explanatory figures teach rather than decorate. In the inspected render, the row-fragment dependency figure on page 259 is especially effective: it makes clear why a global divisor cannot repair fragment-local numerators without implying a universal cross-program barrier. The KV placement and compile-lifecycle figures also match their captions and surrounding calculations. The Part V opener on page 270 follows the chapter's final ownership/failure transition cleanly.

No additional expansion is needed for round 1. After the KV listing is re-rendered successfully, the chapter merits a content pass; the Neuron reorder, opening-taxonomy compression, and whitespace reflow should be handled as bounded editorial/layout polish rather than another technical expansion cycle.

## Round 1 closure

**Final status: pass.** I inspected the rebuilt 130 dpi `tmp/pdfs/accelerator-team/round1-closure/page-263.png`. The parenthesized `logical_kv` expression now renders across two syntactically clear lines; `element_bytes` is intact, there is no renderer-inserted `>` continuation, and the complete listing remains legible without clipping. The required gate above is closed.

The Neuron ordering, opening-taxonomy compression, status-label normalization, and pagination suggestions remain non-blocking round-two backlog. They do not qualify or delay this round-one pass.
