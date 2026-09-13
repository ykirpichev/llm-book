# Round 2 editorial QC after Packages 1–3

**Role:** independent professional-book/story/pedagogy checker, September 13, 2026  
**Material reviewed:** current `manuscript/04_cuda.md:1524–1887`, its full diff from the round-two baseline, the round-one render for comparison, and the initial full round-two render at `tmp/pdfs/accelerator-team/round2-pages/page-250.png` through `page-271.png`.  
**Verdict:** **conditional editorial/visual pass.** The revision preserves the chapter's useful depth, removes the duplicated opening table, and integrates the connector/quantization and DCP/scheduler decisions without turning the chapter into another support catalogue. Four bounded copy fixes are recommended; two visible layout defects require re-rendered closure. No substantive rollback or new example is warranted.

I inspected every rendered chapter page plus the Part V opener. The render is an intermediate artifact because figure compaction and the wrapped KV comment are already being repaired. I did not edit the manuscript, code, figures, or shared files.

## What now works

### The opening has one organizing device

Removing the taxonomy table from former lines 1538–1544 was the right choice. The portability diagram now carries the hierarchy, while the three acceptance tests and the operator → state/variants → deployment roadmap explain how to use it. The later porting-question table remains and is still valuable because it synthesizes mechanisms the reader has already encountered. No necessary source was lost with the opening table.

### RMSNorm teaches before it qualifies

Current lines 1546–1554 now move from the equation and dtype contract to the padding picture and plausible wrong result before discussing range, epsilon, tolerances, model quality, and training. This is more natural and more memorable. The moved validation paragraph remains technically serious without delaying the worked example.

### Neuron has a continuous code-reading lesson

Current lines 1696–1740 put the checked serving route and dated migration boundary before the NKI subheading. From the subheading onward, the reader can follow HBM → SBUF/PSUM → tile schedule → concrete `(128, 512)` call → simulator/device gates without jumping back into release history. The current SDK, migration, memory, API, simulator, and profiling links remain present.

### The two decisions teach rather than catalogue

The `ROCM_ATTN` example at line 1599 gives the abstract instruction “enumerate attention backends” a real consequence: an attention backend can work locally yet fail the required connector gate. It is carefully scoped to the inspected implementation rather than all ROCm.

The state section then separates weight format, compute dtype, and KV representation before contrasting independent CP with nested DCP. The eight-active-decodes/long-prompt replay at lines 1830–1832 converts scheduler terminology into a user-visible question about TTFT and token gaps. These are worked decisions tied to the same assistant, not four detached feature summaries.

The standalone Python calculation, Triton and HIP code, Pallas and NKI fragments, diagrams, variant table, and exercises all remain. The section is only four physical source lines longer than the baseline diff, although its prose is about 279 words longer; the added density is concentrated and can be improved with the cuts below.

## Recommended fix 1 — Tighten the ROCm gate after it has made its decision

**Location:** current lines 1597–1599.

The concrete connector example earns its space, but its last sentence repeats its opening: “Local attention support is only the first gate” and “A passing local attention test cannot repair a missing transfer path” make the same point. The preceding generic paragraph also ends with another version of the fallback warning.

**Exact cut:** delete `Unsupported code may fail loudly, but an unnoticed fallback or layout conversion can be equally costly.` from line 1597, because the following source-inspected example now demonstrates the issue. Delete the final sentence of line 1599, `A passing local attention test cannot repair a missing transfer path.`

Keep the source inspection, its date, narrow scope, record fields, and actual choice: keep the phases together or choose a supported backend. The paragraph will end on a decision rather than a reiterated warning.

**Reader benefit:** the new example feels like book prose instead of a warning followed by a counterexample followed by the same warning.

## Recommended fix 2 — Compress the precision contract before the memory arithmetic compounds

**Location:** current lines 1753–1755, immediately before `More ranks can replicate state`.

This is the densest new passage. It asks the reader to hold three precision meanings, one Neuron example, the 4 GiB/2 GiB contrast, scale and allocator overhead, latency, checkpoint lineage, and quality in two long paragraphs. Every concept is defensible, but the accumulation makes the transition into TP/CP/DCP arithmetic feel heavier than necessary. The opening phrase—“The cache representation completes the backend-and-connector decision”—also reaches back across the TPU and NKI tour to a decision the reader last saw many pages earlier.

**Suggested replacement direction (about half the current length):**

> Before applying the 4 GiB ledger, ask which precision changed: the weight artifact and its scales, the compute dtype, or KV storage and its scale layout. In the checked vLLM Neuron guide, FP8 KV is stored in FP8 but dequantized to BF16 for attention; weight loading is a separate path with its own restrictions.
>
> Changing weights alone leaves this assistant's two-byte KV payload at 4 GiB. A supported one-byte KV format stores 2 GiB of elements before scales, padding, replication, and reservations. Measure the full allocation and latency. Treat a requantized checkpoint as a new artifact, preserve the original baseline, and rerun the quality gate.

Retain both links if the weight-loading sentence depends on the feature guide; do not add a quantization table. The final current sentence about “a specific model artifact, backend, and cache/connector contract—not a precision label” can be cut because the two replacement paragraphs have already demonstrated it.

**Reader benefit:** the arithmetic remains reproducible, but the reader enters the parallelism section carrying one distinction—2 GiB is element payload, not total allocation—instead of a full acceptance checklist.

## Recommended fix 3 — Turn the DCP paragraph into a direct contrast, then shorten the constraint ledger

**Location:** current lines 1789–1793.

The DCP calculation is worth keeping. It corrects a likely misuse of the independent-CP helper and produces a memorable `16 ranks, not 32` result. Its current opening, however, is abstract—“An engine's decode-context-parallel flag can mean something different”—and the following paragraph reads like condensed release notes.

**Exact opening direction:** replace the first two sentences with `The preceding CP example adds ranks; vLLM's decode context parallelism reuses them within TP.` Then continue with the checked Neuron TP=16/DCP=2 mapping and its 256 MiB/rank, 4 GiB aggregate result.

**Constraint cut:** compress line 1791 to three jobs only:

1. state the arithmetic divisibility/bound conditions relevant to this geometry;
2. state that prefill uses different groups and whole-model support remains to be established;
3. keep the independent-CP helper distinct and require recorded group membership.

The sentence `The documented path also excludes attention data parallelism` can move to the linked target record or be folded into the second job; it does not affect the worked payload calculation and currently widens the paragraph into a feature inventory. Avoid repeating both `that alone does not establish support` and another generic validation sentence.

**Reader benefit:** the passage reads as one mathematical contrast rather than a definition, a second definition, five constraints, and a warning. The important technical boundary—DCP is nested inside TP—becomes the grammatical subject.

The admission paragraph at line 1793 should otherwise remain. It translates the calculation into the constraining-rank decision and connects naturally to compiled variants.

## Recommended fix 4 — Let the phase table hand directly to the scheduler replay

**Location:** current lines 1820–1832.

The scheduler replay is strong, but the paragraph between it and the phase table repeats much of the table before reaching quality. Its first two sentences (“A working forward pass…” and “Similarly, fast prefill…”) are now redundant: the table already separates training, prefill, and decode, and the replay immediately demonstrates why prefill and decode differ.

**Suggested replacement:**

> Compare serving candidates on the assistant's request distribution and p99 targets. Check logits against a trusted reference with justified tolerances, then run the task-level quality evaluation. Different reduction orders can change generated tokens without implying a bug; convincing-looking output can also hide one.

Then proceed directly to `Replay eight active decodes while a long prompt arrives.` Keep both scheduler paragraphs and the same-vendor connector callback at line 1834; they form a coherent observation → trace → decision sequence.

**Reader benefit:** the table supplies the taxonomy once, and the prose immediately turns it into an experiment. This is the clearest place to recover words without touching new technical depth.

## One optional positive-voice polish

**Location:** final sentence of the Neuron routing paragraph, current line 1698.

The paragraph is structurally fixed, but its last clause returns to the old caveat cadence: `support in the umbrella SDK does not establish plugin support for older Inferentia hardware.` A more direct ending is: `Match the assistant's attention, quantization, sampling, cache features, and target generation to the plugin's documented support.`

This retains the target-generation boundary and lets the paragraph finish with an actionable route. It is optional because the current sentence is accurate and the lesson already flows.

## Initial round-two render: required repairs and verified strengths

The opening-table removal has fixed the old page-252 problem. Page 251 now contains the portability diagram and roadmap; page 252 keeps the RMSNorm heading, contract, full figure, and numerical explanation together. The status labels and all code remain readable through the HIP, Pallas, and NKI tour. The NKI flow across pages 259–261 is clear, the compile-lifecycle figure and surrounding prose on pages 265–266 need no further compaction, and Exercises 2–3 plus their solutions render cleanly on page 269. The closing transition on page 270 leads naturally to the Part V opener on page 271; the remaining white space at the end of page 270 is appropriate for a part boundary.

Two defects must be closed:

1. **Pages 258–259:** page 258 ends roughly halfway down because the fragment-dependency figure advances to page 259. Compact the figure's internal vertical gaps by the planned 30–40 points while preserving type size, both fragment paths, the combined inverse RMS, and the dependency caption. Re-rendering must show that the figure fits naturally after its explanation or otherwise materially reduces the gap; do not split the figure.
2. **Pages 263–264:** page 263 contains only the KV figure, while its status and code advance to page 264. In the code, the final `d` in `excluded` wraps onto a renderer continuation line (`> d`), making a valid Python listing look damaged. Shorten or remove the inline comment and compact only the figure's internal vertical whitespace, preserving font size and all three layouts. Acceptance requires an intact `one_byte_kv` assertion, no `>` continuation, and a materially better figure/code grouping.

One pagination blemish should resolve through copy rather than art changes: the `ROCM_ATTN` paragraph breaks after `one compatible` at the foot of page 254, leaving its final two lines above the HIP subheading on page 255. Recommended fix 1 removes exactly the repeated sentences responsible for most of that spill. Verify that the revised paragraph ends cleanly and that the HIP heading remains attached to its introduction.

Page 267 is dense but readable; the phase table, scheduler replay, and handoff arithmetic form one continuous argument with no overflow. The copy cut in recommended fix 4 should add breathing room there. Do not reduce code type, split the HIP reduction, or compress the compile figure, which already renders well.

The closure render should again cover the entire chapter because the early copy cuts and figure-height changes can shift all later breaks. Inspect the opening, pages containing the two compacted figures, the KV code, DCP equations, revised phase paragraph, exercises, and Part V transition.

## Final judgment

Packages 1–3 are editorially successful. The chapter now has a natural professional arc: establish the portability boundary, carry one operator across execution models, make two concrete engine/deployment decisions, and finish with observable workload evidence. The remaining work is compression at four exact seams, not restructuring and not expansion.

After those cuts and a clean full render, this revision should pass round-two editorial QC. No target execution or benchmark claim follows from this review.

## Round 2 closure

**Final status: pass.** I inspected every page in the final 458-page build from Chapter 36's opener on page 250 through its conclusion on page 269 and the Part V opener on page 270.

The four copy fixes are reflected cleanly. The `ROCM_ATTN` decision now ends on page 254 before the HIP subheading; the precision, DCP, and scheduler passages retain their sources and worked consequences with materially better cadence. The fragment-dependency explanation and compacted figure fit together on page 258 with all labels and the caption legible. Page 262 now pairs the independent-CP prose with the compacted KV figure, and the complete self-contained Python block on page 263 has an intact `one_byte_kv` assertion and comment with no renderer continuation. The DCP equations and `16 ranks, not 32` result are readable. The compile-lifecycle figure remains clear, the exercises and solutions have no overflow, and the chapter-to-Part V transition is clean.

The unused lower area on page 262 is acceptable: the next artifact is a kept-together executable code block, and forcing it upward would trade page rhythm for a split listing or smaller type. No further figure compaction or prose change is warranted for round-two closure. This editorial pass does not upgrade any source-only accelerator example to device evidence.
