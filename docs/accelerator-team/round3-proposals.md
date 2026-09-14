# Round 3 improvement proposals

**Baseline:** `973d577`  
**Role:** proposer, September 13, 2026

This proposal reconciles the independent technical, editorial, learning-integration, and author-side reviews. No reviewer found a reason to reopen Chapter 36's structure or add another accelerator catalogue. The useful Round 3 work is a bounded numerical contract fix, four scope corrections at existing teaching boundaries, a genuine transfer exercise, exact reader navigation, selective prose repair, and honest current-source metadata.

Keep Chapter 36's current worked decisions, CPU KV calculation, Pallas/NKI/HIP/Triton material, figures, and evidence ladder stable. The only companion change associated with that chapter is a clarifying docstring in its CPU planning helper. Do not add vendor feature matrices, new hardware claims, or another general compatibility checklist.

Finding keys used below:

- `T1`-`T4`: Round 3 technical findings in `round3-technical.md`.
- `E1`-`E5`: Round 3 editorial findings in `round3-editorial.md`.
- `I1`-`I4`: Round 3 integration findings in `round3-integration.md`.
- `A1`-`A2`: attention-range and CuTe notes in `round3-author-notes.md`.

## Package 1 - Make the attention oracle reject numerical failure without breaking masked-shard identity

**Order:** first; this is the only genuine reference-code correctness fix.  
**Closes:** `T1`, `I2`, `I4`, `A1`.

Keep `examples/attention.py` as a clarity-first Python-float semantic reference. Do not replace it with a scaled-accumulator implementation. Add explicit range checks at the four boundaries the algorithm actually relies on:

1. In `partial`, reject any computed visible score that is not finite before taking the maximum or subtracting it.
2. Require shifted mass to be positive and finite for a nonempty visible shard, and require every unnormalized weighted-value component to be finite.
3. In the merge, distinguish the valid zero-mass identity from invalid arithmetic. Skip only states whose mass is exactly zero after their contract has been checked; reject non-finite maxima, factors, denominator, merged numerators, or normalized outputs.
4. Use one named `ValueError` message or small validation helper for out-of-range attention arithmetic. The contract should say that finite inputs are necessary but not sufficient: scores, partial numerators, and merge results must fit Python float.

Extend `tests/test_examples.py::AttentionTests` with four focused cases:

- `attention([1e308], [[1e308]], [[1.0]])` raises for score overflow in dense and explicitly sharded calls;
- `attention([0.0], [[1.0], [2.0]], [[1.7e308], [1.7e308]])` raises both when the partial numerator overflows in one shard and when individually finite partial numerators overflow only during a `shard_size=1` merge;
- a mixed mask with one fully masked shard and one visible shard returns the independent dense result (the small fixture may use zero query, values `[[2.0], [4.0]]`, mask `[False, True]`, and `shard_size=1`, expecting `[4.0]`);
- a genuinely all-masked row still returns the declared zero vector even when the masked score inputs would overflow if evaluated.

Then connect the executable lesson rather than leaving it discoverable by inference:

- Beside the empty/fully masked DCP merge rule at `manuscript/05_distributed.md:841`, link to the dependency-free, single-head semantic reference and its mask/partition tests. Explicitly say it is not a distributed runtime or DCP performance test.
- In the attention row of `examples/README.md`, give the exact test command `python -m unittest tests.test_examples.AttentionTests -v` while retaining `python -m examples.attention` as the demonstration command.
- In the Part IX capstone row at `manuscript/09_appendices.md:41`, either include that exact test command or point unambiguously to the example-index command. Prefer one canonical command rather than duplicating command prose in two places.

**Alternative:** If the Part V source link interrupts the derivation, put the link in a one-sentence note immediately after the equations rather than in line 841's already dense paragraph. The minimum acceptable package remains the arithmetic checks, all four regressions, and one exact learner command.

**Acceptance:** both adversarial inputs fail loudly in dense and sharded paths; ordinary finite fixtures retain their existing answers and tolerance; mixed masked/nonempty shards merge correctly; all-masked rows retain the explicit zero convention; empty overall K/V remains invalid. The full CPU suite passes with no GPU dependency or device claim. The test count recorded later must come from the completed suite, not from the number of test methods estimated here.

**Bounded delegation:** the implementation and unit tests in `examples/attention.py` and `tests/test_examples.py` can be delegated as one isolated code task. The main author should integrate the Part V and capstone wording after inspecting the exact behavior.

## Package 2 - Scope state, capacity, and DCP terms where readers first use them

**Order:** second, after the oracle contract is fixed.  
**Closes:** `T2`, `T3`, `T4`, `I1`.

Make four local corrections; none requires editing Chapter 36.

1. At `manuscript/04_cuda.md:1207`, begin with “For the running full-history GQA model” before defining token-growing KV. Keep the existing formula. Add one short bridge to Part I: hybrid ports and handoffs must preserve recurrent matrices, convolution history, and attention KV at the same accepted prefix boundary. Do not restate the earlier hybrid arithmetic.
2. At `manuscript/04_cuda.md:1294`, replace “Quantized K/V reduces capacity and bandwidth.” State positively that quantization reduces stored KV bytes and potential read traffic, thereby increasing the context/request capacity available from a fixed memory budget, subject to scale metadata, packing, and a supported kernel. Do not infer lower latency from the byte reduction.
3. At `manuscript/05_distributed.md:893`, replace the exhaustive-sounding “Relevant only when...” with a scoped row such as: “May attend to already DCP-sharded history; backend-specific prefill modes require their own group and weight plan.” Add one sentence after the ledger explaining that a shared DCP flag name does not establish identical prefill and decode group geometry; point back to the SDK-scoped Neuron case already taught in Chapter 36. Preserve the present PCP/DCP distinction and do not relabel every prefill implementation PCP.
4. At `examples/accelerator_portability.py:82-87`, define `cp` as the chapter's abstract independent capacity axis: it adds ranks in this model and is neither PCP nor DCP nor a portable engine flag. Keep the public parameter and arithmetic stable; do not turn the helper into an engine-layout planner.

**Recommended scope:** stop after these four corrections. The optional NixlConnector scale example in `T4` is useful only if it replaces the generic cache-format warning at `manuscript/03_inference.md:922`. If selected, add at most two dated sentences: matching KV dtype is necessary, while support for transported scales depends on whether scales are independently loaded/packed or generated as separate runtime block metadata in that connector. Do not generalize the inspected restriction to other connectors and do not add the same note in Part IV.

**Alternative:** Omit the optional connector example and leave line 922's current scale/layout ledger intact. The four corrections above fully close the correctness and terminology findings without new product-specific prose.

**Acceptance:** the uniform KV formula is visibly a running-model assumption; hybrid state is named without a second derivation; “capacity” is no longer reversed; the DCP table admits backend-specific prefill modes without conflating them with decode; the helper cannot reasonably be read as accepting an engine DCP/PCP flag. The existing TP16/DCP2 and independent TP4/CP2 calculations remain unchanged and distinct. Chapter 36's body and figures do not move.

## Package 3 - Replace the duplicated heavy-hitter answer with a bounded transfer problem

**Order:** third; it removes repetition before the final prose/render pass.  
**Closes:** `E1`.

Keep the complete event-time design at `manuscript/06_coding_and_design.md:850-940`; it is the better worked lesson. Replace Exercise 10 at line 953 and its repeated answer at lines 1139-1155 with one changed-assumption problem using the same model:

- five-minute tumbling windows;
- 20 minutes of allowed lateness;
- 50,000 active tenant-subshard pairs;
- `304,528` bytes per dense sketch/window;
- a measured eight-minute watermark lag;
- an operator configured with a hard six-window retention cap.

Ask the reader to calculate the nominal live-window requirement and memory, compare it with the cap, and state the correctness and recovery consequences. The worked answer should derive:

`ceil((5 + 20 + 8) / 5) = 7` nominal windows,

approximately `99.3 GiB` of dense sketch tables for seven windows versus `85.1 GiB` for the six-window cap, a deficit of approximately `14.2 GiB` before candidates, deduplication, object overhead, and snapshots.

The decision is the teaching point: the seventh window cannot be silently finalized or evicted merely to satisfy the cap. The service must increase/tier capacity, apply declared backpressure or durable buffering, or evict only under a visible incomplete-result/late-data policy. Checkpoint and restore must preserve the affected window's completeness/revision state and the corresponding source-offset boundary.

**Alternative:** Delete Exercise 10 and its solution, leaving nine exercises, if even this bounded transfer would make the chapter feel over-assessed. Do not retain the current prompt and merely shorten its answer; that still asks the reader to copy the preceding design.

**Acceptance:** the `70.9 GiB` worked calculation appears only in the main-body design; Exercise 10 changes one operational assumption rather than introducing a new algorithm; its arithmetic reproduces `99.3`, `85.1`, and `14.2 GiB` to the stated rounding; the answer names one capacity response and the non-silent completeness/recovery rule. No sliding-window or new-sketch excursus is added.

## Package 4 - Make navigation exact and the prose selectively more human

**Order:** fourth, after technical wording and exercise pagination settle.  
**Closes:** `E2`, `E3`, `E4`, `E5`.

### Exact destinations

At `manuscript/09_appendices.md:433-453`, keep the symptom and “first model” columns, but replace every shorthand destination with exact table-of-contents titles, preferably with the part number. Use stable internal links only if the builder and PDF verifier preserve them inside tables. At minimum resolve these ambiguous rows as follows:

- “Prefill, scheduling, serving” -> **Prefill, Decode, and Performance Modeling**; **Scheduling, Batching, and Admission Control**; **Production Architecture, Capacity, and Reliability**.
- “Paged KV cache” -> **KV Cache, Paging, and Prefix Reuse**.
- “Speculative decoding” -> **Sampling, Structured Output, and Speculative Decoding**.
- “Optimization, recipe” -> **Optimization as a Coupled Dynamical System**; **Designing a Training Recipe**.
- “Measurement, serving” -> **Measurement and Experimental Judgment**; **Production Architecture, Capacity, and Reliability**.
- “Tiled matrix multiplication” -> **Hierarchical Matrix Multiplication**; optionally add **Kernel Engineering, Profiling, and Correctness** if the symptom is explicitly low achieved throughput.
- “Transformers, distributed systems” -> **Mixture-of-Experts and Sparse Communication** and, only if architectural routing is needed, **Transformer Architecture as Resource Allocation**.
- “Strategy and leadership” -> **Strategy, Vision, and the First 90 Days**.

Apply the same exact-title rule to the already close-but-not-verbatim decode, quantization, and serving rows. Do not turn this into a comprehensive subject index.

### Three seams and one checklist handoff

Add at most one sentence at each location:

1. After `manuscript/05_distributed.md:1440`, connect distributed execution to Part VI by saying that valid-progress claims become operable through replayable telemetry, versioned summaries, and control paths.
2. Before `manuscript/06_coding_and_design.md:1157`, state that the streaming arc is complete and that the next chapter is an optional implementation refresher; application-design readers may continue at the system-design or RAG chapter.
3. After `manuscript/06_coding_and_design.md:1638`, say that Part VII applies these state/evidence contracts to deciding which frontier mechanisms transfer to a real workload.
4. After the platform checklist at `manuscript/02_training.md:1263`, say that the casework tests whether those release claims survive deletion, contamination, partial failure, and mixture changes.

Each sentence should explain the causal handoff; avoid “In the next chapter, we will...” cadence.

### Five lead replacements, no expansion

Revise only the most visible corrective-thesis cluster, keeping each lead the same length or shorter:

- `manuscript/02_training.md:143`: open with tracing a removed source through raw, packed, release, and checkpoint artifacts; make versioned-product status the consequence.
- `manuscript/02_training.md:357`: “Every deduplication or quality threshold changes probability mass in the learned distribution”; then name the statistical-intervention rule.
- `manuscript/04_cuda.md:953`: lead with dense attention computed through a tiled memory schedule; keep the exact-versus-approximate boundary in the second sentence.
- `manuscript/04_cuda.md:1112`: lead with asynchronous-engine supply and intermediate ownership; remove the generic “faster matrix units do not automatically” opener.
- `manuscript/06_coding_and_design.md:297`: say positively that moments, samples, and tail/window summaries answer different questions and therefore carry different contracts.

**Alternative:** If the final render makes page movement costly, treat the exact Decision Index and four seam sentences as required and defer two of the five lead replacements. Do not compensate by adding generic chapter summaries.

**Acceptance:** every Decision Index destination is verbatim-findable in the table of contents and follows one location convention; the four handoffs total four sentences; roughly five leads change with no caveat loss or net paragraph growth; no other chapter ending receives formulaic signposting. Part III/IV maps and the four reference-style Part IX openings remain unchanged.

## Package 5 - Reconcile current metadata, add the missing primary link, and close with proportional QC

**Order:** last, using results produced by Packages 1-4.  
**Closes:** `I3`, `A2`; validates all other packages.

1. In `README.md:15`, change “latest five-pass revision” to an explicitly historical checkpoint description. Keep its 445-page/95-test evidence attached to that checkpoint. Add a separate current-source sentence only after the final suite and PDF complete, linking to `docs/accelerator-team-review-2026-09-13.md` and reporting the actual new test count, page count, outline count, and evidence limits.
2. Update the top “Current reviewed source” block of `docs/release-readiness.md` to the completed accelerator-team checkpoint after final validation. Move or relabel the five-pass numbers as historical; do not rewrite the immutable September 7 or five-pass review records.
3. At the existing CuTe DSL mention in `manuscript/04_cuda.md:1187`, link “CuTe DSL” to the official CUTLASS Python DSL overview recorded in the author notes. This is a citation repair in an existing implementation-boundary sentence, not an invitation to add TileLang or another DSL list.
4. Append the selected Round 3 packages, test/build/link evidence, visual scope, and unrun accelerator gates to the active campaign record. Do not mark the three-hour campaign complete or pause its heartbeat before the stated deadline and final gate.
5. Run the supported CPU suite, manuscript/build verifier, formatting/whitespace checks, and relevant link checks. Rebuild the PDF and inspect every page changed by Packages 2-4, plus both sides of each part seam and the capstone/index pages. Chapter 36 itself needs a regression visual check, not another editorial rewrite.

**Alternative:** Avoid a second volatile hard-coded test count in the README by naming the five-pass number as historical and linking the current campaign record for the live result. `docs/release-readiness.md` should still contain one unambiguous current reviewed-source snapshot after closure.

**Acceptance:** README, release-readiness, and the campaign record agree on the current checkpoint's measured counts and clearly distinguish them from historical 95/445 evidence; all hardware/device checks remain explicitly unrun unless actual captured evidence changes that; the CuTe link resolves and adds no catalogue prose; `git diff --check`, CPU tests, PDF verification, and relevant URL checks pass. Visual closure names exactly which changed pages were inspected and does not claim a new whole-book copy edit.

## Finding-to-package map

| Finding | Package | Resolution |
| --- | --- | --- |
| `T1`, `A1` attention arithmetic overflow | 1 | Reject non-finite score, partial, merge, and output arithmetic with focused regressions. |
| `I2` masked partition merge untested | 1 | Mixed empty/nonempty shard oracle plus Part V semantic-reference link. |
| `I4` capstone lacks exact partition-test command | 1 | One canonical `AttentionTests -v` route from the example index/capstone. |
| `T2` full-history KV universalized | 2 | Scope the formula to the running GQA model and point to hybrid state. |
| `T3` DCP prefill “only” too broad | 2 | Admit backend-specific prefill modes and separate their group/weight plans. |
| `T4` capacity wording reversed | 2 | Distinguish reduced footprint/read traffic from increased fixed-budget capacity and achieved latency. |
| `I1` helper `cp` can be mistaken for an engine flag | 2 | Clarify the abstract independent capacity axis without changing the API. |
| `E1` heavy-hitter design duplicated | 3 | Replace it with the seven-window/six-window-cap transfer decision, or remove it. |
| `E2` Decision Index destinations are shorthand | 4 | Use exact TOC titles and one consistent location convention. |
| `E3` three weak navigation seams | 4 | Add causal one-sentence handoffs only at V-VI, the Part VI route fork, and VI-VII. |
| `E4` clustered corrective leads | 4 | Replace approximately five with positive mechanism/scenario openings. |
| `E5` platform chapter ends as notes | 4 | One evidence-gate sentence into data-system casework. |
| `I3` README current/historical count ambiguity | 5 | Preserve five-pass history and report the completed current checkpoint separately. |
| `A2` unlinked CuTe mention | 5 | Link the existing phrase to the official CUTLASS Python DSL overview. |

## Selection guidance

Packages 1-3 are recommended without expansion: they correct behavior and replace repetition with a better learning transfer. Package 4 is also recommended, but its lead edits should stay selective; exact navigation and seam handoffs matter more than achieving a stylistic quota. Package 5 is the mandatory closure for whichever subset is selected.

The intended result is nearly size-neutral. One repeated solution disappears; a few sentences are replaced rather than added; navigation and metadata become more precise. Chapter 36 remains the stable center of the completed accelerator work instead of becoming the destination for every adjacent correction.
