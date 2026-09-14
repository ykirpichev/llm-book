# Round 3 editorial quality check

## Interim verdict

**Prose and pedagogy pass, with three micro-edits requested and visual closure pending.** The integrated revision is essentially size-neutral across the five manuscript files (46 inserted lines, 42 removed), removes the duplicated heavy-hitter answer, and improves navigation without turning every chapter boundary into a template. Chapter 36's body remains stable.

I read the actual current diff and its surrounding prose in Parts II, IV, V, VI, and IX. I also checked all Decision Index destinations against the current level-two headings and ran the focused `AttentionTests`; all ten pass. This report does not yet claim visual inspection of the changed pages.

## What passes

- The five revised leads are shorter or comparably sized and more affirmative. The Data Contracts, Blackwell, and Online Statistics openings are especially natural.
- The transfer exercise now changes a real operational assumption. Its seven-window derivation and `99.3 GiB` versus `85.1 GiB` comparison are clear, and the answer makes completeness/recovery—not another sketch catalogue—the decision. `70.9 GiB` now appears once, in the main worked design.
- The four handoffs are bounded to one sentence each. The internal Part VI route marker is useful where the reader actually chooses a path; there is no book-wide layer of formulaic chapter summaries.
- Every Decision Index destination is an exact current chapter title prefixed by the correct part. The new manuscript regression test enforces that property rather than relying on editorial memory.
- The full-history GQA qualification and hybrid-state bridge read cleanly. The quantized-KV sentence now distinguishes footprint/read traffic from fixed-budget capacity without promising speed.
- The Part V reference to the CPU attention fixture clearly limits the evidence to single-head semantics, and the companion index now gives the exact test command.

## Requested micro-edits before visual closure

### 1. Make the revised deduplication lead technically direct

**Location:** `manuscript/02_training.md:357`.

“Every deduplication or quality threshold changes probability mass in the learned distribution” attributes the direct effect to the learned model. The direct object of filtering is the empirical training distribution; model behavior is downstream and not mechanically determined.

Suggested direction: “Every deduplication rule or quality threshold changes the empirical training distribution. Evaluate filtering and contamination controls as statistical interventions, with explicit effects on coverage, repetition, and held-out evidence.” This is both more precise and less abstract.

### 2. Remove two local repetitions introduced by the handoff/lead pass

**Part II location:** `manuscript/02_training.md:1265-1269`.

“The following casework...” is immediately followed by the chapter lead “The following cases...”. Keep the lead and make the handoff a conclusion: “A release platform earns trust when those claims survive deletion, contamination, partial failure, and a changed mixture.” The heading and lead can then perform the forward navigation once.

**Part IV location:** `manuscript/04_cuda.md:953-955`.

The revised lead says FlashAttention changes the schedule rather than the attended keys; the next paragraph immediately defines exactness as the same dense operation without sparse or low-rank approximation. Keep the strong first sentence of the lead and let the next paragraph carry the exactness qualification. Removing the lead's second sentence improves cadence without losing content.

### 3. Let the prefill subject do the action in the DCP table

**Location:** `manuscript/05_distributed.md:895` in the current working tree.

“DCP may attend to sharded history” compresses the row enough to make the mechanism sound like the attention subject. Prefer: “A prefill chunk may attend to DCP-sharded history; backend-specific prefill modes require their own group and weight plan.” This retains the technical review's scope correction while reading as finished prose rather than a ledger note.

These edits should be replacements, not additions; none requires reworking the surrounding explanation.

## Visual closure checklist

When the new render is available, inspect:

- both revised Part II leads and the platform-to-casework boundary;
- the FlashAttention and Blackwell opening pages;
- the full-history/hybrid state paragraphs and KV-quantization paragraph;
- the DCP table, new CPU-reference paragraph, Neuron scope paragraph, and Part V-to-VI handoff;
- the transferred Exercise 10 prompt and solution, especially equation/code-style wrapping and the route marker before references;
- the Part VI-to-VII handoff;
- the Part IX capstone command row and every page occupied by the expanded exact-title Decision Index.

Pass conditions are no stranded one-line handoff, no table row whose destination becomes visually ambiguous, no split title that looks like two destinations, no clipped inline arithmetic, and no new near-empty page caused solely by a seam sentence. Whitespace alone is not a defect if the alternative would split a table, code panel, or conceptual unit.

## Final-render inspection

I inspected the requested 460-page final-render PNGs at original detail:

- revised leads on pages 59, 66, 226, 232, and 343;
- Part II, V-VI, the Part VI route fork, and VI-VII seams on pages 94-95, 331-332, 372-374, and 394-395;
- the transferred exercise prompt and solution on pages 366 and 371-372;
- the capstone route on page 444;
- the glossary/index/final-principle sequence on pages 458-460.

The requested copy edits are present. The leads and handoffs are clean and unstranded; the exercise arithmetic and identifiers wrap normally; the reference-only page 373 is sparse but acceptable before a forced chapter opener. The stale verifier phrase “Tiled Matrix Multiplication” was correctly updated to the canonical existing heading “Hierarchical Matrix Multiplication”; that is validation maintenance, not reduced coverage.

**One visual fix remains before pass:** the Decision Index is unbalanced. Page 459 carries 15 rows to the foot; page 460 repeats the header for only two rows, then the final principle, leaving roughly 70 percent of the page blank. This looks like accidental table spill rather than deliberate closing space. Split the index into two balanced tables with the same columns—approximately nine technical/performance symptoms on page 459 and eight operational/product symptoms on page 460—while preserving the exact part-title destinations and final principle. A repeated header is useful when each continuation contains a substantive group; it is awkward for two orphan rows.

After that reflow, rerender pages 459-460. Closure requires both tables to remain legible and all destination titles intact; no further prose changes are requested.

## Author closure of final visual finding

The author split the index into nine/eight-row tables with a deliberate page
break and continuation label, retaining all 17 destinations. Rebuilt proofs in
`tmp/pdfs/accelerator-team/round3-balanced-index/` show legible complete rows and
balanced closing pages; the author inspected pages 459 and 460. The updated
index regression and full 108-test suite pass; PDF verification remains 460
pages. Sol's attempted independent recheck failed because of model capacity,
so this paragraph records author verification, not an additional Sol approval.
