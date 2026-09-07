# Draft review and priorities

Reviewed: 2026-09-07. Status: working draft, not a publication sign-off.

## Overall assessment

The book has a useful identity: engineering decisions across the complete LLM lifecycle, supported by resource models and failure analysis. Inference and CUDA are particularly strong because they connect calculations to a running workload rather than just listing techniques. The design exercises, worked solutions, and field-reference decision index make the material useful beyond a first read.

The next step should be consolidation, not another broad expansion. The main issues are uneven depth, limited executable verification, and inconsistent proximity of evidence to technical claims. Layout is generally coherent, but the sampled pages show a few problems worth fixing before release.

This review covers the expanded local manuscript combined with the remote inference and CUDA improvements through commit `6f1ed81`. It is a structural/editorial review with source spot-checks and sampled visual inspection, not a line-by-line technical certification.

## What is in the draft

The current build is 338 pages and approximately 90,000 words across nine parts. Word counts below are approximate and include Markdown, examples, and exercises; they measure allocation of space, not quality.

| Part | Approximate words | Assessment |
| --- | ---: | --- |
| I — Foundations | 6,800 | Useful systems framing; needs a primary-source reading trail. |
| II — Training | 14,800 | Substantial data and recipe coverage; review the depth of post-training relative to the rest. |
| III — Inference | 15,700 | Strong running-service narrative and quantitative reasoning. |
| IV — CUDA | 13,700 | Strong running workloads; make code verification and assumptions explicit. |
| V — Distributed systems | 14,000 | Broad coverage; extend the shared numerical workload across machine boundaries. |
| VI — Coding and design | 14,200 | Most uneven part: streaming algorithms occupy about 88% of it. |
| VII — Recent state of the art | 3,100 | Keep explicitly dated and distinguish paper results from general conclusions. |
| VIII — Technical leadership | 4,500 | Useful practical guidance; concrete anonymized decisions would strengthen it. |
| IX — Field reference | 2,200 | Useful navigation aid; keep synchronized with chapter revisions. |

## Prioritized work

### 1. Rebalance Part VI — next writing task

In [Coding and system design](../manuscript/06_coding_and_design.md), “Streaming Algorithms and Top-K” is approximately 12,450 words. It covers much more than top-K: windows, statistics, sampling, sketches, membership, quantiles, and event-time pipelines. Meanwhile, production ML algorithms, design method, and RAG/evaluation are approximately 614, 596, and 470 words respectively.

Preserve the valuable streaming material, but split it into navigable chapters with more accurate titles. Then expand RAG and evaluation into one worked service, not a longer catalogue of components.

Suggested case: a documentation assistant with versioned documents, access controls, freshness requirements, and a latency/cost target. Follow ingestion, chunking, indexing, retrieval, reranking, generation, and evaluation. Show at least one failed design and its measured correction.

Completion criteria:

- State the workload, user needs, access policy, and evaluation dataset.
- Separate retrieval quality from answer quality and citation correctness.
- Include an explicit latency/cost budget and a reproducible small example.
- Test stale documents, access-control filtering, unanswerable queries, and conflicting sources.
- End with a design exercise and a worked answer using the same case.

### 2. Turn selected examples into verified teaching material

The manuscript contains roughly 28 fenced code blocks, while the automated tests currently cover the book builder, not the algorithms in those blocks. A successful PDF build does not establish that an example runs or is correct.

Start with a small `examples/` collection: one attention/reference calculation, one streaming algorithm, and one evaluation pipeline. Label every manuscript snippet as runnable code, an excerpt, or pseudocode. Link runnable examples to tests, dependencies, expected outputs, and numerical tolerances. Keep CPU reference tests in ordinary CI; document optional GPU checks separately.

For CUDA examples, distinguish correctness checks from performance measurements. Record shapes, dtype, device assumptions, synchronization, and what the timed region includes. Avoid promising portable speedups from illustrative code.

### 3. Make technical evidence easier to audit

[Foundations](../manuscript/01_foundations.md) currently has no external URLs. Add a short primary-source reading list and nearby citations for architecture, optimization, and scaling claims. A missing URL is not itself a factual error, but readers need a way to distinguish established results, simplifying assumptions, and the author's advice.

Across the book, audit quantitative claims with a small claim-to-source checklist: result, source/version, workload, hardware, baseline, and limitations. In [Recent state of the art](../manuscript/07_recent_state_of_art.md), keep the August 2026 scope explicit and identify whether a number is a paper-reported result or a derived example.

Limited source spot-checks found the expected papers for [test-time scaling](https://arxiv.org/abs/2608.04001), [SparseServe](https://arxiv.org/abs/2509.24626), and [RocketKV](https://arxiv.org/abs/2502.14051). These checks do not validate every related statement or extrapolation in the manuscript.

### 4. Improve continuity without repeating the same advice

Use the inference/CUDA running model as a shared example in distributed serving and selected training calculations. Explicitly flag changed assumptions rather than quietly introducing different models or hardware.

During a prose pass, reduce repeated reminders to define contracts, identify bottlenecks, and measure. Keep the principle once, then use later space for actual numbers, a failure, or a decision. Review exercise coverage so adjacent chapters do not alternate unexpectedly between questions alone and fully worked solutions.

For leadership, add a few clearly labeled original or anonymized cases: a migration decision, an experiment that changed the roadmap, and an incident with competing priorities. Do not invent personal experience.

### 5. Finish the reading experience

The navy/teal hierarchy, callouts, and tables are consistent in the sampled PDF pages. Remaining issues include:

- Chapter titles repeat immediately below the chapter banner (for example PDF pages 10, 20, 100, and 250). Consider using the banner as the sole visible title while preserving navigation anchors.
- The agent-policy diagram on PDF page 302 extends beyond its background panel on the right. Reflow its nodes and arrows within the available width.
- Code on PDF page 85 is small and low-contrast against the dark panel. Check code typography at normal reading size and in print.
- Some chapter endings have substantial whitespace (for example pages 41, 190, and 220). This may be intentional chapter pagination, but review the tradeoff between space and total book length.

These are findings, not fixes applied to the manuscript or typesetter in this cleanup. Page numbers refer to the 338-page checkpoint and will change after revisions.

## Verification performed

| Check | Result and limitation |
| --- | --- |
| Builder unit tests | Six passing tests, including an added inline-code multiplication/emphasis regression. Does not test manuscript algorithms. |
| Build and structural checks | PDF built successfully; 338 pages, 65 outline entries, 73 external link annotations; no suspiciously empty pages flagged. |
| Link reachability | 72 distinct URLs checked; no 404/410 responses. Four publisher/DOI URLs returned HTTP 403 and need manual verification. Reachability does not prove claim accuracy. |
| Visual inspection | 18 sampled pages: 1, 3, 10, 20, 35, 41, 85, 100, 130, 160, 190, 220, 250, 280, 294, 302, 307, 338. Not a complete visual audit. |
| Sources | Limited primary-source spot-checks, not a complete fact-check. |

The four manually unresolved links are DOI `10.1080/00031305.1983.10483115`, DOI `10.1145/2500128`, DOI `10.1145/3600006.3613165`, and the ScienceDirect PDF with identifier `S002001900500298X`.

## Before making the repository public

- Choose explicit publication/reuse terms for prose and code; do not assume they need the same license. The existing copyright notice remains unchanged.
- Review tracked files and Git history for private material and confirm rights/attribution for third-party figures and excerpts. This cleanup is not a complete historical security or rights audit.
- Run the technical and full visual review after the next editorial pass.
- Publish a durable, versioned PDF download, such as a GitHub Release, rather than relying solely on temporary Actions artifacts.
- Keep the editable manuscript, pinned build requirements, tests, and clear build instructions together. Keep unrelated presentation archives out of the book repository.

Repository visibility has not been changed. Recommended next milestone: a rebalanced Part VI with one tested end-to-end RAG case, followed by the evidence and layout passes above.
