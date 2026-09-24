# Public edition release readiness

## Release policy

This free public edition uses AI-assisted technical review and editing, CPU
reference tests, source checks, and PDF verification. Independent human review
and professional copy editing are not release requirements. Their absence is
stated in the README and PDF imprint; no independent human certification is
claimed. This policy supersedes the draft-only restrictions in historical logs.

Book text and original figures use CC BY 4.0; original code uses MIT. The research
cutoff is September 24, 2026 for Part VI. Part VII has worked additions dated
September 24, with its frontier model map still dated September 23. Other parts
retain September 13 unless explicitly dated. This update adds no locally reproduced
accelerator benchmark evidence.

## September 24 Part VII runnable experiments

Versioned release: [public-2026-09-24-experiments](https://github.com/ykirpichev/llm-book/releases/tag/public-2026-09-24-experiments).
This focused revision prioritizes Chapters 55 and 59. See the
[experiment and scope record](part7-experiments-2026-09-24.md).
Earlier releases and their original assets are preserved.

| Check | Result and scope |
| --- | --- |
| Agent review | Mechanisms reviewer independently approved both modules, tests and chapter additions after clarifying rejection-counter precedence. Systems reviewer authored the adoption module, then reviewed the serving simulation and both chapter explanations. Both final verdicts have no blocking findings. |
| CPU/reference/document suite | All 153 tests pass, including 19 new checks for serving event timelines, memory admission, offered-workload accounting, paired evidence, timeouts, failed gates and malformed evaluation records. |
| Final PDF | 485 pages, 80 outline entries and 329 external link annotations. Structural, navigation, text, geometry and sparse-page checks pass. |
| Rendered layout | Front matter pages 1-8 and Part VII through its transition, pages 402-443, rendered at 130 dpi. All 50 pages inspected in contact sheets; new serving table and statistical explanation inspected at full size. |
| Website | 70 chapters, nine parts, 80 reading pages and 49 figures. Reader tests pass. Browser search locates adoption_analysis in Chapter 59; Chapter 55 fits a 390-pixel viewport without document overflow. |
| Sources | New NIST confidence-interval reference inspected directly. The prior Part VII source audit remains recorded below; this focused revision adds no new model-performance claims. |

The examples use invented inputs and Python's standard library. No GPU model,
engine throughput, production latency or adversarial robustness is measured.
The normal-approximation screening rule is explicitly heuristic, and the
adoption script never recommends full rollout. The release package uses the
existing allowlist, bounded source/PDF scan and per-file checksum manifest.

## September 24 Part VII worked additions

Versioned release: [public-2026-09-24-part7](https://github.com/ykirpichev/llm-book/releases/tag/public-2026-09-24-part7).
Read [Part VII online](https://ykirpichev.github.io/llm-book/part-07.html) and the
[chapter-by-chapter additions and review record](part7-followup-2026-09-24.md).
The earlier Part VI release and its original assets are preserved.

| Check | Result and scope |
| --- | --- |
| Mechanisms reviewer, Chapters 54 and 57 | Approved after correcting the AttnRes vector-coordinate example. No outstanding content findings. |
| Systems reviewer, Chapters 55, 56, 58 and 59 | Approved after clarifying paired canary comparisons, validation overhead and timeout accounting. Recovery fixture reviewed with the manuscript. |
| CPU/reference/document suite | All 134 tests pass. Eight new tests cover durable task recovery, exhausted budgets, stale generations, revoked authorization, immutable intent, conflicting receipts and missing or incompatible state. |
| Final PDF | 483 pages, 80 outline entries and 328 external link annotations. Structural, navigation, text, geometry and sparse-page checks pass. |
| Rendered layout | Front matter pages 1-8 and Part VII through its transition, pages 402-441, rendered at 130 dpi. All 48 pages inspected in contact sheets; the new residual-routing diagram and concurrent timeline also inspected at full size. |
| Website | 70 chapters, nine parts, 80 reading pages and 49 figures. Internal-link/anchor/search tests pass. Browser search finds the new recovery material in Chapter 58; Chapter 54 fits a 390-pixel viewport without document overflow. |
| Source links | All 64 distinct Part VII source URLs reachable, with zero missing or unverified targets. Reachability is separate from the reviewers' claim checks. |

The new numbers are explicitly hypothetical worked examples, not model or
accelerator benchmark results. The recovery fixture assumes one active runner
and local SQLite effects; it does not establish remote exactly-once execution.
The release source archive uses the bounded allowlist and per-file checksum
manifest. Consult the release assets and GitHub Actions for hosted build results.

## September 24 Part VI revision

Versioned release: [public-2026-09-24](https://github.com/ykirpichev/llm-book/releases/tag/public-2026-09-24).
Read [Part VI online](https://ykirpichev.github.io/llm-book/part-06.html) and its
[chapter-by-chapter gap and source record](part6-update-2026-09-24.md).
Earlier releases and their original assets are preserved.

| Check | Result and scope |
| --- | --- |
| Technical review | Authoring-agent source, arithmetic, implementation and scope review across Chapters 46-53. No separate reviewer-agent campaign claimed for this revision. |
| CPU/reference/document suite | All 126 tests pass. Eight new tests cover local transaction rollback, lost reply, separate-process replay, abrupt process exit, concurrent retries, tenant identity, payload conflict and revoked authorization. |
| Final PDF | 476 pages, 80 outline entries and 326 external link annotations. Structural, navigation, text, geometry and sparse-page checks pass. |
| Rendered layout | Front matter 1-8, all Part VI pages 332-401 and Part VII transition 402-403 rendered at 130 dpi. Contact-sheet review and full-size quantile-table inspection; revised ordering and bibliography pagination inspected again. |
| Website | 70 chapters, nine parts, 80 reading pages and 48 figures. Internal-link/anchor/search tests pass; browser search finds ToolBench-X in Chapter 53. The new retrieval section fits a 390-pixel viewport without document overflow. |
| Source links | 258 distinct manuscript URLs checked; 254 successful after retrying one temporary DNS failure, zero confirmed missing, four pre-existing publisher access blocks. The final ToolBench-X v2 URL was checked separately after pinning it. Reachability is not claim verification. |

Validation used the pinned Python 3.12/document runtime. The new reference
implements atomic local SQLite effects, not remote exactly-once execution or
production authorization. No accelerator benchmarks or cited retrieval-model
experiments were run. Published workflow results and asset hashes are available
through the release and Actions links; the source archive includes a per-file
checksum manifest and excludes Git history and unrelated output experiments.

## September 23 Part VII revision

Versioned release: [public-2026-09-23](https://github.com/ykirpichev/llm-book/releases/tag/public-2026-09-23).
Read [Part VII online](https://ykirpichev.github.io/llm-book/part-07.html) and the
[chapter-by-chapter gap, source, and review record](part7-update-2026-09-23.md).
The original September public release and its assets are preserved.

| Check | Result and scope |
| --- | --- |
| Reviewer A, Chapters 54-55 | Approved after correcting post-training coverage, account scope, exercises, and a second-pass quantization precision finding. No unresolved substantive findings. |
| Reviewer B, Chapters 56-57 | Approved after distinguishing diffusion mechanisms, concrete limitations, current commercial/open comparisons, and new exercises. No unresolved substantive findings. |
| Reviewer C, introduction and Chapters 58-59 | Approved after adaptive-security qualifications, contamination controls, and recovery/comparison exercises. No unresolved substantive findings. |
| CPU/reference/document suite | All 118 tests pass, including online-reader and source-package checks. |
| Final PDF | 469 pages, 80 outline entries, 310 external link annotations. Geometry, navigation, text, metadata and sparse-page checks pass. |
| Rendered layout | Front matter pages 1-8, all Part VII pages 395-426, and transition page 427 rendered at 130 dpi and inspected using contact sheets plus a full-size table check. Final exercise-numbering and heading-pagination fixes were rerendered; remaining inspected pages were pixel-identical to the previous render. |
| Website | 70 chapters, nine parts, 80 reading pages and 48 figures. Automated internal-link/anchor/search coverage passes. Desktop and 390-pixel browser checks and a search for new material pass. |
| Source links | 245 manuscript URLs checked: 241 successful responses, zero confirmed missing targets, four unchanged older publisher access blocks. Reachability is not claim verification. |

The reviewer verdicts concern manuscript content and primary-source support,
not independent model execution. The publication pipeline builds the exact
pushed commit; consult [Book CI](https://github.com/ykirpichev/llm-book/actions/workflows/book.yml)
and [Pages deployment](https://github.com/ykirpichev/llm-book/actions/workflows/pages.yml)
for hosted results. Release asset hashes are in the attached `SHA256SUMS`;
the source archive contains its own per-file manifest.

## Original public-edition publication record

Publication repository: [ykirpichev/llm-book](https://github.com/ykirpichev/llm-book).
The [Public Edition - September 2026](https://github.com/ykirpichev/llm-book/releases/tag/public-2026-09)
provides the reviewed PDF, source snapshot, and SHA-256 checksums. The README
links directly to the PDF and source download. Current hosted build results
are available in [GitHub Actions](https://github.com/ykirpichev/llm-book/actions/workflows/book.yml).

The author authorized publication in the existing repository, including changing
its visibility to public. This supersedes the earlier plan to use a fresh
repository. Existing branches, history, and earlier releases are retained;
no history rewrite or blanket reference push is part of publication.

The release source archive contains no Git history, local archives, preview
renders, or unrelated output experiments. Repository visibility has a broader
scope: all existing remote branches and the two historical release tags were
checked before publication. Their reachable history contains 278 scanned text
or PDF blobs and nine book-preview images. The bounded scan found no selected
credential signatures. Its only flag is the previously recorded historical
Makefile with a local Python path, not a credential. The nine images were
visually inspected as book previews. Local tool-checkpoint references are not
remote branches and are excluded from the push. See [publishing.md](publishing.md).
The two existing release PDFs (348 and 402 pages), issue/PR descriptions,
issue comments, and PR review comments also passed the bounded text scan.

## Original public-edition evidence

The previous manuscript checkpoint had 70 chapters, a 460-page PDF, and 108
passing CPU/reference/document tests. Its [completed AI review campaign](accelerator-team-review-2026-09-13.md)
records three review and correction cycles and the limits of its visual checks.

Fresh public-edition checks (September 20, 2026):

| Check | Result and scope |
| --- | --- |
| CPU/reference/document suite | 111 tests pass, including three source-package boundary checks. |
| Rebuilt PDF | 460 pages, 80 outline entries, 276 external link annotations; structural, navigation, metadata, and text checks pass; no sparse pages flagged. |
| Visual regression | All 460 pages rendered at 72 dpi and compared to the previous PDF: 457 are pixel-identical. Changed pages 1, 2, and 8 were inspected at 130 dpi; cover, license/review imprint, and scope text are clean. This is targeted release-layout review, not a new full-book copy edit. |
| Manuscript source links | 219 URLs checked; 215 successful HTTP responses, zero confirmed missing targets, four publisher access blocks. Reachability is not content verification. |
| Source boundary | Allowlisted current source files and extracted PDF text pass the bounded credential/personal-path scanner. Git history and unrelated local output are excluded. No claim of an exhaustive secret or rights audit. |
| Archive verification | The 100-file source snapshot extracts without Git metadata; source and asset SHA-256 checksums match, all relative Markdown file links resolve, and `make verify` passes from the extracted source directory (111 tests and the 460-page PDF). |

Local build and extracted-source checks used Python 3.12.14, ReportLab 4.4.9,
pdfplumber 0.11.9, and pypdf 6.10.0. Visual comparisons used pypdfium2 5.13.0
and Pillow 12.3.0. All five package pins match this tested environment; the
renderer pin was updated to the version used for the visual checks. The local
checks and hosted GitHub Actions checks are separate evidence;
consult the linked workflow for the published commit's hosted build status.

The four blocked sources are DOI `10.1080/00031305.1983.10483115`, DOI
`10.1145/2500128`, DOI `10.1145/3600006.3613165`, and the ScienceDirect PDF
identified by `S002001900500298X`. HTTP 403 responses leave them unverified;
they were not counted as successes or treated as proof of a missing citation.

Hardware compilation and execution, real serving-engine integration, distributed
cache behavior, frontier training runs, and accelerator performance remain
unverified by this project. Passing CPU examples does not validate those paths.
External-link reachability does not establish the correctness of a citation.

## Historical evidence

[Earlier release checks](release-readiness-history.md) preserve the draft-era
records and their original verdicts. They are evidence about those dated
checkpoints, not outstanding requirements for this public edition. Their older
licenses describe historical artifacts, not the licenses in this source tree.
