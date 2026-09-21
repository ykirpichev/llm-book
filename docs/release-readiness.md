# Public edition release readiness

## Release policy

This free public edition uses AI-assisted technical review and editing, CPU
reference tests, source checks, and PDF verification. Independent human review
and professional copy editing are not release requirements. Their absence is
stated in the README and PDF imprint; no independent human certification is
claimed. This policy supersedes the draft-only restrictions in historical logs.

Book text and original figures use CC BY 4.0; original code uses MIT. The research
cutoff remains September 13, 2026. Preparing a public edition does not imply a
new survey of later research or new accelerator benchmark evidence.

## Publication status

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

## Evidence

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
