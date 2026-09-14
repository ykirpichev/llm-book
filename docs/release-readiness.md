# Working-draft release checks

## Current source — accelerator-team completion, September 14

The [completed team campaign](accelerator-team-review-2026-09-13.md), through round 3,
builds to 460 pages, 80 outline entries, and 274 external link annotations;
108 CPU/reference and document checks pass. Three independent reviewers feed
each proposer/author/quality-checker cycle. The current changes deepen the
accelerator chapter, fix attention-reference overflow, sharpen phase/state
contracts, replace a duplicated exercise, and make the decision index exact.

The accelerator chapter was inspected page by page in round 2. Round 3 checks
the changed chapter openings, technical regions, exercise, navigation seams,
capstone, and decision index; it is not another word-by-word full-book copy edit.
All 51 Part-IV source URLs responded successfully. Other source checks and their
limits remain in the dated reports below. Astra passed the integrated round-3
technical QC. An additional Astra release pass was unavailable because of model
capacity; it is not counted as a review. Sol checked the final changed-page
layout and requested a balanced index. The author applied and visually checked
that final two-page reflow; Sol's attempted recheck also hit model capacity.

Device compilation, accelerator numerics/performance, real serving engines,
and distributed cache behavior remain untested here. This is a working draft
for technical-reviewer circulation, not a finished public edition. Independent
human technical/copy review and publication-boundary checks remain necessary.
No remote push, release upload, or visibility change accompanied this checkpoint.

## Historical reviewed source — five-pass revision, September 13

The [five-pass review](five-pass-review-2026-09-13.md) records five sequential
coverage, mechanism, source, editorial, and adversarial/production-check cycles,
with a disposition for all 70 chapters. That checkpoint's PDF has 445 pages,
80 outline entries, and 245 external link annotations; 95 CPU tests pass.
All pages were inspected at contact-sheet scale and all 44 distinct diagram
proofs were inspected with captions, with enlarged changed-page checks.

The final link probe reached 196 unique manuscript URLs: 192 HTTP successes,
no confirmed missing targets, and the same four publisher access blocks listed
below. GPU kernels, real serving engines, distributed cache behavior, and
accelerator ports were not executed or benchmarked. The source and PDF remain
a working draft suitable for technical-reviewer circulation, not a finished
public edition. Independent human technical/copy review and publication-boundary
checks remain necessary. The historical releases below are unchanged.

## Historical reviewed source — earlier September 13 multi-model pass

The [multi-model editorial review](multi-model-review-2026-09-13.md) records the
complete Astra technical, Sol prose, and Terra pedagogy review, followed by
post-integration Terra and Astra gates. That source passed 77 CPU tests.
Its rebuilt PDF had 410 pages, 78 outline entries, 168 external link annotations,
and no suspiciously empty pages. All pages were inspected at contact-sheet
scale, with enlarged checks of every materially changed region.

The online checker reached 147 unique manuscript URLs with zero confirmed
missing links; four publisher endpoints returned access blocks and remain
unverified. The repository audit found no selected credential signatures and
again reported the documented developer-local path in one historical Makefile
blob. CUDA excerpts remain uncompiled and unbenchmarked locally.

**Readiness:** suitable for broader peer and reviewer circulation as a clearly
labeled working draft. A finished public edition still needs an independent
human technical review, professional copy edit, and final rights, licensing,
privacy, and repository-history review.

The release evidence below describes the immutable September 7 artifacts rather
than this newer locally rebuilt PDF.

## Historical checkpoint: expanded end-to-end learning draft

The current release is `draft-2026-09-07-expanded`. It preserves the earlier
checkpoint below and keeps the repository private. See
[release notes](release-notes-2026-09-07-expanded.md) and the
[chapter coverage ledger](coverage-audit-2026-09.md).

| Check | Result and limit |
| --- | --- |
| Coverage | 68 chapters across nine parts; ten new chapters and targeted expansions in every part. The ledger distinguishes added, expanded, retained, and corrected chapters. |
| CPU tests | 66 passing tests, including all explicitly runnable manuscript Python blocks and finite-difference checks for the miniature model. |
| PDF | 402 pages, 78 outline entries, 159 external link annotations; structural checks pass with no suspiciously empty pages. |
| Source reachability | 145 unique manuscript URLs; zero reported missing pages (404/410). Four publisher access blocks and one ACL Anthology DNS failure remain unverified by the automated checker. |
| Visual review | All pages reviewed at contact-sheet scale; changed sheets rechecked after pagination fixes. Enlarged checks cover equations, code, tables, chapter openers, and the capstone. This is not a word-by-word copy edit. |
| Experimental scope | CPU mechanism fixtures are tested. CUDA snippets, frontier training, distributed RL, and real-model agent robustness were not experimentally reproduced. |

The additional unverified URL is `https://aclanthology.org/D18-2012/`.
The four publisher URLs are listed in the historical evidence below. No
unverified response is counted as a successful content verification.

Final local review images are in `output/previews/coverage-release/` and
`tmp/pdfs/coverage-release/`. Superseded coverage-review renders are kept in
the ignored, recoverable `.local-archive/book-expansion-review-2026-09-07/`.

## Repository boundary and rights

The release consists of the book branch and its ancestors, not every local Git
reference. Local tool checkpoint references contain presentation experiments
and machine-specific paths; they are not book commits and must not be pushed.
Use explicit branch/tag pushes, never `git push --mirror` or a blanket refspec.
The ignored `.local-archive/` remains local and recoverable.

The existing all-rights-reserved notice is now explicit in `LICENSE`. No new
reuse rights have been granted. The cover is labeled as a working draft, not a
finished first edition. Repository visibility must remain private for this
release.

`python src/audit_repository.py --history` checks reachable book-history text
and extracted PDF text for selected credential signatures and personal local
paths, without printing matched values. This is a bounded check, not a promise
that no secret can exist. Historical PNGs are book-preview artifacts and are
not OCR-scanned by the tool. Current figures are drawn by `src/build_book.py`;
the current build does not import third-party image files. Primary-source
citations identify the research being explained rather than granting rights
to reproduce those sources.

The book-history scan found no selected credential signatures. One historical
Makefile blob (`648f856e5728`) contains a developer-specific Python executable
path; the current Makefile uses `PYTHON ?= python3`. This known historical path
remains in private history; no force-push or history rewrite was performed.
The public university URL containing `/home/` was a false positive and the
scanner now distinguishes such URL paths from local filesystem paths.

## Historical release evidence: original September checkpoint

Checked September 7, 2026, for the `draft-2026-09-07` private review release.

| Check | Result and limit |
| --- | --- |
| CPU test suite | 33 passing tests: attention, top-K/statistics, RAG, resource arithmetic, runnable manuscript excerpt, status labels, audit regexes, and typesetter regressions. |
| PDF build and structure | 348 pages, 68 outline entries, 96 external link annotations; metadata and required-topic checks pass; no suspiciously empty pages flagged. |
| Link reachability | 87 unique manuscript URLs; no 404/410 responses; four publisher URLs returned 403 and remain unverified. |
| Visual review | All 348 pages inspected at contact-sheet scale; enlarged checks of chapter titles, code wrapping, RAG budgets/results, leadership cases, and the agent-policy diagram. Final short-list heading pagination was rerendered and checked. Not a word-by-word copy edit. |
| Numerical evidence | Shared model arithmetic is tested; selected paper claims are linked with explicit baseline/scope limits in `docs/claim-audit.md`. GPU experiments were not reproduced. |
| Repository boundary | Only the book branch and release tag are publication targets. Unrelated presentations and prior review renders remain in the ignored, recoverable local archive. No history rewrite or visibility change. |

The four access-blocked links are DOI `10.1080/00031305.1983.10483115`, DOI
`10.1145/2500128`, DOI `10.1145/3600006.3613165`, and the ScienceDirect PDF with
identifier `S002001900500298X`. A 403 does not show that the citation is broken,
but the automated check cannot validate its content.

## Historical checkpoint deliverables and rebuild

- Editable Markdown, figures/typesetter source, fixtures, tests, pinned Python
  requirements, and build instructions are versioned in the book repository.
- The reviewed PDF is attached to the private GitHub release
  `draft-2026-09-07`, rather than tracked as a generated binary in Git.
- `make verify` runs the CPU tests, rebuilds the PDF, and checks its structure.
  GitHub Actions runs the same workflow on `main` using Python 3.12.
- Current local detailed renders are in `output/previews/`; contact sheets are
  in `tmp/pdfs/release-review/`. Previous review renders were moved, not deleted,
  to `.local-archive/book-review-2026-09-07/`.

The release is suitable for private draft review, not a claim of a finished or
fully independently validated textbook. A public release still needs an
independent technical/copy edit, a final security/rights review, and any desired
reuse-license decision. Those limitations do not block this private checkpoint.
