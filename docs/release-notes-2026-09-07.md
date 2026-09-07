# Engineering Large Language Models — September 2026 Working Draft

Private review release. Repository visibility and all-rights-reserved terms
are unchanged.

## What changed

- Reorganized the editable manuscript and kept unrelated slide experiments out
  of the book branch.
- Clarified phase-specific tensor parallelism versus prefill/decode context
  parallelism, with pinned vLLM implementation references.
- Split the long streaming section into four chapters and expanded RAG into
  a versioned documentation-assistant case with a runnable regression fixture.
- Added tested attention, top-K, and streaming-statistics CPU references.
- Added primary references, a bounded quantitative claim audit, shared resource
  calculations, worked answers, and fictional leadership decision cases.
- Fixed repeated titles, overflowing diagrams, truncated code lines, code
  contrast, and short exercise-set pagination.

## Validation and limits

The reviewed PDF has 348 pages, 68 outline entries, and 96 external link
annotations. All 33 CPU tests and structural PDF checks pass. All pages were
reviewed at contact-sheet scale, with enlarged checks of changed layouts.
Of 87 unique manuscript URLs, none returned 404/410; four publisher URLs block
automated access and remain unverified.

This is not a finished first edition. CUDA excerpts are illustrative, not GPU
benchmarks. The technical evidence audit is targeted, and the small RAG fixture
is a regression test, not a production quality or security benchmark. See
`docs/release-readiness.md`, `docs/claim-audit.md`, and `examples/README.md`.

The PDF is the reviewed reading copy. Markdown sources, tests, build tooling,
and pinned Python dependencies are included in the source snapshot at this tag.
