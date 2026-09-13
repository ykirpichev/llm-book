# Engineering Large Language Models

A working technical book by Yury Kirpichev covering model training, data, distillation, LLM inference, CUDA, distributed systems, ML coding, system design, and technical leadership.

This is a **working draft**, not a finished first edition. Start with the [front matter](manuscript/00_front_matter.md), [current release checks](docs/release-readiness.md), or [September 13 multi-model review](docs/multi-model-review-2026-09-13.md). The existing all-rights-reserved terms are explicit in [LICENSE](LICENSE); no open-source or open-content license is granted at this stage.

The expanded versioned PDF is attached to the [September 2026 end-to-end learning release](https://github.com/ykirpichev/llm-book/releases/tag/draft-2026-09-07-expanded). The [earlier checkpoint](https://github.com/ykirpichev/llm-book/releases/tag/draft-2026-09-07) is preserved. The repository and both releases remain private; access requires repository permission.

The [chapter coverage audit](docs/coverage-audit-2026-09.md) records the expanded pass, with a September 7, 2026 research cutoff. Core mechanisms are explained in the book with examples and failure boundaries; references provide supporting evidence. The book does not claim to catalogue every paper or reproduce frontier training runs.

The [September 8 Part I review](docs/part1-review-2026-09-08.md) records an earlier independent pass. The [September 13 multi-model review](docs/multi-model-review-2026-09-13.md) records the Astra technical, Sol prose, and Terra pedagogy reviews, final Astra gate, 77 passing CPU tests, and rebuilt 410-page PDF. Rebuild from the current sources for these revisions; the September 7 release PDFs remain unchanged historical checkpoints.

## Read the draft

| Part | Manuscript |
| --- | --- |
| I | [Foundations](manuscript/01_foundations.md) |
| II | [Training systems, data, and adaptation](manuscript/02_training.md) |
| III | [Inference systems](manuscript/03_inference.md) |
| IV | [CUDA and accelerator programming](manuscript/04_cuda.md) |
| V | [Distributed ML systems](manuscript/05_distributed.md) |
| VI | [Coding and system design](manuscript/06_coding_and_design.md) |
| VII | [Recent state of the art](manuscript/07_recent_state_of_art.md) |
| VIII | [Technical leadership](manuscript/08_technical_leadership.md) |
| IX | [Field reference](manuscript/09_appendices.md) |

## Build

The build uses ReportLab and Poppler. Install the pinned Python packages with
`python3 -m pip install -r requirements.txt`; ensure `pdfinfo` from Poppler is
available on `PATH`.

Python 3.12 is the CI reference environment. For an isolated local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install Poppler with `brew install poppler` on macOS or `sudo apt-get install poppler-utils` on Ubuntu.

```bash
make book
make test
make verify
make check-links
make previews
```

The final PDF is written to:

`output/pdf/engineering-large-language-models.pdf`

Generated PDFs and previews are ignored by Git. A GitHub Release preserves the reviewed PDF; successful GitHub Actions runs also attach their build as a temporary downloadable artifact. The manuscript and builder are the versioned sources. Platform font availability can affect line breaks, so use the same environment when comparing page layouts.

## Verified teaching examples

See [examples/README.md](examples/README.md) for runnable CPU references covering a complete tiny-model lifecycle, tokenization/loss, recurrent state, post-training, speculative sampling, quantization, attention, streaming, retrieval, and a bounded agent loop. Each has explicit assumptions and tests. Manuscript snippets distinguish runnable code from illustrative excerpts and pseudocode. CUDA excerpts have not been compiled or benchmarked in this release.

The [claim audit](docs/claim-audit.md) records selected source/version checks and the running-model arithmetic. It is not a line-by-line technical certification.

## Source layout

- `manuscript/` - editable Markdown manuscript, ordered by filename.
- `src/build_book.py` - deterministic typesetting, diagrams, cover, table of contents, headers, and PDF outlines.
- `src/verify_pdf.py` - structural, navigation, pagination, and text-quality checks.
- `examples/` - runnable, dependency-light teaching references and evaluation fixtures.
- `tests/` - algorithm, arithmetic, manuscript, audit-tool, and builder checks.
- `docs/` - editorial review, evidence audit, and release records.
- `output/pdf/` - final deliverables.
- `output/previews/` - selected rendered pages used for visual review.
- `tmp/pdfs/` - temporary full-document renders.

`make verify` runs the complete CPU test suite, rebuilds the book, checks every page for
geometry and suspicious emptiness, validates navigation and required topics,
checks publication metadata and front matter, and scans extracted text for
placeholder or stale editorial language.

## Supported manuscript syntax

The builder supports headings, paragraphs, lists, block quotes, fenced code, tables, page breaks, vector diagrams, callouts, and a generated table of contents. Examples:

```text
:::callout insight|Engineering Insight
Compact content for a highlighted box.
:::

:::diagram speculative_decoding|Caption text.
```

## Editorial design

The book is set in a 7 x 10 inch technical-book format with a forest-green, sage, ivory, brass, and terracotta palette, embedded fonts, original vector teaching diagrams, code panels, table styling, chapter openers, PDF outlines, and page headers/footers. Diagram labels are measured during the build and each figure stays with its caption. The new figure library is in `src/book_figures.py`; its diagrams explain tensor layouts, state ownership, timing, and component relationships.
