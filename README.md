# Engineering Large Language Models

A working technical book by Yury Kirpichev covering model training, data, distillation, LLM inference, CUDA, distributed systems, ML coding, system design, and technical leadership.

The manuscript is under editorial review. Start with the [front matter](manuscript/00_front_matter.md) or the [draft review and priorities](docs/editorial-review.md). Publication and reuse terms have not yet been selected; the manuscript currently retains its existing copyright notice.

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
make verify
make check-links
make previews
```

The final PDF is written to:

`output/pdf/engineering-large-language-models.pdf`

Generated PDFs and previews are ignored by Git. Successful GitHub Actions runs attach the PDF as a downloadable artifact; the manuscript and builder are the versioned sources. Platform font availability can affect line breaks, so use the same environment when comparing page layouts.

## Source layout

- `manuscript/` - editable Markdown manuscript, ordered by filename.
- `src/build_book.py` - deterministic typesetting, diagrams, cover, table of contents, headers, and PDF outlines.
- `src/verify_pdf.py` - structural, navigation, pagination, and text-quality checks.
- `tests/` - tests for manuscript parsing and rendering markup.
- `docs/` - editorial review and priorities.
- `output/pdf/` - final deliverables.
- `output/previews/` - selected rendered pages used for visual review.
- `tmp/pdfs/` - temporary full-document renders.

`make verify` runs parser unit tests, rebuilds the book, checks every page for
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

The book is set in a 7 x 10 inch technical-book format with a navy/teal/coral system, embedded fonts, vector architecture diagrams, code panels, table styling, chapter openers, PDF outlines, and page headers/footers.
