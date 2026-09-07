# Engineering Large Language Models

A publisher-style technical handbook covering model training, data, distillation, LLM inference, CUDA, distributed systems, ML coding, system design, and technical leadership.

## Build

The build uses ReportLab and Poppler. Install the pinned Python packages with
`python3 -m pip install -r requirements.txt`; ensure `pdfinfo` from Poppler is
available on `PATH`.

```bash
make book
make verify
make check-links
make previews
```

The final PDF is written to:

`output/pdf/engineering-large-language-models.pdf`

## Source layout

- `manuscript/` - editable Markdown manuscript, ordered by filename.
- `src/build_book.py` - deterministic typesetting, diagrams, cover, table of contents, headers, and PDF outlines.
- `src/verify_pdf.py` - structural, navigation, pagination, and text-quality checks.
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
:::callout insight|Interview Insight
Compact content for a highlighted box.
:::

:::diagram speculative_decoding|Caption text.
```

## Editorial design

The book is set in a 7 x 10 inch technical-book format with a navy/teal/coral system, embedded fonts, vector architecture diagrams, code panels, table styling, chapter openers, PDF outlines, and page headers/footers.
