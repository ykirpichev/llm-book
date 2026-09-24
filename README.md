# Engineering Large Language Models

## [Read the book online — no download needed](https://ykirpichev.github.io/llm-book/)

**[Download PDF](https://github.com/ykirpichev/llm-book/releases/download/public-2026-09-24-part7/engineering-large-language-models.pdf)** · **[Download source](https://github.com/ykirpichev/llm-book/releases/download/public-2026-09-24-part7/engineering-large-language-models-source.zip)** · **[Release notes](https://github.com/ykirpichev/llm-book/releases/tag/public-2026-09-24-part7)**

A free technical book by **Yury Kirpichev** about building and operating LLM systems: foundations, training and distillation, inference, CUDA and other accelerators, distributed systems, production design, and technical leadership.

**Public Edition - September 2026.** Seventy chapters across nine parts, with worked examples, original diagrams, runnable CPU references, and design exercises. Part VI revised through **September 24, 2026**, and Part VII with worked additions on **September 24, 2026** (frontier snapshot: September 23); other parts retain their September 13 research cutoff unless explicitly dated. [What changed in Part VI](docs/part6-update-2026-09-24.md). [Part VII worked additions](docs/part7-followup-2026-09-24.md). [Earlier Part VII revision](docs/part7-update-2026-09-23.md).

## Read the book

The [online reader](https://ykirpichev.github.io/llm-book/) includes chapter navigation, full-book search, equations, and the original diagrams. It works on desktop and mobile.

Start with the [introduction and learning path](https://ykirpichev.github.io/llm-book/introduction.html), or choose a part below. Read Parts I-III in order for the model-to-service path. Parts IV-V cover accelerators and clusters; Part VI develops retrieval, agents, and production design. The [capstone and field reference](https://ykirpichev.github.io/llm-book/part-09.html) connect the examples into a complete learning path.

The **[public edition release](https://github.com/ykirpichev/llm-book/releases/tag/public-2026-09-24-part7)** includes the PDF, editable source, and [SHA-256 checksums](https://github.com/ykirpichev/llm-book/releases/download/public-2026-09-24-part7/SHA256SUMS). No account or payment is required to read or download the book. Build it locally with `make book`, or create all release assets with `make release`. See the [release evidence](docs/release-readiness.md) and [publication instructions](docs/publishing.md). Earlier draft releases are historical checkpoints with their original notices.

| Part | Read online |
| --- | --- |
| I | [Foundations](https://ykirpichev.github.io/llm-book/part-01.html) |
| II | [Training systems, data, and adaptation](https://ykirpichev.github.io/llm-book/part-02.html) |
| III | [Inference systems](https://ykirpichev.github.io/llm-book/part-03.html) |
| IV | [CUDA and accelerator programming](https://ykirpichev.github.io/llm-book/part-04.html) |
| V | [Distributed ML systems](https://ykirpichev.github.io/llm-book/part-05.html) |
| VI | [Coding and system design](https://ykirpichev.github.io/llm-book/part-06.html) |
| VII | [Recent state of the art](https://ykirpichev.github.io/llm-book/part-07.html) |
| VIII | [Technical leadership](https://ykirpichev.github.io/llm-book/part-08.html) |
| IX | [Field reference](https://ykirpichev.github.io/llm-book/part-09.html) |

## Reuse and contributions

The book text and original figures are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original code examples and build tools use the [MIT License](LICENSE-CODE). Both permit commercial reuse under their respective terms. See [LICENSE](LICENSE) for the scope, including embedded code and third-party material.

Corrections, clearer explanations, and reproducible examples are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for reporting errors and proposing changes.

## How this edition was reviewed

The book was developed with AI-assisted writing, technical review, and editing, including Astra, Sol, and Terra passes. It has **not** received independent human technical review or professional copy editing. That is a disclosed limitation, not a prerequisite for this free public edition.

The CPU references and document checks are executable. Accelerator kernels, real serving engines, distributed execution, and performance results have not been validated on hardware in this project. Code blocks distinguish runnable references, illustrative excerpts, and pseudocode. Published performance results are attributed to their sources and bounded by those sources' workloads.

Historical review reports in `docs/` record earlier checkpoints. Their draft-only release restrictions and test/page counts describe those checkpoints; [current release readiness](docs/release-readiness.md) governs this edition.

## Build

The build uses ReportLab and Poppler. Install the pinned Python packages in the
virtual environment below; ensure `pdfinfo` from Poppler is available on `PATH`.

Use Python 3.12 (the CI reference environment); the system Python on macOS may be too old. For an isolated local environment:

```bash
python3.12 -m venv .venv
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
make site
```

The website is written to `output/site/`. Preview it with `python -m http.server --directory output/site`, then open the printed localhost URL. GitHub Actions automatically deploys it from `main`; see [reader maintenance](docs/web-reader.md). The website follows the current manuscript; the versioned PDF remains a fixed release snapshot.

The final PDF is written to:

`output/pdf/engineering-large-language-models.pdf`

Generated PDFs and previews are ignored by Git. Release packages contain the PDF and a source archive; successful GitHub Actions runs also attach their build as a temporary downloadable artifact. The manuscript and builder are the versioned sources. Platform font availability can affect line breaks, so use the same environment when comparing page layouts.

## Verified teaching examples

See [examples/README.md](examples/README.md) for runnable CPU references covering a complete tiny-model lifecycle, tokenization/loss, recurrent state, post-training, speculative sampling, quantization, attention, streaming, retrieval, and a bounded agent loop. Each has explicit assumptions and tests. Manuscript snippets distinguish runnable code from illustrative excerpts and pseudocode. CUDA excerpts have not been compiled or benchmarked in this release.

For an accelerator evaluation, start with the [optional target examples](examples/accelerators/README.md) and the [baseline/candidate acceptance record](docs/accelerator-acceptance-template.md). The latter is an unfilled reporting template, not a recorded benchmark.

The [claim audit](docs/claim-audit.md) records selected source/version checks and the running-model arithmetic. It is not a line-by-line technical certification.

## Source layout

- `manuscript/` - editable Markdown manuscript, ordered by filename.
- `src/build_book.py` - typesetting, diagrams, cover, table of contents, headers, and PDF outlines.
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
