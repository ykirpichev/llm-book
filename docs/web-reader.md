# Online reading edition

The book is available at **https://ykirpichev.github.io/llm-book/** through GitHub
Pages. The README and repository website field point to this reader. The PDF
release remains available for offline reading.

## Content and behavior

`src/build_site.py` converts the same Markdown manuscript used by the PDF builder
into an introduction, nine part introductions, and seventy chapter pages. The
homepage lists every chapter. Previous/next navigation follows manuscript order;
each chapter has section anchors and an expandable local contents list.

The reader converts all book directives: callouts, equations, diagrams, the
contents marker, and print-only page breaks. Equations preserve the manuscript's
Unicode operators and explicit subscript/superscript syntax. Original ReportLab
diagrams are exported to SVG, with captions, alternative text, stable image
dimensions, and full-size links. Wide diagrams, tables, and code scroll within
the page on small screens.

Search runs in the browser against a generated local JSON index. Search queries
are not sent to an external search provider. Reading and ordinary links work
without JavaScript; search, the mobile contents toggle, and copy-code buttons
use JavaScript. No account, analytics integration, external font, or paid service
is required.

The reader tracks `main`; the September 2026 PDF and release source archive are
versioned snapshots. Publishing the website does not silently replace them.

## Build and test

Use Python 3.12 and install `requirements.txt`, then:

```sh
make test
make site
python -m http.server --directory output/site
```

Open the printed localhost URL. Generated site files are ignored by Git.
The stylesheet and browser script live in `src/web/` and are included in newly
built source archives. There is no separate web copy of the manuscript to edit.

Tests verify chapter coverage, search coverage, internal links and fragments,
figure references, exact preservation of fenced code, conversion of special
blocks, and fail-fast handling of unknown/unclosed directives. Inspect desktop
and phone layouts after visual changes. Test a search, a section anchor, chapter
navigation, and a wide diagram before deployment.

## Deployment

`.github/workflows/pages.yml` tests and builds the reader on pull requests.
On pushes to `main` and manual workflow runs, it also uploads `output/site` and
deploys it through the `github-pages` environment. Pages uses the **GitHub
Actions** publishing source, rather than exposing a source folder directly.
Deployment has `pages: write` and `id-token: write`; pull requests cannot deploy.

The workflow follows GitHub's [custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
Check its final deployment result and the public URL after changes. If the
workflow fails, inspect the failed job; the previous successful deployment
remains the reader until a new deployment succeeds.
