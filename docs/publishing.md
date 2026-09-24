# Publishing the public edition on GitHub

## Prepare assets

Use Python 3.12, install `requirements.txt` and Poppler, then run:

```sh
make release
```

The assets are written to `output/release/public-2026-09-23/`:

- `engineering-large-language-models.pdf`
- `engineering-large-language-models-source.zip`
- `SHA256SUMS`

The source archive includes the manuscript, original figure/build source,
examples, tests, licenses, contribution instructions, workflows, and review
evidence. Its internal `SOURCE-SHA256SUMS` records every included source file.
The outer checksums cover the source archive and PDF. The PDF is built and
structurally verified before packaging; inspect rendered pages before upload.

## Publish in the existing repository

The publication target is [ykirpichev/llm-book](https://github.com/ykirpichev/llm-book).
The author explicitly authorized making this existing repository public.
This supersedes the earlier fresh-repository proposal. Preserve existing
history, branches, and historical draft releases; do not force-push or rewrite
history as part of this release.

Before changing visibility, inspect all remote branches, tags, release assets,
and relevant repository discussions. The bounded scan and historical findings
are documented in [release readiness](release-readiness.md). Do not push local
tool-checkpoint references, ignored archives, or unrelated output experiments.
Use an explicit source commit and branch push, never `git push --mirror`.

The source archive is still a portable snapshot with no `.git` directory. It
can be built independently of the repository's development history.

## Publish the release

After the source is pushed, let the Build book workflow pass on `main`. Its
Linux font layout may differ from the locally reviewed PDF; publish the reviewed
PDF from the package unless you inspect the CI-built PDF too.

Preserve the original `public-2026-09` release. Create a new GitHub Release with tag `public-2026-09-23` pointing to the published
source commit. Use the prepared [release notes](release-notes-public-2026-09-23.md)
and attach the PDF, source archive, and `SHA256SUMS`. Mark it as the latest
release. The edition's AI review and hardware-validation limits belong in the
release notes, not behind an implication of independent human review.

Keep the direct PDF and source download links prominent at the top of
README.md. Check the repository and download while signed out, download the
assets, and verify their SHA-256 hashes. Record the public URLs and source
commit in release-readiness.md. Until this is done, local preparation must not
be described as a completed public launch.
