# Contributing

Thank you for helping improve Engineering Large Language Models.

## Report an error

Open an issue in the public GitHub repository with the edition, part and chapter
name, and page number if using the PDF. Quote the relevant sentence or identify
the example. Explain the correction and include a primary source or a minimal
reproduction where possible. Page numbers can differ across build environments,
so a chapter title is especially useful.

For code bugs, include the command, Python/runtime version, expected result,
and actual result. For accelerator results, include hardware, software versions,
shapes, precision, and the complete measurement procedure. Do not include
credentials, personal data, or confidential workloads.

## Propose a change

Keep pull requests focused on one correction or improvement. Edit the Markdown
in `manuscript/`, the examples in `examples/`, or the builder in `src/` rather
than editing generated PDFs. Preserve the distinction between runnable code,
illustrative excerpts, and pseudocode. Cite primary sources for technical claims
and date claims that depend on a moving software implementation.

Use Python 3.12 and the pinned dependencies. Run `make verify` for manuscript,
example, or builder changes. Inspect rendered pages after layout changes;
passing text checks alone does not prove readable layout. Documentation-only
changes need a link and wording check. Include the relevant validation in your PR.

Disclose substantial AI assistance in your contribution and verify its claims.
Do not claim hardware tests that you did not run. Add tests when they protect
nontrivial behavior or fix a reproduced error, not merely to mirror prose.

Contribute only material you have the right to share. Contributions use the
existing licenses: CC BY 4.0 for prose and original figures, MIT for original
code and build configuration. See LICENSE for details. Third-party material
must retain its required notices and compatible terms.
