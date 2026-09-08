# End-to-end learning expansion — September 7, 2026

Private working draft of *Engineering Large Language Models* by Yury Kirpichev.
The earlier `draft-2026-09-07` release is preserved. No repository visibility or
license change is included.

## What changed

- Reviewed the chapter coverage and expanded all nine parts. The book now has
  68 chapters and 402 pages, up from 58 chapters and 348 pages.
- Added dedicated chapters on tokenization and loss; compressed, sparse, and
  recurrent model state; SFT and parameter-efficient adaptation; reasoning and
  tool post-training; Blackwell pipelines; distributed RL; bounded agents;
  multimodal representations; diffusion language generation; and an end-to-end
  learning lab.
- Connected recent methods to their mechanisms and failure boundaries:
  MLA/Gated DeltaNet/hybrids, DPO/GRPO/DAPO/GSPO/self-distillation, EAGLE-3/DFlash,
  low-bit scaling, FlashAttention-4, DeepEP, asynchronous rollout systems,
  hybrid retrieval, and modern multimodal architectures. Research cutoff:
  September 7, 2026. Preprints are not treated as universal production results.
- Added standard-library teaching references for sequence mechanisms,
  post-training objectives, exact speculative sampling, quantization, retrieval,
  a capability-bounded agent loop, and a train/save/load/generate bigram model.
- Added worked calculations, exercises with answers, an ordered capstone,
  updated formula/glossary references, and a 68-row chapter coverage ledger.
- Fixed heading/table pagination and added typesetter regression tests.

## Validation

- `make verify`: 66 CPU tests pass; PDF structure and navigation pass.
- PDF: 402 pages; 78 outline entries; 159 external link annotations.
- 145 distinct manuscript URLs checked: no missing-page responses; five
  automated checks remain unverified (four publisher access blocks, one DNS
  failure). Details are in `docs/release-readiness.md`.
- Full-document contact-sheet review plus enlarged checks of changed equations,
  examples, tables, and chapter openers. This is not an independent copy edit.

## Scope and limits

This is a substantially expanded, self-contained learning draft, not a claim
to cover every paper or certify every implementation. GPU kernel performance,
frontier model convergence, distributed training, and real-model agent behavior
were not reproduced. Runnable fixtures are explicitly educational; the tiny
language model is a bigram, not a transformer. An independent technical and copy
review remains appropriate before a finished public edition.

The repository contains editable Markdown, build code, examples, tests, and
review records. The attached PDF is the reviewed local build; generated binaries
are not committed into the source tree. Use `make verify` to rebuild and test.

Reviewed PDF size: 1,245,589 bytes. SHA-256:
`4c386695d4f0712bc4b6e41b278a6bc05ddd4e2c6c97240115d662e07368e384`.
