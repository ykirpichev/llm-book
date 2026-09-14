# Visual teaching pass - September 13, 2026

## Result

Added 24 original vector diagrams across Parts I-VIII. The book now contains
42 diagram placements using 41 distinct designs. The regenerated PDF has 427
pages. The existing forest, sage, warm ivory, brass, and terracotta palette is
preserved. No external paper artwork or raster screenshots were imported.

This is a targeted teaching pass: figures explain dependencies, tensor axes,
state ownership, timelines, or component relationships in dense sections.
Tables, worked derivations, code, and reference chapters retain their existing
roles. The diagrams illustrate the mechanisms and assumptions in the adjacent
text; timing illustrations are not measured benchmarks.

## New figures in the reviewed build

| Part | Figure | Page | Main relationship |
| --- | --- | --- | --- |
| I | Next-token supervision | 17 | Input positions and shifted prediction targets |
| I | Optimization loop | 20 | Forward, gradients, persistent optimizer state, and updated weights |
| I | Decoder block | 29 | Two pre-norm sublayers and their residual bypasses |
| I | Attention history | 35 | Dense, local, sparse reads versus compression and recurrence |
| II | Deduplication groups | 65 | Pair verification and non-transitive connected components |
| II | Synthetic-data verification | 77 | Candidate generation, rejection, and accepted mixtures |
| II | Data release architecture | 86 | Immutable payloads, lineage, policy, and release manifests |
| II | LoRA | 107 | Frozen projection and trainable low-rank branch |
| II | Policy learning | 116 | Behavior samples, reward, advantages, reference, and learner |
| III | Shared KV pages | 138 | Shared immutable prefix and independent writable tails |
| III | KV handoff | 161 | Reservation, transfer, validation, and execution ownership |
| III | Quantized execution | 168 | Packed values, metadata, activations, and compatible kernels |
| IV | Double buffering | 224 | Transfer completion, buffer reuse, and consumer timing |
| V | ZeRO ownership | 250 | Which state partitions are retained at each stage |
| V | MoE dispatch | 267 | Selected experts, return traffic, and weighted combination |
| V | Prefill/decode context parallelism | 276 | Query-row sharding versus historical KV-token sharding |
| V | Checkpoint publication | 289 | Durable shards, validated manifest, and atomic pointer |
| VI | Event time | 311 | Out-of-order arrivals, event-time windows, and watermarks |
| VI | Count-Min Sketch | 323 | Row hashes, queried counters, and the minimum estimate |
| VI | Streaming telemetry | 332 | Window state, per-key merge, corrections, and recovery offsets |
| VI | RAG | 351 | Offline index construction and online evidence retrieval |
| VII | Multimodal inputs | 374 | Encoders, projection, positions, and language-model context |
| VII | Block diffusion | 379 | Fixed causal prefix and mutable block cache dependencies |
| VIII | Migration gates | 398 | Increasing commitment, evidence, and retirement criteria |

Page numbers identify this build and may change with later manuscript edits.

## Existing diagrams improved

- Memory hierarchy: replaced overlapping nested outlines with separate levels.
- Parallelism map: named each partition axis and removed ambiguous arrows.
- Leadership loop: corrected the sequence and rendered its central label on two lines.
- Continuous batching: showed immediate slot refill and moved the timeline label clear of blocks.
- GEMM tiling: used a multiplication sign between operands instead of implying A flows into B.

## Checks and corrections

The new figure library measures label widths and checks canvas containment,
label overlaps, node overlaps, and horizontal/vertical connectors crossing text.
An unknown diagram name now fails the build instead of producing a placeholder.
Figures remain with their captions and immediate section headings. Displayed
equations also remain with their explanations after pagination changes.

All new diagrams were inspected as rendered figures and in their manuscript
context. Every final book page was included in a contact-sheet layout review;
changed or suspect details were inspected at higher resolution. Corrections
included a crossing in the KV map, an ambiguous policy-data path, a watermark
arrow crossing its label, a lineage arrow crossing text, and separated headings.
No clipping, overlapping labels, or detached new captions remained in the
reviewed output.

Validation: 81 passing CPU tests; 24/24 new figure titles and complete captions
found together on their intended PDF pages; PDF verification passed with 78
outline entries, 199 external links, and no suspiciously sparse pages.

The reproducible proof command is `python3 src/render_figure_proofs.py`; it writes
temporary QA material under `tmp/pdfs/visual-pass/`. The source figures live in
`src/book_figures.py` and are rendered as vector geometry with embedded book fonts.
