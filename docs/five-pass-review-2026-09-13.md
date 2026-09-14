# Five-pass book revision - September 13, 2026

Baseline: `937a2e1`, 69 chapters, 434-page PDF, 87 passing CPU tests.

This record separates five sequential review/revision cycles. A retained chapter
is not certified correct; an added citation is not a reproduced experiment.
The objective is a more useful and defensible engineering reference, not an
unverifiable claim to be the best or an exhaustive catalogue of current work.

## Iteration 1 - Coverage and implementation bridge

Reviewed the complete chapter/section inventory and prior review dispositions,
then closely read the serving lifecycle, production synthesis, and existing
engine references. The largest gap was implementation literacy: vLLM and SGLang
appeared as references without a guided explanation of their execution paths.
The correction is a dedicated engine/cache chapter, a source-to-concept
reading map, a realistic comparison experiment, and a layer diagram.

Source checks also found a current TensorRT LLM migration guide that removes
the TensorRT engine backend. The new chapter distinguishes that current
documentation from older build-engine tutorials. The inference conclusion now
distinguishes a numerical lower bound from an end-to-end guarantee.

Result: added the engine chapter and native layer diagram. All six new chapter
pages were rendered and inspected. The build had 440 pages and 88 passing tests.

## Iteration 2 - Mechanisms, arithmetic, and failure cases

Completed targeted mechanism and arithmetic review across foundations, training,
serving, and distributed recovery. Added zero-failure confidence limits and judge
calibration; corrected the temperature-scaling caveat, top-k distillation tail
semantics, percentile population interpretation, prefix-block alignment, and
weight-versus-KV traffic. Distinguished checkpoint staging from durable upload.
Added four numerical regression tests. Result: 443 pages, 92 passing tests;
representative changed pages 48, 104, and 170 visually inspected. Found an
orphaned example-status line on page 48, queued for the final layout pass.

## Iteration 3 - Primary-source currency and evidence boundaries

Checked current primary documentation for FSDP2, TorchTitan, Megatron-LM, verl,
llama.cpp, checkpointing, and artifact serialization. Added code-reading paths
and training/rollout numerical-mismatch checks; clarified per-parameter FSDP2
versus older flattened storage and selective recomputation. Added the local
inference use case and safe artifact promotion. Corrected an overly broad
management-plane outage rule that could conflict with immediate revocation.
Spot-checked existing 2026 model/reasoning/omni reports and A2A specification;
retained their bounded claims rather than adding unverified leaderboard claims.

Publication-link probe at this stage: 188 unique URLs, 184 HTTP success, no
404/410, four 403 responses (three DOI targets and one ScienceDirect paper).
403 means access unverified, not a confirmed missing paper. Later-added links
are included in the final check. Stable streaming derivations were reviewed as
mechanisms; no new algorithm was added solely because of publication date.
Validation: 444-page build, 92 tests; visually inspected pages 177, 188, 267,
302, and 306, including the checkpoint boundary carried over from iteration 2.

## Iteration 4 - Prose, chapter progression, and holistic story

Reviewed the opening and closing text of all 70 chapters, their section maps,
and the front-matter/capstone route. Replaced the unrelated opening code-assistant
scenario with the recurring documentation service and consistent latency targets.
Rewrote the training-to-serving transition, removed stale "final section" and
fixed-concurrency language, and connected the research snapshot to the engine
chapter. Added a concrete first-90-days decision case and expanded capstone
deliverables to cover engine choice, cache ownership, hardware portability, and
the distinction between measured and unmeasured results. Corrected the tiny
fixture's "proves" language and the streaming notation's chapter scope.
Validation: 446-page build, 92 tests; inspected pages 12, 122, 418, 432, and
433. The capstone expansion stranded its last paragraph; iteration 5 trims
the repetition and checks the repaginated result.

## Iteration 5 - Adversarial reread and final production checks

Completed a fresh read of the revised claims, numerical examples, tests, and
evidence boundaries. Rechecked all 44 distinct diagram proofs (45 placements)
with their captions. Clarified that the new engine figure shows forward dispatch
and omits the output-return path; it is not a complete process topology.
Added typesetter regressions that keep code-status labels with code and short
table introductions with tables. Trimmed the capstone's repeated instructions
so its closing paragraph no longer occupies a separate page. Added an automated
check that this ledger covers each current chapter exactly once.

Final validation:

- 95 passing CPU tests, including runnable manuscript blocks, mechanism arithmetic,
  figure geometry, typesetter grouping, and chapter coverage.
- 445 pages, 80 outline entries, 245 external link annotations; PDF metadata,
  geometry, navigation, required text, and sparse-page checks pass.
- All 445 pages inspected at contact-sheet scale; final enlarged checks on pages
  13, 48–49, 185, 432, and 445, in addition to the earlier changed-page checks.
  This verifies layout at the stated scales, not a human word-by-word proofread.
- 196 unique manuscript URLs checked: 192 HTTP successes, zero 404/410 responses,
  four publisher 403 responses. HTTP success is reachability, not claim validation.
  The blocked URLs are DOI `10.1080/00031305.1983.10483115`, DOI `10.1145/2500128`,
  DOI `10.1145/3600006.3613165`, and the ScienceDirect PDF `S002001900500298X`.
- Git diff whitespace checks pass. The bounded history scan reports only the
  previously documented personal path in historical Makefile blob `648f856e5728`;
  historical images are not OCR-scanned. No history rewrite was performed.

## Holistic verdict and remaining evidence

The story is now consistent: define useful behavior; construct and evaluate the
training signal; budget request phases and state; inspect an actual engine;
map execution to devices and ranks; enforce application authority; and operate
the release through explicit owners. The capstone requires the same chain of
evidence. Streaming chapters are a supporting telemetry/state track, not an
unexplained detour or a prerequisite to reading RAG.

Suitable for broader technical-reviewer circulation as a labeled working draft.
Not yet a certified production manual or finished public edition. The next
high-value evidence is independent human technical/copy review and execution of
the GPU, distributed-cache, engine-comparison, and accelerator-port experiments
on target hardware. This pass did not reproduce frontier training, compile CUDA
examples, benchmark any serving engine, or test real-model agent robustness.
Rights, privacy, and repository-history review remain gates before publication.
The new figure is original vector artwork; no external diagram was reproduced.
No remote publication, release replacement, or repository visibility change was
performed by this revision.

## Chapter-by-chapter disposition

All 70 chapters received a structured coverage/progression review. Close reading,
source checks, and rederivations were concentrated on consequential claims and
changed regions. This is not a claim that all 125,000-plus words were line-edited
five times, every citation's full text was reread, or every equation was proved.
"Retain" records a considered decision not to pad a chapter with unrelated news.

| # | Chapter | Disposition and review focus |
| --- | --- | --- |
| 1 | The Model, Workload, and System Contract | Revise: use the recurring assistant and its actual hypothetical SLOs; preserve outcome-cost denominator caveats. |
| 2 | From Text to Tokens, Targets, and Loss | Retain: token identity, packing masks, byte-normalized comparison, and runnable target tests form a coherent prerequisite. |
| 3 | Optimization as a Coupled Dynamical System | Retain: optimizer state, data order, and skipped-step recovery connect algorithms to distributed execution. |
| 4 | Transformer Architecture as Resource Allocation | Retain: dense attention, GQA, MLP, and resource derivations precede architectural alternatives. |
| 5 | Compressed, Sparse, and Recurrent Model State | Retain: modern attention/state families already distinguish selection, compression, recurrence, and rollback. |
| 6 | Scale, Memory, and Performance Models | Retain: units, lower bounds, memory, and economics; correct misleading downstream uses in inference. |
| 7 | Measurement and Experimental Judgment | Revise: zero-observed-failure confidence, evaluator calibration, and tests-versus-proof boundary. |
| 8 | Designing a Training Recipe | Retain: staged recipe, pilot evidence, stopping, and cross-references avoid prescribing an algorithm by fashion. |
| 9 | Data Contracts, Provenance, and Normalization | Retain: source/derived identity, deletion lineage, normalization, and security limits remain necessary. |
| 10 | Deduplication, Quality, and Contamination | Retain: candidate versus verified similarity, split-family policy, and contamination uncertainty. |
| 11 | Synthetic Data and Mixture Design | Retain: verifier yield and independent generalization, not generation volume, define value. |
| 12 | Training Data Platform and Release Engineering | Retain: immutable manifests and retry/release identity connect data to checkpoints. |
| 13 | Applied Data-System Casework | Retain: cumulative cases apply the preceding four data chapters; no redundant framework catalogue. |
| 14 | Distillation, KL Divergence, and Model Sizing | Revise: temperature scaling is approximate; tail mass does not preserve full KL; add numeric counterexample. |
| 15 | Supervised Adaptation and Low-Rank Training | Retain: gradient path through frozen weights, supervised-token normalization, and retrieval-versus-adaptation decision. |
| 16 | Post-Training and Alignment | Retain: proxy reward, judge bias, behavioral specification, and rollback bundle are explicit. |
| 17 | Reinforcement Learning for Reasoning and Tool Use | Revise conclusion: connect the evaluated checkpoint to serving and the assistant; retain objective/behavior-policy distinctions. |
| 18 | Request Lifecycle, Metrics, and Workload Models | Revise: distinguish per-request acceptance, population percentiles, token-weighted gaps, and session experience. |
| 19 | Prefill, Decode, and Performance Modeling | Retain: phase-specific shapes and illustrative byte ledger; quantify weight/KV distinction later. |
| 20 | KV Cache, Paging, and Prefix Reuse | Revise: saved prefill work depends on full-block and logit requirements; new engine chapter supplies implementation path. |
| 21 | Scheduling, Batching, and Admission Control | Retain: token/KV/slack constraints, starvation, overload, and headroom; engine chapter adds execution context. |
| 22 | Sampling, Structured Output, and Speculative Decoding | Retain: exact proposal/verification semantics, draft overhead, and modern draft examples are already separated. |
| 23 | Parallel, Replicated, and Disaggregated Serving | Retain: phase separation, destination reservation, transfer, and failure reserve; detailed sharding remains in Part V. |
| 24 | Quantization, Compression, and Adapter Serving | Revise: weight-only savings leave KV traffic; replace fixed-concurrency rhetoric with a conditional estimate. |
| 25 | Production Architecture, Capacity, and Reliability | Revise: artifact deserialization/promotion, authorization during management outage, and non-final transition. |
| 26 | Serving Engines and Cache Backends in Practice | Add: vLLM, SGLang, current TensorRT LLM, FlashInfer, cache/fleet responsibilities, llama.cpp, comparison method, and original diagram. |
| 27 | GPU Execution, Memory, and Resource Accounting | Retain: execution/memory/occupancy prerequisites; no portable claim based on one device's limits. |
| 28 | Host-Device Orchestration, Streams, and CUDA Graphs | Retain: async lifetime, warm-up, graph shapes, and end-to-end timing support the overlap discussion. |
| 29 | Hierarchical Matrix Multiplication | Retain: scalar-to-tiled progression, arithmetic intensity, epilogue, and production-library boundary. |
| 30 | Reductions, Prefix Scans, and Histograms | Retain: participation, tree structure, contention, and deterministic-order caveats. |
| 31 | Softmax, Normalization, Top-K, and Sampling | Retain: numerical edge cases and sampling semantics; existing references/tests supply the CPU oracle. |
| 32 | FlashAttention and IO-Aware Exact Attention | Retain: online merge recurrence and exact-operation versus floating-point-result distinction. |
| 33 | Blackwell Pipelines and Modern Attention Kernels | Retain: existing SM100/FlashAttention-4 discussion states architecture scope and provisional low-bit evidence. |
| 34 | CUDA Kernels for LLM Inference | Retain: paged layout, GQA reuse, split-K merge, quantization, and routing contracts. |
| 35 | Kernel Engineering, Profiling, and Correctness | Retain: warm/cold boundaries, measured bottlenecks, tolerance, sanitizer limits, and deployment replay. |
| 36 | Accelerator Ecosystems Beyond CUDA and NVIDIA | Retain new prior-pass chapter: Triton/ROCm/TPU/Neuron/SYCL layers and port acceptance; connect it to capstone. |
| 37 | Communication Models, Collectives, and Topology | Retain: byte accounting and matched collective/application boundaries precede sharding. |
| 38 | Data Parallelism, ZeRO, and Fully Sharded Training | Revise: current FSDP2/DTensor, TorchTitan/Megatron reading path, selective recomputation, and mutable-state caveat. |
| 39 | Tensor, Sequence, and Context Parallelism | Retain: layouts as types and TP/CP tradeoffs; no duplicate serving-only tutorial here. |
| 40 | Pipeline Parallelism and Hybrid Plans | Retain: bubble assumptions, hybrid group geometry, and recovery-adjusted comparison. |
| 41 | Mixture-of-Experts and Sparse Communication | Retain: expert capacity, routed bytes, hot-rank tails, and decode-specific small groups. |
| 42 | Distributed Inference and Stateful Placement | Retain: separate PCP/DCP ownership, partial-attention merge, heterogeneous handoff, and stream failure semantics. |
| 43 | Distributed Reinforcement Learning and Policy Freshness | Revise: verl roles and training/rollout numerical mismatch; preserve queue-selection and policy-age limits. |
| 44 | Distributed Checkpoints, Recovery, and Elasticity | Revise: distinguish staging, durable upload, publication, and mutation fences; require failure tests at both boundaries. |
| 45 | Cluster Scheduling, Observability, and Distributed Diagnosis | Retain: first divergence, topology, admission, and coordinated recovery conclude the distributed sequence. |
| 46 | Exact Streaming Queries and Time Windows | Revise notation scope; retain immutable-record versus mutable-frequency distinction and event-time contract. |
| 47 | Online Statistics and Sampling | Retain: count-weighted moment merge and population-correct sampling, with CPU fixtures. |
| 48 | Heavy Hitters and Probabilistic Sketches | Retain: pointwise versus simultaneous bounds, deletion/merge limits, KLL rank error, and sizing examples. |
| 49 | Building a Streaming Telemetry Service | Retain: query-to-summary selection, event identity, windows, and recovery apply the algorithm sequence. |
| 50 | Compact Review of ML Algorithms in Production | Retain: intentionally compact implementation review, not a substitute for a full ML textbook. |
| 51 | An Engineering System Design and Review Method | Retain: already introduces the assistant's corpus, traffic, budgets, and decision process. |
| 52 | RAG, Vector Search, and Evaluation Pipelines | Retain: RRF/late interaction/hierarchical retrieval plus authoritative revocation; resolve conflict in production chapter. |
| 53 | Building and Evaluating a Bounded Agent Loop | Retain: capabilities, unknown write outcome, memory provenance, protocols, and deterministic-fixture limits. |
| 54 | Efficient Frontier Models and Reasoning Training | Retain: bounded 2026 architecture and inference-budget examples; spot-check primary report identity. |
| 55 | Serving, Attention, and Memory Hierarchies | Revise: link research mechanisms back to the practical engine chapter; retain scoped historical performance claims. |
| 56 | Multimodal Representations: Images, Video, and Speech | Retain: patch/frame/codec arithmetic, coordinate inversion, evidence loss, and audible-output boundary. |
| 57 | Diffusion and Block-Parallel Language Generation | Retain: masked objective, commitment example, mutable-block cache validity, and draft-versus-target distinction. |
| 58 | Multimodal and Tool-Using Systems | Retain: geometric and authority boundaries extend the earlier mechanism chapters rather than replace them. |
| 59 | From Research Result to Production Decision | Retain: result card, falsifiable adoption test, and ownership handoff bridge research and leadership. |
| 60 | Executive Technical Communication | Retain: decision/evidence/risk/request and explicit uncertainty; cumulative answer criteria appear later in the part. |
| 61 | Operating Mechanisms for ML Systems | Retain: owner/input/trigger/action makes governance operational rather than a meeting list. |
| 62 | Incident Leadership and High-Risk Change | Retain: mitigation and diagnosis, measurable incident actions, and migration completion tests. |
| 63 | Strategy, Vision, and the First 90 Days | Revise: concrete fictional release-pipeline decision with day-60/day-90 evidence and reversal condition. |
| 64 | Leadership Evidence and Reflective Practice | Retain: clearly fictional migration/prioritization/incident cases; do not invent author experience. |
| 65 | Cross-Layer Design Prompt Bank | Retain: prompts integrate rather than duplicate full chapters; closing question hands off to capstone. |
| 66 | End-to-End Learning Lab and Capstone | Revise: tests-versus-proof wording, engine/port experiments, evidence record, ownership trace; trim repetition after rendering. |
| 67 | Formula and Capacity Sheet | Retain: formula boundaries, units, ownership, and schedule assumptions remain explicit. |
| 68 | CUDA Engineering Checklist | Retain: semantic oracle, resource ledger, diagnostics, and acceptance rather than untested performance promises. |
| 69 | Diagnostic Question Bank | Retain: compact retrieval cues point back to developed mechanisms. |
| 70 | Glossary and Decision Index | Retain: definitions, navigation, and final assistant-centered principle close the story. |
