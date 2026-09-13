# End-to-end teaching and currency audit

Started September 7, 2026, from private checkpoint `85a4882`. This pass is not
the earlier layout/release review. Its question is whether a reader can explain
and apply the central mechanisms without needing another tutorial.

## Chapter acceptance standard

For each technical chapter: define prerequisites and notation; explain the
mechanism rather than only its name; work a small example; distinguish exact
semantics from approximations; give a failure case and a decision boundary;
provide exercises with answer criteria; cite primary evidence for modern
variants. CPU tests validate selected calculations, not GPU performance.

The research cutoff is September 7, 2026. Recent preprints are labeled as such;
reported maxima and adoption claims are not treated as universal conclusions.
Links are evidence and further reading, not substitutes for the explanation.

## Sequential work queue

| Part | Gaps to resolve | Status |
| --- | --- | --- |
| I: Foundations | Tokenization/targets; explicit transformer operations; MLA, sparse attention, recurrent/hybrid state; Muon; compute allocation and evaluation | Expanded; CPU references tested |
| II: Training | SFT/LoRA/QLoRA implementation; DPO, PPO/GRPO; newer policy-gradient variants; rollout/verifier pipeline; precision and data-mixture experiments | Expanded; arithmetic tested |
| III: Inference | Modern speculative draft families; exactness; low-bit format arithmetic and algorithms; cache/state implications of hybrids | Expanded; output-distribution and quantization tests |
| IV: CUDA | Blackwell execution model, tensor memory and asynchronous ownership; FA4; numerical and kernel decision cases | Expanded; GPU experiments explicitly not reproduced |
| V: Distributed | Modern MoE communication/overlap; parallelism and state ownership in asynchronous RL; failure and memory accounting | Expanded; source-backed mechanisms and worked ledgers |
| VI: Applications | Retrieval alternatives and reranking; a complete bounded agent loop with tools, memory, evaluation, and failure recovery | Expanded; deterministic fixtures tested |
| VII: Current research | Dated mechanism map; self-contained multimodal/audio/video and diffusion explanations | Expanded through September 7 cutoff; preprint limits stated |
| VIII: Leadership | Connect new technical choices to experiment gates, review evidence, and operational ownership | Release decision case and ownership table added |
| IX: Reference | Glossary, formulas, learning path, and chapter navigation | Expanded; complete tiny-model lab and capstone added |

Do not mark the whole book complete because its PDF builds. Final evidence must
include chapter coverage, tests, citation checks, visual QA, and disclosed
limitations. Keep the previous private release intact while this pass evolves.

## Chapter-by-chapter coverage disposition

This is a coverage/depth review with targeted technical and numerical checks,
not an independent line-by-line certification. “Retained” means the chapter's
existing derivation, cases, or exercises already supply its intended teaching
role; it does not mean every statement was experimentally reproduced. Existing
data casework, streaming solutions, and foundation answer criteria serve several
adjacent chapters and are not duplicated at every chapter boundary.

| Chapter | Disposition | Teaching evidence / boundary |
| --- | --- | --- |
| The Model, Workload, and System Contract | Retained | Joint workload, outcome/invariant definitions, design exercises |
| From Text to Tokens, Targets, and Loss | Added | BPE, vocabulary budget, teacher forcing, three masks, numerical loss |
| Optimization as a Coupled Dynamical System | Expanded | Muon alongside momentum/AdamW; precision and committed-step invariants |
| Transformer Architecture as Resource Allocation | Expanded | Explicit shapes, SwiGLU, RMSNorm, RoPE calculation and conventions |
| Compressed, Sparse, and Recurrent Model State | Added | MLA absorption, sparse-selector counterexample, delta recurrence, SSM and hybrid ledgers |
| Scale, Memory, and Performance Models | Expanded | Fixed-budget scaling derivative; running KV and queueing models |
| Measurement and Experimental Judgment | Retained | Confidence, causal comparisons, KV-layout decision and foundation answer criteria |
| Designing a Training Recipe | Retained | Experiment ladder, budget, stages, recipe table and recovery |
| Data Contracts, Provenance, and Normalization | Retained | Canonical schema, transformations, provenance and normalization choices |
| Deduplication, Quality, and Contamination | Retained | MinHash/LSH, cluster/split leakage, filtering and adversarial contamination |
| Synthetic Data and Mixture Design | Expanded | Language-aware curation and controlled filter/mixture experiment |
| Training Data Platform and Release Engineering | Retained | Immutable manifests, pipelines, validation, release and removal protocols |
| Applied Data-System Casework | Retained | Ten worked cases connecting preceding data chapters |
| Distillation, KL Divergence, and Model Sizing | Retained | KL direction, temperature, teacher pipeline, draft and sizing tradeoffs |
| Supervised Adaptation and Low-Rank Training | Added | Loss masks, global token normalization, LoRA count, QLoRA gradient/merge boundaries |
| Post-Training and Alignment | Expanded | DPO derivation, numerical example, reward/refusal caveats and answer criteria |
| Reinforcement Learning for Reasoning and Tool Use | Added | PPO/GRPO, DAPO/GSPO, SDPO, verifiers, rollout cost, pass-at-k |
| Request Lifecycle, Metrics, and Workload Models | Retained | Open/closed loop, arrival distributions, TTFT/TPOT definitions and exercises |
| Prefill, Decode, and Performance Modeling | Retained | Phase-specific FLOPs/bytes, batching and running-model lower bounds |
| KV Cache, Paging, and Prefix Reuse | Retained | Block mapping, identity, sharing, eviction, migration and security |
| Scheduling, Batching, and Admission Control | Retained | Bounded scheduling, token budgets, chunked prefill, cancellation and overload |
| Sampling, Structured Output, and Speculative Decoding | Expanded | Exact residual proof, EAGLE-3/DFlash families, hybrid rollback |
| Parallel, Replicated, and Disaggregated Serving | Retained | Distinct phase groups, KV transfer, pinned PCP/TP/DCP source check |
| Quantization, Compression, and Adapter Serving | Expanded | Scalar error/scales, GPTQ/AWQ/SmoothQuant/SpinQuant, MXFP8/NVFP4 |
| Production Architecture, Capacity, and Reliability | Retained | Capacity, release identity, failure domains, canaries and SLO goodput |
| GPU Execution, Memory, and Resource Accounting | Retained | Coalescing, banks, occupancy and synchronization with worked solutions |
| Host-Device Orchestration, Streams, and CUDA Graphs | Retained | Async host boundaries, graph lifecycle and measured-region semantics |
| Hierarchical Matrix Multiplication | Retained | Scalar/tiled references, register reuse, tensor cores, split-K and grouped GEMM |
| Reductions, Prefix Scans, and Histograms | Retained | Trees, scans, compaction, irregular access and numerical behavior |
| Softmax, Normalization, Top-K, and Sampling | Retained | Stable/online softmax, backward, normalization and distributed selection |
| FlashAttention and IO-Aware Exact Attention | Retained | Tiled recurrence, backward, saved state, masks and partitioning |
| Blackwell Pipelines and Modern Attention Kernels | Added | TMEM/TMA ownership, buffer generations, timing model, FA4 and FP4 limits |
| CUDA Kernels for LLM Inference | Retained | KV append, paged/split attention, GQA, quantized GEMM and MoE |
| Kernel Engineering, Profiling, and Correctness | Retained | Three baseline boundaries, cold/warm measurement, race/numerical tests |
| Communication Models, Collectives, and Topology | Retained | Alpha-beta, ring traffic, overlap, physical groups and deadlock |
| Data Parallelism, ZeRO, and Fully Sharded Training | Retained | State ledger, gathers, checkpoint/offload and numerical agreement |
| Tensor, Sequence, and Context Parallelism | Retained | Row/column layouts, vocabulary loss, CP merge and geometry |
| Pipeline Parallelism and Hybrid Plans | Retained | GPipe/1F1B, stage allocation, microbatch/activation tradeoffs |
| Mixture-of-Experts and Sparse Communication | Expanded | Dispatch/combine mapping, DeepEP V1/V2 boundary, DualPipe and resource contention |
| Distributed Inference and Stateful Placement | Retained | PCP/DCP distinction, rank geometry, transfer and streaming recovery |
| Distributed Reinforcement Learning and Policy Freshness | Added | Role ownership, behavior policy, lag/ESS, backpressure and committed batches |
| Distributed Checkpoints, Recovery, and Elasticity | Retained | Atomic manifests, logical resharding, failure interval and restore drills |
| Cluster Scheduling, Observability, and Distributed Diagnosis | Retained | Gang placement, stragglers, flight recorder and end-to-end diagnosis |
| Exact Streaming Queries and Time Windows | Retained | Exact top-k, window deques, event-time semantics and error boundaries |
| Online Statistics and Sampling | Retained | Welford and reservoir derivations with worked examples |
| Heavy Hitters and Probabilistic Sketches | Retained | Misra-Gries, CMS, HLL, Bloom and quantile error models |
| Building a Streaming Telemetry Service | Retained | Complete service and ten worked solutions |
| Compact Review of ML Algorithms in Production | Retained | Shape/numerical contracts and worked answer criteria |
| An Engineering System Design and Review Method | Retained | Contract, estimation, critical path, failure and decision method |
| RAG, Vector Search, and Evaluation Pipelines | Expanded | RRF/MaxSim, hierarchical/graph alternatives, authorized service fixture |
| Building and Evaluating a Bounded Agent Loop | Added | Executable loop, capability denial, ambiguous writes, memory and task evaluation |
| Efficient Frontier Models and Reasoning Training | Expanded | V4, Qwen3.5, Nemotron 3, mHC and Engram with independent toy examples |
| Serving, Attention, and Memory Hierarchies | Expanded | 2026 mechanism-to-bottleneck map; prior results explicitly historical |
| Multimodal Representations: Images, Video, and Speech | Added | Patch budgets, contrastive/generative objectives, temporal/audio and interaction contracts |
| Diffusion and Block-Parallel Language Generation | Added | Masked objective, generation schedule, cache validity and target/draft distinction |
| Multimodal and Tool-Using Systems | Retained | Grounded coordinates and adversarial control boundary; new chapters supply foundations |
| From Research Result to Production Decision | Retained | Result card, evidence ladder and explicit rejection conditions |
| Executive Technical Communication | Retained | Decision memo, uncertainty and durable decision ledger |
| Operating Mechanisms for ML Systems | Expanded | Cross-stack release gates, ownership and Amdahl decision example |
| Incident Leadership and High-Risk Change | Retained | Causal diagnosis, mitigation clock, migrations and rollback |
| Strategy, Vision, and the First 90 Days | Retained | Staged discovery, commitments and falsifiable milestones |
| Leadership Evidence and Reflective Practice | Retained | Three applied decision cases and evidence-based reflection |
| Cross-Layer Design Prompt Bank | Retained | Fifteen cross-layer scenarios with answer structures |
| End-to-End Learning Lab and Capstone | Added | Train/serialize/generate fixture; connected service design and self-check |
| Formula and Capacity Sheet | Expanded | Hybrid state, LoRA, DPO/PPO/GRPO and ESS with assumptions |
| CUDA Engineering Checklist | Retained | Correctness, race, timing and resource checklists |
| Diagnostic Question Bank | Corrected | Exact attention semantics distinguished from numerical approximations |
| Glossary and Decision Index | Expanded | Modern terms and symptom-to-chapter navigation |

## What this pass does not establish

Release verification is recorded in `docs/release-readiness.md` and
`docs/release-notes-2026-09-07-expanded.md`: 66 passing CPU tests, a 402-page
PDF, and a reachability check of 145 unique manuscript URLs. The manuscript
test suite checks that this ledger contains each of the 68 chapter titles
exactly once; that is a completeness check on the ledger, not a correctness
proof for every claim.

- It does not reproduce GPU kernel speedups, distributed training runs, frontier
  convergence, or real-model agent robustness. Those require suitable hardware,
  checkpoints, datasets, and controlled experiments.
- It does not promise coverage of every recent paper. Techniques are selected
  for distinct mechanisms; names and leaderboard maxima are not a completion
  criterion. The snapshot has an explicit cutoff.
- The standard-library fixtures are educational references. The tiny LM is a
  bigram model, not a transformer; retrieval and agents use synthetic/scripted
  environments, not production quality measurements.
- An independent technical/copy review remains desirable before calling this
  a finished public textbook. This pass produces a substantially more complete
  private learning draft, not an unconditional certification.
