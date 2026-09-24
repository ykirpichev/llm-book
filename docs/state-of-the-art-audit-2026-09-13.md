# State-of-the-Art Audit — September 13, 2026

The [September 24 Part VII worked additions](part7-followup-2026-09-24.md) implement the follow-up teaching and recovery recommendations; the frontier-model map remains dated September 23.

The [September 24 Part VI revision](part6-update-2026-09-24.md) updates Chapters 46-53. Its source and validation record supersedes the corresponding current-state assessment below.

The [September 23 Part VII revision](part7-update-2026-09-23.md) supersedes the current-state assessment of Chapters 54-59 below. Earlier claim checks remain historical evidence, not a survey through the new date.

## Executive finding

The manuscript's technical center remains current. Its strongest choice is to organize around durable resource and correctness models rather than transient leaderboards. A chapter-by-chapter audit found no missing development that justified rewriting the training-data, CUDA-basics, distributed-training, streaming-algorithm, leadership, or appendix chapters. Six targeted updates were warranted:

1. output-length uncertainty is now treated explicitly as a joint KV reservation, routing, prefix-reuse, and parallelism problem;
2. the distributed prefix-cache discussion now maps the design to maintained implementations: Dynamo, LMCache, llm-d, Mooncake, and NIXL;
3. the distributed-inference chapter now separates router belief, cache indexing, transfer completion, admission, and ownership publication;
4. the attention taxonomy now names local/windowed, hierarchical learned-sparse, latent, recurrent/linear, and hybrid state explicitly, with Native Sparse Attention and Kimi Linear as primary examples;
5. the distributed-inference chapter now includes a phase-specific ledger for TP, PCP, DCP, prefill/decode disaggregation, and serving data parallelism;
6. the bounded-agent chapter now distinguishes MCP's host/server interoperability boundary from A2A's agent-to-agent boundary, while keeping authorization in deterministic host code.

The review cutoff is September 13, 2026. “Current” means supported by a primary paper, official specification, official model report, or maintained project documentation available by that date. A new item was integrated only when it changed an engineering decision, failure boundary, measurement, or reference implementation. Benchmark rank alone was not sufficient.

## Evidence and selection method

The audit compared every chapter against six active fronts: model architecture and post-training; inference scheduling and KV state; CUDA and attention kernels; distributed execution; multimodal and agent protocols; and evaluation/reproducibility. Primary evidence included model and systems reports, official runtime documentation, and protocol specifications. In particular, current serving stacks now expose cache-event routing, tiered KV storage, and prefill/decode transfer as separable components rather than one monolithic “distributed cache.”[1][2][3][4] The latest vLLM documentation also labels disaggregated prefill experimental and explicitly warns that it does not inherently improve throughput, reinforcing the book's break-even analysis rather than displacing it.[5]

Recent architecture evidence continues to support the manuscript's treatment of sparse activation, hybrid recurrent/attention state, compressed memory, and variable test-time compute.[6][7][8][9] Current CUDA documentation still makes architecture-aware pipelines, asynchronous movement, and shape-specific validation the relevant engineering frame.[10][11] For agents, MCP and A2A have matured into concrete interoperability specifications, but their own authentication and capability mechanisms do not remove application-level authorization or trust-boundary work.[12][13]

## Chapter-by-chapter disposition

“Current” means the chapter's durable treatment still covers the active frontier. “Integrated” names a change made in this pass. “Covered in Part VII” means a fast-moving development is intentionally localized to the dated survey rather than duplicated in a durable chapter.

### Part I — Foundations

| Chapter | Disposition | Audit note |
| --- | --- | --- |
| The Model, Workload, and System Contract | Current | SLO, quality, security, and cost contracts remain the right boundary for comparing new models. |
| From Text to Tokens, Targets, and Loss | Current | Tokenization and causal-loss mechanics are stable; newer architectures do not alter these definitions universally. |
| Optimization as a Coupled Dynamical System | Current | Numerical state, clipping, scheduling, and reproducibility remain the necessary basis for newer optimizers and low-precision recipes. |
| Transformer Architecture as Resource Allocation | Current | The chapter already derives attention, GQA, RoPE, normalization, and MoE as resource choices. |
| Compressed, Sparse, and Recurrent Model State | **Integrated** | Expanded the mechanism taxonomy and added Native Sparse Attention and Kimi Linear to cover hierarchical learned sparsity and newer gated linear-attention hybrids.[6][7][8][16][17] |
| Scale, Memory, and Performance Models | Current | FLOP, byte, occupancy, and latency models remain architecture-independent acceptance tools. |
| Measurement and Experimental Judgment | Current | Matched boundaries and uncertainty are more useful than adding a transient benchmark table. |

### Part II — Training and Post-Training

| Chapter | Disposition | Audit note |
| --- | --- | --- |
| Designing a Training Recipe | Current | Recipe identity, ablations, and stop criteria already govern newer data and optimizer combinations. |
| Data Contracts, Provenance, and Normalization | Current | Provenance, licensing, privacy, and normalization remain unresolved operational constraints, not superseded techniques. |
| Deduplication, Quality, and Contamination | Current | Exact/near-duplicate separation and contamination auditing remain state of practice. |
| Synthetic Data and Mixture Design | Current | Generator, verifier, filtering, diversity, and mixture feedback loops cover current synthetic-data pipelines. |
| Training Data Platform and Release Engineering | Current | Immutable manifests, lineage, staged publication, and rollback remain the production frontier. |
| Applied Data-System Casework | Current | The cases exercise current failure modes without depending on a vendor stack. |
| Distillation, KL Divergence, and Model Sizing | Current | Reasoning distillation increases the importance of the existing teacher/student boundary; Part VII supplies the dated example.[9] |
| Supervised Adaptation and Low-Rank Training | Current | Adapter rank, memory, merge, and serving implications remain accurate. |
| Post-Training and Alignment | Current | Preference objectives and verifier limitations remain the essential framework. |
| Reinforcement Learning for Reasoning and Tool Use | Current | Group-relative and sequence-level objectives, verifier risk, and rollout accounting already cover recent reasoning-RL directions. |

### Part III — Inference Systems

| Chapter | Disposition | Audit note |
| --- | --- | --- |
| Request Lifecycle, Metrics, and Workload Models | Current | TTFT, inter-token latency, goodput, and correlated workload traces remain the correct comparison boundary. |
| Prefill, Decode, and Performance Modeling | Current | Phase asymmetry remains fundamental, including for hybrid and disaggregated engines. |
| KV Cache, Paging, and Prefix Reuse | **Integrated** | Added risk-calibrated reservation under output uncertainty and an implementation map for Dynamo, LMCache, llm-d, and Mooncake.[1][2][3][14] |
| Scheduling, Batching, and Admission Control | Current | Token budgets, chunked prefill, fairness, cancellation, and early rejection remain the central mechanisms; robust reservation now cross-links conceptually from the KV chapter.[14] |
| Sampling, Structured Output, and Speculative Decoding | Current | Exact verification, feature drafts, block drafts, and branch-state accounting already cover the current frontier. |
| Parallel, Replicated, and Disaggregated Serving | Current | The chapter already derives transfer break-even, handoff state, affinity, and backpressure; implementation evidence was added where cache contracts are introduced. |
| Quantization, Compression, and Adapter Serving | Current | Weight, activation, KV, MXFP8, NVFP4, and multi-adapter tradeoffs already cover current deployment choices. |
| Production Architecture, Capacity, and Reliability | Current | Control/data planes, rollout, observability, degradation, and abuse resistance remain complete at the book's intended depth. |

### Part IV — CUDA and Kernel Engineering

| Chapter | Disposition | Audit note |
| --- | --- | --- |
| GPU Execution, Memory, and Resource Accounting | Current | Execution and occupancy fundamentals remain stable. |
| Host-Device Orchestration, Streams, and CUDA Graphs | Current | Asynchronous orchestration and graph constraints remain current in official CUDA guidance.[10] |
| Hierarchical Matrix Multiplication | Current | Tiling, reuse, precision, and shape dispatch remain the core design method. |
| Reductions, Prefix Scans, and Histograms | Current | Primitive semantics and contention tradeoffs are unchanged. |
| Softmax, Normalization, Top-K, and Sampling | Current | Numerics and fused-reduction reasoning remain current. |
| FlashAttention and IO-Aware Exact Attention | Current | The exact online-softmax derivation remains foundational. |
| Blackwell Pipelines and Modern Attention Kernels | Current | Already covers asynchronous Blackwell-era pipelines and current attention work; official CUDA 13.4 material confirms the hardware framing.[11] |
| CUDA Kernels for LLM Inference | Current | Shape-specific fusion, quantization metadata, and decoding primitives remain appropriate. |
| Kernel Engineering, Profiling, and Correctness | Current | Differential tests, sanitizer/profiler evidence, and dispatch boundaries remain the right standard. |

### Part V — Distributed Systems

| Chapter | Disposition | Audit note |
| --- | --- | --- |
| Communication Models, Collectives, and Topology | Current | Alpha-beta reasoning, collectives, registration, and topology remain durable. |
| Data Parallelism, ZeRO, and Fully Sharded Training | Current | The chapter's logical sharding model remains valid for FSDP2's per-parameter DTensor representation.[15] |
| Tensor, Sequence, and Context Parallelism | Current | Layout-as-type and communication accounting remain the correct abstraction. |
| Pipeline Parallelism and Hybrid Plans | Current | Bubble, partition, and global plan evaluation remain current. |
| Mixture-of-Experts and Sparse Communication | Current | Expert imbalance, all-to-all, replication, and fused communication remain active bottlenecks. |
| Distributed Inference and Stateful Placement | **Integrated** | Added the deployable split among routing, event-fed indexing, direct NIXL transfer, admission, and ownership publication, plus a phase-specific TP/PCP/DCP/P-D/DP sharding ledger.[1][3][4][18] |
| Distributed Reinforcement Learning and Policy Freshness | Current | Role separation, stale-policy accounting, backpressure, and publication remain central to asynchronous RL. |
| Distributed Checkpoints, Recovery, and Elasticity | Current | Logical tensor identity, manifests, resharding, and membership epochs remain current. |
| Cluster Scheduling, Observability, and Distributed Diagnosis | Current | Gang placement, multi-resource admission, and rank-correlated diagnosis remain the production standard. |

### Part VI — Algorithms and Product Systems

| Chapter | Disposition | Audit note |
| --- | --- | --- |
| Exact Streaming Queries and Time Windows | Current | Exactness and ordering contracts are stable algorithmic foundations. |
| Online Statistics and Sampling | Current | Mergeability, numerical stability, and sampling proofs remain current. |
| Heavy Hitters and Probabilistic Sketches | Current | Stream-model distinctions and error guarantees remain canonical. |
| Building a Streaming Telemetry Service | Current | Offsets, windows, watermarking, recovery, and overload remain the correct system boundary. |
| Compact Review of ML Algorithms in Production | Current | The chapter is intentionally compact and decision-oriented; no new catalog entries improve it. |
| An Engineering System Design and Review Method | Current | Explicit contracts, bottleneck models, failure modes, and experiments remain durable. |
| RAG, Vector Search, and Evaluation Pipelines | Current | Hybrid retrieval, ACLs, revision identity, grounding, and abstention already cover current production concerns. |
| Building and Evaluating a Bounded Agent Loop | **Integrated** | Added MCP and A2A interoperability boundaries while preserving deterministic authorization, idempotency, provenance, and budgets.[12][13] |

### Part VII — Recent State of the Art

| Chapter | Disposition | Audit note |
| --- | --- | --- |
| Efficient Frontier Models and Reasoning Training | Current | Already includes DeepSeek-V4, Qwen3.5, Nemotron 3, mHC, Engram, R1, and an August 2026 test-time-scaling taxonomy.[6][7][8][9] |
| Serving, Attention, and Memory Hierarchies | **Integrated** | Updated the cutoff and connected Mooncake's paper architecture to maintained Dynamo, LMCache, llm-d, and NIXL components.[1][2][3][4] |
| Multimodal Representations: Images, Video, and Speech | Current | Covers native-resolution vision, temporal sampling, codecs, speech/text alignment, and end-to-end interaction latency. |
| Diffusion and Block-Parallel Language Generation | Current | Covers masked diffusion, block diffusion, cache validity, and the distinction between draft and target models. |
| Multimodal and Tool-Using Systems | Covered in Part VI | Security results remain current; protocol mechanics were integrated into the bounded-agent chapter instead of duplicating them here. |
| From Research Result to Production Decision | Current | Result cards, tuned baselines, layered reproduction, and retirement criteria are precisely what fast-moving claims require. |

### Part VIII — Technical Leadership

| Chapter | Disposition | Audit note |
| --- | --- | --- |
| Executive Technical Communication | Current | Decision, evidence, uncertainty, and ask remain the durable structure. |
| Operating Mechanisms for ML Systems | Current | Review cadences, ownership, metrics, and escalation are organizational mechanisms, not model-version details. |
| Incident Leadership and High-Risk Change | Current | Incident command, blast-radius control, rollback, and learning remain complete. |
| Strategy, Vision, and the First 90 Days | Current | Portfolio and capability sequencing remain appropriate. |
| Leadership Evidence and Reflective Practice | Current | Evidence-based reflection does not need a SOTA insert. |
| Cross-Layer Design Prompt Bank | Current | Prompts already span model, data, serving, kernel, distributed, and organizational boundaries. |

### Part IX — Appendices

| Chapter | Disposition | Audit note |
| --- | --- | --- |
| End-to-End Learning Lab and Capstone | Current | The lab exercises the book's complete decision chain. |
| Formula and Capacity Sheet | Current | The formulas remain dimensionally and conceptually stable. |
| CUDA Engineering Checklist | Current | The checklist remains aligned with current CUDA workflow.[10][11] |
| Diagnostic Question Bank | Current | Questions test mechanisms rather than vendor trivia. |
| Glossary and Decision Index | Current | New implementation names do not warrant new general-purpose terms. |

## What was deliberately not added

- Unverified September 2026 preprints were not added merely for recency.
- Vendor performance maxima were not copied into durable chapters without matched workload, hardware, quality, and end-to-end boundaries.
- Every chapter did not receive a “latest” paragraph. That would date stable material and obscure the few changes that alter design choices.
- Protocol adoption was not presented as a security solution. Discovery and typed messages improve interoperability; they do not establish authority.
- Disaggregated prefill was not described as a universal throughput optimization. Current vLLM documentation explicitly scopes it to phase tuning and tail-latency control and labels the facility experimental.[5]

## Release judgment

After the targeted integrations, no known state-of-the-art omission changes a core engineering conclusion in the manuscript. The dated Part VII still needs periodic maintenance, and implementation links should be tested at each release because feature matrices evolve quickly. For broad sharing, the book should continue to label paper-reported numbers, pin revisions when describing exact runtime behavior, and keep the durable chapters mechanism-first.

## Sources

1. [NVIDIA Dynamo repository and KV-aware routing documentation](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/cli/kv-aware-routing/overview.mdx).
2. [LMCache official repository](https://github.com/LMCache/LMCache).
3. [llm-d KV Cache Management architecture](https://github.com/llm-d/llm-d/blob/main/docs/architecture/advanced/kv-management/README.md).
4. [NVIDIA Dynamo: Disaggregated Serving](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/disaggregated-serving) and [NIXL official repository](https://github.com/ai-dynamo/nixl).
5. [vLLM: Disaggregated Prefilling](https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md).
6. [DeepSeek-V4 report](https://arxiv.org/abs/2606.19348).
7. [Qwen3.5-35B-A3B official model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base).
8. [NVIDIA Nemotron 3 Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf).
9. [Test-Time Scaling in Reasoning LLMs, version 2](https://arxiv.org/abs/2608.04001v2).
10. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/).
11. [NVIDIA Blackwell Tuning Guide 13.4](https://docs.nvidia.com/cuda/blackwell-tuning-guide/).
12. [Model Context Protocol specification, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25).
13. [Agent2Agent Protocol specification, version 1.0.0](https://github.com/a2aproject/A2A/blob/main/docs/specification.md).
14. [Robust KV Cache Management for LLM Serving under Output Token Length Uncertainty](https://arxiv.org/abs/2607.16892).
15. [PyTorch FSDP2 `fully_shard` documentation](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html).
16. [Native Sparse Attention](https://arxiv.org/abs/2502.11089).
17. [Kimi Linear](https://arxiv.org/abs/2510.26692).
18. [vLLM Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/).
