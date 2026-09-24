# Part VII - Recent State of the Art

This part was substantively updated on September 23, 2026, with worked comparisons, recovery examples, and runnable serving/adoption experiments added on September 24. The frontier-model snapshot below remains dated September 23. It connects current model cards, research reports, and implementation evidence to the mechanisms developed throughout the book. It is a dated engineering snapshot, not an exhaustive catalogue or permanent leaderboard. Earlier results are explicitly retained as foundations. Reported speedups and benchmark scores belong to the cited source's hardware, software, model, workload, baseline, and quality threshold; they were not reproduced on accelerators for this book. Part VI was also revised on September 24; the remaining parts retain their September 13 research cutoff unless explicitly dated.

The durable value of a recent result is usually not its rank. It is the mechanism that changed the resource model: sparse activation, better load balancing, reinforcement learning with verifiable rewards, explicit inference-time compute, asynchronous attention pipelines, disaggregated KV state, hierarchical memory, native-resolution multimodality, or enforceable trust boundaries for tools.

Read this part as a decision filter, not a chronology. For each result, first identify the mechanism and the resource or behavior it changes; then reconstruct the evaluation boundary and compare it with your workload. Finally, name the integration cost, quality guardrail, and experiment that would justify adoption. Return to Parts II–VI when a claim depends on training data, serving state, kernel behavior, topology, or application policy—the dated examples here do not replace those durable models.

:::callout decision|How to read a state-of-the-art claim
Record the evaluated system, baseline, workload distribution, hardware, precision, quality constraint, and end-to-end boundary. Treat an isolated kernel speedup, benchmark score, or best-case throughput number as a hypothesis until the same advantage appears under the production contract.
:::

## Efficient Frontier Models and Reasoning Training

LEAD: Recent frontier systems show that capability is increasingly a co-design problem across architecture, data, optimization, precision, communication, and inference-time computation.

### The frontier on September 23, 2026

The frontier now includes several kinds of evidence: hosted capability, downloadable checkpoints, architectural reports, and experimental inference systems. The following map identifies representative current systems, not a ranking. A model card's existence does not establish access for every account, and open weights do not imply an open training corpus or unrestricted licensing.

| System and primary evidence | What is established | Engineering question |
| --- | --- | --- |
| [GPT-6 Astra, Sol, and Luna](https://developers.openai.com/api/docs/guides/latest-model) | Official API guidance documents three capability/cost tiers, asynchronous tools, and mid-turn steering | How should routing, cancellation, and concurrent tool state change? |
| [Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5), announced September 22 | A current hosted model with revised serving economics and account-dependent preserved-thinking requirements | Does a lower token price reduce cost per verified task? |
| [Claude Fable / Mythos 5.1](https://www.anthropic.com/claude-fable-and-mythos-5-1), September 1 | The same model under different safeguards; Fable is generally available, Mythos has restricted access | Which capability and access policy will the actual application receive? |
| [Gemini 3.8 Flash](https://deepmind.google/models/model-cards/gemini-3-8-flash/), September 2 card | Text output from text, image, audio, and video inputs, with configurable effort | Which modality, reasoning budget, and harness produced a result? |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Downloadable dense vision-language model with hybrid attention | Can recurrent state and attention KV be scheduled and restored correctly? |
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Downloadable multimodal model with redesigned cache ownership | What is persistent, shared, or reconstructed during a cache hit? |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | Downloadable hybrid MoE with model-specific license and quantized deployment recipe | Does active compute fit while total expert storage and routing remain feasible? |
| [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) and [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | Distinct checkpoints: post-training gains versus a new multimodal hybrid base | Is a gain architectural, post-training, or a harness change? |

Mutable API documentation and checkpoint cards in this map were inspected on September 23. Use an immutable checkpoint revision and model/API identifier in an actual evaluation. Preserve the provider's request configuration and response metadata. Product names alone are insufficient to reconstruct a run.

Hosted capability advances also change the application contract. OpenAI's GPT-6 guidance documents tools that can run while reasoning continues and user instructions that can arrive during a turn. This motivates explicit pending-call IDs, dependency tracking, cancellation, and stale-result rejection in the application. It does not reveal a proprietary parameter count or training recipe. Anthropic's Opus 5.5 announcement applies preserved thinking to Opus 5.5 and Fable 5.1 for API accounts created on or after August 31, 2026. A migration therefore needs account-specific conversation-state and access-policy tests in addition to answer-quality tests. These are observable interface changes, not evidence that either vendor has solved autonomous reliability.

### Sparse activation changes the economic unit

[DeepSeek-V3](https://arxiv.org/html/2412.19437v2), a December 2024 foundation for this discussion, reports 671 billion total parameters while activating 37 billion parameters per token. Its technical report describes 14.8 trillion pretraining tokens and 2.788 million H800 GPU-hours for the official training run, including context extension and post-training but excluding prior research and ablation experiments. This is not the total research-and-development cost. The architecture combines a mixture-of-experts design, Multi-head Latent Attention, multi-token prediction, and an auxiliary-loss-free load-balancing strategy.

The system lesson is not that every model should copy one expert layout. Sparse activation separates three quantities that dense scaling often conflates:

- total parameter capacity, which affects checkpoint storage and aggregate expert memory;
- active parameters per token, which more directly control token-level compute;
- routed communication and imbalance, which can erase theoretical savings.

An MoE capacity plan therefore needs distributions, not only averages. Measure tokens per expert by layer and workload slice, overflow or dropped-token behavior, all-to-all bytes, hot-rank tails, expert memory residency, and useful throughput after communication. A balanced global histogram can still hide transient microbatch hotspots.

:::callout insight|Reported training cost is a system result
A GPU-hour number compresses hardware availability, precision, kernel efficiency, network topology, failure rate, checkpoint policy, and experiment reuse. Compare it only after reconstructing what the number includes.
:::

### A 2026 architecture map

The [DeepSeek-V4 report](https://arxiv.org/abs/2606.19348) combines compressed sparse attention (CSA), heavily compressed attention (HCA), constrained residual mixing, and Muon. CSA compresses groups of history entries before sparse selection; HCA uses stronger compression with attention over that shorter representation. A local-window branch retains nearby detail. This differs from merely selecting fewer entries from an unchanged full-resolution cache. The resulting cache has multiple representations and partially formed tail state, so memory admission and reuse must understand each component.

As a generic calculation, compressing 1,024 historical entries in groups of eight yields 128 entries; attending to 16 selected compressed entries is not the same operation as attending to 16 original tokens. Each compressed entry summarizes several positions, and information lost during compression cannot be restored by the selector. This example explains the distinction; its group sizes are not the V4 configuration.

The official [Qwen3.5-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base) documents a hybrid of Gated DeltaNet and full attention. Part I derives why that means recurrent state plus selected attention-layer KV, not one uniformly growing cache. The [Nemotron 3 Super report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) combines Mamba/attention with MoE and introduces LatentMoE: projecting tokens into a smaller expert-computation space changes routed bytes and parameter loading. It also describes an NVFP4 training recipe. Neither model should be described simply as “an MoE transformer” when predicting its state or precision behavior.

For a generic expert layer, reducing communicated width from 4,096 to 1,024 quarters the activation payload per assignment at fixed dtype and routing count. Increasing active experts fourfold can spend that saving again. Projection overhead, scaling metadata, and expert compute remain in the ledger. Latent width, active count, and total count must therefore be listed together when making an efficiency claim.

### Current open models require different state ledgers

The [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B) specifies 64 layers arranged as sixteen groups of three Gated DeltaNet layers and one gated-attention layer. It lists a native context of 262,144 tokens, with extension up to one million. The hosted million-token service is described as forthcoming in the inspected card; do not turn that statement into an availability claim. Native, extended, and hosted context limits are different evidence. Dense feed-forward computation also does not imply full attention at every layer.

The [Kimi K3 card](https://huggingface.co/moonshotai/Kimi-K3) lists 2.8T total and 104B active parameters, 69 KDA layers and 24 gated-MLA layers, and MXFP4 routed-expert weights with MXFP8 activations, while other components retain higher precision. Its [July technical report](https://arxiv.org/abs/2607.24653) provides the research context. Active parameters help estimate computation; they do not make the full checkpoint or expert placement disappear. Hybrid attention adds recurrent snapshots to the attention-cache ledger, and quantized weights add scale metadata and hardware-dependent kernels.

[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) attributes its improvement over GLM-5.2 to post-training on the same base. [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), despite the related name, describes a newly trained multimodal base with sparse/linear attention and mHC. This distinction matters experimentally: changing a suffix can change the attention state, modality contract, and kernels, not only the price or decoding budget. Inspect each checkpoint's license separately from the license of this book.

For a hybrid model, estimate persistent state as the sum of attention-layer KV, recurrent-layer state, compressed history, and any shared representations. Then add temporary workspaces, speculative branches, graph buffers, and allocator reserve. A million-token limit is an interface or evaluation boundary, not a claim that every million-token request fits one device or retains every detail with equal accuracy.

### Residual routing and learned lookup memory

[Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) studies multiple residual streams and constrained mixing between them. A doubly stochastic mixing matrix has nonnegative entries with every row and column summing to one. The intent is to retain richer routing without unconstrained amplification in the residual transport. This is a training architecture change, not a post-hoc serving flag.

For an independent two-stream example, the matrix `[[0.75,0.25],[0.25,0.75]]` maps scalar streams `[2,10]` to `[4,8]`. Each new stream is a convex combination; the sum remains twelve. By contrast, multiplying both streams by two doubles the sum at every layer. The constrained example illustrates transport stability, not a proof that an entire nonlinear network cannot diverge. Layer transformations, gates, finite-iteration constraint enforcement, and optimizer behavior still matter.

#### Compare depth selection with stream transport

[Attention Residuals, March 2026 v1](https://arxiv.org/abs/2603.15031v1), replaces fixed residual accumulation with learned selection over earlier layer outputs at the same token position. A layer-specific learned pseudo-query scores normalized source representations; a softmax weights the original representations as values. The query is learned, while its weights depend on the input through the keys. Block AttnRes reduces the retained sources to the embedding, completed block sums, and the current partial block sum when present.

:::diagram residual_routing|Original schematic of two different routing axes. AttnRes selects sources across depth for one token. mHC transports a widened set of streams at the current depth and separately reads, transforms, and writes a layer update. Neither drawing is token-to-token attention.

For one coordinate of three vector-valued sources, values `[2,4,10]` and illustrative attention weights `[0.2,0.3,0.5]` produce `0.2*2 + 0.3*4 + 0.5*10 = 6.6`. The full source vectors determine the weights. This is a selected aggregate, not the two-stream transport `[2,10] -> [4,8]` above. The [mHC formulation, January 2026 v2](https://arxiv.org/html/2512.24880v2) keeps readout and writeback mappings separate from its constrained residual transport; finite Sinkhorn iterations approximate the doubly stochastic constraint.

The engineering comparison therefore measures different state: earlier depth outputs or block summaries for AttnRes, versus widened current-depth streams and mixing work for mHC. These are activation and training-architecture costs, not interchangeable reductions in historical token KV. Compare matched training budgets, loss, activation memory and end-to-end execution. The figure does not establish which architecture wins.

[Engram](https://arxiv.org/abs/2601.07372) explores learned conditional memory through n-gram-based lookup as a complement to conditional expert computation. The basic distinction is computation versus lookup: a short token pattern selects stored vectors, while contextual processing determines how useful those vectors are. A lookup table can hold recurring local associations without recomputing them through every dense layer. Hash collisions, table size, placement, and the quality of the contextual gate are engineering concerns.

This is not the same as retrieving current documents from an enterprise corpus. Learned lookup parameters do not automatically provide citations, per-user permissions, or instant factual updates. Nor does discussing Engram establish that a particular released model uses it; require that model's own architecture evidence. Similarly, a fixed lookup-time complexity says little about host/device transfer latency or cache misses.

### Reasoning from reinforcement learning

The January 2025 [DeepSeek-R1-Zero](https://arxiv.org/abs/2501.12948) study reports that large-scale reinforcement learning without a supervised fine-tuning warm start can elicit stronger reasoning behavior, but also reports readability and language-mixing problems. DeepSeek-R1 adds cold-start data and a multi-stage training process before and after reinforcement learning. The report also releases distilled dense models from 1.5B through 70B parameters and reports that reasoning behavior can transfer into smaller students.

This evidence changes the post-training design space in three ways.

1. **Verifiable rewards are high leverage.** Mathematics, code, formal constraints, and tool outcomes support scalable outcome checks, but a reward is only as sound as its verifier and sandbox.
2. **Capability and presentation are separate objectives.** A policy can improve task reward while degrading readability, language consistency, calibration, or safety.
3. **Distillation is a deployment lever.** A high-compute teacher can create traces or targets for a smaller model, but the student still requires independent quality, contamination, and serving evaluation.

Reward design must account for false acceptance, reward hacking, length incentives, duplicated samples, and train-evaluation overlap. Log the full rollout policy, sampling configuration, verifier version, reward components, rejected trajectories, and update batch. Without those artifacts, an apparent algorithmic gain may be an unrepeatable data-selection effect.

### From reasoning rewards to agent environments

The [DeepSeek-V4.1 technical report, section 5.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf), explicitly retains supervised fine-tuning, reinforcement learning, and on-policy distillation. It attributes the release's post-training advances to task synthesis, interactive environment construction, filtering, deduplication, and difficulty calibration rather than a new optimization algorithm. Its training unit includes a problem, an environment, and a verification system. This is a broader unit of curriculum design than an isolated answer with a scalar reward.

[Kimi K3's report, version 2](https://arxiv.org/abs/2607.24653v2), likewise connects long-context agentic RL to persistent rollout and sandbox state. A long trajectory consumes environment lifetime, tool execution, checkpoint storage, and policy-freshness budget as well as tokens. The infrastructure must distinguish a slow but productive rollout from a stuck tool, an invalid environment, or a policy exploiting its grader.

In on-policy distillation, a teacher provides learning targets on states visited by the student rather than only a fixed collection of teacher-generated solutions. This can reduce the mismatch between training states and deployed student behavior, but the teacher and its feedback remain fallible. Part II develops the post-training objectives and Part V the asynchronous rollout accounting; these model reports illustrate why the data and environment systems now matter as much as the named objective.

A causal comparison should hold the base checkpoint, rollout budget, verifier, environment family, and evaluation harness constant while changing one training ingredient. Then test unseen environment families, tool errors, changed task wording, and held-out verifiers. Separate gains from additional task diversity, longer reasoning, a different teacher, and a stronger optimizer. A same-base post-training claim, such as GLM-5.3's, does not by itself identify which of those ingredients caused the improvement.

### Test-time scaling is a family of systems

Recent reasoning systems spend variable inference compute rather than mapping every request to one fixed decode. [Test-Time Scaling in Reasoning LLMs, version 2](https://arxiv.org/abs/2608.04001v2), dated August 31, 2026, formalizes three regimes:

- **single-trajectory scaling:** extend one sequential reasoning path;
- **leaf-level scaling:** sample completed candidates and reduce them with voting, ranking, or verification;
- **prefix-level scaling:** branch or search over unfinished states.

These regimes have different state, parallelism, and failure behavior. Sequential scaling increases latency and retained context along one path. Leaf-level scaling is naturally parallel but can waste correlated samples and needs a trustworthy reducer. Prefix search introduces a frontier, partial-state ownership, pruning policy, and potentially large KV reuse opportunities.

A correct test-time compute budget includes input and output tokens, cached-prefix work, verifier or judge calls, tool execution, failed branches, communication, and wall-clock constraints. Accuracy at `N` samples is not comparable when one system uses a hidden candidate bank, a stronger verifier, or more expensive tools.

:::equation expected_value = quality_gain(B) - lambda_latency * latency(B) - lambda_cost * cost(B) - lambda_risk * risk(B) | Choose an inference budget B against product value, latency, cost, and failure exposure rather than accuracy alone.

### Design Exercises

1. Design an MoE experiment that distinguishes expert imbalance from network saturation.
2. What evidence would show that a reasoning RL gain comes from reward hacking?
3. Compare the scheduler state required for sequential scaling, best-of-N, and prefix search.
4. When should a distilled reasoning model replace its teacher in production?
5. Why is a million-token context limit insufficient to compare Qwen3.8, Kimi K3, and a hosted model?
6. How would you separate a post-training improvement from an architecture improvement?

### Worked Solutions

1. Slice expert load by layer, step, sequence class, and rank; replay a fixed routing trace with communication disabled or emulated; compare ideal and observed all-to-all time; and inspect whether the slowest expert or link predicts step time.
2. Use hidden verifiers, adversarially varied problem forms, manual trace audits, reward-component ablations, and pass rates on tasks where superficial shortcuts do not work. Check whether reward rises while independent correctness or readability falls.
3. Sequential scaling owns one growing KV history; best-of-N owns independent completed trajectories plus reducer state; prefix search owns a branching frontier, shared prefixes, pruning metadata, and branch-specific RNG. Their cancellation and memory-reclamation semantics differ.
4. Replace the teacher only when the student meets slice-level quality and safety floors, lowers end-to-end cost or latency under the real workload, and has a fallback for tasks where compression removed necessary capability.
5. Record native versus extended context, input modality, model revision, cache/recurrent layout, numerical format, and long-range quality. A supported length does not establish equal memory, latency, recall, or account access.
6. First hold the base checkpoint and evaluation harness fixed while changing post-training. Then compare architecture changes with training data, compute, effort, and harness differences explicitly recorded. GLM-5.3 and GLM-5.3-Flash illustrate why related names do not imply the same base.


## Serving, Attention, and Memory Hierarchies

LEAD: The recent inference frontier is defined less by one universal engine than by explicit management of asynchronous hardware, phase-specific work, KV placement, and overload.

### Attention pipelines on newer accelerators

The 2024 [FlashAttention-3](https://arxiv.org/abs/2407.08608) paper targets Hopper GPUs with warp specialization, asynchronous Tensor Memory Accelerator transfers, overlap between matrix multiplication and softmax, and an FP8 path with block quantization. The paper reports 1.5 to 2.0 times speedup over FlashAttention-2 on H100, up to 740 FP16 TFLOP/s, and nearly 1.2 PFLOP/s for FP8. These are attention-kernel results over the evaluated shapes, not whole-model or service speedups. It also reports lower numerical error than a baseline FP8 attention implementation.

The durable mechanism is a deeper software pipeline. Once hardware exposes specialized asynchronous movement and matrix units, a kernel must schedule producer and consumer warps, manage barriers and buffers, and interleave non-matrix work so tensor cores remain fed. The optimization surface shifts from tile reuse alone to dependency timing.

The production acceptance test still needs ragged and causal shapes, head dimensions, sequence tails, backward or decode variants, numerical drift by layer, graph capture, and end-to-end model throughput. Peak forward attention throughput does not establish application speedup.

### Blackwell attention and the limits of FP4

[FlashAttention-4](https://arxiv.org/abs/2603.05451), published in March 2026, addresses asymmetric hardware scaling: matrix throughput improves faster than exponential evaluation and shared-memory bandwidth. Its pipeline uses asynchronous matrix operations, tensor memory, and software techniques that reduce softmax overhead; its implementation uses CuTe-DSL. The paper reports up to 1.3 times the BF16 throughput of cuDNN 9.13 on B200 for its evaluated attention kernels. That comparator and hardware are essential; this is not a general service-level improvement over FlashAttention-3.

The September 3 preprint [Hardware-Aware FP4 FlashAttention-4, version 1](https://arxiv.org/abs/2609.04105v1), sharpens the distinction between a fast kernel and a sound training recipe. It reports an FP4 noncausal inference path and a separate causal training path, but also reports divergence in every tested distributed-training trajectory using MXFP4 probabilities and values. Its matched distributed runs retain FP8 for those operands. Treat this as provisional, boundary-specific evidence, not proof that attention can now use four bits everywhere.

The engineering implication is to keep a precision ledger for Q, K, V, scores, probabilities, accumulators, gradients, and scaling factors. Test long reductions, outliers, masks, and backward stability independently. A matrix unit's nominal throughput cannot compensate for conversion overhead, a softmax bottleneck, or failed optimization. Part IV supplies the ownership and synchronization details behind this pipeline.

### Read the 2026 developments through the earlier chapters

Part IV derives Blackwell ownership and the FlashAttention-4 pipeline. Part III compares feature-based and block-parallel speculative drafts, including EAGLE-3 and DFlash, while retaining the target-distribution verification contract. Part II develops group-relative RL, sequence-level ratios, and feedback-conditioned self-distillation; Part V explains why asynchronous training needs policy-freshness control. These changes act on different parts of the system, so each needs its own acceptance test.

| Change | Quantity it tries to improve | Evidence that could reject it |
| --- | --- | --- |
| Better speculative draft | Committed target tokens per unit time | Draft/verification overhead or state rollback defeats the gain |
| New attention pipeline | Kernel critical-path time | Unsupported shapes, numerical drift, or another layer dominates |
| Recurrent / compressed state | Persistent bytes and history processing | Required long-range information is lost |
| Better RL feedback | Useful learning signal per rollout | Verifier shortcuts or held-out regression |
| Asynchronous RL | Useful learner/rollout utilization | Stale-data bias, variance, or selection changes quality |

### Draft quality must repay its own cost

The 2026 speculative frontier extends beyond choosing a fixed draft length. [DFlare, version 2](https://arxiv.org/abs/2606.02091v2), conditions different draft layers on learned combinations of target-layer features to increase block-draft capacity. [CaDDTree, version 1](https://arxiv.org/abs/2606.01813v1), chooses both the candidate-tree structure and node budget against expected throughput, explicitly accounting for verification latency. These are research proposals with evaluated model/task boundaries, not universal engine defaults.

For an independent worked example, a candidate that commits six tokens in a 12-millisecond draft/verify round yields 500 committed tokens per second. A wider candidate that commits eight in 20 milliseconds yields 400. Acceptance length increased while throughput fell. At larger serving batches, extra draft or verification work may also displace other requests. Measure committed tokens and SLO goodput, including rollback and scheduling, rather than acceptance alone. Exact target-distribution preservation still requires the correct proposal and verification procedure; it cannot be inferred from a model's use of the word "speculative."

### KV-centric disaggregation

[Mooncake](https://arxiv.org/abs/2407.00079) reports a disaggregated serving architecture that separates prefill and decode clusters and treats KV state as a distributed object across GPU memory, CPU memory, and SSD. Its scheduler chooses placement and admission under latency objectives. The paper reports up to 525 percent higher throughput than its baseline in selected simulations and 75 percent more handled requests on a production workload. A 525 percent increase means 6.25 times the baseline throughput; it is not the production-workload result and is not a matched comparison against every current serving engine.

The general lesson is that disaggregation works only when state movement is first-class. A phase boundary needs:

- a stable identity for model, adapter, tokenizer, prompt prefix, and KV format;
- an ownership transfer protocol with completion and retry semantics;
- bandwidth and tail-latency budgets for the transfer path;
- admission that rejects work before expensive partial execution when overload is unavoidable;
- observability for cache hit, transfer, queue, recompute, and abandoned state.

Disaggregation can improve independent scaling and isolation while adding network dependence and a new distributed lifecycle. It wins when phase imbalance and placement flexibility repay the transfer and coordination cost.

The implementation frontier now exposes this lifecycle as composable subsystems. [Dynamo](https://github.com/ai-dynamo/dynamo) combines event-informed KV routing with prefill/decode worker pools and NIXL transfer. [LMCache](https://github.com/LMCache/LMCache) focuses on reusable KV across accelerator, host, disk, and remote tiers. [llm-d](https://github.com/llm-d/llm-d) separates approximate or precise prefix routing, cache indexing, offload, and disaggregated orchestration. Their feature matrices and compatibility notes change faster than the mechanism, so record a tested revision. A useful bake-off replays the same trace and measures stale-route rate, reusable-token fraction, transfer versus recompute choice, admission failures after prefill, failover, and SLO goodput—not just a warm-cache microbenchmark.

For the practical reading path, return to **Serving Engines and Cache Backends in Practice** in Part III. It distinguishes an engine iteration from a kernel call, a cache transfer, and a fleet-routing decision. The purpose of this research snapshot is to identify which of those boundaries a result changes, not to select a stack by counting project names.

Within each pool, long-context sharding is also phase-specific. Current vLLM documentation distinguishes prefill context parallelism, which partitions prompt queries and either gathers or circulates K/V, from decode context parallelism, which shards historical KV tokens and merges partial attention. Disaggregation adds another axis: the prefill and decode pools may select different TP, PP, PCP, or DCP plans, but a heterogeneous boundary must reshard compatible state during handoff. Part V gives the full ownership ledger.

### A cache is now part of the architecture

The September-inspected [DeepSeek-V4.1-Flash card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) describes a causal encoder-decoder whose decoder global KV comes from final encoder states. Its CSA2 scheme shares KV and indexing state across layers; a sliding-window replay mechanism reconstructs recent local state. The card reports 890 bytes per token for global KV using FP4 main-cache storage. This is a component-level figure, not the entire request footprint. It also explicitly names Engram and DSpark, so those mechanisms can be attributed to this checkpoint; the earlier V4 discussion alone cannot establish that attribution.

Using the reported global-KV figure, one million tokens occupy about 890 million bytes, or 0.83 GiB, for that component. Do not count that as the memory required to serve the model: weights, local/reconstructed state, vision processing, workspace, and concurrent requests remain. Similarly, a reusable global cache is not necessarily a zero-compute prefix hit if local state must be replayed.

A cache-hit result should therefore report at least bytes reused, bytes transferred, state reconstructed, and resumed computation. A disaggregated handoff must carry the architecture's state schema, not only a list of ordinary per-layer K/V tensors. Mixing model revisions, quantization scales, compression layouts, or recurrent snapshots can produce plausible but incorrect continuations. Validate cold execution against cache reuse, offload/reload, interruption, and branch rollback at the intended numerical tolerance.

### Worked admission ledger: count the complete worker

Consider a hypothetical inference worker with 80 GiB of usable device memory. These are planning numbers, not measurements or the specification of a named checkpoint. All entries below are local to this worker after its chosen sharding; weights include their quantization metadata. Shared fixed allocations appear once, while request allocations scale with concurrent requests.

| Allocation | GiB | Scope and assumption |
| --- | ---: | --- |
| Resident weights and scales | 40 | Shared, already placed on this worker |
| Runtime and communication buffers | 6 | Shared worst-case allocation |
| Captured graphs and fixed scratch | 4 | Shared; excludes scratch below |
| Admission reserve | 6 | Held back for fragmentation and uncertainty |
| Attention KV | 4 | Per request, including reserved output growth |
| Recurrent and local state | 1 | Per request, hypothetical hybrid layers |
| Cache metadata and alignment | 0.25 | Per request, separate from raw KV |
| Replay, draft and transfer scratch | 2.25 | Per request at simultaneous peak |

The raw KV row follows an uncompressed example with 32 attention layers, eight KV heads, head dimension 128, two-byte elements, and a reserved total length of 32,768 tokens: `2 * 32 * 8 * 128 * 32768 * 2 = 4,294,967,296 bytes = 4 GiB`. The leading two counts K and V. It is unrelated to the 890-byte global-cache component reported for the different architecture above.

Fixed allocations plus reserve total 56 GiB. Each request needs at most 7.5 GiB under these assumptions. Three concurrent requests need `56 + 3*7.5 = 78.5 GiB`; four need 86 GiB and must not be admitted. Three leave 1.5 GiB beyond the explicit six-GiB reserve. Counting only raw KV would incorrectly suggest room for six requests. If a request exceeds its reserved output length, admission must obtain more capacity before growth or enforce the declared stop policy.

This is a memory ceiling, not a throughput promise. If every newly admitted request imports 4 GiB and measured usable transfer bandwidth is 20 GiB/s, five such imports per second already consume the entire link budget, before competing traffic. Transfer queues and latency may force a smaller admission rate. In a sharded deployment, repeat the ledger on every worker: free memory on one rank cannot satisfy an allocation on another, and replicated buffers must not be divided by the shard count.

### Runnable load experiment: admission is not useful throughput

Run `python -m examples.serving_load` from the source repository. This deterministic CPU simulation sends 60 requests at five arrivals per second through one FIFO transfer link and one FIFO compute server. Transfers can overlap computation, but compute executes whole requests in admission order; there is no continuous batching, preemption, GPU contention model, or learned quality model. The purpose is to expose queueing and accounting errors, not predict a serving engine's performance.

Each ordinary request has 8,192 prompt tokens and reserves 128 output tokens. The model keeps the ledger's 56-GiB fixed allocation and 3.5-GiB non-KV request allocation, plus `total_tokens/8192` GiB of KV. Resident slots and memory are held from admission through completion, including time waiting for import or compute. Arrivals that exceed either limit are rejected immediately. When both limits would fail, this implementation records a memory rejection first; the rejection counters are mutually exclusive bookkeeping categories. A completion at exactly an arrival time releases its reservation first. These choices are part of the experiment, not universal scheduler rules.

The invented service rates are 32,768 uncached prompt tokens/s, 131,072 cached tokens/s for local replay, and 512 output tokens/s. Reused prompt state arrives from an external tier: imported GiB divided by link GiB/s determines transfer service, and cache reuse does not reduce resident KV allocation. A request passes only if first-token latency is at most one second and completion latency at most two seconds, both measured from arrival. Goodput counts these passing completions divided by elapsed time from the first arrival through the final drain or last arrival, whichever is later. Rejections remain in the 60-request denominator for the pass fraction; accepted-only latency percentiles would hide them.

| Scenario | Admitted / offered | Slot / memory rejects | Pass both SLOs | Goodput, requests/s |
| --- | ---: | ---: | ---: | ---: |
| One resident slot | 20 / 60 | 40 / 0 | 20 | 1.681 |
| Three resident slots | 26 / 60 | 34 / 0 | 3 | 0.231 |
| Six configured slots | 28 / 60 | 0 / 32 | 3 | 0.214 |
| Three slots, 32,768-token prompts | 12 / 60 | 0 / 48 | 0 | 0.000 |
| Three slots, 75% reuse, 20 GiB/s link | 35 / 60 | 25 / 0 | 35 | 2.774 |
| Three slots, 75% reuse, 1 GiB/s link | 18 / 60 | 42 / 0 | 1 | 0.072 |

More slots admit more work while accumulating a compute queue; this server has no batching speedup to repay it. Six configured slots cannot all be occupied under the memory ledger: ordinary requests need about 4.516 GiB each, so only five fit. Longer prompts hit both memory and the first-token floor. Reuse improves goodput on the fast link, but on the slow link its transfer queue makes it worse than cold computation. The example deliberately imports reused state; a production scheduler should compare that choice with recomputation before dispatch.

**Worked check:** should the one-slot policy ship because it beats three slots? No: it rejects two thirds of offered requests. All six configurations need an explicit acceptable rejection rate as well as latency and quality gates. The script teaches why capacity, completed throughput, SLO goodput and offered-workload success are different quantities.

For a real experiment, replace the constant rates with measured shape- and batch-dependent service distributions; include bursty arrivals, cache misses and invalidation, transfer overlap, cancellation, and retries. Record engine revision, hardware, parallelism, warmup and cold-start policy. Replay the same workload against cold recomputation and cache reuse, and validate outputs separately. The simulation and its tests establish event bookkeeping only; they establish no GPU speedup or production capacity.

### Long-context memory is becoming hierarchical

Recent work explores two complementary ways to reduce long-context pressure. [RocketKV, version 1](https://arxiv.org/abs/2502.14051v1), combines coarse eviction with fine-grained sparse attention and reports up to 3 times end-to-end decode speedup and up to 31 percent peak-memory reduction on H100 against a full-KV-cache baseline, with negligible loss on its evaluated tasks. These numbers belong to the February 2025 version; later revisions report a different hardware evaluation. [SparseServe](https://arxiv.org/abs/2509.24626v1) places unselected KV state in host memory, controls batch size from the active working set, and segments prefill by layer; it reports up to a 9.26-fold reduction in mean time to first token and up to 3.14 times higher generation throughput than its evaluated baselines. These maxima need not occur in the same configuration.

These are paper-reported maxima, not portable constants. They do establish a broader systems pattern: when attention becomes sparse, the bottleneck can move from HBM bandwidth to HBM capacity, irregular selection, or host-device movement. A valid comparison must include retrieval quality, task accuracy, selection overhead, cache thrashing, tail latency, and worst-case dense fallbacks.

:::callout pitfall|Compression is not free capacity
Eviction, quantization, sparse selection, and offload change the model's effective context or add selection and transfer work. Capacity is useful only when quality and latency remain inside the workload contract.
:::

### Design Exercises

1. Build an end-to-end benchmark that could reject a FlashAttention-4 deployment despite a faster kernel.
2. Derive the break-even condition for moving KV between prefill and decode workers.
3. How would admission control change when host memory is a KV backing tier?
4. Compare KV eviction, sparse attention, quantization, and offload as long-context policies.
5. A cache reports 890 bytes per token. What must be established before using it for fleet admission?
6. Why might a speculative tree with a longer accepted prefix reduce throughput?

### Worked Solutions

1. Include production sequence and head shapes, padding, causal masks, graph capture, competing kernels, full-layer time, accuracy drift, memory, compile cost, and tail latency. Reject it if layout conversion or unsupported shapes erase the kernel gain.
2. Disaggregation wins when saved queueing and improved phase utilization exceed transfer, synchronization, retry, and additional network-tail cost while KV fits within the destination's admission reserve.
3. Admit from predicted active working set and transfer bandwidth, not nominal host capacity. Reserve headroom, bound concurrent promotions, detect thrashing, and reject or degrade before latency collapses.
4. Eviction removes state and risks quality; sparse attention retains or tiers state but pays selection cost; quantization reduces bytes with numerical risk; offload preserves values but adds transfer latency. Hybrids should be evaluated against a full-attention quality and latency baseline.
5. Establish exactly which state the figure includes, then add weights, local or recurrent state, replay workspace, scales, temporary buffers, concurrency, and reserve. Measure cache-hit reconstruction work and transfer tails rather than treating a component footprint as total service memory.
6. More candidate nodes can increase verification, draft, and rollback costs faster than useful committed tokens. Compare committed tokens per complete round and fleet goodput under concurrency, not accepted length alone.


## Multimodal Representations: Images, Video, and Speech

LEAD: A multimodal model must turn signals with space and time into a representation that a language system can learn from and act on. The encoder, alignment objective, sequence layout, and output contract are as important as the language backbone.

### From pixels to language-model inputs

:::diagram multimodal_path|A common conceptual arrangement maps visual, audio, and text representations into language-model context. Position and time metadata preserve meaning across the interfaces; the exact encoder, projection, and fusion arrangement is model-specific.

Begin with an image of height H and width W. A simple patch encoder divides it into patches of side p, producing approximately `(H/p) × (W/p)` tokens when dimensions divide evenly. Each flattened patch is projected into a vector; a vision transformer mixes these vectors with position information. A projector then maps visual features to the language model's hidden width. Insert them into a declared multimodal sequence or expose them through cross-attention.

In sequence insertion, visual embeddings occupy positions alongside text and consume backbone context/attention work. In cross-attention, text queries read a separate visual representation; visual-token count still changes cross-attention and encoder cost, but it is not necessarily identical to text KV growth. Early joint training can learn richer integration across modalities; it does not remove modality-specific preprocessing or alignment requirements. [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) is a primary example of connecting a vision encoder and language model for instruction-following behavior.

An original geometry example: 448 × 448 pixels with 16 × 16 patches gives 784 patch tokens before any pooling or special tokens. Doubling both image dimensions gives 3,136, four times as many. A merge that combines each 2 × 2 patch group reduces those counts to 196 and 784 respectively, while discarding some spatial detail. These are illustrative dimensions, not a particular model's processor configuration. Always calculate tokens **after** the actual resize, crop, patch, and merge policy.

Small text in a document creates a real tradeoff: aggressive downsampling can erase a decimal point before the language model sees anything. Increasing language-model size cannot recover pixels that were removed. Evaluate encoder resolution, OCR/visual parsing, and language interpretation separately.

### Alignment objectives teach different capabilities

Contrastive alignment teaches matching. In a batch of N image/text pairs, compute an N × N matrix of scaled similarities. For each image, use cross-entropy to favor its paired text over other batch texts; commonly apply the reverse direction too. [CLIP](https://arxiv.org/abs/2103.00020) establishes this form of image-language representation learning. A high matching score is not itself a caption generator or a proof of visual reasoning.

For an original two-pair example, logits `[2,0]` give the correct first text probability `exp(2)/(exp(2)+1) ≈ 0.881`, with loss about 0.127. If two captions are both valid descriptions of the image, treating one as a negative creates a data/objective mismatch. Pair quality, duplicated images, and near-identical captions affect the training signal.

Generative alignment instead predicts text conditioned on visual features—for example a caption or answer. Instruction tuning adds varied questions and response formats. Training a projector with a frozen backbone differs from updating the vision encoder and language model together: their memory, compute, forgetting risk, and data needs differ. Evaluate visual grounding using image changes that should alter the answer, not only questions that a text-only prior can answer.

### Video requires time, not only more images

A straightforward video pipeline samples frames, encodes each, and attaches timestamps before combining them with text or audio. Frame order alone is insufficient if sampling rates vary. The model must distinguish “frame 20” from “twenty seconds elapsed.” Temporal pooling can lower cost while losing short actions; adaptive sampling can retain events while adding another learned or heuristic decision.

At two frames per second, a one-minute clip contains 120 sampled frames. With 196 visual tokens per frame, that is 23,520 tokens before text, audio, and temporal compression. A 100-millisecond event can occur entirely between the sampled frames. No downstream reasoning method can guarantee recovery of an unobserved event. Evaluate temporal localization, ordering, and evidence coverage separately from a general video summary.

Grounded outputs need a coordinate contract. If a detector reports normalized `(x,y)` coordinates, first map them to the pixel frame defined by its output schema, then invert the recorded preprocessing transforms in reverse order to locate the point in the original image. For example, remove model-input padding before undoing resize, then add the crop's original-image offset. Rotation requires its corresponding inverse transform. A correct label with the wrong coordinate frame can trigger the wrong UI action. For video, preserve the source time base and dropped-frame policy similarly.

### Speech has acoustic and linguistic time scales

Audio may enter as waveform-derived features such as a log-mel spectrogram, then pass through an audio encoder. For generation, a neural codec can compress audio into discrete code sequences that a model predicts before a decoder reconstructs a waveform. [EnCodec](https://arxiv.org/abs/2210.13438) is a primary neural-compression reference. Codec tokens are not words: several codebooks may describe different residual details of one audio frame.

Imagine a codec producing 50 frames per second with eight codebooks. A naive flattening has 400 code symbols per second; a model that predicts codebooks partly in parallel has a different sequential-step count. Neither count should be called text tokens per second. Separate semantic content, acoustic detail, frame rate, codec bitrate, model steps, and audible latency.

A cascaded assistant uses speech recognition, a text model, then text-to-speech. It offers clear component boundaries but can lose prosody and compound errors. A speech-text model can integrate these representations more directly. [Moshi](https://arxiv.org/abs/2410.00037) studies real-time speech-text dialogue; the [Qwen3-Omni report](https://arxiv.org/abs/2509.17765) describes a Thinker/Talker organization for multimodal understanding and speech generation. These systems motivate separating language decisions from waveform production while coordinating their streams.

The April 2026 [Qwen3.5-Omni report](https://arxiv.org/abs/2604.15804) describes hybrid-attention MoE components and adaptive text/speech alignment, addressing differing production rates between text and speech tokens. The underlying problem is general: if text outruns audio, buffering grows; if audio commits too early, later reasoning cannot retract spoken words cleanly. The book does not adopt the report's broad benchmark leadership claims as a universal ranking.

### The September frontier has multiple interaction contracts

The [Gemini 3.8 Audio card](https://deepmind.google/models/model-cards/gemini-3-8-audio/), listed as updated September 23 in the publisher's index, distinguishes Live and Live Extended Thinking from Flash TTS and Flash-Lite TTS. The Live variants accept audio, images, video, and text and return audio/text; the TTS variants take text and return audio. These are not interchangeable endpoints. The card also acknowledges hallucinations and occasional slowness or timeouts. A stronger reasoning mode must be evaluated against the interaction's silence and interruption budgets.

[GPT-Live 1](https://developers.openai.com/api/docs/models/gpt-live-1) documents full-duplex speech and delegation to a backend agent. Its own model card lists audio/text, not image/video input. The [delegation guide](https://developers.openai.com/api/docs/guides/live-delegation) separates managed Responses delegation from application-owned client delegation: the latter owns the context and decides which backend results to return. Thus a multimodal backend does not make the live voice model itself a vision model, and finishing a backend request does not establish that its answer was spoken.

[Gemini Omni Flash's August card](https://deepmind.google/models/model-cards/gemini-omni-flash/) describes a different boundary: generating and editing video with audio from multimodal input. It notes remaining problems with edit consistency, complex motion, and accurate text rendering. Understanding a video, generating one, and maintaining a real-time conversation require different evaluations. Treat media-generation quality as an adjacent system concern; it cannot be reduced to text perplexity.

For a live assistant, model the conversation and the delegated task as two concurrent state machines. A correction can invalidate a pending task while audio continues. Attach a task generation number to each delegated result, check it before committing an action or speaking an answer, and record the last audio actually played. This is an application design principle, not a claim that any cited service supplies transactional cancellation automatically.

### Worked concurrent interaction: correction during a tool call

Suppose a visual assistant is describing a diagram while looking up a cited specification. The following original timeline uses milliseconds from session start. Video timestamps identify evidence capture; audio times identify playback, not generation completion. The tool is read-only in this example.

| Time | Concurrent activity | Host state and decision |
| --- | --- | --- |
| 0 | User asks about the left connector; video frame V0 is captured | Start task generation 41 and bind evidence V0 |
| 100 | Specification lookup starts while the assistant begins a short acknowledgment | Tool call carries generation 41; audio queue belongs to that response |
| 200 | New frame V200 shows the user pointing at the right connector | Store capture time and provenance; do not silently replace V0 in the pending task |
| 600 | User interrupts: "I meant the right connector" | Advance to generation 42; request old-tool cancellation and flush unplayed old audio |
| 650 | Playback stop is acknowledged; replacement lookup starts | Record old audio actually heard through 650; bind generation 42 to V200 only if still relevant |
| 900 | Old lookup returns despite cancellation | Reject generation-41 result for speaking or new actions |
| 1200 | Replacement lookup completes | Validate generation 42 and evidence freshness, then queue its answer |
| 1400 | New answer begins playing | Record delivery now; queued audio at 1200 was not yet heard |

If a newer frame contradicts V200, refresh evidence or ask for clarification instead of pairing the correction with stale visual state. A playback device may continue briefly after a stop request; the 600-650 ms interval belongs in the interruption metric. Checks at enqueue time alone are insufficient: invalidate queued old-generation audio and check generation again at delivery.

For a write tool, generation validation must be coordinated with effect commitment through an authoritative gate, version precondition or fencing mechanism. A check followed by an unprotected network write leaves a race with the correction. If the external service already committed generation 41, mark that effect as committed and reconcile it; canceling speech cannot undo it. An endpoint without conditional execution requires an explicit ambiguous-outcome policy. This timeline describes an application contract, not a guarantee supplied by a particular live-model API.

### Budget and evaluate the entire interaction

Time to first audio includes input buffering, encoder work, decision latency, codec generation, waveform decoding, and playback buffering. An illustrative budget assigns 80 milliseconds to input buffering, 60 to encoding, 140 to the first response decision, 40 to codec generation plus waveform decoding, and 80 to playback buffering. These five sequential stages total 400 milliseconds; overlap may reduce the total, while queueing can enlarge it. A first text token is not a first audible response.

For interruption, stop generation and playback coherently, release queued state, and record what the user actually heard. A tool action already committed cannot be canceled by muting its spoken confirmation. Define turn-taking, barge-in, maximum silence, and degradation on packet loss before optimizing throughput.

Evaluate recognition by language, accent, noise, and domain vocabulary; grounding by visible/audible evidence; generated speech by intelligibility and timing; interaction by successful tasks and appropriate interruption. Train/test splits should avoid speaker, video, or near-duplicate scene leakage. Obtain appropriate rights and consent for source material and voices; data availability is not equivalent to permission to clone a speaker.

### Exercises and worked answers

1. **Why does doubling width and height quadruple patch count?** Patches cover area; both axes double. Attention and encoder costs may then grow faster than the token count.
2. **Can a larger model fix text destroyed by downsampling?** Not reliably. Preserve the signal or use a better crop/resolution policy before changing the backbone.
3. **Why is a contrastive model not automatically a generative assistant?** Matching representations and generating conditional responses use different objectives and output mechanisms.
4. **Why can a correct video summary miss an important action?** Sampling or compression can omit a brief event; evaluate coverage and temporal localization, not only summary fluency.
5. **What counts as completed speech output?** Audio actually delivered under the interaction contract, with synchronized cancellation and tool state—not merely predicted text or queued codec tokens.
6. **Can a delegated task finish after the spoken request has changed?** Yes. Track task generations and reject stale results before speaking or acting; cancellation of audio and cancellation of external work are separate operations.
7. **Does a multimodal backend make GPT-Live a native video model?** No. Record the modalities supported by each component and the context actually transmitted across delegation.

## Diffusion and Block-Parallel Language Generation

LEAD: Autoregressive decoding commits a sequence from left to right. Discrete diffusion-style generation instead learns to reconstruct corrupted token sequences and can refine multiple positions in one model call. Parallel positions do not automatically imply lower total computation or identical semantics.

### Train a masked denoiser

Take a clean sequence and randomly replace some tokens with a mask symbol. The network sees the corrupted sequence and predicts original tokens at masked positions. Vary the corruption level during training so it learns both nearly complete and heavily masked contexts. Unlike a causal decoder, a denoiser can use visible context on both sides of a missing position.

For an illustrative objective with independent mask probability t sampled uniformly in `(0,1]`, weight the summed masked-token cross-entropy by `1/t`. The weighting compensates for how often positions are masked; implementations need their precise noise schedule, conditioning, and loss normalization. This is not obtained by adding a mask token to a pretrained causal model and changing its decoder. [LLaDA](https://arxiv.org/abs/2502.09992) studies large language models trained through a masked-diffusion formulation.

### Generate through a schedule of commitments

Start with a requested output length of masks. Predict distributions for currently masked positions, sample or choose candidate tokens, and reveal a subset according to a schedule—possibly using confidence. Repeat until complete. Some methods remask or revise positions. Output length and end-of-sequence handling are part of the method, not details that left-to-right decoding supplies automatically.

Example status: Explanatory pseudocode; model training, sampling policy, and length selection are intentionally abstract.

```text
tokens = MASK repeated output_length times
for step in denoising_schedule:
    predictions = model(prompt, tokens, noise_level=step.level)
    positions = choose_positions_to_commit(predictions, step)
    tokens[positions] = sample(predictions[positions])
assert no_required_position_is_masked(tokens)
```

An eight-position sequence revealed two positions per round requires four denoiser calls. Eight autoregressive output positions require eight sequential decode steps, but each denoiser call can process all eight output positions, while cached autoregressive decode processes only the new position against prior state. Comparing four to eight without FLOPs, memory traffic, batch size, and quality is misleading. Independent guesses can also disagree: choosing a subject and verb simultaneously may commit an incompatible pair.

### Block diffusion and cache validity

:::diagram diffusion_blocks|Three snapshots show an initially masked block and two reveal steps. The causal prefix stays fixed while the current block changes. Question marks denote masked positions. Cache validity depends on which representations can change under the method's attention mask.

[Block Diffusion](https://arxiv.org/abs/2503.09573) interpolates between autoregressive blocks and within-block diffusion. Earlier blocks can become fixed context while positions in the current block are refined together. This introduces a tunable tradeoff among block size, denoising steps, parallelism, and quality.

A causal prefix cache is valid because later output cannot alter its representations. A bidirectional mutable block lacks that property: changing one token can change the other positions' hidden states. Do not reuse its KV as if it were an unchanged autoregressive prefix. Cache reuse must follow the method's attention mask and update dependencies; an approximation needs an accuracy test of its own.

A diffusion **draft** inside speculative decoding is a different system from a diffusion **target** model. In the former, an autoregressive target can still define the output distribution if the proposal and verification procedure is valid; Part III discusses DFlash in that role. In the latter, the denoising model and generation schedule define the output behavior. There is no general promise of equality to an unrelated autoregressive model.

### From a denoising idea to a serving implementation

[DiffusionGemma's July 31 technical report](https://arxiv.org/abs/2608.00146v1) describes an experimental Gemma 4 adaptation with 256-token blocks, supervised denoising, reinforcement learning, and sampler distillation. It reports roughly twenty tokens per forward pass and about 1,500 output tokens per second on one H100 across its evaluation suite. These are source-reported results, not measurements reproduced for this book.

Unlike the masked example above, DiffusionGemma starts with random tokens, re-noises uncertain positions, and uses previous predicted distributions as self-conditioning. Entropy helps control refinement. Its [method and limitations](https://arxiv.org/html/2608.00146v1) also report lower absolute capability than the autoregressive initialization, occasional repetition, and a throughput crossover favoring autoregressive execution at higher concurrency. A crossover measured in that setup is not a universal user-count threshold.

The [vLLM implementation account, June 10](https://vllm-project.github.io/2026/06/10/diffusion-gemma), exposes an important serving detail. Requests can occupy different denoising stages within one batch. A mutable canvas uses bidirectional attention, while accepting a completed canvas requires a causal pass to populate the reusable prefix cache. The implementation therefore uses per-request attention causality rather than one global causal flag for the entire batch. The model's architecture and scheduler must agree about which state is final.

This changes the benchmark contract. Record time to the first committed readable block, total completion time, useful output tokens, denoiser passes, acceptance passes, and memory at the actual concurrent load. Count canceled or revised work. A low-latency single request and a high-throughput batch can prefer different block sizes or even different model families. Include an autoregressive baseline with a tuned speculative path, not only naive token-by-token execution.

A useful toy calculation is a 32-token block refined in four 10-millisecond passes followed by a 4-millisecond acceptance pass. Its model-side rate is about 727 tokens per second, but the first committed block arrives no earlier than 44 milliseconds after the required prefix is ready. Reporting only 800 tokens per second from the four refinement passes omits necessary work. These numbers are illustrative, not measurements of DiffusionGemma.

### Compare generation methods by their commit boundary

| Method | Mutable unit and commitment | Reusable state and comparison boundary |
| --- | --- | --- |
| Autoregressive target | One next token becomes final under the decoding policy | Causal prefix KV; measure useful committed tokens and full serving latency |
| Exact speculative autoregression | Draft block or tree, then target verification and correction | Discard rejected branch state; target-distribution equivalence requires the correct sampling algorithm |
| Full-sequence masked diffusion | Multiple masked positions refined under a schedule | Mutable bidirectional state; no general promise of an early stable readable prefix |
| Block diffusion | Refine one block against fixed earlier blocks | Prefix reuse follows the actual attention mask; mutable-block KV is not automatically reusable |
| DiffusionGemma | Random-token canvas with self-conditioning, then acceptance | Separate causal encode-and-append pass populates prefix cache; count it alongside denoising |
| Sequence editing | Insertions and deletions change token positions and length | Revalidate positional/cache dependencies and termination; support depends on the actual engine |

The rows summarize the cited mechanisms in this chapter, not a speed ranking. In particular, diffusion used only as a speculative draft belongs under the autoregressive target's verification contract. A native diffusion target defines its own output behavior. Benchmark the same workload, quality floor, concurrency and output policy, and include canceled, revised and rejected work. A smaller number of model calls can still mean more arithmetic, memory traffic or time before the first useful commitment.

### Sequence editing and commercial diffusion

The [LLaDA2.2-flash card](https://huggingface.co/inclusionAI/LLaDA2.2-flash), inspected September 23, introduces `DELETE` and `INSERT` controls, block-level MoE routing, and agentic reinforcement learning. This extends refinement beyond replacing tokens at fixed positions: deletion and insertion change sequence structure. Position indices, termination, visible streaming, and cache invalidation must follow the editing method. The card provides a Transformers example but says SGLang deployment support is coming soon; model availability does not establish readiness of every suggested engine.

[Mercury 2.5](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5), announced September 8, supplies a commercial contrast: Inception documents an available diffusion service with configurable reasoning, parallel tool calls, and schema-aligned JSON. Its production and performance claims are vendor evidence, not a matched benchmark against the open implementations above. Test structured-output validity, correct tool arguments, committed-output latency, and concurrent throughput under the same application contract before selecting a generation family.

### Exercises and worked answers

1. **Does half as many model calls mean twice the speed?** No. Each call may process more positions and move more state; measure end-to-end time at matched quality and batch size.
2. **Why train across corruption levels?** Inference encounters different amounts of known context as generation proceeds; training only at one mask rate can mismatch that sequence.
3. **When may an earlier block be cached?** When its representations cannot depend on mutable later positions under the actual attention mask and model computation.
4. **Is masked denoising exact speculative sampling?** No. It can supply proposals, but a target-preserving verifier needs the correct proposal semantics and acceptance/correction logic.
5. **Why does an accepted diffusion block need a separate cache decision?** Mutable bidirectional representations are not automatically valid as causal prefix KV. Follow the method's acceptance/cache-population pass and include its work in latency.
6. **What could reject a fast single-request diffusion deployment?** Worse task quality, slow first committed output, poor batching under mixed refinement stages, excessive workspace, or expensive cancellation can all violate the service contract.
7. **Do all discrete diffusion models repeatedly reveal masked tokens?** No. Masked denoising, random-token corruption with self-conditioning, and length-changing sequence editing have different transition rules. Use the target method's sampler, termination, and cache dependencies.

## Multimodal and Tool-Using Systems

LEAD: Multimodal and agentic models turn context construction into an active systems boundary: inputs have geometry and time, outputs can invoke capabilities, and untrusted observations can influence irreversible actions.

### Native-resolution vision and long video

The [Qwen2.5-VL technical report](https://arxiv.org/abs/2502.13923) connects the representation mechanisms above to document and interface tasks through dynamic resolution, window attention, temporal encoding, and geometric outputs. Its benchmark results are evidence for the evaluated tasks and model sizes; deployment still needs a contract for the actions those outputs can trigger.

For engineers, variable visual resolution makes token count a function of input geometry and preprocessing. Capacity planning must model image area, frame sampling, patching, video duration, visual token compression, text length, and cross-modal attention. A request limit expressed only in text tokens is incomplete.

Document and interface tasks also require geometric output contracts. Bounding boxes, points, table cells, time spans, and actions need coordinate systems, normalization rules, confidence, and provenance. Resize, crop, rotation, frame sampling, and OCR versions must be logged because they can change the answer without changing the model checkpoint.

### An agent is a policy around capabilities

A production agent contains more than a language model. It has a tool registry, schemas, credentials, planner state, observations, retries, budgets, memory, approval boundaries, and an audit log. Reliability depends on the whole trace:

1. interpret the user objective and authority;
2. choose a tool under a capability policy;
3. validate typed arguments;
4. execute with least privilege and idempotency controls;
5. label returned data with trust and provenance;
6. decide whether another action is justified;
7. verify the result against the user's goal.

Evaluate task success together with invalid calls, unnecessary calls, side effects, recovery after tool errors, budget use, latency, and security-policy violations. A benchmark that scores only the final answer can reward unsafe hidden trajectories.

### Long-running agents need durable, testable state

Current model releases put more weight on completing extended workflows, but a longer context window is not a recovery protocol. Persist the objective, current user constraints, tool-call IDs, authoritative resource versions, execution receipts, pending work, and budgets outside the model. After a restart, reconcile actual external state before replaying a write. A model's recollection that an action succeeded is weaker evidence than the service's receipt.

Preserved reasoning history adds another compatibility boundary. The [Kimi K3 usage card](https://huggingface.co/moonshotai/Kimi-K3) requires complete returned assistant messages, including reasoning content and tool calls, to be passed back for multi-turn use. A generic adapter that strips everything except visible text can silently change the supported interaction. Follow the model's contract while applying the application's retention and access policies; raw reasoning history is not a substitute for a compact, authoritative task ledger.

Persistent memory must retain provenance and user scope. A fact extracted from an untrusted page cannot become a higher-priority instruction merely because it was saved in a memory store or summarized. Test revocation, stale facts, contradictory updates, deleted records, and a malicious observation that attempts to survive across sessions. A successful memory retrieval test says little about whether the retrieved material may authorize a side effect.

### Worked recovery: the effect committed before the task checkpoint

Part VI's `examples.durable_effects` stores one local effect with its receipt. The companion `python -m examples.task_recovery` adds a separate persistent task ledger. It records task identity, tenant, operation key, payload, generation, schema version, attempt limit, attempts spent, status and receipt. Its scripted task saves one draft; it does not simulate general model planning.

The failure sequence is deliberately split across two databases. First commit the task's pending intent. Before dispatch, durably charge one effect attempt. Then save the draft and receipt in the effect database. Simulate process failure before marking the task done. On reopening, the task is still pending with one attempt spent, but the effect receipt exists. Recovery checks current host-supplied authority and generation, reads that matching receipt, and marks the task done without a second dispatch. The demo prints `status=done attempts=1 drafts=1` even with a one-attempt limit.

If there is no receipt and no remaining attempt, return `needs_reconciliation`; a crash between charging and dispatch can consume an attempt without creating an effect. This conservative choice prevents restart from resetting the budget. A receipt query is a separate operation: the fixture bounds effect attempts only, while production also needs persistent deadlines and budgets for status reads, model calls and total spend. When another effect attempt is allowed, reuse the original operation key and exact payload.

A changed generation returns `stale` before dispatch; revoked authorization denies receipt replay; changed task parameters cannot overwrite an existing intent. Unsupported ledger schema versions stop recovery. The fixture assumes one active runner per task. A production scheduler needs leases or fencing and a defined authority/commit ordering when corrections race a dispatch. The separate SQLite databases model the checkpoint gap, but the effect itself remains local and transactional; remote services require their own durable idempotency and reconciliation contract.

**Worked check:** a task spent its final attempt, the draft committed, and the reply was lost. Must recovery fail or reset the counter? Neither. If current authority permits it, reconcile the existing receipt and finish with the original counter. If authority was revoked, do not expose the receipt through the denied task.

### Evaluate the model together with its harness

[Terminal-Bench 4.0](https://www.tbench.ai/news/terminal-bench-4-0), released in August 2026, recalibrates time/CPU/memory resources, fixes tasks, and removes saturated tasks. Its authors describe an eight-hour task timeout and reduced infrastructure-induced measurement noise. Consequently, a score on 4.0 is not a direct continuation of a 2.1 or 3.0 score, and that evaluation budget is not a product latency guarantee. The published benchmark tracks an agent as well as a model.

Record the task dataset revision, harness, system prompt, tool versions, sandbox resources, network access, retries, model effort, and timeout. Hold those fixed when asking whether a model changed; deliberately vary them when asking which complete system works best. Use task-level paired outcomes and uncertainty intervals. Count invalid or unauthorized actions separately from ordinary task failure, and keep verifiers outside the agent's writable environment. A final correct artifact should not erase an unsafe intermediate action.

An independent reliability example illustrates the horizon problem. If twenty required stages each succeed with probability 0.98 and failures are independent with no recovery, complete-task success is about 0.668. Real errors are often correlated, so this is an explanatory calculation rather than a forecast. Checkpointing, verification, and targeted recovery change the result; simply selecting a model with a slightly higher short-task score does not measure those mechanisms.

### Prompt injection is a control-flow problem

[AgentDojo](https://arxiv.org/abs/2406.13352) introduced 97 realistic tasks and 629 security test cases for agents operating over untrusted tool data. These counts describe the published benchmark, not every future repository version. Its evaluation found that agents could fail ordinary tasks even without attacks and that then-current attacks and defenses had uneven coverage. [ChatInject](https://arxiv.org/abs/2509.22830), published at ICLR 2026, reports that chat-template and multi-turn payloads substantially increased attack success over traditional prompt injection on AgentDojo and InjecAgent, including against prompt-based defenses.

The engineering conclusion is stronger than "improve the system prompt." Natural-language instructions and retrieved data share a representation, so the model alone should not be the final authorization boundary. Use conventional controls:

- capability-scoped credentials and allowlisted tool identities;
- typed schemas and deterministic validation;
- separation of untrusted content from control metadata;
- information-flow labels for confidentiality and integrity;
- explicit authorization rules and confirmation where the action exceeds the user's existing grant or the product's risk threshold;
- rate, cost, and recursion limits;
- immutable audit logs and replayable security tests.

[Information-flow-control research](https://arxiv.org/abs/2505.23643) explores planners that track integrity and confidentiality labels and enforce policies independently of the model's textual judgment. The broad lesson is durable even while implementations evolve: authorization belongs in code that can deny an action deterministically.

Deterministic enforcement is only as complete as the boundary it mediates. Sensitive tools must pass through protected enforcement code; labels must survive summarization, saved memory, and delegation; and side channels and tool side effects must be included in the threat model. Enforcing a specified policy does not prove that the policy captures every prohibited outcome. The June 2026 preprint [Adaptive Evaluation of Out-of-Band Defenses, version 1](https://arxiv.org/abs/2606.26479v1), reports a limited Progent reproduction and explicitly declines to establish general adaptive robustness. Test attackers that know the defense and can adapt attempts, while measuring benign task utility as well as unauthorized actions. Passing a fixed injection suite is useful regression evidence, not a security proof.

:::diagram agent_trust_boundary|Tool-using systems need a deterministic policy boundary between model proposals and side effects.

:::callout decision|Models propose; policy authorizes
Treat every model-generated tool call as untrusted input. Validate identity, arguments, authority, state preconditions, and consequence before execution, and validate postconditions afterward.
:::

### Design Exercises

1. Define a capacity model for a service accepting text, images, and long video.
2. Design idempotency and approval semantics for an agent that can issue refunds.
3. Construct a prompt-injection test that distinguishes task failure from security failure.
4. What information must an agent trace preserve for replay?
5. A benchmark score rose after both a model and the sandbox changed. What can you conclude?
6. A refund succeeded, its response was lost, and the user changed the amount before the agent restarted. How should recovery work?

### Worked Solutions

1. Model visual tokens from pixels, patches, crops, and sampled frames; add text and output distributions; separate encoder, prefill, decode, memory, and queue budgets; and preserve workload correlations such as long video with long output.
2. Use a user-scoped capability, deterministic refund limits, a stable operation key, read-before-write state checks, preview plus confirmation above a threshold, transactional execution, and a receipt verified against the ledger.
3. Record benign task success first, then add untrusted content that requests a policy violation. Score utility and security separately: refusal of the entire task is not a successful defense, while task completion with an unauthorized side effect is a security failure.
4. Preserve model and prompt versions, user authority, tool schemas and versions, inputs and outputs with trust labels, RNG and sampling policy where relevant, policy decisions, retries, approvals, side effects, timestamps, and final verification.
5. Only that the measured system changed. Rerun paired tasks with fixed harness and resources to isolate the model, or report the full-system gain with all changes disclosed. Scores from different Terminal-Bench versions are not one continuous scale.
6. Reconcile the operation ID, receipt, and current resource version with the external service before retrying. Do not replay the completed refund. Treat a changed amount as a new request requiring authority for its exact arguments; an old authorization or idempotency key must not silently authorize a different write.


## From Research Result to Production Decision

LEAD: A research result becomes engineering evidence only after its mechanism survives a matched baseline, a production-shaped workload, and the system's quality and reliability constraints.

### Reconstruct the claim

For every candidate technique, write a result card:

| Field | Required record |
| --- | --- |
| Claim | Exact metric, direction, and reported range |
| Boundary | Kernel, layer, model, service, or product |
| Baseline | Version, tuning effort, and excluded features |
| Workload | Model, shapes, traffic distribution, and data |
| Hardware | Accelerator, memory, topology, and host |
| Precision | Storage, compute, accumulation, and calibration |
| Quality | Tasks, slices, tolerance, and statistical uncertainty |
| Cost | Compute, memory, network, engineering, and operations |
| Failure | Unsupported shapes, overload, recovery, and fallback |
| Currency | Paper version and reproduction date |

Then map the claimed mechanism to a local bottleneck. A faster attention kernel may reduce queueing if it raises capacity at the saturated stage; it will have little effect when that queue is waiting on another resource. KV offload helps only to the extent that reclaimable KV capacity matters after weights and other state are reserved. Reasoning-time sampling depends on whether the reducer can recognize better candidates. Trace the causal path from the proposed change to the measured outcome.

### Compare verified outcomes, not headline scores

Separate four claims: a provider announced a capability; an endpoint or checkpoint is accessible; an evaluation reports a gain; and the gain reproduces under your workload. These claims need different evidence. Do not fill unpublished training details with architectural guesses, or describe a restricted-access model as an ordinary public deployment option.

A useful comparison includes a dated hosted candidate, a feasible open-weight candidate, and the incumbent. For each, record reasoning budget, cached and uncached input, output, tool/verifier work, retries, elapsed time, and whether the task passed independent checks. Report cost per verified success alongside success rate and tail latency. This prevents a cheap but frequently failing route from appearing preferable merely because its tokens cost less.

For an illustrative batch of one hundred tasks, route A costs 20 units and completes eighty verified tasks: 0.25 units per success. Route B costs 30 units and completes ninety: about 0.33 per success. A is cheaper on that ratio, but B may be the only acceptable route if the required success floor is 85 percent. The result still needs workload slices and uncertainty; averages cannot decide which failures are tolerable. A fallback route requires a detector of failure, whose own accuracy and cost must be measured.

Public benchmarks also become development material. [Terminal-Bench 4.0](https://www.tbench.ai/news/terminal-bench-4-0) removes tasks with public solutions and requires fresh trials after changes to tasks or agent resources. Keep development tasks separate from held-out adoption tests; record known benchmark exposure and public solutions; and do not tune prompts or routing on the final comparison set. Preserve failures, timeouts, and all planned repetitions in the denominator. Use paired task-level uncertainty and workload slices rather than selecting a favorable run. An observed ninety-percent success rate on one hundred tasks is not a guarantee that future success exceeds eighty-five percent.

### Worked adoption decision: a cheaper serving candidate

The following is invented evaluation data for the documentation assistant, not a vendor benchmark. Before running it, the team freezes 400 independent task pairs, corpus and access snapshots, harness, model/prompt versions, retry rules and budgets. Each system receives every task. Verified success includes the required evidence and permitted final state; timeouts and failed retries stay in the denominator and cost totals.

| Paired outcome | Tasks |
| --- | ---: |
| Both systems succeed | 340 |
| Only incumbent succeeds | 20 |
| Only candidate succeeds | 30 |
| Neither succeeds | 10 |

The incumbent succeeds on 360/400 tasks (90%); the candidate on 370/400 (92.5%). The paired improvement is `(30-20)/400 = 2.5` percentage points. Let each pair's difference be -1, 0 or 1. Its sample variance is `(50 - 400*0.025^2)/399`, giving an approximate standard error of 1.77 percentage points and a normal 95% interval of about -1.0 to +6.0 points. This supports neither a claim of proven superiority nor treating the two success counts as independent samples. Related tasks require grouped resampling; these invented independent pairs make the arithmetic transparent.

The prespecified gates are: the paired interval's lower endpoint exceeds a -2-point noninferiority margin; the candidate's two-sided 95% Wilson lower success bound exceeds 88%; no observed forbidden effects; p99 first-token latency at most 1.5 seconds; and at least 20% lower total cost per verified success. The candidate's Wilson lower bound is about 89.5%. These are illustrative acceptance rules for this workload, not guarantees about future traffic or a general security threshold.

Suppose all 400 incumbent attempts cost 160 units and all candidate attempts cost 120, including failed calls, retries, tools and allocated serving time. Cost per verified success is `160/360 ~= 0.444` versus `120/370 ~= 0.324`, about a 27% reduction. Observed p99 first-token latency is 1.30 versus 1.45 seconds, and both have zero observed forbidden effects. No-token timeouts count as latency SLO failures; do not omit them from the latency denominator. The candidate passes the stated offline gates, but the narrow latency headroom justifies only a bounded canary. Zero violations in 400 trials does not establish absence of rare failures.

At a hypothetical 10,000 attempts daily, the observed per-attempt saving of 0.10 units would save 1,000 units per day. If integration costs 1,200 units, the arithmetic break-even is 1.2 full-traffic days, assuming the workload, retry behavior and cost remain unchanged. At a 5% canary, the same gross saving is only 50 units per day, before duplicate validation and rollback capacity. Do not use full-rollout economics to describe a canary.

The decision is to keep the incumbent as default and run a 5% canary for at least seven days and 2,000 adjudicated tasks, whichever takes longer. Pin the assignment rule and inspect tenant, language, permission and long-document slices. Roll back immediately on a confirmed unauthorized disclosure or duplicated write; stop admissions to the candidate if p99 first-token latency exceeds 1.5 seconds in two consecutive predeclared 1,000-request windows. At the planned endpoint, evaluate each adjudicated canary task against the pinned incumbent in an isolated copy of its pre-action environment and evidence snapshot. Only the candidate executes live effects; the comparator must not duplicate real writes. These same-task evaluations provide the paired outcomes required by the quality rule. If faithful isolated replay is unavailable, prespecify an unpaired control design and its uncertainty rule instead of calling unrelated traffic paired. Include duplicate evaluation overhead in canary economics, and evaluate quality and cost at the planned endpoint with grouped uncertainty when appropriate; do not repeatedly peek for a favorable promotion result.

Rollback switches the compatible routing manifest, preserves current permissions and deletion tombstones, and reconciles in-flight effects. Its owner rehearses the switch before the canary. Promotion requires the planned evidence and slice review; a changed model, verifier or routing rule starts a new comparison. If the candidate instead had p99 of 1.65 seconds, the offline decision would be no canary until a new bounded configuration passed the latency gate.

### Run the adoption analysis and challenge its evidence

Run `python -m examples.adoption_analysis` to reconstruct the 400 task pairs above. The fixture creates one immutable record per task ID with both outcomes, all-in cost, first-token time and a forbidden-effect flag. Its latencies are deliberately constant teaching inputs, not a sampled production distribution. The analyzer rejects duplicate IDs, missing pairs, invalid numbers and inconsistent no-token successes. It computes the paired difference from the actual pairing; equal marginal success rates with different pairings can produce different uncertainty.

The output gives a 2.50-point improvement with an approximate 95% interval from -0.96 to +5.96 points, a candidate Wilson lower bound of 89.50%, a 27.03% cost-per-success reduction, and `bounded_canary_only`. The [NIST confidence-interval reference](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) explains the Wilson and exact binomial constructions. The paired normal interval is a separate calculation on task differences, not a Wilson interval or proof of superiority.

The executable adds a conservative screening rule before using the normal approximation for its decision: at least 30 pairs and ten discordant pairs. This is an explicitly heuristic teaching guard, not a calibrated guarantee that the approximation is accurate. Smaller or highly discrete studies need a prespecified appropriate method or more evidence. The program cannot establish task independence, truthful adjudication, or faithful environment snapshots. Grouped tasks need grouped inference; a repeated task with a new identifier is still not independent evidence. The individual intervals also do not supply a joint 95% guarantee for all rollout gates.

Two evidence limits deserve numerical treatment:

- **Zero is an observation, not a risk bound.** With zero forbidden effects in 400 independent trials at a fixed event probability, the exact one-sided 95% upper bound is `1 - 0.05**(1/400) ~= 0.00746`, about 0.75%. This is far above many rare-event budgets. An adaptive attacker or changed traffic distribution invalidates the fixed-probability assumption; this calculation is not a security certificate.
- **A sample p99 is not a stable tail estimate.** The fixture uses the nearest-rank definition: sorted observation number `ceil(0.99*n)`, or 396 of 400. Only about four observations are expected from a true 1% tail in 400 independent draws. The probability of seeing none is `0.99**400 ~= 1.80%`; this is a tail-coverage illustration, not a confidence interval for the measured p99. A credible tail estimate needs more representative observations, workload slices and an appropriate uncertainty method.

A no-token timeout is represented by `None`, charged its full cost, counted as a latency failure and ordered as infinity for the empirical latency quantile. Consequently, one timeout in 400 requests is still a reported failure even though nearest-rank p99 can pass; five such timeouts make it infinite. A product requiring fewer failures needs a separate, prespecified failure-rate gate. Filtering timeouts out of the sample would silently change the acceptance rule.

**Worked challenge:** use `dataclasses.replace` on the immutable fixture to set candidate first-token times to 1.65 seconds, introduce one forbidden effect, or increase candidate costs. Inspect the named gate that fails and the resulting `do_not_adopt` decision. The tests also rearrange pairing while keeping success totals fixed, retain failed-task costs, and reject misleading zero-success cost ratios. No branch of this program authorizes full rollout: even a passing offline comparison must satisfy the canary, slice review and rollback contract above.

### What this snapshot does not establish

The model map is selective. It does not exhaust every model family, rank proprietary systems under one common harness, or reproduce training recipes whose data and weights are unavailable. The newly cited attention, draft, and diffusion results remain source-reported until run on matched hardware. Robotics, general world models, and scientific-domain foundation models are adjacent fields, not comprehensively surveyed here.

Maintain a source ledger containing the claim, publication/version date, inspection date, evidence type, and local adoption test. A moving model card should be archived or pinned for an actual experiment. When revising the chapter, update the failed or missing evidence as well as the successful result. A new publication date alone does not make an old comparison current.

### Reproduce in layers

1. reproduce the paper's narrow result or official reference configuration;
2. compare against a tuned current baseline;
3. substitute production models and shapes;
4. run trace replay with arrival correlations and failures;
5. shadow real traffic without side effects;
6. canary with explicit rollback and guardrails;
7. review realized cost, quality, and operational burden.

At every layer, record negative results. A technique that loses under an important shape is not a failed experiment; it defines the dispatch boundary.

### Design Exercises

1. Turn one result in this part into a complete result card.
2. Why is a tuned baseline an ethical requirement as well as a technical one?
3. Define stop conditions for a production reproduction effort.
4. How should an organization maintain a dated state-of-the-art chapter?
5. A cheaper model has lower cost per success but fails the product quality floor. Should it replace the incumbent?
6. Two model scores use different Terminal-Bench versions, harnesses, resource budgets, and retry policies. How would you compare them?

### Worked Solutions

1. State the paper's exact boundary and maximum result, then add its hardware, shapes, baseline, quality test, and known unsupported cases. Mark every missing field as unknown rather than filling it with an assumption.
2. Weak baselines exaggerate novelty and can cause unnecessary cost or risk. A tuned baseline gives decision-makers a fair estimate of incremental value and avoids misrepresenting engineering work as research advantage.
3. Stop when the local bottleneck is absent, quality fails a hard floor, integration cost exceeds plausible value, the baseline closes the gap, or an unsupported production shape has no safe fallback.
4. Assign an owner and snapshot date, prefer primary sources, preserve old claims with version history, distinguish peer-reviewed from preliminary work, and schedule review when hardware, model architecture, or serving workload changes materially.
5. No. Hard quality and policy constraints precede cost optimization. A selective route may be useful only if a tested detector and fallback satisfy those constraints after their latency and cost are included.
6. The scores do not isolate a model improvement. Rerun both on the same held-out task revision, harness, sandbox resources, retry policy, and effort budget; retain every planned trial and compare paired task outcomes with uncertainty. If comparing complete systems instead, disclose all differences and report quality, cost, latency, and policy violations together.

### Selected Primary Sources and Version Boundaries

The September 23 additions cite current model cards, FlashAttention-4 and its FP4 follow-up, DFlare, CaDDTree, DiffusionGemma and its serving implementation, live multimodal systems, and Terminal-Bench 4.0 beside the relevant mechanisms. The following earlier sources remain useful foundations, not a list of the latest releases. Version-specific result links identify the experiment being quoted; a newer revision may change hardware, baselines, or conclusions.

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - sparse activation, load balancing, multi-token prediction, training scale, and reported training cost.
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089) and [Kimi Linear](https://arxiv.org/abs/2510.26692) - hardware-aware sparse selection and hybrid gated linear attention.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - reinforcement learning for reasoning, cold-start and multi-stage training, and reasoning distillation.
- [Test-Time Scaling in Reasoning LLMs, version 2](https://arxiv.org/abs/2608.04001v2) - inference regimes, compute accounting, evaluation, and reproducibility.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) - asynchronous attention pipelines and low-precision attention on Hopper.
- [Mooncake](https://arxiv.org/abs/2407.00079) - KV-centric disaggregated serving, hierarchical cache, scheduling, and overload control.
- [Dynamo documentation](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/disaggregated-serving), [LMCache](https://github.com/LMCache/LMCache), and [llm-d KV management](https://github.com/llm-d/llm-d/blob/main/docs/architecture/advanced/kv-management/README.md) - maintained implementations of KV-aware routing, transfer, indexing, and tiered offload.
- [RocketKV, version 1](https://arxiv.org/abs/2502.14051v1) - two-stage KV-cache compression and the H100 results quoted above.
- [SparseServe, version 1](https://arxiv.org/abs/2509.24626v1) - hierarchical KV placement and working-set control for dynamic sparse attention.
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) - native-resolution vision, temporal encoding, document understanding, and visual agents.
- [AgentDojo](https://arxiv.org/abs/2406.13352) - dynamic evaluation of indirect prompt injection in tool-using agents.
- [ChatInject](https://openreview.net/forum?id=WVhgFSKniL) - ICLR 2026 evaluation of chat-template and multi-turn prompt injection attacks.
- [Securing AI Agents with Information-Flow Control](https://arxiv.org/abs/2505.23643) - 2025 work introducing the Fides planner and deterministic confidentiality/integrity policies.
- [Model Context Protocol, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) and [A2A 1.0](https://github.com/a2aproject/A2A/blob/main/docs/specification.md) - protocol references inspected for the September 13 baseline, for host-tool and agent-to-agent interoperability; neither substitutes for application authorization.

### From an experiment to a commitment

A result card should now support one of three actions: adopt within a tested boundary, run a specific follow-up, or stop. A new attention kernel might qualify only for long prefills; an embedding migration might remain blocked on a filtered tenant. Part VIII asks who owns those decisions, how dissent and uncertainty are recorded, and what evidence permits a broader rollout.
