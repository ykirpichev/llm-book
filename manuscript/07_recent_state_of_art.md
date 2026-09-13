# Part VII - Recent State of the Art

This part connects selected research and maintained implementation evidence available by September 13, 2026 to the mechanisms developed throughout the book. It is a dated engineering snapshot, not an exhaustive catalogue or permanent leaderboard. Older results remain when they establish a useful mechanism; they are not presented as the newest available implementation. Reported speedups and benchmark scores belong to the cited paper's hardware, software, model, workload, baseline, and quality threshold.

The durable value of a recent result is usually not its rank. It is the mechanism that changed the resource model: sparse activation, better load balancing, reinforcement learning with verifiable rewards, explicit inference-time compute, asynchronous attention pipelines, disaggregated KV state, hierarchical memory, native-resolution multimodality, or enforceable trust boundaries for tools.

Read this part as a decision filter, not a chronology. For each result, first identify the mechanism and the resource or behavior it changes; then reconstruct the evaluation boundary and compare it with your workload. Finally, name the integration cost, quality guardrail, and experiment that would justify adoption. Return to Parts II–VI when a claim depends on training data, serving state, kernel behavior, topology, or application policy—the dated examples here do not replace those durable models.

:::callout decision|How to read a state-of-the-art claim
Record the evaluated system, baseline, workload distribution, hardware, precision, quality constraint, and end-to-end boundary. Treat an isolated kernel speedup, benchmark score, or best-case throughput number as a hypothesis until the same advantage appears under the production contract.
:::

## Efficient Frontier Models and Reasoning Training

LEAD: Recent frontier systems show that capability is increasingly a co-design problem across architecture, data, optimization, precision, communication, and inference-time computation.

### Sparse activation changes the economic unit

[DeepSeek-V3](https://arxiv.org/html/2412.19437v2) reports 671 billion total parameters while activating 37 billion parameters per token. Its technical report describes 14.8 trillion pretraining tokens and 2.788 million H800 GPU-hours for the official training run, including context extension and post-training but excluding prior research and ablation experiments. This is not the total research-and-development cost. The architecture combines a mixture-of-experts design, Multi-head Latent Attention, multi-token prediction, and an auxiliary-loss-free load-balancing strategy.

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

### Residual routing and learned lookup memory

[Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) studies multiple residual streams and constrained mixing between them. A doubly stochastic mixing matrix has nonnegative entries with every row and column summing to one. The intent is to retain richer routing without unconstrained amplification in the residual transport. This is a training architecture change, not a post-hoc serving flag.

For an independent two-stream example, the matrix `[[0.75,0.25],[0.25,0.75]]` maps scalar streams `[2,10]` to `[4,8]`. Each new stream is a convex combination; the sum remains twelve. By contrast, multiplying both streams by two doubles the sum at every layer. The constrained example illustrates transport stability, not a proof that an entire nonlinear network cannot diverge. Layer transformations, gates, finite-iteration constraint enforcement, and optimizer behavior still matter.

[Engram](https://arxiv.org/abs/2601.07372) explores learned conditional memory through n-gram-based lookup as a complement to conditional expert computation. The basic distinction is computation versus lookup: a short token pattern selects stored vectors, while contextual processing determines how useful those vectors are. A lookup table can hold recurring local associations without recomputing them through every dense layer. Hash collisions, table size, placement, and the quality of the contextual gate are engineering concerns.

This is not the same as retrieving current documents from an enterprise corpus. Learned lookup parameters do not automatically provide citations, per-user permissions, or instant factual updates. Nor does discussing Engram establish that a particular released model uses it; require that model's own architecture evidence. Similarly, a fixed lookup-time complexity says little about host/device transfer latency or cache misses.

### Reasoning from reinforcement learning

[DeepSeek-R1-Zero](https://arxiv.org/abs/2501.12948) reports that large-scale reinforcement learning without a supervised fine-tuning warm start can elicit stronger reasoning behavior, but also reports readability and language-mixing problems. DeepSeek-R1 adds cold-start data and a multi-stage training process before and after reinforcement learning. The report also releases distilled dense models from 1.5B through 70B parameters and reports that reasoning behavior can transfer into smaller students.

This evidence changes the post-training design space in three ways.

1. **Verifiable rewards are high leverage.** Mathematics, code, formal constraints, and tool outcomes support scalable outcome checks, but a reward is only as sound as its verifier and sandbox.
2. **Capability and presentation are separate objectives.** A policy can improve task reward while degrading readability, language consistency, calibration, or safety.
3. **Distillation is a deployment lever.** A high-compute teacher can create traces or targets for a smaller model, but the student still requires independent quality, contamination, and serving evaluation.

Reward design must account for false acceptance, reward hacking, length incentives, duplicated samples, and train-evaluation overlap. Log the full rollout policy, sampling configuration, verifier version, reward components, rejected trajectories, and update batch. Without those artifacts, an apparent algorithmic gain may be an unrepeatable data-selection effect.

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

### Worked Solutions

1. Slice expert load by layer, step, sequence class, and rank; replay a fixed routing trace with communication disabled or emulated; compare ideal and observed all-to-all time; and inspect whether the slowest expert or link predicts step time.
2. Use hidden verifiers, adversarially varied problem forms, manual trace audits, reward-component ablations, and pass rates on tasks where superficial shortcuts do not work. Check whether reward rises while independent correctness or readability falls.
3. Sequential scaling owns one growing KV history; best-of-N owns independent completed trajectories plus reducer state; prefix search owns a branching frontier, shared prefixes, pruning metadata, and branch-specific RNG. Their cancellation and memory-reclamation semantics differ.
4. Replace the teacher only when the student meets slice-level quality and safety floors, lowers end-to-end cost or latency under the real workload, and has a fallback for tasks where compression removed necessary capability.

## Serving, Attention, and Memory Hierarchies

LEAD: The recent inference frontier is defined less by one universal engine than by explicit management of asynchronous hardware, phase-specific work, KV placement, and overload.

### Attention pipelines on newer accelerators

[FlashAttention-3](https://arxiv.org/abs/2407.08608) targets Hopper GPUs with warp specialization, asynchronous Tensor Memory Accelerator transfers, overlap between matrix multiplication and softmax, and an FP8 path with block quantization. The paper reports 1.5 to 2.0 times speedup over FlashAttention-2 on H100, up to 740 FP16 TFLOP/s, and nearly 1.2 PFLOP/s for FP8. These are attention-kernel results over the evaluated shapes, not whole-model or service speedups. It also reports lower numerical error than a baseline FP8 attention implementation.

The durable mechanism is a deeper software pipeline. Once hardware exposes specialized asynchronous movement and matrix units, a kernel must schedule producer and consumer warps, manage barriers and buffers, and interleave non-matrix work so tensor cores remain fed. The optimization surface shifts from tile reuse alone to dependency timing.

The production acceptance test still needs ragged and causal shapes, head dimensions, sequence tails, backward or decode variants, numerical drift by layer, graph capture, and end-to-end model throughput. Peak forward attention throughput does not establish application speedup.

### Read the 2026 developments through the earlier chapters

Part IV derives Blackwell ownership and the FlashAttention-4 pipeline. Part III compares feature-based and block-parallel speculative drafts, including EAGLE-3 and DFlash, while retaining the target-distribution verification contract. Part II develops group-relative RL, sequence-level ratios, and feedback-conditioned self-distillation; Part V explains why asynchronous training needs policy-freshness control. These changes act on different parts of the system, so each needs its own acceptance test.

| Change | Quantity it tries to improve | Evidence that could reject it |
| --- | --- | --- |
| Better speculative draft | Committed target tokens per unit time | Draft/verification overhead or state rollback defeats the gain |
| New attention pipeline | Kernel critical-path time | Unsupported shapes, numerical drift, or another layer dominates |
| Recurrent / compressed state | Persistent bytes and history processing | Required long-range information is lost |
| Better RL feedback | Useful learning signal per rollout | Verifier shortcuts or held-out regression |
| Asynchronous RL | Useful learner/rollout utilization | Stale-data bias, variance, or selection changes quality |

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

### Long-context memory is becoming hierarchical

Recent work explores two complementary ways to reduce long-context pressure. [RocketKV, version 1](https://arxiv.org/abs/2502.14051v1), combines coarse eviction with fine-grained sparse attention and reports up to 3 times end-to-end decode speedup and up to 31 percent peak-memory reduction on H100 against a full-KV-cache baseline, with negligible loss on its evaluated tasks. These numbers belong to the February 2025 version; later revisions report a different hardware evaluation. [SparseServe](https://arxiv.org/abs/2509.24626v1) places unselected KV state in host memory, controls batch size from the active working set, and segments prefill by layer; it reports up to a 9.26-fold reduction in mean time to first token and up to 3.14 times higher generation throughput than its evaluated baselines. These maxima need not occur in the same configuration.

These are paper-reported maxima, not portable constants. They do establish a broader systems pattern: when attention becomes sparse, the bottleneck can move from HBM bandwidth to HBM capacity, irregular selection, or host-device movement. A valid comparison must include retrieval quality, task accuracy, selection overhead, cache thrashing, tail latency, and worst-case dense fallbacks.

:::callout pitfall|Compression is not free capacity
Eviction, quantization, sparse selection, and offload change the model's effective context or add selection and transfer work. Capacity is useful only when quality and latency remain inside the workload contract.
:::

### Design Exercises

1. Build an end-to-end benchmark that could reject a FlashAttention-3 deployment despite a faster kernel.
2. Derive the break-even condition for moving KV between prefill and decode workers.
3. How would admission control change when host memory is a KV backing tier?
4. Compare KV eviction, sparse attention, quantization, and offload as long-context policies.

### Worked Solutions

1. Include production sequence and head shapes, padding, causal masks, graph capture, competing kernels, full-layer time, accuracy drift, memory, compile cost, and tail latency. Reject it if layout conversion or unsupported shapes erase the kernel gain.
2. Disaggregation wins when saved queueing and improved phase utilization exceed transfer, synchronization, retry, and additional network-tail cost while KV fits within the destination's admission reserve.
3. Admit from predicted active working set and transfer bandwidth, not nominal host capacity. Reserve headroom, bound concurrent promotions, detect thrashing, and reject or degrade before latency collapses.
4. Eviction removes state and risks quality; sparse attention retains or tiers state but pays selection cost; quantization reduces bytes with numerical risk; offload preserves values but adds transfer latency. Hybrids should be evaluated against a full-attention quality and latency baseline.

## Multimodal Representations: Images, Video, and Speech

LEAD: A multimodal model must turn signals with space and time into a representation that a language system can learn from and act on. The encoder, alignment objective, sequence layout, and output contract are as important as the language backbone.

### From pixels to language-model inputs

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

[Block Diffusion](https://arxiv.org/abs/2503.09573) interpolates between autoregressive blocks and within-block diffusion. Earlier blocks can become fixed context while positions in the current block are refined together. This introduces a tunable tradeoff among block size, denoising steps, parallelism, and quality.

A causal prefix cache is valid because later output cannot alter its representations. A bidirectional mutable block lacks that property: changing one token can change the other positions' hidden states. Do not reuse its KV as if it were an unchanged autoregressive prefix. Cache reuse must follow the method's attention mask and update dependencies; an approximation needs an accuracy test of its own.

A diffusion **draft** inside speculative decoding is a different system from a diffusion **target** model. In the former, an autoregressive target can still define the output distribution if the proposal and verification procedure is valid; Part III discusses DFlash in that role. In the latter, the denoising model and generation schedule define the output behavior. There is no general promise of equality to an unrelated autoregressive model.

### Exercises and worked answers

1. **Does half as many model calls mean twice the speed?** No. Each call may process more positions and move more state; measure end-to-end time at matched quality and batch size.
2. **Why train across corruption levels?** Inference encounters different amounts of known context as generation proceeds; training only at one mask rate can mismatch that sequence.
3. **When may an earlier block be cached?** When its representations cannot depend on mutable later positions under the actual attention mask and model computation.
4. **Is masked denoising exact speculative sampling?** No. It can supply proposals, but a target-preserving verifier needs the correct proposal semantics and acceptance/correction logic.

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

:::diagram agent_trust_boundary|Tool-using systems need a deterministic policy boundary between model proposals and side effects.

:::callout decision|Models propose; policy authorizes
Treat every model-generated tool call as untrusted input. Validate identity, arguments, authority, state preconditions, and consequence before execution, and validate postconditions afterward.
:::

### Design Exercises

1. Define a capacity model for a service accepting text, images, and long video.
2. Design idempotency and approval semantics for an agent that can issue refunds.
3. Construct a prompt-injection test that distinguishes task failure from security failure.
4. What information must an agent trace preserve for replay?

### Worked Solutions

1. Model visual tokens from pixels, patches, crops, and sampled frames; add text and output distributions; separate encoder, prefill, decode, memory, and queue budgets; and preserve workload correlations such as long video with long output.
2. Use a user-scoped capability, deterministic refund limits, a stable operation key, read-before-write state checks, preview plus confirmation above a threshold, transactional execution, and a receipt verified against the ledger.
3. Record benign task success first, then add untrusted content that requests a policy violation. Score utility and security separately: refusal of the entire task is not a successful defense, while task completion with an unauthorized side effect is a security failure.
4. Preserve model and prompt versions, user authority, tool schemas and versions, inputs and outputs with trust labels, RNG and sampling policy where relevant, policy decisions, retries, approvals, side effects, timestamps, and final verification.

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

### Reproduce in layers

Use an evidence ladder:

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

### Worked Solutions

1. State the paper's exact boundary and maximum result, then add its hardware, shapes, baseline, quality test, and known unsupported cases. Mark every missing field as unknown rather than filling it with an assumption.
2. Weak baselines exaggerate novelty and can cause unnecessary cost or risk. A tuned baseline gives decision-makers a fair estimate of incremental value and avoids misrepresenting engineering work as research advantage.
3. Stop when the local bottleneck is absent, quality fails a hard floor, integration cost exceeds plausible value, the baseline closes the gap, or an unsupported production shape has no safe fallback.
4. Assign an owner and snapshot date, prefer primary sources, preserve old claims with version history, distinguish peer-reviewed from preliminary work, and schedule review when hardware, model architecture, or serving workload changes materially.

### Selected Primary Sources for the 2024-2026 Snapshot

Additional architecture, multimodal, and generation references appear beside their mechanisms above. Version-specific result links identify the experiment being quoted; a newer revision may change hardware, baselines, or conclusions.

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - sparse activation, load balancing, multi-token prediction, training scale, and reported training cost.
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
- [Model Context Protocol, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) and [A2A 1.0](https://github.com/a2aproject/A2A/blob/main/docs/specification.md) - current protocol contracts for host-tool and agent-to-agent interoperability; neither substitutes for application authorization.

### Recent State-of-the-Art Principles

1. Compare systems at the same boundary and quality level.
2. Separate total parameters, active compute, and communication.
3. Treat inference-time compute as a schedulable product resource.
4. Move state only with explicit identity, ownership, and admission.
5. Expect bottlenecks to migrate across compute, bandwidth, capacity, and coordination.
6. Model multimodal token load from geometry and time, not text limits alone.
7. Treat tool output as untrusted data and authorization as deterministic code.
8. Preserve the full experimental protocol so a result can be reproduced or retired.
