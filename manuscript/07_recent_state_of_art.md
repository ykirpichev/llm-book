# Part VII - Recent State of the Art

This part records engineering results that materially changed how large language models were trained or served from 2024 through August 2026. It is a dated snapshot, not a permanent leaderboard. Reported speedups and benchmark scores belong to the cited paper's hardware, software, model, workload, baseline, and quality threshold.

The durable value of a recent result is usually not its rank. It is the mechanism that changed the resource model: sparse activation, better load balancing, reinforcement learning with verifiable rewards, explicit inference-time compute, asynchronous attention pipelines, disaggregated KV state, hierarchical memory, native-resolution multimodality, or enforceable trust boundaries for tools.

:::callout decision|How to read a state-of-the-art claim
Record the evaluated system, baseline, workload distribution, hardware, precision, quality constraint, and end-to-end boundary. Treat an isolated kernel speedup, benchmark score, or best-case throughput number as a hypothesis until the same advantage appears under the production contract.
:::

## Efficient Frontier Models and Reasoning Training

LEAD: Recent frontier systems show that capability is increasingly a co-design problem across architecture, data, optimization, precision, communication, and inference-time computation.

### Sparse activation changes the economic unit

DeepSeek-V3 reports 671 billion total parameters while activating 37 billion parameters per token. Its technical report describes 14.8 trillion pretraining tokens and 2.788 million H800 GPU-hours for the complete training run. The architecture combines a mixture-of-experts design, Multi-head Latent Attention, multi-token prediction, and an auxiliary-loss-free load-balancing strategy.

The system lesson is not that every model should copy one expert layout. Sparse activation separates three quantities that dense scaling often conflates:

- total parameter capacity, which affects checkpoint storage and aggregate expert memory;
- active parameters per token, which more directly control token-level compute;
- routed communication and imbalance, which can erase theoretical savings.

An MoE capacity plan therefore needs distributions, not only averages. Measure tokens per expert by layer and workload slice, overflow or dropped-token behavior, all-to-all bytes, hot-rank tails, expert memory residency, and useful throughput after communication. A balanced global histogram can still hide transient microbatch hotspots.

:::callout insight|Reported training cost is a system result
A GPU-hour number compresses hardware availability, precision, kernel efficiency, network topology, failure rate, checkpoint policy, and experiment reuse. Compare it only after reconstructing what the number includes.
:::

### Reasoning from reinforcement learning

DeepSeek-R1-Zero reports that large-scale reinforcement learning without a supervised fine-tuning warm start can elicit stronger reasoning behavior, but also reports readability and language-mixing problems. DeepSeek-R1 adds cold-start data and a multi-stage training process before and after reinforcement learning. The report also releases distilled dense models from 1.5B through 70B parameters and reports that reasoning behavior can transfer into smaller students.

This evidence changes the post-training design space in three ways.

1. **Verifiable rewards are high leverage.** Mathematics, code, formal constraints, and tool outcomes support scalable outcome checks, but a reward is only as sound as its verifier and sandbox.
2. **Capability and presentation are separate objectives.** A policy can improve task reward while degrading readability, language consistency, calibration, or safety.
3. **Distillation is a deployment lever.** A high-compute teacher can create traces or targets for a smaller model, but the student still requires independent quality, contamination, and serving evaluation.

Reward design must account for false acceptance, reward hacking, length incentives, duplicated samples, and train-evaluation overlap. Log the full rollout policy, sampling configuration, verifier version, reward components, rejected trajectories, and update batch. Without those artifacts, an apparent algorithmic gain may be an unrepeatable data-selection effect.

### Test-time scaling is a family of systems

Recent reasoning systems spend variable inference compute rather than mapping every request to one fixed decode. A 2026 study formalizes three regimes:

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

FlashAttention-3 targets Hopper GPUs with warp specialization, asynchronous Tensor Memory Accelerator transfers, overlap between matrix multiplication and softmax, and an FP8 path with block quantization. The paper reports 1.5 to 2.0 times speedup over FlashAttention-2 on H100, up to 740 FP16 TFLOP/s, and nearly 1.2 PFLOP/s for FP8. It also reports lower numerical error than a baseline FP8 attention implementation.

The durable mechanism is a deeper software pipeline. Once hardware exposes specialized asynchronous movement and matrix units, a kernel must schedule producer and consumer warps, manage barriers and buffers, and interleave non-matrix work so tensor cores remain fed. The optimization surface shifts from tile reuse alone to dependency timing.

The production acceptance test still needs ragged and causal shapes, head dimensions, sequence tails, backward or decode variants, numerical drift by layer, graph capture, and end-to-end model throughput. Peak forward attention throughput does not establish application speedup.

### KV-centric disaggregation

Mooncake reports a disaggregated serving architecture that separates prefill and decode clusters and treats KV state as a distributed object across GPU memory, CPU memory, and SSD. Its scheduler chooses placement and admission under latency objectives. The paper reports up to 525 percent higher throughput than its baseline in selected simulations and 75 percent more handled requests on a production workload.

The general lesson is that disaggregation works only when state movement is first-class. A phase boundary needs:

- a stable identity for model, adapter, tokenizer, prompt prefix, and KV format;
- an ownership transfer protocol with completion and retry semantics;
- bandwidth and tail-latency budgets for the transfer path;
- admission that rejects work before expensive partial execution when overload is unavoidable;
- observability for cache hit, transfer, queue, recompute, and abandoned state.

Disaggregation can improve independent scaling and isolation while adding network dependence and a new distributed lifecycle. It wins when phase imbalance and placement flexibility repay the transfer and coordination cost.

### Long-context memory is becoming hierarchical

Recent work explores two complementary ways to reduce long-context pressure. RocketKV combines coarse eviction with fine-grained sparse attention and reports up to 3 times end-to-end decode speedup and up to 31 percent peak-memory reduction on H100 with negligible loss on its evaluated tasks. SparseServe places unselected KV state in host memory, controls batch size from the active working set, and segments prefill by layer; it reports up to 9.26 times lower mean time to first token and up to 3.14 times higher generation throughput than its evaluated baselines.

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

## Multimodal and Tool-Using Systems

LEAD: Multimodal and agentic models turn context construction into an active systems boundary: inputs have geometry and time, outputs can invoke capabilities, and untrusted observations can influence irreversible actions.

### Native-resolution vision and long video

The Qwen2.5-VL technical report describes a native dynamic-resolution vision transformer, window attention, explicit temporal encoding, object localization, document parsing, and long-video processing. It reports competitive or leading results across several document, diagram, localization, and video benchmarks for the evaluated model sizes.

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

AgentDojo introduced 97 realistic tasks and 629 security test cases for agents operating over untrusted tool data. Its evaluation found that agents could fail ordinary tasks even without attacks and that then-current attacks and defenses had uneven coverage. ChatInject, published at ICLR 2026, reports that chat-template and multi-turn payloads substantially increased attack success over traditional prompt injection on AgentDojo and InjecAgent, including against prompt-based defenses.

The engineering conclusion is stronger than "improve the system prompt." Natural-language instructions and retrieved data share a representation, so the model alone should not be the final authorization boundary. Use conventional controls:

- capability-scoped credentials and allowlisted tool identities;
- typed schemas and deterministic validation;
- separation of untrusted content from control metadata;
- information-flow labels for confidentiality and integrity;
- explicit confirmation for destructive, financial, external, or privilege-changing actions;
- rate, cost, and recursion limits;
- immutable audit logs and replayable security tests.

Recent information-flow-control research explores planners that track integrity and confidentiality labels and enforce policies independently of the model's textual judgment. The broad lesson is durable even while implementations evolve: authorization belongs in code that can deny an action deterministically.

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

Then map the claimed mechanism to a local bottleneck. An attention kernel cannot fix a service dominated by queueing. A KV offload system cannot help if weights dominate memory. Reasoning-time sampling cannot improve a task whose verifier selects confidently wrong candidates.

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

### Primary Sources for the 2024-2026 Snapshot

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - sparse activation, load balancing, multi-token prediction, training scale, and reported training cost.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - reinforcement learning for reasoning, cold-start and multi-stage training, and reasoning distillation.
- [Test-Time Scaling in Reasoning LLMs](https://arxiv.org/abs/2608.04001) - inference regimes, compute accounting, evaluation, and reproducibility.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) - asynchronous attention pipelines and low-precision attention on Hopper.
- [Mooncake](https://arxiv.org/abs/2407.00079) - KV-centric disaggregated serving, hierarchical cache, scheduling, and overload control.
- [RocketKV](https://arxiv.org/abs/2502.14051) - two-stage KV-cache compression for long-context decode.
- [SparseServe](https://arxiv.org/abs/2509.24626) - hierarchical KV placement and working-set control for dynamic sparse attention.
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) - native-resolution vision, temporal encoding, document understanding, and visual agents.
- [AgentDojo](https://arxiv.org/abs/2406.13352) - dynamic evaluation of indirect prompt injection in tool-using agents.
- [ChatInject](https://openreview.net/forum?id=WVhgFSKniL) - ICLR 2026 evaluation of chat-template and multi-turn prompt injection attacks.
- [Securing AI Agents with Information-Flow Control](https://openreview.net/forum?id=2FkswFYju5) - 2026 research on deterministic confidentiality and integrity enforcement for agent planners.

### Recent State-of-the-Art Principles

1. Compare systems at the same boundary and quality level.
2. Separate total parameters, active compute, and communication.
3. Treat inference-time compute as a schedulable product resource.
4. Move state only with explicit identity, ownership, and admission.
5. Expect bottlenecks to migrate across compute, bandwidth, capacity, and coordination.
6. Model multimodal token load from geometry and time, not text limits alone.
7. Treat tool output as untrusted data and authorization as deterministic code.
8. Preserve the full experimental protocol so a result can be reproduced or retired.
