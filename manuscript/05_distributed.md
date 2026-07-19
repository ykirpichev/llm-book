# Part V - Distributed ML Systems

Distributed design begins when one device, process, or failure domain cannot satisfy the objective. Every parallelism axis exchanges one scarce resource for another: memory for communication, latency for throughput, or implementation simplicity for scale.

## Parallelism for Training

LEAD: Choose parallelism by locating what does not fit and what communication the workload can amortize. Compose axes only after understanding each one independently.

:::diagram parallelism_map|Data, tensor, pipeline, sequence, and expert parallelism partition different dimensions and create different collectives.

### Data parallelism

Each replica processes different examples and owns a full model copy. Gradients are reduced before the update. Communication volume is proportional to parameter or gradient size, while compute grows with local batch.

Ring all-reduce sends approximately `2 (n - 1) / n` times the tensor size per rank. Tree algorithms reduce latency for smaller messages. Hierarchical collectives exploit fast links within a node and aggregate across slower inter-node links.

Overlap gradient reduction with backpropagation by launching buckets when layers finish. Buckets that are too small pay latency; buckets that are too large delay overlap. The optimal size follows network latency, bandwidth, framework overhead, and layer order.

### Sharded data parallelism

Optimizer states, gradients, and parameters can be partitioned across data-parallel ranks. ZeRO-style stages progressively shard these objects. Fully sharded data parallelism gathers parameter shards before computation and frees or reshares them after use.

Sharding reduces memory per rank but adds collectives and makes prefetch timing critical. Small layers can become latency-bound; flat parameter groups and controlled prefetch help. Activation checkpointing further trades recomputation for memory.

### Tensor parallelism

Tensor parallelism splits a matrix operation across devices. Column and row partitioning of linear layers alternate all-gather or all-reduce patterns. Communication occurs within layers, so fast links and topology awareness matter.

For transformer blocks, choose partitions that avoid unnecessary collectives between adjacent operations. Keep tensor-parallel groups within the highest-bandwidth domain when possible. At high degree, smaller per-rank matmuls reduce efficiency while collective overhead grows.

### Pipeline parallelism

Pipeline stages own layer ranges and pass activations. Microbatches fill the pipeline. A simple schedule has a bubble fraction related to `(stages - 1) / microbatches`. More microbatches reduce bubbles but increase activation state and may change optimization.

One-forward-one-backward schedules reduce peak activation memory. Interleaved schedules place multiple model chunks per device to reduce bubbles at the cost of more communication and complexity. Stage balance must include compute, activation transfer, and heterogeneous layer cost.

### Sequence and context parallelism

Sequence parallelism partitions tokenwise operations and reduces replicated activation memory within tensor-parallel regions. Context parallelism partitions long sequence attention, requiring exchange or ring-style traversal of K/V blocks. It is attractive when activations or attention state dominate and long contexts provide enough work to amortize communication.

### Expert parallelism

Experts are placed across ranks; tokens move via all-to-all according to routing. Load balance, token capacity, and topology dominate. Replicating popular experts, routing within locality groups, or using hierarchical all-to-all can reduce hotspots.

### Composing a 3D plan

Suppose a model does not fit one node even with sharding, matmuls are large, and the cluster spans many nodes. A common plan keeps tensor parallelism inside a fast node, pipelines across a small number of node groups, and applies data parallelism across replicas. The product of degrees equals world size.

For each candidate, estimate:

- parameter, optimizer, gradient, and activation memory per rank;
- collective bytes and latency on each topology link;
- matmul efficiency at local shapes;
- pipeline bubble and imbalance;
- checkpoint and failure-recovery complexity.

:::callout decision|Parallelism is a topology assignment problem
Do not present degrees without mapping groups to physical links. The same logical plan can be fast within an NVLink domain and unusable across oversubscribed Ethernet.
:::

### Principal Interview Review

1. Compare FSDP and tensor parallelism by memory and communication timing.
2. Derive the pipeline bubble for stages and microbatches.
3. How do you place a 3D parallel plan on a multi-node topology?
4. What causes expert-parallel all-to-all imbalance?
5. When does increasing tensor-parallel degree reduce throughput?

## Distributed Inference

LEAD: Inference parallelism must protect latency while accommodating model and state. Communication occurs on every generated token, so patterns acceptable in training may fail in decode.

### Tensor parallel decode

Tensor parallelism makes a large model fit and divides matmul work. It also adds collectives per layer or block. During decode, small local matmuls and repeated collectives can be latency-bound. Keep the group within the fastest fabric and avoid degrees larger than the model or batch can use efficiently.

Batching helps local GEMM efficiency but can conflict with latency. Quantization may reduce compute and weight traffic enough that communication becomes a larger fraction of step time.

### Pipeline inference

Pipeline parallelism partitions layers and passes activations. For a single sequence, token latency includes all stages; with many microbatches or sequences, the pipeline can provide throughput. Dynamic output lengths and continuous batching complicate stage balance.

Pipeline is useful when model memory cannot be partitioned solely by tensor parallelism or when topology favors stage placement. It is less attractive for strict low-batch latency due to bubbles and handoff.

### Replication

Replicas scale request throughput and isolate failures. Route requests by model, adapter, cache affinity, capacity, and tenant. State such as KV cache makes mid-request migration expensive. Sticky routing improves reuse but can create hotspots.

Use power-of-two choices or cost-aware routing rather than raw request count. A request with 100,000 prompt tokens is not equivalent to a short chat turn.

### KV transfer and remote memory

Disaggregated or migrated inference may transfer KV across workers. Compressing transfer saves network bytes but consumes device cycles and may affect quality. Remote KV caches can pool capacity but introduce a latency-sensitive dependency on network and cache service availability.

Transfer priority should account for whether the destination GPU will stall. Chunking enables overlap. Checksums, sequence identity, model version, and page order protect correctness.

### Expert-parallel serving

MoE serving routes tokens to experts at every layer. Batch size by expert can be tiny and imbalanced. Expert replication, token reordering, capacity scheduling, and fused dispatch matter. Cross-node expert traffic can dominate decode.

The scheduler may co-batch tokens across requests before expert dispatch, improving expert GEMMs at the cost of synchronization and latency. Popular experts can be replicated based on observed routing, but replicas complicate placement and memory.

### Fault handling

A collective failure can invalidate all ranks in a model replica. Maintain multiple replicas and drain the failed group. Retrying a generation requires request input, random seed or sampling state, emitted token boundary, and policy on duplicate streaming.

For user-visible streams, do not silently restart and emit a divergent continuation after partial output. Either resume from preserved state with deterministic semantics, mark a retry boundary, or fail clearly.

:::callout pitfall|Failover is not just loading weights elsewhere
Inference is stateful. KV pages, generated tokens, sampler RNG, adapter, tools, and stream acknowledgments determine whether a request can resume safely.
:::

### Principal Interview Review

1. Why is tensor parallelism more latency-sensitive during decode than training?
2. How would you route requests across replicas with different cache state?
3. When is remote KV memory beneficial?
4. Design failure behavior for a streaming generation.
5. How can expert replication improve MoE serving, and what does it cost?

## Designing an LLM Serving Platform

LEAD: A platform turns model execution into a multi-tenant product with admission, isolation, observability, versioning, and cost accountability.

:::diagram system_design|The serving path includes a control loop: observed SLOs and capacity feed admission, scheduling, routing, and cache policy.

### Clarify requirements

Ask for models and sizes, request rate, prompt and output distributions, context limit, streaming, latency SLOs, availability, tenant isolation, data retention, adapter count, regions, hardware, and cost objective. Distinguish interactive from batch traffic.

Create a workload table rather than using only averages. Include p50, p95, and maximum token counts; burst factor; cacheable prefix rate; model popularity; and quality or sampling modes.

### API and gateway

The gateway authenticates, validates limits, applies policy, assigns request identity, supports idempotency where appropriate, and emits structured trace context. It should not become a heavyweight tokenizer bottleneck unless centralized token accounting is necessary.

Rate limits may use request, token, compute, or concurrency budgets. Token budgets better reflect cost but require estimates for output. Reservations can be reconciled after completion.

### Router and model pools

Maintain pools by model version and hardware compatibility. Route using estimated service cost, queue delay, cache affinity, region, and health. A slow-start policy warms new replicas and graphs before receiving full traffic.

Canary by request identity so multi-turn conversations remain on a consistent version. Record model, tokenizer, engine, kernel, quantization, prompt-template, and policy versions in every response trace.

### Engine scheduler

The engine owns continuous batching, prefill chunking, KV pages, prefix cache, speculative policy, graph selection, sampling, cancellation, and streaming. Separate the data plane from a control plane that manages versions, placement, capacity targets, and rollout.

### Capacity model

Estimate separately:

- **weight capacity:** model bytes and parallel replicas;
- **KV capacity:** active token state and fragmentation reserve;
- **prefill compute:** prompt tokens per second by length bucket;
- **decode capacity:** output tokens per second by batch and context;
- **network:** collectives, KV transfer, and client egress;
- **queue headroom:** burst, failure, and canary reserve.

A request may be limited by any one. Provisioning only from average GPU utilization misses KV exhaustion and tail queueing.

### Multi-tenancy and isolation

Protect memory, cache keys, logs, metrics, and timing side channels. Apply per-tenant quotas and fair scheduling. Encrypt transport. Minimize prompt retention and control debug capture. Prefix-cache reuse across tenants should be disabled unless content is explicitly public and keying is safe.

Adapters create another isolation boundary. Validate adapter ownership and model compatibility, cap memory, and prevent an adapter from altering shared persistent state.

### Observability

Expose request decomposition:

- gateway and queue delay;
- prefill time and tokens;
- time to first token;
- per-token and total decode time;
- scheduler wait per iteration;
- batch composition;
- weight and KV memory;
- cache hit and saved work;
- collective time;
- speculative proposal and acceptance;
- cancellation and rejection reason;
- cost allocation.

Use high-cardinality request traces selectively; aggregate durable metrics by model, version, shape bucket, tenant class, and hardware. Correlate engine metrics with kernel traces during investigation.

### Rollout and rollback

Release an immutable model-engine bundle. Validate offline, replay shadow traffic, warm a canary, compare quality and SLOs, and ramp. Rollback must be one control-plane operation. Preserve prior capacity until the new version clears the observation window.

:::callout insight|Name the control plane
Principal system design answers distinguish per-request execution from placement, rollout, policy, and capacity decisions. Without a control plane, the diagram is a collection of workers.
:::

### Principal Interview Review

1. Design an LLM service for interactive and batch workloads on the same fleet.
2. Build a capacity model that includes KV, not only FLOPs.
3. How do you canary a model when conversations are multi-turn?
4. What telemetry localizes time-to-first-token regression?
5. Which serving configuration must roll back atomically with model weights?

## Reliability, Checkpoints, and Evaluation Infrastructure

LEAD: ML reliability is the ability to preserve meaning through failure: the same data, model, metric, and release must remain identifiable and recoverable.

### Failure domains

Enumerate hardware, process, host, rack, network, storage, region, dependency, control-plane, and human configuration failures. Then decide whether each operation retries, resumes, degrades, or fails.

Retries are safe only when side effects are idempotent or deduplicated. Training input consumption, checkpoint publication, model registration, and streaming output each need a durable identity.

### Checkpoint service

A checkpoint manifest maps logical tensors and optimizer state to immutable shards, with checksums, dtype, shape, partitioning, and version. Publication should be atomic: readers see either the previous complete manifest or the new complete one.

Scale-out writes can overload shared storage. Stagger, aggregate locally, compress where CPU does not become the bottleneck, and provision bandwidth from checkpoint size divided by required interval. Async writes need backpressure so multiple pending checkpoints do not consume all host or device memory.

Test cross-topology restore when resharding is supported. Retention policy should include recovery points, milestones, and legal deletion requirements.

### Evaluation service

An evaluation service owns dataset versions, runners, judge versions, execution sandboxes, metrics, confidence intervals, and comparison reports. It must prevent test data from leaking into prompts, logs, or training exports.

For code, isolate execution with resource limits and hidden tests. For model judges, preserve raw judgments, rationales if allowed, order randomization, and calibration results. For human evaluation, record rubric, annotator eligibility, and adjudication.

### Release gates

Define hard blocks and review-required thresholds. Examples:

- no critical safety regression;
- capability primary metric above target with confidence;
- no declared segment below its floor;
- time to first token and inter-token p99 within SLO;
- cost per successful task within budget;
- rollback tested;
- lineage and artifacts complete.

Gates should be sparse enough to be meaningful. A wall of weak thresholds invites metric gaming and exception fatigue.

### Incident response

For a model incident, preserve prompts and outputs according to privacy policy, identify version and configuration, bound affected traffic, stop or reroute exposure, reproduce with a minimal case, and determine whether failure arises from data, model, engine, policy, or dependency.

Separate mitigation from root cause. A prompt filter may stop immediate harm while a model or data fix proceeds. Track follow-up ownership and verify that the fix does not merely move the failure to an adjacent slice.

:::callout decision|Reliability requires semantic versioning
A checksum identifies bytes. A release identity must also bind tokenizer, prompt template, tools, policy, quantization, engine, and evaluation. User-visible behavior is the composition.
:::

### Principal Interview Review

1. Design atomic publication for a 20 TB sharded checkpoint.
2. How do you prevent evaluation leakage through infrastructure?
3. Which release gates should be hard versus reviewable?
4. Design an incident response for a quality regression seen only at long context.
5. Explain semantic release identity for an LLM system.

