# Part III - LLM Inference Systems

Inference turns a fixed model into a variable, stateful, multi-tenant service. Prompt length, output length, sampling policy, cache state, adapter choice, arrival bursts, and hardware topology change the cost of every request. The systems problem is to preserve a declared quality and latency contract while sharing expensive accelerators efficiently.

This part is self-contained. Model phases, KV state, batching, decoding semantics, parallel placement, compression, and operations are developed here from first principles; the only prerequisites are transformer basics and comfort with back-of-envelope arithmetic. Part IV writes the kernels that execute these decisions on one GPU. This part owns the request, the fleet, and the contract. Each part can be read alone.

### The running service used throughout

Every section returns to one illustrative deployment so estimates stay comparable. Treat the numbers as a teaching configuration, not a product benchmark.

| Item | Value in this part |
| --- | --- |
| Model | dense 7B-class transformer: `L = 32` layers, hidden `D = 4096`, GQA with `H_kv = 8`, head dim `d = 128` |
| Weights | FP16, approximately 14 GB |
| KV cost | `2 * L * H_kv * d * 2` bytes = 128 KiB (about 131 kB) per cached token |
| Replica | one 80 GB accelerator, ~3 TB/s HBM, ~50 GB/s usable inter-node path |
| Workload | median prompt 2,000 tokens with a shared 500-token system prefix; median output 300 tokens; bursty arrivals |
| SLOs | streaming; p99 time to first token <= 1.5 s; p99 inter-token gap <= 100 ms |

The model geometry deliberately matches the running example of Part IV, so a reader moving between the parts sees the same tensors from the kernel side and from the service side.

This part follows a request through that service, from API admission to token streaming and cleanup. Every chapter ends with principal-level interview questions and complete answer keys.

:::callout decision|The serving contract comes first
Before choosing an engine or optimization, define model semantics, supported request features, latency objectives, overload behavior, and the quality deviations that are permitted. A faster system that silently changes these is a different product.
:::

### Map of this part

| Section | Role in the story |
| --- | --- |
| Lifecycle and metrics | Define the request path and the measurements every later claim uses |
| Prefill, decode, modeling | Build the bytes-and-FLOPs model of one request on one replica |
| KV cache and prefix reuse | Manage the working set that decode carries between steps |
| Scheduling and admission | Share one replica across requests without breaking the contract |
| Sampling and speculation | Preserve decoding semantics while accelerating token selection |
| Parallel and disaggregated serving | Place phases and state across devices and pools |
| Quantization and adapters | Shrink bytes and multiply variants without changing the product |
| Production and reliability | Operate the whole system through failures and rollouts |

## Request Lifecycle, Metrics, and Workload Models

LEAD: An LLM request is a distributed transaction with a long, streaming response. Correct measurement separates queueing, model phases, transport, and user-visible cadence instead of collapsing everything into one throughput number.

The map starts here because every optimization in this part is judged against these measurements. For the running service the contract is concrete: a request must show its first token within 1.5 seconds at the 99th percentile and must never let the visible gap between tokens exceed 100 milliseconds.

:::diagram request_lifecycle|A request crosses gateway, queue, prefill, decode, and streaming boundaries. Each boundary needs an owner and a timestamp before any latency claim is meaningful.

### The request path

A production request usually crosses these stages:

1. authenticate, rate-limit, and validate the request;
2. normalize the prompt, apply the chat template, and tokenize;
3. route by model, adapter, tenant, locality, and capacity;
4. wait in an admission or execution queue;
5. allocate or reuse KV state and run prefill;
6. sample and stream the first token;
7. repeat scheduling, decode, postprocessing, and streaming;
8. stop on an end token, length limit, constraint, cancellation, or error;
9. release pages, adapter references, graph slots, and accounting state.

Each boundary needs an owner and a timestamp. Without this, a regression in tokenization, routing, queueing, networking, or client backpressure may be misdiagnosed as a slow GPU.

The request contract includes more than input text. It includes model and tokenizer versions, chat template, adapter, maximum context and output, stop rules, logits processors, random seed, sampling parameters, structured-output grammar, priority, deadline, streaming mode, and any approximation policy. These fields affect both semantics and execution shape.

### Latency definitions

Let `a` be arrival at the service boundary, `s` the first model-service start, `f` the first token visible at that boundary, and `o_i` the visible time of output token `i`.

| Metric | Definition | What it reveals |
| --- | --- | --- |
| Queue delay | `s - a` | Admission pressure and scheduler delay |
| Time to first token | `f - a` | Queueing, preprocessing, prefill, first sampling, and transport |
| Inter-token latency | `o_i - o_(i-1)` | Per-step cadence and stalls |
| Time per output token | Usually `(o_last - f) / (N - 1)` | Average generation speed after the first token |
| End-to-end latency | `o_last - a` | Total user wait for a completed response |

Time per output token is an average; it can hide a one-second generation stall among many fast tokens. Report an inter-token latency distribution or a maximum-gap metric for interactive products. Also define whether the first token means sampled on the GPU, serialized by the server, received by the gateway, or observed by the client. A metric without a boundary is ambiguous.

For non-streaming clients, end-to-end latency matters most, but internal first-token and cadence metrics still locate faults. For streaming clients, a reasonable SLO often has separate thresholds for time to first token and inter-token latency.

Decompose the running service's budget before optimizing anything. Of the 1.5-second first-token allowance, transport and tokenization cost tens of milliseconds and prefill for a 2,000-token prompt costs tens to low hundreds of milliseconds on a healthy replica. The remainder - often more than a second - is queueing and scheduling slack. That allocation is why later sections spend more effort on admission, batching, and cache reuse than on shaving prefill kernels.

### Throughput, goodput, and capacity

Report input tokens per second and output tokens per second separately. They exercise different phases and cannot be added without context. Requests per second is meaningful only with a declared prompt/output distribution. Per-GPU token throughput is useful for efficiency, while per-user output rate describes interactivity.

**Goodput** is work that satisfies the product contract. One useful definition is:

`goodput = requests meeting every required SLO / measurement interval`

A request that eventually completes but violates its time-to-first-token or token-cadence requirement consumes capacity without contributing to goodput. This makes goodput a better autoscaling and architecture objective than raw throughput under strict latency objectives.

Capacity is not the peak point on a throughput curve. It is the highest sustained arrival process for which tail latency, error rate, memory reserve, quality, and fairness remain within contract. State the percentile and duration: for example, 99th-percentile first-token latency under a thirty-minute replay with a specified burst model.

### Open-loop and closed-loop load

In a **closed-loop** test, each simulated client waits for completion before sending another request. When the service slows, offered load falls. This is useful for user-session modeling but can hide overload.

In an **open-loop** test, arrivals follow a process independent of completions. Queueing is visible because offered load does not retreat when latency rises. Use open-loop tests to find sustainable capacity, but choose realistic burstiness rather than only a smooth Poisson stream.

Avoid coordinated omission: a load generator that pauses arrivals while the server is slow fails to count requests that would have arrived during the stall. Record intended arrival time separately from actual send time, and include client-side queue delay if the generator cannot keep up.

### A workload is a joint distribution

Average prompt length and average output length are insufficient. The service cost depends on their joint distribution and correlations with:

- arrival time and burst size;
- model, adapter, and tenant;
- sampling or structured-output policy;
- prefix-reuse probability and shared-prefix length;
- cancellation point and client backpressure;
- speculative acceptance;
- multimodal input size;
- priority and deadline.

Long prompts and long outputs create different pressure. A long prompt is a large prefill event; a long output holds KV memory and a decode slot for many iterations. A few long-tail requests can control memory reserve and tail latency even when token-weighted averages look benign.

Build at least three workload views: a stable production replay, a stress distribution that magnifies tails, and synthetic shape sweeps for causal explanation. Preserve request correlations in replay. Independently sampling each feature can create combinations that never occur or remove the bursts that matter.

### Trace decomposition

Use a request identifier across gateway, scheduler, worker, and stream transport. A useful span model includes:

`gateway -> tokenize -> route -> queue -> prefill -> first sample -> decode iterations -> stream flush -> cleanup`

Within the worker, attach phase, batch composition, input/output token counts, allocated and reused KV tokens, model/adapter version, graph or engine variant, device group, and termination reason. Sample detailed traces if volume is high, but retain aggregate histograms for every request class.

Tail diagnosis should compare a slow request with the batch and node state around it. Per-request latency alone cannot reveal that a short decode waited behind a long prefill, a cache eviction triggered recomputation, or a peer stalled a tensor-parallel collective.

### Benchmark protocol

A defensible benchmark declares:

- exact model, tokenizer, precision, engine, kernels, drivers, and hardware topology;
- prompt/output distributions, sampling policy, and dataset transformations;
- concurrency or arrival process, duration, warmup, and random seeds;
- inclusion boundaries for tokenization, network, queueing, and detokenization;
- cold-start and warm steady-state results;
- latency percentiles, inter-token gaps, throughput, goodput, errors, and memory;
- output correctness or quality checks.

Run both a component benchmark and an end-to-end replay. Component results explain a mechanism; the replay proves that scheduling, transport, memory pressure, and application features do not erase the gain.

:::callout pitfall|Tokens per second can be manufactured
Changing prompt/output mix, ignoring queue time, omitting rejected requests, reducing sampling work, or reporting a saturated batch with no latency constraint can raise tokens per second without improving a real service.
:::

### Principal Interview Review

1. Define time to first token, inter-token latency, time per output token, and end-to-end latency at precise boundaries.
2. Why is goodput often a better capacity objective than raw token throughput?
3. Design a load test that exposes overload without coordinated omission.
4. What workload correlations must a production replay preserve?
5. A GPU trace looks healthy, but user latency regressed. How do you localize the problem?
6. What belongs in a reproducible inference benchmark report?

### Answer Key

#### 1. Latency metrics

At the chosen service boundary, time to first token is arrival until the first emitted token. Inter-token latency is the gap between consecutive visible tokens. Time per output token is usually the average generation interval after the first token, while end-to-end latency is arrival until the final token or terminal response. State whether the boundary is worker, gateway, or client and whether tokenization and transport are included. Report an inter-token distribution because its average hides generation stalls.

#### 2. Goodput

Raw throughput counts work even when requests miss latency, error, fairness, or quality requirements. Goodput counts only requests that satisfy every required SLO, so it captures the useful capacity of the system. At overload, batching can raise raw tokens per second while queue delay causes nearly every interactive request to miss its first-token target. Autoscaling against goodput or SLO attainment avoids rewarding that state.

#### 3. Load testing overload

Use an open-loop arrival generator whose intended send times do not depend on response completion. Replay realistic prompt/output correlations and burstiness, increase offered load across a sweep, and measure queue growth, tail latency, goodput, errors, and memory reserve. Timestamp intended and actual send times so generator delay is visible. Run long enough to expose slow leaks and output-length tails, and include a closed-loop session test separately when it reflects product behavior.

#### 4. Workload correlations

Preserve at least prompt versus output length, request features versus shape, tenant/model/adapter locality, prefix sharing, arrival bursts, cancellation behavior, and speculative acceptance. Long prompts may cluster in batch jobs, a specific adapter may have longer outputs, or a shared system prompt may create most prefix hits. Breaking these relationships changes queueing, memory lifetime, cache behavior, and batch compatibility.

#### 5. Healthy GPU, slow users

Trace from the user boundary inward. Split latency into gateway, tokenization, routing, queue, model phases, sampling, serialization, network flush, and client backpressure. Correlate the slow request with scheduler state, batch composition, cache events, collective peers, and node health. A well-utilized GPU can coexist with excessive queueing or stream buffering; utilization proves activity, not useful latency.

#### 6. Reproducible report

Record the model/tokenizer and semantic settings, hardware and topology, software stack and versions, precision and engine configuration, complete workload distribution, arrival process, warmup, measurement boundary, seeds, and all exclusions. Report latency percentiles and gaps, input/output throughput, goodput, errors, memory, power if relevant, and quality validation. Include raw traces or enough artifacts to reproduce them, plus cold and steady-state behavior.

## Prefill, Decode, and Performance Modeling

LEAD: A request moves through phases with different shapes and bottlenecks. Prefill exposes token parallelism and often high arithmetic intensity; decode is sequential across time and repeatedly reads weights and growing KV state.

The lifecycle defined where time can go; this section models how much the model itself must cost. The tool is the same resource ledger used throughout the book: count the bytes each phase must move and the FLOPs it must execute, and let the running service make the quantities concrete.

### Phase anatomy

Text-only autoregressive serving has two principal model phases:

- **Prefill** processes all uncached prompt tokens, produces the first-token state, and materializes their KV cache.
- **Decode** adds one token per active sequence per iteration, reads prior KV state, samples a token, and appends new KV.

Multimodal models may add an encoder phase before prefill. Image, audio, or video preprocessing and encoding can dominate first-token latency, expand into many model tokens, and require a different accelerator profile. Treat encode, prefill, and decode as distinct stages even when one engine executes all three.

| Property | Prefill | Decode |
| --- | --- | --- |
| Token parallelism | Many prompt positions | One new position per sequence |
| Typical linear shape | Large token dimension | Small token dimension, often batch-sized |
| Common limit | Compute or attention IO | Weight/KV bandwidth, collectives, launch overhead |
| KV behavior | Creates prompt state | Reads history and appends one position |
| Primary user metric | Time to first token | Token cadence and output rate |

:::diagram roofline|Arithmetic intensity helps explain why a large prefill can approach compute throughput while small-batch decode remains limited by bytes moved per generated token.

### Transformer work per phase

Each layer performs normalization, projections for attention, attention itself, an output projection, another normalization, and an MLP, with residual paths and optional routing for mixture-of-experts models.

For a dense model, most parameter FLOPs occur in linear layers. Prefill applies them to many prompt positions at once, creating GEMM shapes with substantial reuse. Decode applies them to one new position per sequence, so the effective token dimension is approximately active batch size.

Attention has a different scaling. For sequence length `S` and head dimension `d`, full causal prefill performs work proportional to `S squared d` but IO-aware kernels avoid materializing the full score matrix. One decode query attends to `S` cached keys and values, so per-step attention work and KV traffic grow approximately linearly with context length. At very long context, attention can dominate a decode step even when weight reads dominate at shorter context.

### Arithmetic intensity and roofline reasoning

Arithmetic intensity is useful FLOPs divided by bytes moved from the limiting memory level. A first-order roofline is:

`attainable compute rate <= min(peak compute, bandwidth times arithmetic intensity)`

For an unbatched decode linear layer, a weight may be fetched for only one multiply-accumulate before the next layer. Batching sequences reuses each weight across rows, raising intensity. Weight quantization reduces bytes but adds scale reads and dequantization. A fused epilogue removes intermediate reads and writes. The correct denominator is bytes actually transferred, not tensor size on paper.

Prefill matrix multiplications can become compute-bound because tiled kernels reuse weights and activations. Attention can remain IO-bound if an implementation materializes scores or makes extra HBM passes. The roofline identifies a bound, not a schedule: low achieved performance may still come from poor tiling, serialization, launch gaps, or an imbalanced distributed peer.

### A lower-bound model for decode

Let `W` be active model weight bytes per replica or model-parallel group and `BW_w` sustainable bandwidth for those reads. For small-batch decode:

`t_weights >= W / BW_w`

Let `K(S, B)` be KV bytes read at current context and batch, `BW_kv` the sustainable bandwidth for that access pattern, `C` collective time, and `O` launch, scheduler, sampling, and synchronization overhead. A deliberately conservative step model is:

`t_step >= max(t_weights, t_kv, t_compute) + C + O`

Some traffic and compute overlap, so do not blindly sum all resource bounds. Measure which terms overlap on the real engine. The model is valuable because it predicts how batch, precision, context, and topology should move the result.

If latency is near the weight-read bound, optimizing scalar arithmetic cannot produce a large gain. Reduce bytes, increase batch reuse, improve placement, or change the model. If latency is far above every bound, examine kernel efficiency, launch gaps, collectives, and scheduling.

#### The running service's step floor

Plug in the numbers. Weights are approximately 14 GB in FP16 and the replica sustains roughly 3 TB/s from HBM, so a small-batch decode step cannot beat:

`t_weights >= 14e9 / 3e12 = 4.7 ms`

That is a ceiling near 210 steps per second, shared by every sequence in the batch. KV traffic starts small by comparison: at batch 8 with 4,096-token contexts, a step reads about `8 * 4096 * 131 kB = 4.3 GB`, roughly 1.4 ms of additional bandwidth time. The two terms cross when the batch holds about 107,000 cached tokens (`14 GB / 131 kB`) - for example, eight sequences of about 13,000 tokens each. Below that point the service is mostly buying weight reads; beyond it, context length governs the step. This one number reappears in the KV, quantization, and disaggregation sections.

### KV-cache accounting

For layers `L`, batch `B`, cached sequence length `S`, KV heads `H_kv`, head dimension `d`, and element bytes `b`:

`KV_bytes = 2 L B S H_kv d b`

The factor two represents keys and values. For heterogeneous sequence lengths, replace `B S` with the sum of cached tokens across sequences. Add page-table metadata, partial blocks, allocator reserve, prefix-cache entries, speculative branches, and temporary workspaces.

This equation makes architectural effects explicit. Grouped-query or multi-query attention reduces `H_kv`. Lower KV precision reduces `b`. Sliding-window attention caps effective `S` for selected layers. None of these is free: each changes model quality, kernels, or both.

For the running service the equation gives `2 * 32 * 8 * 128 * 2` = 131 kB per cached token. A median request holding 2,300 tokens of context therefore occupies about 300 MB, and after weights and workspace the 80 GB replica has room for roughly 450,000 cached tokens - about 200 median conversations. Every architectural lever above moves one factor of that product.

### Batch as a reuse and latency control

Increasing active decode batch can reuse weights and amortize launches, but it also:

- increases per-iteration duration;
- consumes more KV capacity;
- may raise collective message size;
- delays newly arrived work until a scheduling boundary;
- makes one iteration include more heterogeneous states.

Therefore the useful batch is SLO-dependent. Throughput-oriented jobs may use a large batch near saturation. Interactive traffic often needs a smaller iteration budget or a deadline-aware scheduler. Report the full throughput-latency curve rather than one maximum-throughput point.

### Long prompts and chunked prefill

A long prefill is efficient in isolation but can monopolize an accelerator long enough to stall active decodes. Chunked prefill divides uncached prompt tokens into bounded pieces and interleaves them with decode iterations. The chunk size trades:

- smaller generation gaps versus better prefill kernel efficiency;
- earlier service for other requests versus later first token for the long request;
- stable iteration shapes versus more scheduling and state transitions.

Chunk by a measured compute or time budget rather than assuming equal token chunks cost the same. Attention work changes with prefix length, and multimodal tokens or architecture-specific layers may have different cost.

### Prefix reuse changes prefill accounting

When a request reuses cached state for `S_hit` prefix tokens and has `S_new` uncached tokens, its prefill work is determined primarily by `S_new` plus attention against the reused history. A high token hit rate can reduce compute while consuming substantial cache memory. Distinguish:

- request hit rate;
- reused-token fraction;
- prefill FLOPs avoided;
- first-token time saved;
- bytes occupied and eviction cost.

Routing a request away from a useful prefix may cost more than waiting briefly for the node that holds it. Cache locality becomes a scheduling input rather than a storage statistic.

### Phase-specific optimization

**Prefill** commonly benefits from efficient attention, large matrix shapes, fused projections and epilogues, balanced long-prompt partitioning, prefix reuse, and topology that supports compute-heavy parallelism.

**Decode** commonly benefits from continuous batching, lower weight and KV bytes, efficient small-row matrix operations, paged attention, launch amortization, graph capture, and carefully chosen model parallelism. At long context, decode attention and KV placement deserve separate analysis from weight-dominated layers.

:::callout insight|Name the limiting resource by phase and shape
"The model is memory-bound" is too coarse. State whether the critical shape is short-prompt prefill, long-prompt attention, small-batch decode, long-context KV reads, a collective, or host scheduling. Each points to a different intervention.
:::

### Principal Interview Review

1. Why can the same model be compute-bound in prefill and bandwidth-bound in decode?
2. Derive the KV-cache memory equation and explain every omitted overhead.
3. How does decode batch size change arithmetic intensity and user latency?
4. When does long-context decode stop being mainly a weight-streaming problem?
5. Design a chunked-prefill policy for interactive and batch traffic.
6. How should a multimodal request change phase accounting and capacity planning?

### Answer Key

#### 1. Different phase bounds

Prefill applies weight matrices to many token rows, so tiled GEMMs reuse weights and can reach high arithmetic intensity. Decode adds one row per sequence; at small batch, weights and KV state are moved for relatively little new computation, so bandwidth and launch overhead dominate. Long prompts can make prefill attention IO-limited, and a large decode batch can move linear layers toward compute-bound. The answer must be shape-specific rather than an absolute label for the model.

#### 2. KV memory

For `L` layers, `B` equal-length sequences of `S` cached tokens, `H_kv` KV heads, dimension `d`, and `b` bytes per element, keys plus values require `2 L B S H_kv d b` bytes. With variable lengths, sum cached tokens instead of multiplying by `B S`. Add partially filled pages, allocator reserve, block tables, shared-prefix reference metadata, speculative branches, quantization scales, temporary attention workspace, and replication or sharding effects.

#### 3. Decode batch

More active sequences reuse each fetched weight across more rows and amortize launch and collective overhead, which raises arithmetic intensity and throughput. The iteration also becomes longer, consumes more KV memory, and may delay admission or token emission. Sweep batch under the actual latency SLO and context distribution; choose the largest budget that improves goodput rather than the batch with the highest unconstrained tokens per second.

#### 4. Long-context decode

Weight traffic is roughly stable per step, while each query reads a KV history that grows with context. When measured KV read time or attention compute approaches and then exceeds the weight-streaming term, attention controls the step. GQA, KV precision, sliding windows, cache layout, page locality, and attention parallelism then matter more. Use per-layer profiling and a bytes model; total model bandwidth alone cannot identify the crossover.

#### 5. Chunked prefill

Reserve a decode budget that maintains the declared maximum token gap, then spend remaining per-iteration compute or token budget on one or more prefill chunks. Adapt chunk size to decode pressure, estimated chunk time, prompt progress, priority, and pipeline shape. Age long prompts so they cannot starve, reserve KV before executing a chunk, and measure both cadence improvement for active decodes and added first-token latency for chunked requests.

#### 6. Multimodal accounting

Add preprocessing and encoder time, memory, and queueing as a separate phase. Account for media-size distributions and the number of model tokens produced by the encoder, which affect later prefill and KV. The encoder may need different hardware or batching, and large media payloads affect network and host memory. Capacity tests must preserve correlations between media size, text length, output length, and model variant; text token counts alone understate cost.

## KV Cache, Paging, and Prefix Reuse

LEAD: KV state is the working set of autoregressive inference. Its lifetime follows requests, not batches, so allocation, sharing, eviction, placement, and security are first-class serving decisions.

The previous section priced KV by the byte; this section manages its lifetime. On the running replica, 131 kB per token means one long conversation can hold hundreds of megabytes hostage, and the shared 500-token system prefix costs about 65 MB per copy - large enough that whether it is shared or duplicated is a capacity decision, not a detail.

### Logical state and physical storage

Each cached token contributes a key and value vector at every attention layer. Logically, a sequence sees positions in order. Physically, those positions do not need to occupy one contiguous allocation. A serving engine can divide KV into fixed-size token blocks, allocate physical blocks on demand, and translate logical block numbers through a per-sequence table.

Paging avoids reserving the maximum possible output for every request and avoids copying the entire cache whenever a sequence grows. It also enables prefix sharing and copy-on-write branching. The attention implementation must accept block tables or an equivalent indirection, so memory management and kernel layout are designed together.

The term **page** may refer to a token block across selected layers, a per-layer allocation, or an engine-specific unit. Always state the unit, byte size, and placement. The important contract is stable logical order over movable or non-contiguous physical storage.

### Fragmentation

Three forms matter:

- **Internal fragmentation** is unused space inside the last allocated block of a live sequence.
- **External fragmentation** is free memory that cannot satisfy an allocation because of size or placement constraints.
- **Reserved slack** is intentionally unallocated or pre-reserved capacity for growth and failure recovery.

With equal fixed-size blocks, external fragmentation in the KV pool is greatly reduced, but internal waste is bounded by at most one partially filled block per independent sequence or branch. Metadata, alignment, and separate pools can reintroduce constraints. Report effective usable KV capacity from the live allocator, not free bytes from a device-wide monitor.

### Block-size tradeoffs

Smaller token blocks reduce tail waste, improve the granularity of prefix sharing, and make cancellation reclaim memory sooner. They increase block-table size, allocator operations, address translation, scatter in physical access, and scheduling metadata.

Larger blocks reduce metadata and can improve contiguous access, but waste more space on short or terminated sequences and require longer exact prefix matches before a whole block is reusable. The optimum depends on the sequence-length distribution, layer layout, attention kernel, and reuse pattern. Benchmark block sizes under memory pressure, not only with an empty cache.

### Allocation and reservation invariants

A robust allocator maintains:

- no live sequence or in-flight kernel references a freed block;
- reference counts protect shared prefixes and speculative branches;
- copy-on-write occurs before a branch mutates shared state;
- reservation makes the next scheduled iteration memory-safe;
- cancellation and errors eventually release every owned reference;
- block tables are updated before kernels consume them, with explicit ordering;
- accounting attributes bytes to model, tenant, request, and cache class;
- migration or restore preserves token order, position state, dtype, and model identity.

Reserve before launch rather than discovering out-of-memory after some requests in a batch have advanced. For uncertain output length, reserve a bounded horizon and repeat admission checks. A reservation is a promise to the scheduler, not necessarily immediate physical zero-filling.

### Prefix cache identity

KV for a prefix is reusable only when the computation that produced it is identical under the product contract. A key commonly includes:

- model weights and engine-compatible model version;
- tokenizer and exact token IDs;
- chat template and special-token policy;
- adapter or prompt-tuning state;
- positional encoding configuration and absolute offset;
- attention mode, relevant multimodal embeddings, and cache dtype;
- tenant or security namespace and cache salt;
- any hidden input that changes activations.

Text equality is insufficient because normalization or templates can change token IDs. Token equality can still be insufficient when adapter, image embedding, position, or model version differs. Cache identity should be a versioned schema with test vectors.

### Lookup structures and partial hits

A trie or radix tree naturally represents shared token prefixes. A block-hash index maps full token blocks and their parent identity to cached blocks. Both can return the longest valid prefix rather than an all-or-nothing hit.

Insert only committed state. A cancelled or speculative branch may have computed tokens that were never part of the accepted sequence. Mark blocks immutable once published for reuse; append new private blocks and publish them only after the corresponding tokens and semantic state are committed.

Measure four distinct outcomes:

1. request-level hit rate;
2. reused-token fraction;
3. prefill time or FLOPs avoided;
4. net first-token improvement after lookup, routing, and transfer.

A high hit rate on tiny prefixes may save little. A lower hit rate on long system prompts can be far more valuable.

The running service makes the distinction concrete. Its shared 500-token system prefix costs about 65 MB to retain and saves 500 tokens of prefill on nearly every request. A per-user greeting of a few tokens may hit just as often while saving almost nothing and fragmenting the pool with tiny published blocks.

### Eviction as value density

Least-recently-used eviction is a baseline, not a universal optimum. The value of an entry depends on expected reuse probability, recomputation cost, bytes, transfer locality, and the opportunity cost of displacing active KV.

A practical priority score may increase with saved prefill time and recent or known future reuse, and decrease with bytes and memory pressure. Pinning known system prompts can help, but permanent pins can crowd out active requests. Set budgets by cache class and expose why an entry survived or was evicted.

Prefix caching competes with decode capacity. Under pressure, retaining a reusable prefix may force rejection of a live request or reduce batch size. Admission and eviction need one memory objective, not separate local heuristics.

### Routing and cache locality

In a replicated fleet, the router chooses between queue delay and cached state. Define an estimated completion score such as:

`cost = queue_time + transfer_time + uncached_prefill_time + policy_penalty`

Sticky routing maximizes locality but can create hot spots. Pure least-loaded routing destroys reuse. A cache-aware router considers both, places common prefixes on enough replicas, and falls back when a cache owner is unhealthy or overloaded.

Do not transfer a prefix merely because it exists elsewhere. Compare transfer time and network contention with recomputation. For a short prefix or a compute-rich prefill worker, recomputing may be cheaper and simpler.

### Offload and migration

KV can be tiered across accelerator memory, host memory, local storage, or remote memory. Offload is useful when pause time is long enough to hide transfer and recomputation is more expensive. It is harmful when every decode step faults pages across a slow link.

For a state object of `X` bytes over sustainable bandwidth `BW`, a lower bound is `X / BW`, plus registration, serialization, queueing, and synchronization. Migration should transfer only live and required blocks, preserve reference ownership, and cancel cleanly if the destination can no longer admit the request.

A session cache and an active decode cache have different latency needs. Separate tiers and policies so inactive conversations cannot evict the working set of interactive generation.

### Security and side channels

Cross-request reuse creates an information boundary. A tenant must not infer another tenant's prompt through timing, hit metadata, or shared identifiers. Namespace or salt cache keys, authorize sharing explicitly, avoid returning cache-hit details to untrusted callers, and consider padding or disabling cross-tenant reuse for sensitive workloads.

Deletion requirements propagate to cache copies, offloaded tiers, and replicas. Logging raw token keys or prefix hashes can itself leak information when prompts have low entropy. Treat cache metadata under the same retention policy as model inputs.

:::callout pitfall|A KV hit is not automatically a win
The useful metric is first-token time or cost saved after lookup, routing, transfer, memory displacement, and security policy. Cache reuse that fragments capacity or creates a hot node can lower fleet goodput.
:::

### Principal Interview Review

1. Why does paged KV reduce waste, and what costs does it introduce?
2. How would you choose token-block size?
3. State the allocator invariants required for prefix sharing and speculative branches.
4. What must be included in a safe prefix-cache identity?
5. Design a cache-aware routing and eviction policy.
6. When should the system transfer, offload, recompute, or discard KV state?

### Answer Key

#### 1. Paging tradeoff

Paging allocates fixed-size blocks as sequences grow, so requests do not reserve maximum length and growth does not copy a contiguous cache. Tail waste is bounded by partially filled blocks, and blocks can be shared or released independently. Costs include block tables, allocation operations, address translation in attention, less contiguous physical access, metadata synchronization, and kernel complexity. The win must be measured with the actual block layout and memory pressure.

#### 2. Block size

Sweep sizes against the production sequence and reuse distributions. Small blocks reduce internal fragmentation, reclaim sooner, and expose finer prefix hits; large blocks reduce metadata, allocation frequency, and translation overhead and may improve locality. Measure usable KV capacity, attention latency across context lengths, allocator CPU cost, block-table traffic, cache hit value, and cancellation reclamation. Choose by end-to-end goodput and SLOs, not fragmentation alone.

#### 3. Sharing invariants

Published shared blocks are immutable and reference-counted. A branch increments references, performs copy-on-write before divergence, and publishes only accepted state. Scheduling reserves private growth blocks before launch. Freeing waits until no sequence or in-flight kernel can reference the block, with ordered block-table updates. Cancellation, rejection, and partial failure release every reference exactly once, and accounting remains attributable.

#### 4. Cache identity

Use exact token IDs plus model and tokenizer versions, chat-template and special-token policy, adapter or prompt state, position configuration and offset, attention/cache mode and dtype, multimodal inputs that affect activations, and an authorization namespace or salt. Include any hidden feature that changes computed K/V. Version the identity schema so rollout cannot mix incompatible cache entries.

#### 5. Routing and eviction

Estimate completion cost on candidate replicas from queue delay, reusable tokens, transfer or recompute time, memory pressure, and tenant policy. Replicate very hot prefixes enough to avoid affinity hot spots and use a no-cache fallback. Eviction should rank saved recompute time or known future value per byte while preserving active-request reserve and per-tenant limits. Continuously compare predicted savings with actual first-token improvement.

#### 6. KV placement decision

Keep active working state on the accelerator. Transfer when another node can finish sooner after including network and admission cost. Offload when expected idle time can hide the round trip and recomputation is expensive. Recompute when the prefix is small, transfer links are contended, or local prefill is cheap. Discard when future reuse probability times saved work is below storage and displacement cost. Enforce deletion and tenant-isolation policy regardless of performance.

## Scheduling, Batching, and Admission Control

LEAD: The scheduler allocates compute time, KV memory, and latency slack across requests whose work is revealed incrementally. It is a policy engine with explicit invariants, not a queue attached to a GPU.

With phase costs and KV lifetime established, the scheduler is where they collide. Every iteration must divide the replica between the token cadence promised to active requests and the first-token latency promised to queued ones - the running service's two SLOs are exactly that tension, and its roughly 200-conversation KV capacity is the budget being divided.

### Iteration-level scheduling

Autoregressive requests finish at different output lengths. A static batch keeps completed slots idle or delays their responses until the longest sequence finishes. Iteration-level scheduling rebuilds the active batch at token or bounded-chunk boundaries. Finished and cancelled requests leave; newly admitted work can enter.

:::diagram continuous_batching|Continuous batching fills freed decode slots over time. The scheduler gains efficiency but must decide admission, phase mixing, fairness, and shape compatibility at every boundary.

An iteration may contain decode tokens, prefill chunks, encoder work, or verification tokens. The engine exposes feasible work units; the scheduler chooses a set under budgets for tokens, sequences, memory, execution time, graph shape, and distributed compatibility.

### Scheduler state

Track per request:

- phase and tokens already processed;
- prompt remainder and generated length;
- deadline, priority, tenant weight, and arrival time;
- estimated next-iteration and remaining work;
- current KV blocks, reserved growth, and prefix ownership;
- model, adapter, sampling, grammar, and speculative mode;
- compatible engine/graph variants and device placement;
- cancellation and client-backpressure state.

Track global state such as free and reserved KV blocks, active batch composition, queue age, measured iteration-time curves, cache locality, device health, adapter residency, and SLO risk.

### Token and time budgets

A maximum sequence count is a weak batching control because one prefill token and one long-context decode token have different cost. Use one or more measured budgets:

- total uncached prefill tokens;
- active decode sequences grouped by context bucket;
- predicted iteration time;
- KV blocks allocated now and at the next horizon;
- communication volume or pipeline microbatches;
- verification tokens for speculation.

Cost estimates need not be perfect, but they should be calibrated from telemetry and bounded conservatively near deadlines. When estimates drift after an engine or model update, the scheduler can violate SLOs even though kernels improved.

### Phase mixing and chunked prefill

Decode-first scheduling protects token cadence but can starve new prefills under sustained generation. Prefill-first scheduling reduces queue delay for new arrivals but can create generation stalls. Chunked prefill bounds interference and creates a tunable compromise.

A robust policy first reserves enough decode work to protect imminent deadlines, then fills remaining predicted iteration time with prefill chunks. Aging increases the priority of waiting prefills. Batch jobs may use a separate queue and consume explicitly allocated slack rather than opportunistically overwhelming interactive traffic.

Pipeline-parallel execution may prefer iterations with similar compute to reduce bubbles. Chunk size should therefore consider stage balance as well as a single-device token budget.

The running service shows why the compromise is necessary. An uninterrupted 2,000-token prefill occupies the replica for tens to low hundreds of milliseconds, which alone can exceed the 100 ms inter-token budget of every active decode. Chunking that prompt into a few hundred tokens per iteration bounds each gap while adding modest total prefill time; the long request pays a slightly later first token so that dozens of active streams keep their cadence.

### Fairness and priorities

FIFO is predictable but allows a huge prompt to delay many small requests. Shortest-remaining-processing-time improves mean latency but can starve large jobs and relies on unknown output length. Earliest-deadline-first protects explicit deadlines when service time is estimable. Weighted fair service can allocate token or compute time across tenants.

Charge service in a resource-correlated unit, not requests. One long-context decode step can cost far more than a short-context step; one prefill can consume thousands of tokens. Maintain deficit or virtual-time accounting and add aging so low-priority work eventually progresses.

Priority inversion appears when a high-priority request depends on a low-priority resource: an adapter load, a shared prefix owner, a pipeline peer, or memory held by a paused request. The scheduler must either inherit priority, preempt or evict safely, or route around the dependency.

### Admission control

Admission determines whether a request can enter without breaking promises already made. Estimate:

- prompt and encoder work;
- minimum and worst-credible KV growth;
- model and adapter residency;
- deadline slack and queue position;
- compatible replica capacity;
- tenant quota and global reserve.

Possible outcomes are admit, defer, route, downgrade under an explicit policy, truncate if the API allows it, or reject quickly with a retry signal. Do not admit unlimited work into an invisible queue. An honest overload response protects existing requests and gives upstream systems a chance to shed or reroute load.

Reserve memory for at least the next scheduling horizon. Reserving maximum output for every request wastes capacity; reserving nothing risks mid-generation failure. A probabilistic growth model can improve utilization, but keep a hard safety reserve and define what happens when predictions are wrong.

### Cancellation and backpressure

Cancellation is a capacity feature. Propagate it from disconnected client to gateway, queue, scheduler, worker, and KV allocator. If a kernel cannot be interrupted, mark the request and release resources at the next safe boundary. Avoid publishing unaccepted speculative state or streaming tokens after cancellation.

A slow client can make stream buffers grow while the GPU continues generation. Set bounded buffers and a policy: pause the request, reduce its scheduling share, spill within limits, or cancel. Paused requests still occupy KV memory, so waiting indefinitely is not free.

### Scheduler pseudocode

```python
def schedule_tick(state, target_ms):
    candidates = state.ready_requests()
    selected = []
    predicted_ms = 0.0

    for req in deadline_and_fairness_order(candidates):
        unit = next_feasible_unit(req, state)
        if not reserve_kv(unit, state.kv_pool):
            continue
        if predicted_ms + estimate_ms(unit, selected) > target_ms:
            release_reservation(unit)
            continue
        selected.append(unit)
        predicted_ms = estimate_ms(selected)

    return group_by_engine_shape(selected)
```

The hard work lives in cost estimation, fairness ordering, memory reservation, graph compatibility, phase mixing, and rollback when a selected distributed peer becomes unavailable.

### Overload and headroom

Queueing latency grows nonlinearly near saturation, and variable output lengths make service time heavy-tailed. Operate with headroom sized for bursts, node loss, and estimation error. The right utilization target is the highest level that preserves goodput and recovery reserve, not a universal percentage.

During overload, apply a declared hierarchy: protect in-flight interactive sequences, stop low-value prefills, enforce tenant quotas, shed batch traffic, reduce optional speculative or best-of work, and reject new requests before memory is exhausted. Quality-changing degradation requires product authorization and telemetry.

:::callout decision|Make the objective executable
"Maximize useful tokens subject to deadline, fairness, memory, and semantic constraints" can be translated into ordering, budgets, and invariants. "Keep the GPU busy" cannot decide whom to delay or reject.
:::

### Principal Interview Review

1. Design an iteration-level scheduler for mixed interactive and batch traffic.
2. How do you prevent long prefills from breaking token cadence without starving them?
3. What should admission control do when KV memory is nearly full?
4. How would you measure and enforce fairness across tenants with different request shapes?
5. Explain cancellation and slow-client handling end to end.
6. Why can intentionally lower accelerator utilization improve fleet goodput?

### Answer Key

#### 1. Mixed scheduler

Maintain separate objective classes but one global capacity and memory governor. At each boundary, reserve imminent interactive decodes by deadline, add aged interactive prefills in bounded chunks, and spend declared slack on batch work using weighted service accounting. Predict iteration time and KV growth, group compatible shapes, and reserve before launch. Enforce tenant quotas, propagate cancellation, and expose quick rejection during overload. Evaluate goodput and starvation, not only average throughput.

#### 2. Long prefills

Split them into cost-bounded chunks. Reserve a decode time budget that keeps active requests below the maximum token gap, then schedule chunks in remaining capacity. Increase a waiting prompt's virtual priority with age and guarantee a minimum service share so continuous decode load cannot starve it. Adapt chunk size to measured phase cost and pipeline balance; report the added first-token latency paid by the long request.

#### 3. Near-full KV

Stop admitting requests whose reserved growth would consume the safety margin. Reclaim cancelled and expired state, evict low-value reusable prefixes or inactive sessions, route to a replica with capacity, and shed lower-priority queued work according to policy. Protect memory promised to active requests. Do not rely on an eventual allocator failure; return a bounded retry or overload response before the pool is exhausted.

#### 4. Fairness

Charge tenants in predicted or measured compute-time/KV-time units rather than raw request count. Use weighted deficit or virtual-time scheduling, cap burst debt, and age waiting work. Track service share, queue delay, SLO attainment, rejections, and starvation by tenant and shape class. Account for shared-prefix savings consistently so a cache hit does not accidentally grant unlimited priority or penalize the tenant twice.

#### 5. Cancellation and backpressure

The gateway propagates cancellation with request identity; queues remove unstarted work; the scheduler marks in-flight work and excludes it at the next safe boundary; the worker stops sampling and streaming; and the allocator releases private and shared references exactly once. Stream buffers are bounded. A slow client is paused, deprioritized, or cancelled by policy, with its KV occupancy still charged. Races are tested around token commit and speculative rollback.

#### 6. Lower utilization

Near saturation, queue delay and tail risk rise sharply, bursts cannot be absorbed, and a node failure can overload the remainder of the fleet. Huge batches may maximize hardware throughput while violating first-token and cadence SLOs. Leaving headroom can admit urgent work sooner, reduce generation stalls, preserve failure reserve, and therefore increase the number of requests completed within contract even though average device utilization is lower.

## Sampling, Structured Output, and Speculative Decoding

LEAD: Decoding is part of the model's externally visible semantics. Acceleration may reorganize computation, but it must preserve the declared probability distribution, constraints, random-number mapping, and termination behavior unless approximation is explicitly allowed.

Scheduling decided when a request runs; this section governs what its tokens mean and how to produce them faster without changing them. The running service's 4.7 ms weight-streaming floor is the number to beat: any acceleration that commits more than one token per floor-priced step is attacking the sequential bottleneck itself.

### The token-selection pipeline

A typical step transforms model logits through an ordered pipeline:

1. apply vocabulary restrictions and bad-token masks;
2. apply repetition, frequency, presence, or custom penalties;
3. apply temperature or other score transformation;
4. truncate with top-k, nucleus top-p, minimum probability, or a product policy;
5. normalize and sample with a defined random stream;
6. test stop tokens, stop strings, grammar acceptance, and length limits;
7. commit token and KV state, detokenize incrementally, and stream.

Order matters because these operations generally do not commute. A fused implementation must match the reference order, tie rules, NaN behavior, and dtype policy. Tokenization creates additional edge cases: a stop string may span tokens, and a token byte sequence may be incomplete until later output arrives.

Greedy decoding is the special case that selects an argmax under deterministic tie policy. Sampling correctness means equality of the declared distribution, not equality of one token trace unless seeds, batching, and random-number mapping are also fixed.

### Randomness and reproducibility

Counter-based random generators can derive values from a key such as request seed, logical output position, branch, and sampling substep. This makes randomness independent of physical thread scheduling and easier to preserve across dynamic batches.

Exact reproducibility across engine versions may still be impossible if floating reductions, truncation boundaries, or probability transforms change. Define a level:

- distributional equivalence;
- same result for fixed seed on one engine version;
- same result across batch compositions;
- bitwise identity across versions and hardware.

Do not promise the strongest level unless the whole logits pipeline, numeric path, and RNG mapping are controlled and regression-tested.

### Structured and constrained decoding

Structured output restricts the next-token set according to a grammar, schema, regular expression, or application state. A common engine compiles the constraint into a finite-state representation and maintains a state per sequence. At each step it finds allowed tokens and masks the rest before sampling.

The hard parts are often outside the model:

- tokenizer tokens may contain multiple characters or partial UTF-8 sequences;
- a grammar state can have a large or tiny allowed set;
- dynamic masks can break graph or batch compatibility;
- state transitions and detokenization must agree;
- fallback or repair changes semantics and latency.

Precompute token-to-transition tables where possible, cache grammar artifacts by versioned identity, and batch sequences with compatible constraint work without starving rare grammars. Measure compile latency separately from per-token mask cost.

### Beam search, best-of, and branch state

Beam search maintains multiple scored hypotheses and selects expansions under a length-normalized objective. Best-of samples several candidates and returns a selected one. Both multiply KV state and scheduling work. Paged copy-on-write can share the common prefix, but diverged branches need private blocks.

Define whether non-returned candidates count against token quotas and billing, how stop rules apply, and whether streaming is possible before the winner is known. Prune and release branches promptly. A scheduler that counts one API request rather than its active branches can under-reserve memory badly.

### Speculative decoding

Speculative decoding uses a cheaper proposal process to reduce sequential target-model steps. A draft proposes `gamma` tokens. The target evaluates the proposed positions in parallel and commits an accepted prefix plus, in common formulations, one token selected from target information at the stopping point.

:::diagram speculative_decoding|A proposal is useful only when the target can verify several positions more cheaply than performing the same number of ordinary sequential decode steps.

Proposal sources include a smaller standalone model, a quantized or pruned target, early-exit layers, multiple-token prediction heads, retrieval or n-gram matches, or a tree of candidates. They differ in draft cost, acceptance, memory, training requirements, and how well they batch.

### Exact stochastic verification

Let target probability be `p(x)` and proposal probability be `q(x)` for a proposed token `x`. Accept the proposed token with probability:

`a(x) = min(1, p(x) / q(x))`

If rejected, sample from the normalized positive residual:

`r(x) proportional to max(0, p(x) - q(x))`

This decomposition preserves the target distribution when probabilities are computed under matching histories and the correction is implemented exactly. Guard zero proposal probability, normalize stably, and apply identical target logits processors before constructing `p`. Vocabulary sharding may require distributed normalization or candidate exchange.

```python
def verify_token(proposed, p, q, uniform):
    qx = max(q[proposed], 1e-30)
    accept = min(1.0, p[proposed] / qx)
    if uniform <= accept:
        return proposed, True
    residual = maximum(p - q, 0.0)
    residual = residual / residual.sum()
    return sample(residual), False
```

Top-k, top-p, temperature, penalties, and constraints define the target distribution. Applying them differently in draft and verification can be legal if the exact acceptance/correction construction uses the resulting `p` and `q`, but it changes acceptance and implementation complexity. An approximation that skips correction must be declared and evaluated as a quality change.

### Expected progress and break-even

If `A` is the number of accepted draft tokens, the tail-sum identity gives:

`E[A] = sum from k=1 to gamma of Pr(A >= k)`

The conditional acceptance probability changes with position and context, so estimate this from traces rather than assuming independent identical trials. Useful committed tokens per cycle may include an additional target token when all or part of the proposal is accepted, depending on algorithm details.

A simple break-even comparison is:

`cycle_cost / expected_committed_tokens < ordinary_target_step_cost`

Cycle cost includes draft, target verification, correction sampling, branch KV work, scheduler gaps, and any lost target batch capacity. Verification latency is shape-dependent and does not usually grow linearly with `gamma`, which is why speculation can win.

For the running service at small batch, an ordinary decode step is bounded near 4.7 ms by weight streaming. A 1B-class draft with about 2 GB of weights has a streaming floor near 0.7 ms per draft step, and target verification of several proposed positions still costs roughly one target-weight pass. Four draft steps plus one target verification therefore have an idealized floor near `4 * 0.7 + 4.7 = 7.5 ms`. If the cycle commits three output tokens on average, ordinary decoding would spend about `3 * 4.7 = 14.1 ms` on the same output work, so the idealized speedup is about 1.9x. Draft arithmetic, correction sampling, scheduler gaps, and extra KV reduce that gain; poor acceptance or lost target batch capacity can erase it.

### Choosing proposal length

Long proposals amortize a target call if acceptance remains high, but waste draft and verification work after early rejection and consume more temporary KV. Choose `gamma` from:

- recent acceptance and draft confidence;
- target/draft cycle-time curves by batch and context;
- queue pressure and target batch opportunity cost;
- memory reserve and graph variants;
- domain, temperature, grammar, and request class.

Use a bounded policy or lookup table before a complex predictor. Optimize measured goodput and tail latency, not acceptance alone. The best length may be zero when the target is already efficiently batched.

### Tree and multi-token proposals

A tree proposes alternatives at several future positions so target verification can recover useful progress even when the top draft path is wrong. Multiple-token prediction heads propose future tokens from a shared target representation. These approaches can reduce separate draft cost but require branch packing, attention masks, verification logic, and often model-specific training.

The scheduler sees a variable number of verification tokens per request. Grouping incompatible trees can waste padding, while separating every shape fragments batches and graph caches. Restrict to a small family of supported tree shapes unless telemetry justifies more variants.

### Interaction with batching and KV

Different sequences accept different prefix lengths, so they advance unevenly. Commit accepted KV, discard or recycle rejected branch state, update logical positions, and ensure random streams follow logical output rather than physical verification slots.

At high target batch, ordinary decode may already reuse weights well. Draft compute can contend with the target or reduce its resident batch because of extra weights and KV. At low-latency batch, eliminating target synchronization steps is often more valuable. Benchmark the joint scheduler with real arrival pressure.

:::callout pitfall|Acceptance is not the objective
A proposal with excellent agreement can lose if it is expensive, reduces target batch capacity, creates awkward verification shapes, or worsens queueing. Report committed output tokens per end-to-end second under the same SLO and semantics.
:::

### Principal Interview Review

1. Why must the order of logits processors be part of the API contract?
2. Design deterministic RNG mapping for continuous batching and cancellation.
3. How does constrained decoding affect the serving system beyond token masking?
4. Derive speculative acceptance and the rejection correction at a high level.
5. How would you choose proposal length online and decide when to disable speculation?
6. Compare standalone drafts, multi-token heads, and tree proposals.

### Answer Key

#### 1. Processor order

Penalties, masks, temperature, and truncation generally do not commute, so reordering changes probabilities and possibly the chosen token. The contract must fix the pipeline, numeric and tie behavior, stop handling, and constraint application. A fused or distributed path is correct only if it matches that reference or the product explicitly accepts a new distribution.

#### 2. RNG mapping

Use a counter-based generator keyed by request seed and logical output position, plus branch or subdraw indices for rejection and best-of. Do not key randomness to batch slot, CUDA thread, or scheduling iteration because regrouping would change results. Commit counters only with logical tokens, discard uncommitted branch draws by identity, and define whether retries reproduce or intentionally resample. Test across batch reshuffles, cancellation neighbors, and speculative accept/reject paths.

#### 3. Constraints

Constraints require compilation and caching of grammar state, per-request state transitions, tokenizer-aware allowed-token computation, dynamic masks, and defined invalid-state behavior. They change CPU work, batch compatibility, graph shapes, sampling cost, and sometimes token-length distribution. The scheduler must account for compile latency and per-step cost, while semantic tests cover UTF-8 boundaries, stop sequences, schema completion, and cancellation.

#### 4. Exact speculation

For a proposed token with target probability `p(x)` and draft probability `q(x)`, accept with `min(1, p(x)/q(x))`. Accepted mass is the overlap between the two distributions. If rejected, sampling from the normalized positive residual `max(0, p-q)` supplies exactly the target mass not covered by accepted proposals. Apply the construction conditionally at each matching history, handle zero and normalization stably, and use the exact target postprocessed distribution.

#### 5. Proposal control

Estimate expected committed tokens and full cycle cost for a small supported set of proposal lengths by batch, context, domain, temperature, and recent acceptance. Select the length with best predicted SLO-constrained goodput subject to memory reserve. Disable when draft plus verification per committed token exceeds ordinary target decode, when target batches are already efficient, acceptance collapses, memory is tight, or unsupported sampling/grammar semantics would require approximation.

#### 6. Proposal designs

A standalone draft is flexible and needs no target modification, but carries separate weights/KV and consumes scheduling capacity. Multi-token heads share target computation and can draft cheaply, but require architecture-specific training and may have correlated errors. Trees provide alternative future paths and recover from top-path errors, but increase verification tokens, branch state, masks, and shape complexity. Compare end-to-end cycle cost, acceptance/progress, memory, training and deployment burden, batching, and exactness.

## Parallel, Replicated, and Disaggregated Serving

LEAD: A serving fleet places model parameters, KV state, and request phases across failure domains. Parallelism is useful only when its memory or throughput gain exceeds communication, synchronization, and scheduling cost at the target shapes.

One replica is no longer enough: a model may not fit one device, traffic exceeds one accelerator, and phases interfere. This section places weights, KV, and phases across devices and pools. Two communication facts drive everything below. A **collective** is a synchronized exchange, such as an all-reduce, whose time is roughly a fixed latency per step plus bytes divided by link bandwidth, and it completes at the pace of the slowest participant. Decode sends small messages every few milliseconds, so the latency term and the slowest peer dominate exactly where training-style large-message efficiency does not apply.

### Replicas and model-parallel groups

**Data-parallel serving replicas** hold equivalent model state and process different requests. They scale aggregate capacity and isolate failures but do not make one request faster unless the request can be divided at a higher level.

**Tensor parallelism** shards large layer operations across devices and introduces collectives in each layer. It reduces parameters per device and can lower single-request compute time, but small decode messages may underutilize links and every step waits for the slowest peer.

**Pipeline parallelism** assigns layer ranges to stages. It reduces per-device weights and can overlap microbatches, but latency includes stage traversal and bubbles depend on the number and uniformity of microbatches. Dynamic continuous batches complicate stable pipeline schedules.

**Expert parallelism** places mixture-of-experts experts across devices. It reduces resident expert weights per device and increases aggregate capacity, but token routing creates all-to-all traffic and load imbalance. Hot experts and small per-expert token counts can dominate decode.

Part V develops these mechanisms in depth. For serving, the decision is driven by request latency, active batch, KV placement, link topology, and the model's fit, not training precedent.

### Choosing the smallest group

Use the smallest model-parallel group that fits weights, KV reserve, runtime workspaces, and failure headroom while meeting latency. Increasing tensor-parallel degree can reduce local compute but adds communication and consumes more devices per replica, reducing the number of independent batches the fleet can run.

A useful comparison includes:

- per-device weight and KV bytes;
- prefill and decode latency curves by batch and context;
- collective time and exposed synchronization;
- replica count and queueing under the arrival process;
- fault blast radius and repair time;
- cost per good request or output token.

The fastest isolated request configuration may have worse fleet goodput because it creates fewer replicas and longer queues.

The running service's 14 GB model fits comfortably on one 80 GB device, so tensor parallelism is a choice, not a necessity. Splitting across two devices moves the weight-streaming floor from 4.7 ms toward 2.4 ms but inserts collectives into all 32 layers of every step and consumes two devices per replica. Unless the fleet is latency-starved at small batch, two independent replicas usually deliver more goodput than one two-way group.

### Topology-aware placement

Map communication-heavy peers onto the fastest available links and keep the mapping stable enough for engine artifacts and cache locality. Know which links share bandwidth, which paths cross sockets or switches, and whether collectives contend with KV transfer or storage traffic.

Topology faults are performance faults before they become hard errors. A degraded link or throttled peer slows every synchronized device. Record per-rank timing and collective skew; aggregate GPU utilization can hide one straggler holding the group.

Heterogeneous accelerators require separate performance models and engine variants. Route shapes to the hardware that meets their phase objective, but include transfer, queue, and fragmentation costs. A nominally faster device can lose if its pool is saturated or lacks the required adapter/precision path.

### Prefill-decode disaggregation

Colocated workers run both phases and retain KV locally. This is simple and avoids a phase-boundary transfer, but long compute-heavy prefills can interfere with cadence-sensitive decode and both phases must share one parallelism and capacity plan.

Disaggregation assigns prefill and decode to different worker pools. It allows phase-specific hardware, batch, parallelism, and scaling, and it isolates interference. The handoff is KV state plus request metadata.

:::diagram disaggregation|Prefill and decode pools specialize independently; the price is a KV handoff whose bytes, protocol, and queueing must be modeled before committing to the architecture.

For transferred KV bytes `X` and sustainable path bandwidth `BW`, a lower bound is:

`t_handoff >= X / BW + fixed transfer overhead`

The actual path includes source readiness, registration, serialization or layout conversion, network queueing, destination placement, and synchronization. Long prompts create more state to transfer but also more prefill work that specialization may save.

For the running service, a 2,000-token prompt hands off about 260 MB of KV (`2000 * 131 kB`). Over the assumed 50 GB/s path that is roughly 5 ms plus protocol overhead - cheap next to the prefill it frees the decode pool from repeating. The same transfer over a contended or slower path can erase the benefit, which is why the bound is a starting point and not a verdict.

:::callout decision|Disaggregate only after modeling the handoff
The architecture wins when phase specialization, independent scaling, and interference isolation exceed KV transfer, additional queueing, network contention, and operational complexity at the real prompt/context distribution.
:::

### Handoff protocol

A correct handoff identifies model/adapter version, token range, position state, KV dtype and layout, source blocks, destination allocation, and ownership. One safe sequence is:

1. destination admission reserves capacity;
2. source finishes and marks transferable committed blocks;
3. data moves with checksums or transport integrity;
4. destination confirms visibility and compatible metadata;
5. ownership transfers or source references are released;
6. decode becomes schedulable.

Retries need idempotent request and transfer identifiers. If the destination fails after receiving some blocks, cleanup must not free source state prematurely or leak destination allocations. Avoid a protocol that can generate the first token twice after a retry.

### Routing with state affinity

Routing decisions occur at several levels:

- gateway to model/version pool;
- request to replica or prefill group;
- prefill result to decode group;
- resumed session to existing or migrated KV;
- expert or adapter work within a group.

Least-loaded routing is insufficient when state has locality. Estimate completion time including queue, cache hit, adapter load, transfer, and phase cost. Use bounded stickiness: prefer the state owner while its delay advantage remains positive, then migrate or recompute rather than creating an unbounded hot spot.

### Independent scaling and backpressure

In a disaggregated system, prefill and decode capacity must balance in units that reflect the workload. Request rate alone is misleading; prefill load follows uncached input tokens and encoder work, while decode load follows generated tokens and retained contexts.

If prefill outruns decode, completed KV accumulates and consumes transfer buffers or destination memory. If decode outruns prefill, expensive decode devices idle. Add bounded inter-stage queues, destination reservations before prefill commitment, and autoscaling signals based on predicted phase work and SLO risk.

### Fault domains and recovery

A model-parallel group generally fails as a unit when one peer disappears. Drain new admissions, fail or retry affected requests according to commit state, and replace the group. Replicas provide capacity redundancy; model-parallel ranks do not.

Replaying a prefill can reconstruct KV if the input and model version are available, but it increases first-token latency and may violate deletion or privacy policy. Replicating KV improves recovery but doubles state traffic and memory. Choose by SLO, prompt cost, and failure frequency.

For streaming responses, retries after emitted tokens are delicate. The service can resume only if it can reproduce exact state and avoid duplicate bytes, otherwise it should terminate with a clear partial-response error. Exactly-once token streaming across arbitrary failures is an application protocol, not an automatic property of the model engine.

### Principal Interview Review

1. Why can increasing tensor-parallel degree reduce fleet goodput?
2. How would you choose between replication, tensor parallelism, and pipeline parallelism for serving?
3. Derive the break-even terms for prefill-decode disaggregation.
4. Design a correct and retry-safe KV handoff protocol.
5. How should routing balance queue delay against state locality?
6. What changes in autoscaling and failure handling after disaggregation?

### Answer Key

#### 1. Tensor-parallel degree

More shards reduce weights and compute per device but add per-layer collective latency and synchronize on the slowest peer. Small decode messages may use links poorly. A larger group consumes more GPUs per replica, so the fleet runs fewer independent batches and may queue longer. Compare SLO-constrained request goodput for the whole fleet, including collective skew and failure blast radius, rather than single-request kernel speed.

#### 2. Parallelism choice

Start with the smallest group that fits weights, KV reserve, runtime workspace, and safety margin. Prefer replicas for aggregate independent traffic. Add tensor parallelism when the model does not fit or its reduced local compute offsets collectives at target batch. Consider pipeline parallelism when layer partitioning fits the topology and enough compatible microbatches exist to control bubbles. Model queueing, replica count, context, prefill/decode separately, and topology rather than copying the training layout.

#### 3. Disaggregation break-even

Benefits are phase-specific batching and hardware, independent scaling, better parallelism choices, and removal of prefill-decode interference. Costs are KV transfer bytes over sustainable bandwidth, fixed handoff and synchronization, extra queueing, destination reservation, network contention, layout conversion, and operational failure modes. Evaluate goodput under both first-token and token-cadence SLOs across the joint prompt/output distribution. The design wins only when the benefit exceeds all handoff terms at fleet scale.

#### 4. KV handoff

Give the transfer an idempotent identity and reserve compatible destination blocks first. The source transfers only committed state with model, adapter, token range, position, dtype, and layout metadata. Verify integrity and visibility before making decode schedulable, then atomically transfer ownership or release source references. Retries discover completed chunks, and failures clean up destination partials without destroying the source's recoverable copy. Token commit and client streaming have separate idempotency rules.

#### 5. Locality-aware routing

Predict time to useful progress on each eligible destination: queue plus missing prefill or transfer plus execution, adjusted for SLO, tenant, health, and memory pressure. Prefer a cache or adapter owner only while its saved work exceeds extra queue delay. Replicate very hot immutable state, use bounded stickiness for sessions, and retain a recompute fallback. Validate the predictor against actual completion time and prevent one popular prefix from hot-spotting a replica.

#### 6. Scaling and failure

Scale prefill from uncached input/encoder work and first-token risk; scale decode from generated-token demand, context distribution, KV occupancy, and cadence risk. Balance the bounded handoff queue and reserve destination memory before producing more state. Failure recovery now spans two pools and a transfer protocol: retry idempotently, reconstruct or replicate KV by policy, avoid duplicate streamed tokens, and keep capacity reserve for loss of a group or network path.

## Quantization, Compression, and Adapter Serving

LEAD: Compression changes capacity, bandwidth, arithmetic, layouts, calibration, and quality risk at once. Adapter serving adds another layer of dynamic state and batch compatibility. Neither is merely a smaller checkpoint.

The performance model made decode a bytes problem, and the fleet section made memory a replica-count problem. Compression attacks both at the source. The running service quantifies the stakes: 14 GB of weights set the 4.7 ms step floor, and KV at 131 kB per token caps concurrency near 200 conversations per replica.

### Precision as a systems choice

Quantization maps values to a lower-precision representation plus scale and possibly zero-point metadata. The useful question is not "How many bits?" but:

- which tensors are quantized;
- which operations consume the format directly;
- the scale granularity and metadata layout;
- accumulation and output precision;
- calibration or optimization data;
- hardware and kernel support;
- quality and performance at production shapes.

Common families include:

| Family | Main bandwidth/capacity effect | Common challenge |
| --- | --- | --- |
| Weight-only | Smaller model reads and footprint | Unpack/dequant overhead, small-row kernels |
| Weight-activation | Smaller operands and low-precision matrix path | Activation outliers and dynamic scaling |
| Floating low precision | Wider dynamic range than integers at similar bits | Scale policy and hardware availability |
| KV quantization | More context or batch, fewer KV bytes | Attention sensitivity and write/read conversion |
| Mixed precision | Protect selected layers or channels | More variants, conversions, and analysis |

The format name alone does not determine speed. Group size, packing order, scale placement, padding, and fused epilogue support can decide whether the kernel reaches the intended hardware path.

### Weight-only quantization

In small-batch decode, weight reads often dominate dense linear layers. Weight-only quantization stores packed low-bit weights, loads scales, reconstructs or directly consumes low-precision values, and multiplies by higher-precision activations.

Smaller weights can:

- reduce step latency when bandwidth-bound;
- make the model fit on fewer devices;
- leave more memory for KV and batch;
- reduce model load and replica startup time.

The path can lose if unpack and scale work is not overlapped, tensor-core instructions do not support the exact format, group metadata is large, layout conversion occurs at runtime, or the compressed model shifts the bottleneck to KV, communication, or launch overhead.

Calibration-aware methods protect important weights or choose scales using representative activations. The serving team still owns format validation because a high-quality checkpoint can be paired with a poor runtime layout.

On the running service, a well-executed 4-bit weight path shrinks streamed bytes from about 14 GB toward 4 GB and the small-batch step floor from 4.7 ms toward 1.5 ms - potentially tripling single-stream token rate. The same checkpoint behind a poor unpack path can sit above the FP16 floor. The format's arithmetic is a promise; the kernel's achieved bandwidth is the delivery.

### Weight-activation and floating formats

Quantizing activations can unlock lower-precision matrix instructions and reduce intermediate traffic. Activation distributions are input-dependent and often contain outliers. Strategies include per-token or per-row dynamic scales, offline transformations that move quantization difficulty between activations and weights, selective higher precision, and block-scaled floating formats.

Dynamic scaling adds reductions, metadata, and synchronization. Offline scaling depends on calibration coverage. Per-tensor scales are cheap but sensitive to outliers; per-channel, row, or block scales track range better but increase metadata and may complicate kernels.

Accumulate sensitive reductions in an appropriate wider type. Protect normalization, logits, embeddings, or selected layers when evidence shows they dominate error. "Everything at one dtype" is an implementation convenience, not a quality principle.

### KV-cache quantization

KV state grows with live cached tokens, so lower precision can increase concurrency or context and reduce attention bandwidth. Keys affect attention scores; values affect the weighted sum, and their sensitivities may differ by layer, head, token position, and model.

Scale choices include per-tensor, per-head, per-channel, per-token, or grouped blocks. Finer scales improve range tracking but consume metadata and append-time work. Dynamic token scales may require a reduction when KV is written. The attention kernel must dequantize without destroying locality or occupancy.

Evaluate long-context retrieval, position-sensitive tasks, multi-turn conversations, rare tokens, tool-use formats, and output distributions, not only short perplexity. A format that doubles theoretical capacity but slows attention can reduce serving capacity under latency SLOs.

For the running service, FP8 KV doubles capacity toward 900,000 cached tokens - about 400 median conversations per replica - and a 4-bit format doubles it again. Each halving also halves long-context attention traffic, which matters once the batch's cached tokens approach the 107,000-token crossover where KV reads rival weight reads.

### Compression shifts the optimum

Suppose uncompressed weights require a four-device tensor-parallel group. A weight format that fits the model plus reserve on two devices may create twice as many replicas, reduce collective degree, and change the best batch. This fleet-level gain can exceed its isolated kernel speedup.

The reverse can occur: a quantized format frees memory, the scheduler admits a larger batch, and longer iterations violate token-cadence SLOs. Re-tune group size, batch, KV budget, graph shapes, and autoscaling after changing precision.

### Quantization validation ladder

Use a staged evaluation:

1. tensor and layer numerical comparisons;
2. teacher-forced token probability or perplexity by domain;
3. generation quality, task, safety, and structured-output suites;
4. long-context, multilingual, code, rare-slice, and adversarial tests;
5. kernel benchmarks across batch, context, and alignment shapes;
6. end-to-end replay with scheduling, KV pressure, and target hardware;
7. canary with slice metrics, rollback, and version isolation.

Measure both mean quality and regressions concentrated in specific layers, languages, adapters, or long contexts. Use the same sampling seeds or distributional test appropriate to the reproducibility contract.

:::callout pitfall|A smaller model can be a slower service
Checkpoint bytes do not reveal unpack cost, scale traffic, unsupported instructions, layout conversion, collective changes, or the batch and group size the scheduler will choose. Benchmark the deployed artifact and re-plan the fleet.
:::

### Multi-adapter serving

Low-rank adapters modify selected linear layers while sharing a base model. Serving many adapters on separate full replicas wastes base weights. A multi-adapter engine keeps the base resident, loads adapter matrices dynamically, and groups adapter work across requests.

For a linear layer with base output `x W`, a low-rank update has the form:

`output = x W + scale times x A B`

Requests in one batch may use different `A` and `B`. The engine can execute grouped low-rank operations, segment tokens by adapter, or fuse adapter work into a specialized path. Tiny ranks make launch and indexing overhead important; large or multiple adapters can create meaningful compute and memory pressure.

Adapter batching should not require every sequence to use the same adapter, but completely unstructured mixtures can reduce locality. Track adapter ID and rank as scheduling features and limit variant explosion.

### Adapter lifecycle and isolation

An adapter has identity, base-model compatibility, rank and target modules, dtype, tokenizer/template expectations, authorization, and version. Loading should verify a manifest and content digest before publication. Keep reference counts while requests execute; evict only when no in-flight use remains.

An adapter cache trades load latency against device memory and batch capacity. Eviction value depends on request frequency, load cost, bytes, and whether the base worker can serve useful traffic while loading. Preload latency-critical adapters, but cap tenants so one customer cannot fill every worker.

Adapter and prefix identities interact: KV produced with one adapter cannot generally be reused under another. A rollout of an adapter version requires cache namespace separation just like a base-model rollout.

### Capacity accounting with adapters

A worker memory budget includes:

`base weights + resident adapters + active KV + reusable KV + workspaces + graph buffers + allocator reserve`

Adapter count alone is misleading because ranks and targeted layers differ. Account bytes and measured per-token work. For sparse popularity, maintain a warm subset and route by adapter locality. For uniformly hot adapters, replicate them. For a long tail, consider host-resident staging, but include transfer time and queue hot spots.

### Principal Interview Review

1. Compare weight-only, weight-activation, and KV quantization as serving interventions.
2. Why can a four-bit checkpoint be smaller but slower than a higher-precision one?
3. Design a quality and performance evaluation for a new quantization recipe.
4. How can quantization change the optimal parallelism and batching plan?
5. Design multi-adapter batching and residency management.
6. What isolation and versioning rules connect adapters, tokenizers, and prefix caches?

### Answer Key

#### 1. Quantization targets

Weight-only primarily reduces model footprint and decode weight bandwidth while leaving activation compute higher precision; it pays unpack and scale cost. Weight-activation can use lower-precision matrix paths and reduce operand traffic, but activation outliers, dynamic scales, and accumulation matter. KV quantization targets context capacity and attention bandwidth, adding append/read conversion and long-context quality risk. Choose the tensor whose bytes control the target phase and require a supported kernel and quality contract.

#### 2. Smaller but slower

The runtime may unpack into a wider type, read many group scales, execute scalar conversions, use an unsupported or poorly tiled matrix path, or convert layouts at load or every step. The shape may be compute-bound already, or compression may expose KV, collective, sampling, or launch overhead. Measure achieved bandwidth/compute, instruction path, scale and conversion traffic, kernel time by layer and shape, group size, batch, and end-to-end goodput.

#### 3. Evaluation design

Validate layers numerically, then probability/perplexity by representative domain, task and safety quality, generation behavior, long-context retrieval, code/multilingual/rare slices, and adapters. Benchmark exact deployed kernels across batch, context, alignment, and hardware. Replay production arrivals with KV pressure and measure first-token/cadence SLOs, memory, energy if relevant, and goodput. Canary by quantization version with rollback and cache isolation; do not infer production quality from one aggregate benchmark.

#### 4. Re-planning after quantization

Smaller weights may let the model use fewer tensor-parallel devices, increasing replicas and reducing collectives. Freed memory permits more KV or adapters and a larger batch. Faster matrix work can make attention, communication, or host overhead dominant. Re-sweep group size, replica count, batch and iteration budget, KV allocation, graph variants, and autoscaling on the new artifact; the old optimum is no longer evidence.

#### 5. Multi-adapter serving

Share one base model, tag tokens or sequences by adapter, and execute low-rank updates through grouped or segmented operations that support mixed adapters without padding every request to one rank. Cache adapters by expected saved load time per byte, preload latency-critical ones, route with bounded locality, and keep per-tenant quotas. Version manifests, validate base compatibility, reference-count in-flight use, and expose fallback or rejection when an adapter cannot load within SLO.

#### 6. Isolation and versioning

Adapter identity includes content digest, base-model version, target modules, rank, scale, dtype, and any tokenizer/template requirements. Authenticate access before routing or loading. Prefix-cache keys include the exact adapter and tokenizer/template identity, so state never crosses incompatible versions or tenants. Rollout publishes a new namespace, drains references to the old one, and deletes adapter and cache artifacts under the same retention policy.

## Production Architecture, Capacity, and Reliability

LEAD: A fast engine becomes a dependable service only when artifacts, control planes, observability, overload, failure recovery, security, and rollout are designed around its stateful streaming behavior.

Everything so far tuned one replica or one mechanism. Production is where the running service must hold its two SLOs through bursts, failures, and rollouts, with a couple hundred conversations of KV state in flight on every replica. This final section closes the loop the part opened: the same boundaries the lifecycle section timestamped are now the boundaries that ownership, capacity, and recovery are organized around.

### Data plane and control plane

The **data plane** handles live requests: gateway, tokenizer, router, scheduler, model workers, KV transfer, sampling, and stream transport. The **control plane** manages model and adapter artifacts, placement, configuration, health, rollout, autoscaling, quotas, and policy.

Keep the hot path independent of a synchronous control-plane lookup. Workers consume versioned snapshots and continue safely during temporary control-plane failure. Conversely, the control plane needs enough data-plane telemetry to stop routing to unhealthy or incompatible workers.

Define ownership at boundaries. The gateway owns authentication and client semantics; the router owns eligible placement; the scheduler owns local admission and execution order; the worker owns model state and token commit; the allocator owns KV lifetime. Ambiguous ownership produces double admission, leaked pages, or duplicate streaming during retries.

### Artifact build and compatibility

A deployable model is a bundle, not one weight file. It may include:

- model and tokenizer with immutable digests;
- chat templates and special-token configuration;
- quantization scales and packing layout;
- compiled engines or kernels by hardware capability and shape;
- parallelism and placement metadata;
- adapter compatibility schema;
- sampling and grammar versions;
- quality, performance, and security attestations.

Build artifacts reproducibly, sign or verify them, and publish atomically. A worker should reject a partial or incompatible bundle before serving traffic. Engine caches and CUDA graphs are derived artifacts and must be invalidated when their relevant inputs change.

### Startup, warmup, and readiness

Startup includes artifact fetch, deserialization, memory allocation, weight placement, communicator creation, compilation or autotuning, graph capture, cache initialization, and warmup. Readiness should require representative model execution, not only a listening socket.

Warm the important precision, batch, context, adapter, and graph variants without trying to enumerate an unbounded shape space. Keep a fallback path for a cold shape. Record cold-start stages so autoscaling knows whether a new replica will arrive in seconds or minutes.

A rolling restart temporarily removes capacity and cache locality. Drain admissions, let bounded in-flight work finish, migrate or recompute state by policy, and maintain enough reserve that the remaining fleet does not enter overload.

### Capacity planning

Start from the workload and SLO, not peak hardware FLOPs. For each request class estimate:

- uncached input and encoder work;
- output tokens and decode context trajectory;
- KV bytes and lifetime;
- prefix and adapter locality;
- phase service curves by batch and topology;
- arrival rate, bursts, and target percentiles.

At a stable high level, Little's Law relates average in-system concurrency `N`, arrival rate `lambda`, and average time in system `T`:

`N = lambda T`

It is an accounting identity, not a tail-latency guarantee. Use it to cross-check concurrency and memory, then use replay or a queueing model for variability and percentiles.

For the running service, a median request spends roughly 20 to 30 seconds in the system: 300 output tokens at a batch-shared decode cadence plus queueing. At `T = 25 s` and about 200 concurrent conversations of KV capacity, one replica sustains `lambda = N / T = 8` requests per second - before failure headroom, bursts, and long-context outliers, which is exactly why the reserve terms in this section exist.

For a worker, enforce:

`weights + adapters + active KV + reusable KV + workspaces + reserve <= usable memory`

Use usable memory after runtime and fragmentation measurements, not device nameplate capacity. For the fleet, include at least one relevant failure or maintenance scenario and the startup delay of replacement capacity.

:::callout insight|Capacity is a memory claim as much as a compute claim
On the running replica, admission is bounded by roughly 200 conversations of KV before compute saturates. State autoscaling, drain, and failure reserve in cached tokens and replicas, not in GPU utilization percentages.
:::

### Autoscaling signals

GPU utilization is a lagging and ambiguous signal. Combine:

- queued predicted prefill and decode work;
- first-token and token-cadence SLO risk;
- active and reserved KV occupancy;
- arrival forecasts and tenant quotas;
- adapter/cache locality pressure;
- replica warmup time and failure reserve.

Scale early enough to cover startup. Scale down by draining workers and accounting for cache coldness that remaining replicas will experience. Hysteresis and minimum residency avoid oscillation when traffic hovers near a threshold.

In disaggregated fleets, scale stages separately but coordinate the handoff queue and destination memory. In model-parallel groups, add or remove complete compatible groups rather than individual ranks.

### Observability

The minimum operational view includes:

- arrival, admission, rejection, cancellation, and completion by class;
- queue delay, first-token, inter-token, and end-to-end distributions;
- input/output tokens, goodput, and batch/iteration shape;
- active, reserved, reusable, fragmented, and offloaded KV bytes;
- prefix hit tokens and actual time saved;
- adapter load, hit, eviction, and mixed-batch behavior;
- speculative proposal, acceptance, committed progress, and cycle cost;
- per-phase device time, collective skew, transfer time, and host gaps;
- errors by stage, model version, worker group, and termination reason.

Control metric cardinality. Put request IDs and exact shapes in sampled traces, not unbounded metric labels. Histograms need buckets that resolve the SLO region; averages are insufficient.

### Failure taxonomy

Separate failures by recoverability and commit point:

- **before admission:** reject or route elsewhere;
- **queued:** retry safely if deadline permits;
- **during prefill before output:** retry on a compatible replica, perhaps using cached prefix;
- **after streaming begins:** resume only with exact committed state and transport protocol, otherwise terminate clearly;
- **allocator or OOM pressure:** stop admissions, reclaim, and protect active reservations;
- **rank or link failure:** fail the whole synchronized group and replace it;
- **control-plane isolation:** continue from a safe snapshot for a bounded interval;
- **semantic corruption:** quarantine the artifact and roll back, not merely restart.

Test fault injection at token-commit, KV handoff, cancellation, adapter eviction, and rollout boundaries. Many leaks and duplicates require a rare race, not a steady benchmark.

### Graceful degradation

Safe overload actions preserve model semantics: reject new work, queue within a bound, route, shed batch traffic, lower speculative effort, disable best-of, or evict reusable caches. Actions that change model, precision, context, safety filters, or sampling require explicit product policy and response metadata.

Protect already admitted requests according to the published contract, but do not let an unbounded generation monopolize the service. Enforce declared maximum output and context, tenant budgets, stream idle timeouts, and cancellation.

### Rollout and rollback

Version model, tokenizer, templates, quantization, engine, sampling, and scheduler configuration independently but deploy them as tested compatibility sets. A rollout sequence can be:

1. offline semantic and performance validation;
2. shadow execution on representative traffic without user impact;
3. canary by tenant or request slice with separate cache namespace;
4. gradual traffic increase under automated guardrails;
5. drain old state only after rollback risk falls;
6. retain a tested rollback artifact and capacity path.

Compare SLOs, goodput, error slices, output distributions, safety, cache behavior, and cost. A scheduler change can alter batching and therefore RNG traces even when the model is unchanged; use the correct equivalence criterion.

Rollback is harder with state. New-version KV and adapters may be incompatible with old workers. Either keep version-sticky requests until completion, reconstruct state from tokens on rollback, or maintain an explicitly compatible format. Never reinterpret state across versions by assumption.

### Security and abuse resistance

Enforce authentication, quotas, maximum context/output, media limits, grammar complexity limits, adapter authorization, and bounded streaming buffers before scarce accelerator work is committed. Tokenization and grammar compilation can themselves be denial-of-service targets.

Isolate tenant cache keys and adapter state. Scrub or overwrite memory according to the threat model before reallocation, avoid logging prompts or reversible low-entropy hashes, and propagate deletion across prefix caches, offload, traces, and backups.

Side channels include timing differences from cache hits, model or adapter residency, and co-tenant load. Reduce observable detail, namespace reuse, and offer dedicated capacity where stronger isolation is required.

### Cost and energy

Useful cost metrics include accelerator-hours per good request, dollars per million accepted output tokens under an SLO, and energy per completed request. Include idle reserve, failed and rejected work, draft computation, cache transfer, and host/network resources. A higher-throughput configuration can cost more per good request if it misses latency or needs a larger failure reserve.

Optimize total service cost after semantic and reliability constraints. The cheapest kernel is irrelevant if its artifact takes too long to roll out safely or its shape specialization multiplies operational burden.

### Production readiness checklist

| Area | Required evidence |
| --- | --- |
| Semantics | Reference logits/sampling/stop/constraint tests and declared approximation |
| Performance | Open-loop replay, SLO-goodput curve, cold and warm behavior |
| Memory | Capacity model, allocator invariants, cancellation and OOM tests |
| Distribution | Topology plan, collective/transfer profiling, group-failure recovery |
| Operations | Metrics, traces, alerts, runbooks, load shedding, autoscaling |
| Rollout | Versioned artifacts, cache isolation, canary guardrails, tested rollback |
| Security | Auth, quotas, tenant isolation, deletion, abuse limits, audit trail |

### Principal Interview Review

1. Produce a capacity plan for a new LLM endpoint from workload traces and SLOs.
2. Why is GPU utilization insufficient for autoscaling?
3. Design readiness, draining, and rollout for stateful model workers.
4. How should the service respond to failure after some tokens have streamed?
5. What observability distinguishes a scheduler regression from a kernel or network regression?
6. Design an overload and rollback policy that preserves explicit semantics.

### Answer Key

#### 1. Capacity plan

Segment traces by prompt, output, context, model, adapter, feature, locality, and burst. Benchmark phase service curves and memory per live token on candidate group layouts. Replay open-loop arrivals to find the highest rate meeting first-token, cadence, error, fairness, and quality SLOs with KV and failure reserve. Convert this to complete replica/group count, add startup and maintenance headroom, validate one-group loss, and report cost per good request with assumptions and sensitivity ranges.

#### 2. Autoscaling signals

Utilization says a device is busy, not whether useful deadlines are met or how much queued work and state is arriving. It can be high during inefficient kernels or doomed overload, and low while memory is full of paused requests. Scale from predicted queued phase work, SLO risk, KV reservations, arrival forecast, locality, warmup delay, and failure reserve, using utilization as supporting evidence rather than the control objective.

#### 3. Stateful worker lifecycle

Readiness requires verified artifact identity, memory/communicator setup, representative warm execution, and registration with compatible capacity. Draining stops new admissions, preserves or transfers committed state by policy, completes bounded in-flight work, and releases cache ownership before termination. Rollout uses separate version/cache namespaces, shadow and canary stages, automatic semantic/performance guardrails, enough old capacity for rollback, and explicit treatment of in-flight KV compatibility.

#### 4. Failure after streaming

Track the last logically committed and acknowledged token. Resume only if a compatible worker can reconstruct exactly the model, sampling/RNG, constraint, and KV state and the transport can suppress duplicate output. Otherwise terminate with a clear partial-response error and stable request identifier; blindly restarting could duplicate or contradict visible text. Release state idempotently and record the commit point for diagnosis and billing.

#### 5. Observability diagnosis

Scheduler regressions change queue delay, batch composition, phase mixing, iteration gaps, reservation failures, and fairness while per-shape kernels may remain stable. Kernel regressions raise device time or reduce achieved bandwidth/compute for the same shape. Network regressions raise collective or KV-transfer time, skew ranks, and correlate with links or paths. Join per-request spans with scheduler decisions, shape-tagged phase timing, per-rank collectives, host gaps, and version/config identity.

#### 6. Overload and rollback

Bound queues and memory reservations, reject early with retry guidance, protect admitted interactive work, shed batch and optional best-of/speculative effort, and enforce tenant quotas. Any change to model, precision, context, sampling, or safety requires an authorized degradation mode and response metadata. Rollout isolates versions and caches, preserves a warm rollback path, drains incompatible in-flight state or reconstructs from tokens, and triggers on goodput, semantic, quality, error, and memory guardrails rather than utilization alone.

### Further Study and Primary References

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) - iteration-level scheduling and selective batching.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://doi.org/10.1145/3600006.3613165) - paged KV memory and serving throughput.
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) - chunked prefill and stall-free scheduling.
- [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) - phase disaggregation and goodput-oriented placement.
- [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html) - exact speculative sampling.
- [SGLang: Efficient Execution of Structured Language Model Programs](https://papers.nips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf) - radix prefix reuse and structured execution.
- [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html) and [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) - activation-aware quantization methods.
- [Punica: Multi-Tenant LoRA Serving](https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html) - shared-base multi-adapter serving.
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/latest/index.html) - maintained engine features, KV reuse, quantization, and deployment guidance.
- [MLPerf Inference Documentation](https://docs.mlcommons.org/inference/submission/) - benchmark scenarios and reproducibility requirements.

### Final Inference Principle

*The unit of optimization is a request completed within its semantic and SLO contract, not a kernel, token, batch, or GPU in isolation.*

The serving design is complete only when performance models predict the important shapes, state transitions are safe under cancellation and failure, measurements include queueing and user-visible cadence, and rollout can reverse without corrupting live state. For the running service of this part, that means the 4.7 ms step floor, the 131 kB-per-token KV ledger, and the two latency SLOs from the opening table survive contact with schedulers, caches, fleets, and rollouts - defended end to end, not per component.
