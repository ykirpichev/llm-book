# Part V - Distributed ML Systems

Distributed design begins when one accelerator, process, memory domain, or failure domain cannot satisfy the objective. Every parallelism axis exchanges one scarce resource for another: memory for communication, latency for throughput, replication for availability, or implementation simplicity for scale.

This part builds from communication cost to complete training and inference plans. It covers collectives, topology, data and fully sharded parallelism, tensor and context parallelism, pipelines, sparse experts, distributed inference, checkpoint recovery, cluster scheduling, and diagnosis. Every chapter ends with advanced engineering questions and complete answer keys.

:::callout decision|Write the physical plan, not only the degrees
A configuration such as TP=8, PP=4, DP=16 is incomplete until every process group is mapped to devices, links, nodes, failure domains, tensors, and schedules. Logical parallelism becomes performance only through physical placement.
:::

## Communication Models, Collectives, and Topology

LEAD: Distributed performance is governed by the bytes, synchronization steps, link paths, and overlap opportunities on the critical path. FLOPs alone cannot explain a job that waits on communication.

### The alpha-beta model

For a message of `n` bytes, a first-order point-to-point time model is:

`T(n) = alpha + n / beta`

`alpha` is fixed latency, including launch, protocol, and synchronization. `beta` is sustainable bandwidth for the path and message regime. The model separates two common failures:

- many tiny operations pay latency repeatedly;
- large operations saturate bandwidth or a shared bottleneck.

Real systems add contention, topology hops, protocol thresholds, device synchronization, reduction compute, and software overhead. The alpha-beta model is still useful because it predicts whether to fuse messages, reduce bytes, change the algorithm, or overlap work.

Measure effective values by message size, source/destination placement, concurrency, and direction. Nameplate link bandwidth is not application bandwidth. A path may traverse a GPU fabric, PCIe switch, host bridge, NIC, top-of-rack switch, and oversubscribed spine.

### Collective semantics

The core collectives have different ownership contracts:

| Collective | Input ownership | Output ownership | Typical ML use |
| --- | --- | --- | --- |
| Broadcast | One rank owns full input | Every rank receives full tensor | Parameters, metadata |
| All-reduce | Every rank owns full contribution | Every rank receives reduced tensor | Replicated gradients |
| Reduce-scatter | Every rank owns full contribution | Each rank receives one reduced shard | Sharded gradients |
| All-gather | Each rank owns one shard | Every rank receives the concatenation | Sharded parameters/activations |
| All-to-all | Each rank owns destination partitions | Each rank receives its incoming partitions | Expert token dispatch |
| Send/receive | One source and destination per operation | Destination receives message | Pipeline activations |

A collective is both data movement and a participation contract. Every rank in the communicator must issue compatible operations in compatible order. Shape, dtype, count, root, and sequence disagreements can hang or corrupt the job.

### Ring all-reduce

For `p` ranks and a tensor of `N` bytes per rank, a ring all-reduce is commonly decomposed into a reduce-scatter and an all-gather. Each rank sends and receives approximately:

`2 (p - 1) N / p bytes`

There are approximately `2 (p - 1)` serialized steps. A simplified time is:

`T_ring ~= 2 (p - 1) alpha + 2 (p - 1) N / (p beta)`

The ring approaches two tensor volumes per rank for large `p`, making it bandwidth-efficient for large messages. Its latency term grows with rank count, so a tree or other algorithm can be better for small messages. Implementations choose protocols and algorithms by topology and size; do not assume one ring for every collective.

### Trees and hierarchical collectives

A balanced tree reduces the number of latency steps to order `log p`, but link utilization and total bytes depend on the tree and topology. It often helps small or medium messages where startup dominates.

Hierarchical collectives first reduce or gather within a high-bandwidth domain, then communicate representatives across slower links, then distribute locally. Common domains are accelerator fabric within a node, NIC rails across nodes, or islands inside a rack.

The hierarchy must match contention. If every local rank sends independently through one NIC, aggregate injection bandwidth, not GPU links, limits the operation. If ranks are mapped inconsistently across nodes, the logical ring can cross slow links more often than expected.

### Bandwidth accounting

Distinguish:

- **algorithm bandwidth:** useful tensor bytes divided by elapsed time;
- **bus bandwidth:** an adjusted measure reflecting bytes carried by the collective algorithm;
- **link bandwidth:** measured bytes on a physical edge;
- **injection bandwidth:** aggregate traffic a node or NIC can enter into the network;
- **bisection bandwidth:** capacity between large partitions of the cluster.

These numbers answer different questions. A collective can report respectable per-rank bandwidth while oversubscribing a shared uplink and slowing other groups. Compare the definition used by the benchmark with the application traffic pattern.

### Topology and process groups

Build a topology inventory before selecting degrees:

- GPUs per high-bandwidth fabric domain;
- PCIe root and CPU socket affinity;
- NIC count, rail mapping, and GPU-to-NIC locality;
- intra-rack and inter-rack oversubscription;
- shared storage and control-plane paths;
- failure boundaries for host, switch, rack, and zone.

Map frequent, latency-sensitive collectives to the fastest domain. Tensor-parallel groups commonly stay within a node or fabric island. Data-parallel groups can span nodes because their larger gradient collectives have more overlap opportunity. Expert and context groups require separate analysis because they move activations and may be sensitive to token imbalance.

Process-group construction is part of correctness. Generate groups from a deterministic rank coordinate system, print membership, and test that every rank derives the same groups. Overlapping groups are legal, but concurrent collectives need a globally consistent launch order or independent synchronization.

### Communication and computation overlap

Overlap removes communication from the critical path only when:

- a dependency-free compute interval exists;
- communication and compute use resources that can progress concurrently;
- network traffic does not starve the compute kernel's memory path;
- buffers remain alive and are not reused early;
- the launch order avoids communicator serialization.

Measure **exposed communication**, not total communication duration. If a 4 ms all-reduce overlaps 3 ms of backward compute, only roughly 1 ms is exposed, assuming no slowdown from contention. Timeline comparison must also check whether overlapped compute became slower.

Chunking creates earlier overlap but increases latency and launch overhead. Large buckets improve bandwidth but start later. The optimum follows layer order, bandwidth curve, compute gaps, and communicator scheduling.

### GPUDirect, RDMA, and registration

Direct device networking avoids staging every transfer through host memory. It still depends on correct PCIe/NIC topology, memory registration, queue resources, and transport configuration. Registration caches and persistent buffers reduce setup cost; unbounded registration can exhaust resources.

Host control remains relevant. CPU oversubscription, interrupt placement, NUMA mismatch, container limits, or a blocked progress thread can lower collective performance even when GPU and network hardware are healthy.

### Collective correctness and deadlocks

Common causes of hangs include:

- one rank takes a different branch and skips a collective;
- communicators issue overlapping collectives in inconsistent order;
- tensor sizes differ because of ragged inputs or a local error;
- a prior CUDA fault prevents a rank from reaching the collective;
- one rank is blocked in input, checkpoint, or host synchronization;
- a failed link or process leaves peers waiting.

Use sequence numbers, communicator IDs, tensor metadata, and call stacks in diagnostics. Timeouts limit damage but do not identify the first divergence. Capture the last successfully completed and first pending collective on every rank.

:::callout pitfall|A slow collective may be an innocent barrier
Because ranks synchronize through collectives, the operation often waits for a straggler created earlier by data loading, a kernel, page fault, or another communicator. Measure arrival skew separately from transport duration.
:::

### Design Exercises

1. Derive ring all-reduce bytes and explain when a tree can be faster.
2. Compare all-reduce with reduce-scatter plus all-gather by ownership and use.
3. How would you map tensor- and data-parallel groups onto a multi-node topology?
4. What evidence proves communication-compute overlap is useful?
5. Diagnose a collective whose duration is high only on some iterations.
6. Why can a bandwidth benchmark disagree with application communication performance?

### Worked Solutions

#### 1. Ring and tree

A ring reduce-scatter sends `(p-1)/p` tensor bytes per rank, and the all-gather sends the same again, for `2(p-1)N/p`. It uses about `2(p-1)` ordered steps, so startup grows linearly with ranks. A tree uses order `log p` levels and can win for small messages or many ranks where latency dominates. The ring is attractive for large messages because it keeps links busy with near-minimal per-rank bandwidth. Measure the implementation and topology because hybrid algorithms are common.

#### 2. Ownership contracts

All-reduce begins with a full-size contribution on each rank and returns the full reduced tensor to each rank. Reduce-scatter produces only one reduced shard per rank; a following all-gather reconstructs a full tensor when needed. Sharded optimizers exploit the interval between these operations to keep gradients or parameters partitioned, while ordinary data parallelism needs the complete gradient on every replica.

#### 3. Group mapping

Keep frequent latency-sensitive tensor-parallel collectives within the fastest GPU fabric and align ranks with PCIe/NIC locality. Form data-parallel groups from corresponding tensor/pipeline coordinates across replicas, preferably spreading them over independent rails without creating oversubscribed hot spots. Print group membership and physical paths, reserve cross-node bandwidth for large overlap-friendly reductions, and validate placement under one-node failure and scheduler fragmentation.

#### 4. Useful overlap

Compare timelines and step time with overlap enabled and disabled at the same workload. Show that communication starts earlier, the dependency-free compute interval covers it, exposed communication decreases, and neither compute kernels nor other collectives slow from contention. Verify buffer lifetime and numerical equivalence. Total communication duration can increase while step time falls; the relevant metric is the critical path.

#### 5. Intermittent collective latency

Separate rank arrival time from transport completion. Correlate slow iterations with input stalls, long kernels, memory pressure, checkpoint activity, network counters, message size, routing, and other communicator traffic. Identify the earliest late rank and its preceding event. If arrivals are aligned but transport is slow, inspect link errors, shared-uplink contention, protocol thresholds, and topology. Compare per-rank traces rather than only the maximum collective span.

#### 6. Benchmark disagreement

Microbenchmarks often use aligned contiguous buffers, one communicator, steady message sizes, dedicated links, warm registrations, and no competing compute. Applications use buckets, nonuniform timing, multiple groups, topology-specific paths, concurrent storage or KV traffic, and may expose only part of the communication. Ensure the benchmark uses the same ranks, sizes, directions, concurrency, and metric definition before treating its bandwidth as a bound.

## Data Parallelism, ZeRO, and Fully Sharded Training

LEAD: Data parallelism scales independent examples. Sharded variants reduce replicated model state, but replace simple gradient synchronization with a schedule of parameter gathers, gradient partitions, prefetches, and memory-lifetime decisions.

### Replicated data parallelism

Each data-parallel rank owns a complete model and optimizer replica and processes different local examples. Backward produces local gradients, and an all-reduce makes the reduced gradient identical before every rank performs the same optimizer update.

For local microbatch `b`, gradient-accumulation steps `g`, and data-parallel degree `d`, the global batch is `B_global = b * g * d`. This assumes `b` is per rank and every rank contributes the same number of examples. Ragged or filtered batches require explicit weighting so the reduced gradient represents the intended global objective.

Scaling `d` increases global batch unless `b` or `g` changes. That can alter optimization, learning-rate schedule, and sample order. System scaling and training semantics must be planned together.

### Gradient reduction and buckets

Frameworks register backward hooks and place parameters into buckets. When all gradients for a bucket are ready, its all-reduce can overlap with backward computation for earlier layers.

Small buckets start early but pay more launch and latency. Large buckets use bandwidth well but may not begin until late in backward. Parameter registration order should roughly match gradient-ready order; unused or conditional parameters complicate readiness.

Gradient accumulation can suppress synchronization for intermediate microbatches and reduce once at the accumulation boundary. This saves communication but holds gradients longer and changes the overlap opportunity. Ensure the final microbatch triggers synchronization on every rank, including early termination paths.

### Memory accounting

For `P` parameters, per-rank training memory includes some combination of:

- model parameters, possibly low precision;
- gradients, possibly accumulated in higher precision;
- optimizer moments and master parameters;
- activations and saved tensors;
- temporary workspaces, communication buckets, and allocator reserve.

For Adam-like training with low-precision parameters, optimizer states and master weights can exceed model parameter bytes several times over. State the exact dtype and optimizer representation rather than quoting a universal bytes-per-parameter number.

Activation memory follows microbatch, sequence length, hidden size, layer count, checkpoint policy, and parallel partitioning. Sharding model state does not automatically solve activation OOM.

### Carry the running model across the training boundary

Use the same nominal seven-billion-parameter model as Parts III and IV, but change the task from inference to full-parameter training. Under an explicitly chosen convention of two-byte weights, two-byte gradients, four-byte master weights, and two four-byte Adam moments, persistent model state is `7e9 * 16 = 112 GB` in decimal units. This excludes activations, temporary buffers, communication buckets, and allocator reserve. Different optimizers or gradient dtypes change the result.

Ordinary data parallelism replicates those 112 GB on each rank; adding replicas does not make the per-rank state fit. Ideal eight-way full sharding reduces the resident shard to `112 / 8 = 14 GB` per rank, before transient all-gathers and the other excluded allocations. That number happens to equal the running model's two-byte inference weight size, but the allocations have different meanings. Training does not become a 14 GB job merely because its persistent state is sharded.

The inference request's KV budget also does not transfer into training unchanged. Its 32 layers, eight KV heads, head dimension 128, and two-byte elements give 128 KiB per cached token, or 250 MiB for a 2,000-token prompt. Training instead retains or recomputes the forward tensors needed by backward; use the chosen microbatch, sequence length, and checkpoint policy to budget those activations. The arithmetic is checked in `tests/test_resource_models.py`; neither calculation establishes measured GPU capacity.

### ZeRO stages

ZeRO-style sharding progressively removes data-parallel redundancy:

| Stage | Sharded across data ranks | Main new communication/lifetime issue |
| --- | --- | --- |
| Stage 1 | Optimizer states | Parameter updates need shard ownership and synchronization |
| Stage 2 | Optimizer states and gradients | Gradients naturally arrive through reduce-scatter |
| Stage 3 | Optimizer states, gradients, and parameters | Parameters must be gathered around computation |

Exact implementations differ. The principle is that each rank owns a partition of persistent state while temporarily materializing what its local compute needs.

Sharding reduces memory approximately with data-parallel degree for shardable objects, but replicas still carry activations, buffers, some metadata, padding, and peak gathered parameters. Peak memory, not steady shard size, decides whether the model fits.

### Fully sharded execution

A fully sharded module generally performs:

1. prefetch or all-gather parameter shards for a module group;
2. execute forward with materialized parameters;
3. reshard or retain parameters according to policy;
4. gather again as backward needs the module;
5. reduce-scatter gradients to owners;
6. update local optimizer shards.

Wrapping granularity controls this schedule. Very small units issue many latency-bound gathers. Very large units raise peak memory and delay prefetch. Flattening compatible parameters reduces metadata and improves collective efficiency but complicates per-parameter tooling and checkpoints.

Backward prefetch can overlap the next parameter gather with current gradient computation. Forward prefetch is useful when execution order is static and CPU issue cannot stay ahead. Too much prefetch creates multiple live gathered units and causes OOM.

### Reshard-after-forward decisions

Resharding immediately after forward frees parameter memory but requires another gather in backward. Retaining parameters saves communication but increases peak live state, particularly with many layers or microbatches.

Choose by available memory, module size, recomputation, pipeline schedule, and bandwidth. Hybrid sharding can fully shard within a node and replicate across nodes, reducing cross-node parameter gathers while retaining some memory savings.

### Activation checkpointing and offload

Activation checkpointing saves selected boundary tensors and recomputes internal forward operations during backward. It trades compute for activation memory and can change overlap timing. Selective checkpointing targets expensive saved tensors while avoiding recomputation of IO-heavy or nondeterministic work.

CPU or storage offload expands capacity but introduces transfer and page-fault risk. It is viable when transfers overlap and the interconnect sustains the required bytes per step. An offload plan that fits but extends step time beyond the training budget is not a solution.

Checkpointed regions must preserve random-number behavior for dropout or other stochastic operations. Distributed recomputation also needs the same collective order as the original forward.

### Mixed precision and global numerical state

Ranks must agree on loss-scaling overflow, skipped optimizer steps, gradient clipping norm, and scheduler progress. A local overflow decision that only one rank follows will desynchronize parameters and future collectives.

Global norm clipping usually reduces partial squared norms across the appropriate parameter ownership group. Avoid double-counting replicated tensors or omitting shards. Optimizer-state sharding changes who owns the update but not the mathematical parameter set.

### Uneven inputs and join semantics

If one data rank exhausts input early while others continue, ordinary synchronized training hangs or changes the effective gradient. Prefer globally constructed batches with equal step counts. When uneven-input support is deliberate, participating ranks must shadow collectives or adjust normalization, and additional model collectives may require stricter termination handling.

Input failures are distributed failures. A corrupt sample on one rank, data-service timeout, or divergent filtering decision can leave peers waiting in gradient reduction. Log sample/shard identity and fail the group coherently.

:::callout pitfall|Sharding is a schedule, not a memory flag
The memory saving appears only if gathers, resharding, prefetch, backward, optimizer ownership, and checkpoints agree on tensor lifetime. A poorly wrapped fully sharded model can both use more peak memory and communicate more.
:::

### Design Exercises

1. Compare replicated DDP, ZeRO stage 2, and fully sharded training.
2. How do bucket size and parameter order affect gradient overlap?
3. Design a wrapping and prefetch policy for a deep transformer under a hard memory limit.
4. When should parameters be retained after forward rather than resharded?
5. What distributed state must agree when mixed-precision overflow occurs?
6. Why can activation checkpointing make a communication plan slower or incorrect?

### Worked Solutions

#### 1. DDP and sharded variants

DDP replicates parameters, gradients, and optimizer state and all-reduces gradients; it is simple and compute-granular but memory-heavy. ZeRO stage 2 shards optimizer and gradients, often using reduce-scatter, while parameters remain replicated. Fully sharded training also shards parameters, requiring gathers around module execution and careful reshard/prefetch timing. Compare peak bytes, collective frequency and size, overlap, checkpoint format, and local compute efficiency, not only persistent memory.

#### 2. Buckets and order

Smaller buckets become ready earlier and overlap more but pay fixed latency and may compete with compute. Larger buckets reach bandwidth efficiency but start later and can leave a reduction tail after backward. If bucket parameter order differs from gradient-ready order, one late gradient blocks an otherwise ready bucket. Profile ready times, exposed reduction tail, and network contention, then tune size and order for the real model and accumulation schedule.

#### 3. Wrapping and prefetch

Start with transformer-block-sized units, measure each unit's gathered bytes and compute window, and choose the largest unit that preserves peak reserve. Prefetch one dependency ahead when its gather can hide under current compute; cap outstanding gathers to a byte budget. Flatten compatible parameters, special-case unusually large embeddings or heads, reshard after forward when memory is tight, and validate with peak snapshots across the full forward/backward and accumulation schedule.

#### 4. Retain versus reshard

Retain when memory headroom exists and the avoided backward all-gather is expensive or difficult to overlap. Reshard when gathered parameters would accumulate to the peak limit, especially across many layers, microbatches, or pipeline stages. Hybrid policies retain small or soon-reused units and reshard large ones. Measure step time and peak live bytes; persistent shard size alone misses the decision.

#### 5. Mixed-precision agreement

All ranks must reach the same overflow decision, loss-scale update, optimizer-step skip, global gradient norm, clipping coefficient, step counter, learning-rate schedule, and RNG/data progression policy. Reduce overflow flags and partial norms over the group that collectively owns the parameters. If one rank skips while another updates, parameters diverge and subsequent collective order or checkpoint state becomes invalid.

#### 6. Checkpointing interactions

Recomputation adds compute where parameter gathers or gradient reductions were expected to overlap, so the critical path can change. It may repeat collectives, random operations, or mutable side effects unless the region is designed for recomputation. Preserve RNG state, use the same process-group order, avoid external side effects, and profile the new timeline. Memory saved is useful only if recompute and additional communication still meet throughput.

## Tensor, Sequence, and Context Parallelism

LEAD: Intra-layer parallelism partitions the tensor dimensions of one model replica. It makes large layers or long activations fit, but puts communication inside every transformer block and can shrink local kernels below efficient shapes.

### Column-parallel linear layers

For a linear operation `Y = X A`, column parallelism partitions output columns of `A` across `p` ranks. Each rank computes a shard `Y_i = X A_i`. The input `X` is replicated across the tensor-parallel group, while the output feature dimension is sharded.

This maps naturally to Q, K, V, or the first MLP projection when attention heads or intermediate features divide evenly. A following operation that can consume the sharded features avoids an immediate all-gather.

Bias, activation, and gating should be applied locally when their partition matches. If the API materializes full `Y` unnecessarily, an all-gather erases the benefit and adds memory.

### Row-parallel linear layers

Row parallelism partitions the input rows of `A` and the corresponding feature dimension of `X`. Rank `i` computes a partial output `X_i A_i`; partials must be summed across ranks. The result is usually produced by all-reduce, or reduce-scatter when the next region can consume a sharded output.

A common transformer plan pairs a column-parallel expansion with a row-parallel contraction. The intermediate activation remains sharded, and only the final partial outputs require reduction. The same principle applies to attention projections: choose adjacent layouts so communication occurs at a small number of deliberate boundaries.

### Tensor layouts are types

Treat each tensor as carrying a distribution type such as:

- replicated;
- sharded on hidden, head, vocabulary, sequence, or batch dimension;
- partial, meaning local values require reduction;
- uneven or padded shard with a validity mask.

Operators transform these types. A compiler or framework can insert collectives, but a robust design should state them. Many accidental all-gathers come from an interface that forgets a tensor is already sharded.

Residual connections require compatible layouts. Dropout masks, bias, and normalization statistics must have a defined logical mapping so changing tensor-parallel degree does not silently change model semantics beyond the reproducibility contract.

### Vocabulary-parallel embedding and loss

Large vocabularies can shard embedding rows or output logits. For embedding lookup, each rank identifies token IDs it owns, produces local vectors for those IDs, and reduces or routes results. For vocabulary-parallel cross-entropy, compute a distributed log-sum-exp:

1. reduce the maximum logit across vocabulary shards;
2. compute local shifted exponentials and reduce their sums;
3. locate the target token's owning shard and reduce its logit contribution;
4. form loss and local gradient without gathering all logits.

Numerical stability and padding vocabulary entries matter. Sampling from sharded logits similarly needs exact global top candidates or distributed normalization, depending on the decoding policy.

### Local GEMM efficiency

Increasing tensor-parallel degree divides matrix dimensions. Communication grows as a fraction of step time, and local GEMMs may lose tile efficiency or tensor-core alignment. The useful degree is bounded by:

- hidden, head, and intermediate dimensions;
- batch-token dimension at the target workload;
- link latency and bandwidth;
- memory required for one shard;
- collective overlap opportunity.

Strong scaling eventually stops when smaller local work no longer amortizes collectives. Quantization or faster kernels can move that crossover to a lower degree because compute shrinks while communication may not.

### Sequence parallelism

Within a tensor-parallel region, operations such as layer normalization and dropout do not need the full hidden dimension to be replicated across ranks for every token. Sequence parallelism can partition activations along sequence for these regions, reducing replicated activation memory.

Transitions often use reduce-scatter to produce sequence shards and all-gather before operations that require another layout. It complements tensor parallelism; it does not by itself partition attention's all-token dependency.

Random operations need a logical mask policy. If dropout is intended to be invariant to partition degree, map randomness by global token and feature coordinate, not local rank index. If exact cross-degree identity is not promised, at least preserve distribution and rank consistency.

### Context parallelism

Context parallelism partitions the entire sequence and its activations across a group. Tokenwise linear and normalization operations run locally. Attention is different: local queries require keys and values from the full logical context.

That description is sufficient for training, where many query positions are processed together, but it is too coarse for serving. Prefill and autoregressive decode have different query-to-history ratios and therefore need different context-parallel layouts. The distributed-inference chapter distinguishes **prefill context parallelism (PCP)** from **decode context parallelism (DCP)**; the two should not be treated as interchangeable settings.

Do not confuse these with **phase-specific tensor parallelism**: a disaggregated service can run ordinary TP at one degree in its prefill pool and another degree in its decode pool. That changes model-group ownership across a KV handoff, whereas PCP and DCP partition sequence work or history. Part III's “Different TP sizes for prefill and decode” gives a concrete TP=4 to TP=2 handoff.

Implementations may:

- all-gather K/V for attention, paying memory;
- circulate K/V blocks in a ring while updating online attention state;
- exchange queries and accumulate partial results;
- use hierarchical variants aligned with topology.

Ring-style attention keeps only a K/V block at a time and overlaps transfer with attention compute. It needs an online softmax merge so partial score ranges combine exactly. Backward must route gradient contributions to the owners of Q, K, and V.

### Causal and ragged context

Causal attention has triangular work. A naive contiguous sequence partition gives early query blocks little valid K/V work and late blocks much more, creating imbalance. Zigzag or interleaved partitions can distribute early and late positions across ranks. The mapping must preserve output order and mask semantics.

Packed ragged sequences add boundaries and different lengths. Do not let an attention block cross sequence identity. Cost models should use valid query-key pairs, not padded maximum length alone. Load balance may require bucketing, packing, or a nonuniform block assignment.

### Choosing TP versus CP

Both can reduce activation memory, but they affect different work. Raising TP shrinks hidden-dimension GEMMs and adds collectives throughout the block. Raising CP shrinks sequence-local activations and attention queries while duplicating weights across the CP group and communicating attention context.

For very long sequence and already efficient hidden-dimension kernels, CP may preserve better local GEMM shapes than another TP step. For huge hidden layers that do not fit, TP remains necessary. Compose them only after measuring the communication paths and divisibility constraints.

### Topology and communicator concurrency

TP collectives are frequent and latency-sensitive, so keep TP groups on the fastest local fabric. CP transfers can be large and may span a broader domain if attention compute hides them. When TP and CP overlap, their communicator traffic can contend for the same links.

Schedule group operations in a consistent order across ranks and reserve streams/buffers deliberately. A theoretically independent CP ring and TP all-reduce can deadlock or serialize if communicator creation or launch order differs.

:::callout insight|Follow the tensor layouts
For every transformer sublayer, label input, intermediate, and output as replicated, sharded, or partial. The required collective and its bytes then become visible, and redundant gathers are hard to hide.
:::

### Design Exercises

1. Derive the collectives in paired column- and row-parallel linear layers.
2. How would you implement vocabulary-parallel cross-entropy without gathering logits?
3. Why can a larger tensor-parallel degree make both compute and communication efficiency worse?
4. Compare sequence parallelism with context parallelism.
5. Design causal context partitioning that avoids severe load imbalance.
6. How do you choose TP and CP degrees for a long-context transformer?

### Worked Solutions

#### 1. Paired linear layers

Column sharding replicates input `X`, partitions output columns of `A`, and produces a sharded feature activation with no reduction. A compatible activation and next linear consume that shard. Row sharding of the contraction computes partial full outputs from each input-feature shard, then all-reduces them, or reduce-scatters if the next region accepts a shard. The efficient pair avoids gathering the expanded intermediate and communicates only at the contraction boundary.

#### 2. Vocabulary-parallel loss

Reduce the maximum across vocabulary shards for stable logits, subtract it locally, reduce the local exponential sums into a global denominator, and reduce the target logit from the one owning rank. Each rank can then compute loss and its local gradient shard. Mask padded vocabulary entries and use consistent reduction precision. This avoids materializing batch-by-full-vocabulary logits on every rank.

#### 3. Excess TP

Each extra shard reduces local matrix dimensions and may break tile alignment, lower arithmetic intensity, and expose launch overhead. Collectives remain inside every layer; smaller messages can become latency-bound and a higher rank count adds steps or contention. The group also spans slower links once it exceeds a fabric domain. Profile local GEMM efficiency, collective exposure, and memory; stop strong scaling when step time or cost per token worsens.

#### 4. Sequence versus context

Sequence parallelism typically shards tokenwise activation regions associated with tensor parallelism, reducing duplicated normalization/dropout activations and using layout transitions around tensor-parallel blocks. Context parallelism partitions full network inputs and activations by sequence. Attention then communicates K/V or Q across context ranks because tokens interact. CP targets long-context activation/attention scaling; SP is a lighter complement to TP.

#### 5. Causal balance

Estimate valid query-key work per sequence block. Assign each rank a mixture of early and late query blocks, such as a zigzag mapping, so triangular work sums are similar. Circulate K/V blocks and skip masked interactions, preserve global position and output order, and handle ragged sequence boundaries explicitly. Validate both per-rank attention FLOPs and communication, because equal tokens do not imply equal causal work.

#### 6. TP and CP selection

Use enough TP to fit large hidden layers while keeping local GEMMs efficient and groups within fast links. Use CP when sequence-driven activation memory or attention work remains the constraint and another TP step would make hidden-dimension kernels too small. Sweep feasible factor pairs with actual batch/sequence, measure peak memory, TP and CP exposed communication, attention balance, and end-to-end tokens per second, then map groups to noncontending topology paths.

## Pipeline Parallelism and Hybrid Plans

LEAD: Pipeline parallelism partitions model depth and turns a batch into a schedule of activation and gradient messages. Throughput depends on filling stages, balancing their service times, and controlling the number of in-flight microbatches.

:::diagram parallelism_map|A hybrid plan partitions batch, hidden dimensions, sequence, layers, and experts. Each axis creates a different process group and communication boundary.

### Pipeline fundamentals

Partition a model into `s` ordered stages. A microbatch executes forward through the stages and backward in reverse, with activations and activation gradients sent between neighbors.

For a forward-only pipeline with `m` equal-time microbatches and balanced stages, a simple bubble fraction is:

`bubble = (s - 1) / (m + s - 1)`

Equivalently, useful stage slots are `m` out of `m+s-1`. For training schedules, exact utilization depends on forward/backward times, memory policy, and schedule. The approximation still shows why more microbatches amortize fill and drain.

More microbatches are not free. They change local matrix shapes, increase scheduler operations and in-flight activation state, and may force a smaller per-microbatch batch size. The global-batch and optimizer semantics must remain deliberate.

### GPipe-style schedule

A flush schedule runs forward for all microbatches, then backward for all. It is simple and has one model-weight version per batch, but retains activations for many microbatches and has a fill/drain bubble.

Activation checkpointing can reduce memory by storing stage inputs and recomputing internal activations. The recompute cost and communication schedule must be included in stage balance.

Flush boundaries simplify checkpointing and failure recovery because no microbatches from the next weight version are in flight.

### One-forward-one-backward

After warmup, a 1F1B schedule alternates forward and backward work. It reduces peak live activations because backward frees a microbatch's saved state earlier. It still has warmup and cooldown bubbles.

The number of in-flight forward microbatches differs by stage. Earlier stages may hold more activation state during warmup. Memory planning needs a per-stage schedule simulation rather than multiplying one activation estimate by the same count everywhere.

Schedulers must define send/receive ordering. Blocking operations issued inconsistently across adjacent stages can deadlock. Use explicit tags or sequence IDs, bounded buffers, and a schedule generated identically on every rank.

### Interleaved and virtual stages

A device can own multiple nonadjacent model chunks. Interleaving gives it useful work while another chunk waits, reducing bubble and improving balance. It also increases the frequency of activation transfers, complicates ordering, and can enlarge live state.

Virtual-stage count should follow measured imbalance and communication. Too many chunks turn large efficient stage work into small launches and more messages. The right goal is critical-path reduction, not the maximum number of virtual stages.

### Stage partitioning

Equal layer count rarely means equal time. Embeddings, output heads, long-context attention, MoE layers, and checkpointed segments have different compute and memory. Partition using:

- measured forward, backward, and recompute time;
- parameter and optimizer bytes;
- peak activation state under the chosen schedule;
- activation-transfer bytes between candidate cuts;
- topology and colocated communicator traffic.

The slowest stage sets steady-state cadence. A 10 percent imbalance can waste roughly that fraction even after bubbles are small. Dynamic sequence length may change which stage is slow, so evaluate workload buckets rather than one synthetic shape.

### Activation communication

Boundary activation bytes scale with microbatch, sequence, hidden dimension, and dtype. Tensor parallelism changes ownership: pipeline sends may be per-TP rank, gathered, or scattered depending on adjacent layouts. Preserve consistent rank pairing between stages so traffic uses intended links.

Compression can reduce bytes but adds conversion and numerical risk. It is attractive only when boundary transfer is exposed and the compressed representation is part of the training-quality contract.

### Hybrid parallel coordinates

Represent each rank by coordinates such as:

`(data, pipeline, tensor, context, expert)`

Not every axis is always independent; expert groups may fold with data or tensor dimensions. Write the exact group construction. A common dense-model relation is:

`world_size = DP times PP times TP times CP`

For each coordinate, list:

- tensors owned persistently;
- activations and parameters temporarily materialized;
- communicators and their physical paths;
- microbatch schedule and random-number mapping;
- checkpoint shard identity.

### A 64-GPU design example

Suppose eight nodes contain eight tightly connected GPUs each. A model needs more memory than one node but has large hidden layers and moderate sequence length. One candidate is:

- TP=8 within each node;
- PP=2 across pairs of nodes;
- DP=4 across four pipeline replicas;
- CP=1 for the initial sequence target.

TP traffic stays local. Pipeline boundaries cross nodes once per microbatch in each direction. Data-parallel reductions connect corresponding TP/PP coordinates across replicas and should be mapped across network rails.

This is a hypothesis, not a default. Compare with TP=4, PP=4, DP=4 or a sharded-data plan. The first may have better local TP bandwidth but larger pipeline stages; the second changes pipeline bubbles and per-rank GEMM shapes. Measure peak memory and step time under the same global batch.

### Selecting the global batch and microbatches

For local microbatch size `b`, accumulation microbatches `m`, and data degree `d`:

`B_global = b m d`

Increasing `m` reduces pipeline bubble but raises global batch unless `b` or `d` falls. `b` may already be one sequence, leaving optimization changes or sequence packing as the remaining levers. Do not silently change training hyperparameters to make a systems schedule look efficient.

### Plan evaluation

For every candidate plan, calculate and then measure:

- persistent and peak memory per rank and stage;
- local matrix dimensions and achieved compute;
- bytes, steps, and exposed time for every communicator;
- pipeline bubble, imbalance, and recomputation;
- data input and checkpoint bandwidth;
- failure blast radius and spare-capacity requirement.

Reject plans that fit only with zero allocator reserve or depend on perfect overlap. Include shape changes across curriculum, sequence-length ramp, or expert routing.

:::callout decision|Choose the simplest plan inside the performance envelope
Every additional axis multiplies process groups, state mappings, checkpoints, tests, and failure modes. Add a parallelism dimension only when it solves a measured memory or throughput constraint that a simpler plan cannot.
:::

### Design Exercises

1. Derive pipeline bubble and explain why more microbatches can still hurt throughput.
2. Compare flush and 1F1B schedules by memory, semantics, and recovery.
3. How would you partition heterogeneous transformer layers into stages?
4. Design rank coordinates and process groups for a 3D or 4D plan.
5. Evaluate the 64-GPU example against an alternative plan.
6. Why is the fastest isolated parallel plan sometimes wrong for the full training run?

### Worked Solutions

#### 1. Bubble and microbatches

With `s` equal forward-only stages and `m` microbatches, useful slots are `m` across `m+s-1`, so bubble fraction is `(s-1)/(m+s-1)`. Training schedules add backward and schedule-specific terms. Raising `m` lowers fill/drain fraction but may shrink local batch below efficient GEMM shapes, increase messages and scheduler overhead, increase live activations, or alter global batch. Sweep complete step time and memory, not the formula alone.

#### 2. Flush versus 1F1B

Flush runs all forwards then all backwards, retains more activations, and has clear weight-version and checkpoint boundaries. 1F1B alternates after warmup, releases activations earlier, and reduces peak memory, but has stage-dependent live state and more intricate ordering. Both can preserve synchronous update semantics when weights do not update until the batch completes. Recovery is simpler at flush points; mid-schedule restart needs microbatch commit state or a full-step replay.

#### 3. Stage partitioning

Profile forward, backward, recompute, parameter bytes, and boundary activations by layer and workload bucket. Solve for balanced maximum stage time subject to per-stage peak memory and legal cut points. Account for embeddings, loss, MoE, and long-context attention separately, then map adjacent stages to links and simulate the actual schedule. Validate with per-stage idle time because equal estimated FLOPs may hide memory or communication differences.

#### 4. Rank coordinates

Assign deterministic `(dp, pp, tp, cp)` coordinates whose product equals world size for the dense case. TP groups vary `tp` while other coordinates are fixed; PP groups vary `pp`; CP groups vary `cp`; DP groups vary `dp`. Map TP to local fabric, PP to low-contention neighbor paths, and DP across replicas/rails. Print memberships and include expert folding explicitly rather than assuming another independent product dimension.

#### 5. Sixty-four GPUs

For TP8-PP2-DP4, verify that TP8 local GEMMs remain efficient, two-node stage memory fits, pipeline activation traffic is acceptable, and DP groups use independent rails. Compare TP4-PP4-DP4: it may improve local GEMM size and lower TP communication but doubles pipeline depth and boundary count. Measure bubble at the same global batch, stage imbalance, peak bytes, collective exposure, and failure recovery. The winner follows the actual model and topology.

#### 6. Full-run objective

An isolated steady-state step excludes startup, data stalls, checkpoint pauses, sequence curriculum, evaluation, failures, and scheduler fragmentation. A plan may require a changed global batch, have fragile zero-reserve memory, or make checkpoints and resharded restore expensive. Choose time-to-valid-model and recovery-adjusted cost under the full run, while preserving optimization semantics and quality.

## Mixture-of-Experts and Sparse Communication

LEAD: Mixture-of-experts models increase parameter capacity while activating only a subset per token. The systems cost moves from dense compute to routing, variable all-to-all traffic, expert memory, and load imbalance.

### The MoE layer

For each token representation, a router scores experts and selects top `k`. The layer then:

1. computes expert assignments and weights;
2. counts tokens per destination expert;
3. packs tokens into expert-contiguous buffers;
4. dispatches tokens to expert owners, often with all-to-all;
5. executes expert MLPs as grouped or batched GEMMs;
6. sends outputs back to original token owners;
7. combines selected expert outputs by routing weights.

Only selected experts compute for a token, but all expert parameters must reside somewhere. Total parameter count drives memory and checkpoint size; activated expert count drives most FLOPs; routing distribution drives communication and local GEMM shapes.

### Capacity and overflow

If `T` tokens are routed across `E` experts with top-1 routing, ideal load is `T/E`. A capacity factor `c` might allocate approximately:

`capacity_per_expert = ceil(c T / E)`

Top-k routing increases total assignments. Real load is not uniform, so some experts can overflow while others idle. Possible policies include dropping overflow tokens, routing to a backup expert, increasing capacity and padding, or dynamically scheduling variable counts.

Dropping or rerouting changes model semantics and training. Padding preserves shape but wastes compute. Dynamic shapes improve useful work but complicate kernels, graphs, and collectives. State the policy in the model contract and report overflow by layer and data slice.

### Load-balancing objectives

Auxiliary losses encourage balanced token counts and router probability mass. They improve system utilization but can oppose expert specialization. Router z-losses, noise, or regularization may stabilize training. The exact objectives belong to the training recipe, not an invisible runtime fix.

Balance has several levels:

- token count per expert;
- weighted compute per expert when ranks or experts differ;
- bytes per destination rank;
- maximum expert time on the critical path;
- balance over a step, accumulation window, and data domain.

Equal counts do not guarantee equal time if sequence packing, expert dimensions, quantization, or hardware differ.

### Expert parallelism

An expert-parallel group partitions experts across ranks. Non-expert transformer layers may use data or tensor parallelism, while MoE layers enter expert-dispatch groups. Token ownership and expert ownership are different coordinates.

If a rank owns multiple experts, it sorts received tokens by local expert and executes a grouped GEMM. Very small expert batches underuse the accelerator. Combining tokens from multiple sequences or microbatches improves efficiency but may increase latency or saved activation memory.

Expert tensor parallelism can shard a single large expert. It lowers expert parameter bytes per rank but adds collectives inside expert computation, on top of dispatch traffic. Use it only when experts do not fit or local expert GEMMs are large enough.

### All-to-all cost

All-to-all traffic is personalized: each source sends a different partition to every destination. Total token bytes may appear balanced while one destination or network link is hot. Fixed-size padded all-to-all is predictable but sends empty capacity; variable-size exchange requires counts and offsets and can create launch variation.

Measure:

- dispatch and combine bytes;
- per-source/destination matrix, not only total bytes;
- packing/unpacking time;
- arrival skew before the collective;
- maximum rank receive count;
- expert GEMM occupancy and tail;
- overlap with attention or other layers.

Hierarchical dispatch can first exchange within a node, aggregate by remote destination, then traverse the network. It reduces small cross-node messages but adds a local pack stage.

### Token packing and ordering

Packing requires a histogram of assignments, prefix offsets, and scatter. Preserve original token identity so outputs return to the right sequence and position. For top-k, one token has multiple branch identities and combine weights.

Stable ordering may be needed for reproducibility or exact checkpoint replay. An unstable parallel sort can change floating accumulation order. Define whether results need bitwise identity or only distributional/numerical equivalence.

Padding, duplicate expert choices, masked tokens, and zero-token experts are edge cases. Test them directly; they often expose mismatched counts that hang all-to-all.

### Expert replication

Replicating a hot expert increases memory but splits its tokens across copies. A placement policy needs:

- expert popularity and variance by layer/domain;
- parameter bytes and load/eviction cost;
- routing distance and topology;
- consistency of replicated weights and optimizer state;
- deterministic mapping or acceptable stochasticity.

In training, replicas of one expert require gradient synchronization and consistent optimizer updates, creating another group. In inference, immutable weights simplify replication, but router-to-replica assignment affects cache locality and batching.

Dynamic expert migration during a step is rarely free. A slower-timescale placement controller can use recent routing histograms, move at safe boundaries, and keep a fallback copy until publication completes.

### Expert, data, and tensor group geometry

Expert parallelism does not always multiply independently with data parallelism. Non-expert weights may be replicated across an expert group while expert weights are partitioned, so gradient groups differ by parameter class. Some systems fold expert and data axes to use the same devices differently in dense and sparse layers.

Draw group membership for:

- dense parameters and their gradient reductions;
- each expert's replicas or shards;
- router parameters;
- token dispatch and return;
- pipeline/context boundaries.

A single global "data parallel degree" is insufficient for an MoE checkpoint or optimizer.

### MoE inference

Decode may provide only one token per active sequence. Per-expert batches become tiny and routing skew varies every iteration. The all-to-all latency can dominate even when experts are computationally cheap.

Serving strategies include larger continuous batches, expert replication, locality-aware request grouping, wide expert-parallel domains for capacity, and small local dispatch groups for latency. These goals conflict. Quantized expert weights reduce memory and load bandwidth but can make communication a larger fraction.

For strict token cadence, cap the work admitted to a group and protect against a rare hot-expert iteration. Report per-token tail latency by routed expert, not only average output tokens per second.

### MoE failure and observability

Failure of one expert owner can invalidate the entire model replica unless another consistent copy exists. Recovery must restore expert placement, router version, optimizer ownership, and process groups together.

Operational dashboards should show expert load histograms, overflow, entropy, dispatch matrix, per-expert compute, routed-token quality slices, and replica placement. A routing collapse may first appear as network imbalance, then training instability.

:::callout pitfall|Sparse FLOPs do not imply cheap execution
An MoE model can activate few parameters yet wait on two all-to-alls, token packing, a hot expert, and tiny grouped GEMMs. Capacity parameters still consume memory and checkpoint bandwidth even when inactive.
:::

### Design Exercises

1. Trace one token through distributed top-k expert routing and back.
2. Compare padding, dropping, backup routing, and dynamic capacity on overflow.
3. Why can balanced expert counts still produce an imbalanced step?
4. Design expert parallel groups and gradient groups for a hybrid MoE model.
5. When should a hot expert be replicated, and how is consistency maintained?
6. Why is expert-parallel decode often harder than expert-parallel training?

### Worked Solutions

#### 1. Token path

The router produces expert IDs and weights. Counts and prefix offsets pack a token copy for each selected expert with original token identity. All-to-all sends each packed copy to the expert owner, which groups by local expert and runs the MLP. A return all-to-all restores source ownership; outputs are scattered to original positions and combined by routing weights. Counts, masks, top-k branches, and ordering must match at both transfers.

#### 2. Overflow policies

Padding to fixed capacity gives predictable shapes but wastes compute and bandwidth. Dropping caps cost but changes model output and gradient. Backup routing preserves more tokens but changes specialization and can move the hot spot. Dynamic variable counts do useful work but create irregular collectives and kernels. Choose in the model/training contract, measure quality and tail performance, and expose overflow per layer and slice rather than hiding it in aggregate throughput.

#### 3. Hidden imbalance

Experts may have equal counts but different token shapes, hardware, quantization, or branch multiplicity. Packing work, source-destination network paths, and arrival times can differ. One rank may own several simultaneously hot experts. Padding to maximum count also makes every rank pay for the worst destination. Profile per-expert compute, per-rank receive bytes, dispatch matrix, and arrival skew; count balance is only one indicator.

#### 4. Group design

Give ranks explicit data, tensor, pipeline, context, and expert coordinates. Define separate gradient groups for dense parameters, router parameters, and each expert's replicas or shards. Define expert dispatch groups between token owners and expert owners, plus expert-TP groups if used. Map frequent dispatch within fast topology where possible and print ownership in checkpoints. Do not assume EP simply multiplies DP; parameter classes can have different replication geometry.

#### 5. Expert replication

Replicate when a persistent hot expert sets the critical path and saved queue/dispatch time exceeds parameter memory, load, and synchronization cost. Place copies near token sources and split assignments using load-aware deterministic mapping. During training, reduce gradients across copies and apply identical optimizer updates; publish placement only at a safe boundary. During inference, immutable replicas simplify consistency, but version and quantization identity must match.

#### 6. Decode difficulty

Training microbatches provide many tokens, creating larger expert GEMMs and amortizing all-to-all. Decode supplies roughly one token per sequence, so per-expert groups are tiny, skew changes each step, and two personalized collectives sit on token latency. Batching helps but conflicts with cadence and KV capacity. Expert replication and topology-aware grouping may be needed even when training used pure partitioning.

## Distributed Inference and Stateful Placement

LEAD: Distributed inference repeats communication for every generated token while carrying request-specific KV state. A layout that scales training throughput can fail interactive latency because decode has small local work and a strict serial cadence.

### Replication first

If a model and its working-state reserve fit on one device or node, independent replicas usually scale serving capacity most simply. Replicas run separate batches, isolate failures, and avoid per-layer cross-replica synchronization.

Model parallelism is needed when the model or KV does not fit, or when one request's latency benefits from multiple devices enough to offset communication. Use the smallest model-parallel group that satisfies fit and SLO; spend remaining devices on replicas.

The fastest single request plan can lower fleet goodput if it consumes more devices per replica and creates longer queues. Evaluate the full arrival process and failure reserve.

### Tensor-parallel decode

Tensor parallelism shards weights and compute but adds reductions or gathers in many layers. Decode local matrix height is approximately active batch size, so high TP degree produces small per-rank GEMMs and latency-bound collectives.

A simplified step decomposition is:

`T_step ~= T_local_compute + T_exposed_collectives + T_KV + T_host`

Quantization reduces local weight traffic and compute, which can make collectives a larger fraction. Batching raises local GEMM efficiency and amortizes communication, but lengthens each iteration and consumes more KV. Choose TP jointly with the scheduler's latency budget.

Collective fusion across operations is constrained by dependencies and residual paths. CUDA graph capture or persistent scheduling can reduce host gaps, but the distributed launch order and communicator buffers must remain stable.

### Pipeline inference

Pipeline stages reduce per-device weight memory. For one request, first-token and token latency include traversal across all stages. With many requests, stages can process different microbatches and improve throughput.

Continuous batching creates dynamic microbatches and output lengths. A stage may receive a different active set each iteration, so sequence IDs and KV ownership must be explicit. Pipeline bubbles arise from too few microbatches, stage imbalance, and pauses for long prefills.

Chunked prefill can create more uniform stage work. Separate prefill and decode pipelines can specialize schedules but introduce KV handoff, as Part III discusses. This chapter focuses on how the chosen parallel groups carry that state.

### KV ownership under model parallelism

KV layout follows attention partitioning:

- head-sharded TP stores only local KV heads when compatible;
- replicated KV simplifies access but multiplies capacity cost;
- context or sequence partitioning stores token ranges on different ranks;
- pipeline stages store KV only for their layers;
- expert parallelism normally does not own attention KV directly.

Grouped-query and multi-query attention may have fewer KV heads than TP ranks. Replicating a KV head across a subgroup can avoid empty shards but increases bytes. Alternatively, ranks can share or exchange K/V, adding latency. The best mapping depends on head count, context, and fabric.

Block tables and prefix references must use the same distributed ownership. A request is not admitted until every required rank can reserve its shard; otherwise one rank can OOM after peers have advanced.

### Phase-specific context parallelism

“Context parallelism” is not one inference plan. Prefill applies many queries to an expanding context and is usually optimized for time to first token. Decode applies one new query per active sequence to a large paged history and is usually optimized for inter-token latency, KV capacity, or batch goodput. A system can use different degrees and even different algorithms for these phases.

#### Prefill context parallelism

Prefill context parallelism (PCP) divides newly processed prompt tokens across PCP ranks. Each rank projects and attends for a subset of query positions, reducing the local attention work and activation footprint. It primarily targets long-prompt time to first token and contexts whose attention working set does not fit conveniently on one rank.

Two implementation families make different memory-communication trades:

- **partial query, full KV:** ranks keep local query chunks but exchange or gather the keys and values needed by those queries;
- **partial query, partial KV:** ranks keep both query and KV partitions and circulate blocks, often with ring attention and an online softmax merge.

The first is simpler but can reproduce the full-KV memory cost. The second bounds local KV working memory but introduces ordered communication and causal-load-balancing work. Contiguous causal partitions are imbalanced because late query blocks see more history; zigzag or interleaved mappings distribute early and late positions more evenly.

PCP is normally an additional process-group dimension. In vLLM's current group geometry, PCP is separate from tensor parallelism and therefore increases the world size for a fixed tensor-parallel group. Support remains version- and attention-backend-dependent, so a deployment must validate the exact release and kernel path rather than assuming that every context-parallel prefill algorithm is production-ready.

#### Decode context parallelism

Decode context parallelism (DCP) instead partitions the historical KV cache along the sequence dimension. Every rank evaluates the new query against its local history, then the ranks merge compact partial-attention state. For a shard `r`, let `m_r` be its maximum score, `l_r` its shifted exponential sum, and `o_r` its shifted weighted-value sum. The exact global result is:

:::equation m = max_{r} m_{r}|The global maximum provides a common numerical reference across history shards.

:::equation l = Σ_{r} exp(m_{r} - m) l_{r}|Shifted local normalizers combine into the exact global softmax denominator.

:::equation o = (1 / l) Σ_{r} exp(m_{r} - m) o_{r}|Only compact statistics need to be reduced; the historical KV tensors need not be gathered.

Interleaving token blocks across DCP ranks spreads future cache growth and attention work more evenly than assigning each rank one permanently contiguous interval.

In the vLLM layout with PCP disabled, DCP reuses ranks inside each tensor-parallel group, and the tensor-parallel degree must be divisible by the DCP degree. This matters most for grouped-query and multi-query attention. With equal head partitions and a compatible TP degree, ordinary TP first shards KV heads; when the tensor-parallel degree `T` exceeds the number of KV heads `H_{kv}`, each head is repeated across `T / H_{kv}` ranks. DCP can use those otherwise duplicate-bearing ranks to shard each head's token history instead.

:::equation 1 ≤ D ≤ T / H_{kv}|Replication-removal sizing for compatible GQA layouts with PCP disabled and T at least H_kv; not a universal DCP limit.

For example, with four KV heads and TP=8, each head has two copies before DCP. Choosing DCP=2 lets each copy-bearing pair store complementary token blocks. There are still eight GPUs, but the pair stores one logical history instead of two full copies. This is sequence sharding inside the allocated model group, not a TP=8 prefill pool handing off to a TP=2 decode pool.

Increasing DCP reduces duplicated KV capacity and may admit a larger decode batch, but it adds communication to every attention layer and generated token. It is therefore not automatically a token-latency optimization. The gain often appears first as more cache headroom, fewer evictions, or higher SLO-compliant fleet goodput.

#### Group geometry and mixed phases

For vLLM's model-worker rank space, excluding API processes and independently launched pools, the layout can be summarized as:

:::equation WorldSize = DP × PP × PCP × TP|DCP reuses this rank space rather than adding another multiplier; the product does not imply every combination is supported.

The implementation is evolving. In [vLLM's parallel configuration at revision 252ed876](https://github.com/vllm-project/vllm/blob/252ed876214a0a01a6d0ce93bb5bbe77685a4160/vllm/config/parallel.py), checked September 7, 2026:

- with PCP=1, DCP must divide TP;
- with PCP greater than one, the accepted DCP sizes are 1, PCP, or TP multiplied by PCP;
- PCP greater than one cannot yet be combined with DP greater than one.

Thus “DCP is always nested inside TP” is too broad: combined layouts can span the PCP axis or the full TP-by-PCP block. These are configuration constraints in the cited revision, not a guarantee that every model or attention backend supports each accepted combination. The older replication-removal bound above explains the no-PCP case; it must not be applied to every combined layout.

| Mechanism | Primary shard | Adds ranks for fixed TP? | Main serving objective | Recurring cost |
| --- | --- | --- | --- | --- |
| TP | weights, activations, and usually KV heads | yes | model fit and layer compute | per-layer reductions or gathers |
| PCP | prompt query positions; optionally KV blocks | yes | long-prompt TTFT and working-set fit | KV exchange or ring traffic during prefill |
| DCP | historical KV tokens across existing ranks | no | remove KV duplication, enlarge batch or context | partial-attention merge during decode |

Chunked prefill blurs the phase boundary. A request may already have DCP-sharded history while a new prompt chunk supplies many query positions. Prefix-cache hits create the same condition. The engine must combine the new chunk with distributed historical state; the label “prefill” alone does not determine which communication appears in the kernel.

A sound selection sequence is:

1. choose the smallest TP degree that fits weights and meets layer latency;
2. add PCP only when long-prompt TTFT or prefill attention memory remains limiting;
3. add DCP when TP has created KV-head duplication or decode capacity is history-bound;
4. measure phase-specific latency, cache occupancy, communication, batch size, and fleet replica count together;
5. verify the current runtime's model, attention-backend, chunked-prefill, speculative-decoding, and multi-token-prediction compatibility.

Do not copy a training CP degree into serving by precedent. PCP and DCP solve different bottlenecks, have different process-group geometry, and can move opposite SLOs.

### KV transfer and remote memory

Migration or prefill-decode disaggregation transfers state between model-parallel groups. A complete transfer plan specifies:

- logical token/layer/head ranges;
- source and destination rank mapping;
- packed page layout, dtype, and scales;
- direct versus staged network path;
- chunk order and overlap;
- integrity, idempotency, and ownership transfer.

If source TP and destination TP degrees differ, the transfer includes resharding. Avoid collecting the entire KV on one coordinator. Each source sends the intersections of its logical shards with destination ownership.

Remote KV memory is useful for inactive sessions or pooled capacity when transfer/recompute cost is below holding scarce device memory. Fetching remote pages on every token makes the network part of the attention critical path and requires a much stronger availability and tail-latency contract.

### Replica routing with distributed state

Route on estimated completion, not request count. Inputs include queue delay, model/adapter residency, prefix locality, KV capacity on every rank, transfer or recompute time, topology health, and deadline.

State affinity should be bounded. Prefer the owner while saved work exceeds its extra queue delay; otherwise migrate, recompute, or start cold. A tensor-parallel group's capacity is the minimum admissible shard across its ranks, not the sum of free bytes.

Hot prefixes can be replicated across complete model groups. Partial replication is useful only if every consumer has a defined way to access the missing shards without creating a worse network dependency.

### Expert-parallel serving

An MoE inference group adds token dispatch to attention and tensor communication. Small decode batches make expert GEMMs inefficient. Replicate popular experts, co-batch compatible tokens, or keep EP within fast links where memory permits.

The router and scheduler can cooperate: predicted expert affinity may improve placement, but future routes are input-dependent and exposing or storing them can raise complexity. Always keep a capacity-safe fallback for unexpected hot experts.

### Streaming failure semantics

A failure of any rank generally invalidates the model-parallel replica. A replacement group can resume only with compatible weights, KV shards, sampling state, constraint state, and the exact emitted-token commit point.

Replication supplies spare groups, not automatic state replication. Recomputing KV from prompt and accepted output is often the simplest recovery before the first token; after streaming begins it may exceed latency and must avoid duplicate visible bytes.

Use request IDs and token sequence numbers. If exact resume cannot be proven, terminate the partial stream clearly rather than emitting a divergent continuation. Part III defines the product contract; the distributed layer must preserve it across all ranks.

### SLO-constrained scaling

Report:

- first-token and inter-token latency by group layout;
- input and output goodput per accelerator;
- collective and KV-transfer tail time;
- queueing under complete replica count;
- memory headroom on the most constrained rank;
- performance after one group or network path fails.

Strong-scaling efficiency is secondary to good requests per fleet cost. A plan that halves kernel time but doubles devices and queueing is not an inference win.

:::callout decision|Inference parallelism spends a device on either one request or another replica
Every added rank must justify the requests it prevents the fleet from serving independently. Fit, latency, queueing, and failure reserve decide together.
:::

### Design Exercises

1. Why is tensor parallelism more latency-sensitive in decode than in training?
2. Design distributed KV ownership for TP, PP, and GQA.
3. Compare PCP and DCP: what does each shard, when does it add ranks, and which serving bottleneck does it address?
4. How do you reshard KV during migration without a central bottleneck?
5. Design state-aware routing to a model-parallel group.
6. Specify safe recovery after one rank fails during streaming generation.

### Worked Solutions

#### 1. Decode sensitivity

Decode has a small token dimension and repeats a serial step for every output token. TP shrinks already-small local GEMMs while collectives remain inside many layers, so latency and launch overhead are exposed. Training has larger microbatches and long backward compute that better amortize or overlap communication. Quantization can worsen the ratio by accelerating compute without reducing collective latency.

#### 2. KV ownership

PP stages own KV only for their layers. Within a stage, TP ranks own compatible attention heads when KV-head count permits. If GQA has fewer KV heads than TP ranks, either form KV-sharing subgroups, replicate selected heads, or lower attention TP; document the added bytes or exchange. Block tables and reservations are sharded identically, and admission succeeds only when every stage/rank reserves its required pages.

#### 3. Prefill versus decode context parallelism

PCP shards prompt query positions across a separate process-group dimension and may also partition the KV working set. It can reduce long-prompt TTFT or make the prefill attention working set fit, but it usually consumes additional ranks and communicates KV blocks or online-attention state. DCP shards historical KV tokens among already allocated ranks: inside TP when PCP is disabled, or across supported PCP/TP layouts when it is enabled. In vLLM it does not increase world size. Removing duplicated GQA or MQA KV heads is one important use case. DCP can recover cache capacity and raise decode batch goodput, but adds a partial-attention merge at every layer and token. Benchmark PCP on prompt length and TTFT; benchmark DCP on context, cache occupancy, batch size, inter-token latency, and SLO-constrained goodput.

#### 4. KV resharding

Describe KV in logical layer, token, head, and feature ranges independent of source layout. Each source rank intersects its owned ranges with destination ranges and sends chunks directly to the corresponding destination ranks. Pre-reserve destination pages, include version/dtype/scale metadata and checksums, and commit ownership after all chunks are visible. Avoid assembling full KV on a coordinator; distribute metadata planning separately from data movement.

#### 5. Group routing

Treat a model-parallel group as one service unit whose admissible memory is constrained by its tightest rank. Estimate queue plus cache miss/recompute or transfer plus execution, adjusted for adapter, topology health, and SLO. Prefer state locality only while it reduces completion time, replicate hot immutable prefixes to whole groups, and retain cold-start fallback. Monitor prediction error and rank-level imbalance.

#### 6. Rank failure during streaming

Abort and quarantine the whole synchronized group. Identify the last token committed by the model and acknowledged by the transport. Resume on a compatible spare only if prompt, accepted tokens, distributed KV or deterministic recompute, RNG, adapter, constraints, and model version reconstruct exactly, with duplicate suppression by token sequence. Otherwise end the stream with a clear partial-response failure and release all sharded state idempotently.

## Distributed Checkpoints, Recovery, and Elasticity

LEAD: A checkpoint is a distributed commit of model meaning. It must bind logical tensors, optimizer and scheduler state, data position, random streams, and topology-independent ownership into one recoverable publication.

### What must be recoverable

A training restart commonly needs:

- model parameters and nonparameter buffers;
- optimizer moments, master weights, and step counters;
- learning-rate and loss-scale scheduler state;
- data-loader epoch, shard, sample, and packing position;
- random-number states for data, dropout, and model operations;
- parallelism metadata and tensor layout;
- training recipe, code/artifact version, and global progress;
- any curriculum, mixture weights, or dynamic routing state.

Saving only weights supports inference or a warm start, not an exact training continuation. Define checkpoint classes such as recovery, milestone, export, and evaluation artifact. They have different contents, retention, and compatibility promises.

### Logical versus physical state

A logical tensor has a global name, shape, dtype, and semantic role. A physical checkpoint contains shards written by ranks under one parallel plan. The manifest maps logical ranges to immutable objects with checksums and ownership metadata.

This separation enables resharded restore. A checkpoint saved with DP8-TP8-PP4 may load into a different compatible plan by computing intersections between stored logical ranges and destination ranges. Avoid formats that encode only rank-local filenames without global offsets.

Optimizer state follows parameter identity, not current rank. Flattening, padding, expert placement, or pipeline cuts can change physical layout. Stable logical IDs and versioned transformation rules are essential.

### Atomic publication

One safe protocol is:

1. coordinator allocates a unique checkpoint ID and immutable object prefix;
2. every rank writes assigned shards and reports size/checksum;
3. metadata validation confirms complete logical coverage without illegal overlap;
4. a manifest is written after all required objects are durable;
5. a small atomic pointer or catalog entry publishes the manifest;
6. retention removes unreachable older objects only after a grace period.

Readers see either the previous complete pointer or the new complete pointer. A partially written checkpoint is never named as latest. Object stores may provide atomic creation of one manifest object even when directory rename is unavailable.

Manifest publication must be idempotent. A retry with the same checkpoint ID either discovers the same complete contents or writes a new immutable attempt; it must not mix shards from different steps.

### Checkpoint bandwidth model

For total checkpoint bytes `C`, aggregate sustainable storage bandwidth `B`, and fixed coordination overhead `L`, a lower bound is:

`T_checkpoint >= C / B + L`

The slowest writer, metadata service, shared network, or storage partition may determine `B`. Per-rank bandwidth multiplied by ranks can exceed the storage system's aggregate limit.

Checkpoint traffic also contends with training collectives and input reads. A burst of synchronized writers can collapse the shared filesystem. Stagger writes, aggregate locally, use topology-aware writers, or provision a separate path when necessary.

Measure time paused, time until durable publication, device-to-host staging bytes, pending checkpoint memory, and impact on concurrent training steps.

### Synchronous and asynchronous save

A synchronous save pauses training until required state is durable. It is simple and bounds the version of tensors being written, but the pause can be large.

An asynchronous save first snapshots or stages a consistent state, then writes while training continues. The snapshot must remain immutable. Copying to host frees device state sooner but consumes host memory and link bandwidth. Copy-on-write or double buffering can reduce pause but may temporarily double large state.

Backpressure is mandatory. If storage is slower than checkpoint production, do not queue unbounded snapshots. Skip a nonessential interval, block at a safe point, or lower frequency while preserving the recovery objective.

### Incremental and local checkpoints

Incremental checkpoints store only changed chunks relative to a base. They save bytes when change is sparse or compression/deduplication is effective, but optimizer tensors often change everywhere. Long dependency chains slow restore and complicate retention; periodically compact into a full base.

Local node or rack checkpoints provide fast recovery from process failure but disappear with larger failure domains. Remote durable checkpoints cover rack or site loss at higher cost. A tiered policy can write frequent local recovery points and less frequent durable milestones.

Parity or erasure coding can tolerate lost shards without full replication, but encoding consumes compute/network and recovery reads many fragments. Use it when object loss risk and storage economics justify the complexity.

### Restore and resharding

A restore planner should:

1. validate manifest, recipe, and requested topology compatibility;
2. allocate destination tensors in their new layouts;
3. assign read ranges across ranks to balance storage and network;
4. stream stored chunks directly into destination shards where possible;
5. transform dtype, padding, or layout only under a versioned rule;
6. reconstruct optimizer, RNG, data, and scheduler state;
7. run cross-rank validation before the next optimizer step.

Avoid loading a full tensor onto one rank and broadcasting it. Parallel reads and range intersection scale better. Test restore into the actual disaster topology, not only the same healthy cluster.

Validate checksums, finite values, representative tensor hashes, global step, optimizer ownership, and a deterministic short replay when feasible.

### Failure interval and checkpoint frequency

Checkpointing too often wastes time; too rarely loses expensive work. If a synchronous checkpoint costs `C_time` and the mean time between relevant failures is `M`, a common first-order heuristic places the useful interval on the order of:

`sqrt(2 C_time M)`

This assumes independent failures and roughly constant checkpoint cost. Large jobs have a lower job-level mean time between failures than one device. Include restart time, queue reacquisition, local versus durable failures, and asynchronous overlap before applying the heuristic.

Optimize expected time to a valid model, not only average step throughput. A plan with slightly slower steps but much faster checkpoint and recovery can finish sooner.

### Coordinated failure handling

Synchronous model-parallel groups usually fail as a unit. When one rank faults:

- stop launching new collectives;
- surface the first known error, not a cascade of timeouts;
- capture rank and communicator state;
- terminate or fence all group members;
- release cluster resources and stale rendezvous state;
- restart from a verified checkpoint or a documented in-memory recovery point.

Trying to keep healthy ranks alive indefinitely can make recovery slower and state ambiguous. Use bounded timeouts and a control-plane lease so an old group cannot rejoin after replacement.

### Elasticity

Data-parallel replicas are the easiest axis to change because model ownership within each replica remains complete. Even then, changing DP changes global batch or accumulation unless adjusted, and it changes sample assignment and random streams.

Changing TP, PP, CP, or EP requires resharding model and optimizer state, rebuilding communicators, and possibly recompiling engines. Treat this as a coordinated restart unless the framework explicitly supports a safe online transition.

Elastic recovery is not free capacity. Keep spare nodes or accept scheduler wait, and include rendezvous, artifact load, cache warmup, and checkpoint read in recovery time.

### Recovery drills

Automate tests for:

- rank process crash during compute and collective;
- node loss during checkpoint write;
- corrupt or missing shard;
- restore with different DP or model-parallel degree;
- stale latest pointer and duplicate publication retry;
- storage throttling and unavailable metadata service;
- data-loader and RNG continuation correctness.

A checkpoint that has never been restored is an untested backup. Track restore success and duration as release metrics.

:::callout decision|Optimize recovery-adjusted throughput
The useful rate is training progress that survives expected failures. Include checkpoint pause, storage contention, lost work, restart, resharding, and queue delay when comparing parallel plans.
:::

### Design Exercises

1. Design atomic publication for a multi-terabyte sharded checkpoint.
2. What distinguishes a recovery checkpoint from a model export?
3. How can asynchronous checkpointing corrupt state or cause OOM?
4. Design topology-independent resharded restore.
5. How would you choose checkpoint frequency for a large cluster job?
6. What does safe elasticity require beyond restarting with fewer data-parallel ranks?

### Worked Solutions

#### 1. Atomic checkpoint

Write all shards under a unique immutable attempt ID, each with logical ranges, size, and checksum. Validate complete tensor coverage and required global state, then write one immutable manifest and atomically publish a pointer/catalog record. Readers ignore unreferenced partial attempts. Publication and retries are idempotent, and garbage collection waits past a grace period so it cannot delete shards still referenced by readers or the previous checkpoint.

#### 2. Recovery versus export

A recovery checkpoint includes optimizer, scheduler, loss scale, global step, data position, RNG, parallel-layout metadata, and recipe identity needed to continue training. An export may contain only inference weights, tokenizer/configuration, and serving metadata in a consolidated or quantized form. A milestone may retain additional audit lineage. Name the class so operators do not attempt exact continuation from an artifact that lacks state.

#### 3. Async risks

If training mutates tensors while writers read them, one checkpoint contains mixed steps. Staging solves consistency but consumes device/host memory and bandwidth; multiple pending stages can OOM. Copy completion and object durability are different boundaries. Use an immutable snapshot, explicit stream synchronization, bounded outstanding saves, checksummed shards, and publish only after durability. Apply backpressure when storage falls behind.

#### 4. Resharded restore

Describe tensors globally by stable identity, shape, dtype, and logical ranges. The destination allocates its new shards and a planner intersects them with stored ranges, assigning parallel reads directly to consumers. Versioned transformations handle flattening, padding, dtype, pipeline cuts, and experts. Rebuild optimizer ownership, RNG, data, and schedulers, then validate hashes and a short replay. Never require one rank to materialize every full tensor.

#### 5. Frequency

Estimate synchronous or exposed checkpoint cost, job-level failure interval, restart/queue time, and value of lost work. The square-root heuristic gives a starting interval near `sqrt(2 C_time M)` under simple assumptions. Simulate failure domains and tiered local/durable saves, then optimize expected time to valid completion under storage limits. Revisit as cluster size, checkpoint bytes, or failure rate changes.

#### 6. Elasticity

Changing DP requires consistent new rank membership, adjustment of local batch or accumulation to preserve the intended global batch, sample reassignment, RNG policy, optimizer ownership, and scheduler progress. Changing model-parallel axes additionally needs resharding, communicator rebuild, engine compatibility, and new placement. Fence the old group, publish one membership epoch, and resume only after all ranks validate the same checkpoint and plan.

## Cluster Scheduling, Observability, and Distributed Diagnosis

LEAD: A distributed job is a coordinated tenant of accelerators, network, storage, CPUs, and control-plane services. Production performance depends on placement, reproducibility, and the ability to identify the first rank or resource that diverges.

:::diagram system_design|The distributed system has a control loop: placement and policy shape execution; telemetry, failures, and checkpoints feed the next scheduling decision.

### Job specification and rendezvous

A reproducible job specification binds:

- code and container or environment digest;
- model architecture, data, tokenizer, and training recipe versions;
- world size and every parallel degree;
- rank-to-device and group mapping policy;
- network, storage, CPU, and memory requirements;
- checkpoint input/output and recovery policy;
- retry, timeout, priority, and preemption semantics;
- observability and security configuration.

Rendezvous assigns one membership epoch and rank mapping. Every process must agree before communicator creation. Reusing stale addresses or allowing two epochs to write the same checkpoint namespace can corrupt a run.

The control plane should fence old workers with leases or epochs. A restarted rank is not allowed to join an existing synchronous group casually; the whole group transitions through a supported recovery protocol.

### Gang scheduling and placement

Synchronous jobs need all required ranks, so gang scheduling avoids holding partial allocations that cannot progress. Large rigid gangs can wait behind fragmented free GPUs even when aggregate capacity is sufficient.

Topology-aware placement chooses nodes and rails that satisfy the communication plan. Pack latency-sensitive TP groups tightly; place DP groups to exploit network bandwidth; avoid sharing oversubscribed links with another communication-heavy job when the scheduler can see that demand.

Placement objectives include:

- expected step time and variance;
- queue delay and fragmentation;
- failure-domain spread or concentration;
- locality to data/checkpoints;
- power, cooling, and maintenance constraints;
- preemption and replacement availability.

Packing one job into a small failure domain improves communication but a single failure loses more of its ranks. Spreading replicas improves fault isolation but may cross slower links. Make the trade explicit.

### Admission and quotas

Reserve accelerators, host memory, network injection, storage bandwidth, and checkpoint capacity. GPU count alone permits too many jobs to saturate a shared filesystem or fabric.

Tenant quotas can be expressed in accelerator time, priority-weighted queue share, or reserved pools. Backfilling short jobs improves utilization if it cannot delay reserved starts. Preemption needs a checkpoint or restart-cost model; killing a job just before its next durable point wastes more than its nominal GPU time.

Use predicted checkpoint and startup duration in scheduling. A large job that needs thirty minutes to restore should not be repeatedly preempted for brief capacity gaps.

### Step-time decomposition

Measure each step as a distributed critical path:

`input -> forward -> backward -> optimizer -> checkpoint/eval boundaries`

Within it, separate compute kernels, exposed collectives, rank arrival skew, pipeline idle, host gaps, memory stalls, and storage. The maximum rank determines synchronous step time, but the earliest divergence often occurs before the maximum event.

Report tokens or samples per second, model FLOP utilization with a declared FLOP convention, scaling efficiency, and time-to-valid-model. FLOP utilization can look low because of sparse experts, recomputation, padding, or a mismatched numerator; define what counts.

### Rank-synchronous observability

Useful telemetry includes:

- global step, microbatch, pipeline slot, and membership epoch;
- rank coordinates and host/device/link identity;
- collective sequence number, group, type, bytes, start, and completion;
- kernel and data-loader spans;
- allocator and activation peak memory;
- expert load and context/pipeline balance;
- checkpoint stage and storage throughput;
- hardware errors, throttling, and network counters.

Align clocks or use logical sequence IDs. Wall-clock traces from unsynchronized hosts can invert causality. Aggregate percentiles without losing per-rank maxima and identities.

Control cardinality: ranks and steps belong in traces or bounded debug logs, while durable metrics aggregate by job, stage, node class, and group. During an incident, promote sampled detail for the affected job.

### Straggler diagnosis

A slow step can originate from:

- data or CPU preprocessing on one rank;
- a long kernel due to shape, thermal throttling, or memory fault;
- pipeline stage imbalance;
- expert routing skew;
- collective transport or link errors;
- another job contending for NIC or storage;
- host scheduling and progress-thread delay.

Find the earliest event whose start or completion diverges. A late collective start indicates upstream skew; aligned starts with a late completion indicate communication or one rank failing to progress. Compare the slow rank with its physical neighbors and logical peers.

Do not begin by averaging. The mean rank is not on the critical path.

### Hang and desynchronization debugging

For each communicator, capture a bounded flight recorder of collective sequence, operation, tensor metadata, stream, completion state, and call stack. On timeout, gather snapshots from all reachable ranks.

Look for:

- a sequence present on some ranks but absent on another;
- the same sequence with mismatched operation or count;
- ranks blocked before the communicator;
- a prior device error;
- cyclic waits across multiple communicators;
- stale membership or duplicate rank.

Reproduce with synchronization checks and smaller scale if timing permits, but remember that added barriers can hide races. Preserve the original flight recorder first.

### Network and hardware isolation

Run layered tests:

1. single-device kernel and memory health;
2. pairwise device links within a node;
3. local collectives;
4. GPU-to-NIC and host affinity;
5. pairwise inter-node paths by rail;
6. collective benchmarks on the exact group;
7. application replay with competing traffic.

Quarantine nodes or links that show repeated correctable errors, bandwidth cliffs, or timeouts. A retry that lands on the same bad path is not recovery.

Firmware, drivers, runtime, collective library, and topology discovery form a compatibility set. Canary infrastructure changes on representative distributed jobs, not only single-GPU tests.

### Reproducibility and semantic identity

Record model, tokenizer, data snapshot, packing, optimizer, precision, kernels, collective library, topology, parallel plan, checkpoint manifest, and membership history. Two jobs with the same source commit can differ because container, data order, or group layout changed.

Bitwise reproducibility across rank counts is often impractical due to reduction order and random mapping. Define the promised level: exact same-plan replay, numerical tolerance, or statistical training equivalence. Deterministic modes can cost performance and still require deterministic data and kernels.

### End-to-end design method

A robust distributed design proceeds in this order:

1. define training/inference objective, SLO, batch semantics, and failure tolerance;
2. build per-rank memory and per-step FLOP/byte models;
3. choose the simplest parallel axes that make the workload fit;
4. map process groups to physical topology;
5. simulate schedules and collective critical paths;
6. plan checkpoints, recovery, and capacity reserve;
7. benchmark components, then full-step and failure scenarios;
8. instrument the invariants and automate regression gates.

Change one axis at a time when possible. Autotuning can search degrees, but constraints and cost models should prune invalid plans before expensive runs.

### Production readiness checklist

| Area | Required evidence |
| --- | --- |
| Fit | Peak per-rank memory with allocator and failure reserve |
| Compute | Local shapes, achieved kernels, recomputation cost |
| Communication | Bytes, groups, topology paths, overlap, tail ranks |
| Schedule | Pipeline/accumulation semantics, bubble, imbalance |
| Correctness | Cross-rank invariants, numerical tests, deterministic contract |
| Recovery | Atomic checkpoints, resharded restore, fault drills |
| Operations | Placement, quotas, alerts, flight recorder, runbooks |
| Economics | Queue plus run time, recovery-adjusted cost, useful output |

### Design Exercises

1. Design topology-aware scheduling for several competing large training jobs.
2. How do you distinguish a network slowdown from rank arrival skew?
3. Design a flight recorder for distributed collective hangs.
4. What should be included in distributed job semantic identity?
5. Compare strong-scaling efficiency with time-to-valid-model as objectives.
6. Walk through the first hour of diagnosing a 5 percent intermittent step-time regression.

### Worked Solutions

#### 1. Topology-aware scheduling

Collect each job's process-group graph, bandwidth/latency demand, gang size, duration, checkpoint/startup cost, and failure policy. Pack TP and other latency-sensitive groups into fast fabric domains, place cross-node groups across sufficient rails, and account for shared storage/NIC contention. Use gang admission, reservations, backfill that cannot delay starts, and preemption based on recoverable work. Balance communication locality against failure-domain and fragmentation cost.

#### 2. Network versus arrival skew

Record per-rank collective enqueue/start and completion using logical sequence IDs. If one rank starts late and peers wait, inspect that rank's preceding input, kernel, pipeline, expert, or host work. If starts align but completion stretches, inspect link/NIC counters, route, contention, transport retries, and progress. Compare the same group/message size across healthy nodes and a topology-matched microbenchmark. The collective span alone cannot separate the cases.

#### 3. Flight recorder

Maintain a bounded per-process-group ring buffer containing membership epoch, communicator ID, sequence number, operation, tensor count/dtype, stream, enqueue/completion timestamps, and call stack. On timeout or signal, persist local buffers and aggregate reachable ranks. The analyzer finds missing sequences, mismatched operations/counts, earliest uncompleted calls, prior device errors, and cycles across communicators. Keep overhead bounded and redact data values.

#### 4. Semantic identity

Bind code/container, model architecture, tokenizer, data snapshot and order, optimizer/scheduler, precision/loss scaling, random seeds and policy, kernels and collective runtime, complete parallel layout/topology, membership epochs, and checkpoint manifest. Add training recipe and evaluation version. A source commit or weight checksum alone cannot explain user-visible or optimization behavior.

#### 5. Scaling versus valid model

Strong-scaling efficiency measures step-speed gain as devices increase at fixed work. It can reward a fragile plan with high checkpoint cost, changed batch semantics, long queue/startup, or poor recovery. Time-to-valid-model includes queueing, data, evaluation, checkpoint pause, lost work, failures, and quality convergence. Use component scaling to diagnose efficiency, but choose plans by recovery-adjusted cost and time to the accepted artifact.

#### 6. Regression diagnosis

Confirm version, workload, topology, and measurement changes; compare the same step and shape buckets. Split the regression into input, kernels, exposed collectives, arrival skew, pipeline idle, optimizer, and checkpoint interference. Identify affected ranks/nodes/rails and earliest divergence from synchronized traces. Check hardware throttling/errors and competing jobs, run topology-matched collectives, bisect software/configuration, and mitigate by quarantining or rollback while preserving evidence.

### Further Study and Primary References

- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) - collective semantics, communicators, topology, and runtime behavior.
- [PyTorch Distributed Documentation](https://docs.pytorch.org/docs/stable/distributed.html) - process groups, collectives, and debugging controls.
- [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [FullyShardedDataParallel](https://docs.pytorch.org/docs/stable/fsdp.html) - replicated and sharded execution contracts.
- [ZeRO](https://doi.org/10.1109/SC41405.2020.00024) - optimizer, gradient, and parameter-state sharding.
- [Megatron-LM](https://arxiv.org/abs/1909.08053) and [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473) - tensor, pipeline, and composed parallelism.
- [Megatron Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) and [Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) - maintained parallel layouts and long-context execution.
- [GPipe](https://papers.nips.cc/paper_files/paper/2019/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html) - microbatch pipeline parallelism.
- [GShard](https://arxiv.org/abs/2006.16668) and [Switch Transformers](https://www.jmlr.org/beta/papers/v23/21-0998.html) - sparse expert routing and scaling.
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) - parallel save/load and resharded restore.
- [PyTorch Flight Recorder](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html) - collective hang and desynchronization diagnosis.
- [vLLM Context Parallel Deployment](https://docs.vllm.ai/en/v0.16.0/serving/context_parallel_deployment/) - maintained distinction between prefill and decode context parallelism, algorithms, and deployment guidance.
- [vLLM Parallel Configuration](https://github.com/vllm-project/vllm/blob/252ed876214a0a01a6d0ce93bb5bbe77685a4160/vllm/config/parallel.py) and [Parallel-State Source](https://github.com/vllm-project/vllm/blob/252ed876214a0a01a6d0ce93bb5bbe77685a4160/vllm/distributed/parallel_state.py) - revision checked September 7, 2026, including combined PCP/TP/DCP geometry; distinguish this from the older deployment guide's no-PCP examples.

### Final Distributed Systems Principle

*A parallel plan is correct only when tensor ownership, communication order, physical placement, numerical semantics, and recovery state agree.*

Scale is not the number of accelerators allocated. It is the amount of valid progress preserved per unit time and cost after communication, imbalance, queueing, checkpointing, and failure are included.
