# Part VIII - Field Reference

The appendices are designed for the final week before an interview and for architecture reviews after it. Each item is a retrieval cue; return to the chapter when the assumptions are not obvious.

## Formula and Capacity Sheet

### Training

| Quantity | Approximation | Use |
| --- | --- | --- |
| Global batch | `microbatch * accumulation * DP` | Schedule and optimizer comparisons |
| Dense training compute | proportional to `parameters * tokens` | Budget and scaling estimates |
| Adam state | weights + gradients + two moments, precision dependent | Device and sharding memory |
| Pipeline bubble | roughly `(stages - 1) / microbatches` for simple schedule | Pipeline efficiency |
| Ring all-reduce traffic | `2 (n - 1) / n * tensor_bytes` per rank | Network estimate |

Always specify which bytes are sharded, replicated, offloaded, or temporarily gathered. Activation memory depends on sequence, microbatch, hidden width, saved intermediates, and checkpointing.

### Transformer inference

`KV_bytes = 2 * layers * batch * sequence * kv_heads * head_dim * bytes_per_element`

`decode_weight_lower_bound = weight_bytes / sustainable_bandwidth`

`roofline = min(peak_compute, bandwidth * FLOPs_per_byte)`

`attention_scores = O(sequence^2)` values if materialized`

FlashAttention preserves `O(sequence^2)` arithmetic for dense attention while reducing HBM materialization.

### Speculative decoding

`accept(x) = min(1, p(x) / q(x))`

`correction(x) proportional to max(0, p(x) - q(x))`

`Pr(accepted_prefix >= k) = product of conditional acceptance through k`

`E[accepted] = sum_(k=1..gamma) Pr(accepted_prefix >= k)`

`speedup = committed_target_equivalent_work / total_draft_plus_verify_time`

The final expression must include draft memory, target batch capacity, queueing, and scheduling when making a deployment decision.

### Distillation

`KL(p || q) = sum p log(p / q)`

`H(p, q) = H(p) + KL(p || q)`

`softmax_i(z, T) = exp(z_i / T) / sum_j exp(z_j / T)`

Classic logit distillation often uses `T^2 * KL(teacher_T || student_T)` plus a hard-label loss.

### Streaming

| Structure | Update / query | Memory | Contract |
| --- | --- | --- | --- |
| Min-heap top-k | `O(log k)` update | `O(k)` | Exact for insert-only global top-k |
| Monotonic deque | amortized `O(1)` | window size worst case | Exact sliding min/max |
| Reservoir | `O(1)` expected | `O(k)` | Uniform sample of unknown-length stream |
| Count-Min Sketch | `O(depth)` | `O(width * depth)` | One-sided frequency overestimate |
| HyperLogLog | `O(1)` | fixed registers | Probabilistic cardinality |
| Bloom filter | `O(hashes)` | bit array | False positives, no false negatives |

### Queueing and capacity

Little's Law under stable conditions: `L = lambda W`, where `L` is average work in system, `lambda` arrival rate, and `W` average time.

It does not predict tail latency or prove capacity. Use it to check dimensional consistency and connect concurrency to service time.

At high utilization with variable service time, queueing rises nonlinearly. Provision headroom based on burst and failure targets, not average utilization alone.

### Precision and quantization

For symmetric signed quantization with scale `s`:

`q = clip(round(x / s), q_min, q_max)`

`x_hat = s q`

Per-group metadata cost is `number_of_groups * scale_bytes` plus optional zero points. Effective model bytes include packing, alignment, scales, outliers, and any higher-precision fallback.

### Communication

Communication time is often modeled as:

`time = latency_term * messages + bytes / bandwidth`

Collective algorithms alter both. Overlap hides only communication that runs concurrently with independent useful compute; report exposed communication on the critical path.

:::callout pitfall|Every formula has a boundary
Write units. Name what is omitted. A lower bound is not a prediction, and a big-O expression is not a performance model.
:::

## CUDA Interview Checklist

### Before writing the kernel

1. State input and output shapes, dtype, layout, alignment, and allowed error.
2. Write the serial reference and complexity.
3. Identify parallel output units and reduction dimensions.
4. Estimate bytes and FLOPs.
5. Choose block and warp ownership.
6. Decide which values live in registers, shared memory, or HBM.
7. Plan edge handling and synchronization.

### Correctness checks

- Every global load and store is bounds-safe.
- Every block barrier is reached uniformly.
- Warp masks match active lanes.
- Shared-memory producers synchronize before consumers.
- Reduction identities initialize inactive lanes.
- Atomics use the correct type and scope.
- Random state is indexed without overlap.
- Numerical accumulation uses adequate precision.
- Empty, singleton, odd, and maximum shapes are tested.
- Results and gradients match a reference within declared tolerance.

### Performance checks

- Global accesses are coalesced or the scatter is unavoidable.
- Shared accesses avoid serious bank conflicts.
- Tiles provide reuse before eviction.
- Register count does not cause damaging spill.
- Shared memory does not eliminate needed residency.
- Launch count and CPU submission are measured.
- Vector width matches alignment and tail policy.
- Tensor-core instructions are actually selected.
- Fusion saves more traffic than it costs in occupancy or flexibility.
- Benchmark covers the production shape distribution.

### Matmul drill

Start with one output per thread. Add shared-memory tiling. Add register tiling. Change the inner loop to tensor-core fragments. Pipeline global-to-shared copies. Discuss split-K, persistent scheduling, epilogue fusion, and why a library may still win.

### Reduction drill

Reduce within a warp using shuffles. Store one value per warp. Synchronize. Let the first warp reduce partials. State which lane owns the output. Extend to max, argmax, Welford state, and multi-block reduction.

### Softmax drill

Find row max, sum shifted exponentials, normalize. Explain stable math, online combination, wide rows, recompute versus store, and fusion into attention.

### LayerNorm drill

Compute mean and variance or Welford state, normalize, apply scale and bias. Explain accumulation dtype, row width, vectorized IO, residual fusion, and backward reductions.

### Attention drill

Contrast naive score materialization with tiled online softmax. Derive rescaling. Cover causal masking, ragged lengths, GQA, backward recomputation, and tile selection.

### Common CUDA interview bugs

| Bug | Symptom | Repair |
| --- | --- | --- |
| Early return before barrier | Deadlock or partial output | Predicate loads and stores; keep barrier uniform |
| Missing edge predicate | Illegal access at nonmultiple shapes | Zero-fill input tile and guard output |
| Lane uses undefined shuffle value | Shape-dependent corruption | Correct active mask and identity values |
| Excess register tile | Spill and sudden slowdown | Inspect register count; retune tile |
| Column access in square shared tile | Bank-conflict serialization | Pad or remap layout |
| Timing without synchronization | Impossible speed | Use events or synchronized benchmark boundary |

:::callout insight|Narrate the resource ledger
As you write, say what each optimization buys and consumes: fewer HBM bytes, more reuse, more registers, more shared memory, fewer launches, or greater specialization.
:::

## Rapid-Fire Question Bank

### Training and data

#### Why AdamW rather than Adam plus L2?

Adam's coordinate-wise scaling also scales an L2 gradient, producing uneven effective shrinkage. AdamW applies weight decay separately from the adaptive gradient update.

#### Why warm up the learning rate?

Early moment estimates, activation scales, and representations are unstable. Warmup limits destructive updates until optimization statistics become informative.

#### What is gradient clipping protecting?

It bounds update-driving gradient norm under outliers or instability. It does not repair a systematically bad learning rate or numerical bug.

#### How do you compare recipes fairly?

Match data, tokenizer, token budget, optimizer-step count or explain the difference, evaluation, precision, and failure handling. Report quality versus tokens, wall-clock, and cost.

#### Exact versus near deduplication?

Exact hashes remove byte-identical content after normalization. Near deduplication clusters similar documents using shingles, MinHash, structural signatures, or semantic methods.

#### Why can a quality filter hurt?

It may remove rare domains, informal language, difficult examples, or diversity while improving average fluency. Train controlled pilots and inspect slice distribution.

#### How do you prevent synthetic collapse?

Use diverse seeds and generators, independent verification, novelty and duplication checks, human or authoritative data anchors, and evaluation outside the teacher distribution.

### Model and inference

#### Why GQA?

It reduces KV heads and cache bandwidth while retaining more K/V diversity than MQA, often improving the quality-throughput tradeoff during decode.

#### What does FlashAttention approximate?

Nothing in dense attention semantics. It changes the IO schedule using tiling and online normalization.

#### Why is decode sequential?

The next token distribution depends on the token just sampled. Parallelism comes from batch, model partition, or predicting and verifying multiple candidates.

#### What limits time to first token?

Queueing, prompt preprocessing, prefill compute and attention IO, model communication, and possibly prefix-cache miss or model cold start.

#### What limits inter-token latency?

Decode iteration cadence: scheduler wait, weight and KV traffic, collectives, launches, sampling, and streaming.

#### When does prefix caching fail?

Low prefix reuse, unsafe cross-tenant reuse, frequent model or template changes, large eviction pressure, or key mismatch from tokenization and position state.

#### Why can quantization improve capacity but not latency?

Unsupported or inefficient kernels, dequantization overhead, small workload, communication dominance, or a new compute bottleneck.

#### Why does speculation preserve quality?

Exact verification and residual correction reproduce the target distribution. Approximate variants need an explicit quality contract.

### Distributed systems

#### Data versus tensor parallelism?

Data parallelism replicates model compute and communicates gradients. Tensor parallelism partitions each layer and communicates activations or partial results inside the forward and backward critical path.

#### What does FSDP trade?

Lower replicated state for parameter gather/reduce-scatter traffic, prefetch complexity, and more sensitive execution order.

#### Why pipeline bubbles?

Stages wait during fill and drain or when work is imbalanced. More microbatches and better schedules reduce but do not erase dependency.

#### Why topology-aware groups?

Collective frequency and volume differ. Put the most latency- and bandwidth-sensitive communication on the fastest links.

#### What makes a retry unsafe?

Non-idempotent side effects or missing request identity. Streaming output, data consumption, and artifact publication need deduplication or transactional boundaries.

### Leadership

#### What is a Principal engineer's output?

Better technical decisions and execution across a scope larger than one person's implementation, expressed through architecture, mechanisms, talent, and aligned ownership.

#### How do you influence without authority?

Frame a shared outcome, include stakeholders early, provide evidence and adoption tooling, give partners ownership, and create mechanisms that reduce ongoing coordination cost.

#### When do you escalate?

When decision rights are blocked, risk exceeds the local mandate, a deadline requires choice, or incentives cannot be resolved at the working level. Escalate with options and evidence, not surprise.

#### How do you handle a wrong decision?

Contain harm, state new evidence, reopen the choice, preserve trust by owning the decision, and improve the mechanism that allowed the miss.

#### What is strategy?

A diagnosis of the controlling challenge plus a small set of coordinated choices that concentrate resources and create leverage toward a defined future state.

## Glossary and Decision Index

### Core terms

**Arithmetic intensity:** useful operations per byte moved across a specified memory boundary.

**Continuous batching:** scheduler policy that admits and removes sequences at iteration boundaries rather than waiting for an entire static batch.

**Control plane:** components that decide placement, version, rollout, policy, and capacity rather than executing individual model steps.

**Data lineage:** the versioned relationship from source through transformations to datasets, checkpoints, evaluations, and releases.

**Forward KL:** divergence weighted by the reference or teacher distribution, penalizing missing its supported outcomes.

**GQA:** grouped-query attention, where multiple query heads share a smaller number of key/value heads.

**KV cache:** stored attention keys and values for prior tokens, avoiding recomputation during autoregressive decode.

**Operational intensity:** a practical form of arithmetic intensity, sometimes including algorithm- and cache-specific byte assumptions.

**Paged attention:** attention over KV state managed in non-contiguous fixed-size blocks with logical-to-physical mapping.

**Prefill:** prompt processing phase that creates KV state and produces first-token logits.

**Reverse KL:** divergence weighted by the learned distribution, often favoring a supported mode when covering all modes is costly.

**Speculative decoding:** exact or controlled approximate generation using cheap proposals and expensive parallel verification.

**Tensor parallelism:** partitioning tensor operations across devices within layers.

**Time to first token:** latency from the declared request boundary until the first output token is available.

### Decision index

| If the symptom is... | First model | Likely chapters |
| --- | --- | --- |
| Slow first token | Queue plus prefill compute/IO | Prefill, scheduling, serving |
| Slow token cadence | Weight/KV bytes plus collectives | Decode, quantization, distributed inference |
| GPU OOM with free fragments | Logical versus physical KV allocation | Paged KV cache |
| High acceptance, no speedup | Joint draft/verify/scheduler cost | Speculative decoding |
| Training instability | Update, precision, data, synchronization | Optimization, recipe |
| Benchmark gain, product loss | Evaluation contract and workload shift | Measurement, serving |
| Low GEMM throughput | Tile, tensor-core, occupancy, shape | Tiled matrix multiplication |
| MoE slowdown | All-to-all and expert imbalance | Transformers, distributed systems |
| Platform not adopted | Migration cost and ownership | Strategy and leadership |
| Recurring disagreement | Goal, facts, risk, or incentives | Executive communication |

### Final principle

> Begin with the objective, quantify the bottleneck, choose the smallest coherent design, and close the loop with evidence.

That principle is equally useful for a KL objective, a CUDA kernel, a distributed scheduler, and a cross-org strategy. The scale changes. The discipline does not.

