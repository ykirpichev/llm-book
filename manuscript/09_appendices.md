# Part IX - Field Reference

The appendices are field references for design, implementation, debugging, and incident response. Each item is a retrieval cue; return to the chapter when assumptions or derivations are not obvious.

## End-to-End Learning Lab and Capstone

LEAD: Learning the field means connecting the boundaries. A tokenization choice changes targets; a target changes training; a checkpoint changes serving; a tool changes what counts as success. This lab makes one small lifecycle executable, then uses the book's running service to connect the larger pieces.

### Run a complete miniature model lifecycle

The module `examples/tiny_lm.py` trains a character bigram model with only Python's standard library. A bigram model predicts the next character from the current character alone. It is deliberately not a transformer and cannot learn general reasoning. Its small size makes every parameter, gradient, checkpoint field, and generation decision inspectable.

For vocabulary size V, parameters form a V × V table of logits. Row a defines the next-character distribution after character a. For target b, the gradient of cross-entropy with respect to a row logit is `p(j) - indicator(j=b)`. Accumulate this over observed transitions, divide by the total number of targets, and subtract a learning-rate-scaled gradient. No automatic differentiation or hidden optimizer is involved. Tests compare the analytic gradient with finite differences.

Example status: Runnable excerpt; execute from the repository root.

```python
from examples.tiny_lm import Bigram

training = ["ababa", "babab"]
model = Bigram(sorted(set("".join(training))))
initial_loss, _ = model.loss_and_gradient(training)
model.fit(training, steps=200, learning_rate=2.0)
final_loss, _ = model.loss_and_gradient(training)
assert final_loss < 0.01 < initial_loss

checkpoint = model.to_json()
restored = Bigram.from_json(checkpoint)
assert restored.generate("a", max_new_tokens=6) == "abababa"
```

Run `python -m examples.tiny_lm` to see the loss decrease and generated sequence. Training data defines the vocabulary; the checkpoint stores both vocabulary order and logits. The decoder uses greedy selection and a hard generation bound, with no learned end token. The examples never join separate documents to create an artificial cross-document training pair.

This experiment checks optimization and serialization behavior on a tiny fixture, not generalization. Evaluating more alternating characters repeats the same transition rule. A real validation split must challenge the intended capability without duplicating training examples, and a real tokenizer needs a declared unknown/byte-fallback policy. Here unknown characters raise an error so the limitation is visible.

### Connect the references in learning order

| Step | Do this in the repository | Explain before moving on |
| --- | --- | --- |
| Targets and state | Run sequence-model tests and the tiny LM | Token IDs, masked loss, gradient, checkpoint identity |
| Attention | Run the attention reference and partition tests | Stable softmax, masking, merge statistics, numerical tolerance |
| Adaptation and RL | Run post-training tests | DPO margin, advantage sign, clipping, group normalization, pass-at-k |
| Inference | Run inference-mechanism tests | Acceptance plus residual correction; quantization range and scale |
| Retrieval | Run the RAG fixture and ranking tests | Authorized current evidence versus mere semantic similarity |
| Agents | Run the bounded agent fixture | Capabilities, observations, unknown write outcome, budgets |

The CPU references teach and test semantics. They do not benchmark a model server, train a transformer, or validate CUDA programs. The hardware chapters explain how to move from semantic references to profiled implementations; their GPU acceptance checks remain work to perform on the actual target system.

When the capstone compares accelerator paths, use the baseline/candidate acceptance record linked from the repository's **Verified teaching examples** index. It keeps model identity, workload, quality, service tails, and cost together; unavailable target measurements stay explicitly unmeasured.

### Design the complete documentation service

Use the running 7B documentation assistant as a capstone. Start with the hypothetical workload and budgets in the RAG chapter, not a preferred framework. Deliver the following artifacts in order; each should be understandable by someone who has not watched the experiments.

1. **Contract and data manifest:** define supported tasks, document owners, permissions, revisions, language slices, freshness, quality, latency, and failure outcomes. Separate development and held-out evaluation before tuning.
2. **Unadapted baseline:** serve an existing compatible checkpoint with a bounded retrieval workflow. Record tokenizer/template, precision, context budget, and generation policy. Measure retrieval and generation separately.
3. **Adaptation decision:** use SFT/LoRA only for demonstrated behavior gaps; use retrieval for changing facts. Require an independent verifier and enough interaction data before adding RL. Compare against the unadapted baseline at the same workload boundary.
4. **Capacity ledger:** calculate weights, layer-specific state, workspaces, reserved memory, phase compute, and transfer budgets. A hybrid model requires a new state ledger; copying the dense transformer's KV formula is not a migration plan.
5. **Engine and performance experiment:** establish one compatible engine baseline, then compare a second using the controlled/deployment distinction in **Serving Engines and Cache Backends in Practice**. Identify the measured bottleneck before adding batching, quantization, speculation, remote cache, or phase sharding. Keep correctness, quality, and SLO gates fixed. If accelerator portability is required, repeat the full acceptance path on one alternative stack; translated code alone is not the result.
6. **Bounded agent extension:** permit only the tools the task requires. Define action identity, timeout/reconciliation, memory provenance, stopping conditions, and final-state verification. Compare completed authorized tasks per budget with the fixed workflow.
7. **Operational release:** bind model/data/index/harness versions, canary, inject failures, rehearse rollback, and assign ownership. Preserve both positive and negative results.

### What a good capstone answer contains

Keep the checkpoint/engine revisions, rejected alternatives, workload trace, resource ledger, quality uncertainty, and latency decomposition together. Distinguish calculations, CPU tests, reported results, and your own measurements. Attach cache ownership through lookup, reservation, transfer, cancellation, and eviction; for a port, include the model/precision/sharding combination and cold-start/recovery results.

If target hardware was unavailable, submit semantic tests and the proposed experiment with performance explicitly unmeasured. Peak specifications cannot fill those cells.

A strong answer explains the bottleneck, calculates a tradeoff, and supplies a rejection test. It need not use every technique: a small model and fixed workflow may solve the task without MoE, distributed RL, or multiple agents.

As a self-check, change one assumption at a time: double input length; revoke a document permission mid-request; replace attention layers with recurrent layers; lose the reply after a successful tool write; or make rollout production faster than the learner. Trace which state, budget, and acceptance rule changes. The relevant derivations and failure protocols are developed in the preceding parts; the reference sheets below help locate them.

## Formula and Capacity Sheet

### Training

| Quantity | Approximation | Use |
| --- | --- | --- |
| Global batch | `microbatch * accumulation * DP` | Schedule and optimizer comparisons |
| Dense training compute | proportional to `parameters * tokens` | Budget and scaling estimates |
| Adam state | weights + gradients + two moments, precision dependent | Device and sharding memory |
| Pipeline bubble fraction | `(stages - 1) / (microbatches + stages - 1)` for balanced forward-only stages | Fraction of schedule spent idle |
| Ring all-reduce traffic | `2 (n - 1) / n * tensor_bytes` sent per rank; the same volume received | Network estimate |

Always specify which bytes are sharded, replicated, offloaded, or temporarily gathered. Adam memory may also include higher-precision master weights. Activation memory depends on sequence, microbatch, hidden width, saved intermediates, and checkpointing. The pipeline expression assumes equal stage times and ignores communication; training schedules need their own backward and overlap accounting. `(stages - 1) / microbatches` is bubble overhead relative to useful work, not the idle fraction of total time.

### Transformer inference

`KV_bytes = 2 * layers * batch * sequence * kv_heads * head_dim * bytes_per_element`

`decode_weight_lower_bound = weight_bytes / sustainable_bandwidth`

`roofline = min(peak_compute, bandwidth * FLOPs_per_byte)`

`attention_scores = O(sequence^2)` values if materialized

FlashAttention preserves `O(sequence^2)` arithmetic for dense attention while reducing HBM materialization.

The KV expression assumes equal sequence lengths and head dimensions, with every listed layer storing full-history K and V. For ragged batches, replace `batch * sequence` by the sum of retained sequence lengths; then account for shared prefixes, block padding, replication, and sharding. The weight-streaming expression is a time bound for one pass over the listed weights, not a per-request latency prediction for a batch.

### Hybrid and recurrent state

For a hybrid, add the KV of attention layers, recurrent state for recurrent layers, local-convolution history, and allocator/workspace overhead. Do not multiply full-attention KV by the total layer count.

`matrix_state_bytes = recurrent_layers * batch * heads * value_dim * key_dim * state_bytes_per_element`

The expression assumes one matrix state per listed head and omits model-specific auxiliary state. MLA stores a compressed representation plus required positional state; sparse selection does not necessarily reduce stored history unless the architecture or cache policy also compresses/evicts it.

### Adaptation and reinforcement learning

`LoRA_parameters = rank * (input_width + output_width)` for one adapted matrix.

`DPO_loss = softplus(-beta * (policy_preference_logratio - reference_preference_logratio))`

`PPO_surrogate = min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)`

The PPO expression is maximized; a minimized policy loss uses its negative. DPO preference log-ratios compare chosen versus rejected response probabilities conditional on the same prompt.

`group_advantage = (reward - group_mean) / group_std` for the stated standardized GRPO variant; zero-variance groups require a defined policy.

`ESS = sum(weights)^2 / sum(weight^2)` for nonnegative weights with a positive total.

Reference-policy KL, behavior-policy ratios, and a learned value baseline have different roles. Clipping or low version lag does not establish on-policy equivalence.

### Speculative decoding

`accept(x) = min(1, p(x) / q(x))`

`correction(x) proportional to max(0, p(x) - q(x))`

`Pr(accepted_prefix >= k) = product of conditional acceptance through k`

`E[accepted] = sum_(k=1..gamma) Pr(accepted_prefix >= k)`

`speedup = baseline_time_for_same_committed_tokens / speculative_time`

Here `p` is the target distribution and `q` is the actual proposal distribution at the same history, after their respective sampling transformations. Acceptance is evaluated on a sampled proposal with `q(x) > 0`; the correction distribution is normalized only after rejection. Committed output also includes the correction token, or the extra target token when every proposal is accepted. Compare matched output work and include draft, verification, sampling, queueing, and scheduling in elapsed time; account for draft memory and lost target batch capacity separately.

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
| Count-Min Sketch | `O(depth)` | `O(width * depth)` | One-sided overestimate for nonnegative current frequencies under standard linear updates |
| HyperLogLog | `O(1)` | fixed registers | Probabilistic cardinality |
| Bloom filter | `O(hashes)` | bit array | False positives, no false negatives for insert-only membership |

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

## CUDA Engineering Checklist

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

### Common CUDA implementation bugs

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

## Diagnostic Question Bank

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

The algorithm preserves dense attention semantics through tiling and online normalization rather than dropping keys. Finite-precision arithmetic, low-bit operands, approximate exponentials, and implementation-specific numerical policies still introduce error. Exact attention semantics is not bitwise equality to every reference.

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

#### What is a senior technical leader's output?

Better technical decisions and execution across a scope larger than one person's implementation, expressed through architecture, mechanisms, talent, and aligned ownership.

#### How do you influence without authority?

Frame a shared outcome, include stakeholders early, provide evidence and adoption tooling, give partners ownership, and create mechanisms that reduce ongoing coordination cost.

#### When do you escalate?

When decision rights are blocked, risk exceeds the local mandate, a deadline requires choice, or incentives cannot be resolved at the working level. Escalate with options and evidence, not surprise.

#### How do you handle a wrong decision?

Contain harm, state new evidence, reopen the choice, preserve trust by owning the decision, and improve the mechanism that allowed the miss.

## Glossary and Decision Index

### Core terms

**Arithmetic intensity:** useful operations per byte moved across a specified memory boundary.

**Continuous batching:** scheduler policy that admits and removes sequences at iteration boundaries rather than waiting for an entire static batch.

**Control plane:** components that decide placement, version, rollout, policy, and capacity rather than executing individual model steps.

**Data lineage:** the versioned relationship from source through transformations to datasets, checkpoints, evaluations, and releases.

**Forward KL:** in this book's teacher/student convention, `KL(teacher || student)`, weighted by the teacher distribution and penalizing missing its supported outcomes.

**GQA:** grouped-query attention, where multiple query heads share a smaller number of key/value heads.

**KV cache:** stored attention keys and values for prior tokens, avoiding recomputation during autoregressive decode.

**Operational intensity:** a practical form of arithmetic intensity, sometimes including algorithm- and cache-specific byte assumptions.

**Paged attention:** attention over KV state managed in non-contiguous fixed-size blocks with logical-to-physical mapping.

**Strategy:** a diagnosis of the controlling challenge plus coordinated choices that concentrate resources toward a defined future state.

**Prefill:** prompt processing phase that creates KV state and produces first-token logits.

**Reverse KL:** in the same convention, `KL(student || teacher)`, weighted by the student distribution and often favoring a supported mode when covering all modes is costly. Always state the operands because naming conventions vary.

**Speculative decoding:** exact or controlled approximate generation using cheap proposals and expensive parallel verification.

**Tensor parallelism:** partitioning tensor operations across devices within layers.

**Time to first token:** latency from the declared request boundary until the first output token is available.

### Modern architecture and training terms

**BPE:** byte-pair encoding; a tokenizer family that repeatedly merges units under learned merge rules. Its vocabulary and normalization are part of checkpoint compatibility.

**SFT:** supervised fine-tuning on selected response targets. The loss mask and chat template define what the model is trained to predict.

**LoRA / QLoRA:** low-rank adaptation; QLoRA trains such adapters through a frozen quantized base. Reduced trainable parameters do not equal the same reduction in all training memory.

**DPO:** direct preference optimization using chosen/rejected responses and a reference policy, without an online rollout loop in the basic objective.

**GRPO:** group-relative policy optimization; response groups provide reward-relative advantage estimates instead of a learned value critic in the basic formulation.

**Behavior policy:** the distribution that actually sampled an action. It may differ from both the current learner and a fixed reference model.

**MLA:** Multi-head Latent Attention; compressed cached representations and projection algebra reduce attention-state storage under a specific architecture.

**Gated DeltaNet:** recurrent matrix memory that combines decay with an error-correcting write along the current key direction.

**Selective SSM:** a structured state-space model whose retention/write/read behavior depends on inputs; not an exact substitute for arbitrary full attention.

**TMEM / TMA:** tensor memory is an architecture-specific on-chip matrix storage resource; Tensor Memory Accelerator handles supported asynchronous transfers. Neither term means ordinary global-memory caching.

**Late interaction:** retrieval that precomputes document-token representations and combines them with query-token representations at query time.

**Denoising language model:** a model trained to reconstruct corrupted token sequences; its generation schedule may refine several positions at once.

**Idempotency key:** a stable operation identity allowing a service to reconcile repeated delivery of the same request without duplicating its effect, subject to that service's transaction guarantees.

**Accelerator ecosystems:** For Triton, ROCm/HIP, XLA/Pallas, Neuron/NKI, and SYCL definitions and hardware mappings, see **Accelerator Ecosystems Beyond CUDA and NVIDIA**.

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
| Accelerator port compiles but misses SLOs | Backend support, layouts, padding, collectives | Accelerator Ecosystems Beyond CUDA and NVIDIA |
| MoE slowdown | All-to-all and expert imbalance | Transformers, distributed systems |
| Platform not adopted | Migration cost and ownership | Strategy and leadership |
| Recurring disagreement | Goal, facts, risk, or incentives | Executive communication |
| Hybrid cache estimate is wrong | Layer-specific attention/recurrent state | Compressed, Sparse, and Recurrent Model State |
| More rollout GPUs hurt training | Queue growth, policy age, ratio variance | Distributed Reinforcement Learning and Policy Freshness |
| Fluent but unsupported answers | Eligible evidence, ranking, generation | RAG, Vector Search, and Evaluation Pipelines |
| Repeated or unauthorized actions | Capabilities, receipts, host-owned budgets | Building and Evaluating a Bounded Agent Loop |
| Fast text but slow speech | Encoder, alignment, codec, playback queues | Multimodal Representations: Images, Video, and Speech |
| Fewer generation calls but no gain | Positions per call, cache validity, quality | Diffusion and Block-Parallel Language Generation |

### Final principle

The documentation assistant owes its user an answer supported by current authorized evidence, delivered within its service target, with failures the team can detect and recover from. The model is essential, but the system around it determines whether that answer is useful.
