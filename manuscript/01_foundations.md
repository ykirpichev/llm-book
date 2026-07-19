# Part I - Foundations for ML Systems Decisions

The foundation is not a list of algorithms. It is a way to connect statistical objectives to the data, precision, memory, and execution constraints that determine whether a model can be trained and served.

## Optimization as a System

LEAD: Training is a feedback control problem implemented by a distributed numerical program. The optimizer, data order, precision policy, parallelism plan, and observability stack jointly define the trajectory.

### Start from the update, not the optimizer name

For parameters `theta`, stochastic optimization applies an update from a noisy gradient estimate. Plain SGD is:

`theta_(t+1) = theta_t - eta_t g_t`

The practical questions are hidden inside the symbols. What examples produced `g_t`? How much variance does the batch remove? In what precision was the reduction accumulated? Was the gradient clipped before or after data-parallel synchronization? Does `eta_t` refer to the nominal learning rate or the effective rate after adaptive normalization?

Momentum adds a state variable that filters short-term gradient noise:

`m_t = beta m_(t-1) + (1 - beta) g_t`

`theta_(t+1) = theta_t - eta_t m_t`

Adam tracks both first and second moments. AdamW decouples weight decay from the adaptive gradient step, avoiding the coordinate-dependent regularization produced by inserting L2 penalty directly into Adam's gradient.

:::callout insight|Say the invariant first
SGD chooses one global scale per step; Adam chooses a coordinate-wise scale from gradient history. AdamW then applies shrinkage independently of that adaptive scale. The right choice depends on optimization geometry, batch regime, memory budget, and generalization evidence.
:::

### Batch size is a systems knob

Larger batches improve hardware utilization and reduce gradient variance, but they also change the number of parameter updates per token. A comparison that holds epochs constant but not optimizer steps can attribute a training difference to the optimizer when the real cause is schedule mismatch.

Define global batch as:

`B_global = B_micro * accumulation_steps * data_parallel_replicas`

Increasing any factor may preserve the nominal batch while changing activation memory, communication cadence, or pipeline bubbles. Gradient accumulation reduces all-reduce frequency per token but lengthens the interval before parameters move. At very large batch, the noise scale can fall below the level that helped exploration or regularization.

#### Learning-rate scaling

Linear scaling is a useful initial hypothesis when the batch grows and the optimization regime is stable. Square-root scaling is more conservative under high gradient noise. Neither is a law. Use a short sweep around the predicted rate and compare loss versus **tokens and wall-clock**, not steps alone.

| Failure signal | Likely mechanism | First check |
| --- | --- | --- |
| Loss spikes early | Update norm too large or precision overflow | Grad norm, loss scale, warmup |
| Stable loss, weak validation | Overfitting or data mismatch | Split integrity, duplication, regularization |
| Throughput high, convergence slow | Batch too large or stale updates | Tokens per update, optimizer steps |
| Replica losses diverge | Synchronization or nondeterministic data | Collective health, seed and sampler state |
| Late NaNs | Accumulated optimizer-state corruption | Moment statistics, checkpoint bisection |

### Precision policy is part of the algorithm

Mixed precision is not "turning on FP16." A production policy specifies:

- storage dtype for parameters and optimizer states;
- compute dtype for matrix multiply and reductions;
- accumulation dtype for sensitive sums;
- loss scaling behavior;
- master-weight policy;
- overflow detection and recovery;
- which operations are forced into higher precision.

BF16 preserves the exponent range of FP32 with fewer mantissa bits, which usually makes it easier to train than FP16 without dynamic loss scaling. FP16 provides more mantissa precision at a narrower range. FP8 increases throughput and reduces bandwidth but requires scale selection, granularity, delayed or current statistics, and careful protection of outliers.

:::callout pitfall|Precision is not only a speed setting
If a candidate says "use lower precision" without naming accumulation, scaling, outliers, and validation criteria, the answer is incomplete. Numerical policy can change convergence and therefore total training cost.
:::

### Initialization, normalization, and residual scale

Deep residual networks work because signal and gradient magnitudes remain controlled across many layers. Initialization sets the first scale; normalization and residual parameterization determine how it evolves.

Pre-norm transformers place normalization before the sublayer, improving gradient flow for deep networks. Post-norm can provide different representation dynamics but is harder to stabilize at scale. RMSNorm removes mean subtraction and reduces work; LayerNorm controls both mean and variance. The performance difference is usually small relative to matmul, but the training dynamics and fusion opportunities matter.

For a residual update `x_(l+1) = x_l + f_l(x_l)`, depth-aware scaling prevents the sum of many residual branches from growing without bound. Techniques differ in where they introduce factors, but the design question is the same: how should the variance of the residual stream behave as depth increases?

### A reproducible training step

```python
def train_step(model, batch, optimizer, scaler, clip_norm):
    optimizer.zero_grad(set_to_none=True)
    with autocast(dtype="bfloat16"):
        logits = model(batch.tokens)
        loss = cross_entropy(logits[:, :-1], batch.tokens[:, 1:])

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = clip_grad_norm_(model.parameters(), clip_norm)
    scaler.step(optimizer)
    scaler.update()
    return {"loss": loss.detach(), "grad_norm": grad_norm}
```

The interview discussion begins after the code: Where is accumulation? Which values are synchronized? Is clipping global across shards? How are token masks normalized? What happens when one worker overflows? Does a skipped step advance the data sampler or learning-rate schedule?

### Scaling laws as resource allocation

Scaling laws relate reducible loss to model size, data, and compute over a particular regime. Their value is not the exact exponent; it is the discipline of treating architecture and data allocation as an optimization problem under a compute budget.

A compute-optimal recipe balances marginal return from more parameters against marginal return from more training tokens. Real products add constraints absent from the ideal curve: inference cost, memory capacity, data rights, latency, fine-tuning behavior, and the value of rapid iteration.

:::callout decision|When the scaling-law optimum is not the product optimum
A smaller, better-trained model may win when serving dominates lifetime cost. A larger model trained on fewer tokens may win when capability at the frontier matters more than utilization. State which lifetime objective you are optimizing.
:::

### Principal Interview Review

1. Explain why doubling global batch without changing the schedule is not a controlled optimizer comparison.
2. Design a numerical policy for FP8 training. Name the values you would keep in BF16 or FP32.
3. A run is 18 percent faster per step but reaches the target loss 25 percent later. What should the team report?
4. How would you localize a NaN that appears after 30,000 steps on one of 256 workers?
5. When would you prefer SGD with momentum to AdamW for a large model?

> A Principal answer connects the loss curve to the execution trace. It does not stop at optimizer folklore.

## Transformers as an Architecture of Tradeoffs

LEAD: A transformer is a collection of decisions about communication, state, and inductive bias. Attention is important, but the systems interview is about why each variant moves cost or quality.

### The block and its cost centers

A decoder block alternates token mixing and channel mixing. Self-attention allows each token to read other positions; the feed-forward network transforms each position independently. Residual connections preserve an information highway; normalization stabilizes scale.

:::diagram attention|Attention is a sequence of projections, a score computation, a normalization, and a weighted reduction. Efficient kernels avoid materializing the large score matrix in HBM.

For sequence length `S`, hidden width `D`, heads `H`, and head dimension `d = D/H`, projection FLOPs scale roughly with `S D^2`, while naive attention score and value products scale with `S^2 D`. At short context, projections and MLP dominate. At long context, quadratic attention becomes the controlling term unless sparsity, locality, or recurrence changes the dependency graph.

The feed-forward expansion is often four times the hidden width in a dense baseline, although gated variants change constants. Because those layers contain much of the parameter count and FLOPs, a discussion focused only on attention misses the dominant compute for many workloads.

### Multi-head, multi-query, and grouped-query attention

Standard multi-head attention stores separate K and V projections for every query head. Multi-query attention shares one K/V head; grouped-query attention shares K/V within groups.

During training, reducing K/V heads modestly changes projection parameters. During autoregressive serving, it can dramatically reduce KV-cache bytes and memory traffic. The quality cost depends on model size, data, and head grouping.

| Variant | Query heads | K/V heads | KV-cache cost | Typical use |
| --- | --- | --- | --- | --- |
| MHA | H | H | Highest | Maximum flexibility |
| GQA | H | G, where 1 < G < H | Medium | Quality-throughput balance |
| MQA | H | 1 | Lowest | Memory-constrained decoding |

:::callout insight|Why GQA exists
GQA is a serving-aware architecture decision. It preserves multiple K/V subspaces while reducing the state that must be read for every generated token.
:::

### Position: RoPE, ALiBi, and context extension

Transformers require a notion of order because attention itself is permutation equivariant. RoPE rotates query and key components by position-dependent angles, making the dot product depend on relative offset. ALiBi adds head-specific linear distance biases to attention scores.

Context extension is not achieved by changing one configuration value. A model trained mostly on short sequences may not learn behavior at long offsets. RoPE scaling changes the mapping between position and phase; interpolation or frequency-aware schemes trade local resolution against extrapolation. Long-context training also changes memory, parallelism, data construction, and evaluation.

#### Long-context evaluation

Needle-in-a-haystack retrieval is a diagnostic, not a complete evaluation. Include:

- retrieval across distance and distractor density;
- multi-hop composition across separated evidence;
- instruction retention over long interaction histories;
- long-form generation coherence;
- latency, prefill memory, and KV-cache pressure;
- degradation on ordinary short-context tasks.

### Mixture of Experts

MoE increases parameter count without activating all parameters per token. A router selects a small number of experts. The promise is higher capacity per unit of compute; the cost is routing instability, load imbalance, all-to-all communication, and operational complexity.

Let `E` be experts and `k` active experts per token. Compute follows `k`, but storage follows `E`. Communication follows token movement across the expert-parallel group. If one expert receives too many tokens, capacity limits cause drops or padding and lower utilization.

Typical objectives combine task loss with an auxiliary load-balancing term. The auxiliary term should be monitored as a means, not a product metric. Perfectly uniform routing can be incompatible with specialization.

:::callout pitfall|Sparse does not mean cheap
An MoE layer may have low arithmetic cost but high network cost. The design is attractive only when expert placement, token dispatch, batching, and capacity produce high accelerator utilization.
:::

### Attention kernel choices

Naive attention materializes `S x S` scores in HBM. FlashAttention tiles Q, K, and V into on-chip memory, computes softmax statistics online, and writes only the final output plus small normalization state. It is exact attention with a different IO schedule, not an approximation.

For backward, the kernel recomputes selected intermediates because recomputation FLOPs are cheaper than reading a quadratic matrix from HBM. This is a recurring accelerator principle: spend abundant arithmetic to avoid scarce data movement.

### Architecture review template

When an interviewer asks you to choose a transformer variant, organize the answer by lifecycle:

1. **Training:** FLOPs, activation memory, stability, parallelism, and data.
2. **Serving prefill:** long-sequence compute, batching, attention kernel, and time to first token.
3. **Serving decode:** weight and KV bytes, batch shape, scheduling, and time per output token.
4. **Quality:** capacity, context behavior, specialization, and fine-tuning response.
5. **Operations:** kernel availability, quantization, observability, fallback, and model portability.

### Principal Interview Review

1. Why can a change from MHA to GQA improve decode throughput more than prefill throughput?
2. Explain FlashAttention without saying "it is faster attention."
3. Design an evaluation that detects long-context gains without hiding short-context regression.
4. When can MoE be slower than a dense model with similar active FLOPs?
5. Why is an architecture decision also a serving decision?

## Measurement, Evaluation, and Experimental Judgment

LEAD: Evaluation is the contract between an optimization and the product. If the contract is weak, a locally correct system can still fail.

### Separate four kinds of metrics

**Capability metrics** measure whether the model can perform tasks: exact match, pass rate, calibrated score, factuality, or human preference. **System metrics** measure latency, throughput, utilization, queueing, and cost. **Guardrail metrics** capture safety, privacy, fairness, regressions, and abuse. **Diagnostic metrics** reveal mechanisms: acceptance rate, router balance, KV hit rate, gradient norm, or cache fragmentation.

No single metric belongs to every layer. Acceptance rate is an excellent diagnostic for speculative decoding, but speedup and quality are the product outcome. Perplexity is useful for broad training progress, but it may fail to predict instruction following or code correctness.

### Build the evaluation matrix

Cross user segments with task categories, difficulty, language, context length, and safety risk. Attach confidence intervals and minimum sample sizes. Preserve immutable benchmark versions and provenance. Track contamination risk explicitly.

| Dimension | Example slices | Why it matters |
| --- | --- | --- |
| Workload | chat, code, reasoning, summarization | Optimizations move tasks differently |
| Context | short, medium, long, multi-turn | Memory and attention behavior shift |
| Traffic | interactive, batch, burst | Queueing changes the system optimum |
| User | locale, domain, experience | Aggregate scores hide regressions |
| Risk | ordinary, sensitive, adversarial | Guardrails need targeted power |

### Offline does not predict online by default

Offline evaluation offers control and reproducibility. Online experiments include user adaptation, traffic mix, cache warmth, queueing, and feedback effects. A robust launch chain is:

1. unit and numerical checks;
2. offline capability and safety suite;
3. replay with production-like shapes;
4. shadow traffic without user exposure;
5. small canary with automatic rollback;
6. controlled experiment with segment analysis;
7. broad rollout with durable monitors.

:::callout decision|Choose the smallest honest experiment
Do not demand an online A/B test for a broken kernel, and do not accept a microbenchmark for a product claim. Match the evidence surface to the decision surface.
:::

### Tail latency and coordinated omission

A latency benchmark that generates the next request only after the previous request completes reduces offered load when the system slows. It can hide queueing collapse. Use an arrival process independent of service completion, record admission rejections, and report the full latency distribution by request shape.

Always define the boundary: client-to-client, gateway-to-gateway, engine-only, or kernel-only. Time to first token, inter-token latency, and end-to-end completion latency answer different user questions.

### Statistical power and practical significance

Confidence intervals quantify sampling uncertainty, not benchmark validity. A highly significant 0.1 percent lift may be irrelevant; a noisy 3 percent loss in a critical safety slice may block launch. Predefine the minimum effect worth acting on and the guardrail boundary that cannot be crossed.

Multiple comparisons inflate false positives when teams inspect many slices. Use a hierarchy: primary metrics, declared guardrails, and exploratory diagnostics. Treat exploratory findings as hypotheses for the next experiment.

### A compact experiment review

```text
Decision:    What choice will this experiment change?
Hypothesis:  What causal mechanism predicts the result?
Population:  Which workload and users are represented?
Metrics:     Primary outcome, guardrails, diagnostics.
Power:       What effect size can we reliably detect?
Execution:   Randomization, duration, ramp, rollback.
Analysis:    Segments, uncertainty, missing data.
Follow-up:   Ship, iterate, or retire - with owner and date.
```

### Principal Interview Review

1. A model improves benchmark accuracy but increases p99 latency. How do you decide?
2. Explain why a closed-loop load generator can lie about overload behavior.
3. Design evaluation for a new data-quality filter.
4. Which metrics would you use for an optimization that changes only KV-cache layout?
5. How do you respond when the aggregate metric is flat but an important segment regresses?

> Measurement is not the final reporting step. It is part of architecture because it determines which failures the organization can see.

