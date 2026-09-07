# Part I - Foundations for ML Systems Decisions

Large language models are statistical programs whose behavior emerges from an unusually tight coupling of data, optimization, architecture, numerical representation, and hardware. None of those layers can be understood safely in isolation. A mathematically valid objective can fail because its gradients overflow. A faster kernel can make a service slower by changing batch formation. A larger model can lower validation loss while making the product economically impossible to operate.

This part develops the reasoning tools used throughout the book. The goal is not to collect definitions. It is to establish a disciplined way to move between four views of the same system:

- the **statistical view**, which asks what distribution and objective define success;
- the **computational view**, which asks what operations and state realize that objective;
- the **systems view**, which asks where bytes, work, synchronization, and failures occur;
- the **product view**, which asks whether the resulting capability is useful, reliable, and affordable.

The central habit is to keep these views connected. Every important design choice should be expressible as a change to the objective, the workload, the resource model, or the evidence required to proceed.

## The Model, Workload, and System Contract

LEAD: An ML system begins with a contract among behavior, workload, resources, and operations. Architecture selection comes later. Without this contract, teams optimize a model that has not been defined as a product.

### Begin with behavior, not model size

“Train a better model” is not an engineering objective. Better for whom, on which tasks, under what constraints, and compared with what baseline? A useful objective identifies the behavior to improve and the failures that may not worsen.

Suppose a code assistant must improve repository-level debugging. The behavior contract might require that the model locate a defect, propose a patch that passes tests, explain the relevant invariant, and abstain when repository context is insufficient. The system contract adds a 4-second time-to-first-token target, a 32,000-token working context, regional data-handling restrictions, and a cost ceiling per completed task.

Those requirements immediately shape the technical design. Repository context creates long prefills and large KV state. Executable verification affects training-data construction and evaluation. The latency target constrains model size and parallelism. The abstention requirement needs calibrated evaluation rather than pass rate alone. The data restriction affects retrieval, logging, and where inference may run.

A compact contract includes:

| Surface | Questions that make it concrete |
| --- | --- |
| Behavior | What should the model produce, refuse, preserve, or explain? |
| Population | Which users, languages, domains, and risk classes matter? |
| Workload | Prompt length, output length, concurrency, arrival process, and tool use? |
| Quality | Primary outcomes, critical slices, guardrails, and minimum useful effect? |
| Service | Latency, availability, durability, freshness, and overload semantics? |
| Economics | Training budget, serving cost, engineering effort, and hardware supply? |
| Governance | Rights, privacy, security, audit, retention, and deletion obligations? |
| Evolution | How will the model, engine, data, and evaluation change independently? |

This contract is not a requirements form completed once. It is the shared model against which experiments and architecture choices are interpreted. When the workload changes, the correct model or serving plan may change with it.

### Define the unit of success

Metrics become misleading when their denominator is vague. Training loss is measured per predicted token, but the product may care about successful tasks, accepted code changes, resolved support cases, or useful sessions. A cheaper token is not necessarily a cheaper outcome if quality declines and users require more attempts.

Let `C_run` be the cost of a set of requests, `N_task` the number of attempted tasks, and `p_success` the probability that a task meets the declared quality bar. The expected cost per successful task is:

:::equation C_{success} = C_{run} / (N_{task} p_{success})|Cost must be normalized by a useful outcome, not only by generated tokens.

This simple normalization changes many decisions. A model that costs 20 percent more per token but raises task success from 40 to 55 percent can be cheaper per successful outcome. Conversely, an aggressive quantization that improves raw throughput may lose economically if retries, escalations, or human review increase.

The same discipline applies to system throughput. Report **goodput**: work completed while satisfying quality and service constraints. Tokens produced after an SLO deadline, requests later discarded, and generations that fail validation consume capacity but do not count as useful output.

### Treat workload as a joint distribution

Averages erase the correlations that determine system behavior. Prompt length, output length, tenant, model variant, adapter, tool calls, latency priority, and arrival time are not independent. Enterprise users may send longer prompts and demand stricter isolation. Code requests may generate longer outputs and have higher speculative-decoding acceptance. Traffic bursts may coincide with colder caches.

Represent the workload as a joint distribution over request attributes:

:::equation W = P(S_{in}, S_{out}, A, Q, T, R)|A workload model preserves correlations among input length, output length, arrival process, quality class, tenant, and request type.

The model need not be analytically elegant. A versioned trace with privacy-safe fields is often more useful than independent parametric distributions. What matters is that benchmarks reproduce the shapes and correlations that influence queueing, memory, kernel efficiency, and quality.

Use percentiles and conditional distributions rather than a single representative request. A p95 prompt combined with a p95 output is not necessarily a real p95 request; it may describe a combination that almost never occurs. Capacity planning should replay or sample from joint observations, then stress explicit adversarial cases separately.

### Establish invariants and failure semantics

Requirements describe desired outcomes. Invariants describe what must remain true while the system changes or fails. Examples include:

- a token may be streamed only from the model and policy versions recorded for the request;
- two tenants may never share a prefix-cache entry without an authorized identity boundary;
- a checkpoint is visible only after every required shard and manifest is durable;
- an optimizer step either advances all participating ranks or none of them;
- a cancelled request eventually releases its KV blocks and scheduler state;
- an evaluation result names immutable model, data, engine, prompt, and metric versions.

Failure semantics say what users observe when an invariant cannot be maintained. A service might reject before admission, terminate a stream with an explicit error, retry only before the first visible token, or fall back to a smaller model. “Highly available” is incomplete without these state transitions.

The most expensive reliability mistakes are often semantic rather than mechanical: duplicating a side effect during retry, mixing shards from two checkpoints, silently changing sampling behavior after failover, or reporting a successful response after a tool action failed. Designing the state machine early is cheaper than inferring it during an incident.

### Make decisions with explicit models

An engineering model is a deliberately simplified relationship between a decision and an outcome. Useful models include:

- a FLOP estimate that identifies the dominant operation;
- a byte model that predicts whether a phase is bandwidth-bound;
- a queueing model that exposes the effect of utilization on latency;
- a statistical model that bounds evaluation uncertainty;
- a cost model that converts fleet resources into product economics.

The purpose is not perfect prediction. It is to identify controlling variables, expose assumptions, and decide what must be measured. A model should be simple enough to challenge and detailed enough to change a choice.

:::callout insight|A model is valuable when it can be wrong usefully
Write down the prediction before measuring. When observation disagrees, the residual points toward a missing effect: launch overhead, communication, cache behavior, skew, numerical instability, or an invalid workload assumption.
:::

### Design Exercises

1. Write a system contract for a long-context research assistant. Include a quality unit, workload distribution, failure semantics, and cost unit.
2. A smaller model produces tokens at half the cost but completes 25 percent fewer tasks. Define the information required to compare cost per successful outcome.
3. Identify three invariants for a service that streams model output while executing tools.
4. Explain why independent prompt-length and output-length histograms can produce a misleading capacity forecast.

## Optimization as a Coupled Dynamical System

LEAD: Training is not an optimizer acting on a fixed objective. It is a coupled dynamical system whose trajectory depends on data order, batch construction, numerical representation, distributed execution, and the rules used to recover from failure.

### From population risk to a training step

Let `x` denote an example drawn from an unknown population distribution `P`, and let `loss(theta; x)` be the loss of parameters `theta` on that example. The population objective is:

:::equation L(θ) = E_{x ~ P}[ℓ(θ; x)]|Population risk is an expectation over the deployment-relevant distribution.

Training does not have direct access to `P`. It operates on a finite, curated dataset and a sampling policy. At step `t`, a batch `B_t` produces a stochastic gradient estimate:

:::equation g_{t} = (1 / size(B(t))) Σ_{x ∈ B(t)} ∇_{θ} ℓ(θ_{t}; x)|The batch gradient depends on both the examples selected and their effective weights.

Plain stochastic gradient descent applies:

:::equation θ_{t+1} = θ_{t} - η_{t} g_{t}|The learning rate converts a gradient estimate into parameter motion.

The compact notation hides nearly every systems decision. Which tokenizer and loss mask defined the tokens? Were examples sampled uniformly or reweighted? Was the loss normalized by sequences, non-padding tokens, or tasks? In what precision were local gradients computed and global gradients reduced? What happened when one worker overflowed? Did a recovered run reproduce the same data position and random streams?

These are not implementation details outside the algorithm. They determine the realized update and therefore the learned model.

### Momentum and adaptive preconditioning

The update rules below follow [Adam](https://arxiv.org/abs/1412.6980) and [decoupled weight decay](https://arxiv.org/abs/1711.05101). The systems tradeoffs that follow are engineering consequences, not a claim that either optimizer dominates every training regime.

Momentum forms an exponentially weighted estimate of recent gradient direction:

:::equation m_{t} = β_{1} m_{t-1} + (1 - β_{1}) g_{t}|The first moment suppresses high-frequency gradient noise and preserves directional persistence.

Adam also tracks an elementwise second moment:

:::equation v_{t} = β_{2} v_{t-1} + (1 - β_{2}) g_{t}^{2}|The second moment estimates a coordinate-wise scale for recent gradients.

After bias correction, the adaptive step is approximately:

:::equation θ_{t+1} = θ_{t} - η_{t} m_{t}^{c} / (√v_{t}^{c} + ε)|The superscript c denotes bias-corrected first and second moments.

AdamW applies weight decay as a separate shrinkage term rather than inserting an L2 penalty into the adaptively scaled gradient:

:::equation θ_{t+1} = (1 - η_{t} λ) θ_{t} - η_{t} m_{t}^{c} / (√v_{t}^{c} + ε)|Decoupled decay preserves a clearer distinction between optimization and parameter shrinkage.

That distinction matters because adaptive scaling would otherwise make the effective regularization coordinate-dependent. It does not imply that AdamW is universally superior. The optimizer trades memory, communication, convergence behavior, and robustness. Optimizer states may consume more memory than the parameters themselves; sharding or quantizing those states changes the execution plan.

### Batch size changes statistics and execution

The global batch combines microbatching, gradient accumulation, and data-parallel replication:

:::equation B_{global} = B_{micro} × N_{accum} × N_{data}|Equal global batch does not imply equal execution behavior.

Increasing `B_micro` usually raises activation memory and can improve local matrix shapes. Increasing `N_accum` delays synchronization and parameter updates while preserving more local work between collectives. Increasing `N_data` expands communication and changes how examples are partitioned. The same global batch can therefore have different throughput, numerical behavior, and failure exposure.

Batch size also changes the number of updates performed for a fixed token budget. If total training tokens are `T` and the global batch contains `B_tokens` tokens, then the number of optimizer steps is approximately:

:::equation N_{steps} = T / B_{tokens}|Schedule comparisons must account for tokens, steps, and wall-clock time.

Doubling batch without revisiting warmup, decay, and total steps is not a controlled optimizer comparison. Linear learning-rate scaling is a useful local hypothesis when gradient statistics and architecture remain similar; square-root scaling is more conservative when noise remains important. Neither is a law. The correct scale is an empirical property of the regime.

Compare experiments along at least three axes:

- loss or capability versus tokens, which measures sample efficiency;
- loss or capability versus wall-clock time, which measures training efficiency;
- final quality versus total cost, which measures economic efficiency.

A configuration can win one axis and lose another. Reporting only step time rewards large batches even when they require many more tokens to reach the target.

### Gradient noise and clipping

The stochastic gradient can be decomposed into a population component and sampling noise. Increasing batch tends to reduce the variance of the mean estimate, but duplicated or strongly correlated examples reduce the effective batch size. Mixture weights and curriculum changes can shift both the mean and variance.

Global norm clipping applies a scale factor when the gradient norm exceeds threshold `c`:

:::equation g' = g × min(1, c / norm_{2}(g))|Clipping bounds update magnitude while preserving direction below the threshold.

In a sharded run, `norm_2(g)` is a global quantity. Clipping each shard independently implements a different rule. The reduction order and accumulation precision also affect the norm. Log the unclipped norm, clipped fraction, update norm, and loss-scale state so that clipping does not silently hide instability.

Clipping is a safety mechanism, not a substitute for diagnosing repeated spikes. Persistent clipping can indicate a learning-rate problem, corrupted data, an unstable loss term, or a precision failure.

### Precision policy is part of the training algorithm

A dtype name does not define a numerical policy. A complete policy specifies:

- storage formats for parameters, gradients, optimizer states, and activations;
- compute formats for matrix multiplication and elementwise operations;
- accumulation formats for reductions and sensitive statistics;
- scaling granularity and how scales are estimated;
- overflow, underflow, and non-finite detection;
- master-weight and checkpoint representation;
- the response to a failed step across every rank.

BF16 preserves FP32's exponent range while reducing mantissa precision, making it relatively robust for activations and matrix multiplies. FP16 has a narrower exponent range and often requires dynamic loss scaling. FP8 can reduce bandwidth and increase tensor-core throughput, but its smaller range makes scale selection, outlier handling, and accumulation policy central to correctness.

Consider loss scaling. Multiplying the loss by scale `s` multiplies gradients by `s` before backward computation; gradients are divided by `s` before the optimizer update. This moves small values into a representable range without changing the ideal real-valued update. If one rank detects an overflow, all ranks must agree to skip the update. Otherwise parameters, optimizer moments, data position, and schedule state diverge.

Sensitive reductions often remain in FP32 even when operands are lower precision. Softmax normalization, variance estimates, gradient norms, and optimizer updates have error structures that differ from matrix multiplication. Precision should be assigned by numerical role, not by a global toggle.

:::callout pitfall|Fast steps can create a slower training run
A lower-precision configuration is valuable only if it preserves the convergence trajectory. A run that is 20 percent faster per step but requires 30 percent more steps to recover quality is not an optimization.
:::

### Residual scale, normalization, and depth

Transformer training depends on maintaining useful signal and gradient scales through many residual blocks. A generic residual update is:

:::equation x_{l+1} = x_{l} + α_{l} f_{l}(x_{l})|Residual parameterization controls how new transformations accumulate with depth.

If residual contributions behave like independent random variables with similar variance, an unscaled sum can grow with depth. Real networks violate the independence assumption, but the model explains why initialization and depth-aware scaling matter. Choices such as pre-normalization, post-normalization, residual scaling, and parameter initialization jointly determine the early training regime.

Pre-normalized blocks place normalization before attention or the feed-forward sublayer. This creates a relatively direct residual path for gradients and is commonly stable in deep models. Post-normalized blocks normalize after the residual addition and can exhibit different representation behavior but often require more careful initialization or scaling.

LayerNorm controls mean and variance; RMSNorm controls root-mean-square magnitude without subtracting the mean. The FLOPs saved by RMSNorm are rarely the primary architectural reason, although simpler normalization can enable efficient fusion. The choice should be evaluated through training stability, quality, precision sensitivity, and kernel support.

For the normalization mechanism and its evaluated benefits, see [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467); do not transfer its reported timings to another model without measurement.

### A training step is a distributed transaction

A production step contains more state than parameters and gradients. It advances optimizer moments, learning-rate schedule, random streams, data-sampler position, loss scale, gradient accumulation counters, and monitoring windows. Checkpoint recovery must restore these surfaces consistently.

Example status: Illustrative Python excerpt; not standalone.

```python
def train_step(state, batch):
    state.optimizer.zero_grad(set_to_none=True)

    with autocast(dtype=state.compute_dtype):
        logits = state.model(batch.tokens)
        token_loss = cross_entropy(
            logits[:, :-1], batch.tokens[:, 1:], reduction="none"
        )
        loss = masked_mean(token_loss, batch.loss_mask[:, 1:])

    state.scaler.scale(loss).backward()
    state.scaler.unscale_(state.optimizer)

    grad_norm = distributed_global_norm(state.model.parameters())
    clip_by_global_norm_(state.model.parameters(), state.clip_threshold)
    all_ranks_finite = distributed_finite_check(state.model.parameters())

    if all_ranks_finite:
        state.scaler.step(state.optimizer)
        state.scheduler.step()
        state.data_cursor.commit(batch.cursor)

    state.scaler.update(all_ranks_finite)
    return {"loss": loss.detach(), "grad_norm": grad_norm}
```

The exact framework API will differ, but the invariants should be visible: loss normalization is explicit, gradient norm is global, every rank agrees on finiteness, and schedule plus data position advance only with a committed update.

### Diagnose the trajectory, not one metric

Training observability should connect statistical and system state. At minimum, record:

- loss and component losses by data slice;
- token counts, effective weights, and mixture proportions;
- gradient norm, update norm, clipping frequency, and loss scale;
- parameter and optimizer-state statistics;
- step time decomposed into input, forward, backward, communication, and optimizer work;
- hardware errors, stragglers, skipped steps, and recovery events;
- validation metrics at immutable checkpoints.

When loss diverges, locate the first inconsistent surface. Did the input distribution change? Did one rank observe a non-finite value? Did a kernel or compiler version change? Did optimizer state fail to restore? Checkpoint bisection combined with replayable batches is far more effective than inspecting the final NaN.

### Design Exercises

1. Explain why two runs with the same global batch can have different numerical and performance behavior.
2. Design a precision policy for FP8 training, naming the values retained in BF16 or FP32 and the evidence required for rollout.
3. A configuration is 18 percent faster per step but needs 25 percent more tokens to reach target loss. Construct an honest comparison.
4. Design a recovery invariant for skipped steps in a 256-worker job.
5. A run clips 40 percent of steps while validation improves. What evidence distinguishes a useful safety bound from concealed instability?

## Transformer Architecture as Resource Allocation

The core attention construction comes from [Attention Is All You Need](https://arxiv.org/abs/1706.03762). This chapter adapts that mechanism to a decoder-serving resource model; its byte and FLOP estimates are derivations under the stated assumptions, not measurements from that paper.

LEAD: A transformer is a schedule for moving information among tokens and channels. Each architectural choice reallocates parameters, arithmetic, memory traffic, communication, and persistent state across training and inference.

### Follow one token through a decoder block

Let the input to layer `l` be a matrix `X_l` with sequence length `S` and hidden width `D`. A pre-normalized decoder block can be written schematically as:

:::equation U_{l} = X_{l} + Attention(Norm(X_{l}))|Attention mixes information across token positions.

:::equation X_{l+1} = U_{l} + MLP(Norm(U_{l}))|The feed-forward network mixes information across channels independently at each position.

This representation highlights two kinds of mixing. Attention creates dependencies among positions; the MLP transforms each position through a larger intermediate space. Residual paths carry earlier representations forward and provide a route for gradients.

For `H` query heads and head dimension `d = D / H`, the projections form query, key, and value tensors. Scaled dot-product attention is:

:::equation A = softmax(Q K^{T} / √d + M)|The mask M encodes causal or structural constraints.

:::equation O = A V|Each output position is a weighted reduction over value vectors.

The equations specify semantics, not an efficient execution plan. Materializing the full `S × S` score and probability matrices creates quadratic memory traffic. IO-aware kernels tile the computation, retain partial softmax statistics on chip, and avoid writing those intermediates to high-bandwidth memory.

### Account for parameters, FLOPs, and activation state

In a dense decoder block, attention projections contribute on the order of `4D²` parameters when query, key, value, and output widths all equal `D`. A gated MLP with intermediate width `F` contributes roughly `3DF` parameters because it commonly uses two input projections and one output projection.

The exact constants matter for capacity planning, but the scaling terms provide the first decision model:

:::equation P_{block} ≈ 4D^{2} + 3DF|Approximate parameters in attention projections and a gated feed-forward network.

For a sequence of length `S`, projection and MLP work scale linearly with `S` and quadratically with width. Attention score and value products scale quadratically with `S` and linearly with `D`:

:::equation FLOPs_{attention-pairs} ≈ 4S^{2}D|Two matrix products account for score construction and weighted value reduction.

At modest context length, dense projections and the MLP can dominate total FLOPs. At sufficiently long context, pairwise attention becomes controlling. The crossover depends on width, MLP ratio, kernel efficiency, causal masking, and whether attention intermediates are materialized.

During training, activation storage and backward work matter as much as forward FLOPs. During serving prefill, long sequences expose substantial parallelism. During autoregressive decode, only one new query position is processed at a time while weights and the accumulated KV cache are repeatedly read. The same architecture therefore presents different bottlenecks across its lifecycle.

### KV state makes architecture a serving decision

For each layer and sequence position, autoregressive serving retains keys and values. If there are `H_kv` key/value heads, head dimension `d`, element size `b` bytes, `L` layers, batch size `B`, and cached sequence length `S`, KV storage is approximately:

:::equation M_{KV} = 2 B L S H_{kv} d b|The factor two accounts for both keys and values.

Standard multi-head attention uses one K/V head per query head. Multi-query attention shares a single K/V head, while grouped-query attention uses an intermediate number. Reducing `H_kv` can cut persistent state and decode memory traffic dramatically without changing the number of query heads.

| Variant | Query heads | K/V heads | Serving consequence |
| --- | --- | --- | --- |
| Multi-head attention | `H` | `H` | Maximum K/V flexibility and largest cache |
| Grouped-query attention | `H` | `G`, with `1 < G < H` | Intermediate quality-state tradeoff |
| Multi-query attention | `H` | `1` | Minimum KV state and bandwidth |

The quality effect is empirical and architecture-dependent. The systems consequence follows directly from the state equation. This is a recurring pattern: the model architecture fixes the lower-level resource problem that serving infrastructure must later solve.

[GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) provides the primary evidence for the intermediate query/KV-head tradeoff. Its quality results depend on the evaluated checkpoints and uptraining recipe.

### Position and long-context behavior

Attention without positional information is permutation equivariant. Position mechanisms introduce order through embeddings, transformations, or score biases. Rotary position embeddings rotate query and key components as a function of position, causing their dot product to depend on relative displacement. ALiBi adds head-specific distance penalties directly to attention logits.

Increasing a configured maximum position does not create long-context capability. A model must encounter relevant long-range structure during training, and its position representation must behave sensibly outside the original distribution. Extension methods trade properties: interpolation can preserve phase behavior while reducing local positional resolution; frequency-aware changes may protect high-frequency components differently from low-frequency ones.

Long-context evaluation should separate capabilities that are often conflated:

- retrieving a known item among distractors;
- composing evidence across distant regions;
- preserving instructions through a long interaction;
- maintaining coherence during long generation;
- using repository, document, or conversation structure;
- avoiding regressions at ordinary context lengths.

The system evaluation must add prefill latency, peak activation memory, KV capacity, scheduling interference, and cancellation cost. A context extension that passes a retrieval diagnostic but destroys fleet goodput is not yet a product improvement.

### Mixture-of-experts changes the communication graph

A mixture-of-experts layer routes each token to a subset of feed-forward experts. If `E` experts exist and `k` are active for each token, parameter storage follows `E` while arithmetic per token follows `k`. That separation can increase model capacity without proportional dense computation.

The savings are not free. Routing creates variable token-to-expert assignments. In an expert-parallel system, tokens move through all-to-all communication, are grouped by expert, processed, and returned to their original positions. A hot expert creates padding, dropped tokens, or stragglers.

Let `N` tokens be routed with top-`k` selection across `E` experts. Under perfectly uniform routing, the expected assignments per expert are `kN/E`. Real routing is not uniform, so implementations define a capacity factor `c`:

:::equation Capacity_{expert} = ceil(c k N / E)|Capacity above the uniform expectation absorbs routing imbalance at additional memory and compute cost.

Auxiliary balancing losses, router noise, expert replication, and capacity policies manage this system. Perfectly uniform routing is not the product objective; useful specialization may be uneven. Measure task quality, overflow, load distribution, communication, and end-to-end utilization together.

### Architecture review across the lifecycle

An architecture should be reviewed as a lifecycle, not only as a forward pass:

1. **Data and objective:** Does the training signal exercise the intended capability?
2. **Optimization:** Are depth, normalization, routing, and precision stable?
3. **Training execution:** What are the dominant parameters, activations, FLOPs, and collectives?
4. **Serving prefill:** What determines time to first token and long-context capacity?
5. **Serving decode:** What state and bytes are read for each generated token?
6. **Quality:** Which capabilities or critical slices could regress?
7. **Operations:** Are kernels, quantization, observability, rollout, and fallback mature?
8. **Economics:** Does the architecture improve cost per successful outcome over its lifetime?

This review prevents local optimization. An MoE model can be compute-efficient and network-inefficient. A context extension can improve a benchmark and reduce concurrency. A novel layer can be mathematically attractive and operationally blocked by missing kernels or export support.

### Design Exercises

1. Derive how changing from 32 K/V heads to 8 affects KV memory while preserving 32 query heads.
2. Explain why attention can be quadratic in sequence length without dominating every transformer workload.
3. Compare a dense and MoE layer using storage, active FLOPs, communication, and failure behavior.
4. Design a long-context evaluation that includes both capability and system constraints.
5. Review an architectural proposal whose training FLOPs fall by 15 percent but whose serving kernel support is immature.

## Scale, Memory, and Performance Models

LEAD: Quantitative reasoning turns architecture into a falsifiable resource plan. Start with lower bounds for work, bytes, state, and synchronization; then measure the gap between those bounds and the realized system.

### Build a dimensional model

Every estimate should carry units. FLOPs, bytes, tokens per second, seconds, watts, and dollars are not interchangeable. Dimensional consistency catches many errors before benchmarking.

If an accelerator sustains `R` FLOPs per second on the relevant operation and a workload requires `F` FLOPs, the compute-only lower bound is `F/R` seconds. If it moves `M` bytes through a memory path sustaining `BW` bytes per second, the bandwidth-only lower bound is `M/BW` seconds. A roofline lower bound takes the larger:

:::equation t_{lower} ≥ max(F / R, M / BW)|Runtime cannot beat either the compute or data-movement requirement.

Arithmetic intensity is the ratio of work to bytes transferred:

:::equation I = F / M|Arithmetic intensity is measured in FLOPs per byte.

The ridge point `R/BW` separates operations that can become compute-bound from those whose intensity is too low to use peak arithmetic throughput. This is a model, not a guarantee. Dependencies, launch overhead, occupancy, layout, and communication can keep measured performance below both ceilings.

### Distinguish storage from traffic

Memory capacity asks how many bytes must coexist. Bandwidth asks how many bytes cross a path during an interval. A tensor can be small in storage and expensive in traffic if it is reread many times. Conversely, a large checkpoint can be operationally manageable when transferred infrequently and off the critical path.

For model training, account separately for:

- parameter storage;
- gradient storage;
- optimizer moments and master weights;
- saved activations;
- temporary workspaces and communication buffers;
- allocator fragmentation and framework overhead.

A rough mixed-precision AdamW configuration can require far more than the low-precision parameter size because gradients and optimizer states may use wider representations. The exact policy and sharding stage determine the total. Write the ledger explicitly rather than relying on a remembered bytes-per-parameter slogan.

### Understand queueing before saturation

Service latency rises nonlinearly as utilization approaches one because variability creates queues. In an idealized M/M/1 queue with arrival rate `λ` and service rate `μ`, utilization is `ρ = λ/μ`, and expected time in the system is:

:::equation E[T] = 1 / (μ - λ)|Even a simple queue shows why latency diverges as offered load approaches service capacity.

Real LLM services are not M/M/1 queues. Service time depends on prompt and output lengths; batching couples requests; decode reveals work one token at a time; priorities and memory admission change scheduling. The formula is useful because it establishes direction, not because it predicts p99 latency.

Capacity should therefore be defined under an SLO and workload distribution. Maximum tokens per second at an overloaded steady state is not sellable capacity. Report the arrival rate or goodput maintained while meeting latency, quality, rejection, and fairness requirements.

### Use scaling laws as allocation models

[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) and [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) are empirical allocation studies, not universal constants. The latter revisits how a fixed training budget is divided between parameters and tokens. Serving cost, data quality, and reuse can change the economic optimum.

Empirical scaling laws often approximate reducible loss with power-law relationships over a bounded regime. A schematic form is:

:::equation L(N, D) = L_{∞} + A N^{-α} + B D^{-β}|Loss decreases with model parameters N and training tokens D within the fitted regime.

If dense training compute is approximately proportional to `ND`, a fixed compute budget creates a constrained allocation between model size and data. The value of the model is not the exact fitted exponent. It is the ability to ask whether the next unit of compute has higher marginal return in parameters, tokens, data quality, or experimentation.

Extrapolation is dangerous. The fitted data distribution, architecture family, tokenizer, optimization recipe, and evaluation metric define the regime. Data exhaustion, repeated examples, context changes, and capability thresholds can break the curve. Always retain uncertainty bands and validate intermediate scales before committing a frontier run.

The compute-optimal training point may also differ from the lifetime product optimum. Serving a model billions of times can dominate its one-time training cost. A smaller model trained on more tokens may cost more to create but far less to operate. A frontier capability project may rationally prefer a larger model even when it is not the lowest-loss allocation under a simple compute constraint.

### Connect utilization to economics

Let `C_fleet` be fleet cost per hour, `G` the goodput in successful tasks per hour, and `C_other` the variable cost of retrieval, tools, storage, and network. A first-order serving cost is:

:::equation C_{task} = C_{fleet} / G + C_{other}|Economic efficiency follows constrained goodput, not nominal accelerator utilization.

High utilization is not automatically efficient. Padding, rejected work, repeated KV movement, or generations that miss the SLO can keep hardware busy without producing goodput. Conversely, deliberate headroom can reduce tail latency and failures enough to lower cost per successful task.

Capacity decisions should include the value of optionality. A design that uses every byte at nominal load may have no room for traffic bursts, longer contexts, model rollouts, or degraded hardware. Headroom is an engineered reliability reserve, not unexplained waste.

### Worked resource example

Consider a decoder with 32 layers, hidden width 4096, 32 query heads, 8 K/V heads, head dimension 128, and BF16 KV storage. For one sequence of 16,384 cached tokens:

:::equation M_{KV} = 2 × 32 × 16384 × 8 × 128 × 2 bytes|Keys and values are stored for every layer, position, and K/V head.

This is approximately 2 GiB per sequence before allocator overhead. A 16-request batch would require roughly 32 GiB of KV state. The estimate explains why paging, admission, GQA, prefix reuse, quantized KV, and context limits are fleet-level design choices.

The estimate is not the capacity plan. Add model weights, workspaces, graph pools, fragmentation, safety reserve, and the distribution of active sequence lengths. Then measure allocation behavior under cancellations, prefix sharing, and mixed prefills. The calculation identifies the controlling state; replay validates the operational policy.

### Design Exercises

1. Construct a parameter, gradient, optimizer-state, and activation memory ledger for a training configuration.
2. Explain why peak FLOPs and peak memory bandwidth cannot simply be added when estimating runtime.
3. A service reaches 95 percent GPU utilization but violates p99 latency. Identify measurements that distinguish useful work from overload.
4. Use the KV equation to compare MHA, GQA, and MQA for a fixed model width.
5. Explain when a compute-optimal model is not the economically optimal product model.

## Measurement and Experimental Judgment

LEAD: Measurement is the contract between a proposed change and the decision it is meant to support. A valid metric can still be useless if it observes the wrong population, boundary, or causal mechanism.

### Separate outcomes, guardrails, and diagnostics

Metrics serve different roles:

- **Outcome metrics** measure useful behavior: task success, accepted patches, resolved cases, or user preference.
- **System metrics** measure service delivery: latency, goodput, availability, rejection, and cost.
- **Guardrails** define unacceptable regressions: safety, privacy, fairness, reliability, or critical capability floors.
- **Diagnostics** expose mechanisms: gradient norm, router imbalance, acceptance rate, cache hit rate, queue depth, or memory fragmentation.

Diagnostics should not be promoted into outcomes merely because they move quickly. Speculative-decoding acceptance rate helps explain performance, but committed tokens per second under quality and latency constraints determines whether the technique is valuable. GPU utilization helps explain resource use, but cost per successful task is closer to the product outcome.

### Build an evaluation matrix

Aggregate scores hide asymmetric failures. Cross task categories with user populations, context length, language, difficulty, risk, and system conditions. Declare which slices are decision-critical before observing results.

| Dimension | Example slices | Failure hidden by an aggregate |
| --- | --- | --- |
| Task | code, reasoning, retrieval, summarization | One high-volume easy task dominates the mean |
| Context | short, medium, long, multi-turn | Long-context regression disappears in averages |
| Population | language, domain, experience | Minority or specialized users lose quality |
| System | warm, cold, burst, degraded | Benchmarks omit queueing and recovery |
| Risk | ordinary, sensitive, adversarial | Rare high-impact failures lack statistical mass |

Every evaluation result should bind the model checkpoint, tokenizer, prompt or policy version, engine, decoding configuration, dataset release, metric implementation, and environment. Without semantic identity, a result cannot be reproduced or compared safely.

### Match the experiment to the claim

Evidence should live at the same boundary as the claim. A unit test can establish an algebraic invariant. A kernel benchmark can establish local speed and numerical error. An engine replay can establish behavior under representative shapes. A shadow or canary can establish integration and operational effects. An online experiment can establish user response.

A robust progression is:

1. analytical model and invariant tests;
2. component correctness and microbenchmarks;
3. integrated offline evaluation;
4. production-shape replay;
5. shadow execution without user-visible output;
6. bounded canary with automatic rollback;
7. controlled rollout with segment analysis;
8. durable monitoring after launch.

Not every change needs every stage. The smallest honest experiment is the least expensive test that observes the claimed effect and its important risks.

### Measure latency without hiding queueing

Define the measurement boundary: client-to-client, gateway-to-gateway, engine-only, or kernel-only. For streaming generation, separate:

- queueing and admission delay;
- time to first token;
- inter-token latency or token cadence;
- end-to-end completion time;
- cancellation and retry behavior.

A closed-loop load generator that waits for one response before sending the next reduces offered load when the system slows. This coordinated omission hides the queueing collapse users would experience. Open-loop generation schedules arrivals independently of completions and records rejection as an outcome.

Report latency conditional on prompt length, output length, request class, and load. A single p99 across a shifting traffic mix cannot distinguish a slower system from a harder workload.

### Quantify uncertainty and practical significance

For an estimated `mean(x)` with sample standard deviation `s` and sample size `n`, a large-sample standard error is:

:::equation SE(mean(x)) = s / √n|Sampling uncertainty falls with the square root of independent sample count.

Independence matters. Multiple turns from one user, repeated prompts, or correlated benchmark variants reduce effective sample size. Bootstrap procedures should resample at the unit of independence, such as user, repository, or conversation, rather than blindly resampling rows.

Statistical significance does not establish practical importance. Predefine the minimum effect that changes the decision. Guardrails may use asymmetric standards: a small uncertain regression on a critical safety slice can block launch even when the primary average improves.

Inspecting many slices creates multiple-comparison risk. Organize metrics into declared primary outcomes, hard guardrails, and exploratory diagnostics. Exploratory findings generate hypotheses for confirmation; they should not be presented as if they were pre-registered conclusions.

### Reason causally about system changes

A before-and-after comparison mixes the effect of a change with traffic, hardware, data, and time. Randomization is powerful when units do not interfere, but system experiments often violate that assumption: requests share queues, caches, replicas, or capacity.

Choose the randomization unit to match interference. Request-level randomization may work for a stateless decoding policy. Replica- or cluster-level randomization may be safer for schedulers and memory allocators. Time-based switchbacks can compare fleet-wide policies but must account for trends and carryover.

Record treatment assignment and exposure separately. A request assigned to a new engine but served by fallback did not receive the intended treatment. Analyze intention-to-treat for product impact and actual exposure for mechanism diagnosis, while preserving the distinction.

### Write the decision before running the experiment

A concise experiment specification contains:

Example status: Explanatory pseudocode.

```text
Decision:     Which choice will this result change?
Mechanism:    Why should the proposed change affect the outcome?
Population:   Which workload and users does the evidence represent?
Treatment:    What exactly differs, and how is exposure recorded?
Metrics:      Primary outcome, system constraints, guardrails, diagnostics.
Sensitivity:  What minimum effect can the design reliably detect?
Execution:    Assignment, duration, ramp, stop, and rollback rules.
Analysis:     Segments, uncertainty, interference, and missing data.
Follow-up:    Ship, revise, or retire, with owner and review date.
```

Writing the decision first prevents metric fishing. It also reveals experiments that cannot change action because no threshold or owner exists.

:::callout decision|Measurement is part of architecture
Systems improve only through failures they can observe and decisions they can revisit. Instrumentation, semantic versioning, replay, and rollback are architectural capabilities, not reporting work added after implementation.
:::

### Worked decision: a new KV-cache layout

Suppose a new layout is expected to reduce decode memory transactions by 18 percent. The component claim is lower bytes per token with identical numerical output. The product claim is higher goodput or lower latency under representative workloads.

The evidence ladder is:

1. property tests confirm logical indexing, masking, sharing, and cancellation invariants;
2. numerical comparison covers dtypes, lengths, page boundaries, and ragged batches;
3. profiler counters test the predicted reduction in bytes;
4. engine replay measures token cadence, batch occupancy, fragmentation, and peak memory;
5. fault tests exercise cancellation, eviction, and worker restart;
6. a canary compares SLO-constrained goodput and critical quality outputs;
7. rollout monitors regressions by sequence-length and hardware class.

If profiler bytes fall but end-to-end latency does not, the result is not a failed experiment. It rejects the assumption that memory traffic controlled the observed workload and directs attention toward launch overhead, synchronization, queueing, or another phase.

### Design Exercises

1. Design an evaluation matrix for a multilingual coding assistant with tool use.
2. Explain why confidence intervals cannot rescue an invalid workload or contaminated benchmark.
3. Choose a randomization unit for comparing two continuous-batching schedulers.
4. A kernel is 25 percent faster in isolation and has no measurable service effect. List the most likely missing mechanisms.
5. Turn “the new model feels better” into an experiment with a decision threshold and guardrails.

### Foundation Exercise Answer Criteria

A strong response to any foundation exercise should make the decision testable rather than merely name a technique. Use these chapter-level criteria to review your work:

1. **System contract:** identify the user outcome, workload distribution, scale, SLO, quality floor, failure semantics, and the observation that would reopen the design.
2. **Optimization:** follow one update through data sampling, forward and backward numerics, clipping, optimizer state, distributed agreement, checkpointing, and recovery. Separate a statistical change from an execution change.
3. **Architecture:** account for parameters, active FLOPs, activations, KV state, communication, and serving consequences. Compare at least one credible alternative under the same workload.
4. **Resource model:** state units, derive the dominant memory or time term, include utilization and queueing headroom, and test the estimate against a measured counter.
5. **Measurement:** define the population, treatment, randomization unit, primary outcome, guardrails, minimum meaningful effect, uncertainty, rollout, and stop or rollback rule.

An answer is incomplete if it depends on an unstated workload, quotes an average where the tail controls the decision, or proposes a metric without naming the action that metric can change.

### Foundation Principles

The rest of this book repeatedly returns to six principles:

1. **Define the outcome and workload before selecting the mechanism.**
2. **Treat numerical and distributed execution as part of the learned algorithm.**
3. **Follow parameters, activations, bytes, and state across the complete lifecycle.**
4. **Use simple quantitative models to expose assumptions and choose measurements.**
5. **Evaluate constrained goodput and useful outcomes, not isolated proxy metrics.**
6. **Design semantic identity, observability, and rollback into the system from the beginning.**

These principles are more durable than any model family or accelerator generation. They make new techniques legible: identify what changes, which resource or behavior should move, what could regress, and what evidence would justify adoption.
