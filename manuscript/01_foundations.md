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

Let `C_run` be the total cost of a representative run, `N_task` its attempted tasks, and `p_success` its task-success fraction. With at least one success, the observed cost per successful task is:

:::equation C_{success} = C_{run} / (N_{task} p_{success})|Cost must be normalized by a useful outcome, not only by generated tokens.

For planning, expected cost per attempt divided by success probability is a long-run ratio under a stable workload, not the expectation of a finite run's random cost/success ratio. A run with zero successes has no finite observed cost per success. Count failed attempts, retries, tools, and review in the numerator; do not silently count only the successful requests' costs.

For example, at equal tokens per attempt and no change in other costs, a baseline costing $1 per attempt with success probability 0.40 costs $2.50 per success in the long run. An alternative costing $1.20 with probability 0.55 costs about $2.18. If its outputs become longer or require more review, those assumptions no longer hold. Conversely, aggressive quantization can improve raw throughput yet lose economically through retries or escalations.

The same discipline applies to system throughput. Report **goodput**: work completed while satisfying quality and service constraints. Tokens produced after an SLO deadline, requests later discarded, and generations that fail validation consume capacity but do not count as useful output.

### Treat workload as a joint distribution

Averages erase the correlations that determine system behavior. Prompt length, output length, tenant, model variant, adapter, tool calls, latency priority, and arrival time are not independent. Enterprise users may send longer prompts and demand stricter isolation. Code requests may generate longer outputs and have higher speculative-decoding acceptance. Traffic bursts may coincide with colder caches.

Represent the workload as a joint distribution over request attributes:

:::equation W = P(S_{in}, S_{out}, A, Q, T, R)|A workload model preserves correlations among input length, output length, arrival process, quality class, tenant, and request type.

Here `S_in` and `S_out` are input and output token counts, `A` records arrival timing or burst context, `Q` is the quality/service class, `T` is tenant identity, and `R` is request type. These symbols are local to this workload model; for example, `T` does not mean a token budget here. Preserve the time order of arrivals when replaying queueing behavior, not only their per-request attributes. For a mixed workload, estimate total costs and successful tasks from the same slice weights or representative replay.

The model need not be analytically elegant. A versioned trace with privacy-safe fields is often more useful than independent parametric distributions. What matters is that benchmarks reproduce the shapes and correlations that influence queueing, memory, kernel efficiency, and quality.

Use percentiles and conditional distributions rather than a single representative request. A p95 prompt combined with a p95 output is not necessarily a real p95 request; it may describe a combination that almost never occurs. Capacity planning should replay or sample from joint observations, then stress explicit adversarial cases separately.

### Establish invariants and failure semantics

Requirements describe desired outcomes. Invariants describe what must remain true while the system changes or fails. Examples include:

- a token may be streamed only from the model and policy versions recorded for the request;
- two tenants may never share a prefix-cache entry without an authorized identity boundary;
- a checkpoint is visible only after every required shard and manifest is durable;
- a run exposes a committed optimizer step only when all participating ranks have advanced consistently; partial failure triggers recovery;
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

## From Text to Tokens, Targets, and Loss

LEAD: Before a model can learn or generate, text must become a sequence of discrete decisions. Tokenization, conversation serialization, and loss masking define those decisions; they are part of the model, not interchangeable preprocessing.

### Why a token is not a word

A vocabulary maps integer IDs to pieces of text or bytes. A word-level vocabulary handles frequent words cheaply but needs a policy for unseen names. Character or byte vocabularies avoid many unknown-token problems but lengthen sequences. Subword vocabularies trade these costs: common strings use fewer positions while rare strings remain representable through smaller pieces.

The [subword BPE paper](https://arxiv.org/abs/1508.07909) adapts repeated pair merging to language segmentation. In a small byte-level example, start with the bytes of `low low`. A merge table first combines `l` with `o`, then `lo` with `w`, giving the pieces `[low, space, low]`. Training the merge table counts pairs in a corpus; encoding new text applies learned merge ranks. Encoding does not retrain the vocabulary on each request. Real tokenizers also specify normalization, pretokenization boundaries, reserved symbols, and decoding rules.

Unigram tokenization instead starts from candidate pieces with probabilities and chooses a segmentation according to their scores. Training removes less useful candidates while retaining coverage. [SentencePiece](https://aclanthology.org/D18-2012/) is a tokenizer implementation framework supporting subword models directly over raw text, not a synonym for one segmentation algorithm. Neither BPE nor a unigram model guarantees that linguistic words, numbers, or code identifiers occupy one token.

In a byte-level scheme, one token can end halfway through a multibyte Unicode character. Decode the accumulated bytes with a streaming decoder rather than decoding every token independently and inserting replacement characters. Normalization can also change the original byte sequence. If exact copying of source code, identifiers, or signed text matters, specify which transformations are permitted and test round trips on those inputs.

The tiny `bpe_pieces` reference in `examples/sequence_models.py` deliberately omits normalization and pretokenization. Its purpose is to make ranked merging and byte reconstruction inspectable, not to reproduce a production tokenizer.

### Vocabulary size moves cost rather than removing it

For vocabulary size `V` and hidden width `D`, an embedding table holds `V*D` parameters. The output projection has the same shape transposed; weight tying shares the parameters but does not remove output-projection computation. A larger vocabulary may shorten a sequence, while increasing the table, logits, and sampling work.

Consider an illustrative model with `D=4096`. Increasing a tied vocabulary from 32,000 to 128,000 entries adds `96000*4096=393216000` parameters, or about 786 MB at two bytes each. If token count falls by 20 percent on a fixed raw-text workload, projection/MLP work per document roughly falls with sequence length, while full attention's pair count falls toward `0.8²=0.64` of the original. These are separate effects; output-head cost and changed language coverage can reverse the overall choice.

Measure tokens per byte or character by language and domain, not only on English prose. A tokenizer that fragments one script heavily imposes a shorter effective context and more generation steps on those users. Compare models on the same raw evaluation text and task outcomes. Per-token perplexities under different tokenizers are not directly comparable because the prediction units differ. Bits per byte can help when normalization and byte accounting are also fixed.

### Autoregression is a factorization, not a decoding trick

:::diagram token_alignment|Inputs and targets refer to different positions. Each prediction uses its causal prefix; the supervised target is the following token. BOS and EOS denote sequence boundaries.

A causal language model represents a sequence probability as a product of next-token conditional probabilities. Taking logs converts that product into a sum:

:::equation log p(x_{1:S}) = Σ_{t=1}^{S} log p(x_{t} given x_{<t})|The prediction at position t must not observe its own target or a later token.

Teacher forcing supplies the known prefix during training. For tokens `[BOS, A, B, EOS]`, the three input positions `[BOS, A, B]` predict `[A, B, EOS]`. A causal mask permits each input position to read itself and earlier input positions, but not later ones. Training evaluates many such predictions in parallel because the true prefix is known. Generation must obtain the next token before it knows the next prefix.

The model emits one logit per vocabulary item. Softmax turns the logits into probabilities, and cross entropy for the observed target is the negative log of its probability. If three valid targets receive probabilities `0.5, 0.25, 0.5`, the mean loss is about `0.924` nats and perplexity is about `2.52`. Perplexity is the exponential of mean negative log probability, not the fraction of correct answers or a calibrated confidence in an entire response.

Compute log probabilities with log-sum-exp: subtract the maximum logit before exponentiating, then restore it in the logarithm. The reference `masked_token_loss` implements this calculation without depending on a deep-learning framework. It rejects a batch with no supervised targets rather than dividing by zero.

The gradient connects this objective to learning. For one target `y` and logits `z`, differentiating `-z_y + log(Σ_i exp(z_i))` gives `p_i - indicator(i=y)` for logit `i`. With two zero logits and target 0, probabilities are `[0.5,0.5]` and the gradient is `[-0.5,0.5]`. An SGD step of size 0.1 on those logits gives `[0.05,-0.05]`, increasing the target probability to about 0.525. A real network propagates this logit gradient through its layers by the chain rule; it does not optimize each token's logits as free parameters.

### Three masks with different jobs

| Mask | What it controls | A common mistake |
| --- | --- | --- |
| Causal/attention mask | Which positions may influence a hidden state | Letting an earlier prediction see a later answer |
| Padding mask | Which positions are real sequence data | Reading padded keys or scoring padding as text |
| Loss mask | Which target predictions contribute to the objective | Training on user/tool text unintentionally |

A loss-masked prompt still influences the assistant's hidden states through attention. Removing its loss does not remove it from context. Conversely, masking attention to a token does not automatically remove that position's target from the objective.

Some loss APIs encode excluded targets with a sentinel such as `-100`, which is not a vocabulary ID. The CPU reference permits such a target only where its explicit loss mask is false; supervised targets must remain valid IDs. Its logits must still be finite even at ignored positions. That is a deliberate validation policy, not a substitute for defining the framework's ignore-index and reduction behavior.

For assistant-only SFT, serialize roles and message boundaries using the model's template, then mark assistant target spans. Decide whether to learn end-of-turn markers and tool-call syntax. Keep a short hand-checked trace showing input IDs, decoded pieces, target IDs, attention visibility, and loss weights. This catches off-by-one labels and incorrect assistant spans before an expensive run.

When packing independent documents into one tensor, decide whether to permit cross-document attention. A block-diagonal causal mask preserves independent examples; plain concatenation changes the training distribution by exposing one document to another. Reset or preserve positions consistently with that policy. Mask the artificial next-document transition if it is not a desired target. Packing is not only a padding optimization.

### A checkpoint includes its text interface

An integer ID has meaning only under the matching vocabulary and embedding row. Reordering IDs while retaining the same tensor shape silently corrupts a model. Adding special tokens requires corresponding embedding/output rows and training for their use. Changing a chat template can alter behavior without touching weights.

Version tokenizer files, normalization, special-token policy, template, stopping rules, and weights together. Test an ordinary conversation, an empty message, a tool exchange, Unicode, code whitespace, and literal strings that resemble control markers. Treat untrusted text as message content; do not let it create privileged roles merely by containing a delimiter.

### Exercises and worked answers

1. **Why can a bigger vocabulary make serving slower?** It adds embedding/logit bytes and output-head work. Measure any sequence shortening against those costs at the actual batch and language mix.
2. **Does masking prompt loss hide the prompt from the model?** No. Attention visibility and supervision are different masks. Prompt tokens can remain the entire conditioning signal.
3. **How do you compare loss across tokenizers?** Use an identical raw-text corpus and a compatible byte-normalized measure, then compare task behavior. Equal per-token loss does not imply equal text likelihood.
4. **What is the minimum packing regression test?** Compare two examples run separately with their packed block-diagonal version, checking valid-position logits and summed loss under the same position policy and numerical tolerance.

## Optimization as a Coupled Dynamical System

LEAD: Training is not an optimizer acting on a fixed objective. It is a coupled dynamical system whose trajectory depends on data order, batch construction, numerical representation, distributed execution, and the rules used to recover from failure.

:::diagram optimization_loop|Follow the state around one optimizer step. The next forward pass uses updated weights; optimizer moments and scheduler progress persist across steps.

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

With moments initialized at zero and the first update numbered `t=1`, bias correction divides by the accumulated averaging weight: `m_t^c=m_t/(1-β_1^t)` and `v_t^c=v_t/(1-β_2^t)`. For the constant first gradient 2, `β_1=0.9`, and `β_2=0.99`, the raw moments are 0.2 and 0.04, but the corrected moments are 2 and 4. The adaptive step is:

:::equation θ_{t+1} = θ_{t} - η_{t} m_{t}^{c} / (√v_{t}^{c} + ε)|The superscript c denotes bias-corrected first and second moments.

AdamW applies weight decay as a separate shrinkage term rather than inserting an L2 penalty into the adaptively scaled gradient:

:::equation θ_{t+1} = (1 - η_{t} λ) θ_{t} - η_{t} m_{t}^{c} / (√v_{t}^{c} + ε)|Decoupled decay preserves a clearer distinction between optimization and parameter shrinkage.

That distinction matters because adaptive scaling would otherwise make the effective regularization coordinate-dependent. It does not imply that AdamW is universally superior. The optimizer trades memory, communication, convergence behavior, and robustness. Optimizer states may consume more memory than the parameters themselves; sharding or quantizing those states changes the execution plan.

### Matrix-aware updates: what Muon changes

AdamW scales gradient coordinates using elementwise moment estimates. Muon instead uses a momentum matrix and approximately orthogonalizes its update direction for selected matrix-shaped parameters. Using a compact singular-value decomposition over the nonzero singular directions, `M=UΣV^T`, the idealized direction `UV^T` removes their relative magnitudes. This produces orthonormal columns for a full-column-rank tall matrix, or rows for a full-row-rank wide matrix, not necessarily a square orthogonal matrix. Rank-deficient directions need a convention; this idealization leaves zero directions at zero, including an all-zero update. Practical matrix iterations approximate the transformation rather than computing a full SVD each step. Update scale, momentum, and numerical precision all matter.

[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) studies weight decay and update-scale choices for large-model use. It does not establish one universal optimizer setting for every architecture. Embeddings, output heads, vectors, and matrix parameters need an explicit optimizer assignment; a hybrid optimizer configuration is not an implementation error.

An original two-dimensional example makes the distinction concrete. For diagonal momentum `diag(100, 1)`, an unnormalized momentum step is dominated by the first direction. Ideal orthogonalization yields `diag(1, 1)` before the chosen overall scale. This balances directions, but also discards magnitude information that might be useful. The method is not equivalent to normalizing the whole matrix by its norm.

Compare optimizers at matched data, model, validation targets, and compute accounting. Include matrix-iteration time and any momentum all-gathers needed by sharded training. A lower number of tokens to reach a loss can coexist with a more expensive optimizer step. Checkpoint the optimizer assignment and states, and rerun learning-rate/decay sweeps rather than copying an AdamW recipe unchanged.

### Batch size changes statistics and execution

The global batch combines microbatching, gradient accumulation, and data-parallel replication:

:::equation B_{global} = B_{micro} × N_{accum} × N_{data}|Equal global batch does not imply equal execution behavior.

Here batch sizes count sequences of a declared shape; with ragged sequences, also count supervised tokens. Increasing `B_micro` usually raises activation memory and can improve local matrix shapes. Increasing `N_accum` delays parameter updates; it reduces gradient communication only if intermediate microbatches suppress synchronization, for example with a framework's no-sync path. Increasing `N_data` expands communication and changes how examples are partitioned. The same global batch can therefore have different throughput, numerical behavior, and failure exposure.

Equal weighting of rank-local means is not a global token mean when supervised counts differ. If rank A has one target with loss 4 and rank B has three targets with loss 0, the mean of local means is 2, but the global token mean is 1. With R replicas whose gradients are averaged, each rank should backpropagate `R * local_loss_sum / global_target_count` to recover the global token-mean gradient. Sum the count over the complete accumulation window; do not independently average differently sized microbatches. A rank with zero targets must still participate in collectives, and a globally empty window needs a coordinated no-update policy. The [DDP reduction documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html) specifies the averaging assumption; custom reduction hooks can change it.

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

:::equation g' = g × min(1, c / norm_{2}(g))|For c > 0, clipping caps gradient norm and preserves the direction of a nonzero finite gradient.

Define the zero-gradient result as zero rather than evaluating `c/0`. A gradient `[3,4]` has norm 5; with threshold 2 it becomes `[1.2,1.6]`. For plain SGD without other update terms, this also bounds parameter motion by `learning_rate * 2`. AdamW's moment history, adaptive scaling, and weight decay mean gradient clipping alone does not impose that same bound on the final parameter update.

In a sharded run, `norm_2(g)` counts each logical gradient element once. Sum squared norms over disjoint shards, then take one square root; do not sum identical data-parallel replicas as though they were distinct shards. Clipping each shard independently implements a different rule. Unscale before computing the norm, reject non-finite values before clipping, and accumulate norm statistics safely enough to avoid overflow. Log the unclipped norm, clipped fraction, update norm, and loss-scale state so that clipping does not silently hide instability.

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
A lower-precision configuration must be compared at matched quality. If step duration falls by 20 percent but reaching the quality target requires 30 percent more steps, total time is `0.80 * 1.30 = 1.04` times the baseline: four percent slower.
:::

### Residual scale, normalization, and depth

Transformer training depends on maintaining useful signal and gradient scales through many residual blocks. A generic residual update is:

:::equation x_{l+1} = x_{l} + α_{l} f_{l}(x_{l})|Residual parameterization controls how new transformations accumulate with depth.

If residual contributions behave like independent random variables with similar variance, an unscaled sum can grow with depth. Real networks violate the independence assumption, but the model explains why initialization and depth-aware scaling matter. Choices such as pre-normalization, post-normalization, residual scaling, and parameter initialization jointly determine the early training regime.

Pre-normalized blocks place normalization before attention or the feed-forward sublayer. This creates a relatively direct residual path for gradients and is commonly stable in deep models. Post-normalized blocks normalize after the residual addition and can exhibit different representation behavior but often require more careful initialization or scaling.

LayerNorm controls mean and variance; RMSNorm controls root-mean-square magnitude without subtracting the mean. The FLOPs saved by RMSNorm are rarely the primary architectural reason, although simpler normalization can enable efficient fusion. The choice should be evaluated through training stability, quality, precision sensitivity, and kernel support.

For the normalization mechanism and its evaluated benefits, see [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467); do not transfer its reported timings to another model without measurement.

### A training step needs a distributed commit and recovery contract

A production step contains more state than parameters and gradients. It advances optimizer moments, learning-rate schedule, random streams, data-sampler position, loss scale, gradient accumulation counters, and monitoring windows. Checkpoint recovery must restore these surfaces consistently.

The following is an execution contract, not a framework API or a crash-atomic transaction. It assumes a replay-on-overflow policy and an update-based schedule; another valid design can discard an overflowing batch, but must record that choice and distinguish consumed tokens from committed-update tokens. Accumulation, loss scaling, and finiteness decisions apply to the whole effective batch.

Example status: Explanatory pseudocode; framework integration and recovery are not implemented.

```text
Reserve the effective batch; retain its replay identity.
Count supervised targets over all ranks and microbatches.
If the global count is zero: record an empty window; skip it.
Zero gradients; keep one loss scale for this window.
For each microbatch:
    Predict shifted targets with causal visibility.
    Normalize the local loss sum by the global token count.
    Compensate for gradient averaging, if the reducer averages.
    Backpropagate; defer collectives only where supported.
Finish gradient reduction; unscale exactly once.
Agree globally that gradients and the logical norm are finite.
If not finite:
    Clear gradients; lower the shared loss scale if applicable.
    Replay the reserved batch, or stop after a bounded retry.
Else:
    Clip the unscaled logical gradient with one global factor.
    Apply the optimizer update consistently on all ranks.
    Advance the update-based schedule and committed cursor.
    Update the shared scale policy; record the step outcome.
```

When implementing token loss in PyTorch, logits shaped `[B,S,V]` cannot be passed unchanged to a class-dimension-one loss. For shifted predictions, flatten `logits[:, :-1, :]` to `[B*(S-1), V]` and targets to `[B*(S-1)]`, or move the vocabulary axis into the required position. Apply the matching shifted target mask to unreduced losses. See [CrossEntropyLoss shapes](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

[PyTorch's AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html) show unscaling before clipping and `scaler.update()` after stepping. A Boolean finite flag is not that API's scale-update argument. Sharded execution needs a compatible distributed overflow/scaler mechanism; a local optimizer or scaler must not independently override the agreed step decision. Do not advance an update-based schedule when the optimizer actually skipped.

Agreement does not make a crash halfway through optimizer writes atomic. On a partial failure, fence the affected run and restore a consistent checkpoint rather than keeping some ranks' advanced parameters. Replay also needs the appropriate RNG, sampler, accumulation, and optimizer state. Distributed Checkpoints, Recovery, and Elasticity develops that recovery protocol.

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
3. A configuration shortens step duration by 18 percent but needs 25 percent more tokens to reach target loss. At unchanged tokens per step, construct an honest comparison.
4. Design a recovery invariant for skipped steps in a 256-worker job.
5. A run clips 40 percent of steps while validation improves. What evidence distinguishes a useful safety bound from concealed instability?

## Transformer Architecture as Resource Allocation

The core attention construction comes from [Attention Is All You Need](https://arxiv.org/abs/1706.03762). This chapter adapts that mechanism to a decoder-serving resource model; its byte and FLOP estimates are derivations under the stated assumptions, not measurements from that paper.

LEAD: A transformer is a schedule for moving information among tokens and channels. Each architectural choice reallocates parameters, arithmetic, memory traffic, communication, and persistent state across training and inference.

### Follow one token through a decoder block

:::diagram decoder_block|The two residual additions preserve a direct path around attention and the MLP. Attention includes its output projection; each sublayer returns to the residual stream's width.

Let the input to layer `l` be a matrix `X_l` with sequence length `S` and hidden width `D`. A pre-normalized decoder block can be written schematically as:

:::equation U_{l} = X_{l} + Attention(Norm(X_{l}))|Attention mixes information across token positions.

:::equation X_{l+1} = U_{l} + MLP(Norm(U_{l}))|The feed-forward network mixes information across channels independently at each position.

This representation highlights two kinds of mixing. Attention creates dependencies among positions; the MLP transforms each position through a larger intermediate space. Residual paths carry earlier representations forward and provide a route for gradients.

For `H` query heads and head dimension `d = D / H`, the projections form query, key, and value tensors. Scaled dot-product attention is:

:::equation A = softmax(Q K^{T} / √d + M)|The mask M encodes causal or structural constraints.

:::equation O = A V|Each output position is a weighted reduction over value vectors.

The equations specify semantics, not an efficient execution plan. Materializing the full `S × S` score and probability matrices creates quadratic memory traffic. IO-aware kernels tile the computation, retain partial softmax statistics on chip, and avoid writing those intermediates to high-bandwidth memory.

### Expand the operations, not only the block names

With row-vector tokens, a projection is `Q = X W_q`: `X` is `S × D`, and `W_q` maps channels into query heads. Split its last dimension into heads, apply the position transformation, and calculate one `S × S` score matrix per head. Causal masking sets scores for future keys to negative infinity **before** softmax. Concatenate the head outputs and apply `W_o` before the residual addition. Batch dimensions repeat this computation; they must never become another attention axis.

For GQA with `H` query heads and `G` KV heads of width `d`, the reshaped tensors are `Q: [B,H,S,d]` and `K,V: [B,G,S,d]`. In the usual evenly grouped layout, `H` is divisible by `G`; each group of `H/G` query heads reads the same KV head. Scores and softmax still have one row per query head and query position. Softmax normalizes over eligible **key positions**, not heads or batch entries. The per-head outputs concatenate back to `[B,S,H*d]` before the output projection.

### Trace one attention row

Use a one-dimensional head so the scaling factor is one. Let query `q=[1]`, keys be `[[0],[log(3)],[100]]`, and scalar values be `[[2],[6],[999]]`. At the second causal position, only the first two keys are visible. Their scores are `[0,log(3)]`; softmax weights are `[1/4,3/4]`, so the output is `2/4 + 18/4 = 5`. The future key contributes nothing despite its enormous score. Masking it after normalization would be wrong because it would already have consumed probability mass.

Example status: Runnable excerpt; execute from the repository root.

```python
from math import log
from examples.attention import attention

q, keys = [1.0], [[0.0], [log(3.0)], [100.0]]
values, visible = [[2.0], [6.0], [999.0]], [True, True, False]
for partition in [1, 2, 3]:
    out = attention(q, keys, values, visible, partition)
    assert abs(out[0] - 5.0) < 1e-12
```

The same reference merges separately scaled partial softmax statistics from key partitions; Part IV derives the stable merge. An all-masked row has no mathematical softmax distribution. This fixture returns zero by convention, while a model integration must explicitly reject the row or define a compatible output policy.

### Channel mixing, normalization, and rotations

In a gated feed-forward block, two projections serve different roles:

:::equation MLP(X) = (SiLU(X W_{g}) ⊙ (X W_{u})) W_{d}|The gate and up projections have width F; the down projection returns to width D.

Here `SiLU(z) = z / (1 + exp(-z))`, and the circled multiplication symbol means elementwise multiplication. The nonlinear gate controls how much of each up-projected channel reaches the down projection. It is not an expert router: every token still executes these dense projections. For a scalar gate input zero, the gated intermediate is zero regardless of the up value; for gate input two and up value three, it is about `1.762 × 3 = 5.285` before the down projection. This explains why a gated MLP has three weight matrices rather than the two of a conventional activated MLP. See [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) for the architectural comparison.

RMSNorm also has a concrete reduction. For channels `[3, 4]`, unit learned gains, and negligible epsilon, the denominator is `sqrt((9 + 16)/2) = 3.536`, giving approximately `[0.849, 1.131]`. Normalization happens along channels within each token, not across the batch. An epsilon protects a zero input, and production reductions generally accumulate more accurately than their stored operands.

For one two-channel pair, RoPE applies the rotation `R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]`. Since `R(a)^T R(b) = R(b-a)`, the rotated query/key dot product depends on their relative angle. A vector `[1, 0]` at angle zero dotted with the same vector at angle `π/2` gives zero, not one. Different pairs use different frequencies, and implementations differ in whether pairs are adjacent or split across the head. Loading the right weights with the wrong pairing convention silently changes the model. See [RoFormer](https://arxiv.org/abs/2104.09864).

### Account for parameters, FLOPs, and activation state

In a dense decoder block, attention projections contribute on the order of `4D²` parameters when query, key, value, and output widths all equal `D`. A gated MLP with intermediate width `F` contributes roughly `3DF` parameters because it commonly uses two input projections and one output projection.

The exact constants matter for capacity planning, but the scaling terms provide the first decision model:

:::equation P_{block} ≈ 4D^{2} + 3DF|Approximate parameters in attention projections and a gated feed-forward network.

For a sequence of length `S`, projection and MLP work scale linearly with `S` and quadratically with width. Attention score and value products scale quadratically with `S` and linearly with `D`:

:::equation FLOPs_{attention-pairs} ≈ 4S^{2}D|Dense all-pairs score and value products, counting a multiply-add as two FLOPs.

This estimate is for all `S²` pairs; an implementation that skips the masked causal triangle has roughly half the pair-product work at long S, plus tile-boundary overhead. Softmax, projections, normalization, and the output vocabulary head are separate costs. For the stated dense MHA block, projections plus the gated MLP require about `8SD² + 6SDF` forward FLOPs. With `F=4D`, the all-pairs term equals that linear-in-S work at `S≈8D`; skipping the causal triangle moves the arithmetic crossover toward `16D`. These are toy FLOP crossovers, not latency predictions. GQA changes projection constants, and real MLP widths and kernel efficiencies change the comparison.

At modest context length, dense projections and the MLP can dominate total FLOPs. At sufficiently long context, pairwise attention becomes controlling. Materializing intermediates can make attention a memory bottleneck before that FLOP crossover.

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

Unmasked, content-only self-attention is permutation equivariant: permuting the input rows permutes the output rows. With a structural mask, this statement requires permuting the mask's rows and columns too. A causal decoder's fixed triangular mask already imposes an order through visibility, so arbitrary input permutations with that mask held fixed do not preserve the claim. Positional mechanisms add explicit location or distance signals through embeddings, transformations, or score biases. Rotary position embeddings rotate query and key components as a function of position; ALiBi adds head-specific distance penalties to attention logits.

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

## Compressed, Sparse, and Recurrent Model State

LEAD: Modern language models do not all retain one explicit key and value per head per token. To compare architectures, ask what information is stored, how a new token reads it, and which approximation or learned bottleneck makes the state cheaper.

### Start from the dense attention contract

:::diagram attention_state_map|Three read patterns over eight historical positions, followed by two different storage mechanisms. A local or sparse read pattern does not itself specify eviction, head sharing, or latent compression.

For one query, softmax attention compares that query with every permitted key and returns a normalized weighted sum of values. GQA shares K/V heads but retains token-addressable history. FlashAttention changes the execution schedule while preserving this mathematical operator, up to floating-point differences. Sparse attention, latent compression, and recurrent state change different parts of that contract.

| Family | Persistent history | New-query work | What must be evaluated |
| --- | --- | --- | --- |
| MHA, GQA, or MQA | Explicit K/V for each token and its KV-head layout | Read all allowed history | Head-sharing quality and KV traffic |
| Local or sliding-window attention | Explicit K/V inside a fixed or layer-specific window | Read the permitted recent region | Lost distant evidence and window-boundary behavior |
| Latent attention | Learned compressed token state plus position state | Read compressed history; transform query/output | Compression capacity and efficient absorbed projections |
| Learned sparse or compressed attention | Token or compressed states plus a selection/index mechanism | Select and read a subset or compressed summary | Evidence missed by compression/selection and index cost |
| Recurrent/linear mixer | Fixed-size state per layer/head | Update/read state | Interference, forgetting, and long-range retrieval |
| Hybrid | A mixture of growing caches and fixed states | Depends on layer schedule | Combined memory, rollback, and quality |

The table is a map of mechanisms, not a quality ranking. A hybrid can use several rows simultaneously, and the training recipe determines whether the cheaper state learns useful behavior.

### Multi-head Latent Attention: compress before caching

[DeepSeek-V2](https://arxiv.org/abs/2405.04434) introduced Multi-head Latent Attention, or MLA, as a learned low-rank representation of K/V history with a separate position-handling path. Its serving attraction is avoiding storage of the fully expanded per-head keys and values. The low-rank representation is learned with the model; it is not a lossless compressor applied afterward to an arbitrary GQA checkpoint.

The underlying algebra can be understood with a simplified single-head, position-free example. Let a token's latent vector be `c` of width `r`, with `k=W_k c` and `v=W_v c`. Its score against query `q` is `q^T W_k c`, which equals `(W_k^T q)^T c`. Transform the query once, then compare it with cached latent vectors. Likewise, `Σ a_i W_v c_i = W_v(Σ a_i c_i)`: aggregate latents first and expand the output afterward. This avoids reconstructing every old value on every step.

These equalities require compatible linear operations. Position-dependent rotations cannot generally be absorbed into one constant matrix, which is why the actual architecture separates positional and non-positional components. Normalization, head grouping, and projection layout also belong in the real execution plan. A kernel that expands all K/V vectors before attention can give back much of the intended bandwidth benefit.

For an illustrative comparison, ordinary GQA with eight KV heads of width 128 stores `2*8*128=2048` values per token per layer. A hypothetical latent cache of width 512 plus 64 positional values stores 576. At two bytes each and 32 layers, that is 128 KiB versus 36 KiB per token. This is a storage calculation, not a claim that the hypothetical models have equal quality or equal compute. Projection weights, workspaces, and any replicated latent state remain additional costs.

### Sparse attention: selection becomes part of the model

Sparse attention replaces the full set of eligible keys with a smaller set. A local window is a fixed rule; a learned indexer predicts relevant positions; block-sparse methods select groups to improve memory locality. The resulting operator can be exact on its selected support, but it is generally not the same result as dense softmax over all history.

Consider scores `[0, 0, log(8)]` and scalar values `[0, 0, 10]`. Dense attention returns `8`. A selector that misses the third key returns zero. Accurate arithmetic inside the sparse kernel cannot repair the missed evidence. This is why indexer recall and end-task quality must be evaluated together.

[Native Sparse Attention](https://arxiv.org/abs/2502.11089) makes learned hierarchical selection a first-class computation, combining compressed global summaries, selected token blocks, and a local window in a hardware-aligned design trained end to end. The [DeepSeek-V4 report](https://arxiv.org/abs/2606.19348) further combines token compression with sparse retrieval through compressed and heavily compressed attention paths. Compression reduces the candidates' representation cost; selection reduces which candidates receive expensive attention. They are distinct levers, and neither implies unbounded lossless memory.

Let context length be `S`, selected count `k`, index cost `I(S)`, and full per-key attention cost `a`. Sparse work is closer to `I(S)+ak` than simply `ak`. A selector that scans a compact representation of all positions can still be linear in `S`, though with a smaller constant than full attention. At short contexts, index launches and gathers may cost more than dense attention. If `k>=S`, a correct fast path can skip selection and attend to all valid positions.

Test repeated identifiers, many similar distractors, multiple required passages, and evidence outside the local window. Measure selector time, index memory, top-k validity, gather locality, attention time, and the dense/short-context fallback. Report the quality budget separately from the speedup.

### From an attention history to a matrix memory

A simple unnormalized linear associative memory stores a matrix `S` with shape `[value_dim, key_dim]`. At each position, it adds an outer product `v k^T`; a query reads `S q`. Expanding the recurrence gives a weighted sum of past values, with weights `k_i^T q`. In this raw-dot-product example the weights need not be positive or sum to one; some other linear-attention formulations add positive feature maps and a normalizing state. A fixed-size matrix also cannot retain arbitrarily many independent associations without interference.

For example, start at zero and write scalar values 2 and 5 with the same unit-norm key. Reading with that key gives 7 under a plain additive update, not the latest value 5. The delta rule corrects what the memory already predicts for the incoming key instead of repeatedly adding the whole value.

### Gated DeltaNet: forget globally, correct selectively

The [Gated DeltaNet paper](https://arxiv.org/abs/2412.06464) combines a decay gate with a key-directed correction. In a simplified head, let `alpha` be the retention gate, `beta` the update gate, and let `k` have unit norm:

:::equation S_{old}' = α S_{old}|Decay applies to the old memory before the correction is computed.

:::equation e = v - S_{old}' k|The error is the incoming value minus the decayed memory's prediction for this key.

:::equation S_{new} = S_{old}' + β e k^{T},  o = S_{new} q|The rank-one correction targets one key direction; the query then reads the updated memory.

Take a one-row state `[2, 9]`, key `[1, 0]`, and incoming value 5. With both gates equal to one, the prediction is 2, the error is 3, and the new state is `[5, 9]`. The unrelated second direction is preserved. With `alpha=0.5` and `beta=0.5`, decay gives `[1, 4.5]`, the error is 4, and correction gives `[3, 4.5]`. Gates therefore control two different operations. The tests in `tests/test_sequence_models.py` check these exact examples and state replay.

This recurrence is the teaching mechanism, not a complete model block. Real implementations add learned projections, head grouping, short convolutions, gates, normalization, and output projections. State dtype can be wider than the input dtype because repeated updates accumulate error. Unit-norm keys make the overwrite interpretation particularly clear; do not assume that behavior for arbitrary key norms.

[Kimi Linear](https://arxiv.org/abs/2510.26692) extends this family with Kimi Delta Attention and combines linear-attention layers with MLA layers. Its reported KV and throughput gains are results for the evaluated architecture, kernels, contexts, and quality comparisons—not a drop-in guarantee for another model. The systems lesson is that “linear attention” now names a co-designed model and execution path: gate granularity, state width, chunkwise training kernels, recurrent decode kernels, and the fraction and placement of full-attention layers must be evaluated together.

Training need not execute a Python loop over all tokens. Rewriting a step as an affine state map permits composition of chunks. For maps `S -> S A_1+B_1` and then `S -> S A_2+B_2`, the combined map is `S -> S(A_1 A_2)+B_1 A_2+B_2`. Associativity creates parallelism, although a practical kernel exploits structure instead of materializing large dense transition matrices. Decode uses the recurrent form because one new token arrives at a time. The two schedules should agree numerically within a declared tolerance.

### State-space models and selective recurrence

A state-space layer also summarizes history, but its parameterization is not the delta rule. A simple discrete system has hidden state `h_t`, input `u_t`, and output `y_t`:

:::equation h_{t} = A_{t} h_{t-1} + B_{t} u_{t}; y_{t} = C_{t} h_{t}|Transition, input, and readout maps control what is retained, written, and observed.

For scalar `A=0.5`, `B=C=1`, zero initial state, and inputs `[2,0,4]`, the states and outputs are `[2,1,4.5]`. The old input decays instead of remaining as an individually addressable KV entry. Structured transitions make larger states affordable. Selective state-space models make aspects of the transition/write/read depend on the current input, so different tokens can be retained differently rather than following one fixed convolution.

[Mamba-2](https://arxiv.org/abs/2405.21060) develops structured state-space duality and an efficient chunked computation connecting these recurrences to structured matrix operations. “Transformers are SSMs” in that paper's title does not mean every full-softmax transformer can be replaced by the same fixed-size state with identical outputs. The useful connection concerns structured computations and schedules, with architectural assumptions.

Compare the scalar example with attention: a later query cannot necessarily recover the exact first input from the single value 4.5. Many input histories lead to that state. Increasing state size, learning selective retention, or adding attention layers changes the capability/resource tradeoff; it does not eliminate compression. During serving, preserve the recurrent state and any local-convolution history together. During training, validate chunked outputs and gradients against a sequential reference on short sequences before trusting a fast scan implementation.

### Hybrids retain more than one kind of cache

The official [Qwen3.5-35B-A3B base model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base) describes a repeated three-Gated-DeltaNet/one-gated-attention pattern, with MoE feed-forward blocks. This is a concrete hybrid example, not a rule for every model carrying the family name. Its recurrent layers and full-attention layers have different serving-state contracts.

Suppose a hypothetical 32-layer model uses 24 recurrent layers and eight GQA layers. With 16 recurrent heads, `d_k=d_v=128`, and four-byte state, matrix memory is `24*16*128*128*4 = 24 MiB` per sequence, before convolution state. The eight GQA layers still need 32 KiB per cached token under the earlier eight-KV-head assumptions, or 64 MiB at 2,048 tokens. At one token, recurrent state can be larger than the tiny attention cache; at long context it avoids growth in those 24 layers. “Linear attention uses less memory” needs a crossover and a full hybrid ledger.

Prefix reuse now means restoring both attention blocks and the matching recurrent boundary state. Speculative rejection must roll back recurrent and convolution state as well as truncate attention KV. Saving only the current matrix cannot recover an arbitrary earlier prefix. Use snapshots at supported boundaries or replay accepted tokens from a saved state. These are architecture requirements that a generic paged-KV allocator alone does not satisfy.

### Exercises and worked answers

1. **Why is MLA not ordinary KV quantization?** It changes the learned representation and permits projection algebra; quantization changes numerical storage of a chosen representation. They can be combined.
2. **Can a perfect sparse kernel recover an omitted key?** No. The operator's support has already excluded that evidence; improve selection or use a fallback.
3. **What does `beta=0` do in the recurrence?** It disables the correction, but decay still acts if `alpha<1`. It is not necessarily a no-op.
4. **What must hybrid speculative rollback restore?** Attention positions, recurrent matrices, convolution buffers, and any position/RNG state required by the exact verification contract.
5. **How would you choose among the families?** First compare task quality at the desired context. Then measure stored state, per-token reads, projection/index work, prefill scheduling, and rollback support under the same service SLO.

## Scale, Memory, and Performance Models

LEAD: Quantitative reasoning turns architecture into a falsifiable resource plan. Start with lower bounds for work, bytes, state, and synchronization; then measure the gap between those bounds and the realized system.

### Build a dimensional model

Every estimate should carry units. FLOPs, bytes, tokens per second, seconds, watts, and dollars are not interchangeable. Dimensional consistency catches many errors before benchmarking.

Choose attainable throughput ceilings `R` FLOPs per second and `BW` bytes per second for the operation, dtype, and memory path. If a workload requires `F` FLOPs and moves `M` bytes through that path, the idealized compute and bandwidth time floors are `F/R` and `M/BW`. A roofline model takes the larger:

:::equation t_{actual} ≥ t_{lower} = max(F / R, M / BW)|The actual runtime cannot beat either requirement under the chosen throughput ceilings.

Arithmetic intensity is the ratio of work to bytes transferred:

:::equation I = F / M|Arithmetic intensity is measured in FLOPs per byte.

The ridge point `R/BW` separates operations that can become compute-bound from those whose intensity is too low to reach the arithmetic ceiling. A throughput measured on an unrelated inefficient workload is not an upper ceiling and cannot establish a strict lower time bound. Dependencies, launch overhead, occupancy, layout, and communication can keep measured performance below both ceilings.

For example, with `F=1e12` FLOPs, `M=1e11` bytes, `R=1e14` FLOPs/s, and `BW=1e12` bytes/s, the floors are 0.01 and 0.10 seconds. The roofline time floor is 0.10 seconds, not their sum; the model permits compute and transfer to overlap. Doubling arithmetic throughput alone leaves this floor unchanged. A dependency that forces phases to run serially can make actual time larger, so apply the model to phases with compatible overlap assumptions.

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

Service latency rises nonlinearly as utilization approaches one because variability creates queues. An idealized M/M/1 queue has Poisson arrivals, independent exponential service times, one server, and an unlimited waiting room. With arrival rate `λ` and service rate `μ`, utilization is `ρ = λ/μ`. A stationary mean exists only when `λ < μ`; in that regime, expected waiting plus service time is:

:::equation E[T] = 1 / (μ - λ)|Even a simple queue shows why latency diverges as offered load approaches service capacity.

For capacity `μ=10` requests/s, raising arrivals from 5 to 9 requests/s raises mean system time from 0.2 to 1.0 seconds, although mean service time stays 0.1 seconds. At or above capacity there is no finite steady-state mean under this unlimited-queue model; do not substitute `λ>μ` and interpret the negative algebraic result as latency.

Real LLM services are not M/M/1 queues. Service time depends on prompt and output lengths; batching couples requests; decode reveals work one token at a time; priorities and memory admission change scheduling. The formula is useful because it establishes direction, not because it predicts p99 latency.

Capacity should therefore be defined under an SLO and workload distribution. Maximum tokens per second at an overloaded steady state is not sellable capacity. Report the arrival rate or goodput maintained while meeting latency, quality, rejection, and fairness requirements.

### Use scaling laws as allocation models

[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) and [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) are empirical allocation studies, not universal constants. The latter revisits how a fixed training budget is divided between parameters and tokens. Serving cost, data quality, and reuse can change the economic optimum.

Empirical scaling laws often approximate reducible loss with power-law relationships over a bounded regime. A schematic form is:

:::equation L(N, D) = L_{∞} + A N^{-α} + B D^{-β}|Loss decreases with model parameters N and training tokens D within the fitted regime.

If dense training compute is approximately proportional to `ND`, a fixed compute budget creates a constrained allocation between model size and data. The value of the model is not the exact fitted exponent. It is the ability to ask whether the next unit of compute has higher marginal return in parameters, tokens, data quality, or experimentation.

To solve the toy allocation, write the fixed budget as `ND = K`, substitute `D=K/N`, and differentiate `A N^(-α) + B K^(-β) N^β` with respect to N. At its interior optimum, `α A N^(-α) = β B D^(-β)`: marginal returns to model size and data balance under the constraint. For positive fitted coefficients and exponents, `N* = (α A / (β B))^(1/(α+β)) * K^(β/(α+β))`, then `D*=K/N*`. Thus parameter and token counts grow with equal exponents in the budget only when `α=β`; their absolute ratio still depends on the fitted constants and units.

For a dimensionless toy problem, let `n` and `d` be parameter and token counts divided by fixed reference scales. Minimize `4/n + 1/d` with `n*d=16`. The optimum is `n=8, d=2`, giving loss contribution 1; choosing `n=d=4` gives 1.25. Equal exponents do not imply equal counts. With equal exponents the optimum balances the two reducible-loss contributions, not necessarily raw parameters and tokens. This derivation does not justify a universal tokens-per-parameter ratio.

Extrapolation is dangerous. The fitted data distribution, architecture family, tokenizer, optimization recipe, and evaluation metric define the regime. Data exhaustion, repeated examples, context changes, and capability thresholds can break the curve. Always retain uncertainty bands and validate intermediate scales before committing a frontier run.

The compute-optimal training point may also differ from the lifetime product optimum. Serving a model billions of times can dominate its one-time training cost. A smaller model trained on more tokens may cost more to create but far less to operate. A frontier capability project may rationally prefer a larger model even when it is not the lowest-loss allocation under a simple compute constraint.

### Connect utilization to economics

Let `C_fleet` be fleet cost per hour, `G` the goodput in successful tasks per hour, and `C_other` the retrieval, tool, storage, and network cost allocated **per successful task**, including those resources spent on failed attempts. A first-order serving cost is:

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

A closed-loop load generator waits for completion before issuing replacement work, so offered load falls when the system slows. That can accurately model a fixed population of sequential clients. It becomes misleading when used to represent externally scheduled arrivals: omitted requests and their waiting times hide the queueing collapse those users would experience. Open-loop generation schedules arrivals independently of completions and records rejection as an outcome. Choose the model to match the workload, and verify that the load generator itself can sustain the intended arrival schedule.

Report latency conditional on prompt length, output length, request class, and load. A single p99 across a shifting traffic mix cannot distinguish a slower system from a harder workload.

### Quantify uncertainty and practical significance

For an estimated `mean(x)` with sample standard deviation `s` and sample size `n`, a large-sample standard error is:

:::equation SE(mean(x)) = s / √n|Sampling uncertainty falls with the square root of independent sample count.

Independence matters. Multiple turns from one user, repeated prompts, or correlated benchmark variants reduce effective sample size. Bootstrap procedures should resample at the unit of independence, such as user, repository, or conversation, rather than blindly resampling rows.

Statistical significance does not establish practical importance. Predefine the minimum effect that changes the decision. Guardrails may use asymmetric standards: a small uncertain regression on a critical safety slice can block launch even when the primary average improves.

For a concrete paired comparison, run both systems on the same 100 independent tasks. Suppose the new system alone succeeds on 15 and the baseline alone succeeds on 5. Define each paired difference as +1, -1, or 0. The mean improvement is 0.10; its sample standard deviation is `sqrt((20 - 100*0.10²)/99)`, about 0.438, giving standard error about 0.0438. A rough normal 95 percent interval is `0.10 ± 1.96*0.0438`, or about 1.4 to 18.6 percentage points. That uncertainty matters even though the point estimate is ten points. This approximation is not a guarantee for small or highly unbalanced samples; use an appropriate paired analysis, and resample repositories or users instead if those are the independent units. To establish a minimum useful improvement of five points, this interval is not yet convincing.

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
Metrics:      Outcome, system constraints, guardrails, diagnostics.
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

1. property tests confirm logical indexing, masking, sharing, and cancellation invariants;
2. numerical comparison covers dtypes, lengths, page boundaries, and ragged batches;
3. profiler counters test the predicted reduction in bytes;
4. engine replay measures token cadence, batch occupancy, fragmentation, and peak memory;
5. fault tests exercise cancellation, eviction, and worker restart;
6. a canary compares SLO-constrained goodput and critical quality outputs;
7. rollout monitors regressions by sequence-length and hardware class.

If profiler bytes fall but end-to-end latency does not, the result is informative. Reduced traffic may lie outside the critical path, or another cost may have offset the saving. Inspect launch overhead, synchronization, queueing, and phase timing before concluding that memory traffic was irrelevant.

### Design Exercises

1. Design an evaluation matrix for a multilingual coding assistant with tool use.
2. Explain why confidence intervals cannot rescue an invalid workload or contaminated benchmark.
3. Choose a randomization unit for comparing two continuous-batching schedulers.
4. A kernel is 25 percent faster in isolation and has no measurable service effect. List the most likely missing mechanisms.
5. Turn “the new model feels better” into an experiment with a decision threshold and guardrails.

:::pagebreak

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
