# Part II - Training, Data, and Distillation

Training systems turn a capability hypothesis into a reproducible model. The hard work is not only gradient descent. It is choosing data, objectives, curriculum, precision, parallelism, checkpoints, evaluation, and stop conditions that reinforce one another.

## Designing a Training Recipe

LEAD: A recipe is an executable theory of how a model will acquire a capability within a compute and time budget. Begin with the target behavior and work backward to data and objective.

### The recipe canvas

Before selecting hyperparameters, write a one-page canvas:

- **Product behavior:** tasks, users, languages, modalities, context, and safety boundary.
- **Model constraints:** parameter budget, serving hardware, latency, throughput, memory, and quantization target.
- **Data assets:** licensed corpora, synthetic generation, preference labels, verifier signals, and contamination risks.
- **Training stages:** pretraining, continued pretraining, supervised fine-tuning, preference or RL stage, distillation, and compression.
- **Evaluation gates:** capability, generalization, safety, calibration, and system performance.
- **Operations:** failure recovery, lineage, reproducibility, checkpoint cadence, and ownership.

:::diagram training_pipeline|A production recipe is a loop. Serving failures become curated examples, evaluation updates, and the next training stage.

### Work backward from serving

If the product needs high-volume interactive decode, architecture and tokenizer choices should reflect KV-cache and output-length costs. If the product needs code completion, data freshness, exact syntax, repository context, and executable verification matter more than generic preference labels. If the product needs a speculative draft, target-distribution agreement and acceptance speed matter more than standalone benchmark leadership.

A common failure is to train the most capable model that fits the training cluster and discover later that it cannot meet the serving SLO. The lifecycle objective is closer to:

`value = quality_gain - training_cost - lifetime_serving_cost - operational_risk`

The terms are not directly commensurate, but writing them forces a complete decision.

### Stage design

**Pretraining** builds broad representations from next-token prediction or a related self-supervised objective. **Continued pretraining** shifts domain, language, freshness, or context distribution while protecting general capability. **Supervised fine-tuning** teaches response formats and high-quality trajectories. **Preference optimization** shifts behavior among plausible outputs. **Distillation** transfers a distribution or capability into a smaller serving envelope.

Stages should be justified by what signal they add. Do not add a preference stage because it is fashionable if verified demonstrations already specify the desired answer. Do not use supervised imitation for an objective that requires exploration and delayed reward.

### Curriculum and mixture

Data mixture weights determine the gradient distribution. Sampling every source in proportion to raw size lets abundant low-value sources dominate. Equal sampling overweights tiny domains and can cause memorization. Temperature sampling interpolates between these extremes.

Curriculum can mean ordering by difficulty, capability, context length, quality, or domain. It is useful when later data assumes skills learned earlier or when expensive long sequences should be introduced after the model is stable. It is harmful when the sequence creates catastrophic forgetting or a distribution cliff.

#### Mixture control loop

1. Define capability slices with leading metrics.
2. Estimate source-to-slice influence through ablations or data attribution proxies.
3. Choose a baseline mixture that covers product distribution.
4. Train small pilots across a sparse set of mixture changes.
5. Fit a response model only as complex as the evidence supports.
6. Update the mixture and preserve held-out confirmation data.

:::callout pitfall|Synthetic volume is not synthetic value
Teacher generations reproduce teacher biases and can collapse diversity. The useful unit is a verified, novel training signal, not a generated token.
:::

### Schedules and stop conditions

Warmup protects early training when moment estimates and activation scales are unstable. Cosine or linear decay then reduces update size as the run approaches a local basin. Restart schedules can help exploration but complicate reproducibility and interpretation.

Stopping only when training loss plateaus wastes compute. Use a portfolio of signals: validation loss, target capability, data-slice trends, gradient noise, checkpoint-to-checkpoint improvement, projected value of more tokens, and schedule milestones. A predeclared review point makes it easier to stop a prestigious run whose marginal return has collapsed.

### Checkpointing and recovery

A distributed checkpoint must capture parameters, optimizer states, scheduler, RNG state, data-loader position, tokenizer and configuration hashes, parallel topology assumptions, and code version. Asynchronous checkpointing reduces pause time but requires a consistency boundary. Sharded checkpoints reduce write hotspots but make restore topology and format evolution first-class design problems.

Use periodic full checkpoints plus more frequent lightweight or incremental protection. Test restore before the expensive run. A checkpoint that has never been restored is a hypothesis.

### Recipe interview: a reasoning model

For a reasoning model, start with a strong base model. Construct tasks with verifiable answers, collect diverse solution trajectories, and filter by both correctness and strategy diversity. Use supervised training to establish format and basic reasoning. Add outcome- or process-based optimization only where the reward is reliable. Prevent reward hacking with hidden tests, adversarial cases, and holdout generators. Track answer accuracy, pass@k, calibration, token efficiency, and failure types.

:::callout insight|A recipe answer should have gates
Name what evidence permits progression from pilot to scale-up, from supervised training to online optimization, and from offline evaluation to serving. Gates show that you can operate the program, not merely describe it.
:::

### Principal Interview Review

1. Design a recipe for a code model that must run on a fixed single-GPU serving tier.
2. When would continued pretraining be safer than changing the base mixture from the start?
3. How would you decide whether to spend the next budget increment on data, model size, or more steps?
4. What belongs in a resumable checkpoint for a sharded optimizer?
5. How do you prevent a reasoning model from learning to exploit its verifier?

## Data Curation and Synthetic Data

LEAD: Data quality is a pipeline of explicit decisions about identity, rights, utility, diversity, and risk. "Clean the data" is not a design.

:::diagram data_pipeline|A dataset is the result of transformations and gates. Every retained example should have provenance and a reason to exist.

### Source contracts and provenance

For every source, record ownership or license, acquisition time, original identifier, permitted uses, deletion obligations, language, domain, and transformation lineage. Hashes enable identity checks but do not replace source IDs because normalization changes bytes.

Provenance must survive shuffling and packing. If a later safety incident requires removal, the system should identify derived examples, synthetic descendants, trained checkpoints, and evaluations affected by the source.

### Normalization without erasing signal

Normalization standardizes encoding, line endings, markup, and obvious boilerplate. Aggressive normalization can destroy code indentation, mathematical structure, conversational turns, or document boundaries. Keep raw and normalized references; version transformation logic; sample diffs by domain.

Language identification should return a distribution or confidence, not only a hard label. Mixed-language documents are common. Route uncertain or code-heavy text through specialized logic instead of dropping it indiscriminately.

### Deduplication at several radii

Exact duplication wastes tokens and increases memorization. Near duplication includes templated pages, mirrors, lightly edited articles, and repeated code. Semantic duplication includes paraphrases or generated variants that carry little new training signal.

Use several stages:

1. content hashes after conservative normalization for exact duplicates;
2. shingled MinHash or locality-sensitive hashing for near-duplicate text;
3. structural signatures for code and templates;
4. embedding or model-based similarity only where its cost is justified;
5. group-aware splitting so related documents cannot cross train and evaluation.

The cluster representative policy matters. Keeping the shortest document can remove context; keeping the longest can preserve spam. Score representatives on quality, provenance, completeness, and recency.

```python
def assign_near_duplicate_cluster(doc, bands, index):
    signature = minhash(shingles(normalize(doc.text), width=5))
    candidates = set()
    for band_id, band_hash in split_into_bands(signature, bands):
        candidates |= index.lookup(band_id, band_hash)
    match = best_jaccard_match(doc, candidates)
    if match and match.score >= doc.threshold:
        return match.cluster_id
    return index.create_cluster(doc, signature)
```

### Quality scoring

Quality is multidimensional. Signals may include language fluency, information density, coherence, citation structure, code validity, authoritativeness, originality, toxicity, and fit to target capability. A single scalar score is useful for ranking but dangerous for governance because tradeoffs become hidden.

Train scorers on carefully defined labels and test calibration by domain. Adversarially inspect high-scoring examples. A model can learn that length, formal tone, or particular sites imply quality, then systematically suppress concise or minority-domain material.

:::callout decision|Filter, weight, or route
Hard-filter examples only for policy, corruption, or very low expected value. Use weights when utility is continuous. Route specialized data into domain mixtures when the data is valuable but distributionally distinct.
:::

### Contamination and benchmark leakage

Contamination can occur through direct benchmark text, solution discussions, derivative tasks, teacher prompts, or synthetic paraphrases. Search exact and fuzzy overlaps before training, but also protect the evaluation supply chain: restrict access, log joins, separate credentials, and rotate private tests.

Report contamination risk rather than pretending it is binary. Evaluate on time-split, source-isolated, and newly authored sets. For code, use hidden tests and repository-level splits. For reasoning, generate parameterized variants with independently verified answers.

### Synthetic generation pipeline

Synthetic data is a programmable acquisition process:

`seed -> generate -> verify -> score -> deduplicate -> balance -> audit -> train`

Seed selection controls coverage. Generation temperature and prompting control diversity. Verification may be executable, symbolic, retrieval-based, consensus-based, or human. Rejection rules must not collapse the exact rare behaviors the project wants to learn.

For coding, compile, run hidden tests, lint, and measure complexity. For mathematics, use symbolic checks and alternative solvers. For open-ended reasoning, require evidence, decompose claims, and use adjudication on uncertain samples. Teacher confidence is not independent verification.

### Distribution balancing

Balancing should reflect the desired post-training workload while reserving capacity for rare but high-cost failures. If 1 percent of requests are safety-sensitive, naive frequency weighting may undertrain them; uniform category weighting may distort ordinary behavior.

Use a constrained objective: maximize expected utility on the product distribution subject to minimum performance on critical slices and limits on regression. Track the effective sample weight, not only raw counts.

### Data pipeline system design

At large scale, separate immutable object storage, metadata catalog, distributed transformations, feature or score stores, and dataset manifests. Make every dataset a content-addressed manifest of shards and transformation versions. Validate row counts, token counts, language mix, duplication, score distributions, and holdout overlap before publication.

Operational requirements include idempotence, partial reruns, backpressure, deletion propagation, audit logs, budget accounting, and reproducible sampling. Data systems fail silently when teams can overwrite a dataset name without changing its visible version.

### Principal Interview Review

1. Design a data-generation pipeline for speculative decoding.
2. How would you evaluate whether a new quality filter helps rather than merely shortens the dataset?
3. Explain train-test contamination beyond exact string matching.
4. A model-based quality scorer rejects dialectal text. How do you diagnose and repair the pipeline?
5. When should synthetic data receive lower sampling weight than human-authored data?

## Distillation, KL Divergence, and Model Sizing

LEAD: Distillation is distribution transfer under a capacity and serving constraint. The central question is not "teacher or student?" It is which information should cross the capacity boundary.

:::diagram distillation|The teacher provides a structured target; the student must convert it into utility inside a different architecture and budget.

### Cross entropy and KL

Let teacher distribution be `p(y|x)` and student distribution be `q(y|x)`. Cross entropy from teacher to student is:

`H(p, q) = - sum_y p(y|x) log q(y|x)`

It decomposes as `H(p) + KL(p || q)`. The teacher entropy does not depend on student parameters, so minimizing teacher cross entropy is equivalent to minimizing forward KL.

`KL(p || q) = sum_y p(y) log(p(y) / q(y))`

Forward KL heavily penalizes the student for assigning too little probability where the teacher has mass. It is often described as mode covering. Reverse KL, `KL(q || p)`, weights regions under the student and tends to concentrate on one teacher mode when representing all modes is expensive.

These slogans are helpful but incomplete. Token distributions are conditional, the student is parameterized, the support is finite after softmax, and sequence-level behavior compounds local choices. Always connect the divergence to the actual objective and sampling process.

### Temperature

With temperature `T`, logits `z` become:

`p_i(T) = exp(z_i / T) / sum_j exp(z_j / T)`

Higher temperature softens the distribution and exposes relationships among non-argmax tokens. Lower temperature sharpens it. In classic distillation, multiplying the KL term by `T^2` compensates for the gradient scale change introduced by temperature.

```python
def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.7):
    teacher = softmax(teacher_logits / T, dim=-1)
    student_logp = log_softmax(student_logits / T, dim=-1)
    soft = kl_div(student_logp, teacher, reduction="batchmean") * (T * T)
    hard = cross_entropy(student_logits, labels)
    return alpha * soft + (1.0 - alpha) * hard
```

The production version masks padding, normalizes by valid tokens, handles sharded vocabulary, controls teacher precision, and avoids storing full teacher logits when bandwidth dominates.

### Why combine soft and hard targets

Soft targets transfer relative preference among tokens. Hard labels preserve ground truth or verified behavior when the teacher is wrong. The balance depends on teacher reliability, student capacity, domain shift, and whether labels represent a unique answer.

For generated sequence data, a hard next-token label from a teacher sample discards uncertainty. Offline logits preserve more information but are enormous. Alternatives include top-k logit storage, quantized logits, on-policy teacher queries, sequence-level ranking, hidden-state targets, or verifier-selected trajectories.

### Sequence and feature distillation

**Sequence distillation** trains on teacher-generated outputs. It can simplify the target distribution and transfer style or reasoning patterns, but it inherits teacher errors and may reduce diversity. **Feature distillation** aligns hidden representations or attention maps, which can help when architectures are compatible. Layer mapping, scale, and representational non-identifiability complicate it.

**Online distillation** queries the teacher on student-relevant states, reducing mismatch between a fixed corpus and the student's current failure modes. It costs teacher inference and creates a moving data distribution.

### Draft-model distillation

A speculative draft is evaluated jointly with the target. The most relevant objectives are acceptance rate, average accepted prefix, draft latency, target verification cost, memory footprint, and final exactness.

Train on real product prompts and target continuations. Include high-entropy positions, domain slices, long-context states, and student failure cases. Match tokenizer and vocabulary unless the system explicitly supports a mapping; otherwise verification and KV reuse become much harder.

An effective loss can combine hard target tokens, forward KL on target logits, and emphasis on positions that control rejection. Evaluate under the actual sampling policy. A draft that agrees under greedy decoding may disagree at production temperature.

:::callout insight|The best draft is not the best small model
The best draft maximizes end-to-end target throughput under memory and latency constraints. Standalone perplexity is a proxy; acceptance per microsecond is closer to the objective.
:::

### Model-size tradeoff

A larger draft costs more per proposed token but can raise acceptance. A smaller draft is cheap but may create verification waste. Let:

- `C_d(m)` be draft cost for model size `m`;
- `C_v(gamma)` be target verification cost for `gamma` proposed tokens;
- `A(m, gamma)` be expected accepted tokens;
- `C_base` be one ordinary target decode step.

A simple speed model is:

`speedup = A(m, gamma) * C_base / (C_d(m, gamma) + C_v(gamma))`

The model omits queueing and overlap but identifies the decision. Benchmark a grid of draft size and proposal length using production shapes. Watch memory capacity: loading a larger draft can reduce target batch size and erase its acceptance benefit.

### Anti-distillation and extraction resistance

"Anti-distillation" covers techniques that make model extraction less effective: rate limits, query anomaly detection, output perturbation, watermarking, restricted logit access, policy controls, and legal or contractual measures. There is no free defense. Perturbing outputs can harm users and may not stop a determined extractor. Detection must distinguish legitimate high-volume customers from imitation workflows.

The Principal framing is a threat model: attacker access, budget, query adaptivity, target fidelity, and acceptable product degradation. Treat claims of protection as measured risk reduction, not impossibility.

### Principal Interview Review

1. Explain forward versus reverse KL using behavior and gradients, not only slogans.
2. Why multiply a temperature-scaled KL term by `T^2`?
3. Design a draft-model training corpus and evaluation.
4. When can a 3B draft outperform a 1B draft end to end?
5. What defenses against extraction preserve normal user quality?

## Post-Training and Alignment

LEAD: Post-training chooses behavior among capabilities the base model can express. The objective, data, and evaluation must agree on what "better" means.

### Supervised fine-tuning

SFT is appropriate when experts can demonstrate the desired behavior. It teaches format, policy, tool use, domain tone, and solution trajectories. High-quality demonstrations often beat a much larger weak corpus.

Mixing must protect broad capability. Include replay from general instruction data when specializing. Mask user or tool tokens according to the learning objective. For multi-turn data, test whether the loss incorrectly rewards copying system content or private tool outputs.

### Preference data

Pairwise preferences can be collected from humans, verifiers, judges, or product behavior. Each source has bias. Human labels reflect rubric clarity and annotator context. Model judges may favor verbosity, familiar style, or their own outputs. Engagement data is confounded by ranking, position, and user intent.

Build calibration sets with expert adjudication. Measure agreement by slice. Retain ties and uncertainty rather than forcing every comparison into a binary label.

### RLHF and direct preference objectives

Classic RLHF trains a reward model and optimizes a policy while constraining divergence from a reference. It supports online sampling and sequence-level reward, but it is operationally complex and vulnerable to reward hacking.

Direct Preference Optimization converts pairwise preferences into a supervised-style objective relative to a reference policy. It removes the explicit reward-model-and-RL loop, simplifying training. Other objectives alter the link function, treatment of a reference, margin, or unpaired feedback.

Do not choose by acronym. Ask:

- Is reward available only at the sequence level?
- Must the policy explore new outputs?
- How reliable and stationary is the preference signal?
- Is an explicit reward model useful for analysis or reuse?
- How much divergence from the base model is safe?

### Verifiable reward and reasoning

Code execution, theorem checks, database answers, and simulated environments provide strong outcome signals. They still permit shortcut exploitation. Split generators and verifiers, randomize hidden tests, audit suspiciously short or repetitive traces, and evaluate on tasks unavailable to the training loop.

Process supervision labels intermediate steps and can improve diagnosis, but step correctness may be ambiguous and expensive to annotate. Outcome supervision is cheaper when final answers are verifiable but provides sparse credit. Hybrid designs use outcome reward plus auxiliary signals for format, tool validity, or progress.

:::callout pitfall|A better reward score is not automatically a better model
Optimization changes the policy distribution, so the reward model is evaluated out of its training distribution. Track independent capability and safety metrics, divergence, response length, and exploit signatures.
:::

### Online learning and rollback

Online post-training creates feedback loops. New policy outputs influence the data later used to judge or train it. Protect a stable holdout, log exposure probabilities, version policies and reward models, and preserve the ability to reconstruct which policy generated each sample.

Use canaries and conservative trust regions. A rollback should restore model, tokenizer, tool policy, prompt templates, safety configuration, and serving parameters as one versioned release.

### Principal Interview Review

1. When does DPO simplify a problem that does not require online RL?
2. Design defenses against reward hacking for a code reasoning model.
3. How do you audit a model judge for verbosity bias?
4. What belongs in an atomic rollback for an aligned assistant?
5. Explain why preference optimization can reduce capability even as reward rises.

