# Part VII - Technical Leadership and the Interview Masterclass

Principal engineering is the ability to improve decisions beyond the code one person can write. The evidence is not senior-sounding vocabulary. It is durable direction, aligned execution, technical risk retired early, and systems that continue working after attention moves elsewhere.

## Executive Technical Communication

LEAD: Executive communication compresses complexity without hiding uncertainty. It gives each audience the decision, evidence, risk, and requested action at the level they can use.

### The decision memo

A useful one-page memo contains:

1. **Decision:** one sentence, including scope and timing.
2. **Why now:** user, reliability, capacity, or strategic trigger.
3. **Evidence:** the few facts that separate options.
4. **Options:** credible alternatives, including doing nothing.
5. **Recommendation:** choice and assumptions.
6. **Risks:** failure modes, mitigations, and reversible boundaries.
7. **Plan:** owner, milestone, metric, and next decision date.

The memo is not a transcript of analysis. Put supporting benchmarks and architecture detail in an appendix. If the recommendation cannot be stated before the history, the problem may not be framed.

### Communicating uncertainty

Separate **known**, **estimated**, **assumed**, and **unknown**. Attach ranges where they change the choice. Name the observation that would invalidate the recommendation.

"We do not know" becomes actionable when followed by: "The uncertainty is draft acceptance on code traffic; a 48-hour replay of 100,000 anonymized prompts can bound it to within two points, and the architecture decision flips below 62 percent."

### Technical deep dive structure

For an engineering audience:

- state the workload and invariant;
- show the resource model;
- explain the baseline and bottleneck evidence;
- walk the proposed data and control paths;
- compare alternatives;
- demonstrate correctness and performance;
- cover rollout, observability, and ownership.

For executives, lead with outcome, exposure, investment, and decision. Preserve the ability to drill down but do not force everyone through kernel details.

:::diagram leadership_loop|Strong technical leadership repeats a loop: frame with evidence, align stakeholders, commit to a decision, learn from execution, and revise.

### Disagreement

Disagreement is valuable before commitment. Identify whether the conflict is about goals, facts, models, risk tolerance, ownership, or incentives. Many arguments persist because participants debate different layers.

Write the shared objective. List assumptions. Seek the smallest experiment that differentiates predictions. Define who decides and by when. After the decision, support execution while preserving a record of risk and trigger conditions.

:::callout pitfall|Consensus is not the same as alignment
Consensus means everyone prefers the choice. Alignment means people understand the decision, rationale, role, and escalation path. High-stakes work often needs the latter before the former is possible.
:::

### Influence without authority

Influence grows from credible problem framing, early stakeholder inclusion, useful artifacts, and reducing the cost of adoption. Build a coalition around the user or system outcome. Give partner teams real ownership rather than presenting a completed plan that requires their labor.

Mechanisms scale influence: design-review templates, shared metrics, reference implementations, migration tooling, office hours, and decision logs. A heroic recurring meeting does not scale.

### Principal Interview Review

1. Turn a complex inference optimization into a one-page executive decision.
2. Describe a disagreement where both sides had valid local incentives.
3. How do you distinguish missing data from incompatible risk tolerance?
4. What mechanism makes a cross-org technical standard durable?
5. When should a Principal engineer escalate rather than seek more consensus?

## Strategy, Vision, and the First 90 Days

LEAD: Vision names a future state and why it matters. Strategy chooses the few coordinated actions that can move the organization there despite constraints.

### A technical vision

A credible vision connects users, system capabilities, and organizational leverage. "Build a world-class platform" is not a vision. A stronger statement is:

> Any product team can take a validated model from checkpoint to a safe regional canary in one day, with predictable cost and no engine-specific integration.

The statement implies developer experience, release identity, evaluation gates, capacity, and isolation. It is measurable and invites architectural consequences.

### Strategy diagnosis

Diagnose the constraint that prevents the vision. It might be fragmented serving engines, weak evaluation, long data iteration, unreliable checkpoints, lack of ownership, or capacity allocation. Strategy concentrates effort on the controlling constraint instead of launching one initiative per complaint.

Choose a small set of mutually reinforcing moves. For example:

- standardize model-engine release bundles;
- create a common evaluation and canary service;
- expose token-level capacity economics;
- migrate the highest-cost workloads with dedicated support.

Each move should have a leading metric, owner, and condition to stop or change course.

### The first 30 days: learn and map

Meet customers and partner teams. Read incident reviews, architecture decisions, roadmaps, cost reports, and reliability metrics. Trace one real request and one model release. Map decision rights and informal influence. Identify which pain is loud and which pain is economically important.

Deliver a small useful artifact - perhaps a capacity model, risk register, or unified metric definition - without pretending to have the full strategy.

### Days 31-60: align and test

Share a diagnosis and candidate principles. Form a working group around a concrete decision, not a generic forum. Run one or two experiments that retire major uncertainty. Draft target architecture, ownership, migration path, and success metrics.

Choose an early project that proves the mechanism and reveals integration cost. Avoid a showcase that succeeds only through exceptional manual effort.

### Days 61-90: commit and start

Publish a decision memo and roadmap. Secure owners and capacity. Define interfaces, review cadence, rollout gates, and a deprecation plan for legacy paths. Start the pilot and establish transparent reporting.

### Prioritization

Score opportunities by user or business impact, risk reduction, strategic leverage, effort, dependency, and reversibility. Do not reduce the decision to one formula; use scoring to expose assumptions.

High-leverage infrastructure often has indirect benefit. Quantify adoption and duplicated work removed. A platform with excellent local performance and no migration path has zero realized leverage.

:::callout decision|Strategy includes a sequence
Order work so early steps generate information and capability for later steps. A list of desirable end states is a roadmap only after dependencies and decision gates are explicit.
:::

### Principal Interview Review

1. Give a measurable vision for ML inference infrastructure.
2. What would you do in the first 90 days at a new organization?
3. How do you prioritize reliability work against model capability?
4. When is standardization premature?
5. How do you measure the leverage of a platform investment?

## Behavioral Stories at Principal Level

LEAD: A Principal story demonstrates scope, judgment, and durable influence. It explains the technical mechanism and the organizational mechanism with equal precision.

### The SCORE structure

Use **SCORE** rather than a mechanical chronology:

- **Situation:** the user and system context, scale, and stakes.
- **Constraint:** the central technical and organizational tension.
- **Options:** credible alternatives and the evidence that separated them.
- **Response:** your decisions, influence, mechanisms, and execution.
- **Effect:** measurable outcome, second-order impact, and what you learned.

The interviewer should know what *you* did without erasing the team. Use "I" for decisions and actions you owned, "we" for collective execution, and name partners' contributions.

### Customer obsession

Weak story: you optimized latency because a dashboard was red. Strong story: you traced how latency disrupted a customer workflow, discovered that the aggregate metric hid long-context users, changed the SLO and scheduler, and created a recurring customer-to-capacity review.

Include a hard tradeoff. Customer obsession is not saying yes to every request; it is understanding the underlying outcome and choosing the most durable response.

### Growth mindset

Choose a failure or changed belief with consequence. Explain the evidence that contradicted you, how you created safety while revising course, and which mechanism changed afterward. "I learned to communicate more" is too generic.

Example: a custom kernel project missed launch because the team optimized a microbenchmark rather than engine replay. You stopped the rollout, rebuilt the benchmark contract, introduced shape-distribution review and numerical gates, and later achieved a smaller but real end-to-end gain.

### Diversity and inclusion

Use engineering mechanisms, not slogans. Examples include changing design-review participation, reducing timezone bias in decisions, auditing data or evaluation slices, improving interview calibration, sponsoring a colleague into visible ownership, or designing accessibility into a developer tool.

Name the observed barrier, your intervention, the voices included, and measured outcome. Avoid telling another person's private story or positioning yourself as the sole rescuer.

### Ambiguity

Ambiguity stories should show how you created clarity without inventing certainty. State what was unknown, how you bounded it, which decisions were reversible, and what you deliberately postponed.

At Principal scope, ambiguity often spans organizations: no shared metric, unclear ownership, or incentives that reward local optimization. The story should show the mechanism that aligned those surfaces.

### Hiring and mentorship

Mentorship is not only advice. It creates increasing ownership. Describe how you diagnosed a growth edge, set a stretch assignment with safety, provided feedback, opened stakeholder access, and stepped back. Hiring stories should include role definition, calibrated signal, closing, and the team's capability after hire.

### Story bank

Prepare six to eight stories that can flex across prompts:

| Story | Primary signal | Useful alternate prompts |
| --- | --- | --- |
| Cross-org platform migration | Influence, vision | Conflict, customer, prioritization |
| Failed optimization and recovery | Learning, judgment | Risk, quality, delivery |
| Severe production incident | Ownership, calm execution | Ambiguity, communication |
| Architecture disagreement | Technical depth, alignment | Backbone, tradeoffs |
| Talent growth | Mentorship | Delegation, inclusion |
| Product-quality tradeoff | Customer judgment | Data, metrics, ethics |

:::callout insight|End with the mechanism that remained
Principal impact is durable. Mention the interface, metric, review, tool, ownership model, or talent growth that continued after the immediate result.
:::

### Principal Interview Review

1. Tell a customer-obsession story with a non-obvious tradeoff.
2. Describe a technical belief you changed after contradictory evidence.
3. Show inclusion through an engineering or decision mechanism.
4. Explain a cross-org disagreement without making the other side irrational.
5. Demonstrate mentorship by the ownership the other person gained.

## Recruiter-Derived Interview Masterclass

LEAD: The following drills are the recurring high-signal questions behind the recruiter hints. Each answer is a compact spine that should expand through assumptions, equations, alternatives, and measurement.

### 1. Design a draft model for speculative decoding

**Answer spine:** Define target workload and exactness. Match tokenizer. Choose draft architecture from memory and latency budget. Train on production prompts with target logits or verified continuations, emphasizing rejection positions. Jointly tune draft size and proposal length. Evaluate acceptance per microsecond, committed tokens per cycle, target batch capacity, tail latency, and exact output distribution. Roll out by traffic slice with automatic fallback to ordinary decode.

**Follow-ups:** Why not only teacher outputs? What changes at high temperature? When does a larger draft win? How do you handle domain shift? Can K/V state be shared?

### 2. Explain forward versus reverse KL

**Answer spine:** Write both expectations. Forward KL weights error under the teacher and strongly penalizes missing teacher-supported modes. Reverse KL weights under the student and can select a mode when capacity is limited. For token distillation, minimizing teacher cross entropy equals forward KL up to teacher entropy. Temperature reveals non-argmax structure; sequence-level behavior still requires task evaluation.

**Follow-ups:** What if teacher assigns zero probability? Why `T^2`? When could reverse KL be useful? How does limited student capacity change behavior?

### 3. Generate training data for reasoning or code

**Answer spine:** Start from a capability taxonomy and seed distribution. Generate diverse candidates. Verify with independent executable, symbolic, retrieval, or expert checks. Score difficulty and novelty. Deduplicate against train and eval. Balance slices and preserve provenance. Train pilots, analyze failures, and feed verified failures back into generation. Protect private holdouts.

**Follow-ups:** How much synthetic data? How do you prevent model collapse? What if the verifier is gameable? How do you measure coverage?

### 4. Evaluate a new data-quality filter

**Answer spine:** Characterize what the filter removes by source and slice. Measure label precision and bias on adjudicated samples. Train controlled pilots with equal token budgets and matched schedules. Compare downstream capability, safety, memorization, and rare-slice floors. Evaluate whether gains come from quality or distribution change. Run counterfactual weight-versus-filter experiments.

**Follow-ups:** What if validation loss improves but target tasks regress? How do you test contamination? How do you handle dialect or domain bias?

### 5. Prefill versus decode

**Answer spine:** Prefill has many prompt positions, large GEMMs, and creates KV; it is often compute- or attention-IO-bound and controls time to first token. Decode adds one token per sequence, repeatedly reads weights and KV, and is commonly bandwidth- or launch-bound; it controls inter-token latency. Use roofline and byte estimates, then choose phase-specific scheduling and kernels.

**Follow-ups:** What changes with batch? Why does GQA help decode? When should phases be disaggregated? What is chunked prefill?

### 6. How many draft tokens should be proposed?

**Answer spine:** Expected accepted prefix grows with products of conditional acceptance, while draft and verification cost grow with proposal length. Benchmark committed tokens divided by cycle time, including memory and scheduling. Tune by batch, temperature, domain, and request class. Use a conservative adaptive policy if workload variation justifies it.

**Follow-ups:** Can confidence predict acceptance? Why can a longer proposal hurt even if verification is parallel? What happens under continuous batching?

### 7. Improve decode efficiency

**Answer spine:** Quantify weight and KV bytes and observed bandwidth. Increase safe batching, reduce bytes with quantization or GQA, improve KV allocation, fuse launch-bound operations, use graphs, cache prefixes, and consider speculation. Evaluate tail latency and capacity under production shapes. Do not optimize FLOPs if bytes are controlling.

**Follow-ups:** Why can 4-bit be slower? When does batching violate the product? Which fusion is safe? How do you profile collectives?

### 8. Parallelize matrix multiplication

**Answer spine:** Partition output tiles across blocks. Cooperatively load A and B tiles into shared memory with coalesced accesses. Accumulate register fragments, use tensor-core instructions when layout and precision allow, pipeline async copies, and handle edge tiles uniformly. Choose tile size from reuse versus registers, shared memory, and occupancy. Compare achieved compute and bandwidth ceilings.

**Follow-ups:** Split-K? Bank conflicts? Register spilling? Persistent kernels? Why use a library?

### 9. Design continuous batching

**Answer spine:** Maintain per-sequence state and admit/remove work at iteration boundaries. Protect decode deadlines, chunk long prefills, reserve KV pages, group compatible graph shapes, and enforce tenant fairness. Admission uses estimated tokens and state, not request count. Track queue, time to first token, inter-token gaps, utilization, and rejection.

**Follow-ups:** Starvation? Cancellation? Prefix caching? Speculative sequences that advance unevenly? Overload policy?

### 10. Walk through a recent ML systems project

**Answer spine:** State user outcome and scale. Explain the previous system and quantified bottleneck. Present alternatives and why your team chose one. Walk the critical technical mechanism. Show your personal decisions and cross-org influence. Report quality, performance, cost, and reliability outcomes. Close with what changed in the platform or organization and what you would do differently.

**Follow-ups:** Hardest disagreement? Largest unknown? How did you test? What failed? How would the design change at ten times scale?

### 11. Design a distributed inference service

**Answer spine:** Clarify model, traffic, SLO, context, availability, and cost. Estimate weights, KV, prefill, decode, and network. Choose parallel group inside fast topology, replicate for throughput and failure isolation, route by work and cache affinity, schedule continuously, and define admission. Add versioned control plane, observability, canary, rollback, and stateful failover semantics.

### 12. Design model evaluation

**Answer spine:** Build a versioned matrix of tasks and critical slices. Separate capability, system, guardrail, and diagnostic metrics. Protect holdouts and provenance. Run unit, offline, replay, shadow, canary, and online stages. Predefine meaningful effect and hard guardrails. Localize regressions by data, model, engine, and policy version.

### 13. Handle distribution shift

**Answer spine:** Define source versus target distribution and whether shift affects covariates, labels, or user behavior. Detect through input, representation, calibration, and outcome monitors. Confirm with labeled samples. Mitigate through reweighting, data acquisition, robust objectives, routing, fallback, or retraining. Protect critical slices and avoid self-reinforcing feedback.

### 14. Explain sampling

**Answer spine:** Temperature rescales logits. Top-k truncates by count; top-p truncates to a cumulative mass. Greedy takes argmax. Sampling policy changes diversity, calibration, acceptance, and evaluation. Implement stable softmax, deterministic seeded streams where required, distributed top-k merge, and exact speculative correction.

### 15. Customer obsession at Principal scope

**Answer spine:** Identify the workflow and consequence behind a request. Segment the metric to reveal affected users. Make a tradeoff that improves the durable outcome, not merely the loud request. Align product and infrastructure owners. Ship with feedback and leave a mechanism that keeps customer evidence in planning.

:::callout decision|Practice expansion, not memorization
For each answer spine, practice a 30-second thesis, a 3-minute structured answer, and a 15-minute deep dive with derivation or code. The structure should remain stable while examples and depth adapt to the interviewer.
:::

### Whiteboard closing checklist

- Did I state the workload and objective?
- Did I estimate the controlling bytes, FLOPs, state, or queue?
- Did I name a credible alternative?
- Did I cover correctness and failure, not only speed?
- Did I define offline, online, and guardrail metrics?
- Did I explain rollout and rollback?
- Did I make my own decision and remaining uncertainty clear?

