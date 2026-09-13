# Part VIII - Technical Leadership and Engineering Judgment

A new embedding model is proposed for the documentation assistant. Aggregate retrieval improves, but one small tenant loses relevant evidence after permission filtering. The retrieval team wants to ship; the service owner is responsible for that tenant's failures. Neither a better kernel nor another average score settles the decision.

This part examines who can stop the rollout, which experiment should come next, and how the decision survives changes in team membership. Its fictional cases return to the assistant's index migration, latency budget, and permission incident. Communication, operating reviews, and incident roles matter because they determine whether the technical safeguards developed earlier are actually used.

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

"We do not know" becomes actionable when followed by a concrete plan. For example: "The uncertainty is draft acceptance on code traffic. We will replay 100,000 representative prompts over 48 hours, report uncertainty by workload slice, and revisit the architecture if acceptance falls below our modeled break-even point of 62 percent." These are hypothetical planning values. A sample count alone does not guarantee a narrow interval: repeated users, correlated prompts, and rare slices change the effective sample size.

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

### Match the artifact to the decision

Technical organizations often use one document shape for every problem. That creates either excessive ceremony or decisions with no durable record. Match the artifact to consequence and reversibility:

| Situation | Minimum useful artifact | Required evidence |
| --- | --- | --- |
| Reversible local implementation | Pull request or short design note | Tests, benchmark if performance-sensitive, rollback path |
| Shared interface or data contract | Request for comments | Consumers, compatibility plan, ownership, migration |
| Expensive architecture choice | Decision record plus prototype | Workload model, alternatives, measured unknowns |
| Reliability or security exposure | Risk acceptance or remediation plan | Impact, likelihood, controls, accountable approver |
| Cross-team investment | Decision memo and roadmap | Outcome, economics, dependencies, staffing, checkpoints |

The artifact is part of the control system. It should say who decides, who must be consulted, when comments close, and what would reopen the choice. Documents without decision rights become discussion archives. Decisions without context force each new participant to reconstruct the argument.

### Build a decision ledger

Keep a lightweight ledger for consequential choices: date, owner, context, options, decision, assumptions, dissent, review trigger, and superseding decision. The purpose is not to prove that an old choice was correct. It is to preserve the conditions under which it was reasonable.

A useful review trigger is observable: traffic exceeds a threshold, a dependency misses an SLO, model size crosses a memory boundary, or migration cost changes materially. “Review later” is not a trigger. When conditions change, append or supersede the record rather than rewriting history.

Decision quality can be reviewed independently from outcome quality. A sound choice can lose because uncertainty resolves badly; a careless choice can win by luck. Post-decision review should ask whether the team framed the objective, considered credible alternatives, calibrated uncertainty, and installed a feedback loop.

### Design Exercises

1. Turn a complex inference optimization into a one-page executive decision.
2. Describe a disagreement where both sides had valid local incentives.
3. How do you distinguish missing data from incompatible risk tolerance?
4. What mechanism makes a cross-org technical standard durable?
5. When should a technical leader escalate rather than seek more consensus?

## Operating Mechanisms for ML Systems

LEAD: A mechanism is a recurring process with an owner, input, decision rule, and observable output. Meetings and dashboards are ingredients; they become mechanisms only when they reliably change action.

### Turn goals into control loops

“Improve inference efficiency” is not executable. A control loop connects a target to decisions:

1. define the outcome and guardrails;
2. measure it by workload slice;
3. identify the owner who can change the system;
4. set thresholds that trigger investigation or action;
5. record the intervention and expected effect;
6. verify the effect and update the model.

For serving efficiency, the outcome might be cost per successful million tokens under latency and quality constraints. Inputs include batch occupancy, memory pressure, acceptance rate, power, and queueing. The weekly mechanism should result in a capacity allocation, experiment, rollback, or explicit no-action decision—not merely a tour of charts.

Metrics need counterweights. Optimizing GPU utilization alone can increase queueing and tail latency. Optimizing mean tokens per second can starve small tenants or long-context requests. Pair resource efficiency with user latency, quality, reliability, and fairness across the workload classes that matter.

### Define ownership at interfaces

Cross-team failures often expose gaps between nominal owners. A model team owns a checkpoint, a serving team owns an engine, and a product team owns traffic; without an explicit release owner, nobody may own whether that exact combination is safe to release.

For each critical interface, define:

- the producer and consumer;
- versioning and compatibility guarantees;
- validation performed on each side;
- the SLO and escalation path;
- who can stop a rollout;
- who owns migration and deprecation;
- the evidence required to declare completion.

Avoid responsibility matrices that assign many “accountable” people. One role owns the decision, even when several teams own execution. Ownership must include authority and resources; assigning accountability without either is organizational fiction.

### Govern a changing model-and-agent stack

A modern release may change the backbone architecture, low-precision kernels, retrieval index, tool permissions, and agent prompt at once. Those changes have different owners and different evidence. Keep a release manifest that binds the compatible versions; avoid making a model-name change stand in for the entire system identity.

| Proposed change | Minimum decision evidence | Accountable boundary |
| --- | --- | --- |
| Hybrid or compressed-state model | Slice-level quality, cache/state accounting, rollback compatibility | Model release owner with serving signoff |
| FP4 or a new attention kernel | Numerical and convergence checks where applicable; end-to-end latency | Kernel/precision owner, not benchmark author alone |
| Asynchronous RL | Quality versus time/cost, stale-sample and verifier diagnostics | Training owner with rollout-platform support |
| New agent tool | Authorized effects, failure reconciliation, adversarial task tests | Product capability owner and security review |
| New retrieval representation | Filtered recall, lineage, deletion/revocation behavior | Data/retrieval owner |

Consider a hypothetical attention replacement that makes a kernel twice as fast. That kernel previously consumed 20 percent of request service time. With all else unchanged, total time becomes `0.8 + 0.2/2 = 0.9` of baseline: a 10 percent reduction, not 50 percent. If layout conversion adds 12 percent of the old total, the replacement loses overall. The decision memo should contain this resource model before the team schedules a large migration.

For an agent improvement, use verified tasks per budget as the outcome. A stricter tool policy can reduce apparent completion while preventing unauthorized actions; a larger model can reduce retries enough to lower cost per success. Separate utility, policy violations, latency, and cost so the decision does not reward unsafe completion or blanket refusal. Define who can stop the rollout when one of these boundaries fails.

The release owner should be able to answer: which immutable baseline did we beat, under whose workload, what changed besides the named technique, which failures remain, and what exactly can be rolled back? That is the leadership counterpart of the book's technical invariants.

### Review architecture without becoming a gatekeeper

Architecture review should improve local decisions and propagate reusable knowledge. It should not route every design through the most senior engineer. Use tiers based on blast radius, reversibility, novelty, and shared dependencies.

Review questions should focus on invariants and evidence: What workload is modeled? Which failure domain is introduced? What state cannot be reconstructed? Which consumer bears migration cost? What observation would show that the design is wrong? Require deeper review for irreversible data formats, security boundaries, fleet-wide control planes, and dependencies that many teams cannot independently replace.

Publish principles, examples, and reference implementations so common choices become self-service. Sample completed designs to test whether the process catches real risk. A review process that approves everything is theater; one that blocks unusual work without timely alternatives drives decisions underground.

:::callout insight|Mechanisms should reduce dependence on their author
The strongest mechanism makes good action easier for people who were not in the original discussion. If it works only while one leader attends every meeting, it is an ongoing intervention, not institutional capability.
:::

### Design Exercises

1. Design an operating mechanism for model-release safety.
2. Which paired metrics prevent a serving team from optimizing utilization at the expense of users?
3. How would you tier architecture reviews by risk?
4. What makes interface ownership testable rather than rhetorical?
5. When should a decision record be reopened?

## Incident Leadership and High-Risk Change

LEAD: During an incident, leadership creates a stable decision process under incomplete information. Afterward, it converts the failure into better system structure rather than a longer list of warnings.

### Establish command and information flow

Separate incident command, technical investigation, operations, and communication when scale permits. The incident commander maintains priorities and the decision log; technical leads form and test hypotheses; operations executes mitigations with verification; communication gives affected users and leaders a consistent view.

Start with user impact, scope, and trajectory. Freeze unrelated changes. Name the current mitigation, its owner, expected observation, and rollback condition. Time-box hypotheses. A busy channel with many unowned suggestions is not parallel investigation.

Maintain two clocks:

- the **mitigation clock** asks how to reduce current harm safely;
- the **diagnosis clock** asks what mechanism created the failure.

Do not block a safe rollback while seeking explanatory completeness. Conversely, do not declare the incident understood merely because traffic recovered. Preserve logs, traces, configurations, model and data versions, scheduler state, and the exact timeline before ephemeral evidence disappears.

### Reason under partial evidence

Use an explicit hypothesis table:

| Hypothesis | Supporting evidence | Contradicting evidence | Next discriminating check | Owner |
| --- | --- | --- | --- | --- |
| KV exhaustion | Allocation failures rise with context | Some short requests also fail | Replay by context bucket | Serving lead |
| Bad model release | Errors begin after canary expansion | Old replicas show smaller spike | Route matched traffic to old version | Release lead |
| Network partition | Collective timeouts cluster by rack | Health probes remain green | Inspect fabric counters and topology | Infrastructure lead |

Prefer checks that separate hypotheses over checks that merely gather more telemetry. State confidence and update it. If a mitigation increases irreversible risk—such as accepting corrupted checkpoints to restore capacity—require an explicit decision owner and containment plan.

### Write a useful incident review

The review should explain the causal chain across technical and organizational layers. “Human error” is a stopping point, not a cause. Ask why the action was easy to take, hard to detect, or difficult to reverse.

Good corrective actions change one of five surfaces: prevention, detection, containment, recovery, or organizational learning. Every action needs an owner, priority, completion test, and link to the risk it reduces. Prefer a few high-leverage changes over dozens of low-confidence tasks.

Verify closure. A runbook is not complete until someone unfamiliar with the incident can execute it. An alert is not complete until replay proves it fires with useful context. A rollback is not complete until its time and state-loss behavior are measured.

### Lead migrations as products

:::diagram migration_gates|Each stage increases commitment and needs evidence from the affected workloads. The old path is retired only after exit criteria are met; rollback must account for the state already changed during adoption.

A migration can be implemented correctly and still fail at adoption. Treat it as a product with users, economics, compatibility, support, and an end state.

Segment adopters by complexity and value. Start with workloads that exercise the important path without requiring every exception. Provide an automated inventory, compatibility test, cost comparison, migration tooling, and staffed escalation path. Publish known gaps rather than allowing each team to rediscover them.

Track realized adoption, not announcements: eligible traffic moved, legacy capacity retired, defects by cohort, support cost, and user outcomes. Define the deprecation authority and exception process early. An indefinite dual stack doubles operational surface and drains investment from the destination.

Use progressive commitments:

1. **shadow:** observe compatibility without serving results;
2. **opt-in canary:** prove the path with motivated users;
3. **default for new workloads:** stop creating more legacy debt;
4. **cohort migration:** move bounded groups with rollback;
5. **deprecation:** remove the old path after verified exit criteria.

:::callout decision|Rollback is a designed capability
A rollback claim must identify which state moves backward, which data remains forward-only, how compatibility is preserved, how long reversal takes, and who may initiate it. “We can redeploy the old version” answers only one of those questions.
:::

### Design Exercises

1. What roles and information flows would you establish during a fleet-wide inference failure?
2. How do mitigation and diagnosis proceed without blocking each other?
3. Turn “operator error” into a causal investigation.
4. Design completion tests for three incident actions.
5. What metrics show that a platform migration has produced realized leverage?

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

Meet customers and partners. Read incident reviews, decisions, roadmaps, cost reports, and reliability metrics. Trace one request and one model release; map decision rights; distinguish loud pain from economically important pain. Deliver a useful artifact such as a capacity model or risk register without pretending to have the full strategy.

### Days 31-60: align and test

Share a diagnosis and candidate principles. Form a working group around a decision, run experiments that retire major uncertainty, and draft the target architecture, ownership, migration, and metrics. Choose an early project that reveals integration cost without depending on exceptional manual effort.

### Days 61-90: commit and start

Publish the decision and roadmap. Secure owners and capacity, define interfaces and rollout gates, start the pilot, and report progress transparently.

### Prioritization

Score opportunities by impact, risk reduction, leverage, effort, dependency, and reversibility. Use scoring to expose assumptions, not replace judgment. Quantify adoption and duplicated work removed: a platform with no migration path has zero realized leverage.

:::callout decision|Strategy includes a sequence
Order work so early steps generate information and capability for later steps. A list of desirable end states is a roadmap only after dependencies and decision gates are explicit.
:::

### Design Exercises

1. Give a measurable vision for ML inference infrastructure.
2. What would you do in the first 90 days at a new organization?
3. How do you prioritize reliability work against model capability?
4. When is standardization premature?
5. How do you measure the leverage of a platform investment?

## Leadership Evidence and Reflective Practice

LEAD: A leadership story demonstrates scope, judgment, and durable influence. It explains the technical mechanism and the organizational mechanism with equal precision.

### The SCORE structure

One optional outline is **SCORE**. Use it to recover the evidence for a real decision, not to turn every experience into the same five-paragraph success story:

- **Situation:** the user and system context, scale, and stakes.
- **Constraint:** the central technical and organizational tension.
- **Options:** credible alternatives and the evidence that separated them.
- **Response:** your decisions, influence, mechanisms, and execution.
- **Effect:** measurable outcome, second-order impact, and what you learned.

The account should make clear what *you* did without erasing the team. Use "I" for decisions and actions you owned, "we" for collective execution, and name partners' contributions. Use real evidence when describing your work; if an outcome was not measured, explain the observable change and the limit of the evidence instead of inventing a metric.

### Customer obsession

Weak story: you optimized latency because a dashboard was red. Strong story: you traced how latency disrupted a customer workflow, discovered that the aggregate metric hid long-context users, changed the SLO and scheduler, and created a recurring customer-to-capacity review.

Include a hard tradeoff. Customer obsession is not saying yes to every request; it is understanding the underlying outcome and choosing the most durable response.

### Growth mindset

Choose a failure or changed belief with consequence. Explain the evidence that contradicted you, how you created safety while revising course, and which mechanism changed afterward. "I learned to communicate more" is too generic.

Example: a custom kernel project missed launch because the team optimized a microbenchmark rather than engine replay. You stopped the rollout, rebuilt the benchmark contract, introduced shape-distribution review and numerical gates, and later achieved a smaller but real end-to-end gain.

### Diversity and inclusion

Use engineering mechanisms, not slogans. Examples include changing design-review participation, reducing timezone bias in decisions, auditing data or evaluation slices, improving hiring calibration, sponsoring a colleague into visible ownership, or designing accessibility into a developer tool.

Name the observed barrier, your intervention, the voices included, and measured outcome. Avoid telling another person's private story or positioning yourself as the sole rescuer.

### Ambiguity

Ambiguity stories should show how you created clarity without inventing certainty. State what was unknown, how you bounded it, which decisions were reversible, and what you deliberately postponed.

At broad organizational scope, ambiguity often spans organizations: no shared metric, unclear ownership, or incentives that reward local optimization. The story should show the mechanism that aligned those surfaces.

### Hiring and mentorship

Mentorship is not only advice. It creates increasing ownership. Describe how you diagnosed a growth edge, set a stretch assignment with safety, provided feedback, opened stakeholder access, and stepped back. Hiring stories should include role definition, calibrated signal, closing, and the team's capability after hire.

### An evidence notebook

Keep a small set of decision records for reflection, mentorship, and interviews. Record the original uncertainty and contrary evidence while they are still available; a polished retrospective can otherwise make an ambiguous choice look inevitable.

| Story | Primary signal | Useful alternate prompts |
| --- | --- | --- |
| Cross-org platform migration | Influence, vision | Conflict, customer, prioritization |
| Failed optimization and recovery | Learning, judgment | Risk, quality, delivery |
| Severe production incident | Ownership, calm execution | Ambiguity, communication |
| Architecture disagreement | Technical depth, alignment | Backbone, tradeoffs |
| Talent growth | Mentorship | Delegation, inclusion |
| Product-quality tradeoff | Customer judgment | Data, metrics, ethics |

:::callout insight|End with the mechanism that remained
Mention the interface, metric, review, tool, ownership model, or talent growth that continued after the immediate result. Explain who maintained it and what evidence showed it still worked after you stepped back.
:::

### Design Exercises

1. Tell a customer-obsession story with a non-obvious tradeoff.
2. Describe a technical belief you changed after contradictory evidence.
3. Show inclusion through an engineering or decision mechanism.
4. Explain a cross-org disagreement without making the other side irrational.
5. Demonstrate mentorship by the ownership the other person gained.

### Leadership exercise criteria

Apply this rubric to every exercise in Part VIII. A strong answer names the decision and accountable owner; distinguishes evidence from uncertainty; presents a credible alternative or dissenting view; installs a mechanism with a cadence or trigger; and defines an observable reversal, escalation, or completion boundary. Prefer a smaller claim supported by durable evidence over a sweeping story whose outcome cannot be verified.

### Three decision cases

The following cases are fictional teaching examples, not claims about the author's employment or measured project outcomes. They serve as model answers for migration, prioritization, and incident decisions: each uses a concrete choice to show what a leadership mechanism changes.

#### A migration that passes the average and fails a customer

A team proposes a new embedding model for the documentation assistant in Part VI. In this scenario, aggregate evidence recall rises from 88 to 92 percent, but recall for a small, heavily filtered tenant falls from 84 to 69 percent. These are invented scenario values. The new index also uses the same vector dimension as the old one, which makes an accidental mixed-version query look superficially valid.

The decision is to delay tenant-wide cutover, not to reject the model permanently. The retrieval owner creates versioned query/index pairs; the access-control owner checks eligibility at both retrieval and context assembly; the release owner adds a per-tenant regression gate. Keep the old pair routable during shadow traffic and canary deployment. The next experiment varies filtered candidate budgets on the failing slice while measuring latency.

The contribution is the release boundary: a visible owner, a customer-level guardrail, and a rollback unit. “The average improved” is insufficient evidence for the affected tenant. Conversely, demanding that every query improve would make change impossible; agree on material slice-level regressions and hard security gates before seeing results.

#### A faster kernel that does not change the roadmap

A kernel experiment is faster in isolation, but trace replay shows that the service still misses its latency target during long-prompt bursts. Profiling identifies queueing ahead of prefill as the dominant tail. The kernel result can be valid while its priority is wrong.

Keep the optimization and its reproducible benchmark, but move the next experiment to admission and prefill scheduling. Assign one engineer to reproduce the end-to-end trace with a fixed arrival process, and another owner to define the acceptable rejection policy with the product team. Compare SLO-constrained goodput, not just completed tokens per second.

The decision record should say which hypothesis failed: “kernel time dominates user-visible tail latency” was not supported by this workload. It should not say that the engineer's work failed. Preserve the useful code, the shapes where it wins, and the evidence that changed prioritization. This makes a negative result reusable instead of encouraging the team to hide it.

#### An incident where speed and confidentiality disagree

After a permissions rollout, the documentation assistant begins returning cached answers under an outdated access scope. Disabling the new permission service would restore response latency but could continue disclosure. The incident commander separates the objectives: stop unauthorized answers first, then recover availability inside the access boundary.

Disable affected answer-cache reads and fail closed for protected content whose authorization cannot be established. Preserve scoped diagnostic evidence without copying confidential answers into a broad incident channel. One owner validates revocation and cache invalidation, another estimates the availability impact, and a communications owner gives users a concrete reduced-service status.

Recovery requires replaying the failing access cases against the canonical authority, not merely observing fewer errors after rejection. The corrective action is a named-owner, authorization-aware cache contract with revocation tests; “be more careful” is not verifiable.

## Cross-Layer Design Prompt Bank

LEAD: The following compact prompts connect the major technical layers of the book. The first ten include diagnostic questions; the final five leave those questions to the reader. Expand every decision path through assumptions, equations, alternatives, failure modes, and measurement.

### 1. Design a draft model for speculative decoding

**Decision path:** Define target workload and exactness. Match tokenizer. Choose draft architecture from memory and latency budget. Train on production prompts with target logits or verified continuations, emphasizing rejection positions. Jointly tune draft size and proposal length. Evaluate acceptance per microsecond, committed tokens per cycle, target batch capacity, and tail latency. Establish target-distribution preservation from the verification algorithm and test its implementation on tractable distributions; finite task evaluations alone cannot prove exactness. Roll out by traffic slice with automatic fallback to ordinary decode.

**Questions to resolve:** Why not only teacher outputs? What changes at high temperature? When does a larger draft win? How do you handle domain shift? Can K/V state be shared?

### 2. Explain forward versus reverse KL

**Decision path:** Write both expectations. Forward KL weights error under the teacher and strongly penalizes missing teacher-supported modes. Reverse KL weights under the student and can select a mode when capacity is limited. For token distillation, minimizing teacher cross entropy equals forward KL up to teacher entropy. Temperature reveals non-argmax structure; sequence-level behavior still requires task evaluation.

**Questions to resolve:** What if teacher assigns zero probability? Why `T^2`? When could reverse KL be useful? How does limited student capacity change behavior?

### 3. Generate training data for reasoning or code

**Decision path:** Start from a capability taxonomy and seed distribution. Generate diverse candidates. Verify with independent executable, symbolic, retrieval, or expert checks. Score difficulty and novelty. Deduplicate against train and eval. Balance slices and preserve provenance. Train pilots, analyze failures, and feed verified failures back into generation. Protect private holdouts.

**Questions to resolve:** How much synthetic data? How do you prevent model collapse? What if the verifier is gameable? How do you measure coverage?

### 4. Evaluate a new data-quality filter

**Decision path:** Characterize what the filter removes by source and slice. Measure label precision and bias on adjudicated samples. Train controlled pilots with equal token budgets and matched schedules. Compare downstream capability, safety, memorization, and rare-slice floors. Evaluate whether gains come from quality or distribution change. Run counterfactual weight-versus-filter experiments.

**Questions to resolve:** What if validation loss improves but target tasks regress? How do you test contamination? How do you handle dialect or domain bias?

### 5. Prefill versus decode

**Decision path:** Prefill has many prompt positions, large GEMMs, and creates KV; it is often compute- or attention-IO-bound and controls time to first token. Decode adds one token per sequence, repeatedly reads weights and KV, and is commonly bandwidth- or launch-bound; it controls inter-token latency. Use roofline and byte estimates, then choose phase-specific scheduling and kernels.

**Questions to resolve:** What changes with batch? Why does GQA help decode? When should phases be disaggregated? What is chunked prefill?

### 6. How many draft tokens should be proposed?

**Decision path:** Expected accepted prefix grows with products of conditional acceptance, while draft and verification cost grow with proposal length. Benchmark committed tokens divided by cycle time, including memory and scheduling. Tune by batch, temperature, domain, and request class. Use a conservative adaptive policy if workload variation justifies it.

**Questions to resolve:** Can confidence predict acceptance? Why can a longer proposal hurt even if verification is parallel? What happens under continuous batching?

### 7. Improve decode efficiency

**Decision path:** Quantify weight and KV bytes and observed bandwidth. Increase safe batching, reduce bytes with quantization or GQA, improve KV allocation, fuse launch-bound operations, use graphs, cache prefixes, and consider speculation. Evaluate tail latency and capacity under production shapes. Do not optimize FLOPs if bytes are controlling.

**Questions to resolve:** Why can 4-bit be slower? When does batching violate the product? Which fusion is safe? How do you profile collectives?

### 8. Parallelize matrix multiplication

**Decision path:** Partition output tiles across blocks. Cooperatively load A and B tiles into shared memory with coalesced accesses. Accumulate register fragments, use tensor-core instructions when layout and precision allow, pipeline async copies, and handle edge tiles uniformly. Choose tile size from reuse versus registers, shared memory, and occupancy. Compare achieved compute and bandwidth ceilings.

**Questions to resolve:** Split-K? Bank conflicts? Register spilling? Persistent kernels? Why use a library?

### 9. Design continuous batching

**Decision path:** Maintain per-sequence state and admit/remove work at iteration boundaries. Protect decode deadlines, chunk long prefills, reserve KV pages, group compatible graph shapes, and enforce tenant fairness. Admission uses estimated tokens and state, not request count. Track queue, time to first token, inter-token gaps, utilization, and rejection.

**Questions to resolve:** Starvation? Cancellation? Prefix caching? Speculative sequences that advance unevenly? Overload policy?

### 10. Walk through a recent ML systems project

**Decision path:** State user outcome and scale. Explain the previous system and quantified bottleneck. Present alternatives and why your team chose one. Walk the critical technical mechanism. Show your personal decisions and cross-org influence. Report quality, performance, cost, and reliability outcomes. Close with what changed in the platform or organization and what you would do differently.

**Questions to resolve:** Hardest disagreement? Largest unknown? How did you test? What failed? How would the design change at ten times scale?

### 11. Design a distributed inference service

**Decision path:** Clarify model, traffic, SLO, context, availability, and cost. Estimate weights, KV, prefill, decode, and network. Choose parallel group inside fast topology, replicate for throughput and failure isolation, route by work and cache affinity, schedule continuously, and define admission. Add versioned control plane, observability, canary, rollback, and stateful failover semantics.

### 12. Design model evaluation

**Decision path:** Build a versioned matrix of tasks and critical slices. Separate capability, system, guardrail, and diagnostic metrics. Protect holdouts and provenance. Run unit, offline, replay, shadow, canary, and online stages. Predefine meaningful effect and hard guardrails. Localize regressions by data, model, engine, and policy version.

### 13. Handle distribution shift

**Decision path:** Define source versus target distribution and whether shift affects covariates, labels, or user behavior. Detect through input, representation, calibration, and outcome monitors. Confirm with labeled samples. Mitigate through reweighting, data acquisition, robust objectives, routing, fallback, or retraining. Protect critical slices and avoid self-reinforcing feedback.

### 14. Explain sampling

**Decision path:** Temperature rescales logits. Top-k truncates by count; top-p truncates to a cumulative mass. Greedy takes argmax. Sampling policy changes diversity, calibration, acceptance, and evaluation. Implement stable softmax, deterministic seeded streams where required, distributed top-k merge, and exact speculative correction.

### 15. Customer-centered engineering at organizational scope

**Decision path:** Identify the workflow and consequence behind a request. Segment the metric to reveal affected users. Make a tradeoff that improves the durable outcome, not merely the loud request. Align product and infrastructure owners. Ship with feedback and leave a mechanism that keeps customer evidence in planning.

:::callout decision|Expand the decision, not the vocabulary
For each decision path, write the governing assumptions, derive the controlling resource model, compare at least one credible alternative, and define the experiment that would change the choice.
:::

### Design review closing checklist

- Did I state the workload and objective?
- Did I estimate the controlling bytes, FLOPs, state, or queue?
- Did I name a credible alternative?
- Did I cover correctness and failure, not only speed?
- Did I define offline, online, and guardrail metrics?
- Did I explain rollout and rollback?
- Did I make my own decision and remaining uncertainty clear?

Return to the documentation assistant: can another team explain its latency budget, update its index without mixing versions, and stop an unauthorized answer? The capstone in Part IX asks you to assemble those decisions into one design and test it by changing the workload or introducing a failure.
