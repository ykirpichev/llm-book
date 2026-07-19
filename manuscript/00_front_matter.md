---
title: "The Principal ML Systems Handbook"
subtitle: "Training, Inference, CUDA, Distributed Systems, and Technical Leadership"
author: "Yury Kirpichev"
edition: "First Edition - 2026"
---

LEAD: A field guide for senior engineers who must connect model behavior, accelerator performance, distributed architecture, and organizational judgment into one coherent system.

This handbook is built around a demanding premise: **the interview is not the subject**. The subject is the ability to design, explain, debug, and lead machine learning systems under real constraints. Interview readiness follows from that mastery.

The material is organized around the questions that repeatedly surface in Staff and Principal loops: How do you design a training recipe? Why does a serving system miss its latency target? What makes one CUDA kernel faster than another? When should a team trade quality for capacity? How do you turn technical ambiguity into an executable cross-org decision?

:::callout insight|How to use this book
For a fast interview pass, read the chapter opener, the decision rules, the pitfalls, and the review drill. For durable mastery, reproduce the derivations, implement the kernels, and defend every architecture against at least two credible alternatives.
:::

### The Principal answer pattern

A strong answer usually moves through six layers:

1. **Frame the objective.** State the user outcome, workload, SLO, quality bar, and constraints before naming a technique.
2. **Build a quantitative model.** Estimate bytes, FLOPs, memory, communication, queueing, or sample complexity at the right level of fidelity.
3. **Choose the bottleneck.** Separate symptoms from the resource or coordination limit that actually controls performance.
4. **Compare alternatives.** Explain why a reasonable competing design loses under the stated conditions.
5. **Design the measurement loop.** Define offline, online, guardrail, and operational metrics with a rollback boundary.
6. **Name the next uncertainty.** A Principal answer makes the remaining risk visible and proposes the cheapest experiment that can retire it.

### Editorial conventions

- Color-coded boxes separate the compact **Interview Insight**, the attractive but incomplete **Common Pitfall**, and the governing **Principal Decision**.
- Equations use implementation-oriented notation; code stays small enough for a whiteboard, then expands into production considerations, and every technical chapter closes with a transfer-focused Principal interview review.

:::toc

### A note on scope

This first edition is a cohesive, recruiter-derived core: foundations, training, inference, CUDA, distributed systems, coding, system design, and leadership. It is intentionally opinionated. The goal is not to catalog every paper. The goal is to teach a reusable method for reaching sound decisions when details change.

The examples assume transformer-style models and GPU-like accelerators, but the reasoning applies more broadly. Hardware names, model families, and framework APIs will evolve. Arithmetic intensity, dependency structure, failure isolation, data provenance, and organizational incentives will not.

### Reader contract

Do not memorize the final recommendation without the assumptions that produced it. A technique is rarely "best" in isolation. It is best for a workload, budget, failure model, and organization. If any of those move, reopen the decision.

:::pagebreak

### The one-page interview operating system

| Phase | What to say | What the interviewer learns |
| --- | --- | --- |
| Clarify | Workload, scale, latency, quality, cost, failure tolerance | You refuse to optimize an undefined system |
| Model | Dominant bytes, FLOPs, state, communication, queueing | You reason quantitatively |
| Propose | Baseline architecture with explicit ownership | You can make a decision |
| Stress | Hot keys, long tails, partial failure, distribution shift | You expect reality |
| Measure | Success, guardrails, observability, rollback | You close the loop |
| Extend | Two alternatives and the condition that flips the choice | You understand the design space |

:::callout decision|The standard of evidence
Use an equation when it changes the decision, a benchmark when implementation details dominate, and an experiment when uncertainty is cheaper to remove than to debate. Do not use precision theater to hide an unknown workload.
:::
