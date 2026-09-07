---
title: "Engineering Large Language Models"
subtitle: "Training, Inference, CUDA, Distributed Systems, and Technical Leadership"
author: "Yury Kirpichev"
edition: "Working Draft - September 2026"
copyright_year: "2026"
publication_date: "September 2026"
keywords: "large language models, LLM systems, model training, inference, CUDA, distributed systems, technical leadership"
---

LEAD: A systems guide to building, operating, and evolving large language models across data, training, inference, accelerators, distributed infrastructure, and technical organizations.

Large language models are not isolated neural networks. They are production systems in which statistical behavior, numerical computation, hardware, data governance, reliability, and economics interact. This book develops the models and engineering methods needed to reason across those boundaries.

The material follows the lifecycle of an LLM system: define the workload and evidence contract; construct data and training recipes; understand inference state and scheduling; optimize accelerator execution; distribute work across machines; design production algorithms and services; examine recent research results; and establish the organizational mechanisms that keep the system operable.

:::toc

### A note on scope

This working draft covers foundations, training, inference, CUDA, distributed systems, production algorithms, system design, recent systems research, and technical leadership. Its research snapshot covers selected work through August 2026, with explicitly dated implementation checks where noted. It is not an exhaustive survey or a guarantee of production readiness. Results identified as recent are reported by their source papers and should be revalidated on the reader's models, hardware, workloads, and quality constraints.

The examples assume transformer-style models and GPU-like accelerators, but the reasoning applies more broadly. Hardware names, model families, and framework APIs will evolve. Arithmetic intensity, dependency structure, failure isolation, data provenance, and organizational incentives will not.

:::callout insight|How to use this book
Begin each part with its conceptual model, then work through the derivations, implementation examples, and production consequences. Use the design exercises to test whether you can transfer the reasoning to a new workload rather than merely recall the conclusion.
:::

### The engineering decision method

A sound technical decision usually moves through six layers:

1. **Frame the objective.** State the user outcome, workload, SLO, quality bar, and constraints before naming a technique.
2. **Build a quantitative model.** Estimate bytes, FLOPs, memory, communication, queueing, or sample complexity at the right level of fidelity.
3. **Choose the bottleneck.** Separate symptoms from the resource or coordination limit that actually controls performance.
4. **Compare alternatives.** Explain why a reasonable competing design loses under the stated conditions.
5. **Design the measurement loop.** Define offline, online, guardrail, and operational metrics with a rollback boundary.
6. **Name the next uncertainty.** Make the remaining risk visible and propose the cheapest experiment that can retire it.

### Editorial conventions

Every code block identifies whether it is an illustrative excerpt or pseudocode. The runnable CPU references live in the companion repository's `examples/` directory, with commands and expected results in `examples/README.md` and tests in `tests/test_examples.py`. They cover attention and partition merging, streaming summaries, and a small retrieval/evaluation fixture. `make test` checks these examples as well as the book builder. CUDA excerpts are not compiled or performance-validated by the CPU test suite.

- Color-coded boxes separate a compact **Engineering Insight**, an attractive but incomplete **Common Pitfall**, and a governing **Engineering Decision**.
- Display equations introduce important relationships with named variables and explicit boundaries. Code begins with the smallest correct mechanism, then expands into production concerns. Technical chapters close with design exercises and worked solutions or answer criteria.
- A `>` in a code panel's left gutter marks a visual continuation of a long source line; it is not part of the code. The editable Markdown and runnable example files remain the source of truth.

:::pagebreak

### The one-page engineering decision loop

| Phase | Engineering action | Result |
| --- | --- | --- |
| Frame | Define workload, scale, latency, quality, cost, and failure tolerance | A testable system contract |
| Model | Estimate dominant bytes, FLOPs, state, communication, and queueing | A predicted bottleneck |
| Propose | Build a baseline architecture with explicit ownership | An executable starting point |
| Stress | Examine hot keys, long tails, partial failure, and distribution shift | Known failure boundaries |
| Measure | Define outcomes, guardrails, observability, and rollback | Evidence that can change the decision |
| Evolve | Compare alternatives and name the condition that changes the choice | A system that can adapt |

:::callout decision|The standard of evidence
Use an equation when it changes the decision, a benchmark when implementation details dominate, and an experiment when uncertainty is cheaper to remove than to debate. Do not use precision theater to hide an unknown workload.
:::

### Reader contract

Do not memorize the final recommendation without the assumptions that produced it. A technique is rarely "best" in isolation. It is best for a workload, budget, failure model, and organization. If any of those move, reopen the decision.
