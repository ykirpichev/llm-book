---
title: "Engineering Large Language Models"
subtitle: "Training, Inference, CUDA, Distributed Systems, and Technical Leadership"
author: "Yury Kirpichev"
edition: "Public Edition - September 2026"
copyright_year: "2026"
publication_date: "September 23, 2026"
keywords: "large language models, LLM systems, model training, inference, CUDA, distributed systems, technical leadership"
---

LEAD: A systems guide to building, operating, and evolving large language models across data, training, inference, accelerators, distributed infrastructure, and technical organizations.

Consider a documentation assistant answering a question about log retention. It retrieves an obsolete policy, waits behind a long prompt, and returns a fluent answer after the user has given up. A better language model might change the wording without fixing any of those failures. The service needs current evidence, a workable latency budget, and a way to discover which stage went wrong.

This book follows those dependencies from model training to an operating service. It explains how data and objectives shape behavior, how attention creates work and persistent state, how kernels and distributed execution pay for that work, and how retrieval, evaluation, and ownership keep the service useful. The documentation assistant is a hypothetical recurring case, developed fully in Part VI and the capstone; it is not a deployment report.

:::toc

### A note on scope

This public edition covers foundations, training, inference, CUDA, distributed systems, production algorithms, agents, multimodal systems, recent research, and technical leadership. Part VII was revised through September 23, 2026; other parts retain the September 13 research cutoff, with explicitly dated implementation checks where noted. It explains the central mechanisms in the text; primary-source links supply evidence and further detail rather than replacing the explanation. Links to moving documentation describe the implementation checked at that time, not a promise about later releases.

The book is not an exhaustive survey of every paper or a guarantee of production readiness. Recent preprints are treated as provisional evidence, and reported results must be revalidated on the reader's models, hardware, workloads, and quality constraints.

The examples develop transformer, recurrent/hybrid, and diffusion-style models, with GPU-like accelerators as the main execution setting. Hardware names, model families, and framework APIs will evolve. Arithmetic intensity, dependency structure, failure isolation, data provenance, and organizational incentives remain useful across those changes.

### Prerequisites and learning path

The book assumes basic Python, vectors and matrix multiplication, probability distributions, logarithms, and derivatives. It develops the LLM-specific uses of those ideas from token targets and optimization through architecture, training, serving, and applications. Read Parts I–III in order on a first pass. Parts IV–V explain how the same computations map to accelerators and clusters. Part VI has two complementary tracks: streaming-state primitives that lead to telemetry and control loops, and production ML design that leads to retrieval and bounded agents. Readers focused on applications may begin Part VI at the RAG chapter and return to the streaming track when they need its state and failure models. Part VII extends the mechanisms to current research and multimodal generation. Part VIII turns technical evidence into operating decisions.

Examples operate at four deliberate levels. Local examples isolate one mechanism. A recurring 7B configuration connects performance arithmetic across parts. The documentation assistant instantiates retrieval, authorization, evaluation, and agent design in one product. The executable bigram fixture in Part IX exposes the complete training-to-generation lifecycle without pretending to reproduce frontier capability.

Use the **End-to-End Learning Lab and Capstone** in Part IX for the runnable lifecycle, an ordered path through the tested references, and the complete documentation-service design exercise.

:::callout insight|How to use this book
Begin each part with its conceptual model, then work through the derivations, implementation examples, and production consequences. Use the design exercises to test whether you can transfer the reasoning to a new workload rather than merely recall the conclusion.
:::

### Editorial conventions

Every code block identifies whether it is runnable, an illustrative excerpt, or pseudocode. The runnable CPU references live in the companion repository's `examples/` directory, with commands and expected results in `examples/README.md` and tests in `tests/`. They cover tokenization/loss, a trainable bigram model, recurrent state, post-training arithmetic, speculative sampling, quantization, attention, streaming summaries, retrieval, and a bounded agent loop. `make test` checks these examples as well as the book builder. CUDA excerpts are not compiled or performance-validated by the CPU test suite; the agent fixtures are deterministic program tests, not measurements of an LLM's autonomous performance.

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
