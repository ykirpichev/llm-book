# Executable teaching references

Run from the repository root with Python 3.12 or newer. These examples use only
the standard library and do not download models, call APIs, or require a GPU.

| Example | Run | Expected result | Verification |
| --- | --- | --- | --- |
| Attention and context-shard merge | `python -m examples.attention` | `[3.0]` | Independent dense formula; masks; irregular partitions; absolute tolerance 1e-12 on ordinary float64 fixtures. |
| Exact streaming top-k | `python -m examples.streaming` | `[(3.0, 'c'), (3.0, 'b')]` | Full-sort oracle and merge of disjoint shards; deterministic ties. |
| Welford moments | `Moments` in `streaming.py` | Tested against `statistics.variance` | Empty-state merge; large offset; absolute tolerance 1e-6 on the declared fixture. |
| Retrieval and evaluation | `python -m examples.rag` | Unsafe baseline: 2/5 correct, one ACL leak; guarded fixture: 5/5 correct, zero ACL leaks. | Stale revision, permission filter, absent evidence, conflicting evidence, deletion. |

Run `make test` for all tests. The RAG experiment is a five-case synthetic
fixture with labeled facts and a deterministic answer function. It is not a
measurement of an LLM, embedding model, ANN index, production accuracy, or
adversarial robustness. It teaches failure isolation and regression contracts.

The attention reference is a single-head semantic calculation, not optimized
code. The top-k example assumes immutable records with unique stable IDs; it
rejects non-finite scores. Some manuscript excerpts illustrate a different
declared policy (such as skipping invalid scores). Policies must not be mixed
silently when comparing implementations.

## GPU examples

CUDA snippets in the manuscript are explanatory excerpts, not validated
standalone programs. A future GPU benchmark must name the device, compiler,
dtype, shape, layout, warmup, timed region, synchronization, allocation policy,
reference tolerances, and repeated-run distribution. CPU tests do not certify
CUDA race freedom or performance. No GPU speedups are claimed by this suite.
