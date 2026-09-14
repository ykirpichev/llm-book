# Executable teaching references

Run from the repository root with Python 3.12 or newer. These examples use only
the standard library and do not download models, call APIs, or require a GPU.

| Example | Run | Expected result | Verification |
| --- | --- | --- | --- |
| Attention and context-shard merge | `python -m examples.attention` | `[3.0]` | Independent dense formula; masks; irregular partitions; absolute tolerance 1e-12 on ordinary float64 fixtures. |
| Exact streaming top-k | `python -m examples.streaming` | `[(3.0, 'c'), (3.0, 'b')]` | Full-sort oracle and merge of disjoint shards; deterministic ties. |
| Welford moments | `Moments` in `streaming.py` | Tested against `statistics.variance` | Empty-state merge; large offset; absolute tolerance 1e-6 on the declared fixture. |
| Retrieval and evaluation | `python -m examples.rag` | Unsafe baseline: 2/5 correct, one ACL leak; guarded fixture: 5/5 correct, zero ACL leaks. | Stale revision, permission filter, absent evidence, conflicting evidence, deletion. |
| Tokenization, loss, recurrent state | `python -m unittest tests.test_sequence_models` | 7 passing tests | Byte merge ordering/round trip; masked stable loss and ignored targets; selective state correction and decay. |
| Train/checkpoint/generate | `python -m examples.tiny_lm` | Loss below 0.01; `abababa` | Analytic gradients against finite differences, serialization, bounded greedy decoding. Bigram model, not a transformer. |
| Post-training arithmetic | `python -m unittest tests.test_post_training` | 6 passing tests | DPO signs/stability, group advantages, clipping, sequence ratios, pass-at-k. |
| Speculation and quantization | `python -m unittest tests.test_inference_mechanisms` | 5 passing tests | Exact output mass including support gaps; randomized distributions; scalar quantization error/outliers. |
| Agent loop | `python -m examples.agent_loop` | `finished 1 0` | Denied capabilities, retry after commit, budgets, key conflicts, revocation/removal. Scripted policy, no LLM. |
| Retrieval ranking | `python -m unittest tests.test_agent_loop.RetrievalMethodTests` | 2 passing tests | Reciprocal rank fusion and normalized token-level MaxSim; no learned encoder or ANN implementation. |
| Foundation calculations | `python -m unittest tests.test_foundations` | 11 passing tests | Loss gradient, causal mask/permutation, rank weighting, clipping, resource and cost models, scaling allocation, paired uncertainty, zero-failure bounds, coarsened KL. |
| Accelerator portability planning | `python -m unittest tests.test_accelerator_portability -v` | 7 passing tests | Higher-precision RMSNorm oracle, padding ratios, modeled physical KV partition/replication, compilation repayment. CPU arithmetic, not device emulation or hardware validation. |

Run `make test` for all tests. The RAG experiment is a five-case synthetic
fixture with labeled facts and a deterministic answer function. It is not a
measurement of an LLM, embedding model, ANN index, production accuracy, or
adversarial robustness. It teaches failure isolation and regression contracts.

The attention reference is a single-head semantic calculation, not optimized
code. Run `python -m unittest tests.test_examples.AttentionTests -v` for its
dense/partition, masked-shard, and numerical-range checks. The top-k example
assumes immutable records with unique stable IDs; it
rejects non-finite scores. Some manuscript excerpts illustrate a different
declared policy (such as skipping invalid scores). Policies must not be mixed
silently when comparing implementations.

## GPU examples

The [optional accelerator examples](accelerators/README.md) include a source-only
Triton RMSNorm kernel and an on-target forward smoke-test harness. Their packages
and hardware requirements are separate from the CPU examples above.

CUDA snippets in the manuscript are explanatory excerpts, not validated
standalone programs. A future GPU benchmark must name the device, compiler,
dtype, shape, layout, warmup, timed region, synchronization, allocation policy,
reference tolerances, and repeated-run distribution. CPU tests do not certify
CUDA race freedom or performance. No GPU speedups are claimed by this suite.
