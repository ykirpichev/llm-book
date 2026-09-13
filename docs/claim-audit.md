# Evidence audit: working-draft checkpoint

Checked September 7, 2026; updated September 13 after the final Astra review.
This ledger covers the highlighted recent-result numbers and the shared
running-model arithmetic. It does not certify every
sentence or reproduce GPU experiments. Source-reported results are labeled
as such; none below were measured on the author's hardware in this pass.

| Claim | Primary evidence | Boundary and limitation |
| --- | --- | --- |
| DeepSeek-V3: 671B total / 37B active, 14.8T tokens, 2.788M H800 hours | [Report v2, introduction and Table 1](https://arxiv.org/html/2412.19437v2) | Official training only; earlier research/ablations excluded. No conversion into total project cost. |
| FlashAttention-3: 1.5-2.0x, 740 TFLOP/s FP16, nearly 1.2 PFLOP/s FP8 | [Paper](https://arxiv.org/abs/2407.08608) | H100 attention kernels; FA2 speedup baseline. Not full-model throughput. |
| Mooncake: +525% simulation throughput / +75% handled production requests | [Paper](https://arxiv.org/abs/2407.00079) | Separate experiments and workloads under SLOs. +525% is 6.25x, not 5.25x. |
| RocketKV: up to 3x decode, 31% lower peak memory | [Paper, version 1](https://arxiv.org/abs/2502.14051v1) | H100; full-KV baseline and evaluated long-context tasks. These figures are from version 1; later revisions report a different hardware evaluation. Not lossless attention. |
| SparseServe: up to 9.26-fold mean TTFT reduction, 3.14x generation throughput | [Paper v1](https://arxiv.org/abs/2509.24626v1) | Hierarchical sparse-serving comparisons; separate maxima, not guaranteed jointly or on all hardware. |
| AgentDojo: 97 tasks, 629 security cases | [Published benchmark](https://arxiv.org/abs/2406.13352) | Original evaluation population, not a current repository-size claim. |
| Three test-time inference regimes | [Study v2, August 31, 2026](https://arxiv.org/abs/2608.04001v2) | Taxonomy of protocols; not a claim of universally monotonic quality gain. |
| ChatInject venue and mechanism | [Author project](https://hwanchang00.github.io/chatinject_project_page/) | ICLR 2026; no attack-success number added without a specific configuration. |
| Fides attribution | [Microsoft Research publication](https://www.microsoft.com/en-us/research/publication/securing-ai-agents-with-information-flow-control/) | Corrected the prior 2026 label to the 2025 work and replaced the ambiguous reference. |
| 128 KiB KV per token; 250 MiB for 2,000 tokens | `tests/test_resource_models.py` | Derived for L=32, eight KV heads, d=128, FP16; excludes padding and runtime workspace. |
| Weight-streaming and handoff floors | `tests/test_resource_models.py` | Assumed sustainable 3 TB/s HBM and 50 GB/s transfer; idealized lower bounds, not benchmarks. |
| Decode-attention KV read: 128 MiB per layer / 4 GiB over 32 layers | `tests/test_resource_models.py` | Derived for batch 8, eight KV heads, 4,096 retained tokens, d=128, FP16. Assumes each KV vector is read once; paging, cache misses, or repeated GQA loads can increase traffic. |

Foundations now has nearby primary references for Adam/AdamW, normalization,
transformer attention, GQA, and empirical scaling laws. Original exercises
and capacity examples use declared assumptions rather than sourced performance
measurements. The recently added TP/PCP/DCP section pins implementation details
to a vLLM revision instead of presenting development-branch constraints as
permanent architecture laws.

For future empirical additions, record model/checkpoint, dataset and shapes,
hardware/topology, precision, software revision, tuned baseline, quality bar,
metric denominator, and the specific table/figure. If those details are not
available, retain only a bounded source-reported description and do not turn
the maximum into a deployment recommendation.
