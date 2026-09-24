# Part VII revision - September 23, 2026

This revision updates every chapter in Part VII, Chapters 54-59, and its
introduction. The rest of the book retains its September 13 research cutoff
unless explicitly dated. The original `public-2026-09` release remains a fixed
historical snapshot; the revised release is `public-2026-09-23`.

## What was missing and what changed

| Chapter | Gap in the preceding edition | Substantive revision |
| --- | --- | --- |
| 54 - Efficient Frontier Models and Reasoning Training | Useful architecture foundations, but no current cross-provider model map; reasoning-training discussion still centered on R1 | Dated hosted/open-weight/access map; current hybrid-state comparisons; 2026 agent-environment and on-policy-distillation discussion; account-specific conversation contracts; additional worked exercises |
| 55 - Serving, Attention, and Memory Hierarchies | FA3 and earlier cache papers led the discussion; newer kernels and drafts were largely cross-references | FA4 and provisional FP4 evidence; cost-aware block drafts; architecture-owned compressed/shared/reconstructed cache; component versus total memory; numerical acceptance/throughput example |
| 56 - Multimodal Representations | Representation foundations did not establish the latest live interaction and media-generation boundaries | Current Live/TTS/video distinctions; native versus delegated modalities; concurrent speech/task state and stale-result rejection; new worked questions |
| 57 - Diffusion and Block-Parallel Generation | Masked denoising and block diffusion did not represent current deployed/experimental systems | DiffusionGemma's different corruption and cache rules, reported limitations, batching crossover; LLaDA2.2 sequence editing; Mercury 2.5 commercial contrast; full-round timing example |
| 58 - Multimodal and Tool-Using Systems | Insufficient long-horizon recovery, persistent-memory provenance, and current harness evaluation | Durable task ledgers, preserved-message contracts, restart reconciliation, Terminal-Bench 4.0, adaptive security tests and enforcement limits, changed-authority recovery exercise |
| 59 - From Research to Production | General result cards lacked concrete availability/evidence distinctions and adoption-bias controls | Verified-outcome economics, quality floors, held-out comparisons, contamination and selection controls, uncertainty, versioned maintenance, and explicit remaining scope limits |

Historical mechanism papers remain, labeled as foundations. This revision does
not equate recency with evidence strength or repeat vendor leaderboard rankings
as independently established facts.

## Source ledger

Sources below were inspected on September 23, 2026. Dates describe a report,
announcement, or card update where stated; they are not inferred launch dates.
A mutable card is not an immutable checkpoint. Deployment experiments should
pin actual checkpoint revisions, engine versions, API identifiers and settings.

| Source | Version/date boundary | Evidence used |
| --- | --- | --- |
| [OpenAI GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model) | Mutable official guide, inspected September 23 | Family and interface contracts; no proprietary architecture inference |
| [Opus 5.5](https://www.anthropic.com/claude-opus-5-5) | September 22 announcement | Availability and account-dependent preserved thinking |
| [Fable/Mythos 5.1](https://www.anthropic.com/claude-fable-and-mythos-5-1) | September 1 announcement | Model versus access/safeguard distinction |
| [Gemini 3.8 Flash](https://deepmind.google/models/model-cards/gemini-3-8-flash/) | September 2 card | Input/output and effort boundary |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Mutable publisher card | Hybrid layout; native/extended/hosted distinction |
| [Kimi K3 card](https://huggingface.co/moonshotai/Kimi-K3) and [report](https://arxiv.org/abs/2607.24653v2) | Report v2, August 7; current card | State/compute/storage distinctions and rollout infrastructure |
| [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) and [Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | Mutable publisher cards | Post-training on existing base versus new architecture |
| [DeepSeek-V4.1 card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and [report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) | Card inspected September 23; report section 5.1 downloaded and read | Cache boundary and data/environment-driven post-training |
| [FlashAttention-4](https://arxiv.org/abs/2603.05451) | March 2026 report | Asymmetric bottlenecks and bounded B200 comparison |
| [FP4 FA4](https://arxiv.org/abs/2609.04105v1) | September 3, v1 preprint | Distinct inference/training precision evidence and negative result |
| [DFlare](https://arxiv.org/abs/2606.02091v2) and [CaDDTree](https://arxiv.org/abs/2606.01813v1) | June 2 v2 / June 1 v1 | Conditioning capacity versus verification-cost-aware budgeting |
| [Gemini 3.8 Audio](https://deepmind.google/models/model-cards/gemini-3-8-audio/) | Card index lists update September 23 | Live versus TTS interfaces and disclosed limitations |
| [GPT-Live model](https://developers.openai.com/api/docs/models/gpt-live-1) and [delegation](https://developers.openai.com/api/docs/guides/live-delegation) | Mutable official documentation | Component modalities and application-owned backend state |
| [Gemini Omni Flash](https://deepmind.google/models/model-cards/gemini-omni-flash/) | August 27 card | Media generation/editing is a different contract |
| [DiffusionGemma](https://arxiv.org/html/2608.00146v1) | July 31, v1 | Corruption/refinement, adaptation, speed boundary and limitations |
| [vLLM implementation](https://vllm-project.github.io/2026/06/10/diffusion-gemma) | June 10 implementation account | Per-request attention causality and acceptance cache pass |
| [LLaDA2.2-flash](https://huggingface.co/inclusionAI/LLaDA2.2-flash) | Mutable publisher card | Sequence editing and explicit deployment-support limitation |
| [Mercury 2.5](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5) | September 8 announcement | Commercial service, vendor evidence only |
| [Terminal-Bench 4.0](https://www.tbench.ai/news/terminal-bench-4-0) | August 2026 release | Resource/task changes, public solutions, and rerun requirements |
| [Adaptive out-of-band defense evaluation](https://arxiv.org/abs/2606.26479v1) | June 25, v1 preprint | Limited adaptive result, not general security assurance |

## Review process

Three separate AI reviewer agents each examined two chapters. They were asked
to check current primary sources, correctness, material omissions, pedagogy,
arithmetic and scope, and to return a revise/approve verdict. They did not edit
the manuscript. The authoring agent applied changes and returned them for review.

The first pass requested stronger 2026 post-training coverage, account-scoped
preserved-thinking wording, distinct diffusion mechanisms and concrete
limitations, commercial/sequence-editing comparisons, adaptive security
qualification, contamination controls, and exercises tied to the additions.
These changes were applied in a second revision. A follow-up check also narrowed
Kimi K3 quantization to routed experts. All three agents then approved their
assigned chapters with no unresolved substantive findings. Final verdicts are
recorded in [release readiness](release-readiness.md).

This is AI review, not independent human review, hardware reproduction, an
exhaustive literature census, or certification of a vendor's claims.

## Remaining limitations

- No common-harness ranking or accelerator benchmarks were run for this revision.
- Proprietary training data, parameter counts and undisclosed recipes remain unknown.
- The model map is selective; robotics, general world models and specialized scientific foundation models are not comprehensively surveyed.
- API and model-card pages can change after this inspection date.
- Local CPU/document tests validate teaching code and publication structure, not the behavior of the cited frontier models.
