# Round 2 — independent technical/source review

Baseline: `a287725`. Reviewed the complete current accelerator chapter,
`manuscript/04_cuda.md:1524–1885`, and the campaign log. Primary documentation
and the selected attention-backend implementation were checked on 2026-09-13.
No other round-2 review was consulted.

## Verdict and scope

No new P0/P1 defect found in this bounded review. The operator examples and
their source-only/device-validation boundaries should remain intact; round-1
findings remain closed. The main opportunity is the missing intermediate
step between a correct RMSNorm and a deployable engine: an actual backend can
implement a familiar flag or feature name differently from the chapter's
abstract placement model.

Four P2 improvements follow, in priority order. These are concrete depth gaps,
not claims that the current hypothetical calculations are wrong. They can be
handled with two short worked decisions and two compact implementation
counterexamples, without another vendor catalogue or rederiving attention.

This review inspected documentation/source, not installed engine dispatch or
accelerator execution. It did not rerun the previously passing CPU suite,
audit every supported model, or establish performance on any device. The
campaign remains active; this report is not its completion gate.

## R2-T1 — distinguish independent CP ranks from an engine's nested DCP group

**Location:** `manuscript/04_cuda.md:1763–1765`, “More ranks can replicate
state,” and the generic process-group requirement at line 1828.

**Current text:** the independent `TP=4, CP=2` model has eight ranks and
512 MiB/rank. Its assumptions and disclaimer are correct. However, a reader
has no concrete example showing why the same-looking engine flags cannot be
substituted into this arithmetic or the CPU helper.

**Evidence:** [vLLM's context-parallel deployment guide](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)
states that DCP reuses TP ranks rather than increasing the launch world size.
Its memory saving comes from partitioning otherwise replicated KV state.
The [Neuron 2.32 DCP design](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/vllm-neuron/docs/design/parallelism/dcp.html)
documents decode TP=16/DCP=2 with adjacent pairs sharing a KV head and splitting
history. It requires TP beyond the KV-head count, bounds DCP by that replication
factor, imposes query-head/group divisibility, and currently excludes attention
DP. Its decode sequence gathers Q and partial log-normalizers, then combines
corrected outputs with reduce-scatter. Prefill uses different groups and an
explicit prefill-DCP path restricted to a disaggregated producer. These are
backend-specific constraints, not a universal definition of CP.

**Proposed insertion:** immediately after the existing independent-CP
calculation, contrast it with the running assistant's 32 query heads and eight
KV heads. A *modeled* TP=16/DCP=2 decode mapping has 16 ranks, not 32. Splitting
each pair's replicated history gives `512 MiB / 2 = 256 MiB` per rank and
`16 * 256 MiB = 4 GiB` aggregate payload, before allocation overhead. The stated
head-divisibility conditions are satisfied; that is not proof the complete
assistant deployment is supported. Keep the current independent-CP example.

At line 1828, replace part of the generic warning with one consequence: record
the phase-specific group membership and which collectives cross a host or
fabric boundary. A two-rank DCP group that stays local and one spanning hosts
have the same payload arithmetic but different communication paths to test.
Do not add a topology-independent performance prediction.

**Acceptance:** the reader can distinguish independent `TP × CP` from DCP
inside TP, can reproduce both rank counts, and is explicitly warned not to
feed the engine's DCP flag directly into the independent-CP helper. Any new
diagram/table names ranks rather than assuming one rank equals one device.
Keep algorithmic derivation of partial-attention normalization in the next
part; this chapter needs only the deployment consequence.

## R2-T2 — show a real attention-backend/connector incompatibility

**Location:** `manuscript/04_cuda.md:1605`, the inventory of attention backends
and quantization formats; line 1830, the cross-vendor cache-ABI warning.

**Current text:** “enumerate … attention backends” is good advice, but the
chapter never shows what to inspect. The later warning might leave readers
thinking that a homogeneous vendor pair is sufficient for a supported
prefill/decode connector.

**Evidence:** the current official [vLLM `RocmAttentionBackend` source](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/rocm_attn/)
declares `supports_kv_connector()` false and explains its incompatible KV
layout. This is a statement about `ROCM_ATTN`, not all ROCm attention backends
or all AMD disaggregation. The same implementation also makes head sizes,
cache dtypes, and cache-block requirements explicit. The [attention-backend
selection design](https://docs.vllm.ai/en/latest/design/attention_backends/)
describes configuration validation and notes that ROCm has its own selection
logic. Do not copy a CUDA backend priority list into a ROCm recipe.

**Proposed insertion:** after line 1605, add a small “one backend, two gates”
example. First validate the assistant's attention geometry/dtypes/block
configuration against the selected backend; then independently validate the
intended KV connector. A backend that passes local prefill/decode correctness
can still fail the second gate. Give the concrete `ROCM_ATTN` declaration as a
dated source-inspected example and leave installed-version validation open.
At line 1830, add “even on the same vendor” and refer back instead of repeating
the complete cache-ABI list.

The useful record is small: engine/build identifier, selected attention backend,
model/head geometry, KV dtype/block/layout, required connector, and observed
startup/dispatch evidence. Until a run exists, mark the final field unmeasured.

If one additional ROCm detail is desired, the [ROCm 7.2.4 vLLM guide](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)
separates attention selection from AITER GEMM/RMSNorm/MoE enablement. Therefore
an attention-backend flag does not establish which quantized linear kernel
the engine executes. That makes a useful bridge from the printed RMSNorm to
actual dispatch; the many tuning flags and vendor speedup figures do not.

**Acceptance:** one concrete backend declaration is cited; no blanket claim
that ROCm lacks disaggregation; no fabricated logs or “verified on hardware”
label. The example stops or chooses a separately supported path when a required
capability is absent, rather than presenting a cache-format conversion as an
already implemented fix.

## R2-T3 — separate the three meanings of “quantized model support”

**Location:** `manuscript/04_cuda.md:1757–1759`, before using the weight/KV
traffic ledger; line 1709 currently mentions quantization only as an item to
validate. Earlier CUDA chapters explain quantization mechanics, so this
addition should concern loading and execution contracts rather than repeat
that material.

**Gap:** checkpoint encoding, arithmetic precision, and KV storage are separate
compatibility questions. The running two-byte KV calculation is sound, but
there is no worked example preventing a reader from halving the entire memory
or latency ledger merely because an engine advertises FP8.

**Evidence:** the [current vLLM Neuron feature guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/features-guide.html)
describes static per-tensor ModelOpt FP8 checkpoint loading, rejects
compressed-tensors *weight* quantization in that path, and treats FP8 KV as a
separate option. That cache path stores quantized K/V but dequantizes them for
BF16 attention computation. This is a loader/representation/compute contract,
not one generic “FP8 supported” bit. The [upstream quantized-KV guide](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
also distinguishes scaling granularity and currently restricts per-head scale
support to the FlashAttention backend; it should not be generalized to every
alternative backend.

**Proposed insertion:** add a three-row acceptance record: weight artifact and
packing/scales; actual linear/attention compute path; KV storage format and
scale layout. Apply it to the assistant: changing only weight precision leaves
its two-byte KV payload at 4 GiB. A supported one-byte KV representation would
reduce the *element payload* to 2 GiB, plus scales/padding/reservations, without
establishing a twofold speedup or an FP8 attention matmul. This arithmetic can
be one additional CPU-executable assertion pair, not another framework import
example. Preserve the required preceding `Example status:` label if a new
manuscript code fence is introduced.

Keep the unmodified checkpoint as the portability baseline. If a backend needs
a requantized artifact, name its lineage and distinct artifact identity, and
rerun the fixed quality evaluation before making a serving comparison. Do not
quietly call differently quantized files the same checkpoint.

**Acceptance:** the three contracts are explicit, both payload totals can be
reproduced, and neither format names nor bit widths imply hardware execution,
quality equivalence, or latency. Source-date the Neuron example; the `latest`
feature guide was retrieved, but its guessed 2.32 versioned URL returned a
browser retrieval error, not a confirmed 404. Do not falsely mark that URL
verified or report it as broken.

## R2-T4 — make phase-specific engine support observable, not a feature label

**Location:** `manuscript/04_cuda.md:1818–1828`, phase acceptance table and
following generic serving requirements. Fold this into that section rather
than insert another accelerator overview.

**Gap:** the chapter identifies prefill/decode differences but offers no
concrete example where the same scheduling vocabulary hides a materially
different execution contract. This is a high-value connection between the
kernel lessons and the existing token-gap SLO.

**Evidence:** the [vLLM Neuron feature guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/features-guide.html)
documents segmented prefill and continuous batching, but explicitly keeps
prefill and decode in separate batches, unlike upstream mixed chunked-prefill
batching. This does not mean requests cannot interleave over time. Separately,
the [vLLM TPU feature matrix](https://docs.vllm.ai/projects/tpu/en/latest/features/)
distinguishes release/nightly, Flax/Torchax paths, and single-/multi-host
parallelism results. Its “untested” status means functionality exists without
recent/thorough verification, not necessarily absent implementation. A compiler
collective primitive is not evidence that an engine's entire topology and
feature combination has passed.

**Proposed insertion:** add one hypothetical trace exercise: eight active
decodes plus an arriving long prompt. Ask the reader to establish whether
chunks and decodes share a batch or alternate batches, which rank groups
participate, and which interval determines the worst token gap. Do not invent
timings. The result to record is the installed scheduler behavior plus TTFT
and token-gap measurements under the same replay, not just a checked
“chunked prefill” box. Reuse the chapter's current SLO rather than introduce a
new one.

A compact topology gate can then require the exact model/backend/path,
release, group arrangement, and host count to intersect in a documented or
locally validated configuration. Cite the TPU matrix as an example of how to
read evidence status; do not freeze its entire fast-changing support table
into the book. Cut the corresponding generic sentences at line 1828 to keep
the section moving.

**Acceptance:** “separate batches” is not misrepresented as no continuous
batching; “untested” is not rewritten as unsupported; no support inferred from
XLA/RCCL primitive availability alone; one concrete replay links a scheduling
choice to the already defined service objective.

## Source-selection caution

Prefer these narrow, inspectable counterexamples over a cross-vendor
quantization support chart. During this review, the broad [upstream vLLM
quantization table](https://docs.vllm.ai/en/latest/features/quantization/)
and AMD's versioned optimization guide did not agree on blanket AMD AWQ/GPTQ
availability. This is a reason to pin the build, model recipe, loader, and
selected kernel—not evidence that the current manuscript makes a false claim.
No new “all AMD supports/does not support format X” sentence is warranted.
