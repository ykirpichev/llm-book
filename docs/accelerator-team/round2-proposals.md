# Round 2 improvement proposals

**Baseline:** `a287725`  
**Role:** proposer, September 13, 2026

This proposal reconciles the independent technical, editorial, and integration reviews. No reviewer found a new P0/P1 correctness defect in the chapter. The useful round-two work is to replace broad warnings with two worked deployment decisions, make future target evidence auditable, and improve the reading rhythm without removing the concrete operator material added in round one.

Preserve the Triton body, HIP reduction and launch, Pallas `BlockSpec`, NKI schedule and API fragment, the self-contained CPU KV calculation, all four new diagrams, the variant table, and five exercises. Do not add another accelerator catalogue, full attention derivation, support matrix, speculative benchmark, or optional JAX dependency.

Finding keys used below:

- `T1`–`T4`: Round-two technical findings R2-T1 through R2-T4.
- `E1`–`E5`: the five priority sections in `round2-editorial.md`.
- `I1`: engine acceptance report; `I2`: automatic target provenance; `I3`: target-only Triton contract/stream tests; `I4`: HIP evidence checklist from `round2-integration.md`.

## Package 1 — Reclaim space and put explanations before qualifications

**Order:** first. This creates the prose and page budget for Packages 2 and 3.  
**Closes:** `E1`, `E2`, `E3`, `E4`; prepares `E5`.

Make four bounded editorial changes in `manuscript/04_cuda.md`:

1. At lines 1532–1550, retain the accelerator-portability diagram but remove the immediately following five-row taxonomy table. The surrounding prose already distinguishes device, stack, and kernel language, and the later table at lines 1746–1751 provides the useful synthesis after the reader has seen the mechanisms. Keep the three acceptance tests, three-stage roadmap, and one linked SYCL sentence.
2. At lines 1554–1562, keep the tensor/dtype contract beside the RMSNorm equation, then show the padding diagram and three-element result before discussing bounded activation range, epsilon rounding, backward coverage, and tolerance selection. Fold those qualifications into the validation paragraph rather than expanding them.
3. At lines 1701–1715, state the checked Neuron serving path and dated NxD/vLLM boundary directly after defining Trainium, Inferentia, and Neuron. Compress the current two migration paragraphs into one positive routing paragraph. Then let the NKI lesson run continuously from HBM/SBUF/PSUM through namespaces, tile schedule, API fragment, and simulator/device gates.
4. Normalize listing provenance into short, stable forms: `Source-only`, `Explanatory pseudocode`, or `Runnable Python`. In particular, replace `Hardware-dependent HIP...` with `Source-only HIP kernel excerpt; no compile or device run recorded`, and replace `Illustrative on-target...` with `Source-only rocprofv3 recipe; not run here`. Keep input and lifetime contracts in the explanatory prose instead of repeating them in status labels.

The Neuron compatibility paragraph should lead with what to inspect: under the checked SDK 2.32 documentation, begin with the vLLM Neuron plugin's supported models and targets; it is beta, currently targets Trn2/Trn3, and no longer depends on NxD Inference; NxD Inference entered maintenance mode with SDK 2.32. Then use the migration guide to separate a supported model move from custom-modeling work. Retain the existing primary links and checked date.

**Why this earns priority:** the chapter currently teaches the right facts but occasionally presents limits before the reader has a usable model. This pass makes the prose more natural and affirmative, removes immediate diagram/table duplication, and should reclaim enough lines to absorb the two worked decisions without growing the chapter materially.

**Alternative:** if the opening table is required for indexing, reduce it to a single sentence naming the four paths plus SYCL; do not retain both the full table and diagram. If the Neuron release paragraph cannot be shortened safely, move it intact before the NKI memory paragraph—the order matters more than the word saving.

**Acceptance:** no mechanism, link, date, or provenance boundary is lost; all `Example status:` proximity tests still pass; a reader reaches the RMSNorm picture before the range caveats and can follow the NKI tile without a release-history interruption. The later porting-question table remains. A manuscript word-count comparison confirms that this package removes enough prose to offset a substantial part of Packages 2 and 3.

## Package 2 — Worked decision 1: admit a backend, cache format, and connector together

**Order:** second.  
**Closes:** `T2`, `T3`; replaces generic warnings near current lines 1605, 1757–1759, and 1830.

Build one worked decision around the running assistant rather than adding separate connector and quantization mini-catalogues.

### Decision sequence

1. **Can the selected attention backend execute the local model contract?** Record engine/build identity, selected attention backend, head geometry, cache dtype/block/layout, and observed startup/dispatch selection.
2. **Can that exact backend participate in the required KV connector?** Use the current official [`RocmAttentionBackend` source](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/rocm_attn/) as a dated code-reading example: its `ROCM_ATTN` implementation declares `supports_kv_connector()` false because its KV layout is incompatible. State narrowly that this blocks the proposed connector for that inspected backend/version; it is not a claim that every ROCm backend or AMD deployment lacks disaggregation. A local prefill/decode correctness pass does not answer the connector question.
3. **Which object is quantized?** Name three independent contracts: weight artifact/packing/scales, actual linear or attention compute path, and KV storage/scale layout. Use the current [vLLM Neuron feature guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/features-guide.html) only as a dated example of why “FP8 support” is not one bit: its model-loading and FP8-KV paths are separate, and quantized K/V storage does not by itself mean FP8 attention computation. The [upstream quantized-KV guide](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) can support the separate scale-layout point.
4. **Recompute only what changed.** Changing weight precision alone leaves the chapter's two-byte KV payload at 4 GiB. A supported one-byte KV representation changes the element payload to 2 GiB for the same geometry, before scales, padding, allocator reservations, or replication. It does not establish half the memory allocation, twice the speed, equal quality, or an FP8 attention matmul.
5. **Decide.** If the required connector is absent, keep prefill and decode on one compatible path or select a separately supported backend; do not present cache conversion as an implemented remedy. If a backend requires a requantized checkpoint, give it distinct artifact lineage and rerun the fixed quality gate.

Place the concrete backend gate after the current ROCm engine inventory around line 1605, and put the three quantization contracts plus arithmetic beside the model/KV ledger around lines 1757–1759. At current line 1830, replace the long generic cache-ABI list with a short callback: even a same-vendor pair must pass the backend/connector/representation decision above.

Extend the existing self-contained KV listing rather than add a second code fence: one or two named values/assertions may show that two-byte and one-byte element payloads are 4 GiB and 2 GiB. Keep `Example status: Runnable Python` and update the existing CPU test/helper output only if the printed calculation depends on them.

**Alternative:** if a source-level backend declaration is considered too volatile for body prose, put the inspected symbol, URL, and retrieval date in a compact code-reading note while keeping the decision sequence in the main text. Do not replace it with a cross-vendor feature table. If the extra assertions make the visible listing wrap poorly, keep the 2 GiB calculation in one displayed sentence and add a CPU unit test behind it.

**Acceptance:** the reader can explain why local attention support and connector support are different gates; the text names `ROCM_ATTN`, not all ROCm; the three quantization contracts are explicit; 4 GiB and 2 GiB are reproducible under stated element-payload assumptions; no format name implies a compute path, quality result, allocation total, or speedup. The unmodified checkpoint remains the comparison baseline, and any requantized artifact has distinct identity. No fabricated logs or target results appear.

## Package 3 — Worked decision 2: distinguish rank arithmetic from engine scheduling

**Order:** third.  
**Closes:** `T1`, `T4`; replaces generic process-group prose near lines 1763–1765 and 1818–1828.

Use one source-grounded deployment scenario to connect the chapter's placement arithmetic to observable scheduling.

### Decision sequence

1. **Name the parallelism semantics before multiplying ranks.** Preserve the existing independent `TP=4, CP=2` model: it launches eight ranks and stores 512 MiB/rank, 4 GiB aggregate modeled KV. Then contrast it with the current [vLLM context-parallel deployment model](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/), where decode context parallelism reuses ranks inside TP rather than multiplying world size.
2. **Work the running geometry once.** For the assistant's 32 query heads and eight KV heads, present the [Neuron SDK 2.32 DCP design](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/vllm-neuron/docs/design/parallelism/dcp.html) as a backend-specific example. A modeled `TP=16, DCP=2` decode mapping has 16 ranks, not 32. Starting from the current TP=16 model's 512 MiB/rank, splitting each replicated pair's history yields 256 MiB/rank and 4 GiB aggregate element payload before overhead. State the documented head/group divisibility conditions and that this arithmetic does not prove the full assistant configuration is supported. Do not pass an engine DCP flag into the independent-CP helper.
3. **Make phase behavior observable.** Use a hypothetical replay with eight active decodes and one arriving long prompt. Ask whether the installed engine mixes prompt chunks with decode work in one batch or alternates phase-specific batches, which rank groups participate, and which interval creates the worst visible token gap. The current [vLLM Neuron feature guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/features-guide.html) is a useful narrow example: segmented prefill and continuous batching coexist with separate prefill and decode batches. Do not rewrite that as “no continuous batching,” and do not invent timings.
4. **Decide with the existing SLO.** Record installed scheduler behavior, TTFT, token-gap tails, group membership, and whether collectives stay within a host or cross the fabric under the same replay. Identical payload arithmetic can traverse different communication paths. A feature flag or available collective primitive is not the result.

Place the DCP contrast directly after the independent CP example. Replace the generic process-group paragraph around line 1828 with the replay and its evidence record, leaving detailed partial-attention normalization to Part V. Revise existing Exercises 2 and 3 rather than adding more exercises: one can ask for the scheduler trace/TTFT-token-gap diagnosis; the other can ask why `TP=16,DCP=2` is 16 ranks while `TP=4,CP=2` is eight and why the helper cannot treat the flags as synonyms.

**Alternative:** if Neuron-specific DCP would overconcentrate the chapter, keep the same numeric contrast but label it as the inspected engine's nested-DCP contract and move the extra Neuron divisibility details into a linked note. The minimum acceptable version still needs both rank counts and the eight-decodes-plus-long-prompt replay. Do not add a new parallelism diagram; the existing KV placement figure should remain the abstract model.

**Acceptance:** independent CP and DCP-within-TP are not conflated; both rank counts and per-rank/aggregate payloads are reproducible; the backend-specific conditions are named without becoming a universal CP definition; “separate batches” is not misrepresented as lack of continuous batching; no timings or topology-independent performance claims are invented. The scenario leads to measurements under the chapter's existing 1.5-second TTFT and 100-millisecond token-gap objectives.

## Package 4 — Make a future target run auditable without claiming one occurred

**Order:** fourth, after the chapter decisions stabilize.  
**Closes:** `I1`, `I2`, `I3`, `I4`.

Add a compact `docs/accelerator-team/engine-acceptance-template.md` and link it from the final evidence section and `examples/accelerators/README.md`. Require one row or record for both candidate and baseline, using the same denominators:

- identity: checkpoint/tokenizer, engine/backend/compiler/runtime, target, TP/CP/DCP/PP, attention backend, KV dtype/layout/quantization, required features;
- workload: prompt/output cohorts, arrival process, offered load, cancellation/retry policy, warm/cold mix, and duration;
- correctness: numerical/tolerance and task-quality gates, failures/rejections, connector/conversion status;
- service: TTFT and token-gap percentiles by cohort, goodput, accepted concurrency, queue time, cold readiness, and compile/cache misses;
- capacity/cost: constraining-rank peak memory, host/network resources, and cost per SLO-qualified request;
- decision: pass only at the same required features, quality, offered load, and SLO; otherwise record the limiting evidence and fallback/no-go.

Make target harness provenance machine-copyable before any case output. For Triton and HIP, emit the source revision when supplied (otherwise `unknown`), device and architecture, framework/runtime/compiler identity available through that stack, shapes/strides/dtypes attempted and skipped, epsilon, and tolerances. Never synthesize a revision or driver value that the program cannot determine.

For Triton, add a separately invoked target-only `contract` and `stream-ordering` mode, or an equivalent GPU-only test file. Cover the wrapper's declared rejection paths and one producer-event/consumer-wait case on non-default streams. Keep it outside the default CPU suite and report unsupported dtype/stream cases as explicit skips. For HIP, put a matching target-evidence checklist beside the build command: compilation tuple, bounded output comparison, launch/asynchronous error boundary, producer-event case, non-finite/range policy, then integrated quality/SLO replay. State the sample's actual limits—FP32, one device, traditional allocation/pageable host vectors—without turning the checklist into another page of caveats.

**Alternative:** if implementation time is tight, land the acceptance template, README link, and symmetric Triton/HIP checklists first; defer harness-emitted provenance and the stream test to a later device-enabled change. This smaller alternative improves reporting but should not be marked as automatic provenance closure (`I2`) or executable target-test closure (`I3`).

**Acceptance:** the template forces baseline/candidate identity and workload parity; default CPU commands gain no GPU dependency and still pass; target-only modes are opt-in; skipped cases are distinct from passes; a successful sample run is labeled device-operator evidence, not integrated-engine or benchmark evidence. No checked-in report claims a Triton, HIP, TPU, or Neuron run unless an actual captured run supports it. Do not add AST/source-string tests or a Pallas interpreter test solely to manufacture another green check.

## Package 5 — Integrated prose, exercise, and visual QC

**Order:** last in round two.  
**Closes:** `E5` and verifies all other findings.

After Packages 1–4 selected for implementation are complete:

1. Read the chapter continuously for human cadence. Remove any residual sentence that merely restates “support must be validated” after a worked decision has already shown what to inspect. Keep mechanism-specific constraints and dates; avoid stacking `do not`, `not proof`, and `does not imply` clauses in adjacent paragraphs.
2. Confirm there are exactly two new worked decisions, not a connector section, quantization section, DCP section, and scheduler section masquerading as four additions. The backend/format/connector decision and the rank/scheduler decision should each culminate in an observable go/no-go record.
3. Recheck the revised exercises and solutions against the new decisions. They should require arithmetic or evidence interpretation, not introduce a third scenario or repeat the acceptance table.
4. Render the entire chapter before changing figure geometry. The Package 1 cuts may cure the large gaps previously seen on pages 252, 255, 262, and 264. If a material gap remains before `kv_port_placement` or `compile_lifecycle`, reduce internal vertical padding/connector spacing by roughly 15–20% while preserving font size, labels, branches, and captions. Do not split the HIP reduction or shrink code typography to fill a page.
5. Run the supported CPU suite, manuscript tests, link/reference checks, and source formatting checks. Inspect every changed page plus the Part V transition. Record unavailable target checks separately.

**Alternative:** accept some whitespace when the only cure would split a code block, detach a figure from its explanation, or reduce legibility. A lower page count is not an acceptance criterion.

**Acceptance:** no overflow, split identifier, orphaned status label, detached caption, broken link, or unreadable figure; the standalone CPU calculation remains visibly executable; source-only labels remain accurate; the closing sentence still hands ownership and failure to Part V without another summary. The final round-two report distinguishes CPU/build evidence from every unrun target gate and does not claim campaign completion before the remaining scheduled review time.

## Finding-to-package map

| Finding | Package | Resolution |
| --- | --- | --- |
| `E1` opening duplication | 1 | Keep diagram; remove immediate taxonomy table; preserve later synthesis table. |
| `E2` Neuron flow | 1 | Current serving route and dated boundary first; uninterrupted NKI tile lesson second. |
| `E3` caveat-led RMSNorm opening | 1 | Worked visual example before range/backward qualifications. |
| `E4` inconsistent status register | 1 | Short, stable provenance vocabulary. |
| `T2` backend/connector mismatch | 2 | One dated `ROCM_ATTN` code-reading gate, narrowly scoped. |
| `T3` meanings of quantized support | 2 | Weight artifact, compute, and KV storage contracts plus 4 GiB/2 GiB arithmetic. |
| `T1` independent CP versus nested DCP | 3 | Reproducible rank/payload contrast; no helper conflation. |
| `T4` phase-specific scheduler behavior | 3 | Eight-decode/long-prompt replay tied to existing SLOs. |
| `I1` report denominator | 4 | Candidate/baseline engine acceptance template. |
| `I2` provenance | 4 | Machine-copyable target identity and fixture header. |
| `I3` target-only contract/stream tests | 4 | Opt-in labeled test mode; default CPU suite unchanged. |
| `I4` HIP evidence asymmetry | 4 | HIP checklist parallel to Triton, with explicit sample limits. |
| `E5` pagination | 5 | Full reflow after cuts; only then bounded internal figure compaction. |

## Selection guidance

Packages 1–3 are the recommended chapter revision and should be implemented in order. Package 4 is useful evidence infrastructure, with its smaller alternative acceptable if device-harness work would displace chapter quality. Package 5 is mandatory QC for whichever subset is selected.

The expected result is not a larger survey. It is a more natural chapter in which two concrete decisions demonstrate why engine flags, quantization labels, and parallelism names are insufficient—and exactly what a team measures next.
