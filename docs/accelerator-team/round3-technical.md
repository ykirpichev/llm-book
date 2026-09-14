# Round 3 — independent technical continuity review

Reviewed the current working tree after round 2 on September 13, 2026. This
report checks continuity between the accelerator chapter and its attention,
state, serving-engine, and phase-sharding prerequisites. It is not a completion
gate for the three-hour campaign. No manuscript or implementation was edited.

## Verdict and scope

No P0/P1 publication blocker found. The core teaching sequence and running-model
arithmetic are coherent. Four bounded corrections follow: a real CPU-oracle
range bug, two overgeneralized state/phase statements, and reversed capacity
wording. An optional concrete scale-transfer illustration is included with the
last item; it does not justify another general compatibility checklist.

I read the complete current accelerator chapter; Part I's transformer and
compressed/sparse/recurrent-state chapters; Part III's service geometry,
prefill/decode, KV, scheduler, parallel/disaggregated serving, quantization, and
engine sections; and Part V's collectives, TP/SP/CP, and distributed-inference
sections. I also read the attention CPU reference and its tests. Other book
parts, unrelated examples, and rendered-page layout were not audited in this
pass. No other round-3 formal review was read before this report.

Primary-source checks covered current vLLM parallel configuration, hybrid-cache
management and NixlConnector contracts; Neuron SDK 2.32 DCP; the TensorRT LLM
backend-removal guide; and the cited FlashAttention-4/FP4 papers. Accelerator
execution was not available and is not claimed. The earlier full suite passed
103 tests during round-2 closure; the new adversarial probes below are additional
CPU observations, not yet regression tests.

## T1 — P2: finite-input validation does not make the attention oracle safe

**Location:** `examples/attention.py:18`,
especially score construction at lines 24–25, partial accumulation at 31–32,
the mass filter at 48, and final merge at 54–55. Tests:
`tests/test_examples.py:13`.

Root supplied two adversarial inputs, both independently reproduced with the
bundled CPU Python:

```python
attention([1e308], [[1e308]], [[1.0]])
# actual [0.0]; the sole visible key should contribute its value,
# or the reference should explicitly reject the numerical range.

attention([0.0], [[1.0], [2.0]], [[1.7e308], [1.7e308]])
# actual [inf]; the mathematical weighted average is 1.7e308.
```

The first case overflows a dot product, then obtains NaN from `inf - inf`.
Because `NaN > 0` is false, the merge silently discards a nonempty state as if
it were fully masked. The second overflows the unnormalized weighted-value
sum. It fails both with the default single shard and with `shard_size=1`, where
the local numerators are finite but their merge overflows. Max-subtracted
softmax prevents exponential overflow for finite scores; it does not protect
dot products or weighted-value accumulations.

**Correction:** retain this simple float64 reference and reject unrepresentable
intermediate/output ranges explicitly. Check computed visible scores before
softmax, positive finite mass and finite partial numerators, and finite merge
numerators/results. Only an actual zero-mass identity may be skipped; an invalid
state must not become an all-masked answer. Document the restricted range in the
reference contract. A scaled-accumulator redesign is unnecessary for this book.

**Acceptance:** both reproducers raise a clear range error in dense and sharded
paths; finite normal cases retain their current answers. Preserve zero output
for a genuinely all-masked nonempty row, including huge but masked inputs.
Add a mixed case where one entire shard is masked and another is valid; the
existing implementation correctly returns `[4.0]` for zero query, values
`[[2.0], [4.0]]`, mask `[False, True]`, and shard size one. That identity behavior
must survive the fix. Empty overall key/value input remains invalid under the
existing API; this is distinct from a masked local shard.

## T2 — P2: the serving-kernel introduction universalizes full-history KV

**Location:** `manuscript/04_cuda.md:1207`,
“Serving contracts restated.” It says autoregressive decode retains every
processed token's K/V for every layer and then gives the uniform KV formula.
Compare `manuscript/01_foundations.md:594`,
“Hybrids retain more than one kind of cache,” and the accelerator's explicitly
full-history GQA ledger at `manuscript/04_cuda.md:1761`.

This is true of the running full-history attention model, not autoregressive
decoding as a category. The earlier hybrid example has fixed-size recurrent
matrices plus convolution history and token-growing state only in its attention
layers. Sliding-window layers need not retain every prior token either. The
current [vLLM hybrid-cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
likewise distinguishes layer groups and computes prefix hits across their
state requirements; a token match in one group alone is insufficient.

**Correction:** start the paragraph with “For the running full-history GQA
model…” and retain its calculation. Add a short bridge, locally or beside the
accelerator ledger, saying that hybrid porting/admission/handoff must also
preserve the recurrent and convolution state at the same accepted prefix
boundary. Refer to Part I's existing example rather than restating its chapter.

**Acceptance:** the uniform formula is visibly a model assumption, not a
definition of all autoregressive state. A reader is directed to the existing
hybrid ledger and rollback contract. No new vendor support table or hybrid
kernel is needed. The existing hypothetical hybrid arithmetic is correct:
24 MiB of matrices plus 64 MiB of attention KV at 2,048 tokens, before convolution
state and allocation overhead.

## T3 — P2: the phase ledger's “only” excludes a documented DCP prefill mode

**Location:** `manuscript/05_distributed.md:893`,
“Phase-specific sharding ledger.” The DCP prefill cell says it is relevant
“only when a prefill chunk attends to an already DCP-sharded history.”

That is a useful canonical history-sharding case, but not an exhaustive rule
for the backend terminology the accelerator chapter now introduces.
[Neuron SDK 2.32's DCP documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/vllm-neuron/docs/design/parallelism/dcp.html)
also defines a prefill mode controlled by `apply_prefill_dcp` and the same DCP
degree flag: it partitions prompt work across subgroups, with separate weight
and process-group rules. Its stated preconditions include a disaggregated
KV-producer role, not an already populated historical cache. Prefill and decode
even derive their group rank differently.

**Correction:** replace “Relevant only when…” with wording scoped to the
history-sharding mechanism, such as “May attend to already sharded history;
backend-specific prefill modes require their own group/weight plan.” Add one
sentence explaining that a shared flag name does not establish identical
prefill and decode geometry, pointing back to the SDK-scoped accelerator case.
Do not rename every backend's prefill mode PCP by definition.

**Acceptance:** the table no longer excludes the documented prefill mode and
does not imply that the TP16/DCP2 decode constraints automatically validate a
prefill executable. This is separate from rank multiplication: the current
accelerator paragraph already says PCP is disabled before describing DCP within
TP. That earlier concern is closed; Part III and Part V's combined-layout
qualifications are consistent with [current vLLM parallel configuration](https://docs.vllm.ai/en/latest/api/vllm/config/parallel/).

## T4 — P3 correction, optional depth: distinguish footprint from capacity

**Location:** `manuscript/04_cuda.md:1294`,
“KV quantization and dequantization”: “Quantized K/V reduces capacity and
bandwidth.” This reverses the capacity consequence taught by the later 4 GiB
to 2 GiB ledger and by Part III.

**Correction and acceptance:** say that quantization reduces stored KV bytes
and potential read traffic, increasing the context/request capacity available
from a fixed memory budget, subject to metadata and kernel support. Do not
promise lower achieved latency merely from the storage reduction. This is a
one-sentence repair.

If adding one practical bridge between quantization and handoff, use the
existing [NixlConnector compatibility source](https://docs.vllm.ai/en/latest/features/nixl_connector_compatibility/#quantized-kv-cache):
its current contract requires matching cache dtype, supports independently
loaded checkpoint scales and packed inline scales, but does not transport
runtime-generated separate per-block scales. Thus identical FP8 dtype is
necessary but not sufficient. Insert two sentences after the existing
heterogeneous-TP compatibility paragraph in
`manuscript/03_inference.md:922`
or in the accelerator's precision paragraph, not both. This is optional depth,
not a claim that all connectors share that restriction. The current general
handoff ledger already names scales and is not itself wrong.

## N1 — justified nonfindings: retain these boundaries and calculations

- **Arithmetic and phase placement agree.** CPU recomputation gives 131,072
  bytes/token, exactly 250 MiB for a 2,000-token prompt, and 4 GiB for eight
  4,096-token sequences. At 25 GiB/s plus 4 ms overhead the handoff is
  13.765625 ms. Part III's approximately 5 ms wire floor uses a different,
  explicitly declared 50 GB/s link; it is not a contradictory measurement.
  TP16/DCP2's 256 MiB/rank and 4 GiB aggregate are correct under the cited
  replicated-GQA decode mapping. Keep independent CP arithmetic separate.
- **Modern attention is not treated as one interchangeable optimization.**
  Part I separates retained representation, sparse support, and recurrent
  state. Part IV correctly separates scheduling changes from omitting keys and
  separates low-bit QK from the probability/value path. The core Blackwell
  scheduling description agrees with the [FlashAttention-4 paper](https://arxiv.org/abs/2603.05451),
  and the cautious account of the September FP4 experiment matches its
  [authors' reported training limitation](https://arxiv.org/abs/2609.04105).
  No extra benchmark catalogue is needed.
- **The engine chapter is not stale on its major migration claim.** The
  current [TensorRT LLM migration guide](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html)
  confirms that the TensorRT engine backend was removed in favor of the
  PyTorch execution path. Preserve the book's warning about old build tutorials.
  Likewise, do not erase Part V's dated/pinned vLLM restrictions merely because
  a current configuration differs. The historical GitHub revision could not
  be retrieved again in this pass; current configuration was checked, but this
  report does not certify every historical condition independently.

Recommended authoring order: T1 regression and range contract; T2/T3 scope
repairs; T4 wording. The optional scale-transfer example should replace generic
compatibility advice if adopted, rather than add another checklist. Independent
QC should rerun the CPU suite and check these exact acceptance criteria.
