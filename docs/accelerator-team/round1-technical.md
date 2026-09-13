# Round 1 independent technical and source review

Reviewer: Astra technical review. Date: September 13, 2026.
Campaign baseline: `1fb029c`; reviewed the expanded working tree, not just that
commit. This review is one campaign stage, not a claim that the three-hour
campaign is complete. No other formal review was read.

## Verdict

No P0/P1 technical blocker found. The chapter is substantially stronger than
the baseline: the numerical operation stays fixed, the Triton indexing and HIP
barriers are explicit, and the distinctions among CPU semantics, device
compilation, device performance, and deployment evidence are consistently made.
The TP/CP arithmetic and compile-amortization example are correct under their
stated assumptions.

Three localized P2 findings should be fixed. The highest-value next depth work
is a concrete legal tile/placement example for Pallas or NKI, not more ecosystem
names or unexecuted performance claims.

## Findings

### T1 — P2: TorchNeuron's access restriction is missing from the deployment options

Location: `manuscript/04_cuda.md:1695`, **AWS accelerators: Neuron is the stack,
not the chip**.

The paragraph presents native PyTorch through TorchNeuron alongside PyTorch
NeuronX and JAX as distinct integration choices, but omits that the cited
release's native backend is restricted to a closed beta. This matters in a
chapter whose recurring decision is to find a supported, usable deployment
stack: documentation of an interface is not access to that interface.

The primary [SDK 2.32 native PyTorch overview](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/frameworks/torch/pytorch-native-overview.html)
explicitly states that TorchNeuron is currently available only through a closed
Beta program, with participation through an AWS Neuron representative. It also
distinguishes it from the older integration paths.

Minimal correction: qualify the sentence with “currently a closed-beta path in
the checked SDK documentation” and link directly to that overview. Keep the
other integration paths distinct. Do not infer that native TorchNeuron's
support or compilation rules apply to vLLM Neuron, PyTorch NeuronX, or JAX.

### T2 — P2: The CPU RMSNorm oracle silently accepts an overflowing denominator

Location: `examples/accelerator_portability.py:36`, `rmsnorm_reference`; related
contract at lines 18–22 and tests in `tests/test_accelerator_portability.py`.

The function checks the squared sum and final outputs for finiteness but does
not check `square_sum / len(row) + epsilon`. This can overflow even when the
squared sum, epsilon, and true normalized result are individually representable.
The overflow becomes an infinite denominator and then a zero inverse; the final
finite-output check misses the error.

Reproduced using Python 3.12.14:

```python
rmsnorm_reference([1e154], [1.0], epsilon=1e308)
# Actual: [0.0]
# Mathematical result: approximately [0.7071067811865475]
```

This is an extreme-domain oracle issue, not a claim that the bounded Triton
example fails at its documented epsilon range. The CPU function nevertheless
explicitly accepts arbitrary finite positive epsilon and says range errors
raise rather than silently becoming invalid results.

Minimal correction: compute the denominator argument in a named variable,
reject nonfinite values with `ValueError`, and document that all intermediate
normalization quantities must fit the reference's arithmetic. Add the case
above as a rejection test. Alternatively, implement a scaled/hypot formulation
if broad-domain robustness is intentionally desired; that is unnecessary for
this teaching scope.

### T3 — P2: The Neuron Explorer capture link points to a missing page

Location: `manuscript/04_cuda.md:1722`, **NKI: follow the tile between memory
spaces**.

Current target:
`https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-explorer/capture-profiles.html`.

The author independently reported an HTTP 404 while checking all chapter URLs.
I followed the SDK's own navigation and verified the replacement primary page:
[Capture profiles with Neuron Explorer, SDK 2.32](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/tools/neuron-explorer/how-to-profile-workload.html).
Its current alias is under `/en/latest/tools/neuron-explorer/how-to-profile-workload.html`.

Replace the target with the versioned capture guide. If a short concrete command
is added, the current guide documents `neuron-explorer capture`, replacing the
older `neuron-profile capture`. It executes a compiled NEFF; pass representative
named input files rather than interpreting default zero-filled inputs as a
workload benchmark. System traces and device traces answer different questions,
and device capture can reserve substantial HBM. None of these commands was run
in this review.

## Useful missing depth, not additional correctness findings

### D1 — Make one Pallas tile choice explicit

Locations: `manuscript/04_cuda.md:1687–1691`.

The text correctly asks the scheduling questions, but a reader cannot yet see
one answer. Add a compact worked `BlockSpec` choice for a bounded two-dimensional
array, such as eight complete 4096-element rows per program, and contrast it
with a row fragment that needs a second reduction stage. Count the input tile's
FP32 bytes: `8 * 4096 * 4 = 128 KiB`, then account separately for output,
weights, temporaries, and additional pipeline buffers. Do not equate this byte
count with a proof of VMEM fit or an optimal tile.

The primary [Grids and BlockSpecs guide](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
gives the TPU shape restrictions, including full-dimension exceptions; these
are more precise than a blanket “multiples of 8 and 128” rule. The
[TPU pipelining guide](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
explains VMEM block windows and additional buffering. A small legal/illegal
layout comparison would teach more than another disclaimer.

### D2 — Expose the NKI operator's actual tile arguments

Locations: `manuscript/04_cuda.md:1707–1720`.

The pseudocode and PSUM qualification are sound. To make them actionable, show
the correspondence between the mathematical width and `nl.rms_norm`'s `axis`
and `n` parameters, using one SDK-matched tile example. A row fragment is not
made globally normalized merely by passing the global N: its squared sum must
also represent the full row. Show where weights are broadcast or tiled and
state the computation/output dtypes.

The official [NKI language source reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_modules/nki/language.html#rms_norm)
documents the operator and a `(128,512)` SBUF example with `axis=1, n=512`.
The chapter's versioned generated API link was not readable through my web
retrieval tool; the author confirmed HTTP 200, and the official source reference
independently confirms the API. This is not an additional broken-link finding.

### D3 — Add a compilation-cache identity counterexample

Locations: `manuscript/04_cuda.md:1778–1797`.

The prose is accurate but a small contrast would make it operational: two calls
with the same prompt length can still require different artifacts because of
dtype, sharding, or static configuration, while a changed runtime valid-length
value inside the same bounded storage need not do so. Give a hypothetical
cross-product of shape buckets and static configurations, and state that the
integration determines which dimensions actually specialize. Include the
artifact compatibility identity in the prewarm manifest.

The [JAX persistent-cache guide](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
supports the compatibility/trust distinction. The
[shape-polymorphic export guide](https://docs.jax.dev/en/latest/export/shape_poly.html)
explicitly distinguishes reused tracing/lowering from compilation for concrete
shapes. Preserve the chapter's existing wording on that distinction.

## Checks that passed and evidence limits

- Read the entire expanded chapter, `manuscript/04_cuda.md:1524–1857`, the
  optional Triton file and README, the CPU planner/oracle, and its seven tests.
- Ran `tests.test_accelerator_portability` under bundled Python 3.12.14: all
  seven tests passed. The additional denominator-overflow reproducer returned
  `[0.0]` as recorded above. No GPU, TPU, or Neuron compilation/execution was
  attempted, and no timing claim is made.
- Source review found the Triton cast-before-square, true-N denominator, masked
  loads/stores, independent read strides, contiguous output, int64 offset
  arithmetic, zero-row launch suppression, epsilon range, and forward-only
  wrapper consistent. The HIP excerpt's 256-thread reduction reaches every
  barrier, including threads that own no input elements.
- Verified the general Pallas interpret contract against
  [`pallas_call`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.pallas_call.html)
  and its distinct TPU simulator against
  [`InterpretParams`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.InterpretParams.html).
  The manuscript does not confuse either with device performance evidence.
- Verified NKI 0.6/SDK 2.32 mapping and shared-HBM returns against the
  [migration guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/migration/index.html),
  and sequential functional simulation against the
  [CPU simulator documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/nki_simulator.html).
  NxD maintenance and the newer vLLM Neuron path agree with the
  [component release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/nxd-inference.html)
  and [SDK announcement](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html).
- Verified the TPU vLLM integration claim against
  [Google's inference guide](https://docs.cloud.google.com/tpu/docs/tpu-inference).
  HIP/Triton/profiling facts were cross-checked with the primary sources recorded
  in `prep-gpu.md`; no contradictory change was found.
- TP=4, TP=16 KV replication, TP4×CP2 partitioning, padded token arithmetic,
  serial compile break-even, and the serialized 110 ms handoff example are
  internally consistent. They appropriately avoid claiming engine support.
- I did not inspect rendered pages in this role or independently verify every
  chapter hyperlink over HTTP. The author supplied the 44-URL HTTP audit; only
  the Neuron capture failure/replacement was incorporated here. No other
  review's judgment was used.

Reviewed snapshot SHA-256 prefixes: manuscript `3580fd0e18ab`, Triton example
`2aa374d8c856`, CPU planner `fda66a8275f8`. Only this review file was authored.
