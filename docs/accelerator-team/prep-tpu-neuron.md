# TPU/Pallas and AWS Neuron/NKI preparation brief

Source snapshot: 2026-09-13. This is a source-research note for the accelerator
chapter, not a hardware validation report. I checked current primary JAX and
AWS documentation. I did not have a TPU or Neuron device, did not compile any
kernel for either target, and did not measure latency, bandwidth, memory use,
or numerical error. Any chapter code derived from this note still needs to be
run against the pinned software and named accelerator generation.

## Editorial recommendation

Give each ecosystem one short, honest development loop rather than a
CUDA-shaped pseudo-port:

1. State the scalar or array contract in ordinary JAX/NumPy.
2. Explain the target's memory movement and tiling model.
3. Show a deliberately small kernel skeleton or link to a maintained example.
4. Separate CPU semantic checks from device compilation, numerical validation,
   and hardware profiling.
5. Name the SDK/JAX versions and target generation beside any runnable code.

For RMSNorm, the valuable comparison is the invariant contract and the
different schedules: rows and weights begin in device HBM, working tiles move
on chip, the reduction accumulates more accurately than the input format where
supported, and results return to HBM. Avoid four superficially parallel code
listings that imply equivalent APIs, supported shapes, or measured speed.

## JAX Pallas on TPU

### What is current and safe to say

Pallas is an experimental JAX extension for writing custom kernels. Its shared
surface does not erase backend differences: TPU uses the Mosaic lowering and
TPU-specific memory spaces and pipeline helpers; the GPU paths use different
lowerings and programming details. The current overview explicitly warns that
Pallas and its backends are experimental and under active development.

Primary references:

- [Pallas overview](https://docs.jax.dev/en/latest/pallas/)
- [Pallas quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html)
- [TPU quickstart](https://docs.jax.dev/en/latest/pallas/tpu/quickstart.html)
- [TPU details](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
- [Pallas changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html)

The memory distinction worth teaching is qualitative, not a portable capacity
number:

- HBM is the large off-chip/device working store.
- TPU VMEM is software-visible on-chip vector memory; SMEM is on-chip scalar
  memory; semaphore memory coordinates asynchronous work.
- TPU computation normally operates on VMEM/SMEM data, not directly on an HBM
  reference. A `pallas_call`/pipeline can arrange an HBM-to-VMEM window before
  the body executes. Lower-level code can use `pltpu.sync_copy`; pipelined code
  can use `pltpu.emit_pipeline` to stage and overlap transfers.
- In generic Pallas examples, a `Ref` means a mutable memory view. Its actual
  memory space depends on the backend and how the kernel is built; do not define
  every `Ref` as either “HBM” or “SRAM.”

The official [TPU pipelining guide](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
is the strongest source for `pl.ANY`, `pltpu.VMEM`, `pltpu.SMEM`, DMA, scratch
buffers, and `emit_pipeline`. The [TPU matrix-multiplication
guide](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html) is a better
code-reading reference than an invented full RMSNorm: it shows how a grid,
blocked windows, reduction-axis semantics, and arithmetic intensity fit
together.

Do not generalize one TPU generation's vector-tile dimensions, VMEM capacity,
or number of TensorCores into a timeless statement. The TPU details guide also
documents shape/layout restrictions and the possibility of VMEM exhaustion;
those constraints must be checked against the target generation and installed
JAX/jaxlib.

### `interpret=True` is not TPU execution

This distinction should be explicit in the chapter. The current
[`pallas_call` API](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.pallas_call.html)
defines `interpret=True` as a `jax.jit`-compiled scan over the grid whose body
is the kernel lowered as a JAX function. It needs neither GPU nor TPU and is the
only Pallas-on-CPU route. It is useful for debugging array semantics, indexing,
and comparison with a reference implementation.

It does **not** validate TPU code generation, TPU-supported operations, block
or layout legality, VMEM fit, DMA scheduling, multicore work distribution,
device numerical behavior, or performance. A kernel that passes interpret mode
can still fail compilation or behave differently once target-specific details
are introduced. Conversely, an interpret-mode number has no standing as a TPU
latency measurement.

There is also a distinct TPU-specific CPU interpreter,
[`jax.experimental.pallas.tpu.force_tpu_interpret_mode`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.force_tpu_interpret_mode.html),
which models TPU memory spaces, DMA, semaphores, and barriers more closely and
offers debugging checks. It is still a CPU simulation, not evidence of Mosaic
code generation or hardware performance. If mentioned, keep the two mechanisms
separate rather than calling both simply “the TPU simulator.”

### Small chapter-safe pseudocode

This is a teaching skeleton, not asserted runnable TPU code. The omitted block
shapes, layouts, reduction schedule, dtype choices, and compiler parameters are
exactly the target-specific work a real kernel must supply.

```python
# Semantic structure only; not hardware-validated Pallas.
def rmsnorm_tile(x_vmem_ref, w_vmem_ref, y_vmem_ref):
    x = read_current_row_tile(x_vmem_ref, accumulator_dtype=float32)
    mean_square = reduce_sum(x * x) / hidden_size
    inv_rms = rsqrt(mean_square + epsilon)
    y_vmem_ref[...] = cast(x * inv_rms * w_vmem_ref[...], output_dtype)

# A real TPU wrapper chooses a grid and block/window specifications so that
# HBM rows and weights are staged into VMEM, then copied back after the body.
```

The prose should call out three unresolved design choices rather than hiding
them: whether one program handles a row or a row fragment; how fragments combine
their sum of squares; and whether the weight tile is reloaded, retained, or
otherwise reused. These choices drive transfers, synchronization, and numeric
behavior.

If the chapter needs runnable syntax, prefer a tiny elementwise example adapted
from the current quickstart, pin its JAX/jaxlib version, and label its observed
targets. Do not represent the skeleton above as a drop-in RMSNorm.

### Compilation and shape behavior

JAX tracing and compilation are part of the operational story. The official
[`jax.jit` guide](https://docs.jax.dev/en/latest/jit-compilation.html) and
[JIT compilation notes](https://docs.jax.dev/en/latest/jit-compilation.html)
describe first-call compilation and caching. Cache keys are sensitive to such
properties as concrete shapes/dtypes and static arguments, so serving a stream
of novel shapes can create repeated compilation work. For a serving example,
bounded length buckets plus masking are a defensible planning pattern, but the
chapter should describe them as an application design, not as a Pallas API
guarantee.

JAX [shape-polymorphic
export](https://docs.jax.dev/en/latest/export/shape_poly.html) can avoid
re-tracing and re-lowering a function for a family of shapes. The exported
program is nevertheless compiled for each concrete input shape when invoked.
It is therefore not evidence that one device executable has unbounded dynamic
tensor extents or that every Pallas block schedule supports them.

Within TPU Pallas pipelines, the current pipelining guide documents bounded
dynamic windows (`pl.BoundedSlice` with a maximum size) and dynamic slices.
That is a more precise model for a ragged tile: storage and the legal maximum
remain bounded while a runtime value selects the active portion.

Useful diagnostics:

- `JAX_LOG_COMPILES=1` / `jax_log_compiles` and
  `jax_explain_cache_misses` expose unexpected recompilation.
- `pallas_call(debug=True)` prints intermediate compiler forms; it does not
  benchmark the device.
- [JAX profiling](https://docs.jax.dev/en/latest/profiling.html) supports
  `jax.profiler.trace`, Perfetto, and XProf. Because JAX dispatch is
  asynchronous, a timed region must wait for results (for example with
  `.block_until_ready()`).
- The [persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
  can reduce repeated compilation across processes. Its documentation includes
  a cache-security warning; it changes startup economics, not kernel semantics.

### Version/API caveats

Pallas examples age quickly. As one concrete marker, the current changelog says
JAX 0.9.1 removed `pallas_call(backend=...)` in favor of backend-specific
`compiler_params`. Other releases have renamed TPU types and removed older
load/store/atomic forms. A chapter snippet should therefore record the exact
JAX and jaxlib versions used and link the changelog. Do not combine an older
`pallas_call` sample with a newer `pl.kernel` TPU sample without verifying the
whole file against one installed release.

A modest device validation ladder is:

1. Compare the ordinary JAX reference with a CPU interpret-mode kernel over
   adversarial shapes and values.
2. Compile and run on the named TPU generation and pinned JAX/jaxlib; record
   unsupported-operation, layout, and memory errors.
3. Compare device outputs across supported dtypes, boundary tile sizes, and
   masked tails.
4. Warm up, synchronize, and profile on device; separately measure end-to-end
   framework latency and the kernel region.

Only stages 1's conceptual scope was researched here; no stage was executed in
this repository.

## AWS Neuron and NKI

### Current version line and namespace

As of the source snapshot, the current AWS documentation announces Neuron SDK
2.32.0. Its [NKI migration index](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/migration/index.html)
maps NKI 0.6.0 to SDK 2.32.0, 0.5.0 to 2.31.0, 0.4.0 to 2.30.0, 0.3.0 to
2.29.0, and 0.2.0 to 2.27/2.28. New chapter code should use current imports:

```python
import nki
import nki.language as nl
import nki.isa as nisa
```

Do not copy `neuronxcc.nki.*` into new examples. That is the NKI 0.1 namespace.
Also avoid repeating the historical claim “`nl.load` and `nl.store` were
removed” as a current rule. Those high-level calls disappeared at an earlier
beta boundary but returned as experimental `nki.language` convenience APIs in
NKI 0.3. Current 2.32 documentation uses them. The durable lesson is to pin the
SDK and consult every intervening migration guide, not to infer current APIs
from an old beta sample.

Other current migration points that can make copied code misleading:

- Block dimensions (a partition dimension somewhere other than the leftmost
  dimension) have been removed.
- Since NKI 0.3, returned tensors must be allocated in `nl.shared_hbm`; string
  buffer names such as `"sbuf"` are replaced by objects such as `nl.sbuf`.
- `nki.jit(platform_target=...)` is no longer the way to choose a target; use
  `NEURON_PLATFORM_TARGET_OVERRIDE` as documented for the installed SDK.
- NKI 0.6 deprecates `nl.dynamic_range(...)` and a bare `while reg:`. Current
  runtime-driven loops use `nl.fori_loop(...)` or `nl.while_loop(...)` under
  documented state and body restrictions.

The [NKI 0.3 migration
section](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/migration/index.html#migrating-from-nki-0-2-0-to-nki-0-3-0)
is especially useful for explaining the modern namespace and `shared_hbm`
contract; the top of that same document captures the later 0.4–0.6 changes.

### Memory hierarchy

Use the [NKI memory hierarchy
overview](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/memory-hierarchy-overview.html)
and the architecture guide for the named device generation. The stable
conceptual distinction is:

- HBM holds device inputs, outputs, and large working state.
- SBUF is software-managed on-chip storage used for working tiles and is
  accessible to compute engines.
- PSUM is smaller on-chip accumulation storage associated with matrix results;
  moving a PSUM result to HBM can require an intermediate SBUF copy depending
  on the current operation/API.
- `nki.isa.dma_copy(dst=..., src=...)` expresses explicit HBM/SBUF movement;
  `nl.load` and `nl.store` are higher-level convenience APIs in current NKI.

Do not collapse SBUF and PSUM into one generic “cache,” and do not describe
them as hardware-managed caches. Avoid copying bandwidth ratios, capacities,
or tile limits without naming the generation. Even when a symbolic constant
such as `nl.tile_size.pmax` appears portable, the code still needs target-level
validation. Current API pages are often tagged for particular generations;
the presence of a platform target in `nki.jit` documentation does not prove
that every new language/ISA operation is supported on that platform.

Primary references:

- [NKI language guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/nki-language-guide.html)
- [NKI tiling overview](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/tiling-overview.html)
- [`nki.isa.dma_copy`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.dma_copy.html)
- [`nki.language.load`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.language.load.html)
- [Trainium2 architecture guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html)

### Small current-style example and its exact claim

This example is adapted from the SDK 2.32 CPU-simulator quick start. It is
useful because it names the transfers without pretending to be an optimized
RMSNorm. It is current-style source, but it was **not** run in this repository.

```python
import nki
import nki.language as nl

@nki.jit
def add_one_tile(a_hbm, b_hbm):
    a_sbuf = nl.load(a_hbm)
    b_sbuf = nl.load(b_hbm)
    sum_sbuf = nl.add(a_sbuf, b_sbuf)
    out_hbm = nl.ndarray(a_hbm.shape, dtype=a_hbm.dtype,
                         buffer=nl.shared_hbm)
    nl.store(out_hbm, value=sum_sbuf)
    return out_hbm
```

Exact support boundary: the source page demonstrates a single legal tile shape
with equal input shapes. This shortened example does not handle a tensor larger
than one legal tile, tails, broadcasting, aliasing, multiple cores, or framework
integration, and it is not a performance example. The [`nki.jit`
reference](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.jit.html)
documents dispatch for NumPy, PyTorch, and JAX argument types, but each integration
has separate setup and support limits.

For a chapter RMSNorm, show its schedule as pseudocode first:

```text
for each bounded row tile:
    DMA/load x and weight from HBM to SBUF
    accumulate sum(x*x) in the supported reduction/accumulator format
    compute inv_rms and scale the SBUF tile
    DMA/store the output tile to shared HBM
```

Then direct readers to the current [`nki.language.rms_norm` API
reference](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.language.rms_norm.html)
for the tile-level operation, or to a current library implementation for a
larger code-reading example. The API is explicitly experimental and its page is
tagged for Trn2 and Trn3. The older standalone RMSNorm tutorial still returned
by search is versioned to an older SDK and imports `neuronxcc.nki`; it is useful
history, not current copy-and-paste source. The chapter should not assert that
the pseudocode is one pass, numerically identical to a framework reference in
every dtype, or optimal on every NeuronCore generation.

### Compilation and dynamic values

The [NKI language
guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/nki-language-guide.html)
describes specialization before machine-code compilation. Tensor shape, stride,
offset, and memory-buffer information are known during specialization; tensor
values are not. Ordinary Python control flow over compile-time values can be
evaluated or unrolled in that phase.

The practical inference for serving is that “dynamic sequence length” must be
made precise. A runtime loop bound is not the same thing as an unbounded dynamic
tensor allocation. NKI 0.6 supplies `nl.fori_loop` and `nl.while_loop` for
runtime-driven control flow, with restrictions documented in the migration
guide. Storage should still be bounded, with a valid length or mask controlling
work, and the deployed shape/bucket set should be measured for specialization
and compilation behavior. Do not promise one executable for arbitrary shapes
without a primary source and a device test for that exact integration.

### Simulator, device run, and tooling are different evidence

The current [NKI CPU Simulator
guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/nki_simulator.html)
documents `nki.simulate(kernel)` as a no-hardware functional/debug path. It can
catch classes of invalid shapes, buffer misuse, dtype errors, and uninitialized
reads. It executes NKI calls eagerly and sequentially through NumPy, with no
instruction scheduling or engine parallelism, so it cannot establish hardware
latency, overlap, engine utilization, or memory bandwidth. The guide also says
that `NKI_SIMULATOR=1` does not currently support JAX; JAX callers must use the
explicit `nki.simulate()` API.

Recommended validation ladder:

1. Compare NumPy/framework reference results with `nki.simulate` across dtypes,
   legal tile boundaries, tails, zeros, and high-magnitude inputs. Test both
   precise low-precision simulation and the documented float32-isolation mode
   when diagnosing numeric error.
2. Compile and run on the named Trn/Inf target with the pinned Neuron SDK. Test
   the framework integration as well as any standalone call path.
3. Benchmark both the warmed kernel and end-to-end model path. Do not treat
   compilation time as steady-state execution or a bare kernel as request
   latency.
4. Capture a device profile and inspect transfers, compute-engine use, stalls,
   and on-chip allocation in the current Neuron tooling.

Tooling references and limits:

- [`nki.benchmark`](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/nki/api/generated/nki.benchmark.html)
  requires a Trn/Inf instance with a v2-or-later NeuronDevice and Neuron tools.
  The page reached from `latest` currently redirects to older versioned docs,
  so verify that decorator against the installed SDK before publishing syntax.
- [`nki.profile`](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/nki/api/generated/nki.profile.html)
  likewise has older versioned documentation. Prefer the current [Neuron
  Explorer capture workflow](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/how-to-profile-workload.html)
  for a 2.32-oriented narrative, and keep any decorator code explicitly
  versioned.
- [Neuron Explorer](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/index.html)
  exposes device/system traces, source, tensor, and memory-oriented views. A
  trace is device evidence; the simulator is not a substitute for it.

### Code-reading references

Use primary, maintained examples and record the revision or SDK version read:

- [AWS NKI samples](https://github.com/aws-neuron/nki-samples) contains
  tutorials and contributed kernels. Its own README distinguishes educational
  examples from contributed code and warns that contributed kernels do not have
  the same public-SDK compatibility guarantee; do not present repository
  presence as a support promise.
- [NKI Library documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/index.html)
  links current open-source library code and its kernel utilities.
- [`nki.language.rms_norm`](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.language.rms_norm.html)
  is the current tile-level API reference; [RMSNorm-Quant in NKI
  Library](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/rmsnorm-quant.html)
  is a current but materially different fused-and-quantized implementation.
  Neither should be described as a generic drop-in framework RMSNorm.
- The [2.32 migration index](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/migration/index.html)
  and current component release notes should be checked immediately before
  freezing a runnable example; the search-visible “NKI Known Issues” page is an
  older beta document and should not be cited as the current support matrix.

## Compact comparison for the chapter

| Question | Pallas/TPU | NKI/Neuron |
|---|---|---|
| Large device store | HBM | HBM / shared HBM for returned tensors |
| On-chip working store | VMEM; SMEM for scalar data | SBUF; PSUM for matrix accumulation |
| CPU correctness aid | `pallas_call(..., interpret=True)`; separate TPU-specific interpreter | `nki.simulate(kernel)` |
| What CPU aid does not prove | TPU lowering, legality, layout, memory fit, timing | Neuron compilation, scheduling, overlap, timing |
| Runtime irregularity | Concrete JIT shapes can recompile; use bounded windows/buckets deliberately | Specialization knows tensor metadata; use bounded storage and current runtime-loop APIs |
| Version warning | Experimental API; `backend=` was removed in current changelog | Namespace and APIs changed across 0.1–0.6; pair NKI with SDK version |

This table is conceptual. It is not an assertion that similarly named stores,
interpreters, or compilation stages are interchangeable.

## Claims to reject during review

- “`interpret=True` runs the Pallas kernel on TPU.”
- “Passing interpret/simulator tests means the kernel is supported or fast on
  hardware.”
- “HBM is directly computed on like registers/on-chip SRAM.”
- “Pallas is one portable optimized kernel language across TPU and GPU.”
- “JAX shape polymorphism means one already-compiled device executable accepts
  every sequence length without recompilation.”
- “NKI examples should import `neuronxcc.nki`.”
- “`nl.load`/`nl.store` are removed from current NKI.”
- “A runtime loop makes NKI tensor allocation unbounded or fully dynamic.”
- “A capacity, bandwidth ratio, tile size, or accumulator behavior applies to
  every TPU or Neuron generation.”
- “A CPU simulator or bare-kernel timing establishes model-serving latency.”

## Validation status

The URLs above were checked as primary documentation on 2026-09-13. The Pallas
API and NKI/SDK migration claims were checked against their current changelogs
or migration index. No local package imports, compilation, simulator run, device
run, or performance measurement was performed. The two code blocks are scoped
as, respectively, pseudocode and a shortened current-documentation example; the
NKI block still requires execution under the pinned SDK before publication.
