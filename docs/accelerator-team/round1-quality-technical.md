# Round 1 technical quality check

Reviewer: Astra technical QC. September 13, 2026. Reviewed the integrated
Packages 1–5 working tree, not the campaign baseline alone. This is a bounded
quality-check stage, not completion of the three-hour campaign.

## Verdict

**No P0/P1 technical blocker found.** The new Pallas and NKI details are correct
against the checked primary documentation, the fragment figure represents the
right information dependency without inventing a device barrier, and the
handoff/cache examples preserve their assumptions. The requested denominator
and repayment-overflow corrections work. All 103 CPU tests pass.

Two localized P2 items remain in the snapshot below: an extreme positive-cost
repayment underflow, and the already-author-identified HIP kernel-name mismatch.
Resolve these before recording the round as clean. The HIP header/discovery
note also needs its already-planned fixture clarification. None requires new
hardware evidence or a broader chapter rewrite.

## Findings

### Q1 — P2: positive compile cost can still produce zero repayment calls

Location: `examples/accelerator_portability.py:116–119`,
`compile_break_even`; test extension belongs in
`tests/test_accelerator_portability.py:test_compilation_amortization`.

The new finite-quotient check correctly rejects overflow, but a positive
quotient can underflow to zero before `ceil`. Reproduced under Python 3.12.14:

```python
compile_break_even(1e-308, 1e308, 1e307)
# Actual: 0
# Required: 1 call (or an explicit documented range rejection)
```

The mathematical quotient is approximately `1.1111e-616`. It is below one, but
positive, so zero calls cannot repay a positive compilation cost. This is a
low-frequency range-contract issue, not a defect in the printed 960-call
example. A small correct fix is to return at least one call when compilation
cost is positive and the candidate is faster, while preserving zero for truly
zero compilation cost. Alternatively, explicitly reject positive-quotient
underflow. Add this regression alongside the overflow case; retain `None` for
the existing non-improving-candidate meaning.

### Q2 — P2: printed HIP launch names a different kernel from its companion

Locations: `manuscript/04_cuda.md:1638` and
`examples/accelerators/hip_rmsnorm.cpp:75`.

The chapter launches `rmsnorm_f32`, but the supplied source defines and launches
`rmsnorm_rows`. The excerpt is source-only, so it need not be independently
compilable, but its stated companion should provide the named kernel. Align the
name and recheck argument order. The author independently identified this
before my report; it is recorded here as an outstanding integration check, not
as a newly discovered algorithmic defect.

Related minor documentation cleanup: the optional README says the HIP header
describes its supported input contract, but the reviewed header contains only
the source-only status. Add the fixed positive-width FP32 fixture/stride/range
contract there, or change that README pointer. Do not call the ordinary
`hipMalloc` allocations asynchronous allocations.

## Source and mathematical checks that pass

### Pallas windows and memory accounting

At `manuscript/04_cuda.md:1673–1699`, input/output `(8,4096)` windows over
`(128,4096)` arrays use correct blocked coordinates. Program 3 maps to rows
24–31, and 16 programs cover all 128 rows. The `(1,4096)` weight window matches
the full weight-array dimensions, so its leading dimension is permitted by the
full-dimension exception; it need not be divisible by eight. Both major window
dimensions satisfy the applicable two-dimensional TPU rules.

The FP32 input footprint is 131,072 bytes, or 128 KiB. An equal output and
16 KiB logical weight row are correctly separated from temporary/layout and
pipeline-buffer costs. The chapter does not equate those logical counts with
compiler-proven VMEM allocation. The warning about unspecified padded inputs
is also correct. Sources: [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html),
[TPU pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html).

### NKI API and complete-row reduction

At `manuscript/04_cuda.md:1731–1742`, the same-shaped FP32 `(128,512)` input and
weight tiles in SBUF match the official example. `axis=1`, `n=512`, and the
`compute_dtype`/`dtype` keyword names are documented. Repeating the learned
row weights across partitions preserves RMSNorm semantics; the text does not
claim that materialization is optimal. The missing-fragments explanation is
correct: substituting a global N does not supply the omitted squared values.

I independently retrieved the **versioned** [SDK 2.32 language source](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_modules/nki/language.html#rms_norm),
which confirms the signature and example, as well as the
[SDK/NKI version mapping](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/migration/index.html).
The generated API page still produced a web-retrieval error here; that is not
evidence of a broken URL, and the author reports successful HTTP verification.

The source of `src/book_figures.py:rmsnorm_fragments` has both fragments feed
the combine/inverse node, then distributes that inverse to both scaling paths.
The caption/footer explicitly require a supported multi-stage implementation
and retaining/reloading values. No cross-program barrier or PSUM requirement is
invented. I checked the source dependency graph, not a rendered proof.

### Access, source repair, numerical contracts and deployment examples

- The TorchNeuron qualifier now agrees with the checked [closed-beta access
  note](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/frameworks/torch/pytorch-native-overview.html).
  The replacement [Neuron Explorer capture guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/tools/neuron-explorer/how-to-profile-workload.html)
  is valid and relevant.
- `rmsnorm_reference([1e154], [1.0], 1e308)` is covered by the new normalization
  argument range rejection. `compile_break_even(1e308, 1e-308, 5e-309)` is
  covered by the new repayment-overflow rejection. Ordinary examples, true
  zero compile cost, and non-improving candidates retain their tested behavior.
- The renamed `aggregate_modeled_kv_bytes` and self-contained printed KV
  calculation distinguish logical versus modeled physical KV. The three
  layouts produce `(1024,4096)`, `(512,8192)`, `(512,4096)` MiB per-rank/aggregate
  pairs. The printed calculation explicitly limits itself to those layouts.
- The handoff now names the eight-request, 4096-token whole batch behind
  4 GiB. The separate request exercise computes 262,144,000 bytes = 250 MiB;
  at 25 GiB/s, transfer is 9.765625 ms, total serialized cost 13.765625 ms,
  leaving 1.234375 ms of the stated 15 ms budget. It does not approve an
  unmeasured transfer path or confuse this budget with token-gap SLOs.
- Two buckets × two dtypes × two static layouts gives eight hypothetical
  variants. Calls A/B share the explicitly assumed runtime-mask variant;
  C/D change specialization dimensions. These are examples, not a universal
  cache-key claim. The distinctions agree with the [persistent-cache guide](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
  and [shape-polymorphic export documentation](https://docs.jax.dev/en/latest/export/shape_poly.html).

## HIP companion source check

Read the complete available HIP source. Its fixed 256-thread reduction keeps
all threads at every barrier; int64 input/output offsets and the true-width
divisor agree with the chapter. The output pass uses the full-row inverse and
the independent weight stride. The bounded fixtures include zero and
mixed-sign rows and tail widths 3, 129 and 4097. The CPU comparator uses a
higher-precision expression with explicit tolerances, not claimed FP32
instruction emulation.

The successful path submits copies, kernel and result copy to the same
non-default stream and synchronizes before host comparison and deallocation.
The host vectors are pageable: calling `hipMemcpyAsync` does not make this an
overlap demonstration. AMD's [memory API reference](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html)
documents synchronous copies for unpinned host buffers and implicit device
synchronization in traditional `hipFree`. Thus the observed RAII destruction
order is not, by itself, a confirmed lifetime bug for these `hipMalloc`
allocations. A future async-allocation conversion needs a new lifetime audit.
This is a fail-fast correctness fixture, not a tested recovery design.

## Evidence and scope limits

- Read the entire current Chapter 36, CPU planner and tests, optional Triton
  source/README, new HIP companion, and fragment-figure source.
- Bundled Python 3.12.14: targeted portability/manuscript suite **12/12 passed**;
  full `unittest discover -s tests` **103/103 passed**. This also executes the
  manuscript's explicitly runnable CPU blocks and checks per-fence status labels.
- Independently recalculated Pallas row/byte counts, variant count and transfer
  arithmetic. Reproduced Q1 with a high-precision decimal cross-check.
- `git diff --check` passed. No accelerator framework, compiler, simulator or
  hardware execution was performed. The author owns current PDF rendering and
  visual inspection; this report makes no new rendered-page claim.
- No shared implementation/manuscript files were edited. Snapshot SHA-256
  prefixes: manuscript `a74471d434e3`; CPU planner `7b8087d4d65a`; tests
  `a70958e605af`; HIP source `339429077502`; figures `18922d3a21ff`.

After Q1/Q2 and the small HIP contract note are integrated, recheck those exact
changes and rerun the CPU suite. The substantive Package 3 technical gate is
clear; device compilation/performance and final visual QC remain separate
evidence stages, not implied by this report.

## Closure verification — September 13, 2026, 22:03 UTC

The author integrated the fixes while the initial report was being written.
The historical findings above are retained; their current status is **closed**.

- **Q1 closed:** positive compilation cost now returns at least one call even
  when the quotient underflows. The exact reproducer has a regression assertion
  and returns 1. Independent checks still return 0 for zero compilation cost,
  960 for the ordinary example, and `None` for a slower candidate; the overflow
  rejection test remains in place.
- **Q2 closed:** the HIP companion definition and launch now both use
  `rmsnorm_f32`, matching the chapter. Pointer/width/stride/epsilon argument
  order matches. The header now declares the fixed FP32, positive-stride,
  bounded-value, epsilon and 256-thread fixture contract, and identifies
  pageable host staging without claiming overlap. The cleanup note correctly
  identifies traditional `hipMalloc`/`hipFree`, not async allocation.
- Re-ran the complete suite with bundled Python 3.12.14: **103/103 passed**.
  `git diff --check` passed. These are CPU/source checks only; no HIP compile,
  accelerator execution, or new visual inspection is claimed.

The technical round-1 gate is now **clear with no open finding from this
review**. This does not complete the overall campaign or replace its separate
visual/device-evidence gates.

For source consistency, the author can pin the remaining `latest` NKI language
source link at `manuscript/04_cuda.md:1742` to the independently retrieved
[SDK 2.32 source reference](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_modules/nki/language.html#rms_norm).
That versioned page contains the checked signature, computation/output dtype
parameters, and same-shaped `(128,512)` FP32 SBUF example. Pinning is appropriate
because the prose explicitly identifies SDK 2.32/NKI 0.6; this is maintenance
consistency, not an additional technical blocker.

Closure snapshot SHA-256 prefixes: manuscript `42d815e1d1ef`; CPU planner
`0e8afa2b3d71`; tests `2b24124d31a9`; HIP source `9ba8b5ed7e06`.
