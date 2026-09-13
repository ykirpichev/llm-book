# Accelerator chapter — round 1 integration review

**Scope.** Independent review of `manuscript/04_cuda.md` lines 1524–1857,
`examples/accelerator_portability.py`,
`tests/test_accelerator_portability.py`, optional
`examples/accelerators/triton_rmsnorm.py` and its README, and the four new
figure functions in `src/book_figures.py`. This is a CPU/source review only;
I did not run a GPU, TPU, Neuron, HIP, JAX, Triton compilation, or a serving
engine.

## Verdict

The chapter is unusually disciplined about its evidence boundary. The CPU
oracle is correctly described as higher-precision host semantics rather than
FP32 emulation; the optional Triton code is explicitly source-reviewed, and
the placement and compile calculations are consistently called planning
models rather than engine guarantees. The model/engine distinction, readiness
path, and evidence ladder make this materially more deployable than a
device-comparison survey. No P0 correctness or safety issue was found in the
reviewed CPU code.

One P1 clarification is needed before treating the transfer example as a
deployment decision aid. The remaining items are P2 hardening/discoverability
work, appropriate before any claim of on-target validation but not blockers
for this chapter's present CPU-only evidence status.

## Findings

### P1 — Define the ownership and scope of the 4 GiB handoff payload

**Location:** `manuscript/04_cuda.md:1811-1813`.

The preceding KV calculation defines 4 GiB as the cache for *eight sequences*
at 4096 tokens (`:1740-1741`), but the transfer example simply says a
"handoff moves 4 GiB." A reader can easily treat that as the normal cache
payload for one prefill-to-decode request. In a disaggregated system, the
handoff unit is normally a request (or an explicitly named batch); its bytes
depend on the actual committed prefix length, batch membership, layer/head
geometry, dtype, and cache representation. This ambiguity can turn a useful
worst-case example into an admission/latency error.

**Smallest fix:** say that 4 GiB is deliberately the earlier `B=8`,
`S=4096` *whole-live-batch* KV state, not a per-request universal number, and
give the payload expression (or a one-request 2000-token contrast) before the
80 ms calculation. Require the decision record to state whether handoff is
per request, microbatch, or pool migration. The existing ABI caveat remains
right; this makes its latency calculation operationally well-scoped.

### P2 — Make the CPU plan’s aggregate field self-identifying

**Location:** `examples/accelerator_portability.py:67-94`,
`manuscript/04_cuda.md:1753-1768`.

`aggregate_bytes` includes `tp * cp` padded shard allocation. It therefore
means aggregate *physical modeled KV allocation*, not logical cache bytes
and not total executable memory. The prose and diagram explain this well, but
the public data-field name and `main()` label (“MiB/aggregate”) are easy to
reuse incorrectly in a capacity worksheet.

**Smallest fix:** rename it to `aggregate_modeled_kv_bytes` (or make the
dataclass/docstring and printed label use that full meaning). Preserve the
current arithmetic and tests; it is correct under its stated equal-padded CP
assumption. This is an interface clarity change, not an engine-support claim.

### P2 — Turn the stated Triton pre-adoption checks into an executable target-suite checklist

**Location:** `examples/accelerators/triton_rmsnorm.py:57-84`,
`examples/accelerators/README.md:23-28`,
`manuscript/04_cuda.md:1598-1600`.

The positive on-device sweep is thoughtfully shaped (tails, zero rows,
strides, dtype-dependent tolerances, and an FP32 framework expression), but
it does not exercise the wrapper’s rejection contract or ordering/resource
behavior. The README already says to add API rejection and concurrent-stream
tests before adoption, so no current validation claim is false. However, a
future reader could run the script, see `PASS`, and miss that its coverage is
only forward numerical smoke testing on the default execution path.

**Smallest fix:** next to the command, enumerate the required target-suite
gates as unchecked/required evidence: invalid rank/device/dtype/epsilon and
`requires_grad` rejection; at least one non-default stream or explicit event
dependency case; error/synchronization boundary; and a bounded adversarial
value-range comparison. Keep them separate from the CPU suite and continue
to record device, driver/runtime, PyTorch, Triton, and skipped BF16 status.

### P2 — Add a top-level discovery link for the CPU portability exercise

**Location:** `examples/accelerators/README.md:38-49` and the chapter’s CPU
example at `manuscript/04_cuda.md:1558-1567`.

The runnable CPU oracle/planner lives at `examples/accelerator_portability.py`,
but its discoverability is nested under the optional accelerator README. It
is valuable without an accelerator installation and should be reachable from
the repository’s normal runnable-example index as well.

**Smallest fix:** add one table/list entry to the top-level examples index
that names the CPU oracle and planning exercises, with their unittest command
and an explicit “not device emulation or hardware validation” label.

## Checks performed

- `/opt/miniconda3/bin/python3.13 -m unittest tests.test_accelerator_portability -v`:
  **7/7 passed**.
- `/opt/miniconda3/bin/python3.13 -m py_compile examples/accelerator_portability.py examples/accelerators/triton_rmsnorm.py`:
  **passed**.
- `git diff --check`: **passed**.
- Full `make test` reached the new 7 tests and 76 other tests successfully,
  but the local Python 3.13 environment lacks `pypdf` and `reportlab`, so
  `test_audit_repository` and `test_build_book` could not import. That is an
  environment/dependency limitation, not evidence that the new chapter or
  figures fail; I did not claim PDF validation.

## Evidence boundary retained

The chapter should retain its current labels: CPU tests establish only CPU
semantics/planning; successful Python parsing is not a device compile;
on-target compilation/correctness/performance and engine behavior still need
pinned-stack evidence. The visual functions accurately reinforce that
boundary: the KV figure says it is KV-only and assumption-dependent, and the
compile lifecycle makes the miss policy and compatible artifact identity
explicit.
