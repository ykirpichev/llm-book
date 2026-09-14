# Narrow pre-expansion review: accelerator portability CPU helper

Scope: `examples/accelerator_portability.py` and
`tests/test_accelerator_portability.py` only. This is preparation for the
accelerator-chapter expansion, not the later whole-chapter review.

## Result

The helper has a useful boundary: it is explicit that its arithmetic is a
semantic/planning aid, not an accelerator emulator or latency model. The KV
planner correctly illustrates the chapter's 4 GiB logical KV example, CP
rounding, and TP-induced KV-head replication under its stated simplified
geometry. The compile calculation also correctly returns no finite repayment
when the candidate is not faster.

The focused tests pass under the repository's CI-compatible Python 3.13:

```text
Ran 7 tests in 0.000s
OK
```

The default local `python3` is Python 3.9 and cannot import the `int | None`
annotation. This is not a repository defect because the README and CI require
Python 3.12; test with the project interpreter rather than treating the
system Python result as a portability claim.

## Actionable findings

### P1 — Reject or define overflow in the RMSNorm output path

`rmsnorm_reference` validates input finiteness and the squared sum, but its
final list comprehension can still return `inf`: a finite dominant row entry
is normalized by an RMS that permits an `O(sqrt(D))` normalized component, and
multiplying that component by a finite near-maximum `weight` can overflow a
Python float. For example, a length-4096 row with one nonzero element and
weights near `1e308` reaches this path even though the square sum is finite.

This matters because the helper describes itself as the semantic oracle for
the RMSNorm worked case. Either:

1. add an explicit supported-domain precondition that bounds scales and
   epsilon in addition to activations; or
2. compute outputs into a temporary list and raise a named `ValueError` when
   any result is non-finite, so the caller can exercise the declared fallback
   path.

Add a regression test for this case. Do not fix it by silently clipping: that
would create a new RMSNorm semantic policy requiring separate documentation.

Related documentation correction has been applied to `prep-cases.md`: finite
host inputs and a positive host epsilon do not establish finite FP32
intermediates or outputs, and a Python-float oracle is not FP32 emulation.

### P2 — Make the new executable discoverable in the example index

`examples/README.md` does not yet list `python -m
examples.accelerator_portability`, expected output, or its contract. Add one
row stating that it tests mathematical RMSNorm semantics, padded-work ratios,
simplified KV sharding/replication, and compile amortization—not CUDA/ROCm/
TPU/Neuron support or timing. This keeps the new testable artifact aligned with
the front-matter claim that runnable CPU references live in `examples/`.

### P2 — Add the portability cases the current helper intentionally omits

The present tests are a sound seed, but the chapter’s planned worked cases need
explicit coverage for:

- a `D=4097` tail with an invalid padded sentinel, proving the logical-width
  denominator and no tail contribution;
- same logical values through a strided/noncontiguous representation;
- explicit epsilon-cast/range policy (the Python oracle accepts values that a
  target FP32 conversion can round to zero);
- a rank-fit planner with weights, active/reusable KV, workspace,
  communication, graph/compile pool, and reserve per rank; the existing helper
  models KV only and should not be portrayed as a full placement validator;
- a cache-ABI/repack handoff planner and event-accounting fixture before any
  cross-vendor handoff discussion claims a break-even.

Keep those additions CPU-only and label every duration/bandwidth as a caller
input or measurement. They should validate arithmetic and acceptance logic,
not emulate a backend or claim hardware validation.
