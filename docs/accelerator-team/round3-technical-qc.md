# Round 3 — independent technical quality check

Reviewed the integrated Packages 1–4 on September 13, 2026, including Terra's
final expanded mixed-mask regression. Review only: no manuscript, code, or test
edits. The earlier technical review's local-machine links were converted to
repository-relative path-and-line references without changing its findings.

## Verdict

Pass. Round-3 T1–T4 are closed. No new actionable P0/P1/P2 technical finding in
the inspected changes. The optional NixlConnector scale illustration was not
required for closure and was appropriately left out of this bounded revision.

## Confirmed acceptance

### CPU numerical reference and abstract CP contract

`examples/attention.py:33–89` now rejects nonfinite visible scores before
softmax, rejects nonfinite partial and merged numerators, checks positive
finite mass/denominator, and validates final outputs. A true zero-mass identity
is checked explicitly instead of using a comparison that silently discards NaN.
The module contract clearly says that finite inputs alone are insufficient.
Rejecting a mathematically representable result when the chosen intermediate
overflows is intentional under this clarity-first reference's stated range;
it is not a claim of an unrestricted robust attention algorithm.

`tests/test_examples.py:23–65` covers score overflow, dense partial-numerator
overflow, sharded merge-numerator overflow, huge masked score inputs, and a
mixed valid/masked-shard identity. The final seven-key, two-output fixture uses
an independent uniform-attention formula and checks shard sizes 1, 2, 3, and 4.
The existing nonuniform dense-formula comparisons remain intact.

Additional direct CPU probes confirmed:

- a masked huge value and overflowing masked score still produce `[0.0]`;
- an overflowing masked shard plus a valid shard returns the valid `[4.0]`;
- a representable single-key value of `1.7e308` is preserved;
- a finite score span from `-1.7e308` to `1.7e308` still returns the dominant
  key's value, allowing the negligible exponential to underflow to zero.

Empty overall K/V remains invalid. The code does not attempt to certify a
distributed implementation. The revised `kv_shard_plan` docstring at
`examples/accelerator_portability.py:82` explicitly defines an independent
capacity axis that adds ranks, not PCP, DCP, or a portable engine flag; its
existing arithmetic is unchanged.

### State and phase terminology

`manuscript/04_cuda.md:1207–1209` limits the token-growing KV formula to the
running full-history GQA model, distinguishes element payload from physical
allocation, and connects hybrid state to the same accepted token boundary.
`04_cuda.md:1296` now says reduced stored bytes and potential read traffic,
not reduced capacity or a guaranteed latency improvement.

`manuscript/05_distributed.md:895–901` no longer excludes backend-specific DCP
prefill modes. It names separate group/weight plans and explains that a common
flag does not establish the same prefill/decode geometry. This closes the
SDK-2.32 source discrepancy recorded in T3 without changing the valid no-PCP
decode example. The nearby CPU-reference pointer explicitly disclaims
distributed execution and DCP performance evidence.

### Changed watermark-lag exercise

Read the complete event-time design, changed exercise, and worked answer in
`manuscript/06_coding_and_design.md`. The nominal model is locally defined as
`ceil((window_width + allowed_lateness + watermark_lag) / window_slide)`;
the answer does not claim an exact runtime peak from this approximation.

Independent CPU recomputation gives:

| Quantity | Result |
| --- | ---: |
| Nominal windows, `ceil((5 + 20 + 8) / 5)` | 7 |
| Seven-window dense sketch payload | 99.264830 GiB |
| Six-window dense sketch payload | 85.084140 GiB |
| One-window payload deficit | 14.180690 GiB |

Thus the printed 99.3, 85.1, and 14.2 GiB values are correctly rounded under the
declared model. Candidates, deduplication, snapshots, and object overhead are
explicitly excluded; boundary alignment is not claimed to be solved exactly.

The policy is also correct: a retention cap is not evidence that event-time
progress advanced. The [Apache Beam programming guide](https://beam.apache.org/documentation/programming-guide/)
places normal window-state expiry after the watermark crosses the window end
plus allowed lateness. The exercise therefore properly distinguishes
capacity/tiering/backpressure responses from an explicitly incomplete eviction.
Its caveat that buffering cannot free already retained state avoids a false
recovery promise. Completeness flags and revisions are checkpointed with their
source-offset boundary; the unchanged main-body design still supplies the
transactional/idempotent sink and replay rules.

### Lead and navigation caveats

The revised FlashAttention lead retains dense-attention semantics. Its next
paragraph still defines exactness as no sparsity/low-rank approximation while
allowing floating-point-order and precision differences. The data-lineage lead
does not replace the detailed source-rights, identity, descendant tracing,
release, and model-remediation requirements that remain in the body. It does
not claim that deleting a raw object undoes a checkpoint's learning. The other
changed leads/seams introduce no new technical promises.

## Validation evidence and limits

After the final mixed-mask test edit, the complete bundled-Python command
`python -m unittest discover -s tests` passed **108 tests**. `git diff --check`
passed. The new decision-index regression also passed and validates the named
chapter/part destinations against the manuscript.

This QC covers source correctness, the scoped prose changes, and CPU behavior.
It does not claim accelerator compilation/execution, distributed runtime
testing, device profiling, a new whole-book audit, or rendered-page inspection.
The three-hour campaign remains active; this is round-3 technical closure only.
