# Part I: independent-model review and fixes

Date: September 8, 2026. Starting source: `329dc21`.

The user requested a different-model review followed by fixes. GPT-5.5 reviewed
all seven chapters of `manuscript/01_foundations.md`, the sequence/attention
references, and their tests, read-only. The editing agent separately inspected
the mathematics, implementation assumptions, and rendered pages. GPT-5.5 then
reviewed the changes and reran all tests; its second pass reported no remaining
concrete issues in the reviewed changes. This is a model-assisted technical
review, not independent human certification of the book.

## Findings and resolution

| Finding | Resolution |
| --- | --- |
| Positional equivariance omitted the causal-mask boundary | Distinguished unmasked attention, permuted structural masks, and a fixed triangular mask; added a numerical counterexample test. |
| Masked targets rejected common sentinel IDs | Validate target IDs only at supervised positions. Keep the finite-logit policy explicit; test ignored `-100`, invalid supervised IDs, and non-finite ignored logits. |
| Training example resembled incorrect PyTorch usage | Replaced it with explicit execution-contract pseudocode. Explained `[B,S,V]` class-axis handling, unscale/clip ordering, global overflow agreement, and the real scale-update API using pinned PyTorch 2.14 sources. |
| Gradient clipping was described as bounding the parameter update | Corrected the gradient/update distinction, zero-gradient case, disjoint-shard norm, and replica counting. Tested an Adam counterexample. |
| Roofline inequality used the lower-bound variable ambiguously | Defined actual time versus the model floor, specified throughput ceilings, and added a bandwidth-bound example. |
| Cost per success mixed observed ratios with expectations | Defined the empirical ratio, zero-success boundary, long-run interpretation, failed-attempt costs, and consistent mixed-workload weighting. An observed aggregate does not require homogeneous task probabilities. |
| Muon SVD explanation omitted rectangular/rank conventions | Specified a compact decomposition over nonzero directions, row/column orthonormality, and the zero-direction convention of the idealized model. |
| Scaling-law symmetry could imply equal raw counts | Derived the interior optimum, distinguished budget exponents from count ratios, and added a dimensionless counterexample with optimum `n=8, d=2`. |

## Teaching improvements

- Defined the workload symbols immediately after their equation.
- Connected cross entropy to the logit gradient and a numeric update.
- Explained Adam bias correction with first-step numbers.
- Showed why averaging rank means differs from a global supervised-token mean.
- Made gradient accumulation's communication savings conditional on deferred
  synchronization, and distinguished replay-on-overflow from discard policies.
- Added GQA tensor shapes, the correct softmax axis, and a runnable masked
  attention row that agrees across partition sizes.
- Specified multiply-add counting and causal attention constants, with a
  projection/MLP versus pair-product FLOP crossover.
- Distinguished raw unnormalized associative memory from other linear-attention
  formulations and supplied the unit-key/zero-initial-state assumptions.
- Added the queue stability condition, normalized resource/cost calculations,
  and a paired evaluation uncertainty example.
- Qualified closed-loop load testing: it is valid for sequential clients, but
  can misrepresent externally scheduled arrivals.

## Verification and scope

- `make verify`: 76 CPU tests pass, including runnable manuscript excerpts.
- Ten tests were added: one sentinel-target regression and nine foundation
  checks. These test semantics and arithmetic, not GPU performance.
- Part I link check: 20 unique URLs, zero missing or unverified responses on
  this date. The previously unavailable ACL Anthology link now resolves; the
  September 7 release record remains a historical observation.
- Full rebuilt PDF: 403 pages, 78 outline entries, and 163 external link
  annotations; structure/navigation checks pass. Part I was inspected
  at contact-sheet scale with enlarged checks of changed calculations, code,
  tables, and the transition into Part II.
- Other manuscript parts were not rewritten. Their pagination and the table of
  contents update automatically when the full book is rebuilt.

The reviewed PDF remains at `output/pdf/engineering-large-language-models.pdf`.
The previously published private September 7 release is unchanged; this review
does not itself publish a new release or change repository visibility.
