# Round 3 code and learning-integration review

**Scope.** Read-only review of the current executable-example index, Chapter
36 companion material, Part IX capstone, and Part V's attention-merge
reference. This is not hardware validation. The optional Triton and HIP paths
were inspected as source and documentation only; no GPU, TPU, Neuron, HIP, or
Triton compilation/execution is claimed here.

## Verdict

The new material has a sound evidence boundary. The optional companion README,
the Chapter 36 evidence ladder, and the unfilled acceptance record consistently
separate CPU semantics, device-operator checks, integrated-engine evidence,
and deployment decisions. The Part IX route to the root **Verified teaching
examples** index is now navigable, and the Chapter 36/Part V distinction
between abstract independent CP and PCP/DCP is correct in prose.

Four small fixes would make that distinction and the reader's runnable path
survive contact with the actual reference code and current repository state.

## Findings

### P1 — Put the independent-CP caveat beside the planner API, not only in Chapter 36

**Location:** `examples/accelerator_portability.py:79-98`; compare
`manuscript/04_cuda.md:1763,1791-1793` and
`manuscript/05_distributed.md:831-897`.

The chapter now carefully calls `CP=2` an *abstract independent context-axis
capacity model*, distinct from PCP and DCP. The executable helper instead
exposes an unqualified `cp` argument and calls it “equal padded CP shards.” A
reader who opens the helper from Chapter 36 can again treat `cp=2` as a
portable engine PCP/DCP setting, precisely the inference the manuscript has
just prevented.

**Smallest fix:** retain the API and arithmetic, but add a first-sentence
docstring/parameter comment: `cp` is the chapter's abstract independent
capacity axis; it adds ranks in this model and is neither PCP nor DCP nor an
engine flag. Optionally name the displayed examples `abstract_cp` while keeping
the callable parameter stable. Do not extend the helper into an engine-layout
planner.

### P1 — Exercise the empty-shard branch of the partition merge that Part V teaches

**Location:** `manuscript/05_distributed.md:833-841`,
`examples/attention.py:30-47`, and `tests/test_examples.py:20-34`.

Part V correctly makes an empty or fully masked history shard a central DCP
edge case: it must contribute zero state without evaluating
`exp(-inf - -inf)`. The CPU reference implements this safely by filtering
zero-mass partial states, but its partition-merge test has every key visible.
The only mask tests cover a one-shard visible-key case and an all-masked row;
neither tests a merge containing both an empty shard and a nonempty shard.

**Smallest fix:** add one dependency-free fixture with irregular partitions
and a mixed mask that makes at least one shard entirely masked while other
shards remain visible. Compare it with an independently computed dense result
over just the allowed keys, and retain the all-masked-row convention test.
Add a brief Part V link to this *semantic, single-head* reference so readers do
not mistake it for a distributed DCP runtime test. This directly turns the
chapter's most error-prone merge condition into an executable learning check.

### P1 — Reconcile the public “latest” test count with the current suite

**Location:** `README.md:15,81-84`.

The README says the latest five-pass revision has 95 passing CPU tests, while
the current full suite completes successfully with 103 tests. The 95-test
number may be valid for the historical five-pass snapshot, but “latest” next
to current build instructions reads as a current release claim. It also makes
the new seven-test portability addition harder for a reader to place.

**Smallest fix:** either label 95 explicitly as the recorded five-pass
snapshot and state the separately verified current full-suite result, or update
the revision/review metadata and count together after the release validation.
Avoid calling all 103 tests device validation; the suite remains CPU/source and
document/build checking.

### P2 — Give the capstone's “partition tests” one exact command

**Location:** `manuscript/09_appendices.md:38-45` and
`examples/README.md:6-21`.

Part IX directs the learner to “Run the attention reference and partition
tests,” but the example index supplies only `python -m examples.attention`,
which prints the simple `[3.0]` demonstration. The actual partition assertions
live in `AttentionTests`; a learner can infer `make test`, but the capstone's
otherwise concrete sequence loses its first reproducible command here.

**Smallest fix:** add `python -m unittest tests.test_examples.AttentionTests
-v` to the attention row (or spell that command in the Part IX table), while
keeping `make test` as the full-suite route. This requires no new test or
hardware dependency.

## Verified non-findings

- The optional accelerator README and Chapter 36 are professionally honest
  about evidence: target-only commands, source-only HIP status, planned versus
  passed fixtures, and required engine/deployment evidence are consistently
separated. No hardware result is implied by the 103 CPU-suite passes.
- The acceptance record is correctly an unfilled decision template, linked
  from both the root teaching-examples index and the Part IX capstone route; it
  does not imply a vendor benchmark or use a nonportable local/PDF link.
- Chapter 36's independent capacity calculation, its nested DCP comparison,
  and Part V's PCP/DCP ownership table now agree: independent CP adds ranks;
  no-PCP DCP reuses rank space. The requested fixes above concern the helper's
  nearby wording and the executable merge edge case, not a prose contradiction.

## Checks run

- `git diff --check` completed without whitespace errors.
- The bundled Python runtime completed `python -m unittest discover -s tests
  -v`: **103 tests passed**.
- No optional accelerator compiler, device runtime, or target-only command was
  run.
