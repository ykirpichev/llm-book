# Round 3 author-side checks

These are supplementary checks while the three independent reviewers work,
not a replacement for their critiques or the proposer stage.

## Attention arithmetic can masquerade as an empty shard

The current CPU reference accepts finite inputs but does not reject
nonrepresentable intermediate arithmetic. Two actual local reproductions:

- `attention([1e308], [[1e308]], [[1.0]])` returns `[0.0]`.
- `attention([0.0], [[1.0], [2.0]], [[1.7e308], [1.7e308]])`
  returns `[inf]`.

The first case creates an infinite score and a NaN shifted mass. Filtering
states with `l > 0` then treats the invalid state as empty, conflating a
numerical failure with the documented all-masked-zero convention. The second
overflows a weighted sum even though the normalized mathematical result fits.

Ask the proposer to consider a bounded teaching-reference fix: require scores,
partial weighted sums, and merged arithmetic to fit Python float, reject
out-of-range calculations with a named error, and test both partial and
merge-only overflow. Preserve the legitimate all-masked-zero convention and
the existing independent dense oracle; do not turn this into a new production
attention implementation. Astra is independently verifying these cases.

## The kernel-language boundary already has a place for CuTe

Part IV's existing **Choose the implementation boundary** paragraph names
CuTe DSL beside Triton but gives no direct reference for the former. The
[official CUTLASS Python overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
was retrieved September 13, 2026. It describes Python kernel authoring through
the CUDA toolchain, with explicit layouts, tiled operations, and pipelines.
A linked phrase at the existing mention would be sufficient; a new vendor or
DSL survey would not improve this chapter's argument.

TileLang's current official documentation was also inspected as a possible
omission. No change is proposed merely to name another available DSL. The
chapter already states its representative scope and teaches the selection
boundary rather than promising exhaustive tool coverage.

## Checkpoint evidence

Round 2 was saved locally as `973d577`. The HEAD-only repository audit scanned
85 blobs and flagged none. This says nothing about uncommitted round-3 files
or a fresh historical publication audit. No remote push or release occurred.
