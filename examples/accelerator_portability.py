"""CPU oracles and planning arithmetic for the accelerator-portability chapter.

These functions do not emulate an accelerator, predict a measured latency, or
validate a compiler. Every plan has intentionally narrow, explicit assumptions.
"""
from dataclasses import dataclass
import math
from collections.abc import Sequence


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def rmsnorm_reference(row: Sequence[float], weight: Sequence[float],
                      epsilon: float = 1e-6) -> list[float]:
    """Higher-precision semantic oracle, not a bitwise FP32/GPU reference.

    Normalize one nonempty row using its actual length, not a padded tile size.
    Inputs must be finite; squared sum, normalization argument, and outputs
    must fit Python float.
    Out-of-range results raise ValueError rather than silently becoming infinity.
    """
    if len(row) == 0 or len(row) != len(weight):
        raise ValueError("one nonempty row and matching weights required")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("finite positive epsilon required")
    if not all(math.isfinite(v) for v in (*row, *weight)):
        raise ValueError("finite row and weights required")
    try:
        square_sum = math.fsum(float(v) * float(v) for v in row)
    except OverflowError as error:
        raise ValueError("squared sum exceeds reference range") from error
    if not math.isfinite(square_sum):
        raise ValueError("squared sum exceeds reference range")
    normalization_argument = square_sum / len(row) + epsilon
    if not math.isfinite(normalization_argument):
        raise ValueError("normalization argument exceeds reference range")
    inverse_rms = 1 / math.sqrt(normalization_argument)
    result = [float(v) * inverse_rms * float(w) for v, w in zip(row, weight)]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("output exceeds reference range")
    return result


@dataclass(frozen=True)
class PaddingWork:
    actual_tokens: int
    bucket_tokens: int
    linear_ratio: float
    dense_pair_ratio: float


def padding_work(tokens: int, buckets: Sequence[int]) -> PaddingWork:
    """Smallest fitting bucket; ratios assume fully padded dense execution."""
    _positive_int("tokens", tokens)
    if not buckets:
        raise ValueError("at least one bucket required")
    for bucket in buckets:
        _positive_int("bucket", bucket)
    fits = [bucket for bucket in buckets if bucket >= tokens]
    if not fits:
        raise ValueError("request exceeds all buckets")
    chosen = min(fits)
    ratio = chosen / tokens
    return PaddingWork(tokens, chosen, ratio, ratio * ratio)


@dataclass(frozen=True)
class KVShardPlan:
    heads_per_rank: int
    tokens_per_rank: int
    kv_replication: int
    bytes_per_rank: int
    aggregate_modeled_kv_bytes: int


def kv_shard_plan(*, layers: int, batch: int, context: int, kv_heads: int,
                  head_dim: int, element_bytes: int, tp: int = 1,
                  cp: int = 1) -> KVShardPlan:
    """Uniform full-history GQA with an abstract independent capacity axis.

    ``cp`` adds ranks in this teaching model; it is neither PCP nor DCP nor a
    portable engine flag. The model has no pipeline partition and uses equal
    padded context shards.

    TP partitions whole KV heads when possible; TP beyond KV-head count
    replicates each head. Only exact head divisibility is modeled. An engine
    may use another layout or reject this plan entirely.
    """
    for name, value in locals().copy().items():
        _positive_int(name, value)
    if kv_heads >= tp and kv_heads % tp == 0:
        local_heads, replication = kv_heads // tp, 1
    elif tp > kv_heads and tp % kv_heads == 0:
        local_heads, replication = 1, tp // kv_heads
    else:
        raise ValueError("TP must divide KV heads or be an exact multiple")
    local_tokens = (context + cp - 1) // cp
    local = 2 * layers * batch * local_tokens * local_heads * head_dim * element_bytes
    return KVShardPlan(local_heads, local_tokens, replication, local, local * tp * cp)


def compile_break_even(compile_seconds: float, baseline_seconds: float,
                       candidate_seconds: float) -> int | None:
    """Calls needed to repay compilation at fixed serial per-call times.

    Excludes caching, overlap, queues, varying shapes, and quality differences.
    The repayment quotient must fit Python float; otherwise raises ValueError.
    None means no finite repayment for a candidate that is not faster.
    """
    if not all(math.isfinite(v) for v in
               (compile_seconds, baseline_seconds, candidate_seconds)):
        raise ValueError("finite durations required")
    if compile_seconds < 0 or baseline_seconds <= 0 or candidate_seconds <= 0:
        raise ValueError("nonnegative compilation and positive execution times required")
    if candidate_seconds >= baseline_seconds:
        return None
    repayment = compile_seconds / (baseline_seconds - candidate_seconds)
    if not math.isfinite(repayment):
        raise ValueError("repayment count exceeds planning range")
    # A positive cost needs at least one call, even if the ratio underflows.
    return max(1, math.ceil(repayment)) if compile_seconds > 0 else 0


def main() -> None:
    print("rmsnorm", [round(v, 6) for v in rmsnorm_reference([1, 2, 3], [1, 1, 1])])
    work = padding_work(2300, [2048, 4096])
    print("padding", work.bucket_tokens, round(work.linear_ratio, 3), round(work.dense_pair_ratio, 3))
    common = dict(layers=32, batch=8, context=4096, kv_heads=8, head_dim=128, element_bytes=2)
    for tp, cp in [(1, 1), (4, 1), (16, 1), (4, 2)]:
        plan = kv_shard_plan(**common, tp=tp, cp=cp)
        print("kv", tp, cp, plan.bytes_per_rank // 2**20, "MiB/rank",
              plan.aggregate_modeled_kv_bytes // 2**20, "MiB/modeled physical KV aggregate")
    # Binary-exact durations make the printed teaching value exact.
    print("compile_break_even", compile_break_even(60, .125, .0625), "calls")


if __name__ == "__main__":
    main()
