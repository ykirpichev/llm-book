"""Scalar objective references; no model training or GPU performance claims."""
from math import comb, exp, isfinite, log1p, sqrt


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=.1):
    values = [chosen, rejected, reference_chosen, reference_rejected, beta]
    if not all(map(isfinite, values)) or beta <= 0:
        raise ValueError("finite log probabilities and positive beta required")
    margin = beta * ((chosen-rejected) - (reference_chosen-reference_rejected))
    return max(-margin, 0) + log1p(exp(-abs(margin)))


def group_advantages(rewards, normalize=True):
    if not rewards or not all(map(isfinite, rewards)):
        raise ValueError("finite nonempty reward group required")
    mean = sum(rewards)/len(rewards)
    centered = [r-mean for r in rewards]
    std = sqrt(sum(x*x for x in centered)/len(centered))
    if not normalize:
        return centered
    return [x/std for x in centered] if std else [0.]*len(rewards)


def clipped_surrogate(ratio, advantage, lower=.2, upper=.2):
    if not all(map(isfinite, [ratio, advantage, lower, upper])) or ratio <= 0 or not 0 <= lower < 1 or upper < 0:
        raise ValueError("invalid ratio or clipping interval")
    clipped = min(max(ratio, 1-lower), 1+upper)
    return min(ratio*advantage, clipped*advantage)


def sequence_ratio(new_logps, old_logps):
    if not new_logps or len(new_logps) != len(old_logps):
        raise ValueError("matching nonempty token log probabilities required")
    if not all(map(isfinite, new_logps+old_logps)):
        raise ValueError("finite log probabilities required")
    return exp(sum(n-o for n, o in zip(new_logps, old_logps))/len(new_logps))


def pass_at_k(n, correct, k):
    """Unbiased estimator from n sampled candidates, under standard assumptions."""
    if not all(isinstance(x, int) for x in [n, correct, k]) or not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("require 0 <= correct <= n and 1 <= k <= n")
    return 1.0 if n-correct < k else 1.0-comb(n-correct, k)/comb(n, k)
