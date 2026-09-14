"""Single-head attention and exact partition merging, using Python float64.

This clarity-first reference materializes scores and is not a GPU benchmark.
Finite inputs are necessary but not sufficient: visible scores, partial
numerators, merged numerators, and outputs must fit Python float. An
all-masked row returns zero by explicit convention.
"""
from __future__ import annotations

import math


_RANGE_ERROR = "attention arithmetic exceeds Python float range"


def finite(value):
    if not math.isfinite(value):
        raise ValueError(_RANGE_ERROR)
    return value


def validate(q, keys, values):
    if not q or not keys or len(keys) != len(values) or not values[0]:
        raise ValueError("nonempty query and matching nonempty key/value rows required")
    if any(len(k) != len(q) for k in keys):
        raise ValueError("key dimension differs from query")
    if any(len(v) != len(values[0]) for v in values):
        raise ValueError("value rows must have equal width")
    if any(not math.isfinite(x) for row in [q, *keys, *values] for x in row):
        raise ValueError("inputs must be finite")


def partial(q, keys, values, allowed):
    """Return (maximum score, shifted mass, shifted weighted-value sum)."""
    pairs = []
    for key, value, keep in zip(keys, values, allowed):
        if keep:
            score = finite(sum(a * b for a, b in zip(q, key)) / math.sqrt(len(q)))
            pairs.append((score, value))
    if not pairs:
        return -math.inf, 0.0, [0.0] * len(values[0])
    maximum = finite(max(s for s, _ in pairs))
    weights = [finite(math.exp(score - maximum)) for score, _ in pairs]
    mass = finite(sum(weights))
    if mass <= 0:
        raise ValueError(_RANGE_ERROR)
    output = [finite(sum(weight * value[j]
                         for weight, (_, value) in zip(weights, pairs)))
              for j in range(len(values[0]))]
    return maximum, mass, output


def attention(q, keys, values, allowed=None, shard_size=None):
    """Compare dense (one shard) and context-sharded attention semantics."""
    validate(q, keys, values)
    allowed = [True] * len(keys) if allowed is None else list(allowed)
    if len(allowed) != len(keys):
        raise ValueError("mask must match key count")
    shard_size = len(keys) if shard_size is None else shard_size
    if not isinstance(shard_size, int) or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    states = [partial(q, keys[i:i + shard_size], values[i:i + shard_size],
                      allowed[i:i + shard_size])
              for i in range(0, len(keys), shard_size)]
    active = []
    for maximum, mass, numerator in states:
        if mass == 0:
            if maximum != -math.inf or any(value != 0 for value in numerator):
                raise ValueError(_RANGE_ERROR)
            continue
        if mass < 0:
            raise ValueError(_RANGE_ERROR)
        finite(maximum)
        finite(mass)
        for value in numerator:
            finite(value)
        active.append((maximum, mass, numerator))
    if not active:
        return [0.0] * len(values[0])
    maximum = finite(max(m for m, _, _ in active))
    factors = [finite(math.exp(m - maximum)) for m, _, _ in active]
    denominator = finite(sum(factor * mass
                             for factor, (_, mass, _) in zip(factors, active)))
    if denominator <= 0:
        raise ValueError(_RANGE_ERROR)
    merged = [finite(sum(factor * numerator[j]
                         for factor, (_, _, numerator) in zip(factors, active)))
              for j in range(len(values[0]))]
    return [finite(value / denominator) for value in merged]


if __name__ == "__main__":
    print(attention([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]],
                    [[2.0], [4.0]], shard_size=1))
