"""Single-head attention and exact partition merging, using Python float64.

This clarity-first reference materializes scores and is not a GPU benchmark.
An all-masked row returns zero by explicit convention.
"""
from __future__ import annotations

import math


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
    pairs = [(sum(a * b for a, b in zip(q, k)) / math.sqrt(len(q)), v)
             for k, v, keep in zip(keys, values, allowed) if keep]
    if not pairs:
        return -math.inf, 0.0, [0.0] * len(values[0])
    maximum = max(s for s, _ in pairs)
    weights = [math.exp(s - maximum) for s, _ in pairs]
    mass = sum(weights)
    output = [sum(w * v[j] for w, (_, v) in zip(weights, pairs))
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
    states = [(m, l, o) for m, l, o in states if l > 0]
    if not states:
        return [0.0] * len(values[0])
    maximum = max(m for m, _, _ in states)
    factors = [math.exp(m - maximum) for m, _, _ in states]
    denominator = sum(a * l for a, (_, l, _) in zip(factors, states))
    return [sum(a * o[j] for a, (_, _, o) in zip(factors, states)) / denominator
            for j in range(len(values[0]))]


if __name__ == "__main__":
    print(attention([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]],
                    [[2.0], [4.0]], shard_size=1))
