"""Finite-distribution and quantization teaching references."""
from math import isclose, isfinite


def validate_distribution(values):
    if not values or not all(isfinite(x) and x >= 0 for x in values) or not isclose(sum(values), 1., abs_tol=1e-12):
        raise ValueError("finite normalized probabilities required")


def acceptance_and_residual(p, q, proposed):
    validate_distribution(p)
    validate_distribution(q)
    if len(p) != len(q) or not 0 <= proposed < len(q) or q[proposed] <= 0:
        raise ValueError("proposal must have positive mass under q")
    acceptance = min(1., p[proposed]/q[proposed])
    positive = [max(a-b, 0.) for a, b in zip(p, q)]
    mass = sum(positive)
    residual = [x/mass for x in positive] if mass else None
    return acceptance, residual


def speculative_output_mass(p, q):
    """Enumerate accepted and rejected mass; no Monte Carlo noise."""
    validate_distribution(p)
    validate_distribution(q)
    if len(p) != len(q):
        raise ValueError("matching vocabularies required")
    overlap = [min(a, b) for a, b in zip(p, q)]
    residual = [max(a-b, 0.) for a, b in zip(p, q)]
    missing = sum(residual)
    reject = 1-sum(overlap)
    return [o + reject*r/missing if missing else o for o, r in zip(overlap, residual)]


def symmetric_quantize(values, bits=4):
    """One scale, signed symmetric integer grid, round-to-nearest-even."""
    if not values or not all(map(isfinite, values)) or not isinstance(bits, int) or not 2 <= bits <= 16:
        raise ValueError("finite values and 2..16 bits required")
    qmax = 2**(bits-1)-1
    amax = max(abs(x) for x in values)
    scale = amax/qmax if amax else 1.
    codes = [min(qmax, max(-qmax, round(x/scale))) for x in values]
    return codes, scale, [q*scale for q in codes]
