"""Tiny CPU mechanisms, not pretrained tokenizers or production model layers."""
from math import exp, isfinite, log


def bpe_pieces(text, merges):
    """Ranked byte-pair merges, without normalization or pretokenization."""
    pieces = [bytes([b]) for b in text.encode("utf-8")]
    ranks = {pair: rank for rank, pair in enumerate(merges)}
    while len(pieces) > 1:
        choices = [(ranks[pair], pair) for pair in zip(pieces, pieces[1:])
                   if pair in ranks]
        if not choices:
            break
        _, chosen = min(choices)
        merged, i = [], 0
        while i < len(pieces):
            if i + 1 < len(pieces) and (pieces[i], pieces[i+1]) == chosen:
                merged.append(pieces[i] + pieces[i+1])
                i += 2
            else:
                merged.append(pieces[i])
                i += 1
        pieces = merged
    return pieces


def masked_token_loss(logits, targets, mask):
    """Mean cross entropy over supervised positions, using log-sum-exp.

    Ignored positions may use sentinel targets such as -100. All logit rows
    must still be nonempty and finite; only supervised targets index a row.
    """
    if not (len(logits) == len(targets) == len(mask)) or not any(mask):
        raise ValueError("matching nonempty supervision required")
    losses = []
    for row, target, use in zip(logits, targets, mask):
        if not row or not all(map(isfinite, row)):
            raise ValueError("invalid logits")
        if use:
            if not isinstance(target, int) or not 0 <= target < len(row):
                raise ValueError("invalid supervised target")
            maximum = max(row)
            losses.append(log(sum(exp(x-maximum) for x in row)) + maximum - row[target])
    return sum(losses) / len(losses)


def delta_step(state, key, value, query, alpha=1.0, beta=1.0):
    """Gated delta recurrence, state [value_dim][key_dim]; pure, no mutation.

    Unit-norm keys make beta=1 an exact overwrite in that key direction.
    Projection, convolution, head grouping, output gating/norm are omitted.
    """
    if not key or not value or len(query) != len(key) or len(state) != len(value):
        raise ValueError("invalid dimensions")
    if any(len(row) != len(key) for row in state):
        raise ValueError("invalid state dimensions")
    numbers = key + value + query + [x for row in state for x in row] + [alpha, beta]
    if not all(map(isfinite, numbers)) or not 0 <= alpha <= 1 or not 0 <= beta <= 1:
        raise ValueError("finite inputs and gates in [0, 1] required")
    decayed = [[alpha*x for x in row] for row in state]
    prediction = [sum(x*k for x, k in zip(row, key)) for row in decayed]
    updated = [[x + beta*(v-p)*k for x, k in zip(row, key)]
               for row, v, p in zip(decayed, value, prediction)]
    output = [sum(x*q for x, q in zip(row, query)) for row in updated]
    return updated, output
