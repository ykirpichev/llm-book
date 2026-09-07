"""Exact immutable-record top-k and mergeable Welford moments."""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heapreplace
import math


def top_k(records, k):
    """Records are (finite score, unique stable string ID); larger wins ties.

    Unique IDs are a caller precondition; maintaining an unbounded ID set
    inside this O(k)-memory primitive would defeat the streaming contract.
    """
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    heap = []
    for score, identifier in records:
        if not math.isfinite(score) or not isinstance(identifier, str):
            raise ValueError("finite score and string ID required")
        record = (score, identifier)
        if len(heap) < k:
            heappush(heap, record)
        elif k and record > heap[0]:
            heapreplace(heap, record)
    return sorted(heap, reverse=True)


@dataclass(frozen=True)
class Moments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x):
        if not math.isfinite(x):
            raise ValueError("finite observations required")
        n = self.count + 1
        delta = x - self.mean
        mean = self.mean + delta / n
        return Moments(n, mean, self.m2 + delta * (x - mean))

    def merge(self, other):
        if not self.count:
            return other
        if not other.count:
            return self
        n = self.count + other.count
        delta = other.mean - self.mean
        return Moments(n, self.mean + delta * other.count / n,
                       self.m2 + other.m2 + delta * delta * self.count * other.count / n)

    def sample_variance(self):
        if self.count < 2:
            raise ValueError("sample variance requires at least two observations")
        return self.m2 / (self.count - 1)


if __name__ == "__main__":
    print(top_k([(1.0, "a"), (3.0, "b"), (3.0, "c")], 2))
