"""Deterministic teaching simulator, not a GPU performance predictor.

One FIFO transfer link can overlap one FIFO compute server. Compute executes
whole requests without batching or preemption. Resident slots include queueing.
Rates are invented, constant service rates; no quality or failure model is implied.
Run: python -m examples.serving_load
"""
from dataclasses import dataclass, replace
import heapq
import math


@dataclass(frozen=True)
class Request:
    arrival: float
    prompt: int = 8192
    output: int = 128
    reuse: float = 0.0  # Fraction of prompt KV imported from an external tier.


@dataclass(frozen=True)
class Config:
    slots: int = 3
    capacity_gib: float = 80.0
    fixed_gib: float = 56.0  # Includes the six-GiB admission reserve.
    other_request_gib: float = 3.5
    kv_gib_per_token: float = 1 / 8192
    bandwidth_gib_s: float = 20.0
    prefill_tokens_s: float = 32768.0
    replay_tokens_s: float = 131072.0
    decode_tokens_s: float = 512.0
    ttft_limit_s: float = 1.0
    completion_limit_s: float = 2.0


@dataclass(frozen=True)
class Result:
    offered: int
    admitted: int
    rejected_slots: int
    rejected_memory: int
    within_slo: int
    horizon_s: float
    peak_gib: float
    ttft_s: tuple[float, ...]
    completion_s: tuple[float, ...]
    transfer_gib: float

    @property
    def goodput(self):
        return self.within_slo / self.horizon_s if self.horizon_s else 0.0


def _finite(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def simulate(requests, config=Config()):
    """Reject immediately on slot/memory pressure; admitted work drains to completion.

    Rejections fail the offered-workload SLO fraction. Latencies are conditional
    on admission; never compare those percentiles without the rejection count.
    The rate denominator includes the final drain, from first arrival to last
    completion (or last rejected arrival, whichever is later). Ties free memory
    before admission. Input order breaks equal-arrival ties. Memory rejection takes
    precedence when both memory and slot limits fail.
    """
    requests = tuple(requests)
    if type(config.slots) is not int or config.slots < 1:
        raise ValueError('slots must be a positive integer')
    for name in ('capacity_gib', 'bandwidth_gib_s', 'prefill_tokens_s',
                 'replay_tokens_s', 'decode_tokens_s', 'ttft_limit_s',
                 'completion_limit_s'):
        if not _finite(getattr(config, name), positive=True):
            raise ValueError(f'{name} must be finite and positive')
    for name in ('fixed_gib', 'other_request_gib', 'kv_gib_per_token'):
        if not _finite(getattr(config, name)):
            raise ValueError(f'{name} must be finite and nonnegative')
    if config.fixed_gib > config.capacity_gib:
        raise ValueError('fixed allocation exceeds device capacity')
    previous = -1.0
    for request in requests:
        if not _finite(request.arrival) or request.arrival < previous:
            raise ValueError('arrivals must be finite, nonnegative and sorted')
        if any(type(n) is not int or n < 1 for n in (request.prompt, request.output)):
            raise ValueError('prompt and output lengths must be positive integers')
        if not _finite(request.reuse) or request.reuse > 1:
            raise ValueError('reuse must be in [0, 1]')
        previous = request.arrival
    residents = []
    used = config.fixed_gib
    peak = used
    link_free = compute_free = 0.0
    rejected_slots = rejected_memory = within_slo = 0
    ttfts, completions = [], []
    transfer_gib = 0.0
    for index, request in enumerate(requests):
        while residents and residents[0][0] <= request.arrival:
            _, _, size = heapq.heappop(residents)
            used -= size
        size = (config.other_request_gib +
                (request.prompt + request.output) * config.kv_gib_per_token)
        if used + size > config.capacity_gib + 1e-10:
            rejected_memory += 1
            continue
        if len(residents) >= config.slots:
            rejected_slots += 1
            continue
        used += size
        peak = max(peak, used)
        reused = request.prompt * request.reuse
        imported = reused * config.kv_gib_per_token
        transfer_gib += imported
        ready = request.arrival
        if imported:
            ready = max(ready, link_free) + imported / config.bandwidth_gib_s
            link_free = ready
        # A later request cannot overtake an earlier admitted compute reservation.
        start = max(ready, compute_free)
        preparation = ((request.prompt - reused) / config.prefill_tokens_s +
                       reused / config.replay_tokens_s)
        first = start + preparation + 1 / config.decode_tokens_s
        end = start + preparation + request.output / config.decode_tokens_s
        compute_free = end
        ttft, completion = first - request.arrival, end - request.arrival
        ttfts.append(ttft)
        completions.append(completion)
        within_slo += (ttft <= config.ttft_limit_s and
                       completion <= config.completion_limit_s)
        heapq.heappush(residents, (end, index, size))
    horizon = (max(requests[-1].arrival, compute_free) - requests[0].arrival
               if requests else 0.0)
    return Result(len(requests), len(ttfts), rejected_slots, rejected_memory,
                  within_slo, horizon, peak, tuple(ttfts), tuple(completions),
                  transfer_gib)


def scenarios():
    base = Config()
    trace = tuple(Request(i / 5) for i in range(60))
    yield 'one-slot', trace, replace(base, slots=1)
    yield 'three-slots', trace, base
    yield 'six-slots', trace, replace(base, slots=6)
    yield 'long-prompts', tuple(replace(r, prompt=32768) for r in trace), base
    cached = tuple(replace(r, reuse=.75) for r in trace)
    yield 'reuse-fast-link', cached, base
    yield 'reuse-slow-link', cached, replace(base, bandwidth_gib_s=1.0)


if __name__ == '__main__':
    print('SYNTHETIC FIFO MODEL: 60 arrivals at 5/s; no batching or measured GPU rates')
    print('scenario admitted slot_reject memory_reject SLO_pass goodput/s peak_GiB')
    for name, trace, config in scenarios():
        r = simulate(trace, config)
        print(f'{name:16s} {r.admitted:3d} {r.rejected_slots:3d} '
              f'{r.rejected_memory:3d} {r.within_slo:3d} '
              f'{r.goodput:.3f} {r.peak_gib:.3f}')
