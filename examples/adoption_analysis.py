"""Analyze Chapter 59's invented paired evaluation; no model/API calls.

Intervals assume independent task pairs. The paired normal interval is only an
approximation, not a general noninferiority or production-safety certification.
Costs include unsuccessful attempts and retries. None means no first token.
Run with ``python -m examples.adoption_analysis``.
"""
from dataclasses import dataclass
import math
from statistics import NormalDist


def _nonnegative_finite(value, name):
    if type(value) not in (int, float) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass(frozen=True)
class Outcome:
    success: bool
    cost: float
    first_token_seconds: float | None
    forbidden_effect: bool = False

    def __post_init__(self):
        if type(self.success) is not bool or type(self.forbidden_effect) is not bool:
            raise ValueError("success and forbidden_effect must be booleans")
        _nonnegative_finite(self.cost, "cost")
        if self.first_token_seconds is None:
            if self.success:
                raise ValueError("a no-token timeout cannot be a verified success")
        else:
            _nonnegative_finite(self.first_token_seconds, "first_token_seconds")


@dataclass(frozen=True)
class TaskPair:
    task_id: str
    incumbent: Outcome
    candidate: Outcome

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a nonempty string")
        if not isinstance(self.incumbent, Outcome) or not isinstance(self.candidate, Outcome):
            raise ValueError("each task must contain both validated outcomes")


def _wilson_lower(successes, count):
    """Lower endpoint of the two-sided 95% Wilson score interval."""
    z = NormalDist().inv_cdf(0.975)
    p = successes / count
    return (p + z*z/(2*count) - z*math.sqrt(
        p*(1-p)/count + z*z/(4*count*count))) / (1 + z*z/count)


def _metrics(outcomes):
    count = len(outcomes)
    successes = sum(item.success for item in outcomes)
    try:
        total_cost = math.fsum(item.cost for item in outcomes)
    except OverflowError as error:
        raise ValueError("aggregate cost exceeds numeric range") from error
    latencies = sorted(math.inf if item.first_token_seconds is None
                       else item.first_token_seconds for item in outcomes)
    violations = sum(item.forbidden_effect for item in outcomes)
    return {
        "successes": successes,
        "success_rate": successes / count,
        "wilson_lower_95": _wilson_lower(successes, count),
        "total_cost": total_cost,
        # Infinite ratio means no verified successes; it is never cost evidence.
        "cost_per_success": total_cost / successes if successes else math.inf,
        "p99_seconds": latencies[math.ceil(0.99 * count) - 1],
        "timeouts": sum(item.first_token_seconds is None for item in outcomes),
        "latency_slo_failures": sum(value > 1.5 for value in latencies),
        "forbidden_effects": violations,
        # Exact binomial upper bound, only for zero observed events and iid trials.
        "zero_event_upper_95": -math.expm1(math.log(0.05)/count)
                               if violations == 0 else None,
    }


def analyze(pairs):
    """Apply the chapter's fixed offline gates; never recommend full rollout.

    The normal-approximation precheck (n >= 30 and >= 10 discordant pairs) is
    an illustrative screening rule, not a calibrated accuracy guarantee. Smaller
    studies need an appropriate prespecified inference method or more evidence.
    This function cannot verify pair independence, snapshot fidelity or provenance.
    """
    pairs = tuple(pairs)
    if len(pairs) < 2:
        raise ValueError("at least two task pairs are needed for sample variance")
    if any(not isinstance(pair, TaskPair) for pair in pairs):
        raise ValueError("validated TaskPair records required")
    if len({pair.task_id for pair in pairs}) != len(pairs):
        raise ValueError("duplicate task IDs would count the same pair twice")
    count = len(pairs)
    incumbent = _metrics([pair.incumbent for pair in pairs])
    candidate = _metrics([pair.candidate for pair in pairs])
    differences = [int(pair.candidate.success) - int(pair.incumbent.success)
                   for pair in pairs]
    delta = math.fsum(differences) / count
    variance = math.fsum((value-delta)**2 for value in differences) / (count-1)
    standard_error = math.sqrt(variance/count)
    width = NormalDist().inv_cdf(0.975) * standard_error
    interval = (delta-width, delta+width)
    discordant = sum(value != 0 for value in differences)
    base_cost = incumbent["cost_per_success"]
    new_cost = candidate["cost_per_success"]
    savings = (1-new_cost/base_cost if math.isfinite(base_cost) and base_cost > 0
               and math.isfinite(new_cost) else None)
    gates = {
        "normal_approximation_precheck": count >= 30 and discordant >= 10,
        "paired_noninferiority": interval[0] > -0.02,
        "candidate_success_floor": candidate["wilson_lower_95"] > 0.88,
        "no_observed_forbidden_effects": candidate["forbidden_effects"] == 0,
        "empirical_latency": candidate["p99_seconds"] <= 1.5,
        "cost_reduction": savings is not None and savings >= 0.20,
    }
    return {
        "tasks": count,
        "paired_counts": {
            "both": sum(p.incumbent.success and p.candidate.success for p in pairs),
            "incumbent_only": differences.count(-1),
            "candidate_only": differences.count(1),
            "neither": sum(not p.incumbent.success and not p.candidate.success for p in pairs),
        },
        "incumbent": incumbent,
        "candidate": candidate,
        "paired_improvement": delta,
        "paired_standard_error": standard_error,
        "paired_normal_interval_95": interval,
        "cost_reduction": savings,
        # Chance of sampling zero observations from a true 1% tail in iid trials.
        # This is not a confidence interval for the measured latency quantile.
        "probability_miss_one_percent_tail": 0.99**count,
        "gates": gates,
        "decision": "bounded_canary_only" if all(gates.values()) else "do_not_adopt",
    }


def synthetic_fixture():
    """400 invented independent pairs; constant latencies are teaching inputs."""
    pairs = []
    for incumbent_success, candidate_success, count in (
            (True, True, 340), (True, False, 20),
            (False, True, 30), (False, False, 10)):
        for _ in range(count):
            pairs.append(TaskPair(f"task-{len(pairs):03d}",
                                  Outcome(incumbent_success, 0.4, 1.30),
                                  Outcome(candidate_success, 0.3, 1.45)))
    return tuple(pairs)


def demo():
    result = analyze(synthetic_fixture())
    low, high = result["paired_normal_interval_95"]
    print(f"Invented pairs: {result['tasks']}; decision={result['decision']}")
    print(f"Paired outcomes: {result['paired_counts']}")
    print(f"Paired gain: {result['paired_improvement']:.2%}; "
          f"approximate 95% interval=[{low:.2%}, {high:.2%}]")
    print(f"Candidate Wilson lower bound: {result['candidate']['wilson_lower_95']:.2%}; "
          f"cost reduction: {result['cost_reduction']:.2%}")
    print(f"Cost per verified success: {result['incumbent']['cost_per_success']:.3f} "
          f"vs {result['candidate']['cost_per_success']:.3f}; "
          f"empirical p99: {result['incumbent']['p99_seconds']:.2f}s "
          f"vs {result['candidate']['p99_seconds']:.2f}s")
    print(f"Zero-event one-sided 95% upper bound: "
          f"{result['candidate']['zero_event_upper_95']:.2%}")
    print(f"Chance of missing a true 1% tail: "
          f"{result['probability_miss_one_percent_tail']:.2%}")
    print("No production evidence: normal/iid assumptions and empirical p99 need review.")


if __name__ == "__main__":
    demo()
