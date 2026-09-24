# Part VII reproducible experiments - September 24, 2026

This revision prioritizes Chapters 55 and 59 from the follow-up recommendations:
readers can now run the capacity and adoption calculations. The other optional
extensions remain future work; this revision does not claim new experiments for
Chapters 54, 56, 57 or 58. The September 23 frontier-model map and earlier
September 24 worked additions remain intact.

## Serving-load experiment, Chapter 55

`python -m examples.serving_load` runs six deterministic scenarios using invented
service rates, 60 arrivals, per-request memory reservations, resident-slot limits,
a FIFO transfer link, and a FIFO compute server. It reports admissions, mutually
exclusive rejection categories, completions within both latency targets, and
goodput with final drain included. It demonstrates why more admitted requests
can miss more deadlines and why remote cache reuse can lose on a slow link.

There is no batching, preemption, measured GPU rate, quality model, or hardware
execution. The text specifies all rates, lengths, latency gates and denominators.
Tests check independent event timelines, transfer/compute overlap, exact-time
release, memory versus slot accounting, output reservations and rejected work.
The rejection breakdown gives memory precedence if both limits would fail.

## Executable adoption analysis, Chapter 59

`python -m examples.adoption_analysis` generates immutable paired task records
for the chapter's invented 400-task example. It calculates paired uncertainty,
the Wilson lower success bound, costs including failures, empirical p99 and the
fixed offline gates. It never recommends full rollout. The text and code
explicitly label the additional normal-approximation screening rule as heuristic.

Tests challenge pairing, latency, forbidden effects, costs, timeouts, sample size,
zero successes, invalid numeric inputs and duplicate IDs. No-token timeouts
remain in the denominator and costs, count as SLO failures and enter quantiles
as infinity. The text distinguishes an empirical p99 from tail uncertainty and
an observation of zero forbidden effects from a bound on future event rates.
The new primary reference is the [NIST confidence-interval handbook](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm),
inspected September 24. Synthetic examples are not production measurements.

## Scope and maintenance

Both reviewer agents approved the final additions. The mechanisms reviewer
independently checked both examples and requested the memory-rejection precedence
clarification. The systems reviewer authored the adoption module and then
reviewed the serving module and both chapter explanations. Their final findings
and artifact verification are recorded in
[release readiness](release-readiness.md). No independent human certification or
accelerator reproduction is claimed. CPU arithmetic checks cannot establish
production quality, tail behavior, security robustness or a serving-engine ranking.
