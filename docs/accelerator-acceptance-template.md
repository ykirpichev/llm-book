# Accelerator engine acceptance record

Use this template to compare a proposed accelerator path with the current
baseline. It is a reporting contract, not a support matrix or a benchmark.
Leave an unavailable field as `not measured` and explain why; do not substitute
a different workload or feature set without recording the change.

## Decision

- Decision owner/date:
- Candidate and baseline revision:
- Required outcome: `go`, `no-go`, or `fallback`.
- Pass rule: the candidate preserves the required features and quality and
  meets the stated SLO at the same offered load. Otherwise record the limiting
  evidence and a fallback/no-go decision.

## One record for the baseline and each candidate

| Field group | Baseline | Candidate |
| --- | --- | --- |
| Model identity | Checkpoint, tokenizer, revision | Checkpoint, tokenizer, revision |
| Execution identity | Engine/backend; compiler/runtime; target; TP/CP/DCP/PP | Engine/backend; compiler/runtime; target; TP/CP/DCP/PP |
| Required features | Attention backend; sampling; paging; KV dtype/layout/quantization; connector/cache ABI | Same fields; list an unsupported or changed feature explicitly |
| Workload | Prompt/output cohorts; arrival process; offered load; cancellation/retry policy; warm/cold mix; duration | Same fields and values, or documented exception |
| Correctness | Logit/reference check and tolerance; quality gate; failures/rejections; conversion status | Same checks and results |
| Service | TTFT and token-gap p50/p95/p99 by cohort; goodput; accepted concurrency; queue time | Same measurements |
| Cold path | Ready time; compile/artifact-cache misses; miss policy | Same measurements |
| Capacity and cost | Peak memory on constraining rank; host/network resources; SLO-qualified requests; cost per SLO-qualified request | Same measurements |
| Evidence and decision | Run IDs/log locations; unmet gate; conclusion | Run IDs/log locations; unmet gate; conclusion |

## Interpretation boundaries

- A CPU oracle, interpreter, or source parse does not populate this record's
  device-operator, engine, or deployment fields.
- A device operator sweep establishes only pinned-stack operator evidence. It
  does not establish the model's feature set, quality, traffic behavior, or
  cost.
- A warm microbenchmark is not a passing cold-readiness or overload result.
- A connector/cache conversion is a separate feature and correctness gate,
  even when the checkpoint or vendor is shared.

Attach the target harness provenance header, command, environment/driver
record, shape and dtype cases, skipped cases, and raw SLO/quality artifacts.
Never convert an unrecorded or skipped target case into a pass.
