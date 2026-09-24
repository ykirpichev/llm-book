# Engineering Large Language Models - runnable Part VII experiments

**[Read Part VII online](https://ykirpichev.github.io/llm-book/part-07.html)**

This revision prioritizes two practical improvements:

- **Chapter 55:** a runnable serving-load simulation with six scenarios, memory
  and slot admission, transfer queues, latency gates and rejected-work accounting.
- **Chapter 59:** an executable paired adoption analysis, with explicit treatment
  of failed-task costs, timeouts, rare-event evidence and empirical p99 limitations.

Both examples run on the CPU using only Python's standard library. They use
invented inputs and demonstrate reasoning and bookkeeping; they are not GPU
benchmarks or evidence that a production system passes the illustrated gates.

The [change record](https://github.com/ykirpichev/llm-book/blob/main/docs/part7-experiments-2026-09-24.md)
and [validation record](https://github.com/ykirpichev/llm-book/blob/main/docs/release-readiness.md)
describe the scope and review. The frontier-model snapshot remains September 23.
Earlier Part VI and Part VII improvements are included, and prior release assets
are preserved.

Download the reviewed PDF, editable source ZIP and SHA-256 checksums below.
Text and original figures: CC BY 4.0. Original code/build tools: MIT.
Third-party materials retain their own terms.
