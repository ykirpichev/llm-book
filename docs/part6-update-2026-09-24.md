# Part VI revision - September 24, 2026

All eight chapters (46-53) and the Part VI introduction were reviewed and
expanded. The established algorithms remain useful; the largest omissions
concerned measurement bias, operational boundaries, retrieval representations,
and recovery testing. Part VII retains its September 23 revision; other parts
retain their September 13 research cutoff unless explicitly dated.

## Recommended improvements implemented

| Chapter | Previous gap | Added or corrected |
| --- | --- | --- |
| 46 - Exact Streaming Queries and Time Windows | Window finalization did not sufficiently distinguish runtime TTL, replay and deduplication horizons | Outage/replay example, event-time versus processing-time expiration, repair epochs and sink fencing |
| 47 - Online Statistics and Sampling | Sampling algorithms were not connected to selectively retained production traces | Head/tail sampling, inclusion-probability weighting, biased denominators, zero-probability blind spots, worked 1% versus 50.25% example |
| 48 - Heavy Hitters and Probabilistic Sketches | Relative error could be read as meaning only relative numerical-value error | KLL/REQ/DDSketch distinction, tail rank versus value guarantees, collapse/near-zero limits, threshold SLO counters and invalid percentile averaging |
| 49 - Building a Streaming Telemetry Service | Missing current GenAI trace-schema and transaction-boundary guidance | OpenTelemetry GenAI repository migration, logical request versus attempts, chunk/token timing, usage accounting, bounded metric cardinality and outbox effects |
| 50 - Compact Review of ML Algorithms in Production | Classical predictors lacked an application to selective model routing | Coverage/risk/cost tradeoff, calibration and pipeline drift, worked routing arithmetic, zero-failure uncertainty |
| 51 - An Engineering System Design and Review Method | Review process emphasized a steady-state design | Compatible-version manifests, shadow/cutover/rollback, authority-preserving rollback, Little's law, admission and shared deadlines |
| 52 - RAG, Vector Search, and Evaluation Pipelines | Text-centric evidence and insufficient separation of retrieval failure layers | ColPali and 2026 Qwen3-VL retrieval, page/region/time provenance, supported dimension reduction, exact-vector versus relevance oracles, selective filters and paired evaluation |
| 53 - Building and Evaluating a Bounded Agent Loop | Retry teaching fixture was in-memory only; limited controlled-failure coverage | SQLite atomic effect/receipt fixture, process restart and abrupt-exit tests, concurrent retries, tenant-scoped keys, checkpoint-versus-effect distinction, ToolBench-X and repeated reliability |

The additions include original worked checks. They make no vendor ranking or
performance claim. Contemporary examples are selected for the engineering
boundary they explain, not represented as an exhaustive state-of-the-art census.

## Primary-source ledger

Inspected September 24, 2026. Mutable documentation is an inspection snapshot;
readers should pin the library, schema and model revisions they actually deploy.

- [Flink state](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/datastream/fault-tolerance/state/): documented processing-time TTL boundary.
- [OpenTelemetry sampling](https://opentelemetry.io/docs/concepts/sampling/): head/tail decisions; the numerical weighting example is original.
- [DataSketches KLL](https://datasketches.apache.org/docs/KLL/KLLSketch.html) and [REQ](https://datasketches.apache.org/docs/REQ/ReqSketch.html): different rank-error profiles.
- [DDSketch, 2019](https://www.vldb.org/pvldb/vol12/p2195-masson.pdf): relative-value bucket mechanism; historical foundation, not a new 2026 algorithm.
- [OpenTelemetry v1.42.0](https://github.com/open-telemetry/semantic-conventions/releases/tag/v1.42.0) and [GenAI repository](https://github.com/open-telemetry/semantic-conventions-genai): migration and schema-version boundary.
- [Kafka Streams concepts](https://docs.confluent.io/platform/current/streams/concepts.html): transaction-coupled offsets, state and Kafka outputs; external effects require a separate contract.
- [ColPali v6, February 2025](https://arxiv.org/abs/2407.01449v6): page-image late interaction.
- [Qwen3-VL retrieval v2, January 2026](https://arxiv.org/abs/2601.04720v2) and [publisher card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B): multimodal embeddings/reranking and supported output dimensions; no leaderboard ranking adopted.
- [Vespa nearest-neighbor guide](https://docs.vespa.ai/en/querying/nearest-neighbor-search-guide.html): exact/approximate search and filters, without transferring example timings to this book's workload.
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence): checkpoint/store distinction and in-memory restart limitation.
- [ToolBench-X v2, June 27, 2026](https://arxiv.org/abs/2606.25819v2): controlled, recoverable tool-environment hazards; no production reliability guarantee inferred.

## Review and validation scope

The authoring agent checked primary sources, arithmetic, implementation bounds,
source changes and rendered layout. This revision does not claim a separate
reviewer-agent campaign, independent human review, production load testing,
hardware validation, or execution of the cited retrieval models/benchmarks.
See [release readiness](release-readiness.md) for final test and artifact results.

The new SQLite example is a local transaction fixture. Its trusted host supplies
identity and current authorization; it does not implement authentication, remote
transactions, checkpoint migration, power-loss certification or a security sandbox.
Existing in-memory agent tests remain useful and explicitly retain their narrower
scope. Production integration still requires real-service failure tests and
representative retrieval/calibration data.
