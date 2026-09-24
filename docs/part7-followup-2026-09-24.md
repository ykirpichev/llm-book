# Part VII worked additions - September 24, 2026

This follow-up implements all six recommendations made after the September 23
Part VII revision. The frontier-model map remains a September 23 snapshot;
these additions improve explanation and reproducibility, not the map's cutoff.
The September 24 Part VI revision is preserved in full.

| Chapter | Recommendation implemented |
| --- | --- |
| 54 | Original AttnRes/mHC vector diagram: depth selection versus parallel-stream transport, with distinct layer-update branch, block-source description and worked vector-coordinate arithmetic |
| 55 | Complete hypothetical 80-GiB worker ledger: shared weights/runtime, per-request KV/recurrent state, scratch, output reservation, reserve, concurrency and transfer-bandwidth ceiling |
| 56 | Concurrent video/audio/tool timeline with evidence timestamps, task generations, interruption acknowledgment, stale-result rejection, actual playback and write-commit race boundaries |
| 57 | Six-family generation comparison organized by mutable unit, commitment, cache validity and evaluation boundary; no universal speed ranking |
| 58 | Runnable persistent task ledger over the Part VI effect fixture: crash between effect commit and task completion, receipt reconciliation at exhausted attempt budget, immutable intent and generation/authority checks |
| 59 | Complete hypothetical adoption case with paired outcomes, uncertainty, quality/cost/latency gates, migration economics, bounded canary, isolated paired evaluation and rollback |

## Evidence and review

New mechanism sources inspected September 24:

- [Attention Residuals v1, March 16](https://arxiv.org/abs/2603.15031v1): depth-wise attention, normalized keys, original values and block summaries.
- [mHC v2, January 5](https://arxiv.org/html/2512.24880v2): constrained stream transport distinct from readout and writeback.

The generation table was checked against the chapter's existing primary sources:
[Block Diffusion](https://arxiv.org/abs/2503.09573),
[DiffusionGemma v1](https://arxiv.org/html/2608.00146v1), its
[vLLM implementation account](https://vllm-project.github.io/2026/06/10/diffusion-gemma),
and the [LLaDA2.2-flash card](https://huggingface.co/inclusionAI/LLaDA2.2-flash).
The diagram is original artwork. Ledger, timeline and adoption numbers are
explicitly hypothetical; they are not local measurements of the cited systems.

Two AI reviewer agents independently reviewed the additions. The mechanisms
reviewer covered Chapters 54 and 57 and required the AttnRes arithmetic to be
labeled as one coordinate of vector sources: normalized positive scalar keys
would not justify the chosen unequal weights. The systems reviewer covered
Chapters 55, 56, 58 and 59 plus the fixture, and required a valid same-task
comparator for canary paired statistics and explicit latency treatment of
no-token timeouts. All findings were applied, the draft returned for review,
and both agents approved with no remaining blocking content findings.

This is AI review, not independent human certification or accelerator benchmark
reproduction. The task fixture assumes one active runner, a trusted host's
identity/authority/generation inputs, and local SQLite effect atomicity. It
bounds effect attempts, not model spend or all reconciliation calls. Remote
services still require their own idempotency, authority ordering and recovery
contracts. See [release readiness](release-readiness.md) for artifact validation.
