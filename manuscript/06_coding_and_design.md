# Part VI - Coding and System Design

Interview coding at senior levels tests more than syntax. The candidate must choose a data structure, state invariants, analyze complexity, handle streams and concurrency, and show how the local algorithm fits a production system.

## Streaming Algorithms and Top-K

LEAD: A stream cannot be revisited cheaply. The design must decide which information to retain, which approximation to accept, and how error is communicated.

### Exact top-k

For a finite stream with known scores, maintain a min-heap of size `k`. Each item above the current minimum replaces the root. Time is `O(n log k)` and memory `O(k)`.

```python
def top_k(stream, k):
    heap = []
    for item in stream:
        scored = (item.score, item.id, item)
        if len(heap) < k:
            heappush(heap, scored)
        elif scored > heap[0]:
            heapreplace(heap, scored)
    return [x.item for x in sorted(heap, reverse=True)]
```

The stable `id` breaks score ties. Production code defines updates, deletions, late data, serialization, and merge across partitions.

For distributed top-k, each shard emits local top-k; the coordinator takes top-k of their union. This is exact because an item outside a shard's local top-k cannot be in the global top-k. With per-key groups, the same argument applies per key but state can explode.

### Sliding windows

Window semantics may be count-based, processing-time, or event-time. Event-time windows require watermarks and a lateness policy. A monotonic deque maintains min or max over a count window in amortized `O(1)` time: remove expired indices from the front and dominated values from the back.

```python
def sliding_max(values, window):
    q = deque()  # indices; values decrease from front to back
    for i, value in enumerate(values):
        while q and q[0] <= i - window:
            q.popleft()
        while q and values[q[-1]] <= value:
            q.pop()
        q.append(i)
        if i + 1 >= window:
            yield values[q[0]]
```

### Online mean and variance

Welford's update avoids the cancellation of `E[x^2] - E[x]^2`:

`n <- n + 1`

`delta = x - mean`

`mean <- mean + delta / n`

`M2 <- M2 + delta * (x - mean)`

Variance is `M2 / (n - 1)` for a sample. Two Welford states can be merged, which makes the method useful in distributed streams.

### Reservoir sampling

To sample `k` items uniformly from a stream of unknown length, fill the reservoir with the first `k`. For item `i` using one-based indexing, choose a random integer in `[1, i]`; replace a reservoir slot if the result is at most `k`.

The proof is inductive: after processing `i` items, every item is retained with probability `k/i`. Weighted reservoirs require different keys or priority sampling.

### Heavy hitters and Count-Min Sketch

A Count-Min Sketch maintains several hash rows. To update key `x` by count `c`, increment one counter per row. Estimate by the minimum of those counters. Collisions only overestimate for nonnegative updates.

With width proportional to `1/epsilon` and depth proportional to `log(1/delta)`, error is bounded by `epsilon` times total count with probability at least `1 - delta`. Hash functions and adversarial robustness matter in exposed systems.

Combine the sketch with a candidate heap for heavy hitters. The sketch estimates frequency, while the heap retains keys worth reporting. Merging is elementwise addition if dimensions and hashes match.

### HyperLogLog and Bloom filters

HyperLogLog estimates cardinality from the position of the first set bit in hashed values across many registers. Its relative error is roughly proportional to `1 / sqrt(registers)`. Bias correction and small-range behavior matter in implementation.

A Bloom filter answers set membership with false positives and no false negatives under append-only use. With `m` bits, `n` elements, and `k` hashes, false-positive rate is approximately:

`(1 - exp(-k n / m))^k`

Counting Bloom filters support deletions with counters but use more memory and can underflow if updates are inconsistent.

:::callout insight|State the approximation contract
For every sketch, say whether error is one-sided, probabilistic, mergeable, and sensitive to adversarial keys. That is the difference between naming a structure and designing with it.
:::

### Continuous log-stream design

If asked to process billions of events, clarify keys, window, ordering, lateness, update rate, query rate, and error tolerance. Partition by stable key, pre-aggregate locally, checkpoint state, and use watermarks for event time. Hot keys may need salting plus a second-stage merge.

Backpressure is part of correctness. Dropping events can bias top-k or averages. If approximation under overload is allowed, make sampling or shedding explicit and account for it in the estimator.

### Principal Interview Review

1. Implement exact top-k and merge distributed results.
2. Explain a monotonic deque and its amortized complexity.
3. Prove reservoir sampling is uniform.
4. Compare Count-Min Sketch, HyperLogLog, and Bloom filters.
5. Design a late-data policy for event-time heavy hitters.

## ML Algorithms in Production Code

LEAD: Production ML code turns mathematical assumptions into shapes, invariants, numerical policies, and measurable failure behavior.

### Vectorized logistic regression

For examples `X`, labels `y`, and weights `w`, logits are `z = Xw`. Binary cross entropy should use a stable logits form rather than computing `log(sigmoid(z))` directly.

```python
def logistic_loss_and_grad(X, y, w, l2=0.0):
    z = X @ w
    # softplus(z) - y*z is stable binary cross entropy from logits.
    loss = mean(softplus(z) - y * z) + 0.5 * l2 * dot(w, w)
    p = sigmoid(z)
    grad = (X.T @ (p - y)) / X.shape[0] + l2 * w
    return loss, grad
```

Discuss sparse features, class weighting, calibration, distributed reduction, feature normalization, leakage, and stopping. Complexity is `O(nnz(X))` for sparse matvec plus the update.

### K-means

Lloyd's algorithm alternates nearest-centroid assignment and centroid recomputation. Complexity is `O(n k d)` per iteration. Initialization strongly affects the local optimum; k-means++ spreads initial centers.

For large data, mini-batch k-means updates centroids from samples. Distributed implementations compute local counts and sums, then reduce. Empty clusters require a policy: reinitialize to a far point, split a large cluster, or retain the old centroid.

The Euclidean objective assumes roughly spherical clusters and comparable feature scales. If those assumptions fail, changing the optimizer will not fix the model.

### Decision trees

A tree searches splits that reduce impurity. For classification, Gini or entropy is common; for regression, squared error. Efficient continuous-feature search sorts values or uses histograms. Missing values and categorical features require explicit routing.

Depth, minimum leaf size, feature subsampling, and pruning trade fit against variance. Distributed histogram methods reduce candidate statistics instead of shipping examples. Production code also handles monotonic constraints, stable serialization, and feature schema evolution.

### Attention from primitives

```python
def attention(q, k, v, mask=None):
    scale = 1.0 / sqrt(q.shape[-1])
    scores = q @ swapaxes(k, -1, -2) * scale
    if mask is not None:
        scores = where(mask, scores, -inf)
    probs = softmax(scores, axis=-1)
    return probs @ v
```

Follow-ups include causal masks, padding, batched heads, mixed precision, stable softmax, dropout, KV caching, GQA, and memory-efficient attention. The concise code is a semantic reference, not a performant long-context implementation.

### Beam search

Beam search keeps the best partial sequences by cumulative log probability. Length normalization prevents systematic preference for short outputs. Finished beams must remain candidates without being expanded. Per-step top-k can be taken over `beam * vocabulary` scores.

Beam search is not sampling. It searches high-probability sequences under the model and may reduce diversity. Diverse beam variants, constraints, or stochastic beams modify the objective.

```python
for step in range(max_steps):
    logits, state = model.step(beams.tokens, beams.state)
    scores = beams.scores[:, None] + log_softmax(logits)
    candidates = top_k(scores.reshape(-1), beam_width)
    beams = gather_parent_state_and_append(candidates, state)
    if beams.all_finished():
        break
return rank_with_length_penalty(beams)
```

### Nearest-neighbor search

Exact search computes all distances. Approximate methods reduce query work through graphs, inverted files, product quantization, or trees. The design must specify recall target, latency, update rate, filtering, memory, and metric.

Embedding normalization makes cosine similarity equivalent to inner product ranking. Product quantization compresses vectors and uses lookup tables for approximate distance, trading recall for capacity and bandwidth. Metadata filters can destroy index efficiency if applied after retrieval; integrate filters or route to partitions.

:::callout pitfall|Complexity without shapes is incomplete
`O(nkd)` does not reveal whether the implementation is a dense GEMM, sparse reduction, or cache-unfriendly loop. State shapes, memory layout, and the operation the hardware will actually execute.
:::

### Principal Interview Review

1. Derive stable logistic loss from logits.
2. How do you distribute k-means updates?
3. Implement beam search state gathering correctly.
4. Explain why the reference attention code is memory-heavy.
5. Choose an ANN index for high update rate and metadata filtering.

## A Principal System Design Method

LEAD: The best system design answer is a sequence of decisions tied to requirements. A diagram is evidence of that reasoning, not a substitute for it.

### Step 1: define the contract

Clarify users, operations, scale, latency, consistency, availability, durability, privacy, compliance, cost, and evolution. For ML systems, add model quality, freshness, feedback, evaluation, and failure containment.

Convert vague statements into working numbers. If the interviewer withholds them, declare reasonable assumptions and make the design parameterized. Distinguish hard requirements from preferences.

### Step 2: estimate

Estimate storage, throughput, state, bandwidth, and hotspots. Use powers of ten. Identify peak-to-average and read/write ratio. For model systems, estimate training tokens, model bytes, KV state, feature volume, embedding count, or evaluation cost.

The purpose is to choose architecture, not impress with arithmetic. Recompute when an assumption changes.

### Step 3: draw the baseline

Show clients, API, durable state, compute workers, queues or streams, indexes or caches, and observability. Draw trust and failure boundaries. Assign ownership of each state transition.

Start with the simplest design that meets the contract. Premature multi-region, five cache tiers, or exotic consensus consumes time without proving judgment.

### Step 4: walk the critical path

Trace one write and one read. For training, trace one example from acquisition to a checkpoint and evaluation. For inference, trace prompt admission to streamed token. State where retries, ordering, idempotency, and version checks occur.

### Step 5: stress the design

Apply load, skew, failure, change, and abuse:

- burst traffic and hot keys;
- long-tail request sizes;
- partial network partition;
- corrupted or late data;
- model or schema rollout;
- dependency slowdown;
- malicious tenant;
- cost or capacity shortfall.

### Step 6: close the loop

Define metrics, alerts, experiments, rollback, and capacity planning. State unresolved risks and the next experiment. This is often where a Staff answer becomes Principal: the design becomes operable and evolvable.

:::callout decision|Name what you would not build yet
Restraint is a design skill. Defer components whose complexity is not justified by the stated scale or risk, and identify the trigger that would cause you to add them.
:::

### Tradeoff matrix

| Decision | Option A | Option B | Flip condition |
| --- | --- | --- | --- |
| Sync vs async | Immediate consistency | Isolation and smoothing | User needs result in request path |
| Push vs pull | Low update latency | Consumer-controlled load | Fanout or offline consumers dominate |
| Replicate vs shard | Availability, simple reads | Capacity, write scaling | One node or replica no longer fits |
| Exact vs approximate | Strong semantics | Lower cost or latency | Bounded error is product-acceptable |
| Shared vs isolated | High utilization | Failure and security isolation | Noisy neighbor or compliance risk dominates |

### Communication during the interview

Signpost transitions: "I will first establish the workload, then estimate the dominant state, then draw a baseline and stress it." Keep a visible list of requirements and risks. When interrupted, answer the question and return to the structure.

If new information invalidates the design, revise it. Defending an obsolete choice signals rigidity, not leadership.

### Principal Interview Review

1. Run the method on a feature store or evaluation service.
2. Which estimates change a vector database architecture?
3. How do you show failure domains on a whiteboard?
4. What makes an approximation acceptable?
5. Name a component you would defer in a new LLM platform and the trigger to add it.

## RAG, Vector Search, and Evaluation Pipelines

LEAD: Retrieval-augmented generation is a data and evaluation system wrapped around a model. Retrieval quality, access control, freshness, and citation behavior matter as much as the generator.

### Ingestion

Connectors fetch documents under source-specific permissions. Normalize without destroying structure. Chunk using semantic and layout boundaries while retaining document, section, time, and ACL metadata. Generate embeddings under a versioned model and write both canonical content and index entries.

Updates need identity and deletion. A replaced document should tombstone or supersede old chunks. Permission changes must propagate to caches and indexes. Store content hashes to avoid re-embedding unchanged chunks.

### Retrieval

Hybrid retrieval combines lexical and dense signals. Lexical search handles rare names and exact identifiers; dense search handles semantic paraphrase. A reranker spends more compute on a small candidate set.

Filters should be enforced before content leaves the retrieval trust boundary. Post-filtering an approximate index can reduce recall; use filter-aware partitions, bitsets, or over-retrieval with a measured bound.

Query rewriting and decomposition can improve recall but also drift intent. Preserve original query, log transformations, and evaluate each stage.

### Context assembly

Select chunks under a token budget, remove redundancy, preserve useful order, and include source identifiers. Position effects can make evidence in the middle less influential. Context compression saves tokens but can remove qualifications or provenance.

Prompt injection in retrieved content is an adversarial input. Separate instructions from evidence, apply source trust policy, constrain tool use, and evaluate injection attacks. The model should cite evidence and abstain when evidence is insufficient.

### Evaluation

Decompose end-to-end quality:

- retrieval recall at k for required evidence;
- reranker quality;
- context precision and redundancy;
- answer correctness;
- faithfulness to provided evidence;
- citation correctness and completeness;
- abstention quality;
- latency and cost by stage;
- freshness and ACL correctness.

An answer can be correct despite failed retrieval due to model memory, or incorrect despite good retrieval due to generation. Stage metrics localize the failure.

### Serving architecture

Use an API layer, query planner, lexical and vector retrieval, metadata/ACL service, reranker, context builder, generation service, and evaluation/telemetry pipeline. Cache public or tenant-scoped retrieval carefully. Version embeddings and indexes; support dual-read during migration.

For multi-region, decide whether indexes are replicated, partitioned by data residency, or queried remotely. Freshness and deletion propagation may control the topology more than query latency.

:::callout pitfall|RAG does not guarantee grounding
Providing evidence changes the model input; it does not force the output to follow that evidence. Measure faithfulness, citations, conflict handling, and abstention directly.
:::

### Principal Interview Review

1. Design a secure RAG system for enterprise documents.
2. How do you evaluate retrieval and generation separately?
3. What happens when an embedding model changes?
4. How do ACL filters interact with approximate search?
5. Design defenses against prompt injection in retrieved content.

