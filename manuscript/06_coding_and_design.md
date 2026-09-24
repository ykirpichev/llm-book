# Part VI - Coding and System Design

Part VI follows two paths that meet in production services. **Streaming-state primitives** develop exact windows, sampling, sketches, and mergeable statistics, then assemble them into telemetry and control loops. **ML implementation and system design** turns numerical assumptions into code, applies a repeatable review method, and builds retrieval and bounded-agent systems. Readers focused on application architecture may begin at **RAG, Vector Search, and Evaluation Pipelines** and return to the streaming chapters for the state, approximation, and failure models those services depend on.

Suppose the serving fleet's token-gap SLO begins to regress. Which tenants are affected? Are long prompts responsible, or one hot error signature? Answering those questions requires summaries of request events across workers, including late and replayed events. The first four chapters build that telemetry path. The ML algorithms review is an optional implementation refresher; the remaining chapters then use the same care with state and evidence to design the documentation assistant and its bounded actions.

This part was reviewed and expanded on September 24, 2026. The additions connect established algorithms to current telemetry, multimodal retrieval, and recoverable agent services. Dated implementation examples are selective; they do not imply a universal best framework or a new benchmark ranking.

## Exact Streaming Queries and Time Windows

LEAD: A stream is a sequence that may be too large, too fast, or too expensive to replay. A correct design begins with an explicit query and error contract, then retains the minimum state that satisfies it. The algorithm, window semantics, merge rule, and failure policy must agree.

### Start with the streaming contract

Let the stream be updates `u_1, u_2, ...`. An update may be:

- an insertion `(key, value, event_time)`;
- a weighted increment `(key, delta)`;
- a replacement of a previously identified record;
- a deletion or negative increment;
- a duplicate delivery of an earlier update.

Those cases are not interchangeable. A cash-register stream permits only nonnegative increments. A strict-turnstile stream permits negative increments but requires every true frequency to remain nonnegative. A general-turnstile stream permits signed frequencies. Standard linear Count-Min Sketch has its familiar one-sided guarantee for nonnegative frequencies, including valid strict-turnstile updates; it does not have that guarantee for arbitrary signed frequencies. A heap over immutable records does not implement updates or deletion.

Define the query before the data structure:

- top `k` immutable records by score;
- top `k` keys by frequency;
- maximum over the most recent `w` arrivals;
- maximum over an event-time interval;
- mean or variance since a checkpoint;
- a uniform sample of all records seen;
- approximate membership, cardinality, frequency, or quantile.

Then define the contract along six dimensions:

| Dimension | Questions that change the algorithm |
| --- | --- |
| Scope | All history, count window, event-time window, or exponential decay? |
| Accuracy | Exact, deterministic rank error, one-sided additive error, or probabilistic relative error? |
| Updates | Insert-only, replacements, deletions, or signed increments? |
| Ordering | In arrival order, event-time order, or no ordering guarantee? |
| Distribution | Must states merge exactly? Are partitions disjoint? Can records move between shards? |
| Recovery | Are updates replayable? Are sources idempotent? Must snapshots align with offsets? |

The space lower bound follows the query. Exact arbitrary-frequency counts require state proportional to the number of distinct keys. Exact arbitrary quantiles can require retaining the stream. Bounded memory therefore usually means a restricted window, a restricted input model, or an approximation.

### Notation and error vocabulary

The four-chapter streaming sequence uses the following notation:

- `N` is the number of processed updates or the current window width, as stated locally;
- `f(x)` is the true nonnegative frequency of key `x`;
- `F_1 = sum_x f(x)` is total frequency mass;
- `D = |{x : f(x) > 0}|` is distinct cardinality;
- `R(v) = |{i : x_i <= v}|` is the rank of value `v`;
- `p` is a probability and `delta` is usually a failure-probability budget;
- `epsilon` is an application-selected error tolerance, not machine epsilon.

An additive frequency bound `|f_hat(x) - f(x)| <= epsilon F_1` scales with total mass. A relative bound `|f_hat(x) - f(x)| <= epsilon f(x)` scales with the queried frequency and is much stronger for rare keys. A rank bound `|R_hat(v) - R(v)| <= epsilon N` says nothing directly about numerical value error. A false-positive rate is a probability over queries and hash construction, not a count-error interval.

Also state the scope of probability:

- **pointwise:** a fixed query chosen independently of the sketch succeeds with probability at least `1 - delta`;
- **simultaneous:** all queries in a declared set succeed together with probability at least `1 - delta`;
- **empirical:** an observed error percentile from experiments, not a mathematical guarantee.

To convert a pointwise failure probability to a simultaneous guarantee over `Q` fixed queries, a union bound can allocate `delta/Q` to each query. Adaptive queries chosen after inspecting outputs need a separate analysis; blindly reusing the pointwise statement can be invalid.

Finally, "mergeable" does not mean byte-for-byte identical to processing one serial stream. It means a documented merge operation produces a summary of the multiset union while preserving the promised error contract. Floating-point states, randomized compaction, and order-sensitive implementations can produce different internal bytes while remaining valid.

### Exact top-k for immutable records

Suppose each record has a total ordering key `(score, tie_breaker)`. The tie breaker must be deterministic and unique enough for the application, such as a record ID. Define "larger" as better. Maintain a min-heap `H` containing at most `k` records.

Assume finite scores and unique, comparable IDs; duplicate deliveries must be deduplicated or assigned distinct occurrence IDs according to the query. The invariant after processing the first `i` accepted records is:

> `H` contains exactly the best `min(i, k)` records among the prefix, and `H[0]` is the worst retained record.

The update rule follows directly:

1. If `len(H) < k`, insert the record.
2. Otherwise compare it with `H[0]`.
3. If it is not better, discard it.
4. If it is better, replace the root.

Example status: Illustrative Python excerpt; not standalone.

```python
from math import isfinite
from heapq import heappush, heapreplace

def exact_top_k(stream, k):
    if k < 0:
        raise ValueError("k must be nonnegative")
    if k == 0:
        return []

    heap = []  # (score, stable_id, payload)
    for item in stream:
        if not isfinite(item.score):
            # A production API must choose reject, canonical ordering, or alert.
            continue
        entry = (item.score, item.id, item)
        if len(heap) < k:
            heappush(heap, entry)
        elif entry > heap[0]:
            heapreplace(heap, entry)

    return [entry[2] for entry in sorted(heap, reverse=True)]
```

#### Correctness proof

Use induction on the prefix length.

- **Base case:** before any record, the heap contains the best zero records.
- **Inductive step:** assume the invariant holds after `i - 1` records. If the heap has fewer than `k` records, adding record `i` clearly gives the best prefix of size `i`. Otherwise, `H[0]` is the `k`th-best retained record. If the new record is no better than `H[0]`, at least `k` prefix records are at least as good, so the new record cannot belong to the top `k`. If it is better, the old root becomes rank `k + 1` or worse and replacing it produces exactly the new top `k`.

Heap construction for the first `k` records costs `O(k)` with bottom-up heapify or `O(k log k)` with repeated insertion. Each of the remaining `n - k` records costs `O(1)` for comparison and `O(log k)` only when it enters the heap. The worst-case bound is `O(n log k)` time and `O(k)` state. Producing sorted output adds `O(k log k)`.

A full sort costs `O(n log n)` and retains `O(n)` records. Selection by partitioning can find the threshold in expected `O(n)` time for a materialized finite array, but it is not a bounded-state one-pass streaming algorithm. The heap is the appropriate choice when `k << n`, arrivals are incremental, and only retained records need storage.

#### Worked heap trace

Let `k = 3` and let the total-order keys arrive as `(8,a), (2,b), (5,c), (9,d), (5,e)`, where a larger score wins and the ID breaks ties. Heap storage order is an implementation detail; the retained sets evolve as:

| Arrival | Retained top set | Boundary root |
| --- | --- | --- |
| `(8,a)` | `{(8,a)}` | `(8,a)` |
| `(2,b)` | `{(2,b), (8,a)}` | `(2,b)` |
| `(5,c)` | `{(2,b), (5,c), (8,a)}` | `(2,b)` |
| `(9,d)` | `{(5,c), (8,a), (9,d)}` | `(5,c)` |
| `(5,e)` | `{(5,e), (8,a), (9,d)}` | `(5,e)` |

The last replacement depends on the declared ID order. If equal scores should preserve first arrival, use arrival sequence as an inverse tie breaker. If the product must return every record tied with the kth score, a fixed-size heap is only the first phase: find the kth score, then retain or replay all records at that score.

#### Boundary semantics

Production correctness includes details hidden by the asymptotic result:

- Define whether ties at the boundary return exactly `k` records or all tied records.
- Reject or order `NaN`; language comparisons involving `NaN` can violate heap assumptions.
- Include version and score definition in serialized state.
- Decide whether payloads live in the heap or the heap stores IDs into durable storage.
- Bound record size; `O(k)` records is not `O(k)` bytes when payloads vary.
- Make query snapshots atomic relative to updates if readers require a coherent top `k`.

### Exact distributed top-k

Let shards partition immutable records into disjoint sets `S_1, ..., S_p`. Let `T_k(S)` denote the best `k` records in set `S`. The coordinator can compute:

`T_k(S_1 union ... union S_p) = T_k(T_k(S_1) union ... union T_k(S_p))`.

To prove this, take a record `x` not in `T_k(S_j)` for its owning shard. At least `k` records in `S_j` rank above `x`; those same records exist globally. Therefore `x` cannot be in the global top `k`. Every possible global winner is present in some local top `k`, so taking top `k` of at most `p k` candidates is exact.

Each shard spends `O(n_j log k)` time and sends `O(k)` records. A coordinator heap costs `O(p k log k)` time and `O(k)` working memory if candidates stream in. A tree reduction uses the same compositional rule and avoids one coordinator receiving all `p k` candidates at once.

The proof fails when:

- records are replicated and duplicates are not deduplicated;
- scores change after local emission;
- a key's frequency is split across shards and must be summed;
- shards use inconsistent score versions or tie breakers;
- the query is "top keys by global count" rather than "top immutable records."

For global frequency top-k, first aggregate each key's partial counts, or use a mergeable frequency summary with a stated error bound. Local top-k by partial count is not generally exact: a globally frequent key can rank just below `k` on every shard while its sum exceeds every local winner.

A minimal counterexample uses `k = 1`. Shard 1 observes `A:6, X:5`; shard 2 observes `B:6, X:5`. The local winners are `A` and `B`, so merging only local top-1 candidates returns frequency six. Globally, `X` has frequency ten and is the true winner. Record top-k composes because each record has one owner and one score; frequency top-k does not compose until partial counts for the same key are combined.

### Mutable scores, updates, and deletions

A size-`k` heap alone loses information needed when a retained record is deleted or its score decreases. Common exact patterns include:

- a map from ID to current version plus a heap with lazy stale-entry removal;
- two indexed heaps or an order-statistics tree when arbitrary updates and rank queries are frequent;
- a durable full state store plus a materialized top-k view;
- periodic rebuilds when lazy garbage grows beyond a threshold.

Lazy deletion stores `(score, id, version)` in the heap and the current version in a map. Before reading or replacing the root, pop entries whose version is stale. This removes obsolete versions but does not recover candidates that a size-k heap already discarded. Exact arbitrary decreases and deletions require all live candidates in a score index, or an authoritative store from which to refill the top-k view. Stale versions add further memory when updates outpace cleanup; rebuild when physical heap size exceeds a multiple of live state.

For windowed top-k, expiration is also deletion. Exact implementations often combine a time-indexed expiration structure with a score-indexed structure. A single heap ordered by score cannot efficiently find all expired items, and a heap ordered by time cannot answer top-k efficiently. State and index count must be included in the design estimate.

### Sliding windows and the monotonic deque

A count window of width `w` at arrival index `i` contains indices `[i - w + 1, i]`. A monotonic deque gives the maximum using only candidates that can still win.

Maintain two invariants:

1. indices increase from front to back;
2. values strictly decrease from front to back.

Example status: Illustrative Python excerpt; not standalone.

```python
from collections import deque

def sliding_max(values, window):
    if window <= 0:
        raise ValueError("window must be positive")

    q = deque()  # (arrival index, value), values decrease front to back
    for i, value in enumerate(values):
        while q and q[0][0] <= i - window:
            q.popleft()                 # expired
        while q and q[-1][1] <= value:
            q.pop()                     # dominated
        q.append((i, value))
        if i + 1 >= window:
            yield q[0][1]
```

When a new value `x_i` arrives, any older candidate at the back with value `<= x_i` is dominated: `x_i` is at least as large and expires later, so the older value can never again be the maximum. After removing expired and dominated entries, the front is the largest live candidate.

#### Amortized analysis

One update can pop many entries, so its worst-case time is `O(w)`. Across `n` updates, however, each index is:

- appended exactly once;
- removed from the back at most once;
- removed from the front at most once.

The total number of deque operations is at most `3n`, so total work is `O(n)` and amortized work is `O(1)` per update. The deque stores at most `w` index/value pairs and accepts an iterator without retaining the full input. Values must have a consistent total order; reject or explicitly order NaNs. A minimum uses the reversed comparison. Returning both minimum and maximum uses two deques.

The deque assumes an arrival-order count window. It does not directly solve out-of-order event-time windows because a late event can be inserted into the middle of the logical order and can invalidate previously emitted results.

### Approximate sliding counts with exponential histograms

The monotonic deque exploits domination and is exact for min or max. A different problem is counting the number of ones among the last `W` bits. An exact algorithm may need enough information to know whether every expiring bit was zero or one, which requires `Theta(W)` bits in the worst case. An exponential histogram compresses old arrivals while keeping recent arrivals precise.

Each bucket represents a consecutive group of one-bits and stores:

- a size that is a power of two;
- the timestamp of the bucket's most recent one-bit.

Buckets are ordered newest to oldest. The following conservative variant makes the relative-error bound simple to prove: for `0 < epsilon < 1`, choose `b = ceil(1/epsilon) + 1` and keep at most `b` buckets of each size. On arrival:

1. expire buckets whose newest timestamp is outside the window;
2. ignore a zero, or create a newest size-one bucket for a one;
3. if a size has `b + 1` buckets, merge its two oldest buckets;
4. give the merged bucket double size and the newer of the two timestamps;
5. cascade the same rule to larger sizes.

Return zero when no buckets remain. A size-one oldest bucket is known exactly from its timestamp, so return the full sum in that case. Otherwise sum every bucket size but count only half of the oldest bucket:

`count_hat = total_bucket_size - oldest_bucket_size / 2`.

Every bucket except the oldest is fully inside the window. Only the oldest bucket can straddle the boundary. If its size is `C`, the estimate's absolute error is at most `C/2`.

Why is that a relative-error bound? A bucket of size `C >= 2` is created by merging two size-`C/2` buckets when there are `b + 1` of them. At least `b - 1` newer buckets of size `C/2` remain, contributing `(b - 1) C/2` ones. Their mass stays newer through subsequent merges and cannot expire while this older bucket remains. Thus:

`(C/2) / (1 + sum_newer_bucket_sizes) <= 1/(b - 1) <= epsilon`.

The `1` is justified because the boundary bucket's newest one has not expired; otherwise the whole bucket would have been deleted. The true live count is at least that one plus all fully live newer buckets. Therefore:

`|count_hat - true_count| / true_count <= (C/2) / (1 + sum_newer_bucket_sizes) <= epsilon`.

This variant uses more buckets than tightly optimized exponential histograms but keeps the same asymptotic bound. There are `O((1/epsilon) log W)` buckets. A size needs `O(log log W)` bits and a timestamp stored modulo a suitable multiple of `W` needs `O(log W)` bits, giving `O((1/epsilon) log^2 W)` bits. Unbounded absolute timestamps instead grow with stream lifetime. With constant-time access to the oldest buckets of each size, cascading can take `O(log W)` for one arrival but is `O(1)` amortized because every merge reduces bucket count.

#### Worked exponential-histogram query

For `epsilon = 0.5`, choose `b = 3`. After 24 consecutive ones, one reachable bucket state from newest to oldest is `[1, 1, 2, 2, 2, 4, 4, 8]`. Suppose the size-eight bucket crosses the window boundary. The stored total is `24`; the estimate is `24 - 8/2 = 20`. The true contribution of the boundary bucket is somewhere from one through eight, so the true count is from `17` through `24`. The worst absolute error is four, exactly half the uncertain bucket; the relative error stays within the stated bound.

This is a count-window algorithm. Event-time disorder still needs watermark and revision semantics. Also do not use the relative guarantee to hide near-zero behavior: when the true count is tiny, an exact sparse representation may be simpler and more useful.

### Event time, watermarks, and late data

:::diagram event_time|Arrival order and event timestamps disagree. With the illustrative watermark at 18, the window ending at 20 has not reached its completion boundary. Allowed lateness and the late-update policy determine retention and corrections.

Streaming systems distinguish:

- **event time:** when the source says the event occurred;
- **ingestion time:** when the platform first accepted it;
- **processing time:** when an operator executes it.

A fixed event-time window of width `T` can assign timestamp `t` to `[floor(t / T) T, (floor(t / T) + 1) T)`. A sliding window has a width and a smaller slide, so one event may update several windows. A session window groups events separated by less than an inactivity gap.

A watermark `W` is a progress claim: the system does not expect ordinary future arrivals with event time below `W`. It is not proof that no such event can arrive. Once `W` passes a window end, the system may emit an on-time pane. Allowed lateness `L` keeps state until approximately `window_end + L`; later records are dropped, side-output, or sent to a correction workflow.

An event-time contract must define:

1. **assignment:** fixed, sliding, or session windows;
2. **watermark generation:** source offsets, observed timestamp lag, or explicit source progress;
3. **triggers:** early speculative, on-time, and late correction emissions;
4. **accumulation:** each pane contains only new changes or the full revised aggregate;
5. **finalization:** when state is deleted and no further in-place corrections occur;
6. **sink semantics:** append, upsert by `(key, window)`, or retract-and-replace.

If a heavy-hitter result changes after a late event, an append-only sink that prints another list creates ambiguity. An upsert should include `window_id`, `revision`, watermark, completeness state, and estimator parameters. Consumers can then distinguish speculative results from final results.

Watermarks in a multi-partition operator usually follow the minimum non-idle input watermark. One stalled partition can therefore hold every window open. Idleness detection needs a policy because marking a truly delayed partition idle can advance the watermark and make its later records late.

### Retention clocks and replay horizons

Expiration is part of correctness. A processing-time time-to-live (TTL) limits storage according to a runtime clock; an event-time timer expresses when a window may finalize. They are not interchangeable. The [Apache Flink state documentation](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/datastream/fault-tolerance/state/), inspected September 24, 2026, describes state TTL as processing-time based. Check the actual engine/version rather than assuming that a TTL follows watermarks.

Consider a five-minute window ending at 12:05 with 20 minutes of allowed lateness. An outage pauses event-time progress while wall-clock time advances to 12:40. Deleting its state solely because 25 wall-clock minutes elapsed can destroy the state needed to accept a replayed 12:04 event. Restore the watermark and source positions with the state; either retain the window until its declared event-time finalization or mark the result incomplete and rebuild from a retained source.

Deduplication needs its own horizon. If operators may replay seven days of input but event-ID receipts expire after one day, a two-day-old event can count twice. Longer receipts cost storage; rebuilding a historical window in a separate output epoch can be cheaper. Bind output identity to that epoch and do not let an old worker overwrite a repaired result merely because its local revision counter is larger. A monotonically ordered epoch, or a fenced writer lease, must be checked at the sink.

**Worked check:** a watermark remains at 12:03 while the runtime clock reaches 12:40. Is the 12:00-12:05 window final? No. Its progress contract has not reached the window end, much less the lateness boundary. A storage cap can force an incomplete result, but cannot turn missing evidence into completeness.

### Check your design: exact queries

Before moving on, explain why local top-k can be merged for immutable records but not by simply summing local frequency ranks. Then specify what a late event changes in an already emitted time window. A complete answer names a total order for records, retained candidate evidence for frequencies, and a revision-aware sink for windows. The cumulative exercises later in this part give worked solutions.

## Online Statistics and Sampling

LEAD: Moments, samples, and tail or window summaries answer different questions. Stable moments describe center and spread; uniform samples preserve inclusion probabilities. Each needs a contract for the quantity and time range it represents.

The runnable `examples/streaming.py` reference includes mergeable Welford moments and exact immutable-record top-k. Its tests compare moments against Python's independent statistics implementation and compare partitioned top-k against a full sort. The remaining algorithms below are derivations and illustrative excerpts, not implementations certified by that suite.

### Online mean and variance

For observations `x_1, ..., x_n`, define:

- count `n`;
- mean `mu_n = (1/n) sum_i x_i`;
- centered sum of squares `M2_n = sum_i (x_i - mu_n)^2`.

The population variance is `M2_n / n`. The unbiased sample variance is `M2_n / (n - 1)` for `n >= 2`.

The algebraically simple formula `variance = mean(x^2) - mean(x)^2` can catastrophically cancel when the variance is small relative to the squared mean. For values near `10^9` separated by units, both terms are near `10^18`; subtracting them can discard the meaningful low-order bits.

#### Deriving Welford's update

Assume state `(n, mu_n, M2_n)` and a new observation `x`. Let `n' = n + 1` and `delta = x - mu_n`.

The new mean is:

`mu_n' = (n mu_n + x) / n' = mu_n + delta / n'`.

Define `delta2 = x - mu_n'`. Since `mu_n' = mu_n + delta / n'`, we have `delta2 = delta * n / n'`.

For the old observations, shifting the center from `mu_n` to `mu_n'` gives:

`sum_{i=1}^n (x_i - mu_n')^2 = M2_n + n (mu_n - mu_n')^2`,

because the cross term contains `sum_i (x_i - mu_n) = 0`. Adding the new observation and simplifying yields:

`M2_n' = M2_n + delta * delta2`.

That produces the stable update:

Example status: Illustrative Python excerpt; not standalone.

```python
def update(state, x):
    n, mean, M2 = state
    n2 = n + 1
    delta = x - mean
    mean2 = mean + delta / n2
    delta2 = x - mean2
    M2_2 = M2 + delta * delta2
    return n2, mean2, M2_2
```

The method avoids subtracting two nearly equal large accumulated moments. It does not make floating point exact: use an adequate accumulator dtype, reject non-finite inputs according to policy, and consider compensated summation or pairwise reduction for extreme dynamic range.

#### Merging two variance states

Let disjoint partitions `A` and `B` have states `(n_A, mu_A, M2_A)` and `(n_B, mu_B, M2_B)`. Let `n = n_A + n_B` and `delta = mu_B - mu_A`.

The combined mean is:

`mu = mu_A + delta * n_B / n`.

The combined centered sum is:

`M2 = M2_A + M2_B + delta^2 * n_A n_B / n`.

The final term accounts for separation between partition means. It follows by writing each deviation from the combined mean as `(x - mu_A) + (mu_A - mu)` or `(x - mu_B) + (mu_B - mu)`. Within-partition cross terms vanish because centered deviations sum to zero.

This merge is associative over exact arithmetic and therefore supports tree reduction. Floating-point results still depend slightly on merge order. Balanced pairwise merging usually limits error better than repeatedly merging a huge state with a tiny state.

Handle empty states explicitly. For vector features, `mean` and elementwise `M2` have feature shape. A full covariance needs a matrix outer-product accumulator and costs `O(d^2)` state, not `O(d)`.

#### Weighted updates and removals

For positive weights, store total weight `W`, weighted mean `mu`, and:

`M2 = sum_i w_i (x_i - mu)^2`.

Adding `(x, w)` is the same as merging the current state with a one-point state:

`W' = W + w`,

`delta = x - mu`,

`mu' = mu + (w/W') delta`,

`M2' = M2 + w delta (x - mu') = M2 + (W w/W') delta^2`.

Two weighted states merge by replacing counts `n_A, n_B` in the parallel formula with total weights `W_A, W_B`.

If an exactly known point `(x, w)` leaves a window and `W' = W - w > 0`, reverse the update:

`mu' = (W mu - w x) / W'`,

`M2' = M2 - w (x - mu)(x - mu')`.

Removal is more sensitive to floating-point error, especially when `W'` is tiny relative to `W`. Periodically rebuilding from live window state can be safer. If the removed point is not stored exactly, Welford state alone cannot reconstruct it.

Roundoff can leave a tiny negative `M2` after removal or merging. Clamp only a value within a justified floating-point tolerance of zero; a materially negative result indicates corrupted state, mismatched removal, or an implementation error.

"Sample variance" for weights is not one universal formula. Frequency weights model repeated observations; reliability or inverse-variance weights use a different effective degrees-of-freedom correction. State the statistical interpretation before dividing weighted `M2`.

#### Worked variance trace

For `[4, 7, 13, 16]`, the state evolves:

| New value | `n` | `mean` | `M2` |
| --- | ---: | ---: | ---: |
| `4` | 1 | 4 | 0 |
| `7` | 2 | 5.5 | 4.5 |
| `13` | 3 | 8 | 42 |
| `16` | 4 | 10 | 90 |

The population variance is `90/4 = 22.5`; the unbiased sample variance is `90/3 = 30`. Splitting into `A=[4,7]` and `B=[13,16]` gives means `5.5` and `14.5`, with `M2_A=M2_B=4.5`. The merge correction is `(14.5-5.5)^2 * 2*2/4 = 81`, so combined `M2 = 4.5 + 4.5 + 81 = 90`.

### Uniform reservoir sampling

The goal is a simple random sample without replacement of size `min(k, N)` from a stream whose final length `N` is unknown. For `N >= k`, every size-k subset must be equally likely; each item then has inclusion probability `k / N`. If `N < k`, retain every item.

Algorithm R fills the reservoir with the first `k` items. For the item at one-based position `i > k`, draw `j` uniformly from `{1, ..., i}`. If `j <= k`, replace reservoir slot `j`; otherwise discard the item.

Example status: Illustrative Python excerpt; not standalone.

```python
def reservoir_sample(stream, k, rng):
    if k < 0:
        raise ValueError("k must be nonnegative")
    reservoir = []
    for i, item in enumerate(stream, start=1):
        if i <= k:
            reservoir.append(item)
        else:
            j = rng.randint(1, i)  # inclusive
            if j <= k:
                reservoir[j - 1] = item
    return reservoir
```

#### Uniformity proof

Proceed by induction on processed length `i`.

- After `i = k`, every item is present with probability `1 = k/k`.
- Assume every earlier item is present after `i - 1` updates with probability `k/(i - 1)`.
- New item `i` is inserted exactly when `j <= k`, which has probability `k/i`.
- An earlier retained item is replaced only when the algorithm chooses its particular slot. That occurs with probability `1/i`, so it survives the update with probability `(i - 1)/i`.

Therefore an earlier item's final inclusion probability is:

`(k / (i - 1)) * ((i - 1) / i) = k / i`.

The invariant holds for all `i >= k`, and at `N >= k` every item has probability `k/N`. Equal marginal probabilities alone do not prove uniform subsets. For a fixed size-k subset after step `i`, if it excludes item `i`, its probability is `(1 - k/i) / choose(i-1, k) = 1 / choose(i, k)`. If it includes item `i`, there are `i-k` possible predecessor subsets, each replacing its one extra member with probability `1/i`; the total is again `(i-k) / (i * choose(i-1, k)) = 1 / choose(i, k)`. This establishes the stronger uniform-subset invariant.

The algorithm processes every record and makes one random draw after the reservoir fills. Skip-based reservoir algorithms improve constants for very long streams by sampling how many records to skip before the next replacement, while preserving the same sample distribution.

#### Distributed sampling

Concatenating arbitrary per-shard reservoirs and sampling uniformly from their items is biased when shard sizes differ. A record on a small shard has a larger chance of entering its local reservoir.

Two exact approaches are:

- assign every record an independent continuous random priority derived from a stable record ID and sampling seed, retain the `k` smallest priorities per shard, then retain the global `k` smallest priorities; or
- merge local uniform reservoirs with weights derived from shard counts using the appropriate multivariate hypergeometric allocation.

The priority approach has the same compositional proof as distributed top-k: a record outside a shard's local `k` smallest priorities cannot be among the global `k` smallest. Stable hash-derived priorities also make replay deterministic, but only if record identity and seed are stable and the hash behaves as a suitable pseudorandom function.

For the shard-allocation approach, let shard sizes be `n_1, ..., n_p`, total `N`, and let `s_j` be the number of final sample records drawn from shard `j`, with `sum_j s_j = k`. A uniform global subset induces the multivariate hypergeometric law:

`P(s_1, ..., s_p) = product_j choose(n_j, s_j) / choose(N, k)`.

First draw the allocation vector from this distribution. Then choose a uniform size-`s_j` subset from each shard. A local uniform reservoir of size `min(k, n_j)` is sufficient: choosing a uniform `s_j` subset of that reservoir is itself uniform over size-`s_j` subsets of the shard. This method requires trustworthy shard counts and coordinated random allocation.

#### Weighted sampling without replacement

For positive item weights `w_i`, "weighted" needs a target law. One common law chooses each next item with probability proportional to its weight among items not yet selected. Generate independent `U_i` uniform on `(0,1]`, define:

`priority_i = -ln(U_i) / w_i`,

and retain the `k` smallest priorities. Since `-ln(U_i)` is exponential with rate one, dividing by `w_i` gives an exponential random variable with rate `w_i`. For `k = 1`, the probability that item `i` has the minimum clock is:

`P(i first) = w_i / sum_j w_j`.

The memoryless property of exponential clocks makes the next minimum proportional to the remaining weights, producing sequential probability-proportional-to-size sampling without replacement.

The priority is composable across shards: keep local `k` smallest, then global `k` smallest. Compute `-ln(U)/w` rather than `U^(1/w)` to avoid numerical underflow. Zero or negative weights are outside this model. Inclusion probability is not generally `k w_i / sum_j w_j`; that expression can exceed one and ignores without-replacement dependence.

Sampling with replacement, weighted sampling, and time-decayed sampling have different distributions. Do not label them "reservoir sampling" without naming the target probability and random-priority construction.

### Sampled traces are not the request population

[OpenTelemetry sampling guidance](https://opentelemetry.io/docs/concepts/sampling/) distinguishes head decisions, made before a trace completes, from tail decisions that can retain traces based on observed latency or errors. Tail sampling is useful for diagnosis, but its selected traces generally do not represent production traffic uniformly. Measure request counters and SLO histograms before selective trace export, or use a defensible sampling estimator.

Here is an original example. A service handles 10,000 requests: 100 fail and 9,900 succeed. Retain every failure and independently retain each success with probability 0.01. If the realized sample contains 100 failures and 99 successes, the raw sampled error fraction is `100/199 ~= 50.25%`, while the population fraction is 1%.

When each request has a known nonzero inclusion probability `p_i`, estimate a total by `sum(y_i / p_i)` over retained requests. In the example, weighted failures are 100 and weighted requests are `100 + 99/0.01 = 10,000`. Their ratio is 1%. The total estimator is unbiased under its sampling design; a ratio of two estimated totals is not generally unbiased, and this realized sample is unusually convenient. If the complete request count is independently known, divide the estimated failure total by that known count instead.

Keep inclusion probabilities, policy versions, and sampling units with the data. A request-level metric cannot silently use span-level inclusion probabilities. Adaptive quotas, head-plus-tail selection, collector drops, and incomplete traces can make the effective probability unknown. A class sampled with probability zero cannot be recovered by weighting. Report that blind spot, and do not derive a population p99 from a failure-enriched trace sample. Weighted moments and quantiles need estimators that explicitly support those weights.

**Worked check:** does keeping every error make an unweighted sample suitable for an error-rate dashboard? No. It improves diagnostic coverage while altering the denominator. Separate diagnostic trace selection from population measurement.

### Check your design: sampling and moments

When merging two partitions, explain why averaging their means without counts is wrong. For sampling, explain why equal-size samples from unequal-size partitions cannot simply be concatenated and treated as uniform. The answer must weight observations by the populations represented, not by how many summary objects happen to arrive.

## Heavy Hitters and Probabilistic Sketches

LEAD: A sketch trades retained state for a precisely defined error. Frequency, cardinality, membership, and rank are different queries; their guarantees and merge operations cannot be interchanged.

### Deterministic heavy hitters with Misra-Gries

For an insertion-only stream of length `N`, a `phi`-heavy hitter is a key with frequency greater than `phi N`. Exact counts may require one counter per distinct key. Misra-Gries provides deterministic candidates with bounded error.

Choose `r` counters. For each key `x`:

1. if `x` is tracked, increment its counter;
2. else if fewer than `r` keys are tracked, insert `x` with count one;
3. else decrement every counter and delete zeros.

With `r = ceil(1/phi) - 1`, every key with frequency greater than `N/(r + 1)` is retained as a candidate.

Why? Each global decrement can be paired with the untracked arriving item and one occurrence from each of the `r` tracked keys, canceling `r + 1` distinct occurrences. There can be at most `floor(N/(r + 1))` such cancellation rounds. A key absent from the final table must have had all of its occurrences canceled, so its true frequency is at most that number.

If `c_hat(x)` is the final counter for a retained key, then:

`c_hat(x) <= f(x)` and `f(x) - c_hat(x) <= g <= N/(r + 1)`,

where `g` is the number of decrement-all rounds. Also,

`g = (N - sum_x c_hat(x)) / (r + 1)`,

because every ordinary increment increases counter mass by one, while a decrement-all round consumes the arriving untracked item and removes one from each of `r` counters. A second pass over replayable data can compute exact frequencies for the candidates. Without replay, report estimates and bounds, not exact counts.

Misra-Gries is excellent when a deterministic no-false-negative candidate guarantee matters. It does not directly retain the numerical top `k` under arbitrary signed updates or event-time expiration.

#### Reference implementation and cost

Example status: Illustrative Python excerpt; not standalone.

```python
def misra_gries(stream, capacity):
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    counters = {}
    for key in stream:
        if key in counters:
            counters[key] += 1
        elif len(counters) < capacity:
            counters[key] = 1
        else:
            dead = []
            for candidate in counters:
                counters[candidate] -= 1
                if counters[candidate] == 0:
                    dead.append(candidate)
            for candidate in dead:
                del counters[candidate]
    return counters
```

This literal implementation spends `O(r)` time on a decrement-all step, but there are at most `N/(r+1)` such steps. With expected constant-time hash-map operations, total processing time is therefore `O(N)`, with `O(1)` amortized update time and `O(r)` worst-case work for one arrival. Global offsets, counter buckets, or another batched-decrement representation can reduce update spikes. Those optimizations must preserve deletion at logical zero; merely postponing decrements without handling zero crossings changes the candidates.

As a trace, process `a, b, a, c, a, b, d, a` with `r = 2`. The table evolves:

| Arrival | Counters after update | Explanation |
| --- | --- | --- |
| `a` | `{a:1}` | Free counter |
| `b` | `{a:1, b:1}` | Free counter |
| `a` | `{a:2, b:1}` | Tracked increment |
| `c` | `{a:1}` | Cancel `a`, `b`, and arriving `c` |
| `a` | `{a:2}` | Tracked increment |
| `b` | `{a:2, b:1}` | Free counter |
| `d` | `{a:1}` | Cancel `a`, `b`, and arriving `d` |
| `a` | `{a:2}` | Tracked increment |

Here `N = 8`, final counter mass is two, and `g = (8 - 2)/3 = 2`. Thus `2 <= f(a) <= 4`; the true count is four. The estimate is deliberately a lower bound.

#### Merging Misra-Gries states

Suppose two disjoint partitions have the same capacity `r`, stream masses `N_A, N_B`, and counter maps `c_A, c_B`. First add counters keywise:

`c(x) = c_A(x) + c_B(x)`.

There are at most `2r` keys. If there are more than `r`, let `C` be the `(r + 1)`th-largest positive combined counter. Replace every counter by:

`c'(x) = max(c(x) - C, 0)`.

At most `r` positive counters remain. This is a weighted reduction: it is equivalent to performing enough cancellation rounds on the summarized weighted items.

The error proof is worth stating. Define each input error budget:

`Delta_A = (N_A - sum_x c_A(x))/(r + 1)`

and likewise for `B`. Before pruning,

`0 <= f(x) - c(x) <= Delta_A + Delta_B`.

Pruning decreases any one counter by at most `C`, so its error rises by at most `C`. At least `r + 1` combined counters were at least `C`, hence pruning removes at least `(r + 1)C` total counter mass. Therefore the merged budget

`Delta' = (N_A + N_B - sum_x c'(x))/(r + 1)`

is at least `Delta_A + Delta_B + C`, which covers the new pointwise error. The merged state retains the same deterministic contract. Repeated merges remain valid, although they need not produce the same counters as one serial ordering.

#### Space-Saving: tighter candidates in practice

Space-Saving also keeps `r` entries but uses a different replacement rule:

1. increment a tracked key;
2. insert an untracked key with `(estimate=1, error=0)` if a slot is free;
3. otherwise replace a minimum entry of estimate `c_min` by the new key with `(estimate=c_min+1, error=c_min)`.

For a tracked key:

`estimate(x) - error(x) <= f(x) <= estimate(x)`.

For an untracked key, its frequency is at most the current minimum estimate. Unlike Misra-Gries, Space-Saving estimates tracked keys from above and often allocates counters more effectively to skewed streams. It guarantees that every key with frequency greater than `N/r` is tracked.

These intervals can certify a ranking only when they separate. If the kth candidate's lower bound exceeds every other tracked candidate's upper bound and the untracked upper bound, the top-k set is established. Otherwise the result is a candidate ranking, not an exact one.

The ordinary Space-Saving update is nonlinear, so two tables cannot be merged by simply adding entries or taking local winners. Use a published error-aware merge procedure, convert to a summary with a proved reduction rule, or use Misra-Gries when simple deterministic merging is a core requirement. Record which variant is implemented; the shared name alone does not specify merge correctness.

### Count-Min Sketch and its error derivation

:::diagram count_min|Three independent row hashes select counters for a key. Querying takes their minimum; collision mass explains why the result can overestimate a nonnegative frequency. Blank cells contain other counters whose values are irrelevant to this query.

A Count-Min Sketch estimates nonnegative key frequencies using a table with depth `d` and width `w`. Row `j` has an independent pairwise-independent hash `h_j`. An increment `(x, c)` with `c >= 0` adds `c` to cell `(j, h_j(x))` in every row. The estimate is:

`f_hat(x) = min_j table[j, h_j(x)]`.

For a fixed row, the queried counter equals the true frequency plus collision noise:

`C_j(x) = f(x) + sum_{y != x, h_j(y)=h_j(x)} f(y)`.

Because every current frequency is nonnegative, `C_j(x) >= f(x)`, so the estimate never undercounts. The insertion-only stream used in this derivation satisfies that condition directly.

Let total mass be `F_1 = sum_y f(y)`. For a pairwise-independent hash into `w` columns, any other key collides with probability `1/w`. The expected collision noise in one row is at most:

`E[noise_j] <= (F_1 - f(x)) / w <= F_1 / w`.

Choose `w = ceil(e / epsilon)`. Then `E[noise_j] <= epsilon F_1 / e`. By Markov's inequality:

`P(noise_j >= epsilon F_1) <= 1/e`.

The estimate takes the minimum over `d` independent rows, so it exceeds `f(x) + epsilon F_1` only if every row has large collision noise:

`P(f_hat(x) > f(x) + epsilon F_1) <= e^(-d)`.

Choosing `d = ceil(ln(1/delta))` gives, for a fixed queried key:

`f(x) <= f_hat(x) <= f(x) + epsilon F_1`

with probability at least `1 - delta`.

The table uses `O((1/epsilon) log(1/delta))` counters. Update and query cost are `O(log(1/delta))`. Counter width must accommodate total mass; overflow silently destroys the guarantee.

#### Pointwise versus many-query guarantees

The derivation is pointwise: fix `x` independently of the randomly chosen hashes, then ask for its estimate. If a job will issue `Q` fixed queries and wants every one to satisfy the bound with total failure probability at most `eta`, the union bound is:

`P(any query fails) <= Q delta`.

Allocate `delta = eta/Q`, which changes the depth to:

`d = ceil(ln(Q/eta))`.

This is conservative but explicit. It does not automatically cover an adversary that adaptively chooses the next query after seeing previous outputs or keys crafted after learning the hash functions. For exposed services, protect seeds, rotate versions deliberately, and validate the threat model.

As a sizing example, `epsilon = 0.001` and pointwise `delta = 10^(-6)` give:

- `w = ceil(e/epsilon) = 2,719`;
- `d = ceil(ln(1/delta)) = 14`;
- `38,066` counters;
- about `297.4 KiB` with unsigned 64-bit counters, before array and metadata overhead.

If `F_1 = 100,000,000`, the additive error allowance is `epsilon F_1 = 100,000`. That may be excellent for million-count heavy hitters and useless for hundred-count rare keys. Count-Min Sketch does not turn an additive guarantee into a relative one merely because the observed estimate is small.

#### From frequency estimates to top-k

The sketch contains counters but not the original keys, so it cannot enumerate heavy hitters by itself. Pair it with a candidate mechanism:

- a bounded heap or Space-Saving summary updated from observed keys;
- a Misra-Gries candidate table;
- an external dictionary for a constrained key universe;
- a second pass that queries known candidates.

Using sketch estimates as heap priorities can admit false positives due to overestimation. A candidate near the boundary should be reported with its error interval. If the gap between the `k`th and `(k+1)`th true frequency is smaller than the possible sketch error, exact ordering is not guaranteed.

Standard sketches merge by elementwise addition only when width, depth, hash functions, seeds, counter type, and update semantics match. Include those fields in the serialized schema. For disjoint stream partitions, the merged sketch represents the union multiset. It does not deduplicate replicated events.

For an unweighted increment, conservative update raises only counters at the current row minimum and often reduces empirical overestimation. For a weighted nonnegative increment `c`, let `m = min_j C_j(x)` and set every addressed counter to `max(C_j(x), m + c)`; merely adding `c` to counters equal to `m` is correct only when `c = 1`. Conservative update is not the same state transition as ordinary Count-Min. Adding two conservatively updated arrays produces useful combined counters in some implementations, but not the state that serial conservative updates would have produced; use only a merge contract proved by the chosen library and do not silently attach the standard linear-state interpretation.

Valid strict-turnstile deletions preserve this argument for the standard linear sketch: apply the signed delta to every hashed row, and require all true frequencies to remain nonnegative. The bound then uses current mass `F_1`. A sketch cannot itself certify that a deletion was valid; authoritative event or keyed state must enforce that contract. In a general-turnstile stream, negative true frequencies can cancel collision mass and break the minimum estimator's one-sided guarantee. Use a sketch and estimator analyzed for that model, or exact keyed state.

### Cardinality with HyperLogLog

HyperLogLog estimates the number of distinct keys, not total frequency. Hash each key to a uniformly distributed bit string. Use the first `p` bits to choose one of `m = 2^p` registers. In the remaining suffix, let `rho` be one plus the number of leading zeros. Update the selected register:

`M[j] = max(M[j], rho)`.

The intuition begins with one bit string: the event `rho > r` means the first `r` suffix bits are zero, which has probability `2^(-r)`. Seeing a very long zero prefix is therefore evidence of many distinct hashes. Multiple registers reduce variance, and the harmonic mean limits domination by unusually large registers.

The raw estimator is:

`E_raw = alpha_m * m^2 / sum_{j=1}^m 2^(-M[j])`,

where `alpha_m` is a bias-correction constant determined by the estimator analysis. In the original estimator, for `m >= 128`:

`alpha_m = 0.7213 / (1 + 1.079/m)`.

For the usual range and sufficiently large `m`, relative standard error is approximately:

`RSE ~= 1.04 / sqrt(m)`.

Choosing `p = 14` gives `m = 2^14 = 16,384` and:

`RSE ~= 1.04 / 128 = 0.008125 = 0.8125%`.

A plain byte per register therefore uses `16 KiB`, plus metadata; packed implementations can use less. This RSE is a standard deviation, not a hard maximum error. A rough normal-model 95% interval would be about `+/- 1.96 RSE`, but real implementations should expose their calibrated interval and estimator version.

Sparse representations save memory at low cardinality. If `V` registers are still zero, the classical small-range estimate is linear counting:

`E_small = m ln(m/V)`.

For example, with `m = 16,384` and `V = 10,000`, this gives about `8,089` distinct items. Implementations switch estimators using tested thresholds and bias tables; very large ranges also need a hash-width correction. Use the library's published estimator rather than applying every correction unconditionally.

The update itself is idempotent. Repeating an identical canonical key produces the same register index and `rho`, and `max(M[j], rho)` does not change. That means HLL estimates set cardinality despite duplicate occurrences. It does not mean two different events with the same business identifier should always be deduplicated; canonical key choice defines the set being counted.

Two compatible HLL states merge by registerwise maximum because each register summarizes the maximum `rho` seen in that bucket. This produces a cardinality estimate for set union. Inclusion-exclusion, `|A intersect B| = |A| + |B| - |A union B|`, can be numerically poor when the intersection is small relative to estimation error. Use a sketch designed for set expressions when intersections are a primary query.

Ordinary HLL cannot delete a key: a register maximum does not reveal the second-largest `rho` that should replace it. Windowed distinct counting therefore uses separate sketches per pane/window, a specialized time-aware sketch, or recomputation from retained state. Subtracting registers or cardinality estimates is invalid.

Hash width bounds useful cardinality and collision behavior. Seeds, precision `p`, estimator version, and canonical key encoding are part of the state definition. An attacker who can choose hashes can bias registers or create denial-of-service behavior, so exposed systems need keyed hashing or controlled input.

### Bloom filters and the false-positive derivation

A Bloom filter represents approximate set membership in `m` bits using `h` hash locations per inserted key. Insert sets all `h` bits. Query returns "possibly present" if all `h` bits are one and "definitely absent" otherwise.

Under independent uniform hashing, after inserting `n` distinct keys:

- one hash leaves a particular bit zero with probability `1 - 1/m`;
- all `h n` hash placements leave it zero with probability `(1 - 1/m)^(h n)`;
- for large `m`, this is approximately `exp(-h n / m)`;
- a queried bit is therefore one with probability `1 - exp(-h n / m)`;
- all `h` queried bits are one with approximate probability:

`p_fp ~= (1 - exp(-h n / m))^h`.

This is the false-positive probability for a key not inserted. There are no false negatives only if the filter has not lost bits, hashing and encoding are consistent, and no unsupported deletion occurs.

#### Optimal number of hashes

For fixed `m` and expected `n`, let `a = m/n` and minimize:

`log p_fp(h) = h ln(1 - exp(-h/a))`.

Set `y = exp(-h/a)`. Differentiating with respect to continuous `h` gives:

`d(log p_fp)/dh = ln(1-y) + h y/(a(1-y))`.

At `y = 1/2`, the relation `h = a ln 2` makes the two terms `-ln 2` and `+ln 2`, so the derivative is zero. This is the unique interior minimum. Thus the optimum occurs when about half the bits remain zero and:

`h_opt = (m / n) ln 2`.

At that point:

`p_min ~= (1/2)^h = exp(-(m/n) (ln 2)^2) ~= (0.6185)^(m/n)`.

Solving for a desired false-positive rate `p` gives:

`m ~= -n ln p / (ln 2)^2`

and:

`h ~= (m/n) ln 2 = -ln p / ln 2`.

For `n = 10,000,000` and `p = 0.001`, the design needs about `143,775,876` bits, or `17.1 MiB`, and roughly `10` hash probes. Capacity planning must use the number of distinct inserted keys during the filter's lifetime, not request count.

The optimum is real-valued, but an implementation uses an integer. Evaluate the predicted false-positive rate for the two neighboring positive integers and choose based on both probability and CPU cost. The common double-hashing construction derives locations as:

`g_i(x) = (h_1(x) + i h_2(x)) mod m`, for `i = 0, ..., h-1`.

This avoids computing `h` unrelated hashes, but `h_1`, `h_2`, modulus choice, and domain separation must follow a construction with an appropriate independence argument. If `h_2` shares a factor with `m`, locations can cycle through only part of the bit array.

If actual `n` exceeds the design capacity, more bits become one and false positives rise. If observed fraction `q` of bits is one, the same occupancy model estimates:

`n_hat = -(m/h) ln(1-q)`.

Use that as an operational saturation signal, not as an exact distinct counter. Rotate, layer, or rebuild before occupancy approaches one.

Counting Bloom filters replace bits with counters so deletion can decrement locations. They cost more memory and still require exact update discipline: deleting a key that was never inserted, applying a duplicate deletion, or losing an insertion can reduce counters shared by other keys and create false negatives. Stable Bloom filters intentionally forget old membership and therefore change the error contract.

Compatible Bloom filters can be unioned with bitwise OR. Their bitwise AND has no false negatives for keys in the intersection under the same valid-use assumptions, but its false positives are not determined by intersection cardinality alone: keys present in only one input may still pass. Do not treat AND as a freshly built filter sized for the true intersection, or use it to count the intersection. Filters must share `m`, hash count, seeds, and encoding.

### Streaming quantiles

A percentile query asks for a value by rank, not by numeric distance. For sorted values `x_(1) <= ... <= x_(N)`, a target quantile fraction `phi` corresponds to rank near `phi N`, with an explicitly chosen rounding convention. An `epsilon`-approximate quantile summary commonly guarantees that the returned value for target rank `r` has true rank in:

`[r - epsilon N, r + epsilon N]`.

Clamp the interval to `[1, N]`. With duplicates, value `v` occupies ranks from `1 + |{i : x_i < v}|` through `|{i : x_i <= v}|`; a returned value is valid when that interval intersects the allowed target interval. Using only the upper rank would incorrectly reject the median of a stream of identical values. Cumulative-rank queries can still use the earlier `R(v)` convention.

This rank contract does not guarantee a small value error. Consider 500 zeros followed by 500 values equal to one billion. Around the median, a rank error of only ten may still permit returning either endpoint, a numerical difference of one billion. In a dense region, the same rank error may have negligible value effect.

Without assumptions on the value distribution, an exact arbitrary quantile in one pass requires linear state in the worst case: an adversary can arrange unseen order information so that many retained distinctions may determine the final rank. Approximate rank is what permits sublinear summaries.

The deterministic Greenwald-Khanna summary stores ordered tuples that bound the minimum and maximum possible rank of retained values. It periodically compresses adjacent tuples while preserving a maximum rank uncertainty proportional to `epsilon N`. Its worst-case state is `O((1/epsilon) log(epsilon N))` for a stream of length `N`.

#### How KLL compaction works

KLL is a randomized hierarchy of compactors. Level `ell` stores values with implicit weight `2^ell`. New items enter level zero with weight one. When a level exceeds its configured capacity:

1. sort the level;
2. choose odd or even positions with an independent fair coin;
3. discard the other positions;
4. promote the survivors to level `ell + 1`, doubling their implicit weight;
5. recursively compact an overflowing higher level.

For an even buffer of `2s` items, compaction retains `s` items of double weight, so represented total mass remains `2s * 2^ell`. An odd buffer must leave one item at its original level and compact an even subset; otherwise mass changes. For any fixed query threshold `v`, sorted pairing means the number of retained representatives at or below `v`, after doubling, differs from the original prefix count by at most one level-`ell` item. Thus that compaction contributes rank error at most `2^ell`. Choosing odd versus even positions makes the contribution conditionally mean zero. Later buffers depend on earlier choices, so the errors are not simply independent; the capacity schedule and concentration analysis control their accumulation.

A tiny example shows the approximation. Compact level-zero values `[1, 2, 4, 7]`. If the coin retains `[2, 7]`, each survivor has weight two. Estimated cumulative weights at thresholds `1, 2, 4, 7` are `0, 2, 2, 4`, whereas true ranks are `1, 2, 3, 4`; the absolute rank error is at most one, exactly the level-zero bound. Retaining `[1, 4]` gives the opposite signed error.

To query, gather retained items from all levels, attach weight `2^ell`, sort by value, and scan cumulative weight until reaching the target rank. A production sketch caches or incrementally maintains enough structure to avoid rebuilding everything for each query.

Compatible KLL states merge by concatenating corresponding levels and running the same compaction rule until capacities are restored. The merge represents the union's weighted items, but random choices and merge-tree shape can change the retained sample. Compatibility includes the capacity parameter, comparator or numeric encoding, weight semantics, randomization/serialization version, and the library's declared merge contract.

The canonical analysis achieves space logarithmic in the failure-probability logarithm, commonly stated as `O((1/epsilon) log log(1/delta))` words for its core probabilistic quantile contract. Constants, all-quantiles versus single-query guarantees, and confidence interpretation vary by implementation. Size from the selected library's published normalized-rank-error table rather than treating its tuning parameter as literally `1/epsilon`.

### Three different tail guarantees

[KLL](https://datasketches.apache.org/docs/KLL/KLLSketch.html) is a useful general rank sketch. [REQ](https://datasketches.apache.org/docs/REQ/ReqSketch.html) concentrates rank accuracy toward a configured end of the distribution. [DDSketch](https://www.vldb.org/pvldb/vol12/p2195-masson.pdf), a 2019 foundation still relevant to telemetry, uses logarithmic value buckets to bound relative value error for supported quantiles. These solve different measurement problems.

| Requirement | Error quantity | Engineering consequence |
| --- | --- | --- |
| General percentiles | Absolute normalized-rank error | A small rank interval can span a large latency jump |
| Extremely high or low ranks | Rank error concentrated toward the chosen tail | Choose and record which tail the REQ configuration favors |
| Latency values within a percentage | Relative numerical-value error | Specify value range, near-zero treatment, and bucket-collapse policy |

For an original numerical example, a true p99 of 2,000 ms with a valid 1% relative-value guarantee permits 1,980-2,020 ms. A normalized-rank error of 0.01 around p99 instead permits a broad rank neighborhood; it does not promise a 20 ms error. Near zero, a relative-value contract needs a separate absolute tolerance. If a bounded sketch collapses buckets, establish which quantiles retain its guarantee rather than applying the unbounded analysis to every query.

A hard SLO such as "at least 99% of requests finish within 2,000 ms" often needs only a threshold counter: increment total requests and requests meeting that boundary. That avoids estimating a percentile to answer a threshold question. Choose the treatment of cancellation and timeout first; excluding failed requests can make a broken service look faster. Merge compatible histograms or sketches, never average per-worker p99 values.

**Worked check:** two shards have p99 values of 10 ms and 1,000 ms. Is the fleet p99 505 ms? No. Those two numbers omit counts and the distributions needed to locate the fleet's 99th percentile.

#### Choosing a quantile sketch

Distribution-oriented structures such as t-digest are often chosen for accurate tails, but their guarantee and merge behavior differ from deterministic uniform rank error. The phrase "relative error" is ambiguous: it may refer to the rank domain or to numerical value. Name the quantity before choosing a library. A design should name:

- the exact rank-error definition;
- pointwise versus simultaneous rank queries;
- deterministic versus probabilistic failure and the value of `delta`;
- support for weights and deletions;
- merge behavior and whether repeated merging degrades accuracy;
- tail accuracy requirements;
- serialization compatibility.

Ordinary KLL and GK summaries support insertion, not arbitrary deletion. Sliding-window quantiles need pane decomposition, an expiration-aware algorithm, or retained raw data; subtracting one sketch from another is invalid. Weighted updates also require a library whose compaction and error analysis cover weights.

Do not use a histogram with arbitrary fixed buckets as if it were a general quantile sketch. Its value resolution is fixed by bucket boundaries, and distribution drift can make the approximation useless. Conversely, do not claim KLL bounds numerical interpolation error: it bounds rank.

### Check your design: approximation

Before choosing a sketch, state whether a false positive, a missed candidate, or a rank error is acceptable. Give the guarantee's failure probability and update model. A correct answer rejects using a membership filter as a frequency estimator, subtracting an insertion-only quantile sketch, or advertising a point-query guarantee as an unlimited simultaneous guarantee.

## Building a Streaming Telemetry Service

LEAD: A complete streaming service combines algorithms with event identity, expiration, partitioning, recovery, and versioned output. The final result is only as strong as the weakest of these contracts.

### Choosing the correct summary

| Query | Representative state | Error contract | Merge operation |
| --- | --- | --- | --- |
| Immutable record top-k | Size-`k` min-heap | Exact with total order | Top-k of local candidates |
| Insertion-only heavy hitters | Misra-Gries | Deterministic additive count bound | Merge and reduce counters carefully |
| Point frequency | Count-Min Sketch | One-sided additive, probabilistic | Elementwise counter addition |
| Distinct count | HyperLogLog | Approximate relative error | Registerwise maximum |
| Membership | Bloom filter | False positives, no false negatives under valid use | Bitwise OR |
| Quantile | GK, KLL, or other quantile summary | Rank error | Algorithm-specific merge |
| Mean and variance | `(n, mean, M2)` | Floating-point numerical error | Parallel variance formula |
| Uniform sample | Random priorities or reservoir | Exact inclusion probability under model | Global priority top-k |

:::callout insight|State the approximation contract
For every summary, say what is approximated, the unit of error, whether the error is one-sided, the probability and scope of failure, the supported update model, and the exact compatibility conditions for merging. A data-structure name is not a correctness contract.
:::

### A complete event-time heavy-hitter design

:::diagram telemetry_pipeline|A streaming service carries event identity, key ownership, window state, and output versions across distinct boundaries. Split-key counts must be merged before ranking. Recovery binds the summary state to consumed offsets; late data follows a declared correction policy.

Consider a service that reports the top `k` error signatures per tenant for each five-minute event-time window. Traffic peaks at millions of events per second, events can arrive 20 minutes late, and results should update quickly.

This is a hypothetical workload for design, not a measured deployment. The cumulative exercises at the end of this chapter review all four streaming chapters.

#### 1. Identity and ingestion

Each event carries `event_id`, tenant, event timestamp, signature, source partition, and source offset. Canonicalize the signature before partitioning. Deduplicate by stable `event_id` within the required replay horizon, or make the source and sink exactly-once enough that duplicate effects are bounded and documented.

Partition first by tenant and then by a stable subshard to distribute hot tenants. A signature must map deterministically so its partial count can be merged. If salting splits one signature across workers, the second stage must sum its partials before final ranking.

Partitioning by signature avoids split counts but sends an extremely hot signature to one worker. Salting by `(signature, event_id)` distributes its updates, at the cost of a second keyed aggregation that recombines all salts for that signature. Use measured skew to enable salting only for hot keys; salting every key multiplies shuffle and intermediate state.

#### 2. Window state

Assign each event to a half-open interval `[start, start + 5 minutes)`. On every worker and window, maintain:

- a Count-Min Sketch for approximate frequency queries;
- a Misra-Gries table with capacity `r` for deterministic candidate recall;
- total accepted mass `F_1`;
- estimator parameters and hash seed;
- revision and source progress metadata.

If the distinct-key count is affordable, exact keyed counts are simpler and may be the right answer. Approximation is justified only by a quantified state or throughput constraint.

The candidate contract is explicit: every signature with frequency greater than `F_1/(r+1)` is present. To guarantee recall of all true top-k keys, the design must establish that the kth frequency exceeds this threshold, increase `r`, or use an exact fallback. Count-Min accuracy cannot repair a key that the candidate stage omitted.

#### 3. Watermark and triggers

Derive per-source watermarks from source progress and observed delay, then take the minimum across non-idle inputs. Emit:

- early speculative panes every 30 seconds of processing time;
- an on-time pane when the watermark passes the window end;
- late correction panes after accepted late arrivals;
- a final pane when the watermark passes `window_end + 20 minutes`.

Do not describe the on-time pane as complete. It is complete only relative to the watermark model. Records after allowed lateness go to a durable late-data stream with counts and alerts, not silent deletion.

For example, suppose the on-time result is `A:100, B:90`. A late batch contributes 20 occurrences of `B`, so revision two must atomically replace the order with `B:110, A:100`. Appending a second unversioned list would leave consumers unable to determine which result is current.

#### 4. Merge and ranking

Workers emit compatible sketch state plus candidates. The reducer adds Count-Min counters, merges Misra-Gries maps with the reduce-and-prune rule, queries merged frequency estimates, and emits the best `k` with:

- estimated count;
- interval `[max(0, f_hat - epsilon F_1), f_hat]` and failure parameter `delta`;
- window and revision;
- pane timing class: early, on-time, late, or final;
- completeness and dropped-event metrics.

Allocate the failure budget over the complete candidate set before sorting by Count-Min estimates. Misra-Gries candidates are selected independently of Count-Min's hashes, so an upper bound of `Q` queried candidates permits `delta = eta/Q`. Allocating only across the displayed winners would ignore their selection by noisy estimates. If ranking intervals overlap, emit "order uncertain" or verify exact counts from retained/durable data. A numerical sort of estimates is not a proof of exact order.

#### 5. Sink and correction semantics

Upsert by `(tenant, window_start)` and require monotonically increasing revisions. The sink stores the entire ranked list for each revision or applies an atomic replacement. Downstream consumers should not independently merge top-k lists from multiple revisions. Include an estimator schema version so a rolling deployment cannot merge incompatible hashes or silently compare different error contracts.

Checkpoint operator state together with source offsets or use a framework snapshot that provides the equivalent barrier. At a checkpoint barrier, each operator snapshot must represent exactly the input prefix named by the saved offsets; in-flight records belong either to the snapshot/channel state or to the replay suffix, not neither and not both. Commit sink writes transactionally with the checkpoint, or make `(tenant, window, revision)` upserts idempotent. On restore, load the snapshot, seek every source partition to the recorded offset, and reject a stale revision at the sink. Event-ID deduplication is still needed if upstream redelivery can cross the framework's consistency boundary.

#### 6. Capacity and failure policy

Estimate active windows as approximately:

`active_windows ~= ceil((window_width + allowed_lateness + watermark_lag) / window_slide)`,

allowing another boundary window depending on trigger timing. Here watermark lag is the nonnegative gap between the newest accepted event time and the watermark; allowed lateness is the separate retention period after the watermark passes a window's end. Multiply by tenants, subshards, sketch bytes, candidate bytes, deduplication state, and checkpoint copies. A stalled watermark can increase retained windows beyond the nominal estimate, so alert on oldest open window and state bytes.

The arithmetic can reject a design before implementation. Five-minute tumbling windows with 20 minutes of allowed lateness retain about five windows in steady state. If there are 50,000 active tenant-subshard pairs and each window uses the earlier `297.4 KiB` Count-Min configuration, dense sketch tables alone require:

`50,000 * 5 * 304,528 bytes ~= 70.9 GiB`.

Candidates, hash metadata, deduplication, object overhead, snapshots, and watermark stalls add more. That budget suggests tiering: use exact sparse maps for low-volume tenants, allocate dense sketches only after a threshold, reduce unnecessary subshards, choose `epsilon` from a business-relevant count gap, and evict finalized state. Parameters belong in the capacity model, not only in a correctness proof.

Under backpressure, silently dropping events biases frequency and can change ranking. Prefer source throttling and durable buffering. If controlled sampling is allowed, attach inclusion probabilities and use an estimator designed for weighted observations; do not feed sampled counts into an ordinary sketch and report them as exact.

Define a hard failure policy for a watermark that never advances. Options include marking a demonstrably idle source, quarantining a damaged partition, or capping retention and marking affected windows incomplete. Any cap changes the completeness contract and must surface in output metadata and alerts.

### Trace contracts for a generative service

Instrument the request, retrieval, generation, tool attempts, and outcome verification as related operations. Preserve a logical task ID across retries while giving each attempt its own identity. A retry should increase attempt cost without inventing another user request. For asynchronous work, retain causal links when a single parent-child span tree no longer describes execution.

This schema needs versioning. [OpenTelemetry semantic conventions v1.42.0](https://github.com/open-telemetry/semantic-conventions/releases/tag/v1.42.0) moved GenAI conventions to a [dedicated repository](https://github.com/open-telemetry/semantic-conventions-genai); the old repository's GenAI definitions were deprecated. As checked September 24, 2026, integrations should pin the applicable conventions and schema URL, not assume that an old attribute spelling remains the current contract.

Record model and application versions, operation type, outcome, token-accounting semantics, cache use, and phase durations. Distinguish time to first response chunk from time to first useful text token: a protocol event or reasoning-only chunk may arrive first. Likewise, do not add reasoning tokens to output totals when the provider already includes them. Missing usage is unknown, not zero. Keep task IDs and document IDs in traces with suitable access controls, not unbounded metric labels. Raw prompts and tool outputs require a separate retention and disclosure policy.

The [Kafka Streams processing contract](https://docs.confluent.io/platform/current/streams/concepts.html) can couple consumed offsets, state changes, and output records within its transaction boundary. An HTTP notification or remote model call is outside that boundary. For such effects, write an outbox intent with the committed state, dispatch using a stable operation key, and reconcile the receiver's receipt. Delivery may repeat; the externally visible effect must tolerate that repetition.

**Worked check:** a tool times out, then returns its existing receipt on retry. Count one logical effect and two attempts, and include both attempts in latency and cost. A dashboard that counts only successful spans hides the recovery overhead.

### Production readiness checklist

Before shipping a streaming summary, verify:

- **semantics:** query, window, update model, ordering, ties, and finalization;
- **accuracy:** bound, probability, target population, and adversarial assumptions;
- **numerics:** counter overflow, accumulator dtype, `NaN`, infinity, and hash width;
- **distribution:** partition rule, merge compatibility, duplicate handling, and skew;
- **state:** bytes per key/window, expiration, compaction, and schema version;
- **recovery:** snapshot-to-offset consistency, replay determinism, and idempotent sink;
- **operations:** watermark lag, state growth, false-positive or error calibration, and hot keys;
- **degradation:** backpressure behavior, sampling policy, and visible completeness metadata.

### Design Exercises

1. Implement exact top-k for immutable records, prove the heap invariant, and prove the exact distributed merge. Give a counterexample for merging local frequency top-k.
2. Derive both exact sliding maximum with a monotonic deque and approximate sliding one-counts with an exponential histogram. State their ordering assumptions and complexity.
3. Derive Welford's online variance update, weighted update, removal formula, and parallel merge. State the variance denominator for each interpretation.
4. Prove Algorithm R reservoir sampling is uniform. Then derive exact distributed uniform and weighted-without-replacement samples.
5. Derive Misra-Gries candidate and error guarantees, show how two states merge, and contrast its bounds with Space-Saving.
6. Derive the Count-Min Sketch point-frequency bound, size it numerically, convert to a many-query failure budget, and build a valid heavy-hitter pipeline around it.
7. Derive the HyperLogLog update, estimator intuition, error-versus-memory tradeoff, union merge, and limitations for deletion and intersection.
8. Derive Bloom-filter false-positive probability, optimal hash count, and memory formula. State the assumptions behind "no false negatives."
9. Explain KLL compaction from first principles, including its rank-error mechanism, query, merge, and the difference between rank and value error.
10. Keep the service's five-minute windows, 20-minute allowed lateness, 50,000 tenant-subshard pairs, and 304,528-byte sketches. Watermark lag grows to eight minutes, but retention is capped at six windows. Using the chapter's nominal window-count model, calculate the state requirement and explain the capacity, completeness, and recovery decision.

### Worked Solutions

#### 1. Exact record top-k

Maintain a min-heap ordered by `(score, stable_id)` with at most `k` records. The invariant after a prefix of length `i` is that the heap contains the best `min(i, k)` prefix records and its root is the worst retained record. If the heap is not full, insert. If it is full, a new record no better than the root has at least `k` records ahead of it and cannot qualify; a better record replaces exactly the old boundary item. That proves the update by induction.

The implementation is:

Example status: Illustrative Python excerpt; not standalone.

```python
from heapq import heappush, heapreplace

def top_k(records, k):
    if k < 0:
        raise ValueError("k must be nonnegative")
    if k == 0:
        return []
    heap = []
    for record in records:
        entry = (record.score, record.stable_id, record)
        if len(heap) < k:
            heappush(heap, entry)
        elif entry > heap[0]:
            heapreplace(heap, entry)
    return [x[2] for x in sorted(heap, reverse=True)]
```

This excerpt assumes finite scores and unique, comparable stable IDs. Reject or canonicalize non-finite scores before calling it. The worst-case update cost is `O(log k)` for `k >= 2`, scan time is `O(n log k)`, state is `O(k)`, and sorted materialization is `O(k log k)`; `k = 1` is a linear scan. Also state whether the API returns exactly `k` records or all score ties at the boundary.

For disjoint immutable shards `S_j`, each shard emits `T_k(S_j)`. Any record omitted locally has at least `k` same-shard records above it and therefore cannot rank globally. Thus the coordinator or reduction tree computes `T_k` over at most `p k` candidates exactly. Communication is `O(p k)` records for a flat gather.

The proof depends on each immutable record having one owner and one final score. It is false for keys whose frequency is split. With `k = 1`, shard one has `A:6, X:5`, and shard two has `B:6, X:5`. Local winners contain only `A` and `B`, but global `X:10` is the true winner. Partial counts must first be aggregated by key or represented by a mergeable frequency summary. Replication also requires deduplication, and mutable scores or deletion require retained full state because a size-k heap has forgotten the next replacement.

#### 2. Exact and approximate sliding windows

For a count window of width `w`, store candidate indices in increasing index order and decreasing value order. First remove front indices `<= i - w` because they expired. Then remove back indices whose values are `<= x_i`. Each removed value is dominated by the new value: the new one is at least as large and expires later, so the old one can never again be a maximum. Append `i`; the front is the maximum live candidate.

An individual update can remove `O(w)` entries, but each index is appended once and removed at most once from either end. Across `n` inputs there are `O(n)` deque operations, so amortized update time is `O(1)` and state is `O(w)`.

For counts of one-bits over the last `W` positions, an exponential histogram stores power-of-two buckets and the timestamp of each bucket's newest one. In the conservative variant above, choose `b = ceil(1/epsilon) + 1`, allow at most `b` buckets per size, and merge the two oldest when a size overflows. Return zero for no buckets and the exact sum when the oldest has size one. Otherwise sum all bucket sizes but only half of the oldest:

`count_hat = sum(bucket sizes) - oldest_size/2`.

Only the oldest bucket can straddle the boundary, so absolute error is at most half its size `C/2`. When this bucket was formed, at least `b-1` newer buckets of size `C/2` remained; their mass cannot expire first. Consequently:

`(C/2)/(1 + newer represented ones) <= 1/(b-1) <= epsilon`,

which converts this to relative error against the true live count. It stores `O((1/epsilon) log W)` buckets, uses `O((1/epsilon) log^2 W)` bits with timestamps modulo a suitable multiple of `W`, and has amortized `O(1)` update work despite occasional merge cascades when the oldest buckets of each size are directly accessible.

Both algorithms assume an arrival-order count window. A late event belongs in the middle of event-time order and may revise an emitted answer. A reorder buffer, watermark, allowed-lateness policy, and revision protocol are separate system responsibilities.

#### 3. Online, weighted, removable, and parallel variance

Store `(n, mu, M2)` where `M2 = sum_i (x_i - mu)^2`. For a new `x`, set `n' = n + 1` and `delta = x - mu`. Expanding the mean gives `mu' = mu + delta/n'`. Let `delta2 = x - mu'`. Re-centering the old observations makes cross terms vanish because their deviations around `mu` sum to zero. Adding the new deviation yields `M2' = M2 + delta * delta2`.

Population variance is `M2/n`; unbiased sample variance is `M2/(n-1)` for `n >= 2`. This avoids the dangerous subtraction in `mean(x^2) - mean(x)^2`, though adequate accumulator precision and non-finite input policy are still required.

For a positive frequency weight `w`, replace count by total weight `W`. With `delta = x - mu`:

`W' = W + w`,

`mu' = mu + (w/W') delta`,

`M2' = M2 + w delta (x - mu') = M2 + (Ww/W') delta^2`.

If removing a previously included `(x,w)` and `W > w`:

`mu' = (W mu - w x)/(W-w)`,

`M2' = M2 - w(x-mu)(x-mu')`.

Removal is numerically more fragile than addition and requires exact knowledge that the observation was present. Frequency weights, reliability weights, and probability weights have different unbiased-variance denominators; do not automatically use `W-1` outside replicated-observation frequency weights.

For disjoint states `A` and `B`, let `delta = mu_B - mu_A` and `W = W_A + W_B`. Re-centering both groups around the combined mean gives:

`mu = mu_A + delta W_B/W`,

`M2 = M2_A + M2_B + delta^2 W_A W_B/W`.

For unweighted observations, set `W_A=n_A` and `W_B=n_B`. The extra term is the between-partition sum of squares. Handle empty partitions explicitly, use a balanced merge tree for numerical quality, and ensure each record belongs to exactly one state.

#### 4. Uniform, distributed, and weighted sampling

Algorithm R stores the first `k` items. At one-based position `i > k`, it selects a uniform integer in `[1, i]` and replaces that reservoir slot if the integer is at most `k`. The new item enters with probability `k/i`.

Assume every earlier item is present after `i-1` positions with probability `k/(i-1)`. Conditional on being present, its particular slot is replaced with probability `1/i`, so it survives with probability `(i-1)/i`. Its new inclusion probability is `(k/(i-1))((i-1)/i) = k/i`. The induction proves that every item has inclusion probability `k/N` at the end.

Naively combining equal-size shard reservoirs is biased if shard sizes differ. A robust distributed construction assigns each record an independent continuous priority, keeps the `k` smallest priorities locally, then keeps the `k` smallest globally. An item omitted locally already has `k` lower-priority same-shard items, so it cannot qualify globally. Independent continuous priorities produce the same exact uniform size-`k` subset as a global priority sample because ties occur with probability zero. A stable finite hash and deterministic tie-breaker make the result reproducible, but the construction is only approximately uniform when collisions are possible; use enough hash bits to make that bias negligible for the population and risk tolerance.

An alternative first draws shard sample counts from:

`P(s_1,...,s_p) = product_j choose(n_j,s_j)/choose(N,k)`,

then takes a uniform size-`s_j` subset from each shard's uniform reservoir. This is exact but requires trustworthy shard sizes and coordinated hypergeometric allocation.

For positive weights and sequential probability proportional to remaining weight, draw independent `U_i` in `(0,1]`, set:

`priority_i = -ln(U_i)/w_i`,

and keep the `k` smallest. These are independent exponential clocks of rates `w_i`; the chance clock `i` rings first is `w_i/sum_j w_j`, and memorylessness repeats the rule among remaining items. Local priority top-k then global priority top-k is exactly composable. Record the seed, priority/hash algorithm, canonical ID encoding, target population, and whether weights are supported; time decay and sampling with replacement require different laws.

#### 5. Misra-Gries and Space-Saving

With capacity `r`, Misra-Gries increments a tracked key, inserts into a free slot, or otherwise decrements all `r` counters and discards zeros. Each decrement round cancels `r+1` distinct occurrences: the arrival plus one from every tracked key. If there are `g` rounds, then `g <= N/(r+1)`. For any tracked key:

`c_hat(x) <= f(x) <= c_hat(x) + g`.

An untracked key has `f(x) <= g`, so every key above `N/(r+1)` must be tracked. Moreover:

`g = (N - sum_x c_hat(x))/(r+1)`.

To merge states `A` and `B`, add their maps keywise. If more than `r` keys remain, let `C` be the `(r+1)`th-largest combined counter, subtract `C` from every counter, and discard nonpositive values. Pruning lowers any estimate by at most `C` and removes at least `(r+1)C` total counter mass. Therefore the new budget:

`Delta' = (N_A+N_B-sum c')/(r+1)`

increases by at least `C` beyond the two input budgets and still covers every key's underestimate. This proves deterministic merge validity, though the merged table may differ from serial processing order.

Space-Saving replaces the minimum estimate `c_min` with a new key at estimate `c_min+1` and stored error `c_min`. A tracked key then satisfies:

`estimate-error <= f <= estimate`,

and an untracked key is bounded by the current minimum. It often gives tighter practical candidates on skewed data, while Misra-Gries has the simpler cancellation merge. A Space-Saving table is not merged by adding local winners; use a proved error-aware variant. Either summary needs a second pass or separated intervals to certify exact order.

#### 6. Count-Min Sketch and heavy hitters

In one row, the counter for `x` is `f(x)` plus nonnegative collision mass. Therefore every row and their minimum are at least `f(x)`. With width `w`, pairwise-independent hashing makes expected collision noise at most `F_1/w`. Set `w = ceil(e/epsilon)`; then expected noise is at most `epsilon F_1/e`. Markov's inequality bounds the chance that one row's noise exceeds `epsilon F_1` by `1/e`.

With `d` independent rows, the minimum is too large only if every row has excessive noise, with probability at most `e^(-d)`. Choosing `d = ceil(ln(1/delta))` gives `f(x) <= f_hat(x) <= f(x) + epsilon F_1` with probability at least `1-delta` for a fixed query.

For `epsilon=0.001` and `delta=10^(-6)`, choose `w=2,719` and `d=14`: `38,066` counters or about `297.4 KiB` at eight bytes each. At `F_1=100,000,000`, the allowance is 100,000 counts. This is additive, so it is meaningful only relative to the application's frequency scale.

For `Q` fixed queries with overall failure budget `eta`, allocate `delta=eta/Q`, so `d=ceil(ln(Q/eta))`; this follows from the union bound. Adaptive adversarial queries need separate analysis.

The sketch does not store keys, so add Misra-Gries, Space-Saving, an external dictionary, or a replay pass for candidate discovery. Merge compatible standard sketches by elementwise addition, merge candidates with their proved rule, and query the merged sketch. Report `f(x)` in `[max(0,f_hat-epsilon F_1), f_hat]`. If intervals around the kth boundary overlap, the exact order is unresolved.

The guarantee assumes nonnegative current frequencies, no counter overflow, independent hash rows, secret or nonadversarial hashing as appropriate, and identical width, depth, seeds, encoding, counter type, and update rule across merged states. Standard linear updates also support valid strict-turnstile deletions. Arbitrary signed frequencies, duplicates relative to the intended population, and conservative-update variants require separate treatment.

#### 7. HyperLogLog

Hash each canonical key uniformly. Use `p` prefix bits to choose one of `m=2^p` registers and let `rho` be one plus the leading-zero count in the remaining bits. Update `M[j]=max(M[j],rho)`. Since `P(rho>r)=2^(-r)`, a large maximum is evidence of many distinct hashes. The harmonic-mean estimator:

`E = alpha_m m^2 / sum_j 2^(-M[j])`

has normal-range relative standard error about `1.04/sqrt(m)`. With `p=14`, `m=16,384`, the RSE is about `0.8125%` and byte registers occupy `16 KiB`. If `V` registers are zero in the small range, linear counting uses `m ln(m/V)`; production libraries also apply calibrated bias and large-range corrections.

Duplicate keys are idempotent because the same register maximum is repeated. Compatible states merge by registerwise maximum, exactly representing the register state of set union. Precision, hash seed/width, canonical encoding, and estimator version must match.

HLL does not estimate per-key frequency or membership. It cannot delete because a register does not retain the second-largest `rho`. Inclusion-exclusion for intersection subtracts three noisy estimates and is unstable for small intersections. Use per-window HLLs or a purpose-built set-expression sketch when those are core queries.

#### 8. Bloom-filter sizing and validity

After `h n` uniform bit placements in `m` bits, a bit remains zero with probability `(1 - 1/m)^(h n)`, approximately `exp(-h n/m)`. A key not in the set becomes a false positive when all `h` queried positions are one, so:

`p_fp ~= (1 - exp(-h n/m))^h`.

Let `a=m/n` and `y=exp(-h/a)`. Then:

`d[ln p_fp]/dh = ln(1-y) + h y/(a(1-y))`.

Setting `y=1/2` and `h=a ln 2` makes the derivative zero. Therefore:

`h_opt = (m/n) ln 2`,

`p_min ~= exp(-(m/n)(ln 2)^2)`,

`m ~= -n ln p/(ln 2)^2`,

`h ~= -ln p/ln 2`.

For ten million distinct keys and `p=0.001`, this is about `143,775,876` bits, `17.1 MiB`, and ten hash probes. Round `h`, then recompute `p_fp` and consider CPU cost. Double hashing can derive `g_i=h_1+i h_2 mod m`, provided the construction and modulus avoid short cycles.

"No false negatives" assumes append-only valid use, no lost or corrupted bits, identical encoding/seeds at insert and query, and correct concurrent writes. An ordinary filter cannot delete. A counting filter can decrement only if every insertion and deletion is balanced; deleting an absent key can clear shared counters and create false negatives. Overfilling raises false positives, so monitor bit occupancy and rotate or rebuild.

#### 9. KLL quantiles

KLL stores a hierarchy. A level-`ell` item has weight `2^ell`. Items enter level zero. When a level exceeds capacity, sort it, flip a fair coin to retain odd or even positions, discard the rest, and promote survivors one level with doubled weight. For `2s` compacted items, represented mass is preserved.

For any fixed threshold `v`, the compacted sorted prefix differs from its original represented rank by at most one level item, or `2^ell`. The random parity makes this a conditionally zero-mean signed error. The level capacities limit high-weight compactions; concentration over their errors gives the probabilistic rank bound. Preserve an unpaired item at its current level when a buffer has odd length.

To query rank `r`, attach each retained item its level weight, sort all retained items by value, and return the value where cumulative weight reaches `r`. To merge compatible states, concatenate each level and compact overflows. Merge order can change retained samples but preserves the library's contract.

Rank error is not value error. With 500 zeros and 500 values equal to one billion, a small median rank interval can allow either value. State whether the guarantee is pointwise or simultaneous, the failure probability, handling of duplicates and weights, and the exact quantile convention. KLL does not support arbitrary subtraction; sliding-window deletion needs pane decomposition, an expiration-aware sketch, or retained data.

#### 10. A retention cap cannot advance a watermark

The nominal requirement becomes `ceil((5 + 20 + 8) / 5) = 7` windows. Dense sketch tables alone need `50,000 * 7 * 304,528 / 2^30`, about 99.3 GiB. Six windows consume about 85.1 GiB, leaving a 14.2 GiB shortfall before candidates, deduplication, snapshots, and object overhead. Trigger alignment can require another boundary window; this estimate is a planning floor under the stated model, not an exact peak allocation.

The seventh window still accepts legitimate corrections. Increase or tier capacity, or use declared backpressure and durable buffering while recovering source progress; buffering alone cannot free already retained state. If the hard cap forces eviction, mark the affected result incomplete and preserve the late-data/reconstruction policy. Memory pressure cannot justify labeling an unfinished window final.

Checkpoint completeness flags and output revisions with their source-offset boundary. Otherwise a restore can resurrect an evicted window as apparently complete, lose an accepted correction, or publish a stale revision. The changed workload therefore needs both a larger state budget and a tested degraded-result contract.

The streaming arc is complete; the next chapter is an optional implementation refresher, while application-design readers can continue at **An Engineering System Design and Review Method** or **RAG, Vector Search, and Evaluation Pipelines**.

### Further Study and Primary References

- [Vitter, Random Sampling with a Reservoir](https://www.cs.umd.edu/~samir/498/vitter.pdf) - uniform one-pass sampling and skip-based improvements.
- [Efraimidis and Spirakis, Weighted Random Sampling with a Reservoir](https://www.sciencedirect.com/science/article/pii/S002001900500298X/pdf) - exponential-key weighted sampling without replacement.
- [Datar et al., Maintaining Stream Statistics over Sliding Windows](https://perso.ens-lyon.fr/bruno.salvy/INF431/Projet-hyperloglog/INF431_-_Projet_Informatique_files/DatarGionisIndykMotwani2002.pdf) - exponential histograms and sliding-window approximation.
- [Cormode and Muthukrishnan, Count-Min Sketch](https://www.cs.helsinki.fi/u/jilu/paper/countMin.pdf) - the sketch construction and additive error analysis.
- [Misra and Gries, Finding Repeated Elements](https://khoury.northeastern.edu/home/pandey/courses/cs7800/spring26/papers/mg.pdf) - deterministic heavy-hitter candidates.
- [Metwally, Agrawal, and El Abbadi, Efficient Computation of Frequent and Top-k Elements](https://www.cs.ucsb.edu/sites/default/files/documents/2005-23.pdf) - the Space-Saving algorithm and its frequency bounds.
- [Agarwal et al., Mergeable Summaries](https://doi.org/10.1145/2500128) - merge models and deterministic summary reductions.
- [Flajolet et al., HyperLogLog](https://dmtcs.episciences.org/3545) - cardinality estimator analysis.
- [Bloom, Space/Time Trade-offs in Hash Coding with Allowable Errors](https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p422-bloom.pdf) - approximate membership origins.
- [Chan, Golub, and LeVeque, Algorithms for Computing the Sample Variance](https://doi.org/10.1080/00031305.1983.10483115) - numerical and pairwise variance algorithms.
- [Greenwald and Khanna, Space-Efficient Online Computation of Quantile Summaries](https://www.cs.dartmouth.edu/~ac/Teach/CS49-Fall11/Papers/greenwald-quantiles.pdf) - deterministic rank-error summaries.
- KLL: [Karnin, Lang, and Liberty](https://arxiv.org/abs/1603.05346) for compaction analysis; [DataSketches](https://datasketches.apache.org/docs/KLL/KLLSketch.html) for parameters and merge behavior.
- [Apache Beam Programming Guide](https://beam.apache.org/documentation/programming-guide/) - event time, watermarks, triggers, and allowed lateness.

## Compact Review of ML Algorithms in Production

LEAD: This compact reference shows how production ML code turns mathematical assumptions into shapes, invariants, numerical policies, and measurable failure behavior. It emphasizes implementation contracts rather than surveying each algorithm exhaustively.

### Vectorized logistic regression

For examples `X`, labels `y`, and weights `w`, logits are `z = Xw`. Binary cross entropy should use a stable logits form rather than computing `log(sigmoid(z))` directly.

Example status: Illustrative Python excerpt; not standalone.

```python
def logistic_loss_and_grad(X, y, w, l2=0.0):
    z = X @ w
    # softplus(z) - y*z is stable binary cross entropy from logits.
    loss = mean(softplus(z) - y * z) + 0.5 * l2 * dot(w, w)
    p = sigmoid(z)
    grad = (X.T @ (p - y)) / X.shape[0] + l2 * w
    return loss, grad
```

Production implementations must specify sparse-feature handling, class weighting, calibration, distributed reduction, feature normalization, leakage controls, and stopping criteria. The sparse matrix-vector product and update cost `O(nnz(X))` per pass.

### K-means

Lloyd's algorithm alternates nearest-centroid assignment and centroid recomputation. Complexity is `O(n k d)` per iteration. Initialization strongly affects the local optimum; k-means++ spreads initial centers.

For large data, mini-batch k-means updates centroids from samples. Distributed implementations compute local counts and sums, then reduce. Empty clusters require a policy: reinitialize to a far point, split a large cluster, or retain the old centroid.

The Euclidean objective assumes roughly spherical clusters and comparable feature scales. If those assumptions fail, changing the optimizer will not fix the model.

### Decision trees

A tree searches splits that reduce impurity. For classification, Gini or entropy is common; for regression, squared error. Efficient continuous-feature search sorts values or uses histograms. Missing values and categorical features require explicit routing.

Depth, minimum leaf size, feature subsampling, and pruning trade fit against variance. Distributed histogram methods reduce candidate statistics instead of shipping examples. Production code also handles monotonic constraints, stable serialization, and feature schema evolution.

### Attention from primitives

Example status: Illustrative Python excerpt; not standalone.

```python
def attention(q, k, v, mask=None):
    scale = 1.0 / sqrt(q.shape[-1])
    scores = q @ swapaxes(k, -1, -2) * scale
    if mask is not None:
        scores = where(mask, scores, -inf)
    probs = softmax(scores, axis=-1)
    return probs @ v
```

Follow-ups include causal masks, padding, batched heads, mixed precision, stable softmax, dropout, KV caching, GQA, and memory-efficient attention. The concise code is a semantic reference, not a performant long-context implementation. It assumes at least one permitted key per query: a fully masked row otherwise feeds all negative infinities to softmax. The runnable single-head reference in `examples/attention.py` defines that case as a zero vector and tests dense versus chunked reduction. A production API may reject that input instead; the caller and kernel must agree.

### Beam search

Beam search keeps the best partial sequences by cumulative log probability. Length normalization can reduce the preference for short outputs, but changes the ranking objective. Finished beams must remain candidates without being expanded. Per-step top-k can be taken over `beam * vocabulary` scores.

Beam search is not sampling. It searches high-probability sequences under the model and may reduce diversity. Diverse beam variants, constraints, or stochastic beams modify the objective.

Example status: Illustrative Python excerpt; not standalone.

```python
for step in range(max_steps):
    active, finished = partition_by_finished(beams)
    if not active:
        break
    logits, next_state = model.step(active.tokens, active.state)
    scores = active.scores[:, None] + log_softmax(logits, axis=-1)
    # Each candidate retains its parent ID and proposed token.
    expanded = candidates_with_parent_ids(active, scores, next_state)
    # Finished sequences retain score, length, and state unchanged.
    selected = select_best(finished + expanded, beam_width)
    beams = materialize_selected_sequences_and_parent_states(selected)
    if beams.all_finished():
        break
return rank_with_length_penalty(beams)
```

The helpers above are schematic, not library APIs. Candidate materialization appends a token only for an expanded candidate, marks EOS as finished, and gathers every layer's cache from the same parent. Final length-penalty reranking does not make the intermediate raw-score pruning exact for the normalized objective.

### Nearest-neighbor search

Exact search computes all distances. Approximate methods reduce query work through graphs, inverted files, product quantization, or trees. The design must specify recall target, latency, update rate, filtering, memory, and metric.

Embedding normalization makes cosine similarity equivalent to inner product ranking. Product quantization compresses vectors and uses lookup tables for approximate distance, trading recall for capacity and bandwidth. Metadata filters can destroy index efficiency if applied after retrieval; integrate filters or route to partitions.

:::callout pitfall|Complexity without shapes is incomplete
`O(nkd)` does not reveal whether the implementation is a dense GEMM, sparse reduction, or cache-unfriendly loop. State shapes, memory layout, and the operation the hardware will actually execute.
:::

### From classifier scores to routing decisions

The logistic and tree examples also apply to model routing: predict whether a cheaper path will satisfy the task contract, then escalate difficult cases. A similarity score, softmax maximum, or fluent self-assessment is not automatically a calibrated probability of success. Train and evaluate against the actual outcome definition, including unsupported answers and unavailable evidence.

Use a held-out calibration split to choose a threshold, then freeze it before acceptance testing. Report coverage (the fraction sent to the cheap path), conditional failure among accepted requests, total cost, and latency by task slice. A router can look accurate by escalating nearly everything. A changed generator, retriever, prompt, or user population can invalidate calibration even when the router weights stay fixed.

In an original cost example, the cheap path costs one unit and escalation costs ten additional units. If 60% of requests stop after the cheap path, mean cost is `1 + 0.40 * 10 = 5` units. If accepted answers are correct 95% of the time and escalated answers 99%, overall correctness is `0.60 * 0.95 + 0.40 * 0.99 = 96.6%`. This misses a hypothetical 98% requirement despite the savings. Direct routing to the expensive path has a different cost formula because it need not pay for the initial cheap attempt.

Small tests also leave uncertainty. Zero failures among 100 independently sampled accepted cases gives a one-sided 95% binomial upper failure bound of `1 - 0.05^(1/100) ~= 2.95%`. It does not establish a failure rate below 1%. Correlated cases, threshold tuning on the same set, or distribution shift invalidate that simple interpretation. If evidence is insufficient, gather more representative cases or reduce coverage; changing a confidence label does not improve reliability.

**Worked check:** after replacing the retriever, can the old threshold ship unchanged because the classifier code is identical? Only after revalidation. Its target is the complete pipeline's outcome, which has changed.

### Design Exercises

1. Derive stable logistic loss from logits.
2. How do you distribute k-means updates?
3. Implement beam search state gathering correctly.
4. Explain why the reference attention code is memory-heavy.
5. Choose an ANN index for high update rate and metadata filtering.

### Worked answer criteria

1. Use `max(z, 0) - y*z + log1p(exp(-abs(z)))` for binary cross entropy from logits, then average consistently with the gradient. Test large positive and negative logits and reject incompatible shapes.
2. Reduce each cluster's count and vector sum across workers; divide only after the global reduction. Workers must share a centroid revision and an empty-cluster policy. Averaging local centroids without weighting their counts is wrong.
3. Test two winning candidates from one parent, an EOS candidate, a retained finished beam, and reordered parents. Tokens, scores, lengths, and every layer's KV cache must follow the same ancestry.
4. The dense reference materializes a query-by-key score matrix and usually probabilities of the same size. Chunked attention retains online normalization state instead; compare against the dense result within the declared dtype tolerance.
5. Benchmark the actual update/delete workload and filtered recall, not only unfiltered search throughput. Keep an exact-search oracle for small partitions and test selective ACLs before committing to an ANN layout.


## An Engineering System Design and Review Method

LEAD: A strong system design or design review is a sequence of decisions tied to requirements. A diagram is evidence of that reasoning, not a substitute for it.

Use this chapter to prepare the documentation-assistant design that follows. Its assumed corpus contains 200,000 chunks, traffic peaks at 60 queries per second, and protected evidence must obey current permissions. The six steps below should produce a baseline and a short list of experiments, not six separate checklists to complete indefinitely.

### Step 1: define the contract

Clarify users, operations, scale, latency, consistency, availability, durability, privacy, compliance, cost, and evolution. For ML systems, add model quality, freshness, feedback, evaluation, and failure containment.

Convert vague statements into working numbers. When measurements are unavailable, declare reasonable assumptions, expose their sensitivity, and keep the design parameterized. Distinguish hard requirements from preferences.

### Step 2: estimate

Estimate storage, throughput, state, bandwidth, and hotspots. Use powers of ten. Identify peak-to-average and read/write ratio. For model systems, estimate training tokens, model bytes, KV state, feature volume, embedding count, or evaluation cost.

For the assistant, 200,000 vectors of 768 float32 components occupy about 614 MB before indexes and metadata. That does not establish search latency, but it removes raw vector capacity as an immediate reason to shard. Measure filtered search and reranking before adding a distributed index.

### Step 3: draw the baseline

Show clients, API, durable state, compute workers, queues or streams, indexes or caches, and observability. Draw trust and failure boundaries. Assign ownership of each state transition.

For the first assistant design, keep canonical documents and permissions separate from the search index. A retrieved chunk is a candidate, not permission to disclose its text. Draw the authority check before the reranker or generator receives protected content.

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

Define metrics, alerts, experiments, rollback, and capacity planning. State unresolved risks and the next experiment. Assign an owner to each release gate and specify what evidence would stop the rollout.

:::callout decision|Name what you would not build yet
Restraint is a design skill. Defer components whose complexity is not justified by the stated scale or risk, and identify the trigger that would cause you to add them.
:::

### Review migrations and overload, not only the steady state

An architecture review should specify the transition between versions. For the documentation assistant, define a routing manifest that binds the parser, chunker, query/document encoders, index generation, reranker, prompt, and verifier. Build a compatible candidate beside the current version, shadow a bounded sample, compare on the same corpus and authority snapshot, and switch the manifest atomically. Retain the previous compatible set for rollback. A rollback must still consult current permissions and deletion tombstones; it must not resurrect revoked documents.

Use Little's law on a stable operating interval as a first capacity check: average in-flight requests equal arrival rate times mean time in the system. At 60 requests per second and a two-second mean, expect about 120 requests in flight. If mean time rises to eight seconds while admissions stay at 60, the corresponding average is 480. This is not a p99 guarantee or a model for an indefinitely growing queue. Set bounded queues and admission limits before overload makes that growth self-reinforcing.

Give the request one end-to-end deadline. Each stage consumes the remaining budget; a retry does not reset the clock. Cancellation should stop avoidable downstream work, but cancellation of a waiting client cannot undo a remote write that already committed. Preserve that ambiguous outcome for reconciliation. Shadow experiments also consume capacity, so budget them separately and suppress external effects.

**Review exercise:** the new index improves recall but doubles reranker candidates. What must the review revisit? Candidate quality, reranker queueing, generation admission, end-to-end tails, and cost per verified answer. Approving the index from isolated recall alone misses the changed critical path.

### Tradeoff matrix

| Decision | Option A | Option B | Flip condition |
| --- | --- | --- | --- |
| Sync vs async | Caller waits for completion | Caller receives an acceptance handle | Choose from completion latency and retry semantics; neither alone guarantees consistency |
| Push vs pull | Low update latency | Consumer-controlled load | Fanout or offline consumers dominate |
| Replicate vs shard | Availability, simple reads | Capacity, write scaling | One node or replica no longer fits |
| Exact vs approximate | Strong semantics | Lower cost or latency | Bounded error is product-acceptable |
| Shared vs isolated | High utilization | Failure and security isolation | Noisy neighbor or compliance risk dominates |

### Communicating the design

In a live review, signpost transitions: "I will first establish the workload, then estimate the dominant state, then draw a baseline and stress it." Keep a visible list of requirements and risks. Answer questions directly, then reconnect the discussion to the decision currently being tested. In a written record, use the same structure as section headings and preserve unresolved objections.

If new information invalidates the design, revise it. Defending an obsolete choice signals rigidity, not leadership.

### Design Exercises

1. Run the method on a feature store or evaluation service.
2. Which estimates change a vector database architecture?
3. How do you represent failure domains clearly in an architecture diagram?
4. What makes an approximation acceptable?
5. Name a component you would defer in a new LLM platform and the trigger to add it.

### Worked answer criteria

A strong response states the workload and SLO, quantifies the dominant state or resource, presents one coherent baseline with ownership and failure domains, and compares at least one credible alternative. It also defines observable success and guardrail metrics, rollout and rollback, and the condition that would reverse the decision. A polished diagram cannot compensate for a missing consistency, security, or overload contract.


## RAG, Vector Search, and Evaluation Pipelines

LEAD: Retrieval-augmented generation is a data and evaluation system wrapped around a model. Retrieval quality, access control, freshness, and citation behavior matter as much as the generator.

:::diagram rag_pipeline|The offline path creates a versioned index with source identities. The online path retrieves, filters, reranks, packs evidence, and generates an answer with citations. Retrieval authorization and evidence quality must survive every stage.

### Ingestion

Connectors fetch documents under source-specific permissions. Normalize without destroying structure. Chunk using semantic and layout boundaries while retaining document, section, time, and ACL metadata. Generate embeddings under a versioned model and write both canonical content and index entries.

Updates need identity and deletion. A replaced document should tombstone or supersede old chunks. Permission changes must propagate to caches and indexes. Store content hashes to avoid re-embedding unchanged chunks.

### Retrieval

Hybrid retrieval combines lexical and dense signals. Lexical search handles rare names and exact identifiers; dense search handles semantic paraphrase. A reranker spends more compute on a small candidate set.

Filters should be enforced before content leaves the retrieval trust boundary. Post-filtering an approximate index can reduce recall; use filter-aware partitions, bitsets, or over-retrieval with a measured bound.

Query rewriting and decomposition can improve recall but also drift intent. Preserve original query, log transformations, and evaluate each stage.

### Calculate hybrid ranking and late interaction

Dense retrieval commonly maps a query and a passage independently into fixed-length vectors, then ranks a dot product or cosine similarity. Independence allows passage vectors to be precomputed; compressing a passage to one vector can lose rare token-level matches. A cross-encoder instead processes query and passage together and learns a relevance score. That richer interaction costs a forward pass per candidate, so it usually reranks a shortlist rather than every document.

Hybrid fusion need not compare incomparable raw scores. Reciprocal rank fusion assigns each document a contribution from its position in each ranking:

:::equation RRF(d) = Σ_{r} 1 / (k + rank_{r}(d))|Ranks are one-based; a document absent from a ranking contributes zero, and k is a chosen smoothing constant.

If lexical search returns `[A, B]` and dense search returns `[B, C]`, with `k=60`, B receives `1/62 + 1/61 ≈ 0.0325`, A about `0.0164`, and C about `0.0161`. B wins through agreement even though it is not first lexically. This does not establish relevance: two correlated bad rankings can agree. Choose candidate depths and the constant on development data, retaining an independent evaluation split. The original [RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) studies this rank-based combination.

[ColBERT](https://arxiv.org/abs/2004.12832) offers a different middle ground: encode passages into token vectors offline, then use query-token interactions at retrieval time. A simplified normalized MaxSim score is:

:::equation score(q,d) = Σ_{i} max_{j} q_{i}^{T} d_{j}|Each query token finds its best document-token match; the matches are summed.

For query vectors `[1,0]` and `[0,1]`, a passage containing both directions scores two; one containing only `[1,0]` scores one. This preserves two distinct matching needs that a single passage vector might blur. It uses more index state than one vector per passage and requires appropriate indexing/pruning for scale. The independent reference `examples/retrieval_methods.py` tests RRF and this simplified score; it does not implement trained encoders or a production index.

### Hierarchical retrieval, long context, or more search?

[RAPTOR](https://arxiv.org/abs/2401.18059) recursively clusters and summarizes text to create a retrieval hierarchy. [GraphRAG](https://arxiv.org/abs/2404.16130) builds graph-derived communities and summaries for corpus-level questions. Their shared motivation is that “What themes recur across this corpus?” may not be answered by retrieving five locally similar paragraphs. They differ in structure and retrieval strategy; a graph is not a required component of every RAG service.

| Need | Simple starting point | Added technique and its cost |
| --- | --- | --- |
| Exact product or error identifier | Lexical retrieval | Hybrid search for paraphrases; extra index and fusion |
| Several precise semantic matches | Dense candidates plus reranker | Late interaction; more vectors and scoring work |
| A global corpus summary | Authorized aggregate evidence | Hierarchical/graph summaries; build cost and derived-data freshness |
| A small, bounded document set | Put the eligible documents in context | Longer prefill and context; not guaranteed evidence use |
| Unknown subquestions | Bounded query decomposition | More model/search calls, drift, and stopping decisions |

Derived summaries inherit access and deletion obligations. A public summary cannot silently contain facts from a restricted child document. Rebuilding leaves without invalidating an ancestor summary preserves stale information. Keep source-to-summary lineage, permissions appropriate to the combined content, and evidence links. For a claim requiring an exact number or exception, expand the summary back to authorized primary passages before answering.

Long context and retrieval are complementary resource choices. Loading an entire 20-page manual may be simpler than maintaining an index for it. Loading every document in a changing enterprise corpus is different. Compare answer quality, first-token latency, cost, freshness, and disclosure scope using the same eligible evidence—not a context-heavy system with hidden extra documents against an artificially restricted retriever.

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

### Visual evidence and multimodal retrieval

Text extraction can lose the evidence in a chart, a table's column structure, or a screenshot. [ColPali, revised February 2025](https://arxiv.org/abs/2407.01449v6), provides a foundation for page-image embeddings with late interaction. A different current example, [Qwen3-VL-Embedding and Qwen3-VL-Reranker, January 2026](https://arxiv.org/abs/2601.04720v2), combines multimodal embeddings with a pairwise reranker across text, images, document images, and video. These are different retrieval architectures, not interchangeable implementations of MaxSim.

The [Qwen3-VL-Embedding-8B publisher card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B), inspected September 24, describes selectable embedding dimensions through Matryoshka training. Dimensional truncation is a model-supported contract, not a general property of arbitrary embeddings. Pin preprocessing, modality limits, pooling, normalization, query instructions, output dimensions, and checkpoint revision together. Test smaller dimensions or output quantization against full-precision retrieval before adopting their storage savings.

For the documentation assistant, keep page or media identity, document revision, access scope, and a region or time reference with each hit. A retrieved page image is useful only if the answering model can inspect the relevant evidence; converting it back to lossy OCR can discard the original advantage. Cite the page and region, and retain accessible text where possible. For video, record frame sampling and time intervals so the system does not claim evidence for an event between sampled frames.

Compare OCR-plus-text retrieval, multimodal dense retrieval, and page-level late interaction on the same authorized corpus. Include chart lookup, visually ambiguous tables, exact identifiers, multilingual pages, and ordinary prose. Count image preprocessing, vectors per page, reranking, and generator image tokens in the budget. Select by measured workload quality and cost, not a published benchmark's overall rank.

### Separate approximation loss from representation loss

An exact vector-search oracle only measures the nearest neighbors under one chosen representation and metric. It is not ground truth for semantic relevance. First compare ANN with exact search using the same vectors and eligible corpus; then compare those retrieved documents with labeled evidence. This separates index approximation errors from an embedding model's inability to represent the task.

Filters change the operating point. If a tenant can access 0.1% of one million vectors, only 1,000 are eligible. An exact scan of that subset may be competitive with traversing a globally shared graph. [Vespa's nearest-neighbor guide](https://docs.vespa.ai/en/querying/nearest-neighbor-search-guide.html) documents exact and approximate modes and their interaction with filtering. This is an implementation example; measure the crossover for your index, selectivity, concurrency, and memory layout. Overfetching globally and then filtering does not guarantee eligible top-k recall.

Use paired query-level comparisons for a migration. Freeze evidence labels, corpus revisions, roles, budgets, and the evaluation rubric. Report wins, losses, and uncertainty, with resampling at the user or document group when queries share evidence. Synthetic questions and model judges can expand diagnostics, but their shared generation biases do not create an independent acceptance set. Keep judge versions and rubric changes in the experiment manifest, and inspect disagreement cases separately from the average score.

**Worked check:** ANN matches 98% of the exact vector top ten, but neither retrieves the required chart. Increasing graph search effort addresses approximation loss; it cannot recover information omitted by the representation. Add the visual-evidence slice before selecting a remedy.

### Serving architecture

Use an API layer, query planner, lexical and vector retrieval, metadata/ACL service, reranker, context builder, generation service, and evaluation/telemetry pipeline. Cache public or tenant-scoped retrieval carefully. Version embeddings and indexes; support dual-read during migration.

For multi-region, decide whether indexes are replicated, partitioned by data residency, or queried remotely. Freshness and deletion propagation may control the topology more than query latency.

:::callout pitfall|RAG does not guarantee grounding
Providing evidence changes the model input; it does not force the output to follow that evidence. Measure faithfulness, citations, conflict handling, and abstention directly.
:::

### Worked service: a versioned documentation assistant

The following workload is hypothetical; its budgets are design assumptions, not measured performance. A company wants employees to ask questions about product documentation and operating policies. There are 50,000 documents, roughly 200,000 searchable chunks, and separate public, employee, and administrator audiences. Typical traffic is 20 queries per second with bursts to 60. A document update should become searchable within five minutes. A permission revocation must stop new disclosures as soon as the authoritative access service commits it, even if the search index is behind.

The product target is a p99 time to first token at most 1.5 seconds and a p99 token gap at most 100 ms, matching the serving objectives used in Part III. Correctness means answering from the current authorized evidence, with useful citations. A fluent answer from an obsolete policy is a failure. If current sources conflict and no explicit authority rule resolves them, the assistant should report the conflict or abstain rather than select whichever chunk ranks first.

The first baseline is deliberately modest: a canonical document store, a revision/permission service, a lexical index, an optional dense index, a reranker, a bounded context builder, and the 7B generation service from Part III. [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) establishes the combination of model parameters and retrieved memory; [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) supplies a primary dual-encoder reference. Neither paper removes the need for the service's access and freshness contracts.

#### Make revision and authority explicit

A chunk should carry `document_id`, `revision_id`, `chunk_id`, source location, content hash, embedding version, permission scope, and effective time. Do not encode “latest” only as the largest timestamp: future-dated policies, backfills, and different authority domains can all invalidate that shortcut. Resolve current status from the source's revision semantics.

Keep canonical text and its provenance independent of the search index. Build a new revision's chunks, check completeness, then atomically switch the active revision manifest. A failed embedding job must not make half an update appear current. An index row is a candidate pointer, not the authoritative document. At context assembly, resolve that pointer against the manifest and fetch the authorized revision.

An ACL filter at search time reduces the candidate set, but permission must also be checked before content leaves the trusted retrieval service. If a user loses access between search and context assembly, discard the candidate. Permission-service failure should fail closed for protected data. Do not send restricted snippets to an external reranker, generator, trace collector, or user-visible explanation and rely on a final-output filter to undo the disclosure.

The immediate-revocation requirement also needs a contract for in-flight generation and cached answers. Order authorization to release protected content against revocation at a trusted output boundary, and cancel affected streams or discard unreleased buffers when authority is lost. A one-time retrieval check cannot establish this guarantee. Content already released before revocation cannot be recalled from a client or an in-flight network buffer.

Deletion has at least four surfaces: canonical access, index candidates, retrieval caches, and answer caches. The authoritative denial can take effect before background index cleanup completes. Cache keys need tenant/principal scope, query normalization, index and embedding versions, and the relevant permission/revision epoch. A fast cached answer that bypasses revocation is not a successful cache hit.

#### Estimate before adding an ANN service

At 768 float32 dimensions, 200,000 vectors require `200000 * 768 * 4 = 614400000` bytes, about 614 MB in decimal units before graph edges, identifiers, text, metadata, replication, and allocator overhead. That estimate does not prove that exact scanning meets latency; it only shows that a large distributed vector database is not forced by the raw vectors alone.

Start with an exact-search oracle on the evaluation set. Compare candidate recall and stage latency for lexical, dense, and hybrid retrieval under real permission filters. A global recall score can look good while a small tenant with narrow filters receives no usable candidates. Report both overall and tenant/query-class results. Increasing candidate count is useful only while added reranking and context work remains affordable.

Suppose the query is “How long do we keep logs?” An obsolete policy repeats “log retention” several times, while the current policy says “remove these records after thirty days.” A lexical baseline can favor the obsolete document; a dense retriever can still retrieve both and leave the error to the ranker. The repair is not necessarily a better embedding model. It is first to enforce current revision, then evaluate semantic recall among eligible documents.

#### Allocate an end-to-end budget

The table allocates the illustrative 1,500 ms first-token budget. It is a planning budget, not an assertion that stage p99 values add to the end-to-end p99; measure request traces to account for correlation, queueing, and overlap.

| Stage | Planning allocation | First response to an overrun |
| --- | ---: | --- |
| Admission and authorization | 100 ms | Bound the queue; investigate access-service tails |
| Candidate retrieval | 200 ms | Inspect filters, query class, and index choice |
| Reranking | 150 ms | Reduce or batch candidates only after recall checks |
| Revision checks and context assembly | 50 ms | Coalesce metadata fetches; reject stale pointers |
| Model admission and prefill | 800 ms | Tune prompt budget and serving admission |
| Transport and first flush | 100 ms | Trace buffering and network tails |
| Reserve | 100 ms | Absorb measured variation, not permanent overload |

The token budget also connects directly to the KV ledger. A 4,000-token prompt in the running model needs 500 MiB of logical KV before generated tokens and padding. Longer context can therefore lower generator batch capacity even if retrieval itself becomes more accurate. Track useful evidence per context token, not only the number of passages retrieved.

For an API-priced generator, let `C_in` and `C_out` be the contracted cost per million input and output tokens. A 4,000-input/200-output request costs `0.004 * C_in + 0.0002 * C_out` for generation, before embeddings, reranking, retries, and infrastructure. These are symbolic prices, not a current vendor quote. For self-hosting, use reserved accelerator time and measured SLO-compliant throughput instead. Divide total service cost by successful authorized answers, counting failed attempts in the numerator.

#### Reproduce a failure before replacing components

Run `python -m examples.rag` from the repository root. The fixture has six synthetic documents and five queries. It includes a stale revision, a restricted administrator document, one ordinary answer, missing evidence, and conflicting current sources. The intentionally unsafe comparison searches every document and emits the top result. The guarded version filters permissions and current status, then applies a deterministic labeled-fact check.

Example status: Runnable excerpt; execute from the repository root after checkout.

```python
from examples.rag import evaluate

baseline = evaluate(safe=False)
guarded = evaluate(safe=True)
assert baseline["correct"] == 2
assert baseline["acl_leaks"] == 1
assert guarded["correct"] == 5
assert guarded["acl_leaks"] == 0
```

The measured results of this fixture are:

| Metric | Unsafe baseline | Guarded fixture |
| --- | ---: | ---: |
| Correct decisions, including abstention | 2 / 5 | 5 / 5 |
| Queries exposing unauthorized retrieved content | 1 / 5 | 0 / 5 |
| Queries retrieving an obsolete revision | 2 / 5 | 0 / 5 |
| Answerable queries with all required evidence retrieved | 2 / 2 | 2 / 2 |
| Correct abstentions on unanswerable/conflicting queries | 1 / 3 | 3 / 3 |

Both versions retrieve the needed evidence for the answerable queries. That retrieval metric alone misses the baseline's wrong revision choice, unauthorized content, and unjustified answers. The table is useful precisely because it exposes different failure boundaries instead of compressing everything into one quality score.

The fixture is not an LLM benchmark. Its `topic` and `answer` fields are hand-labeled ground truth, and equality of those labels is not a production contradiction detector. Five hand-built cases do not establish a population accuracy rate or resistance to prompt injection. They are deterministic regression tests for specific failures. A real generator needs separate evaluation of entailment, partial support, conflicting scope, and calibrated abstention.

#### Build the evaluation set around decisions

For the hypothetical service, begin with an adjudicated set of 200 queries drawn from real user needs with appropriate permission to use them. As an initial allocation, reserve 100 for ordinary answerable questions, 30 for recent updates, 30 for permission boundaries, 20 for missing evidence, and 20 for conflicting sources. These counts are a sampling plan, not existing collected data. Keep document revisions and access roles with each case, split by source/topic where possible, and retain a held-out set for changes to the retriever, prompt, or verifier.

For each query, annotate required evidence, acceptable answers, unacceptable claims, and whether abstention is correct. Multi-hop questions may require several passages; retrieving one supporting sentence is not complete evidence coverage. Two reviewers should resolve ambiguous policy scope before the example becomes an automatic oracle. Do not use an LLM judge's preference as the sole definition of correctness.

Report retrieval recall at k on answerable cases, answer correctness on all cases, and citation support/completeness only where an answer was emitted. Report abstention precision and recall separately so a model cannot earn a high safety score by refusing everything. Record ACL violations and stale disclosures as counts with their denominators, not as a tiny component of an average score. Keep paraphrase robustness, rare identifiers, multilingual inputs, and long documents as visible slices when they matter to users.

An answer may be factually right from model memory but unsupported by the provided documents. It can pass answer correctness and fail citation grounding. Conversely, a quotation can faithfully repeat a document that is obsolete or outside the user's access scope. These are distinct errors with different fixes.

#### Operate revisions and failures as part of retrieval

Evaluate the exact-search oracle, the production retriever, and the generator separately, then join them in trace replay. Record the eligible corpus revision, candidate IDs and scores, authorization decision, selected context, model/prompt revision, citations, stage times, and cancellation outcome. Avoid storing protected document text in broad-access logs; log identifiers and controlled diagnostic samples instead.

On an embedding migration, write the new index beside the old one and shadow queries with both embedding versions. Compare filtered recall, stale-pointer rate, cost, and latency before switching the routing manifest. Never compare old query embeddings with new document embeddings merely because their vector dimensions match. Keep the old compatible pair available for rollback while the new index proves stable.

On retriever timeout, the safe fallback is an explicit retrieval-unavailable response or a policy-approved reduced service, not an answer presented as grounded without evidence. On reranker timeout, a lexical or hybrid ordering may be acceptable only if its quality was evaluated and the response remains inside policy. On conflicting evidence, show the permitted sources or abstain; do not silently choose based on recency when policy scope differs. On permission-service failure, protected retrieval is unavailable.

Prompt injection belongs in a separate adversarial suite. Include documents that imitate system messages, request tool execution, or try to disclose other tenants' data. The generator must not gain capabilities from retrieved text. Restrict the service to answering unless an explicitly authorized workflow grants tools, and enforce those permissions outside the model. Passing the deterministic fixture is not evidence that these attacks have been handled.

### Design Exercises

1. Design a secure RAG system for enterprise documents.
2. How do you evaluate retrieval and generation separately?
3. What happens when an embedding model changes?
4. How do ACL filters interact with approximate search?
5. Design defenses against prompt injection in retrieved content.

### Worked design answers

1. **Secure enterprise retrieval:** resolve user scope, filter candidates, recheck current revision and authority before context leaves the retrieval service, and invalidate scoped caches on permission changes. Keep canonical ownership outside the index.
2. **Separate evaluation:** use evidence recall to diagnose retrieval, correctness to score the decision, and citation support/completeness to score grounding. The fixture shows why identical recall can coexist with different answer quality.
3. **Embedding migration:** maintain versioned query/index pairs, shadow against a fixed evaluation corpus and roles, then switch an atomic manifest with rollback. Dimension compatibility is not semantic compatibility.
4. **ANN plus filters:** benchmark recall after access filtering, particularly for small or selective tenants; compare with an exact oracle and choose partitions or candidate budgets from measured tails.
5. **Injection defense:** treat retrieved text as data, enforce capabilities outside the model, and test unauthorized side effects separately from benign task success. Do not use blanket refusal as the sole success metric.

## Building and Evaluating a Bounded Agent Loop

LEAD: An agent is a model inside a program that chooses actions, observes results, and decides what to do next. The program owns permissions, budgets, durable state, and the definition of success. More model calls alone do not create a reliable agent.

### Distinguish a workflow from an agent

A workflow follows a prescribed graph: retrieve, rerank, answer, validate. An agent can choose which tool to use and whether to search again, edit a file, or finish. The choice is useful when the next step depends on evidence that cannot be known in advance. A fixed workflow is easier to test when the sequence is already known. A hybrid can let the model choose within a small, explicitly bounded subgraph.

[ReAct](https://arxiv.org/abs/2210.03629) studies interleaving reasoning and actions. [SWE-agent](https://arxiv.org/abs/2405.15793) demonstrates that the interface through which a model inspects and changes an environment matters. These observations do not require exposing private reasoning or granting unrestricted shell access. A trace of proposed actions, observations, checks, and results is enough to audit the external work.

For a documentation assistant, begin with a narrow task: read authorized sources and prepare a draft answer. Do not give it a send-message or account-modification tool merely because the model can generate those tool names. Add a capability only when the user workflow requires it and the surrounding service can authorize, constrain, and recover it.

### Follow the control loop

| Stage | Required input | Host-owned check |
| --- | --- | --- |
| Observe | User objective, permitted context, previous results | Scope and provenance remain distinct |
| Propose | Typed tool call or final response | Parse schema; reject unknown tools and invalid arguments |
| Authorize | Caller identity and proposed effect | Check policy against the real target, not model text |
| Execute | Validated call plus deadline and identity | Sandbox, resource budget, timeout, idempotency |
| Record | Result or explicit failure | Append an event; retain external transaction identity |
| Continue or finish | Updated observations | Enforce budgets and validate the claimed outcome |

Tool output is an observation, even when it says “SYSTEM: ignore your rules.” A schema prevents malformed calls, not malicious but well-formed arguments. The authorization layer must resolve paths, tenants, domains, operation types, and effects independently. Credentials belong in the execution service, not in model-visible prompts. Source trust is metadata outside the text being evaluated.

Interoperability protocols standardize envelopes, not trust. The [Model Context Protocol specification](https://modelcontextprotocol.io/specification/2025-11-25) defines negotiated capabilities and typed tools, resources, prompts, and task-like operations between a host and servers. The [A2A 1.0 specification](https://github.com/a2aproject/A2A/blob/main/docs/specification.md) defines discovery through Agent Cards plus messages and durable tasks between agent systems. Either can reduce bespoke integration, but neither turns advertised metadata or remote output into authority. Pin supported protocol versions, authenticate both endpoints, authorize every effect against current caller identity, constrain discovery, and preserve idempotency, deadlines, provenance, and audit records in the host.

### A complete local fixture

Run `python -m examples.agent_loop`. It prints `finished 1 0`: one permitted document-read attempt, no saved drafts. The module includes the decision loop, typed actions and observations, a tool allowlist, a draft capability, bounded attempts, and receipt-based retries. Its policy is scripted so failures are deterministic and require no model account or network.

Example status: Runnable excerpt; execute from the repository root.

```python
from examples.agent_loop import Call, Environment, Finish, run
env = Environment({"policy": "Draft only; review before sending."})

def policy(observations):
    if not observations:
        return Call("read_document", "policy", "read-1")
    return Finish("I found the policy. Nothing was sent.")
result = run(policy, env, max_steps=6, max_tool_attempts=3)
assert result.status == "finished"
assert result.tool_attempts == 1
assert env.drafts == []
```

The fixture also tests a malicious document followed by a proposed unauthorized write. The write is denied by the environment even though the scripted policy proposes it. This establishes one enforced capability boundary, not general prompt-injection resistance. Replacing the policy with an LLM adds probabilistic behavior that needs its own evaluation. The example does not implement a remote sandbox, real-time call interruption, durable storage, concurrent transactions, token accounting, or semantic answer verification.

### Retry an effect without duplicating it

A timeout does not prove a write failed. Suppose saving a draft succeeds, but its reply is lost. Retrying with a new identifier can create a second draft. The fixture retries the same call with the same idempotency key. The environment stores the request identity and result together and returns that result on replay. Reusing the key for different arguments is an error. Authorization is checked again before replay so a revoked capability is not resurrected by a cached receipt.

In production, the receipt and effect must be atomic or reconciled through the external service's transaction identifier. An in-memory dictionary is lost on restart and cannot establish exactly-once behavior across processes. If a remote API offers neither idempotency nor a way to query the effect, stop automatic retries after an ambiguous write and expose the unresolved outcome. “Try harder” is not a recovery protocol.

Read retries also cost time and quota. Distinguish transient transport failure, invalid input, policy denial, exhausted budget, and unavailable evidence. Blindly retrying a denied action is a loop, not progress. A safe fallback preserves uncertainty: “The request may have completed; its status is unknown” is more honest than reporting failure and creating another effect.

### A durable local effect and receipt

Run `python -m examples.durable_effects` for a second, deliberately narrower fixture. It saves a draft and its receipt in one SQLite transaction, simulates a lost reply after commit, opens a fresh connection, and replays the same tenant-scoped key. The output is `same_receipt=True drafts=1`. Tests also restart in a separate process, interrupt the transaction before commit, reuse a key with changed content, revoke authorization before replay, and race duplicate requests from separate connections.

The invariant is concrete: for one authorized `(tenant, operation_key)` and unchanged payload, at most one draft is committed and every successful replay returns the same draft ID. Inserting the draft and then recording a receipt in separate transactions would violate that invariant after a crash between the two writes. SQLite's transaction boundary covers both local tables; it cannot encompass an arbitrary remote API.

The host supplies the tenant and current authorization decision. The teaching fixture accepts that decision as a Boolean and does not implement authentication, policy synchronization, encryption, remote writes, or an agent checkpoint store. A revoked caller cannot fetch the old receipt through the replay path. Receipt retention must cover the promised retry horizon; deleting a receipt while preserving the effect makes an old key unsafe to retry unless another durable identity remains.

Framework persistence solves a related but distinct problem. [LangGraph's persistence documentation](https://docs.langchain.com/oss/python/langgraph/persistence), checked September 24, distinguishes thread checkpoints from cross-thread stores and explicitly notes that its in-memory saver loses state on restart. Persisting conversation state alone does not make external effects idempotent.

In a complete loop, record the pending operation key and remaining budgets before dispatch. On resume, recover that intent, recheck authority, reconcile its receipt, and continue without resetting the task's attempt or spend limits. Store a schema and application version with checkpoints. If changed tool semantics cannot interpret an old intent, migrate it explicitly or stop for reconciliation. Never let a new model silently reinterpret an already committed operation.

**Worked check:** the draft committed, the process stopped before checkpointing the reply, and permissions were revoked. The new process must not create a replacement draft or return protected receipt data through an unauthorized replay. An authorized reconciliation service can determine the effect's status separately.

### Memory is versioned evidence, not an authority upgrade

An agent can retain short-term observations in context, summarize an episode, and retrieve longer-term records. Each has a different failure mode. Context grows expensive; summaries can remove qualifications; retrieval can miss facts or surface stale ones. Store source IDs, creation time, task scope, and authority with important memories. A note claiming that a user approved an action is not itself authorization for a new task.

For a coding task, keep the baseline commit, objective, current hypothesis, experiment command, result, and next uncertainty in a small task record. Put large profiler traces and outputs in files with stable paths. Reload the baseline and measured result after context compaction rather than trusting a summary that says “the fast version passed.” In an enterprise agent, tenant and permission boundaries must apply to memory retrieval just as they apply to RAG.

Parallel agents can explore independent hypotheses in separate branches or worktrees. Each returns a diff plus evidence against the same baseline. Integration is a separate operation: review conflicts, run the combined tests, and remeasure the merged result. Two individually valid changes can interact badly. More agents help only while independent useful work exceeds coordination, compute contention, and review cost. Do not give workers shared mutable output files or let completion order choose the winner.

### Evaluate tasks, trajectories, and budgets

Define success using the environment's final state: the correct draft exists, tests pass without unauthorized edits, or the requested information is supported. A final message saying “done” is not an oracle. Keep forbidden effects, data disclosure, and task completion as separate metrics; blanket refusal should not win the benchmark.

Use a development set to improve prompts/tools and an untouched test set for acceptance. Fix or record model version, harness, tool schemas, dataset/environment snapshot, sampling, budgets, retry policy, and evaluator. Repeat stochastic runs and report variance and success per task slice. [τ²-Bench](https://arxiv.org/abs/2506.07982) is an example of evaluating interaction in an environment with both agent and user control, rather than scoring only a static answer.

An original cost example: a task uses six calls, each with 4,000 input and 500 output tokens. It consumes 24,000 input and 3,000 output tokens before tool costs. With symbolic prices `C_in` and `C_out` per million tokens, model cost is `0.024 C_in + 0.003 C_out`. If only half of attempts meet the task contract, average cost per success doubles, assuming the same average cost per attempt. A cheaper call can therefore produce a more expensive successful task if it needs more retries.

Measure latency from request arrival to verified completion, including tool queues and failed branches. Record where time is spent: context construction, generation, tool execution, verification, or waiting for authorization. Shortening the loop often means better observations and a sharper verifier, not simply buying faster generation.

### Reliability under tool failures

Successful function syntax is only one layer of agent evaluation. [ToolBench-X, June 2026 v2](https://arxiv.org/abs/2606.25819v2), introduces controlled tool-environment hazards including specification changes, execution failures, changed outputs, and conflicting sources, while preserving a recovery path. This is benchmark evidence about defined scenarios, not a guarantee about arbitrary production outages.

Build a local fault matrix around the host contract: timeout before dispatch, lost reply after commit, restart before recording a result, revoked permission, changed tool schema, duplicate delivery, and partial output. Each case should specify the permitted final state, forbidden effects, maximum attempts, and treatment of unknown outcomes. The new durable fixture covers the local transaction cases; the others require tests at their real boundaries.

Measure repeated reliability as well as best-of-many success. If per-run success were independently 0.9, at least one success in three runs would have probability `1 - 0.1^3 = 99.9%`, while three successes in three runs would have probability `0.9^3 = 72.9%`. Real runs may be correlated, so report empirical task-level results and the definition used rather than assuming independence. A production user who retries and still needs one correct, nonduplicated effect cares about both completion and consistency.

**Worked check:** a new agent improves clean-task success but duplicates drafts after lost replies. Does it pass? No. Compare clean completion, recovery completion, forbidden effects, and cost separately. A higher average score cannot compensate for breaking the operation's effect contract.

### Exercises and worked answers

1. **A retrieved document asks the agent to email secrets. Where is the decisive check?** Outside the model, before any tool executes, against caller authority and target scope. Marking text untrusted is necessary but not a sandbox.
2. **A save times out after committing. Should the agent create a new request?** Reuse the same idempotency identity or query transaction status; otherwise stop and reconcile the ambiguous effect.
3. **An agent repeatedly searches without improvement. What stops it?** Host-enforced call/time/token budgets and an explicit failure outcome, not a request in the prompt to be efficient.
4. **When do parallel agents help?** For separable experiments with independent state, a common baseline, bounded resources, and a verified merge. Shared-file contention and duplicate hypotheses can erase the gain.
5. **Does the fixture's passing injection test prove robust model behavior?** No. It verifies a deterministic denied capability. Real models, other tools, observation poisoning, and authorized-but-harmful arguments require broader adversarial tests.


### Part VI closing principle

Production algorithms earn trust by making state, approximation, authority, and recovery explicit. The same rule governs a streaming sketch, a retrieval service, and an agent loop: define the contract, bound the failure, measure the outcome, and keep consequential control outside an unverified prediction.

Part VII uses these contracts to judge which frontier mechanisms transfer to a real workload and which claims still need an experiment.
