# Part II - Training Systems, Data, and Adaptation

Training turns a capability hypothesis into a reproducible model release. The work is not a sequence of independent choices about data, loss, optimizer, and scale. It is a controlled program in which evidence from pilots determines what deserves more compute, every dataset release can be reconstructed, and every checkpoint can be evaluated against the behavior and serving envelope it is meant to satisfy.

This part follows that program from end to end. It begins with recipe design and experiment gates, then develops data contracts, filtering, synthetic generation, mixture control, and release engineering. It closes with distillation and post-training, where a broad base distribution is transformed into a model with a specific capacity, policy, and operating role.

## Designing a Training Recipe

LEAD: A recipe is an executable theory of how a model will acquire a capability within a compute and time budget. Begin with the target behavior and work backward to data and objective.

### The recipe canvas

Before selecting hyperparameters, write a one-page canvas:

- **Product behavior:** tasks, users, languages, modalities, context, and safety boundary.
- **Model constraints:** parameter budget, serving hardware, latency, throughput, memory, and quantization target.
- **Data assets:** licensed corpora, synthetic generation, preference labels, verifier signals, and contamination risks.
- **Training stages:** pretraining, continued pretraining, supervised fine-tuning, preference or RL stage, distillation, and compression.
- **Evaluation gates:** capability, generalization, safety, calibration, and system performance.
- **Operations:** failure recovery, lineage, reproducibility, checkpoint cadence, and ownership.

:::diagram training_pipeline|A production recipe is a loop. Serving failures become curated examples, evaluation updates, and the next training stage.

### Work backward from serving

If the product needs high-volume interactive decode, architecture and tokenizer choices should reflect KV-cache and output-length costs. If the product needs code completion, data freshness, exact syntax, repository context, and executable verification matter more than generic preference labels. If the product needs a speculative draft, target-distribution agreement and acceptance speed matter more than standalone benchmark leadership.

A common failure is to train the most capable model that fits the training cluster and discover later that it cannot meet the serving SLO. The lifecycle objective is closer to:

:::equation Value = QualityGain - TrainCost - ServeCost - OperatingRisk|A training program optimizes lifetime utility, not validation loss in isolation.

The terms are not directly commensurate, but writing them forces a complete decision.

### Budget the program, not only the final run

The nominal training run is only one consumer of compute. A credible budget includes data ablations, architecture pilots, hyperparameter sweeps, failed runs, checkpoint evaluation, teacher inference, safety testing, and the reserve needed to repeat a result. Spending nearly all available compute on one scale-up removes the ability to diagnose or correct it.

Partition the budget into three envelopes:

- **learning budget:** small experiments that estimate response to data, model, and schedule changes;
- **commitment budget:** the selected training run and its planned continuations;
- **assurance budget:** evaluation, red-teaming, reproducibility, rollback preparation, and one credible recovery path.

For dense transformer training, a first-order compute model is proportional to parameters times training tokens:

:::equation C_{train} ≈ k P T|The constant k depends on architecture, forward-backward accounting, sparsity, and implementation.

The estimate is useful for comparing candidate scales, not for purchasing a cluster. Convert it into wall-clock time using measured utilization from representative pilots, then add evaluation and checkpoint overhead. A model that fits the arithmetic budget can still miss its deadline because of input stalls, collectives, failures, or poor local matrix shapes.

### Stage design

**Pretraining** builds broad representations from next-token prediction or a related self-supervised objective. **Continued pretraining** shifts domain, language, freshness, or context distribution while protecting general capability. **Supervised fine-tuning** teaches response formats and high-quality trajectories. **Preference optimization** shifts behavior among plausible outputs. **Distillation** transfers a distribution or capability into a smaller serving envelope.

Stages should be justified by what signal they add. Do not add a preference stage because it is fashionable if verified demonstrations already specify the desired answer. Do not use supervised imitation for an objective that requires exploration and delayed reward.

Each transition needs an entry contract and an exit gate. Continued pretraining should name the capability or distribution shift it targets and the general capabilities it must preserve. Supervised fine-tuning should specify which response structures or tool trajectories demonstrations teach. Preference optimization should begin only when comparisons express a stable ordering that is not already captured by verified labels.

### Use an experiment ladder

Training evidence should become more expensive only as uncertainty falls:

1. **pipeline proof:** hundreds of steps establish data, loss, checkpoint, and metric correctness;
2. **overfit test:** a tiny dataset verifies that the model can drive the intended objective down;
3. **small-scale pilot:** alternative mixtures, schedules, or losses are compared at matched tokens;
4. **scaling pilot:** two or more scales test whether the observed gain persists and estimate utilization;
5. **confirmation run:** the selected recipe is repeated or validated on a protected dataset;
6. **production run:** the full budget is committed with predeclared review points;
7. **release qualification:** immutable checkpoints pass capability, safety, and serving gates.

Do not let the ladder become ritual. A deterministic code change may need correctness and performance tests but no statistical pilot. A new data mixture needs controlled training evidence because its effect cannot be inferred from the pipeline alone.

### Curriculum and mixture

Data mixture weights determine the gradient distribution. Sampling every source in proportion to raw size lets abundant low-value sources dominate. Equal sampling overweights tiny domains and can cause memorization. Temperature sampling interpolates between these extremes.

Curriculum can mean ordering by difficulty, capability, context length, quality, or domain. It is useful when later data assumes skills learned earlier or when expensive long sequences should be introduced after the model is stable. It is harmful when the sequence creates catastrophic forgetting or a distribution cliff.

#### Mixture control loop

1. Define capability slices with leading metrics.
2. Estimate source-to-slice influence through ablations or data attribution proxies.
3. Choose a baseline mixture that covers product distribution.
4. Train small pilots across a sparse set of mixture changes.
5. Fit a response model only as complex as the evidence supports.
6. Update the mixture and preserve held-out confirmation data.

:::callout pitfall|Synthetic volume is not synthetic value
Teacher generations reproduce teacher biases and can collapse diversity. The useful unit is a verified, novel training signal, not a generated token.
:::

### Schedules and stop conditions

Warmup protects early training when moment estimates and activation scales are unstable. Cosine or linear decay then reduces update size as the run approaches a local basin. Restart schedules can help exploration but complicate reproducibility and interpretation.

Stopping only when training loss plateaus wastes compute. Use a portfolio of signals: validation loss, target capability, data-slice trends, gradient noise, checkpoint-to-checkpoint improvement, projected value of more tokens, and schedule milestones. A predeclared review point makes it easier to stop a prestigious run whose marginal return has collapsed.

Compare checkpoints by tokens and wall-clock, not by step number alone. Track the marginal gain between evaluation points:

:::equation MarginalGain = (Metric(t_{2}) - Metric(t_{1})) / (Tokens(t_{2}) - Tokens(t_{1}))|A declining marginal gain supports an explicit continue, change, or stop decision.

The metric in this calculation must be tied to the program objective. Validation loss can guide broad pretraining, while executable pass rate or calibrated task success may be more appropriate for a capability-specific continuation. Guardrails remain constraints rather than terms that can be averaged away.

### Checkpointing and recovery

A distributed checkpoint must capture parameters, optimizer states, scheduler, RNG state, data-loader position, tokenizer and configuration hashes, parallel topology assumptions, and code version. Asynchronous checkpointing reduces pause time but requires a consistency boundary. Sharded checkpoints reduce write hotspots but make restore topology and format evolution first-class design problems.

Use periodic full checkpoints plus more frequent lightweight or incremental protection. Test restore before the expensive run. A checkpoint that has never been restored is a hypothesis.

### Worked recipe: a reasoning model

For a reasoning model, start with a strong base model and define the target beyond “longer reasoning.” Separate mathematical derivation, code execution, structured planning, evidence synthesis, and tool use because they have different verification surfaces.

Construct tasks with independently verifiable answers and retain diverse solution trajectories rather than only the teacher's preferred style. Use supervised training to establish tool syntax, decomposition patterns, and termination behavior. Add outcome- or process-based optimization only where the reward remains reliable under policy improvement.

A staged program might proceed as follows:

| Stage | Training signal | Entry evidence | Exit gate |
| --- | --- | --- | --- |
| Capability pilot | Verified demonstrations | Base model can solve a measurable fraction | Gain persists on protected task families |
| SFT | Correct and diverse trajectories | Format and verifier are stable | Better success without excessive length growth |
| Verifiable optimization | Hidden executable outcomes | Reward resists simple exploit probes | Independent evaluation improves with bounded divergence |
| Distillation | Teacher distributions and selected traces | Serving target is defined | Cost-adjusted task success beats the baseline |
| Release | No additional gradient updates | Immutable candidate checkpoint | Quality, safety, latency, and rollback gates pass |

Prevent reward hacking with hidden tests, adversarial cases, generator-verifier separation, and audits of suspiciously short or repetitive traces. Track answer accuracy, pass@k, calibration, token efficiency, tool validity, and a taxonomy of failure mechanisms. A reasoning program succeeds when the model solves more valuable tasks reliably, not when it merely emits more intermediate tokens.

:::callout insight|A recipe should have gates
Name what evidence permits progression from pilot to scale-up, from supervised training to online optimization, and from offline evaluation to serving. Gates show that you can operate the program, not merely describe it.
:::

### Design Exercises

1. Design a recipe for a code model that must run on a fixed single-GPU serving tier.
2. When would continued pretraining be safer than changing the base mixture from the start?
3. How would you decide whether to spend the next budget increment on data, model size, or more steps?
4. What belongs in a resumable checkpoint for a sharded optimizer?
5. How do you prevent a reasoning model from learning to exploit its verifier?

## Data Contracts, Provenance, and Normalization

LEAD: A training dataset is a versioned product, not a folder of text. Its contract defines what may be used, what each record means, how it was transformed, and how a source can be traced or removed after publication.

:::diagram data_pipeline|A dataset is the result of transformations and gates. Every retained example should have provenance and a reason to exist.

### Start with the data contract

The word "example" hides several distinct objects:

- a **source object** is the retrieved file, page, repository revision, conversation, image, or record;
- a **document** is a parsed logical unit with boundaries and metadata;
- a **training example** is a selected or generated unit before token packing;
- a **sequence** is the tokenizer-specific tensor presented to the model;
- a **dataset release** is an immutable manifest selecting exact versions of those objects.

Counts at these layers are not interchangeable. Deduplicating documents can leave repeated substrings inside long documents. Packing can join examples and obscure their source boundaries unless lineage is carried separately. A report that says "we trained on 10 billion examples" is incomplete without the unit, tokenizer, filtering version, and sample weights.

Before building the pipeline, define:

| Dimension | Questions that determine the implementation |
| --- | --- |
| Intended use | Pretraining, supervised tuning, preference learning, retrieval, evaluation, or red-teaming? |
| Unit | Document, turn, function, repository, image-text pair, trajectory, or packed sequence? |
| Rights and policy | What use is permitted, for how long, in which products, and under what deletion obligations? |
| Identity | What makes two records the same source revision, the same bytes, or the same semantic example? |
| Utility | Which capabilities should improve, and what evidence makes an example useful? |
| Risk | Privacy, toxicity, security, leakage, poisoning, and representational harms? |
| Distribution | Which domains, languages, difficulty levels, and rare critical slices need what mass? |
| Reproducibility | Can a release be reconstructed byte-for-byte from immutable inputs and versioned code? |

Use notation consistently:

- `N` is a document or example count, as stated locally;
- `T` is a token count under a named tokenizer version;
- `n_d` is available token mass in domain `d`;
- `q_d` is the probability of sampling domain `d`;
- `w_i` is the effective training weight of example `i`;
- `H(bytes)` is a cryptographic content digest, not a semantic identifier.

The main design principle is separation of concerns. A policy gate answers "may this data be used?" A corruption gate answers "can it be parsed safely?" A quality score estimates expected utility. A mixture policy decides how often an eligible example should influence gradients. Combining all four into one opaque scalar makes failures difficult to diagnose and reverse.

### Source contracts and provenance

For every source, record the acquisition and use contract before expensive processing. At minimum:

- source namespace, stable source ID, revision, retrieval time, and retrieval method;
- owner or publisher, asserted license or agreement, policy decision, and decision owner;
- permitted purposes, geographic or product restrictions, retention period, and deletion obligations;
- original URI or object location, raw byte digest, media type, and declared encoding;
- consent or privacy classification where applicable;
- language, domain, temporal coverage, and known collection bias;
- parent objects, transformations, generated descendants, and release memberships.

This metadata supports engineering and governance review; it is not a substitute for legal or privacy judgment. Unknown rights should be represented as unknown, not silently mapped to permitted.

#### Identity is layered

Use different identifiers for different questions:

`source_revision_id = H(source_namespace || source_id || revision)`

identifies one logical revision from one source. A raw content hash:

`raw_content_id = H(raw_bytes)`

finds byte-identical objects across locations. A normalized content hash:

`normalized_content_id = H(normalizer_version || normalized_bytes)`

finds exact duplicates after a declared transformation. These IDs must remain separate. If the source ID is mixed into the content digest, mirrors will no longer deduplicate. If only a content hash is kept, two independently licensed copies become indistinguishable for deletion and rights analysis.

Hashes also have scope. A hash collision is unlikely with an appropriate cryptographic digest, but a digest does not prove authorship, permission, semantic equivalence, or absence of personal information. Store the algorithm and canonical serialization in the schema.

#### Model provenance as a graph

Represent provenance as a directed acyclic graph:

- **entities:** raw objects, parsed documents, normalized examples, synthetic records, shards, manifests, and checkpoints;
- **activities:** fetch, parse, normalize, deduplicate, score, generate, verify, sample, tokenize, and pack;
- **agents:** source owners, pipelines, models, reviewers, and approving teams.

An output entity points to its input entities and the exact activity version that produced it. A packed sequence may therefore point to several examples; a synthetic answer points to its seed, generator checkpoint, prompt template, decoding parameters, verifier, and adjudication record.

The graph must answer both forward and reverse questions:

- Which release and checkpoint contain descendants of source object `s`?
- Which raw objects, transforms, scorers, and random seeds produced sequence `z`?
- Which examples changed when normalizer version `v7` replaced `v6`?
- Which evaluations used a benchmark item that entered a training release?

Lineage that exists only in logs is fragile. Keep durable edge tables or manifests, version their schema, and test impact queries before an incident.

#### A minimal record envelope

Example status: Illustrative Python excerpt; not standalone.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class DataEnvelope:
    source_revision_id: str
    raw_content_id: str
    parent_ids: tuple[str, ...]
    acquisition_time: str
    use_policy_id: str
    transform_version: str
    payload_uri: str
    metadata_uri: str
```

The payload can remain in object storage while the envelope flows through distributed transforms. Derived records receive new IDs and retain parent edges. Mutating a record in place destroys the ability to explain an old release.

#### Worked deletion trace

Suppose repository revision `R17` produced 800 parsed files, 620 normalized examples, 410 deduplicated representatives, and 75 synthetic debugging tasks. Three released manifests selected 190 of those descendants, and two checkpoints trained on the releases.

A valid deletion workflow:

1. tombstones `R17` and blocks it from future selection;
2. traverses all descendant edges, including synthetic records;
3. creates replacement manifests without affected records;
4. invalidates caches, indexes, and pending generation jobs;
5. records which checkpoints and evaluations were exposed;
6. applies the organization's model-remediation policy;
7. emits an auditable completion record.

Deleting the current copy of `R17` from raw storage is insufficient. It leaves derived shards, packed sequences, indexes, and model artifacts unexplained.

:::callout insight|Provenance is an operational capability
The test is not whether a dataset has a descriptive card. The test is whether the system can reconstruct a release, explain one example, propagate a deletion, and enumerate affected models under incident pressure.
:::

### Normalization without erasing signal

Normalization converts heterogeneous source objects into stable, comparable records. It should remove representation accidents while preserving task-relevant meaning. The contract is domain-specific: whitespace collapse that helps web prose can corrupt Python; stripping markup can destroy tables, equations, links, or speaker roles.

#### Separate parsing from canonicalization

A robust pipeline uses explicit stages:

1. **decode:** validate bytes, encoding, decompression limits, and media type;
2. **parse:** recover logical structure such as title, headings, paragraphs, code blocks, turns, and tables;
3. **canonicalize:** standardize safe representation details;
4. **clean:** remove source-specific boilerplate using versioned rules;
5. **segment:** create training units without crossing protected boundaries;
6. **validate:** check invariants and compare with the raw object.

Keep the raw payload immutable. Store the normalized representation, structured parse, offset map where feasible, normalizer version, and a diff summary. If a later parser improves table recovery, the release can be rebuilt without recrawling or guessing what was removed.

#### Unicode and encoding choices

Canonical Unicode normalization such as NFC maps canonically equivalent sequences to a stable representation. Compatibility normalization such as NFKC can also fold distinctions including width, superscripts, and some mathematical character variants. That may help identifier matching and deduplication, but it can erase meaning in mathematics, code, linguistics, and historical text.

Therefore:

- default prose may use NFC after valid decoding;
- code should preserve bytes or apply a language-aware representation;
- mathematical text should retain compatibility distinctions unless a reviewed transform says otherwise;
- case folding belongs in a comparison view, not necessarily in the training payload;
- control characters, bidirectional markers, and zero-width characters need explicit policy and security review.

Do not silently replace decoding errors with an undifferentiated placeholder and then treat the result as clean text. Track replacement count and byte ranges, route high-error documents to quarantine, and retain the original bytes.

#### Preserve structure that carries supervision

Document and example boundaries affect the objective. Preserve:

- indentation and newline structure in code;
- cell, header, and row relationships in tables;
- equation spans and surrounding prose;
- speaker, tool, and system roles in conversations;
- repository path and file relationships for code;
- title, heading, list, citation, and hyperlink structure where useful;
- page or section location when later grounding or deletion requires it.

Boilerplate removal should operate on parsed regions, not arbitrary repeated lines alone. A copyright footer repeated across a site may be removable; a repeated function signature or safety warning may be the exact signal the model needs.

#### Determinism and invariants

For fixed raw bytes and configuration, normalization should be deterministic:

`normalize(raw, version, config) -> (structured_record, audit_metadata)`.

Useful invariants include:

- **idempotence:** `normalize(normalize(x)) = normalize(x)` for the canonical text view;
- **bounded loss:** removed characters and regions are counted and classified;
- **boundary preservation:** protected code, table, and conversation regions cannot merge accidentally;
- **stable identity:** content IDs include the normalizer version;
- **resource safety:** decompression ratio, nesting depth, field length, and parse time are bounded;
- **replayability:** locale, Unicode version, parser model, and rule bundle are pinned.

Sample diffs by source, language, and document type. Aggregate averages can hide a rule that corrupts every document in a small language.

#### Language and modality routing

Language identification should return scores and coverage, not only one label. A document can contain English explanation, a Spanish quotation, and code. Record predictions at document and segment level, plus model version and confidence.

Route low-confidence, mixed-language, code-heavy, OCR-heavy, and formula-heavy content to specialized parsers or conservative paths. A hard English-only threshold can remove code, named entities, transliteration, and minority dialects while appearing highly accurate on a benchmark dominated by clean monolingual prose.

#### Normalization review checklist

Before publishing normalized data, verify:

- round-trip links to raw objects and offset maps for audited samples;
- per-domain distributions of length, replacement characters, removed fraction, and parse failures;
- code compilation or parser success where applicable;
- table, equation, and conversation-turn preservation;
- deterministic rerun hashes;
- security limits against archive bombs, malformed markup, and adversarial nesting;
- no train/evaluation split decision was made using a representation unavailable during reconstruction.

## Deduplication, Quality, and Contamination

LEAD: Filtering changes the empirical distribution learned by the model. Deduplication, quality scoring, and contamination controls must therefore be evaluated as statistical interventions, not treated as generic cleaning steps.

### Deduplication at several radii

Deduplication changes the empirical training distribution. It can reduce wasted compute, benchmark overlap, and repeated memorization, but an overly broad rule can erase legitimate repetition, minority sources, translations, version history, or pedagogically useful variants.

Define the target relation before the algorithm:

- **byte identity:** exactly the same raw bytes;
- **canonical identity:** exactly the same normalized payload;
- **near duplication:** high surface overlap after limited editing or templating;
- **containment:** most of a short item appears inside a longer one;
- **structural duplication:** the same code or template with renamed identifiers and literals;
- **semantic duplication:** equivalent meaning with little lexical overlap.

These relations are not transitive in the same way. Exact identity forms clean equivalence classes. Thresholded similarity can create chains: `A` is close to `B` and `B` to `C`, while `A` is not close to `C`. Blind connected components can therefore merge an entire topic through bridging documents.

#### Exact deduplication

Compute a cryptographic digest over a declared canonical serialization. Preserve all source memberships even when one content object becomes the representative:

Example status: Illustrative Python excerpt; not standalone.

```python
def exact_identity(record, normalizer_version, digest):
    canonical = serialize_canonical(record)
    payload = normalizer_version.encode() + b"\x00" + canonical
    return digest(payload).hexdigest()
```

The null separator prevents ambiguous concatenation. The production schema also stores digest algorithm, serialization version, byte length, and source revision IDs.

Decide the scope:

- **global deduplication** removes repeats across all snapshots and sources;
- **snapshot-local deduplication** preserves temporal recurrence across snapshots;
- **source-local deduplication** prevents one source from dominating without erasing corroboration across independent sources.

The right scope depends on the objective. News recurrence may encode importance and temporal change; repeated scraped navigation usually does not.

#### Shingles and Jaccard similarity

Represent a document by a set of shingles, such as word 5-grams:

`S(D) = {all distinct length-5 token windows in D}`.

For sets `A` and `B`, Jaccard similarity is:

:::equation J(A,B) = size(A ∩ B) / size(A ∪ B)|Jaccard similarity measures intersection relative to union.

This treats repeated occurrences inside one document as one set element. Multiset Jaccard or frequency-aware methods are different contracts. Token shingles are robust to small character changes but depend on tokenizer and language; character shingles handle tokenization variation but can overemphasize boilerplate.

Containment is asymmetric:

:::equation C(A,B) = size(A ∩ B) / size(A)|Containment asks what fraction of the smaller reference set appears in the candidate.

If a 100-shingle answer appears inside a 10,000-shingle book, containment can be one while Jaccard is near `0.01`. Use containment when short benchmark items, quotations, or snippets embedded in long documents matter.

#### Deriving MinHash

Computing exact Jaccard for every pair is quadratic. MinHash creates a fixed-size signature. Choose a random permutation `pi` of the shingle universe and define:

`h_pi(A) = min_{x in A} pi(x)`.

In the union `A union B`, every element is equally likely to be first under a random permutation. The two minima agree exactly when that first element belongs to `A intersect B`. Therefore:

:::equation P(h_{π}(A) = h_{π}(B)) = J(A,B)|A MinHash collision occurs with probability equal to Jaccard similarity.

With `k` independent permutations or suitably constructed hash functions:

:::equation J_{hat} = (1/k) Σ_{j=1}^{k} 1[h_{j}(A) = h_{j}(B)]|The matching fraction across independent hashes estimates Jaccard similarity.

Each indicator is Bernoulli with mean `J`, so:

:::equation E[J_{hat}] = J|The MinHash collision estimator is unbiased for Jaccard similarity.

and, under independence,

:::equation Var(J_{hat}) = J(1-J)/k|Estimator variance falls inversely with the number of hashes.

The standard error is largest near `J=0.5`. With `k=128`, it is at most `sqrt(0.25/128) ~= 0.044`. A signature is a noisy similarity estimate, not proof of duplication.

Real systems approximate independent permutations using seeded hash families. Seeds, shingling, tokenizer, hash width, empty-document behavior, and signature length are part of the serialized state.

#### Locality-sensitive hashing

Even fixed-size signatures are expensive to compare all-pairs. Divide a `k = b r` MinHash signature into `b` bands of `r` rows. Make a candidate pair when at least one complete band matches.

If true Jaccard similarity is `s`, one band matches with probability `s^r`. Assuming independent bands:

:::equation P(candidate;s) = 1 - (1 - s^{r})^{b}|LSH banding converts similarity into a tunable candidate-retrieval probability.

This is an S-shaped retrieval curve, not a hard threshold. With `k=128`, `b=16`, and `r=8`:

- at `s=0.8`, candidate probability is about `0.947`;
- at `s=0.5`, candidate probability is about `0.061`.

Increasing `r` reduces low-similarity candidates but can miss real duplicates. Increasing `b` improves recall and increases index and verification work. Select parameters from labeled duplicate pairs and operational budgets, then compute exact Jaccard or a stronger comparison for retrieved candidates.

#### Scale accounting

A 128-entry signature with 64-bit hashes consumes `1,024` bytes before IDs, index structures, and compression. For 100 million documents, raw signatures alone are:

`100,000,000 * 1,024 bytes ~= 95.4 GiB`.

The band index, shuffle copies, candidate edges, and cluster metadata can exceed that. Capacity planning must include:

- signatures and their replication;
- band-key postings and skew from empty or boilerplate documents;
- candidate verification reads;
- graph or cluster state;
- intermediate shuffle and checkpoint storage;
- incremental updates and old-version coexistence.

Exclude extremely short or low-information documents from ordinary LSH paths or use separate rules; otherwise common boilerplate shingles create huge hot buckets.

#### Cluster construction and representative choice

Candidate generation is not clustering. After verification, choose one of:

- pairwise suppression against already selected representatives;
- star clusters around a stable representative;
- bounded-diameter clustering;
- connected components with an explicit chaining-risk audit;
- source-aware quotas that keep independent or temporally important variants.

Representative selection is a product decision. The shortest item may lose context; the longest may preserve spam. Use a deterministic ranking over:

1. policy and rights confidence;
2. source authority and provenance completeness;
3. parse integrity and information density;
4. contextual completeness;
5. recency or temporal relevance;
6. stable source and content IDs as final tie breakers.

Retain cluster membership even when only one item trains. This enables deletion propagation, audit, and later policy changes.

#### Code, conversations, and semantic deduplication

Code requires repository-aware and structure-aware handling. Exact file hashes miss renamed variables; token or abstract-syntax signatures can catch copied functions. But common licenses, generated clients, test fixtures, and API boilerplate should not bridge unrelated repositories into one cluster. Split by repository or repository family before file-level assignment.

Conversation deduplication should distinguish identical user prompts, identical assistant answers, and identical full trajectories. Repeated prompts with independently diverse valid answers may be useful; repeated template refusals may dominate behavior.

Embedding similarity can retrieve paraphrases, translations, and generated variants, but it also merges distinct facts within one topic. Use it as candidate generation with domain-specific adjudication, not a universal delete rule. Record the embedding model and threshold because a model change changes cluster membership.

#### Split before leakage

All related items must land in the same split. A safe order is:

1. construct source and duplicate groups;
2. add repository, conversation, author, temporal, or generation-family constraints;
3. assign groups, not individual documents, to train, validation, or evaluation;
4. freeze evaluation IDs;
5. run a final contamination scan from every training descendant back to evaluation groups.

Splitting individual rows and deduplicating afterward can keep one duplicate in train and one in evaluation, producing an optimistic metric.

#### Reference near-duplicate flow

Example status: Illustrative Python excerpt; not standalone.

```python
def near_duplicate_candidates(record, lsh_index, config):
    shingles = make_shingles(record, config.shingle_width)
    if len(shingles) < config.minimum_shingles:
        return route_short_record(record)

    signature = minhash(
        shingles,
        seeds=config.hash_seeds,
        hash_width=config.hash_width,
    )
    candidates = set()
    for band_id, band_key in make_bands(signature, config.rows_per_band):
        candidates.update(lsh_index.lookup(band_id, band_key))

    verified = []
    for candidate in candidates:
        score = exact_jaccard(shingles, load_shingles(candidate))
        if score >= config.threshold_for(record.domain):
            verified.append((candidate, score))
    return signature, verified
```

The code makes the two error layers visible: LSH can miss a pair, and the verification threshold can misclassify a retrieved pair. Measure both recall and precision on representative labeled pairs.

### Quality scoring

Quality means expected value for a declared training objective, not resemblance to one prestigious writing style. A document can be fluent but false, authoritative but irrelevant, noisy but uniquely useful, or safe for pretraining but unsuitable for supervised imitation.

#### Keep a score vector

Useful dimensions include:

| Dimension | Example evidence | Common proxy failure |
| --- | --- | --- |
| Integrity | Parse success, low corruption, coherent boundaries | Penalizes poetry, logs, or unconventional layout |
| Factual support | Citations, retrieval agreement, trusted source | Rewards citation-shaped hallucinations |
| Instructional value | Explanations, worked steps, clear examples | Confuses verbosity and formal tone with teaching |
| Originality | Low duplicate/boilerplate score | Penalizes legitimate standards and shared facts |
| Domain utility | Executable code, symbolic validity, expert rubric | Overfits to one toolchain or evaluator |
| Safety and privacy | Policy classifiers, PII detection, red-team rules | Creates false confidence outside labeled risks |
| Representation | Language, dialect, source, topic, difficulty coverage | Treats demographic inference as ground truth |

Store raw features and component scores alongside any aggregate. A scalar:

:::equation S(x) = Σ_{j} λ_{j} z_{j}(x)|A composite score is interpretable only when its components and weights remain visible.

is convenient for ranking but hides compensation: a very high fluency score could offset a privacy warning unless policy gates are separated. Policy-disallowed records should not become eligible merely because their weighted sum is high.

#### Define labels by downstream use

Scorer labels should answer a concrete question, such as:

- "Would an expert include this document in a general pretraining mixture?"
- "Does this solution provide a correct and teachable code trajectory?"
- "Is this conversation an exemplary response for supervised imitation?"
- "Will this record add novel capability after existing mixture coverage?"

These are different targets. Write a rubric with positive, negative, and borderline examples. Measure annotator disagreement; ambiguity in the label is a property of the task, not just noise to average away. Adjudicate policy labels separately from utility preferences.

Split scorer training and validation by source, time, and duplicate cluster. A random row split lets the scorer memorize site templates and produces misleading accuracy.

#### Calibration and threshold metrics

Suppose a binary scorer emits `p_i`, its estimated probability that example `i` satisfies a defined usefulness rubric, with label `y_i in {0,1}`. The Brier score is:

:::equation Brier = (1/N) Σ_{i} (p_{i} - y_{i})^{2}|The Brier score measures probability calibration and discrimination together.

Calibration means that among records assigned score near `0.8`, roughly 80 percent satisfy the rubric. Check reliability curves and Brier score by domain, language, source type, and length. A globally calibrated score can be badly miscalibrated on a small domain.

For a hard threshold `tau`, report:

- retention rate;
- precision among retained examples;
- false-rejection rate among useful examples;
- token mass, not only document count;
- per-slice versions of all three;
- score drift on new crawl snapshots.

ROC area alone is insufficient when the operating point retains 10 percent of a billion-document pool. Precision and slice-specific false rejection at the actual threshold determine what trains.

#### Filter, weight, or route

Use three actions deliberately:

- **filter** for prohibited use, severe corruption, confirmed leakage, or extremely low expected value;
- **weight** when utility is continuous and uncertainty should influence exposure gradually;
- **route** when data is valuable but needs a specialized parser, mixture, objective, or reviewer.

If a score becomes a sampling weight, bound it:

`w_i = clip(g(S_i), w_min, w_max)`.

Without a cap, a small score error can cause one style or source to repeat for many epochs. Record both raw count and effective token weight.

:::callout decision|Filter, weight, or route
Hard-filter examples only for policy, corruption, or very low expected value. Use weights when utility is continuous. Route specialized data into domain mixtures when the data is valuable but distributionally distinct.
:::

#### Evaluate a filter causally

A shorter filtered dataset can appear better for the wrong reason:

- it may receive more epochs under a fixed step budget;
- it may change token length and therefore optimization dynamics;
- it may remove evaluation-like material while improving genuine generalization, or the reverse;
- it may change language and domain mixture;
- its scorer may have been trained on the same benchmark family used for evaluation.

Run controlled pilots:

1. **fixed training tokens and compute:** compare candidate and baseline mixtures at equal optimizer steps and sequence length;
2. **fixed unique-data exposure:** compare whether gains come from repeating a small retained set;
3. **matched mixture:** hold domain/language mass constant while changing the quality rule;
4. **filter complement:** train or evaluate on removed data to learn what signal was lost;
5. **threshold sweep:** estimate utility versus retention rather than testing one arbitrary cutoff;
6. **fresh holdouts:** use evaluations not seen by scorer training or threshold selection.

Track validation loss by source slice, downstream capability, calibration, safety, memorization, and rare-domain regressions. A filter is justified by the model trained from its output, not by aesthetically pleasing retained samples.

As a concrete confound, suppose a filter reduces 100 million documents to 30 million. If both runs train on 100 million document draws, the filtered run sees about 3.3 nominal passes through its retained pool while the baseline sees one. Any gain may come from repeated exposure rather than quality. Report unique documents and effective repeats.

#### Diagnose representation bias

If a scorer rejects dialectal text:

1. freeze the scorer and preserve affected records;
2. quantify rejection by dialect, source, topic, length, code-switching, and annotation confidence;
3. manually audit matched pairs that differ in dialect but not utility;
4. inspect labels and annotator guidance for prestige-language bias;
5. test whether fluency, perplexity, or source-domain proxies dominate the score;
6. relabel with qualified annotators and explicit dialect coverage;
7. recalibrate or use slice-specific routing rather than hiding the issue with one threshold;
8. retrain small controlled models and check both target utility and cross-dialect regressions.

The goal is not demographic parity for every data property. It is to prevent an irrelevant proxy from suppressing valuable signal and to make intentional distribution choices visible.

#### Adversarial and operational testing

Quality scorers can be gamed by keyword stuffing, citation-shaped text, formal formatting, or generated prose tuned to the scorer. Inspect the highest-scoring new records, not only random samples. Maintain challenge sets for spam, duplicated templates, fabricated citations, and model-generated style imitation.

Version scorer weights, feature code, calibration data, and thresholds independently. A scorer update should create a new feature column and release candidate, not rewrite historical scores in place.

### Contamination and benchmark leakage

Contamination is information from an evaluation target entering model development in a way that makes the reported metric overstate the intended generalization. It is broader than copying an exact test string.

#### Contamination taxonomy

| Path | Example | Why exact matching misses it |
| --- | --- | --- |
| Direct | Test question and answer in pretraining | Formatting or tokenization changed |
| Partial | Answer explanation without the prompt | Only a small substring overlaps |
| Derivative | Tutorial solves the same benchmark task | Surface wording differs |
| Paraphrase or translation | Synthetic rewrite of a test item | Semantic identity, low lexical overlap |
| Code-family | Same repository or solution with renamed symbols | Function text is modified |
| Generator leakage | Teacher saw the benchmark, then generates a variant | Benchmark text may never appear |
| Process leakage | Evaluation data influences filter, prompt, or checkpoint selection | No training row contains the item |
| Interactive leakage | Repeated evaluation feedback drives model changes | Human optimization targets the test |

A public benchmark can remain useful for iteration, but then it is a development set. Final claims require protected or freshly constructed evaluations.

#### Detection layers

Freeze a canonical evaluation registry before training. For each evaluation item, retain raw form, canonical forms, n-grams, structural features, source family, creation time, and access policy.

Use multiple scans:

1. exact raw and normalized hashes;
2. substring and n-gram overlap;
3. MinHash or suffix-based retrieval for partial overlap;
4. code AST, repository, or test-signature matching;
5. embedding retrieval for paraphrase candidates;
6. model- or human-assisted adjudication of high-risk candidates;
7. lineage checks for shared generators, sources, and annotators.

For evaluation n-gram set `G(E)` and training document n-grams `G(D)`, asymmetric containment is:

`C(E in D) = |G(E) intersect G(D)| / |G(E)|`.

This is more appropriate than Jaccard when a short question appears in a long document. Thresholds must depend on item length and n-gram width. One matching 8-gram is strong evidence for a ten-word item and weak evidence for a long standards document.

Embedding similarity increases recall but can flag legitimate shared concepts. Treat it as retrieval, then adjudicate using answer identity, uncommon details, source history, and temporal evidence. A high semantic score alone does not prove leakage.

#### Quantify risk without hiding uncertainty

Report at least:

:::equation ContaminationRate = N_{flagged} / N_{evaluation}|Report the affected evaluation mass together with detection thresholds and uncertainty.

and:

`mass_contamination_rate = sum_i weight_i * flagged_i / sum_i weight_i`.

The second can weight items by token count, benchmark importance, or another declared measure. Also report severity:

- exact prompt and answer;
- exact prompt only;
- substantial n-gram containment;
- likely paraphrase or derivative;
- shared source family;
- uncertain candidate.

Publish clean-slice and full-set scores when possible, but do not assume removing flagged items fixes selection bias. If only the hardest or most popular items leaked, the clean remainder is a different benchmark.

#### Prevention is stronger than post hoc scanning

Protect the evaluation supply chain:

- separate storage accounts, credentials, and compute roles for private evaluation;
- prevent evaluation objects from entering crawl, labeling, generation, and retrieval indexes;
- log every access and join against evaluation IDs;
- prohibit benchmark text in scorer prompts and synthetic seeds;
- use canary IDs or content markers where appropriate;
- rotate private tests and maintain one-time final evaluations;
- assign duplicate, repository, author, and generation families to one split before row-level sampling.

For code, split at repository or repository-family level and use hidden tests. For temporal knowledge, use time-split sources whose facts did not exist at the training cutoff. For mathematical or reasoning tasks, generate parameterized variants with independent solvers and keep final generators private.

#### Remediation and interpretation

When contamination is found:

1. quarantine matching training records and descendants;
2. identify releases, checkpoints, scorers, and generation jobs exposed through lineage;
3. rerun clean-slice and fresh evaluations;
4. decide whether a new training release or checkpoint is necessary;
5. document detection limits and residual risk;
6. rotate the compromised evaluation if it influenced repeated decisions.

Overlap does not prove the model memorized an item, and absence of detected overlap does not prove independence. Use extraction tests, perturbations, fresh variants, and behavior analysis to distinguish memorization from robust capability. Report evidence and uncertainty rather than a binary "decontaminated" label.

## Synthetic Data and Mixture Design

LEAD: Synthetic data is useful when generation, verification, and sampling create training signal that the available corpus cannot provide. Its value is determined by verified novelty and distributional effect, not by generation volume.

### Synthetic generation pipeline

Synthetic data is a programmable acquisition process, not a free source of truth:

`seed -> generate -> verify -> score -> deduplicate -> balance -> audit -> train`

Every arrow needs a versioned contract. Synthetic data is most valuable when it creates verified signal that is expensive or rare in natural data: new exercises with executable tests, counterexamples, controlled difficulty, tool-use traces, multilingual variants, or targeted failures from serving.

#### Define the target before prompting

A generation specification should state:

- target capability and failure mode;
- input and output schema;
- allowed knowledge sources and temporal cutoff;
- difficulty, language, domain, and format coverage;
- correctness oracle or adjudication process;
- novelty relation against training and evaluation data;
- privacy, safety, and rights constraints;
- generator, verifier, and student independence requirements;
- maximum generation and verification cost per accepted example.

Without this contract, prompt iteration optimizes what looks plausible to the team rather than what produces a useful gradient.

#### Coverage begins with seeds

Seed selection controls the reachable distribution. If prompts are sampled only from existing easy examples, high-temperature decoding produces stylistic variety around an easy core, not new capability.

Construct a coverage matrix across dimensions such as:

- domain and subdomain;
- skill and reasoning operator;
- difficulty and required trajectory length;
- input shape and edge case;
- language, locale, and notation;
- tool or API;
- expected failure mode;
- verifier type and confidence.

Use under-covered cells to drive generation. Keep seed IDs and generator-family IDs so later splitting and deduplication do not place siblings across train and evaluation.

Generation can use decomposition:

1. propose a task specification;
2. validate that the specification is well-posed;
3. generate one or more solutions;
4. derive tests or invariants independently;
5. execute or adjudicate;
6. generate critiques and repairs only when repair lineage is retained.

The task generator and answer generator need not be the same model. Separation can reduce self-confirmation, although shared pretraining still creates correlated errors.

#### Pipeline yield and cost

Let stage `s` retain fraction `y_s` of its input. Starting from `N_0` candidates:

:::equation N_{accepted} = N_{0} Π_{s} y_{s}|End-to-end yield is the product of stage retention rates.

To obtain `M` accepted examples, expected initial generation volume is:

:::equation N_{0} = M / Π_{s} y_{s}|Required generation volume grows rapidly when several filters have modest yield.

If task validation retains `0.8`, solution verification `0.6`, deduplication `0.75`, and final policy review `0.9`, total yield is:

`0.8 * 0.6 * 0.75 * 0.9 = 0.324`.

One million accepted examples require about `3.09` million initial candidates. A budget that prices only accepted generations understates model calls, verifier compute, storage, and failed-sample analysis.

With per-item cost `c_s` applied to the number entering stage `s`, expected total cost is:

`Cost = sum_s c_s N_0 product_{j < s} y_j`.

Place cheap, high-recall gates early and expensive, high-precision verification later. An early gate with poor recall can permanently erase rare valuable cases.

#### Verifier error from Bayes' rule

A verifier's pass rate is not its precision. Let:

- `p = P(good)` before verification;
- `alpha = P(accept | good)` be true-accept rate;
- `beta = P(accept | bad)` be false-accept rate.

Then:

:::equation P(bad;accept) = β(1-p) / (αp + β(1-p))|Bayes' rule converts verifier error and generator quality into accepted-data risk.

If 60 percent of candidates are good, `alpha=0.95`, and `beta=0.10`, then:

`P(bad | accept) = 0.10*0.40 / (0.95*0.60 + 0.10*0.40) ~= 0.0656`.

About 6.6 percent of accepted examples are still bad. At billion-example scale that is not a rounding error. Improve the generator prior `p`, reduce verifier false acceptance `beta`, add independent checks, or lower the weight of uncertain examples.

Running two verifiers does not multiply error rates unless their failures are conditionally independent. Two models trained on the same data and prompted with the same rubric may confidently accept the same false solution. Measure joint errors on a human- or execution-adjudicated set.

#### Verification by data type

| Data type | Stronger verification | Residual risk |
| --- | --- | --- |
| Code | Compile, type-check, sandbox, hidden tests, property tests, complexity bounds | Weak tests, nondeterminism, unsafe behavior, test leakage |
| Mathematics | Symbolic equivalence, independent solver, randomized substitution, proof checker | Domain restrictions, solver bugs, invalid proof steps |
| Retrieval-grounded QA | Source allowlist, claim-evidence alignment, citation entailment | Source error, stale evidence, unsupported synthesis |
| Tool use | Execute in sandbox, validate state transition and side effects | Simulator differs from production, hidden permissions |
| Open-ended reasoning | Rubric decomposition, multiple critics, human adjudication | Correlated preferences, persuasive but false reasoning |
| Safety behavior | Adversarial prompts, policy rule checks, expert review | Coverage gaps, evaluator over-refusal bias |

Teacher confidence is not independent verification. A teacher can be confidently wrong, can recognize its own preferred style, and can leak benchmarks into both task and answer.

#### Rejection changes the distribution

Accepted data follows:

`P(x | accept) proportional to P(accept | x) P_generated(x)`.

If a verifier is better at short algebra than geometry proofs, filtering shifts the accepted mixture toward short algebra even when the generator proposed balanced tasks. Track yield by every coverage cell and inspect rejected examples.

Verifier gaming is the synthetic analogue of reward hacking. A generator can learn to emit solutions that exploit weak tests, include scorer-friendly phrases, or avoid hard cases. Keep hidden tests, rotate verifier families, audit unusually high-yield generators, and evaluate accepted examples with checks that were not used during generation.

#### Diversity and novelty

Measure more than surface lexical diversity:

- unique task specifications and solution strategies;
- semantic and structural cluster sizes;
- coverage-cell entropy;
- rare skill and failure-mode recall;
- distance to seed and teacher examples;
- execution trace or proof-shape diversity;
- student gradient or loss diversity in small pilots.

Deduplicate after generation, but keep controlled variants when they isolate a meaningful factor. Ten paraphrases of one arithmetic problem are not ten reasoning skills. Conversely, two solutions with the same answer but different valid algorithms may be valuable.

#### Synthetic lineage and leakage

Each record should include:

- seed and source IDs;
- generator checkpoint, prompt template, decoding parameters, and random seed;
- retrieved context and tool versions;
- candidate family and repair chain;
- every verifier result and confidence;
- human adjudication;
- duplicate cluster and evaluation-overlap results;
- final sample weight and release membership.

If the teacher or retrieval index may have seen a benchmark, synthetic paraphrases can carry contamination without literal copying. Scan task semantics, generator lineage, and retrieved evidence against the evaluation registry.

#### Recursion and human-data anchors

Repeatedly training generators on generated descendants can narrow tails and amplify model errors. Preserve human-authored or otherwise independently grounded data, label generation depth, and cap recursive descendants in each mixture. Monitor lexical, semantic, factual, and difficulty diversity across generations.

Synthetic data should receive lower weight when:

- verification is weak or correlated with the generator;
- it mostly repeats teacher style or facts;
- coverage differs from product traffic;
- recursive generation depth is high;
- a small set of templates creates high effective repetition;
- human data provides contradictory or more diverse signal;
- pilot training shows gains on verifier-shaped metrics but regressions on fresh independent tests.

It may deserve equal or higher weight when correctness is mechanically verified, novelty is established, the target capability is underrepresented, and controlled pilots show improvement without distributional regressions.

:::callout pitfall|The verifier defines the surviving world
Generation proposes possibilities; verification determines which possibilities become training reality. Audit verifier recall on rare valid behavior as carefully as verifier precision on bad behavior.
:::

### Distribution balancing

Balancing controls the distribution of gradient updates after eligibility, deduplication, and scoring. Raw source size is not a neutral default: large scraped domains then dominate by collection convenience. Uniform domains are not neutral either: tiny sources repeat and can overfit.

#### Domain mixture

Let domain `d` contain `n_d` eligible tokens. A common temperature mixture is:

:::equation q_{d} = n_{d}^{τ} / Σ_{j} n_{j}^{τ}, 0 ≤ τ ≤ 1|Temperature sampling interpolates between proportional and uniform domain sampling.

- `tau=1` samples in proportion to available tokens;
- `tau=0` samples domains uniformly;
- intermediate values upsample smaller domains smoothly.

For raw domain shares `90%`, `9%`, and `1%`, `tau=0.5` uses square roots:

`sqrt(90) : sqrt(9) : sqrt(1) = 9.49 : 3 : 1`,

which normalizes to approximately `70.3%`, `22.2%`, and `7.4%`. The rarest domain is upsampled by about 7.4 times relative to its raw share, so unique-example reuse and memorization risk must be measured.

Sampling by documents rather than tokens changes the objective when lengths differ. State whether `q_d` governs documents, examples, tokens, or packed sequences.

#### Importance weights and target distributions

If product traffic has target distribution `p_d` but training samples domains from `q_d`, an unbiased estimate of product-distribution loss can use:

:::equation w_{d} = p_{d} / q_{d}|Importance weighting corrects a sampling distribution toward a declared target distribution.

For a sampled loss `ell(x)`:

`E_{d~q}[w_d ell_d] = sum_d p_d E[ell_d]`.

Large weights create high-variance gradients. In practice, cap or smooth weights and accept controlled bias. Report the effective sample size:

:::equation ESS = (Σ_{i} w_{i})^{2} / Σ_{i} w_{i}^{2}|Effective sample size reveals concentration hidden by the nominal example count.

If 1,000 examples have weight one, `ESS=1,000`. If a few examples carry most weight, ESS can be far smaller than the row count. Token count without ESS hides concentration.

#### Constrained capability objective

Product frequency should not be the sole objective when rare failures are costly. One formulation is:

:::equation minimize_{q} Σ_{d} p_{d} L_{d}(q)|Choose mixture q to reduce target-weighted domain loss subject to data and risk constraints.

subject to:

`Metric_c(q) >= target_c` for every critical slice `c`,

`Regression_j(q) <= budget_j`,

`q_d >= q_min,d`,

and compute, data-reuse, safety, and rights constraints.

This expresses a common decision: optimize average product utility while requiring minimum safety, language, or reliability performance. Constraint thresholds come from product risk, not from whichever dataset happens to be largest.

#### Adaptive mixture selection

Static heuristics are a baseline. A principled control loop:

1. define domains and held-out losses that are stable enough to guide decisions;
2. train small proxy models on candidate mixtures;
3. estimate which domains are underfit relative to a reference or target;
4. update mixture weights subject to floors and reuse caps;
5. confirm on fresh runs and larger scale;
6. stop adapting before the same holdouts become over-optimized.

Methods such as group distributionally robust reweighting can emphasize domains with excess loss, but proxy-to-large-model transfer is an empirical assumption. Domains must be meaningful: if one "web" bucket mixes every style and language, its aggregate loss gives weak guidance.

#### Multiple balancing axes

One domain label is rarely enough. Balance or audit:

- source and license class;
- language, script, locale, and dialect;
- topic and task;
- difficulty and context length;
- human versus synthetic origin and generation depth;
- quality-score bands;
- code language and repository family;
- safety risk and rare critical failure;
- time and freshness.

The full Cartesian product is sparse. Use hierarchical quotas, constrained sampling, or iterative proportional fitting rather than requiring every combination. Monitor which constraints conflict.

#### Repetition and curriculum

Track expected exposure:

:::equation ExpectedEpochs_{d} = SampledTokens_{d} / UniqueTokens_{d}|Expected reuse exposes memorization risk hidden by aggregate token counts.

Two domains with equal sampled tokens can have very different reuse. Set caps for exact examples, duplicate clusters, sources, and synthetic families. Shuffle at the group level so repeated siblings do not appear back-to-back.

Curriculum changes mixture over training step `t`, so write `q_d(t)`. Introduce long contexts or difficult verified reasoning when the model can use them, but preserve replay of earlier domains to avoid forgetting. A curriculum needs transition gates and regression monitoring, not only a schedule.

#### Worked balancing decision

Suppose product traffic is 95 percent general assistance, 4 percent code, and 1 percent security-sensitive support. Raw eligible data is 98 percent general, 1.9 percent code, and 0.1 percent security.

A reasonable pilot might:

- keep the majority of mass near general traffic;
- upsample code while monitoring repository repetition;
- allocate a larger-than-traffic security floor because failure cost is high;
- use independent safety evaluations as constraints;
- compare fixed-token pilots at several floors;
- record both nominal proportions and effective weights.

Calling this mixture "representative" would be inaccurate. It is deliberately risk-adjusted, and the rationale should be part of the release manifest.

### Language-aware curation and a fixed-budget ablation

A quality threshold learned on English web text can reject useful text in another writing system. Token fertility, punctuation, document length, and repeated boilerplate have language-dependent distributions. Language identification should therefore route to an appropriate quality policy, with an explicit mixed-language path, rather than serve only as a final reporting label. [FineWeb2](https://arxiv.org/abs/2506.20920) studies language-adapted filtering and deduplication, and combines duplication counts with quality when rebalancing the resulting corpus. Its lesson is to evaluate the curation pipeline through trained models across languages, not just retained bytes.

Consider an original experiment with a 10-billion-token budget. Recipe A spends 8 billion tokens on English and 2 billion on other languages. Recipe B uses 6 and 4 billion. Train both with the same architecture, optimizer, total tokens, and evaluation checkpoints. Report English regression as well as gains by language; an average weighted by the new training mixture would move the goalposts. If B's tokenizer produces twice as many tokens per document for a target language, doubling tokens may not double semantic coverage. Track documents, bytes, unique content, and tokens separately.

To isolate filtering from mixture changes, first hold per-language sampling proportions fixed while varying the filter. Then hold the chosen filter fixed while varying the mixture. Preserve an untouched evaluation set and check contamination before either experiment. A synthetic-data generator must use the training pool, not read held-out questions and paraphrase them into the corpus. The acceptance rule is useful generalization per training budget, not agreement with the quality classifier that produced the data.

## Training Data Platform and Release Engineering

LEAD: A data pipeline becomes a training platform when it can reproduce releases, absorb partial failure, enforce policy, explain examples, and connect every selected sequence to the models and evaluations that consumed it.

### Data pipeline system design

At large scale, the architecture should make the correct path reproducible and the incorrect path visible. Separate immutable payloads, metadata and lineage, distributed transforms, indexes, scores, release manifests, and policy decisions.

#### Logical architecture

##### 1. Acquisition gateway
- authenticates sources and records source contracts;
- assigns source-revision and raw-content IDs;
- applies size, media, malware, and decompression limits;
- writes immutable raw objects and acquisition envelopes.

##### 2. Metadata and lineage catalog
- stores source, rights, policy, schema, transform, and parent-child edges;
- supports forward and reverse impact queries;
- separates factual metadata from mutable review decisions.

##### 3. Distributed transformation plane
- parses, normalizes, segments, detects language, and extracts structure;
- uses idempotency key `(input_id, transform_version, config_hash)`;
- writes new immutable outputs rather than overwriting inputs.

##### 4. Index and feature plane
- exact-hash tables, MinHash/LSH indexes, evaluation fingerprints, embeddings, and model-based scores;
- versions every model, seed, tokenizer, and threshold;
- keeps expensive features reusable across release candidates.

##### 5. Selection and mixture plane
- applies policy eligibility, duplicate representative rules, quality weights, quotas, and deterministic sampling;
- produces a release manifest without copying payload bytes unnecessarily.

##### 6. Tokenization and packing plane
- pins tokenizer and packing policy;
- records example-to-sequence spans and loss masks;
- prevents protected boundaries or access classes from merging.

##### 7. Release registry
- stores manifest digest, shard digests, row and token statistics, lineage root, approvals, validation results, and supersession history;
- makes aliases such as `production-current` pointers to immutable versions, never mutable datasets.

##### 8. Consumption and feedback
- training jobs resolve a manifest digest and emit consumed-shard receipts;
- evaluations, incidents, and serving failures create new metadata or candidate records rather than modifying the old release.

#### Immutable layers and materialization

A useful organization is:

- **raw:** source-faithful bytes and acquisition metadata;
- **parsed:** structured documents with recoverable boundaries;
- **eligible:** policy-approved, normalized records plus features;
- **candidate release:** selected IDs, scores, weights, and split assignments;
- **published release:** frozen manifest, tokenization, and validation report.

Not every layer needs a full physical copy. Content-addressed objects and manifests can share unchanged payloads. But logical immutability matters: rerunning a parser must create a new version even if most output bytes match.

#### Idempotence and partial reruns

For transform `F_v`:

`output_id = H(input_id || F_v || config_hash || deterministic_seed)`.

A retry first checks whether that output already exists and verifies its digest. Nondeterministic generation includes generator checkpoint, decoding configuration, and per-record seed in identity.

Partition work by stable input IDs rather than worker count so changing cluster size does not reshuffle identity. Store per-partition completion manifests. A failed 10,000-partition job should rerun failed partitions, not rewrite successful output.

Idempotence does not imply semantic correctness. A broken normalizer can reproducibly corrupt every record; validation and version rollback are still required.

#### Capacity and yield example

Suppose acquisition begins with 100 million documents averaging 2 KiB of normalized text. Normalized payload alone is about 190.7 GiB. A 128-by-64-bit MinHash adds about 95.4 GiB before indexing.

Assume stage yields:

- parse success: `0.95`;
- policy eligibility: `0.98`;
- quality selection: `0.60`;
- exact-dedup representatives: `0.90`;
- near-dedup representatives: `0.80`.

Expected retained documents:

`100M * 0.95 * 0.98 * 0.60 * 0.90 * 0.80 ~= 40.2M`.

This estimate sizes downstream scoring and tokenization. It is not a quality target: surprising yield changes should trigger investigation, not threshold adjustment to force the forecast.

If parsing sustains 100,000 documents per second, ideal compute time is 1,000 seconds, but end-to-end time also includes object reads, skew, shuffle, index writes, retries, and stragglers. Quote both compute throughput and wall-clock critical path.

#### Shuffles, skew, and backpressure

Exact deduplication groups by content hash; LSH groups by band key; balancing groups by domains. Each creates a shuffle. Hot keys from empty documents, boilerplate, or one enormous source can overload reducers.

Mitigations include:

- early quarantine of empty and low-information records;
- key salting with a second merge stage for hot buckets;
- size-aware partitioning;
- capped candidate postings and a separately audited overflow path;
- spillable external sorting;
- backpressure into acquisition rather than dropping records silently.

Dropping failed or slow partitions biases the dataset toward easy-to-parse sources. Completeness metrics must include missing source partitions and retry exhaustion.

#### Release validation

Every release candidate should pass gates:

| Gate | Required evidence |
| --- | --- |
| Integrity | Shard digests, schema validation, no missing partitions, deterministic sample replay |
| Rights and policy | Eligible source coverage, decision versions, unresolved-policy count |
| Composition | Documents, unique tokens, domains, languages, sources, time, lengths, and weights |
| Deduplication | Exact and near-duplicate rates, candidate recall audit, cluster-size tails |
| Quality | Score distributions, threshold precision, slice false-rejection audit |
| Leakage | Evaluation overlap by severity, source-family checks, private-registry scan |
| Safety and privacy | Classifier and sampled-review results, PII or secret handling |
| Training behavior | Small-pilot loss, capabilities, memorization, safety, and regressions |
| Reproducibility | Manifest reconstruction, transform versions, seeds, tokenizer, and packing |

Compare every metric to the previous release with explicit tolerances. A row-count drop of 20 percent, disappearance of a language, or sudden rise in average quality score can all indicate a pipeline bug.

#### Deletion, incident response, and model impact

Deletion is a state transition, not an object-store command:

1. receive and authenticate a source or subject request;
2. resolve matching raw and logical identities;
3. mark future-use eligibility false;
4. traverse descendants and release memberships;
5. invalidate indexes, caches, pending jobs, and derived manifests;
6. publish replacement releases;
7. identify affected checkpoints and downstream artifacts;
8. apply the model governance policy and record completion.

Removing data from future releases does not remove its influence from an existing model. Keep the model impact record separate: retraining, machine unlearning, access restriction, or documented residual exposure may be required by the applicable policy.

For a quality or leakage incident, preserve the faulty release for forensics, revoke its production alias, and roll consumers back to the last approved immutable manifest. Overwriting the bad version destroys evidence.

#### Observability

Monitor:

- documents, bytes, and tokens entering and leaving every stage;
- yield and error code by source, language, parser, and worker version;
- duplicate cluster sizes and hot LSH buckets;
- quality-score drift and slice retention;
- contamination matches and evaluation-registry access;
- synthetic generation yield, verifier disagreement, and cost;
- mixture weights, expected epochs, and effective sample size;
- lineage edge completeness and deletion service-level objective;
- release-to-checkpoint consumption.

Data bugs are often silent because a training job still receives valid tensors. Alert on distribution and lineage invariants, not only job failure.

### End-to-end design: speculative-decoding data

Consider generating training data for a draft model used in speculative decoding. The draft should approximate the target model on the serving prompt distribution while being much cheaper to execute.

#### 1. Contract

Each example contains:

- production-like prompt prefix and optional conversation state;
- target-model token distribution or sampled continuation under a pinned tokenizer;
- decoding configuration and target checkpoint;
- draft context limit and serving policy;
- provenance, privacy classification, and retention limit.

The goal is not to imitate arbitrary teacher prose. It is to improve target-token agreement and downstream acceptance under the exact speculative algorithm.

#### 2. Acquisition and privacy

Sample consented or otherwise permitted prompt traffic with stable event IDs and bounded retention. Remove secrets and disallowed content through explicit policy gates. Add curated rare shapes: long prefixes, code, multilingual text, tool syntax, and safety-sensitive prompts.

Separate train, calibration, and final evaluation by user or conversation family and time. Do not let online acceptance metrics from the final holdout drive prompt generation repeatedly.

#### 3. Generation

Run the pinned target model and store logits only when the training objective requires them; full-vocabulary logits are expensive. Alternatives include top-`m` logits plus residual mass, sampled target tokens, or online teacher inference. Record temperature and truncation because they change the target distribution.

Generate counterfactual prompt variants only when privacy and semantic-validity checks pass. Synthetic prompts receive generator-family IDs and separate mixture accounting.

#### 4. Verification

Verify tokenizer identity, finite logits, probability mass, sequence alignment, stop-token handling, and reproducibility. Replay a sample through the target checkpoint. Deduplicate prompt families and scan against private evaluations.

The strongest downstream verifier is an offline speculative-decoding simulation that measures:

- target acceptance rate by proposed position;
- accepted tokens per target call;
- draft and target latency;
- distributional correctness of the final sampler;
- slice regressions and worst-case tails.

#### 5. Balance and publish

Weight examples toward the serving prompt distribution, with floors for rare high-cost failures and caps on user, template, source, and synthetic family. Report effective epochs and ESS. Publish a manifest binding prompts, target outputs, tokenizer, objective, weights, and verification results.

#### 6. Feedback loop

After deployment, join acceptance and latency telemetry back to coarse, privacy-safe slices. Add failure examples to the next candidate release, not directly to the training set. Require a fresh holdout gate before replacing the draft.

### Production readiness checklist

Before publishing a training dataset, verify:

- **contract:** intended use, unit, tokenizer, rights, policy, retention, and owners;
- **identity:** source revision, raw and normalized hashes, duplicate and family IDs;
- **transforms:** pinned code, models, configs, random seeds, and resource limits;
- **lineage:** forward and reverse impact queries across synthetic and packed descendants;
- **quality:** label rubric, calibration, threshold operating point, bias and adversarial audits;
- **leakage:** frozen evaluation registry, multi-layer scan, split families, and residual risk;
- **synthetic:** generator/verifier independence, false-accept estimate, yield, novelty, and recursion depth;
- **mixture:** token-level weights, constraints, reuse caps, expected epochs, and ESS;
- **release:** immutable manifest, digests, validation report, rollback pointer, and approvals;
- **operations:** partial reruns, backpressure, deletion, incident response, cost, and monitoring.

## Applied Data-System Casework

LEAD: The following cases integrate identity, normalization, deduplication, quality, contamination, synthetic generation, mixtures, and release engineering. The worked solutions demonstrate a decision process rather than prescribe one universal pipeline.

### Design Exercises

1. Design source identity and provenance so one deletion request can find raw, normalized, synthetic, packed, release, and checkpoint descendants.
2. Specify a normalization contract for web prose, code, mathematics, and mixed-language documents. Which invariants make it reproducible?
3. Derive MinHash and the LSH banding probability. Size a near-duplicate system and explain clustering and split hazards.
4. How do you prove a new quality filter helps rather than merely shrinking and repeating the dataset? Include calibration and slice bias.
5. Explain contamination beyond exact strings and design both detection and prevention for public and private evaluations.
6. Derive accepted synthetic-data quality from generator prevalence and verifier error. Explain why two model verifiers may not be independent.
7. When should synthetic data receive lower or higher weight than human-authored data? Address novelty, verifier gaming, and recursive generation.
8. Derive temperature sampling, importance weights, expected domain epochs, and effective sample size. Build a risk-constrained mixture.
9. Design a reproducible billion-scale data platform with idempotent reruns, immutable releases, validation gates, rollback, and deletion propagation.
10. Design a data-generation pipeline for speculative decoding and name the offline metrics that establish serving value.

### Worked Solutions

#### 1. Identity, lineage, and deletion

Separate logical identity from content identity. A source revision can use:

`source_revision_id = H(namespace || source_id || revision)`,

while `raw_content_id = H(raw_bytes)` finds byte-identical mirrors and:

`normalized_content_id = H(normalizer_version || normalized_bytes)`

finds exact canonical duplicates. Never discard source memberships when content deduplicates; rights and deletion can differ across identical copies.

Represent the pipeline as a graph of entities, activities, and agents. Raw objects, parsed documents, normalized records, synthetic descendants, shards, manifests, and checkpoints are entities. Each derived entity records parent IDs and exact transform, model, configuration, and seed. Packed sequences need example span lineage; synthetic records need seed, generator, retrieval, repair, and verifier lineage.

For a deletion, authenticate the request, resolve logical source IDs, tombstone them for future use, traverse every descendant edge, invalidate indexes and jobs, create replacement manifests, and enumerate affected checkpoints and evaluations. Keep a separate model-impact decision because deleting future training records does not remove influence from an existing model.

Test both reverse lineage, "what produced this sequence?", and forward impact, "where did this source propagate?" A descriptive dataset card without executable impact queries is documentation, not an operational deletion capability.

#### 2. Structure-preserving normalization

Use distinct decode, parse, canonicalize, clean, segment, and validate stages. Preserve raw bytes and create new immutable derived records. Pin parser, Unicode version, rules, locale, model versions, and configuration.

Web prose can usually use validated decoding, NFC, line-ending normalization, parsed boilerplate removal, and structure-aware segmentation. Code should preserve indentation, newlines, path, repository, and language semantics. Mathematical content should avoid blind NFKC because compatibility folding can erase semantically distinct symbols. Conversations must retain speaker and tool roles. Mixed-language documents need segment-level language scores and conservative routing rather than one hard document label.

Required invariants include:

- idempotence of the canonical view;
- deterministic hashes for fixed input and version;
- bounded and classified character/region removal;
- no merging across protected boundaries;
- raw-to-derived links and offset maps where feasible;
- bounded decompression, nesting, parse time, and output size;
- replay with pinned dependencies.

Validate by domain and language. Inspect diff samples, removed fraction, replacement characters, parse failure, code compilation, table/equation preservation, and deterministic rerun hashes. A global success rate can hide complete corruption of a rare slice.

#### 3. Near-duplicate derivation and system

Shingle each document into set `A`. Jaccard similarity is:

`J(A,B)=|A intersect B|/|A union B|`.

For random permutation `pi`, MinHash stores the minimum permuted shingle. The minima match exactly when the first element of `A union B` lies in the intersection. Thus:

`P(h_pi(A)=h_pi(B))=J(A,B)`.

Across `k` independent hashes, the matching fraction is unbiased with variance `J(1-J)/k`. At `k=128`, maximum standard error is about `0.044`.

For LSH, split the signature into `b` bands of `r` rows. A band matches with probability `s^r`; at least one band matches with:

`1-(1-s^r)^b`.

With `b=16,r=8`, probability is about `0.947` at similarity `0.8` and `0.061` at `0.5`. LSH retrieves candidates; verify them with exact Jaccard, containment, or a domain-specific comparator.

At 100 million documents, 128 64-bit hashes require about `95.4 GiB` before band indexes and candidates. Account for shuffles, hot boilerplate buckets, verification reads, clusters, and replication.

Do not treat threshold similarity as a clean equivalence relation. Connected components can chain unrelated endpoints. Use bounded-diameter or representative-based clusters and deterministic representative ranking. Build source, duplicate, repository, conversation, and generation families before assigning whole groups to splits; row-level splitting first causes leakage.

#### 4. Quality-filter evaluation

First define the rubric and intended use. Store a vector of integrity, factual support, instructional value, originality, domain utility, safety/privacy, and representation signals. Keep policy gates separate from utility scores.

For a binary usefulness probability `p_i`, evaluate calibration with reliability curves and:

`Brier=(1/N) sum_i(p_i-y_i)^2`.

At the operating threshold, report retained precision, useful-example false rejection, document and token retention, and all metrics by language, dialect, source, domain, and length. Split scorer train/validation by source, time, and duplicate family.

Then test the filter through models:

1. train baseline and candidate at fixed tokens, steps, sequence length, and compute;
2. compare at fixed unique-data exposure to reveal repetition effects;
3. match domain and language mixture where possible;
4. inspect the removed complement;
5. sweep thresholds;
6. evaluate on fresh holdouts not used by the scorer.

If 100 million documents become 30 million but both runs draw 100 million documents, the filtered pool receives about 3.3 nominal passes. Gains may reflect repetition. Report unique exposure and effective weights.

For dialect bias, audit matched examples, labels, annotators, features, and calibration. Relabel with qualified coverage, route uncertain slices, and confirm with controlled model pilots. Do not repair one aggregate metric while hiding a slice regression.

#### 5. Contamination detection and prevention

Contamination includes direct prompt/answer copies, partial explanations, paraphrases, translations, code-family variants, teacher leakage, scorer or prompt tuning on evaluation data, repeated benchmark feedback, and shared generators. Exact hashes find only the simplest path.

Freeze an evaluation registry with raw and canonical forms, n-grams, structural signatures, source/generation family, time, and access policy. Scan training descendants using exact hashes, substring or n-gram containment, MinHash, code AST and repository matching, embedding retrieval, and lineage. For short evaluation item `E` inside long document `D`, use:

`C(E in D)=|G(E) intersect G(D)|/|G(E)|`.

Adjudicate semantic candidates because related concepts are not automatically leakage. Report item rate, weighted mass, and severity categories; also give clean-slice and full scores when defensible.

Prevention is stronger: isolate private evaluation storage and credentials, log access, ban evaluation records from retrieval and generation seeds, group related sources before splitting, use hidden tests, maintain time-split and newly authored sets, and rotate one-time finals.

If leakage is found, quarantine descendants, identify exposed releases and decisions, rerun fresh tests, and rotate evaluations that influenced development. Overlap does not prove memorization; no detected overlap does not prove independence. State residual risk.

#### 6. Synthetic verifier mathematics

Let `p=P(good)`, `alpha=P(accept|good)`, and `beta=P(accept|bad)`. Bayes' rule gives:

`P(bad|accept)=beta(1-p)/(alpha p+beta(1-p))`.

For `p=0.60`, `alpha=0.95`, and `beta=0.10`, accepted bad fraction is about `6.6%`. The pass rate alone cannot reveal this. Improve candidate prior, verifier false-accept rate, or add adjudication; weight uncertain data less.

Two verifiers multiply false-accept probabilities only under conditional independence. Models sharing pretraining data, architecture, prompts, or rubrics have correlated blind spots. Estimate their joint confusion on execution- or human-adjudicated samples.

Verification must match the artifact. Code uses sandboxed compilation, hidden and property tests, plus resource bounds. Mathematics uses symbolic or proof checks with domain restrictions. Grounded QA checks claim-evidence support and source validity. Open-ended reasoning needs decomposed rubrics and independent adjudication.

Track yield by coverage cell because filtering changes the distribution:

`P(x|accept) proportional to P(accept|x)P_generated(x)`.

Keep hidden checks and audit unusually high generator yield for verifier gaming.

#### 7. Weighting synthetic data

Lower synthetic weight when verification is weak, correlated with the generator, or optimized on the same metrics used for evaluation; when examples repeat templates or teacher style; when generated coverage misses product tails; when generation depth is recursive; or when fresh independent pilots regress despite gains on verifier-shaped tests.

Raise weight when correctness is mechanically established, tasks are novel relative to existing data, target capabilities are scarce, coverage is controlled, and fixed-token pilots show independent gains without memorization or safety regression.

Preserve seed, generator, retrieval, decoding, repair, verification, duplicate, and contamination lineage. Deduplicate by task and strategy family, not only wording. Measure semantic, structural, strategy, difficulty, and coverage diversity.

Recursive training can narrow tails and amplify model errors. Label generation depth, cap family exposure, preserve independently grounded human data, and monitor diversity and factual accuracy across generations. "Human" is not synonymous with good and "synthetic" with bad; provenance and verification determine confidence. The mixture should reflect evidence.

#### 8. Mixture mathematics

For domain token mass `n_d`, temperature sampling uses:

`q_d=n_d^tau/sum_j n_j^tau`.

`tau=1` follows raw tokens; `tau=0` is uniform domains. For raw shares 90, 9, and 1 with `tau=0.5`, the mixture is about 70.3, 22.2, and 7.4 percent. The smallest domain is reused about 7.4 times relative to raw proportional sampling.

To estimate target product loss under product distribution `p_d` while sampling `q_d`, use importance weight:

`w_d=p_d/q_d`.

This is unbiased in expectation, but large weights raise gradient variance. Clip or smooth with an explicit bias tradeoff. Report:

:::equation ESS = (Σ_{i} w_{i})^{2} / Σ_{i} w_{i}^{2}|Effective sample size reveals how concentrated weighting reduces statistical support.

and:

`expected_epochs_d=sampled_tokens_d/unique_tokens_d`.

Optimize product-weighted utility with minimum performance constraints for safety, language, or rare high-cost slices, plus regression and reuse caps. Balance on tokens or packed sequences explicitly; document sampling gives a different mixture when lengths differ.

Use small proxy experiments or group-robust reweighting to propose mixtures, then confirm at larger scale and on fresh holdouts. Proxy transfer and domain definitions are assumptions, not guarantees.

#### 9. Reproducible data platform

Use immutable raw storage; a metadata and lineage catalog; distributed, idempotent transforms; versioned exact, LSH, embedding, contamination, and score indexes; a selection/mixture service; tokenizer-aware packing with span lineage; and an immutable release registry.

Derive transform output identity from input ID, transform version, config hash, and deterministic seed. Partition by stable input ID, store partition-completion manifests, and retry only missing partitions. Generated records include checkpoint and decoding configuration.

Publish manifests that bind exact record IDs, shard digests, weights, splits, tokenizer, transforms, approvals, and validation results. Release gates cover integrity, rights, composition, deduplication, quality calibration, leakage, privacy/safety, pilot training, and reconstructability.

Capacity planning includes raw and normalized bytes, reusable features, MinHash and band indexes, shuffle copies, candidate verification, synthetic failures, tokenization, manifests, checkpoints, and old-version coexistence. Handle hot keys with quarantine, salting, size-aware partitions, and audited overflow.

Never drop failed partitions silently; it biases toward easy sources. For incidents, preserve the bad release, revoke its alias, roll back consumers, and rebuild a new immutable version. For deletion, traverse descendants and separately record model impact.

#### 10. Speculative-decoding dataset

The contract is target-distribution agreement under serving constraints. Each record binds a permitted production-like prompt, conversation state, pinned target checkpoint, tokenizer, target output representation, decoding configuration, and draft serving limits.

Acquire consented or otherwise eligible traffic and curated rare prompt shapes. Split by user or conversation family and time. Generate target labels as full logits only when necessary; top-`m` logits with residual mass, sampled tokens, or online teacher calls can reduce storage. Record temperature and truncation.

Verify tokenizer and sequence alignment, finite probability mass, stop tokens, target replay, privacy, deduplication, and evaluation overlap. Simulate the exact speculative algorithm offline. Measure acceptance probability by proposed position, accepted tokens per target call, target calls per output token, draft and target latency, final-distribution correctness, and slice tails.

Balance near serving traffic with floors for rare costly failures and caps on users, templates, and synthetic families. Publish a manifest binding prompts, target outputs, weights, objective, and verifier results. After deployment, feed privacy-safe slice failures into the next candidate release and require a fresh holdout before promotion.

### Further Study and Primary References

- [W3C PROV Overview](https://www.w3.org/TR/prov-overview/) - entities, activities, agents, and interoperable provenance.
- [Gebru et al., Datasheets for Datasets](https://www.microsoft.com/en-us/research/uploads/prod/2019/01/1803.09010.pdf) - structured dataset motivation, composition, collection, use, and maintenance documentation.
- [Unicode Standard Annex 15, Normalization Forms](https://unicode.org/reports/tr15/) - canonical and compatibility normalization contracts and caveats.
- [Broder, On the Resemblance and Containment of Documents](https://doi.org/10.1109/SEQUEN.1997.666900) - shingling, resemblance, containment, and MinHash foundations.
- [Lee et al., Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) - deduplication, memorization, and train-validation overlap.
- [Carlini et al., Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) - empirical extraction and memorization risk.
- [Yang et al., Rethinking Benchmark and Contamination for Language Models with Rephrased Samples](https://arxiv.org/abs/2311.04850) - paraphrase and semantic contamination beyond string matching.
- [Wang et al., Self-Instruct](https://arxiv.org/abs/2212.10560) - iterative synthetic instruction generation and filtering.
- [Shumailov et al., The Curse of Recursion](https://arxiv.org/abs/2305.17493) - distributional degradation under recursive generated-data training.
- [Xie et al., DoReMi](https://arxiv.org/abs/2305.10429) - proxy-based domain reweighting for pretraining mixtures.
- [Li et al., DataComp-LM](https://arxiv.org/abs/2406.11794) - controlled experiments for filtering, mixing, and data-quality evaluation.
- [Penedo et al., The FineWeb Datasets](https://arxiv.org/abs/2406.17557) - documented web-scale filtering and deduplication ablations.

## Distillation, KL Divergence, and Model Sizing

LEAD: Distillation is distribution transfer under a capacity and serving constraint. The central question is not "teacher or student?" It is which information should cross the capacity boundary.

:::diagram distillation|The teacher provides a structured target; the student must convert it into utility inside a different architecture and budget.

### Define the transfer contract

Distillation can transfer several different objects: token probabilities, selected sequences, rankings, hidden representations, tool trajectories, or verified outcomes. These targets are not interchangeable. Token probabilities preserve local uncertainty but are expensive to store. Generated sequences are compact but collapse the teacher distribution to sampled paths. Feature targets require compatible internal shapes and introduce a layer-mapping problem.

Before choosing a loss, specify:

- the student serving envelope and capacity limit;
- the teacher version, decoding policy, and known failure regions;
- the prompts and states on which transfer should occur;
- which information is stored, recomputed, or queried online;
- the quality, latency, memory, and cost metrics that define success.

A student should not be optimized to imitate teacher behavior that the product does not want. Verified labels, safety policy, and human adjudication can override the teacher where it is unreliable.

### Cross entropy and KL

Let teacher distribution be `p(y|x)` and student distribution be `q(y|x)`. Cross entropy from teacher to student is:

:::equation H(p,q;x) = - Σ_{y} p(y;x) log q(y;x)|Teacher cross entropy weights student error by probability under the teacher.

It decomposes as `H(p) + KL(p || q)`. The teacher entropy does not depend on student parameters, so minimizing teacher cross entropy is equivalent to minimizing forward KL.

:::equation KL(p,q) = Σ_{y} p(y) log(p(y) / q(y))|Forward KL penalizes missing probability mass where the teacher assigns support.

Forward KL heavily penalizes the student for assigning too little probability where the teacher has mass. It is often described as mode covering. Reverse KL, `KL(q || p)`, weights regions under the student and tends to concentrate on one teacher mode when representing all modes is expensive.

These slogans are helpful but incomplete. Token distributions are conditional, the student is parameterized, the support is finite after softmax, and sequence-level behavior compounds local choices. Always connect the divergence to the actual objective and sampling process.

### Temperature

With temperature `T`, logits `z` become:

:::equation p_{i}(T) = exp(z_{i}/T) / Σ_{j} exp(z_{j}/T)|Temperature reveals or suppresses probability structure among non-argmax tokens.

Higher temperature softens the distribution and exposes relationships among non-argmax tokens. Lower temperature sharpens it. In classic distillation, multiplying the KL term by `T^2` compensates for the gradient scale change introduced by temperature.

Example status: Illustrative Python excerpt; not standalone.

```python
def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.7):
    teacher = softmax(teacher_logits / T, dim=-1)
    student_logp = log_softmax(student_logits / T, dim=-1)
    soft = kl_div(student_logp, teacher, reduction="batchmean") * (T * T)
    hard = cross_entropy(student_logits, labels)
    return alpha * soft + (1.0 - alpha) * hard
```

The production version masks padding, normalizes by valid tokens, handles sharded vocabulary, controls teacher precision, and avoids storing full teacher logits when bandwidth dominates.

### Engineer the teacher data path

Full-vocabulary logits can exceed the storage cost of the original training examples by orders of magnitude. A data path must choose among online teacher inference, offline dense logits, top-k logits plus residual mass, quantized scores, or sampled sequences.

Online inference avoids a static logit corpus and can follow student-relevant prompts, but it couples training throughput to teacher capacity and version availability. Offline targets make runs reproducible but can become stale and create heavy object-store traffic. Top-k storage is effective when most useful probability mass is concentrated, provided the omitted tail is represented consistently rather than silently renormalized into a different target.

Version every teacher artifact with teacher checkpoint, tokenizer, prompt template, decoding or temperature configuration, precision, and generation code. If the student and teacher tokenize differently, sequence alignment and probability transfer require an explicit mapping; matching strings does not imply matching token events.

### Why combine soft and hard targets

Soft targets transfer relative preference among tokens. Hard labels preserve ground truth or verified behavior when the teacher is wrong. The balance depends on teacher reliability, student capacity, domain shift, and whether labels represent a unique answer.

For generated sequence data, a hard next-token label from a teacher sample discards uncertainty. Offline logits preserve more information but are enormous. Alternatives include top-k logit storage, quantized logits, on-policy teacher queries, sequence-level ranking, hidden-state targets, or verifier-selected trajectories.

### Sequence and feature distillation

**Sequence distillation** trains on teacher-generated outputs. It can simplify the target distribution and transfer style or reasoning patterns, but it inherits teacher errors and may reduce diversity. **Feature distillation** aligns hidden representations or attention maps, which can help when architectures are compatible. Layer mapping, scale, and representational non-identifiability complicate it.

**Online distillation** queries the teacher on student-relevant states, reducing mismatch between a fixed corpus and the student's current failure modes. It costs teacher inference and creates a moving data distribution.

### Draft-model distillation

A speculative draft is evaluated jointly with the target. The most relevant objectives are acceptance rate, average accepted prefix, draft latency, target verification cost, memory footprint, and final exactness.

Train on real product prompts and target continuations. Include high-entropy positions, domain slices, long-context states, and student failure cases. Match tokenizer and vocabulary unless the system explicitly supports a mapping; otherwise verification and KV reuse become much harder.

An effective loss can combine hard target tokens, forward KL on target logits, and emphasis on positions that control rejection. Evaluate under the actual sampling policy. A draft that agrees under greedy decoding may disagree at production temperature.

:::callout insight|The best draft is not the best small model
The best draft maximizes end-to-end target throughput under memory and latency constraints. Standalone perplexity is a proxy; acceptance per microsecond is closer to the objective.
:::

### Model-size tradeoff

A larger draft costs more per proposed token but can raise acceptance. A smaller draft is cheap but may create verification waste. Let:

- `C_d(m)` be draft cost for model size `m`;
- `C_v(gamma)` be target verification cost for `gamma` proposed tokens;
- `A(m, gamma)` be expected accepted tokens;
- `C_base` be one ordinary target decode step.

A simple speed model is:

:::equation Speedup = A(m,γ) C_{base} / (C_{draft}(m,γ) + C_{verify}(γ))|The useful numerator is committed target-equivalent progress; the denominator includes draft and verification time.

The model omits queueing and overlap but identifies the decision. Benchmark a grid of draft size and proposal length using production shapes. Watch memory capacity: loading a larger draft can reduce target batch size and erase its acceptance benefit.

### Evaluate transfer, not resemblance alone

Lower distillation loss does not prove that the student is more useful. Evaluate four layers:

1. **distribution agreement:** KL, calibration, rank correlation, or acceptance under the relevant prompts;
2. **capability transfer:** task success, critical slices, robustness, and failures inherited from the teacher;
3. **systems outcome:** latency, memory, throughput, energy, and target batch capacity;
4. **independent evidence:** verified labels or human judgment that can disagree with the teacher.

Compare against a student trained on hard labels with the same tokens and compute. Otherwise, gains attributed to soft targets may come from additional teacher-generated data or a different curriculum.

### Anti-distillation and extraction resistance

"Anti-distillation" covers techniques that make model extraction less effective: rate limits, query anomaly detection, output perturbation, watermarking, restricted logit access, policy controls, and legal or contractual measures. There is no free defense. Perturbing outputs can harm users and may not stop a determined extractor. Detection must distinguish legitimate high-volume customers from imitation workflows.

A rigorous framing is a threat model: attacker access, budget, query adaptivity, target fidelity, and acceptable product degradation. Treat claims of protection as measured risk reduction, not impossibility.

### Design Exercises

1. Explain forward versus reverse KL using behavior and gradients, not only slogans.
2. Why multiply a temperature-scaled KL term by `T^2`?
3. Design a draft-model training corpus and evaluation.
4. When can a 3B draft outperform a 1B draft end to end?
5. What defenses against extraction preserve normal user quality?

## Supervised Adaptation and Low-Rank Training

LEAD: A useful fine-tuning run begins with a reproducible data-to-gradient path. Understand which targets are learned, which parameters move, and which evaluation would reveal that the adaptation damaged the base model.

### Choose between continued pretraining and instruction learning

Continued pretraining learns from the next-token structure of a domain corpus. SFT learns from demonstrations of desired responses under an explicit interaction format. They can use the same cross-entropy machinery while supervising different targets. A repository dump does not automatically teach a model to follow a debugging request; a short answer demonstration does not expose the full distribution of a technical domain.

For a documentation assistant, distinguish domain knowledge from current facts. Fine-tuning can teach terminology, tool-call syntax, and answer conventions. Frequently changing permissions and policies should remain in a governed retrieval system. Putting them into weights does not create reliable deletion or authorization semantics.

Begin with a base or instruction model that already fits the required language and task family. Establish a no-training baseline with the intended prompt and retrieval. If the failure is a missing document or a wrong access filter, model training attacks the wrong layer. If examples consistently show a learnable output or decision pattern, adaptation becomes a testable intervention.

### Make one supervised batch completely visible

Take a serialized exchange containing a user question, an assistant tool call, a tool result, and a final assistant answer. For assistant-only supervision, mark target positions belonging to the two assistant messages, including whichever delimiters the deployment must generate. Leave user and tool-result targets unscored while keeping them visible as context. The exact template and token spans come from Part I's text-interface contract.

For each batch, retain the number of valid target tokens. Across unequal microbatches, accumulate the sum of token losses and divide by the global supervised-token count. Averaging each microbatch mean equally implements a different weighting when the target counts differ. For example, a two-target microbatch and a 100-target microbatch should not receive equal weight under a token-mean objective.

Inspect a tiny overfitting test before a full run: can the model drive loss down on a handful of valid demonstrations, and does decoded behavior match the target format? If not, first check token shifts, masks, frozen parameters, optimizer membership, and gradient flow. Success on this test establishes plumbing, not generalization. A held-out split must separate related documents, templates, or tasks to avoid measuring memorized variants.

### LoRA reduces trainable state, not all training memory

[LoRA](https://arxiv.org/abs/2106.09685) freezes a base matrix and learns a low-rank update. Using the book's row-vector convention, let `W` have shape `[d_in,d_out]`, `A` shape `[d_in,r]`, and `B` shape `[r,d_out]`:

:::equation Y = X W + s (X A) B|Only the low-rank matrices are trained when the base weight W is frozen.

The adapter contains `r(d_in+d_out)` parameters instead of `d_in*d_out`. For a 4096-by-4096 matrix and rank 16, that is 131,072 versus 16,777,216 parameters: 128 times fewer trainable parameters for this matrix. This does not mean 128 times less total GPU memory. The frozen weights still reside somewhere; activations must support backward; temporary GEMMs and communication buffers still exist.

A common initialization sets one factor randomly and the other to zero so the initial update is zero. If both factors were zero, their product's gradients would also be zero and the adapter would not begin learning. Scaling conventions vary, so record the actual scale rather than assuming the rank alone defines the update.

Target-module selection matters. Attention-only adapters and adapters on all large linear layers have different capacity and memory cost. Increase rank only after separating underfitting from poor data or incorrect targets. Measure broad capability and domain behavior, not only training loss. Adapter merging forms `W+sAB` for a fixed adapter; serving many adapters simultaneously generally preserves the factors instead.

### QLoRA separates storage precision from the gradient path

[QLoRA](https://arxiv.org/abs/2305.14314) combines a frozen quantized base with trainable low-rank adapters, including NF4 storage, quantized scale metadata, and memory-management techniques. It does not backpropagate updates into the packed four-bit base weights as if those integers were ordinary trainable floating-point parameters.

The layer reconstructs or otherwise consumes the base weights in the supported compute path, adds the adapter contribution, and propagates gradients to the adapter and required activations. Four-bit storage therefore does not imply every multiplication or gradient uses four bits. NF4 is a nonuniform quantization codebook, not the same format as hardware FP4.

A memory ledger needs packed base weights, scales and metadata, adapters, their gradients and optimizer states, activations, temporary dequantization/workspaces, and allocator reserve. On a nominal 7B model, ideal four-bit weight payload is 3.5 GB before metadata. Use that only as a lower bound; it is not a promised device-memory requirement.

Merging a trained adapter into a low-bit checkpoint requires care. Forming a higher-precision merged matrix and requantizing can introduce a different error from serving the unmerged quantized base plus adapter. Evaluate the exact deployment artifact, not just the training-time representation.

### A controlled adaptation experiment

Use an illustrative 2,000-example domain task set with a separate held-out group split. Compare the unchanged model, prompt/retrieval-only improvements, SFT with LoRA, and full fine-tuning if resources permit. Keep the source corpus, serialization, target masks, evaluation prompts, and generation policy fixed. Sweep a small learning-rate/rank grid instead of changing every variable together.

Report target-task accuracy, schema validity, unsupported claims, broad-capability retention, and training/serving cost. Add early stopping based on validation, but do not repeatedly tune against the final test set. Test multiple random seeds when differences are small. These are experiment instructions and illustrative counts, not results already measured for a particular model.

If the adapter improves answer style while factuality is unchanged, say so. If a full fine-tune improves the domain but loses multilingual behavior, the release decision must include that regression. Parameter efficiency is useful only when the retained behavior meets the product contract.

### Exercises and worked answers

1. **Why not update only an embedding table for every adaptation?** The failure may require changed transformations or decisions, not just different token vectors; choose trainable modules from the task and compare controls.
2. **Why can a frozen base still require activations for backward?** Gradients must traverse its operations to reach earlier trainable adapters even though the base's own weights are not updated.
3. **What breaks when loss is averaged by microbatch?** Unequal supervised-token counts receive unintended equal weight. Accumulate numerators and denominators consistently with the intended objective.
4. **When is retrieval preferable to memorizing documents?** When evidence changes frequently, must be cited, or has deletion/access requirements that weights cannot enforce reliably.

## Post-Training and Alignment

LEAD: Post-training changes a model's behavior using demonstrations, preferences, and interaction. Its gains depend on the starting model, exploration, data, and objective; it is not limited to formatting, nor does a higher reward establish general reasoning improvement.

### Start with a behavior specification

Post-training should begin with observable behavior, not an algorithm acronym. Define how the model should follow instructions, use tools, express uncertainty, handle conflicting authority, refuse unsafe requests, and preserve capability across languages and domains. Convert those expectations into demonstrations, comparisons, executable checks, and protected evaluations.

Separate three kinds of change:

- **capability acquisition**, where the model learns a task or tool it could not perform reliably;
- **policy selection**, where training shifts probability among behaviors already available;
- **interface conditioning**, where output format, tone, or interaction protocol changes.

SFT, preference optimization, and reinforcement learning can affect all three, but they provide different evidence and failure modes. Naming the intended change makes regressions easier to interpret.

### Supervised fine-tuning

SFT is appropriate when experts can demonstrate the desired behavior. It teaches format, policy, tool use, domain tone, and solution trajectories. High-quality demonstrations often beat a much larger weak corpus.

Mixing must protect broad capability. Include replay from general instruction data when specializing. Mask user or tool tokens according to the learning objective. For multi-turn data, test whether the loss incorrectly rewards copying system content or private tool outputs.

### Preference data

Pairwise preferences can be collected from humans, verifiers, judges, or product behavior. Each source has bias. Human labels reflect rubric clarity and annotator context. Model judges may favor verbosity, familiar style, or their own outputs. Engagement data is confounded by ranking, position, and user intent.

Build calibration sets with expert adjudication. Measure agreement by slice. Retain ties and uncertainty rather than forcing every comparison into a binary label.

A preference record should preserve both candidates, generation policies, presentation order, rubric, annotator or judge version, confidence, and adjudication status. Randomize order to detect position bias. Control length or explicitly model it so that verbosity does not become an accidental reward dimension.

### RLHF and direct preference objectives

Classic RLHF trains a reward model and optimizes a policy while constraining divergence from a reference. It supports online sampling and sequence-level reward, but it is operationally complex and vulnerable to reward hacking.

Direct Preference Optimization converts pairwise preferences into a supervised-style objective relative to a reference policy. It removes the explicit reward-model-and-RL loop, simplifying training. Other objectives alter the link function, treatment of a reference, margin, or unpaired feedback.

Do not choose by acronym. Ask:

- Is reward available only at the sequence level?
- Must the policy explore new outputs?
- How reliable and stationary is the preference signal?
- Is an explicit reward model useful for analysis or reuse?
- How much divergence from the base model is safe?

The KL constraint or reference policy is not merely a mathematical convenience. It discourages large changes into regions where the reward model has little evidence, without guaranteeing safety there. Monitor divergence by task slice and response length; a single aggregate can hide a policy that changes radically on rare but important prompts.

### Derive the DPO training signal

For one prompt, let `y+` be the preferred answer and `y-` the rejected answer. Compute each answer's log probability as the sum over its scored response tokens, conditioned on the same prompt. Let `pi` be the policy being trained and `pi_ref` the frozen reference. Define:

:::equation Δ = (log π(y^{+}) - log π(y^{-})) - (log π_{ref}(y^{+}) - log π_{ref}(y^{-}))|The margin compares the policy's preference with the reference's preference for the same pair.

:::equation L_{DPO} = -log σ(β Δ)|Beta sets the scale of the relative log-probability margin; sigma is the logistic function.

This is the core objective from [Direct Preference Optimization](https://arxiv.org/abs/2305.18290). The prompt is omitted from the notation but remains part of every conditional probability. The reference contributes fixed log probabilities and receives no gradient. The loss is a binary preference objective, not next-token imitation of the preferred answer alone.

Suppose the current model's chosen/rejected log probabilities are `-2` and `-4`, while the reference's are `-3` and `-4`. The relative margin is one. With `beta=0.5`, the loss is about `0.474`; if the policy matches the reference's relative preference, the loss is `log(2)`, about `0.693`. Moving probability toward the chosen answer relative to the rejected one lowers the loss. The scalar reference in `examples/post_training.py` tests this direction and numerical stability.

The reference is not the policy that generated the pair unless the experiment chose it that way. Keep those identities separate. Standard sequence log probabilities sum over tokens; replacing the sum with a length average changes the objective. A model can also improve the pairwise margin while lowering both candidates' absolute probabilities. Monitor held-out generation behavior and likelihood, not only training pair accuracy.

Preference labels may be wrong or weakly ordered. Ties, contradictory rubrics, long-versus-short confounds, and judge familiarity can all drive a clean mathematical objective toward undesirable behavior. DPO removes an explicit online rollout stage during each training update; it does not remove the need for representative candidate generation, evaluation, or a reliable preference dataset.

### Verifiable reward and reasoning

Code execution, theorem checks, database answers, and simulated environments provide strong outcome signals. They still permit shortcut exploitation. Split generators and verifiers, randomize hidden tests, audit suspiciously short or repetitive traces, and evaluate on tasks unavailable to the training loop.

Process supervision labels intermediate steps and can improve diagnosis, but step correctness may be ambiguous and expensive to annotate. Outcome supervision is cheaper when final answers are verifiable but provides sparse credit. Hybrid designs use outcome reward plus auxiliary signals for format, tool validity, or progress.

:::callout pitfall|A better reward score is not automatically a better model
Optimization changes the policy distribution, so the reward model is evaluated out of its training distribution. Track independent capability and safety metrics, divergence, response length, and exploit signatures.
:::

### Online learning and rollback

Online post-training creates feedback loops. New policy outputs influence the data later used to judge or train it. Protect a stable holdout, log exposure probabilities, version policies and reward models, and preserve the ability to reconstruct which policy generated each sample.

Use canaries and conservative trust regions. A rollback should restore model, tokenizer, tool policy, prompt templates, safety configuration, and serving parameters as one versioned release.

### Build a post-training evaluation ladder

Evaluate the data before the model and the model before deployment:

1. audit label consistency, judge bias, leakage, and slice coverage;
2. run loss and reward sanity checks on fixed candidate pairs;
3. train a small pilot and inspect length, style, refusal, and capability shifts;
4. compare against SFT-only and base-model controls;
5. run adversarial and out-of-distribution evaluations;
6. replay representative tool and multi-turn traffic;
7. canary the atomic release with automatic rollback criteria.

Post-training often produces improvements that are easy to notice and regressions that are diffuse. Preserve broad capability floors and report changes by slice, not only a blended preference win rate.

### Design Exercises

1. When does DPO simplify a problem that does not require online RL?
2. Design defenses against reward hacking for a code reasoning model.
3. How do you audit a model judge for verbosity bias?
4. What belongs in an atomic rollback for an aligned assistant?
5. Explain why preference optimization can reduce capability even as reward rises.

### Worked answer criteria

1. DPO is attractive when fixed, trustworthy comparisons cover the desired change and online exploration is not essential. Compare it with SFT and an unchanged-policy baseline.
2. Run generated code in an isolated environment with protected tests and bounded resources; distinguish wrong answers from infrastructure errors and test for attempts to tamper with the checker.
3. Randomize candidate order, stratify by length, use expert-adjudicated ties, and compare judges against a held-out rubric. Do not silently equate verbosity with correctness.
4. Roll back the compatibility set: weights, tokenizer, template, tool policy, serving/sampling configuration, and relevant reward/evaluation versions.
5. Reward is a proxy. Distribution shift, biased labels, overoptimization, and loss of rare capabilities can all coexist with a rising proxy score.

## Reinforcement Learning for Reasoning and Tool Use

LEAD: Reinforcement learning adjusts the probability of actions using outcomes from sampled trajectories. To understand current reasoning training, follow one trajectory from generation through reward, advantage, policy update, and a fresh held-out evaluation.

### Tokens are actions; the conversation is state

For text generation, the state includes the prompt and previously generated tokens. An action is the next token. For a tool-using agent, external observations and environment state also influence future decisions. A trajectory ends at an answer, failure, timeout, or explicit task boundary. Rewards may arrive at intermediate steps or only at the end.

A policy-gradient update increases log probability for actions with positive advantage and decreases it for actions with negative advantage. Advantage means outcome relative to an appropriate baseline, not simply “the answer was correct.” Subtracting a baseline can reduce estimator variance, but the baseline and normalization determine how prompts are weighted.

Terminal reward offers weak credit assignment: every sampled action may inherit a signal from the final result, including unnecessary prose and accidental shortcuts. A process reward can give finer feedback but may itself be incorrect. Neither establishes that a visible chain of thought faithfully describes the computation that produced an answer.

### PPO: distinguish the old policy from the reference

[PPO](https://arxiv.org/abs/1707.06347) uses a clipped surrogate to limit incentives for large policy changes on sampled actions. Let `r_t` be the new policy's probability of the sampled action divided by its probability under the behavior policy that collected the rollout. A simplified term to maximize is:

:::equation J_{t} = min(r_{t} A_{t}, clip(r_{t}, 1-ε, 1+ε) A_{t})|Clipping limits the benefit of moving a sampled action too far in the advantageous direction.

For advantage `+2`, ratio `1.4`, and epsilon `0.2`, the clipped term is `2.4`, not `2.8`. For advantage `-2` and ratio `0.6`, it is `-1.6`, not `-1.2`. The minimum matters for both signs. This does not guarantee a hard bound on policy divergence; other actions and shared parameters also move.

The **old/behavior policy** supplies the sampling probabilities for the ratio. The **reference policy** supplies a regularization anchor, often an SFT checkpoint. A **critic** estimates future return or a value baseline. A **reward model or verifier** scores outcomes. These roles can share architectures but are not interchangeable. Classic actor-critic PPO usually trains a value estimator and may use generalized advantage estimation; omitting it changes how advantage is obtained.

### GRPO: use a group of outcomes as the baseline

[DeepSeekMath](https://arxiv.org/abs/2402.03300) introduced Group Relative Policy Optimization. Sample several responses to the same prompt, evaluate their rewards, and obtain a relative advantage from the group's mean and, in the normalized form, standard deviation. This avoids a separate learned critic for that baseline, but still requires policy, rollout, and reward computation.

For rewards `[1,1,0,0]`, the mean is `0.5` and population standard deviation is `0.5`; standardized advantages are `[1,1,-1,-1]`. For `[1,1,1,1]`, there is no within-group contrast. A safe implementation assigns zero contrast instead of dividing by zero. The group therefore needs useful diversity: all-correct and all-wrong groups carry little or no relative reward signal.

This creates an important data tradeoff. Easy prompts can become uninformative, while very hard prompts may never yield a successful trajectory. Increasing group size can expose contrast but consumes more rollout tokens and verifier calls. Filtering zero-variance groups changes the effective training distribution; record attempted prompts and discarded groups, not just the accepted training batch.

Reward normalization is a choice, not a harmless cosmetic step. Dividing by group standard deviation changes the scale across prompts; averaging each response's token loss can change length weighting. An implementation must specify reward normalization, token/sequence normalization, response masks, KL placement, and clipping. Two programs both called GRPO can optimize measurably different objectives.

### What newer variants are trying to repair

| Approach | Mechanism-level change | Decision to inspect |
| --- | --- | --- |
| DAPO | Decoupled clipping, dynamic sampling, token-level loss aggregation, and explicit treatment of overlong responses | Are entropy collapse, uninformative groups, or truncation dominating? |
| GSPO | A length-normalized sequence likelihood ratio and sequence-level clipping/optimization | Is token-level importance weighting unstable for the model and rollout distribution? |
| Outcome/process reward mixtures | Add signals at different parts of the trajectory | Does finer credit improve held-out success, or only reward the checker's preferred style? |
| On-policy distillation | Train on states visited by the student using teacher feedback | Is teacher-forced training missing the student's actual error distribution? |

The primary descriptions are [DAPO](https://arxiv.org/abs/2503.14476) and [GSPO](https://arxiv.org/abs/2507.18071). These methods should be compared as concrete objectives and pipelines, not stacked as independent switches. A wider upper clipping range, for example, changes the incentive to increase low-probability successful actions; it is not a universal fix for poor exploration.

For GSPO's normalized sequence ratio, average token log-probability differences and exponentiate. If two token probability ratios are 2 and 0.5, their geometric mean is 1, while their arithmetic mean is 1.25. Using the latter implements the wrong quantity. Length normalization helps control scale but is not the full trajectory importance ratio, which would multiply all token ratios. The scalar reference checks this distinction.

For truncation, distinguish a genuinely wrong completed answer from a trajectory cut off by a budget or failed environment. Giving every timeout the same semantic reward as an invalid solution can teach the wrong lesson. Conversely, dropping all failures without reporting them hides the true cost and can bias training toward easy tasks. Define the censoring and reward policy before interpreting improvements.

### Rich feedback and self-distillation: a 2026 direction

A scalar failure says little about which action to change. [Self-Distillation Policy Optimization](https://arxiv.org/abs/2601.20802v2), a 2026 study, conditions a self-teacher on feedback and distills its feedback-informed predictions into the policy. The teaching idea is to turn information such as a compiler error into a richer learning signal rather than reducing every failed attempt to the same zero reward.

For an original toy task, suppose a model writes a function with the wrong argument order. A trusted error message identifies the mismatch. A feedback-conditioned teacher distribution may put more mass on the corrected call, while the student must learn to make that call without seeing the future error at deployment. The training pipeline must keep teacher-only feedback out of the student's evaluation input and detach the target distribution appropriately. Otherwise it measures access to an answer hint, not learning.

This is not a license to trust every self-generated correction. The July 2026 preprint [Denser Is Not Better](https://arxiv.org/abs/2607.01763) reports forgetting and drift in its continual post-training experiments. The September 2 preprint [Learn from Whoever Is Right](https://arxiv.org/abs/2609.02548) explores answer-verified multi-teacher distillation. These are emerging findings with different experimental settings, not a settled ranking of RL objectives.

When studying this family, track who supplies feedback, which tokens the teacher sees, which distribution supplies student trajectories, which parameters are updated, and whether old capabilities survive. An external teacher, a self-teacher with privileged context, and an outcome verifier supply different information even if all produce a training target.

### Verifiable reward needs an independently protected verifier

RL with verifiable rewards is useful when outcomes can be checked: execution tests, formal proof checking, exact structured answers, or environment success conditions. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) is an important example of reasoning-oriented reinforcement learning, but its reported results do not imply that any base model becomes broadly capable from a sparse binary reward alone.

Consider a code task with visible examples and hidden tests. The model receives a sandbox with source files and a way to submit a patch. Hidden tests and the scoring program live outside its writable scope. The verifier distinguishes compilation failure, test failure, infrastructure failure, timeout, and valid success. Network, filesystem, subprocess, and resource permissions are enforced by the environment, not by an instruction asking the model to behave.

Protect the held-out task set from prompt generation, selection, reward tuning, and synthetic-data construction. Generate adversarial verifier tests: empty output, fake success logs, modified tests, hardcoded examples, and attempts to read the checker. A rising training reward accompanied by unchanged hidden-test success is a reward-model problem until shown otherwise.

### From one batch to the next

Example status: Explanatory pseudocode.

```text
publish immutable actor snapshot v
sample prompt groups from a versioned task mixture
generate trajectories with v; save behavior log probabilities
run protected verifiers; distinguish failures from missing labels
compute advantages under the declared normalization policy
recompute current-policy log probabilities on scored action tokens
apply clipped objective and any explicit reference regularizer
check finite gradients; update actor and optimizer consistently
evaluate on independent tasks at a fixed sampling/token budget
publish v+1 only after release checks; retain rollback snapshot
```

Tool-result tokens are observations, not actions sampled from the actor, and normally should not enter the actor's action log-probability objective. Store token/action masks explicitly. For multi-turn trajectories, preserve tool versions, environment seeds, reset state, and termination reasons; replaying only the visible final answer is insufficient.

Suppose 128 prompts receive eight rollouts each, averaging 1,024 generated tokens. That is 1,048,576 generated tokens before verification and training. At an illustrative aggregate rollout rate of 20,000 tokens per second, generation alone takes at least 52.4 seconds. If optimization takes 12 seconds, making the optimizer twice as fast saves only six seconds from a serial cycle dominated by generation. This calculation explains why inference kernels and rollout scheduling are part of training research.

[HybridFlow](https://arxiv.org/abs/2409.19256) addresses the interaction between model roles and distributed execution. The architectural lesson is to plan training and generation layouts together. Their best parallel groups and memory allocations can differ, and switching representations or moving weights has a cost. Part V follows the ownership and staleness consequences rather than treating the RL library as a black box.

### Measure learning, not only pass at a larger budget

Report single-sample success under a fixed generation budget, success with multiple candidates, the selection rule, and total inference/verifier cost. The probability that at least one candidate is correct is not the same as the ability to choose that candidate without an oracle. Majority voting, a learned judge, and executable selection have different failure modes.

For `n` independent identically sampled candidates containing `c` correct ones, the common pass-at-k estimator is `1 - choose(n-c,k)/choose(n,k)`, with result one when fewer than `k` candidates are wrong. The reference `pass_at_k(10,2,3)` returns about `0.533`. It estimates an at-least-one-success event under the sampling assumptions, not the quality of the model's first answer. Correlated task variants and repeated tuning on a benchmark require separate caution.

Track answer length, entropy, group reward variance, discarded prompts, clipping fraction, KL estimates, verifier error rates, and behavior-policy age. Separate gains from better reasoning, more sampling, different tools, and relaxed time budgets. A shorter iteration loop is valuable only if it produces valid evidence about learning.

### Exercises and worked answers

1. **Why is a reference model not the PPO denominator?** The denominator is the policy that sampled the action. The reference is a regularization choice and may remain fixed across many updates.
2. **What happens to an all-wrong binary-reward group?** Its within-group contrast is zero. Resampling or changing task difficulty may help, but changes the compute cost and training distribution.
3. **Can pass-at-8 rise while pass-at-1 falls?** Yes. Distributional changes can improve coverage while hurting the most likely or typical answer. Report both and the actual selection mechanism.
4. **What is wrong with rewarding a process's printed `PASS`?** The actor can print it. The trusted verifier must inspect independently protected execution results, not accept self-reported success.
5. **Which first optimization fits the numerical cycle above?** Improve valid rollout throughput or overlap before assuming optimizer speed dominates; include verifier capacity and policy staleness in the measurement.
