"""Small arithmetic references: ranking fusion and normalized token late interaction."""
import math


def reciprocal_rank_fusion(rankings, k=60):
    if not math.isfinite(k) or k <= 0:
        raise ValueError("k must be positive")
    scores = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("a ranking must not repeat a document")
        for rank, document in enumerate(ranking, 1):
            scores[document] = scores.get(document, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda row: (-row[1], row[0]))


def maxsim(query_tokens, document_tokens):
    if not query_tokens or not document_tokens:
        raise ValueError("token collections must be nonempty")
    dimension = len(query_tokens[0])
    normalized = []
    for token in list(query_tokens) + list(document_tokens):
        if len(token) != dimension or not all(math.isfinite(x) for x in token):
            raise ValueError("incompatible token vectors")
        norm = math.sqrt(sum(x*x for x in token))
        if norm == 0:
            raise ValueError("zero vector has no cosine direction")
        normalized.append([x/norm for x in token])
    queries = normalized[:len(query_tokens)]
    documents = normalized[len(query_tokens):]
    return sum(max(sum(a*b for a, b in zip(q, d)) for d in documents) for q in queries)
