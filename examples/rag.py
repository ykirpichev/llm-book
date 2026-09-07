"""Deterministic retrieval/grounding fixture, not a real LLM or ANN benchmark.

The intentionally unsafe baseline exists only for the printed comparison.
It must never be used for a live retrieval service.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re


@dataclass(frozen=True)
class Document:
    id: str
    topic: str
    text: str
    answer: str
    roles: frozenset[str]
    current: bool = True


@dataclass(frozen=True)
class Case:
    query: str
    topic: str
    role: str
    expected: str | None
    evidence: frozenset[str]


DOCUMENTS = (
    Document("retention-v1", "retention", "retention logs policy logs", "90 days", frozenset({"reader"}), False),
    Document("retention-v2", "retention", "retention logs policy", "30 days", frozenset({"reader"})),
    Document("admin-key", "admin", "admin access secret key", "restricted value", frozenset({"admin"})),
    Document("reset", "reset", "password reset help", "Use the reset page", frozenset({"reader"})),
    Document("region-a", "region", "backup region policy", "Region A", frozenset({"reader"})),
    Document("region-b", "region", "backup region policy", "Region B", frozenset({"reader"})),
)
CASES = (
    Case("retention logs policy", "retention", "reader", "30 days", frozenset({"retention-v2"})),
    Case("admin access secret key", "admin", "reader", None, frozenset()),
    Case("password reset help", "reset", "reader", "Use the reset page", frozenset({"reset"})),
    Case("lunar office hours", "hours", "reader", None, frozenset()),
    Case("backup region policy", "region", "reader", None, frozenset()),
)


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(case, documents=DOCUMENTS, *, safe=True, k=4):
    if k < 1:
        raise ValueError("k must be positive")
    candidates = [d for d in documents if not safe or
                  (d.current and case.role in d.roles)]
    # Term-count overlap is intentionally simple and exposes stale duplicates.
    scored = [(sum(word in tokens(case.query) for word in d.text.lower().split()), d)
              for d in candidates]
    return [d for score, d in sorted(scored, key=lambda item: (-item[0], item[1].id))
            if score > 0][:k]


def answer(case, docs, *, safe=True):
    relevant = [d for d in docs if d.topic == case.topic] if safe else docs
    if not relevant:
        return None, ()
    # The topic and answer fields are fixture labels, NOT a production verifier.
    if safe and len({d.answer for d in relevant}) != 1:
        return None, ()
    chosen = relevant[0]
    return chosen.answer, (chosen.id,)


def evaluate(*, safe=True, documents=DOCUMENTS, cases=CASES):
    correct = leaks = stale = supported = emitted = recalled = answerable = 0
    abstained_correctly = unanswerable = 0
    for case in cases:
        docs = retrieve(case, documents, safe=safe)
        prediction, citations = answer(case, docs, safe=safe)
        correct += prediction == case.expected
        leaks += any(case.role not in d.roles for d in docs)
        stale += any(not d.current for d in docs)
        if case.expected is not None:
            answerable += 1
            recalled += case.evidence.issubset({d.id for d in docs})
        else:
            unanswerable += 1
            abstained_correctly += prediction is None
        if prediction is not None:
            emitted += 1
            supported += bool(citations) and all(any(
                d.id == identifier and d.answer == prediction and d.current
                and case.role in d.roles and d.topic == case.topic for d in docs
            ) for identifier in citations)
    return {"cases": len(cases), "correct": correct, "acl_leaks": leaks,
            "stale_retrievals": stale, "answerable_cases": answerable,
            "evidence_recalled": recalled, "answers_emitted": emitted,
            "answers_with_valid_citations": supported,
            "unanswerable_cases": unanswerable,
            "correct_abstentions": abstained_correctly}


if __name__ == "__main__":
    print(json.dumps({"unsafe_baseline": evaluate(safe=False),
                      "guarded_fixture": evaluate()}, indent=2))
