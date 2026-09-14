from __future__ import annotations

import math
import random
import statistics
import unittest

from examples.attention import attention
from examples.streaming import Moments, top_k
from examples.rag import CASES, DOCUMENTS, answer, evaluate, retrieve


class AttentionTests(unittest.TestCase):
    def test_uniform_scores(self):
        self.assertEqual(attention([0.0], [[1.0], [2.0]], [[2.0], [4.0]]), [3.0])

    def test_one_visible_key(self):
        self.assertEqual(attention([2.0], [[1.0], [3.0]], [[9.0], [5.0]], [True, False]), [9.0])

    def test_all_masked(self):
        self.assertEqual(attention([2.0], [[1.0], [3.0]], [[9.0], [5.0]], [False, False], 1), [0.0])

    def test_all_masked_skips_overflowing_scores(self):
        for shard_size in (None, 1):
            self.assertEqual(attention([1e308], [[1e308]], [[1.0]], [False], shard_size), [0.0])

    def test_partition_merge_against_independent_dense_formula(self):
        rng = random.Random(17)
        for length in [1, 2, 7, 11]:
            q = [rng.uniform(-1, 1) for _ in range(3)]
            keys = [[rng.uniform(-1, 1) for _ in q] for _ in range(length)]
            values = [[rng.uniform(-2, 2) for _ in range(2)] for _ in keys]
            weights = [math.exp(sum(x*y for x, y in zip(q, k))/math.sqrt(3)) for k in keys]
            expected = [sum(w*v[j] for w,v in zip(weights, values))/sum(weights) for j in range(2)]
            for size in [1, 2, 4, length + 1]:
                result = attention(q, keys, values, shard_size=size)
                for actual, target in zip(result, expected):
                    self.assertAlmostEqual(actual, target, delta=1e-12)

    def test_partition_merge_ignores_fully_masked_shard(self):
        result = attention([0.0], [[1.0], [2.0]], [[2.0], [4.0]],
                           [False, True], shard_size=1)
        self.assertEqual(result, [4.0])
        query = [0.0, 0.0]
        keys = [[float(i), -float(i)] for i in range(7)]
        values = [[float(i), 2.0 * i] for i in range(7)]
        allowed = [False, False, True, False, True, True, False]
        expected = [sum(value[j] for value, keep in zip(values, allowed) if keep)
                    / sum(allowed) for j in range(2)]
        for shard_size in (1, 2, 3, 4):
            result = attention(query, keys, values, allowed, shard_size)
            for actual, target in zip(result, expected):
                self.assertAlmostEqual(actual, target, delta=1e-12)

    def test_attention_rejects_score_overflow_dense_and_sharded(self):
        for shard_size in (None, 1):
            with self.assertRaisesRegex(ValueError, "attention arithmetic"):
                attention([1e308], [[1e308]], [[1.0]], shard_size=shard_size)

    def test_attention_rejects_partial_and_merged_numerator_overflow(self):
        args = ([0.0], [[1.0], [2.0]], [[1.7e308], [1.7e308]])
        for shard_size in (None, 1):
            with self.assertRaisesRegex(ValueError, "attention arithmetic"):
                attention(*args, shard_size=shard_size)

    def test_extreme_scores_are_stable(self):
        self.assertEqual(attention([1000.0], [[1.0], [-1.0]], [[2.0], [8.0]], shard_size=1), [2.0])

    def test_invalid_shapes(self):
        for q, k, v in [([], [[1]], [[1]]), ([1], [], []), ([1], [[1, 2]], [[1]]),
                        ([1], [[1], [2]], [[1], [2, 3]]), ([float('nan')], [[1]], [[1]])]:
            with self.assertRaises(ValueError):
                attention(q, k, v)
        with self.assertRaises(ValueError):
            attention([1], [[1]], [[1]], [])
        with self.assertRaises(ValueError):
            attention([1], [[1]], [[1]], shard_size=0)


class StreamingTests(unittest.TestCase):
    def test_top_k_matches_sort_and_distributed_merge(self):
        rng = random.Random(42)
        for n in [0, 1, 20, 100]:
            records = [(rng.randrange(5), str(i)) for i in range(n)]
            for k in [0, 1, 7, 101]:
                expected = sorted(records, reverse=True)[:k]
                self.assertEqual(top_k(records, k), expected)
                local = [record for p in range(3) for record in top_k(records[p::3], k)]
                self.assertEqual(top_k(local, k), expected)

    def test_top_k_validation(self):
        for k in [-1, 0.5]:
            with self.assertRaises(ValueError):
                top_k([], k)
        for x in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                top_k([(x, 'a')], 1)

    def test_variance_merge_and_large_offset(self):
        data = [1e9 + i for i in range(100)]
        states = []
        for shard in [data[:37], data[37:]]:
            state = Moments()
            for value in shard:
                state = state.add(value)
            states.append(state)
        result = states[0].merge(states[1])
        self.assertEqual(result.count, len(data))
        self.assertAlmostEqual(result.mean, statistics.mean(data), delta=1e-6)
        self.assertAlmostEqual(result.sample_variance(), statistics.variance(data), delta=1e-6)
        self.assertEqual(result.merge(Moments()), result)
        self.assertEqual(Moments().merge(result), result)

    def test_variance_insufficient_data(self):
        with self.assertRaises(ValueError):
            Moments().add(1).sample_variance()


class RAGTests(unittest.TestCase):
    def test_stale_revision_is_excluded(self):
        docs = retrieve(CASES[0])
        self.assertTrue(all(d.current for d in docs))
        self.assertEqual(answer(CASES[0], docs)[0], "30 days")

    def test_acl_before_ranking(self):
        self.assertEqual(retrieve(CASES[1]), [])

    def test_unanswerable_abstains(self):
        self.assertEqual(answer(CASES[3], retrieve(CASES[3])), (None, ()))

    def test_conflicting_current_sources_abstain(self):
        self.assertEqual(answer(CASES[4], retrieve(CASES[4])), (None, ()))

    def test_version_or_permission_removal_does_not_reuse_answer(self):
        remaining = tuple(d for d in DOCUMENTS if d.id != "retention-v2")
        self.assertEqual(answer(CASES[0], retrieve(CASES[0], remaining)), (None, ()))

    def test_retrieval_success_does_not_guarantee_correct_answer(self):
        baseline = evaluate(safe=False)
        guarded = evaluate()
        self.assertEqual(baseline["evidence_recalled"], 2)
        self.assertEqual(baseline["correct"], 2)
        self.assertEqual(baseline["acl_leaks"], 1)
        self.assertEqual(guarded["correct"], 5)
        self.assertEqual(guarded["acl_leaks"], 0)
        self.assertEqual(guarded["stale_retrievals"], 0)
        self.assertEqual(guarded["answers_with_valid_citations"], guarded["answers_emitted"])


if __name__ == "__main__":
    unittest.main()
