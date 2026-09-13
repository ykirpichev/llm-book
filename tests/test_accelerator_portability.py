import math
import unittest

from examples.accelerator_portability import (
    compile_break_even, kv_shard_plan, padding_work, rmsnorm_reference,
)


class AcceleratorPortabilityTests(unittest.TestCase):
    def test_rmsnorm_uses_actual_width(self):
        expected = [v / math.sqrt(14 / 3 + 1e-6) for v in [1, 2, 3]]
        self.assertEqual(rmsnorm_reference([1, 2, 3], [1, 1, 1]), expected)
        wrong_padded_denominator = 1 / math.sqrt(14 / 4 + 1e-6)
        self.assertNotAlmostEqual(expected[0], wrong_padded_denominator)

    def test_rmsnorm_zero_weight_sign_and_inputs_unchanged(self):
        row, weights = [0., 0.], [-2., 3.]
        self.assertEqual(rmsnorm_reference(row, weights), [0., 0.])
        self.assertEqual(row, [0., 0.])
        self.assertEqual(weights, [-2., 3.])
        result = rmsnorm_reference([1., -1.], [2., 3.])
        self.assertGreater(result[0], 0)
        self.assertLess(result[1], 0)

    def test_rmsnorm_rejects_invalid_inputs(self):
        for row, weights, eps in [([], [], 1e-6), ([1], [1, 2], 1e-6),
                                  ([math.nan], [1], 1e-6), ([1], [math.inf], 1e-6),
                                  ([1], [1], 0), ([1], [1], math.nan),
                                  ([1e308], [1], 1e-6),
                                  ([1, 0], [1.7e308, 1], 1e-6),
                                  ([1e154], [1], 1e308)]:
            with self.assertRaises(ValueError):
                rmsnorm_reference(row, weights, eps)

    def test_bucket_padding_and_exact_fit(self):
        work = padding_work(2300, [4096, 2048, 4096])
        self.assertEqual(work.bucket_tokens, 4096)
        self.assertAlmostEqual(work.linear_ratio, 1.7808695652173914)
        self.assertAlmostEqual(work.dense_pair_ratio, 3.1714964083175805)
        self.assertEqual(padding_work(2048, [2048, 4096]).linear_ratio, 1)
        for tokens, buckets in [(0, [1]), (2, [1]), (1, []), (1, [0]), (1, [True])]:
            with self.assertRaises(ValueError):
                padding_work(tokens, buckets)

    def test_kv_partition_and_replication(self):
        shape = dict(layers=32, batch=8, context=4096, kv_heads=8, head_dim=128, element_bytes=2)
        unsharded = kv_shard_plan(**shape)
        self.assertEqual(unsharded.bytes_per_rank, 4 * 2**30)
        tp4 = kv_shard_plan(**shape, tp=4)
        self.assertEqual(tp4.bytes_per_rank, 2**30)
        self.assertEqual(tp4.aggregate_modeled_kv_bytes, unsharded.aggregate_modeled_kv_bytes)
        tp16 = kv_shard_plan(**shape, tp=16)
        self.assertEqual(tp16.kv_replication, 2)
        self.assertEqual(tp16.bytes_per_rank, 512 * 2**20)
        self.assertEqual(tp16.aggregate_modeled_kv_bytes, 8 * 2**30)
        combined = kv_shard_plan(**shape, tp=4, cp=2)
        self.assertEqual(combined.bytes_per_rank, 512 * 2**20)
        self.assertEqual(combined.aggregate_modeled_kv_bytes, 4 * 2**30)

    def test_kv_context_padding_and_invalid_geometry(self):
        shape = dict(layers=1, batch=1, context=5, kv_heads=8, head_dim=1, element_bytes=2)
        plan = kv_shard_plan(**shape, tp=4, cp=2)
        self.assertEqual(plan.tokens_per_rank, 3)
        self.assertEqual(plan.aggregate_modeled_kv_bytes, 192)  # six padded tokens, not five
        for tp in [0, 3, 12, True]:
            with self.assertRaises(ValueError):
                kv_shard_plan(**shape, tp=tp)

    def test_compilation_amortization(self):
        self.assertEqual(compile_break_even(60, .125, .0625), 960)
        self.assertEqual(compile_break_even(0, .125, .0625), 0)
        self.assertEqual(compile_break_even(1e-308, 1e308, 1e307), 1)
        self.assertIsNone(compile_break_even(60, .1, .1))
        self.assertIsNone(compile_break_even(60, .1, .2))
        for times in [(-1, 1, .5), (1, 0, .5), (1, 1, 0), (math.inf, 1, .5),
                      (1e308, 1e-308, 5e-309)]:
            with self.assertRaises(ValueError):
                compile_break_even(*times)
