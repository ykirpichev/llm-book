"""Independent event timelines and admission conservation for the teaching model."""
from dataclasses import replace
import math
import unittest
from examples.serving_load import Config, Request, scenarios, simulate


class ServingLoadTests(unittest.TestCase):
    def test_one_request_analytic_service_and_memory(self):
        r = simulate([Request(7, prompt=8192, output=512)])
        self.assertEqual(r.completion_s, (1.25,))
        self.assertAlmostEqual(r.ttft_s[0], .25 + 1/512)
        self.assertEqual(r.horizon_s, 1.25)
        self.assertEqual(r.peak_gib, 56 + 3.5 + 8704/8192)

    def test_fifo_transfer_and_compute_overlap(self):
        cfg = Config(bandwidth_gib_s=1, replay_tokens_s=8192,
                     decode_tokens_s=512, ttft_limit_s=10, completion_limit_s=10)
        # Imports [0,1], [1,2]; compute [1,3], [3,5].
        r = simulate([Request(0, 8192, 512, 1), Request(0, 8192, 512, 1)], cfg)
        self.assertEqual(r.completion_s, (3., 5.))
        self.assertEqual(r.transfer_gib, 2.)
        self.assertEqual(r.within_slo, 2)
        self.assertAlmostEqual(r.goodput, .4)

    def test_release_precedes_admission_at_same_time(self):
        r = simulate([Request(0, 8192, 512), Request(1.25, 8192, 512)],
                     Config(slots=1))
        self.assertEqual(r.admitted, 2)
        self.assertEqual(r.peak_gib, 56 + 3.5 + 8704/8192)

    def test_memory_and_slot_rejections_are_distinct(self):
        trace = [Request(0)] * 6
        memory = simulate(trace, Config(slots=6))
        slots = simulate(trace, Config(slots=1))
        self.assertEqual((memory.admitted, memory.rejected_memory), (5, 1))
        self.assertEqual((slots.admitted, slots.rejected_slots), (1, 5))
        self.assertLessEqual(memory.peak_gib, 80)

    def test_rejections_and_drain_stay_in_accounting(self):
        r = simulate([Request(0), Request(.1), Request(2)], Config(slots=1))
        self.assertEqual((r.offered, r.admitted, r.rejected_slots), (3, 2, 1))
        self.assertEqual(r.horizon_s, 2.5)
        self.assertAlmostEqual(r.goodput, 2/2.5)
        r = simulate([Request(0, prompt=10**7)])
        self.assertEqual((r.admitted, r.within_slo, r.goodput), (0, 0, 0))

    def test_sweep_conservation_and_counterexamples(self):
        results = {name: simulate(trace, cfg) for name, trace, cfg in scenarios()}
        for r in results.values():
            self.assertEqual(r.offered, r.admitted+r.rejected_slots+r.rejected_memory)
            self.assertLessEqual(r.within_slo, r.admitted)
            self.assertLessEqual(r.peak_gib, 80 + 1e-10)
        self.assertLess(results['six-slots'].goodput, results['three-slots'].goodput)
        self.assertLess(results['reuse-slow-link'].goodput, results['three-slots'].goodput)
        self.assertGreater(results['reuse-fast-link'].goodput, results['three-slots'].goodput)

    def test_invalid_inputs_and_empty_trace(self):
        self.assertEqual(simulate([]).offered, 0)
        for cfg in (Config(slots=0), Config(slots=True), Config(bandwidth_gib_s=0),
                    Config(capacity_gib=math.inf), Config(fixed_gib=81)):
            with self.assertRaises(ValueError): simulate([], cfg)
        for trace in ([Request(math.nan)], [Request(1), Request(0)],
                      [Request(0, reuse=1.1)], [Request(0, output=0)]):
            with self.assertRaises(ValueError): simulate(trace)
