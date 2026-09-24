from dataclasses import FrozenInstanceError, replace
import math
import unittest

from examples.adoption_analysis import Outcome, TaskPair, analyze, synthetic_fixture


class AdoptionAnalysisTests(unittest.TestCase):
    def test_chapter_fixture_reproduces_counts_costs_and_decision(self):
        result = analyze(synthetic_fixture())
        self.assertEqual(result["paired_counts"],
                         dict(both=340, incumbent_only=20, candidate_only=30, neither=10))
        self.assertEqual(result["incumbent"]["successes"], 360)
        self.assertEqual(result["candidate"]["successes"], 370)
        self.assertAlmostEqual(result["incumbent"]["total_cost"], 160)
        self.assertAlmostEqual(result["candidate"]["total_cost"], 120)
        self.assertAlmostEqual(result["paired_improvement"], 0.025)
        self.assertAlmostEqual(result["paired_standard_error"],
                               math.sqrt((50 - 400*.025**2)/(399*400)), places=12)
        low, high = result["paired_normal_interval_95"]
        self.assertAlmostEqual(low, -0.009603, places=5)
        self.assertAlmostEqual(high, 0.059603, places=5)
        self.assertAlmostEqual(result["candidate"]["wilson_lower_95"], 0.894956, places=5)
        self.assertAlmostEqual(result["cost_reduction"], 0.27027027027)
        self.assertEqual(result["incumbent"]["p99_seconds"], 1.3)
        self.assertEqual(result["candidate"]["p99_seconds"], 1.45)
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(result["decision"], "bounded_canary_only")

    def test_pairing_changes_uncertainty_even_with_same_marginal_success(self):
        pairs = synthetic_fixture()
        aligned = tuple(TaskPair(p.task_id, Outcome(i < 360, .4, 1.3),
                                Outcome(i < 370, .3, 1.45)) for i, p in enumerate(pairs))
        original, changed = analyze(pairs), analyze(aligned)
        self.assertEqual(original["paired_improvement"], changed["paired_improvement"])
        self.assertLess(changed["paired_standard_error"], original["paired_standard_error"])

    def test_latency_or_forbidden_effect_blocks_canary_despite_cheap_success(self):
        pairs = synthetic_fixture()
        slow = tuple(replace(p, candidate=replace(p.candidate, first_token_seconds=1.65))
                     for p in pairs)
        self.assertFalse(analyze(slow)["gates"]["empirical_latency"])
        unsafe = list(pairs)
        unsafe[0] = replace(unsafe[0], candidate=replace(unsafe[0].candidate,
                                                      forbidden_effect=True))
        result = analyze(unsafe)
        self.assertEqual(result["decision"], "do_not_adopt")
        self.assertFalse(result["gates"]["no_observed_forbidden_effects"])
        self.assertIsNone(result["candidate"]["zero_event_upper_95"])

    def test_timeouts_remain_in_latency_denominator_and_cost(self):
        pairs = list(synthetic_fixture())
        # Five of 400 timeouts exceed the 1% tail; none has verified success.
        for index in range(395, 400):
            pairs[index] = replace(pairs[index], candidate=Outcome(False, 0.3, None))
        result = analyze(pairs)
        self.assertEqual(result["candidate"]["timeouts"], 5)
        self.assertEqual(result["candidate"]["latency_slo_failures"], 5)
        self.assertEqual(result["candidate"]["p99_seconds"], math.inf)
        self.assertEqual(result["candidate"]["total_cost"], 120)
        self.assertEqual(result["tasks"], 400)
        self.assertEqual(result["decision"], "do_not_adopt")

    def test_quality_and_cost_gates_can_independently_reject(self):
        pairs = synthetic_fixture()
        costly = tuple(replace(p, candidate=replace(p.candidate, cost=.5)) for p in pairs)
        result = analyze(costly)
        self.assertFalse(result["gates"]["cost_reduction"])
        self.assertTrue(result["gates"]["candidate_success_floor"])
        self.assertEqual(result["decision"], "do_not_adopt")
        failing = tuple(replace(p, candidate=replace(p.candidate, success=False))
                        for p in pairs)
        result = analyze(failing)
        self.assertFalse(result["gates"]["candidate_success_floor"])
        self.assertFalse(result["gates"]["paired_noninferiority"])
        self.assertEqual(result["candidate"]["total_cost"], 120)

    def test_single_timeout_is_still_a_slo_failure_when_p99_passes(self):
        pairs = list(synthetic_fixture())
        pairs[-1] = replace(pairs[-1], candidate=Outcome(False, 0.3, None))
        result = analyze(pairs)
        self.assertEqual(result["candidate"]["p99_seconds"], 1.45)
        self.assertEqual(result["candidate"]["latency_slo_failures"], 1)

    def test_zero_successes_and_free_baseline_never_create_cost_evidence(self):
        pairs = tuple(TaskPair(str(i), Outcome(False, 0, None), Outcome(False, .3, None))
                      for i in range(30))
        result = analyze(pairs)
        self.assertEqual(result["candidate"]["cost_per_success"], math.inf)
        self.assertIsNone(result["cost_reduction"])
        self.assertEqual(result["decision"], "do_not_adopt")
        free = tuple(replace(p, incumbent=replace(p.incumbent, cost=0))
                     for p in synthetic_fixture())
        self.assertFalse(analyze(free)["gates"]["cost_reduction"])

    def test_small_sample_and_zero_discordance_cannot_pass_approximation_gate(self):
        for pairs in (synthetic_fixture()[:2], synthetic_fixture()[:100]):
            result = analyze(pairs)
            self.assertFalse(result["gates"]["normal_approximation_precheck"])
            self.assertEqual(result["decision"], "do_not_adopt")
        for pairs in ((), synthetic_fixture()[:1]):
            with self.assertRaises(ValueError):
                analyze(pairs)

    def test_zero_event_bound_and_tail_miss_are_not_proofs_of_safety(self):
        result = analyze(synthetic_fixture())
        bound = result["candidate"]["zero_event_upper_95"]
        self.assertAlmostEqual((1-bound)**400, .05)
        self.assertGreater(bound, .007)
        self.assertLess(bound, .008)
        self.assertAlmostEqual(result["probability_miss_one_percent_tail"], .01795055)

    def test_immutable_ids_and_nested_outcomes_and_duplicate_rejection(self):
        pair = synthetic_fixture()[0]
        with self.assertRaises(FrozenInstanceError):
            pair.task_id = "changed"
        with self.assertRaises(FrozenInstanceError):
            pair.candidate.cost = 0
        with self.assertRaises(ValueError):
            analyze([pair, pair])
        for identity in ("", "  ", None, 12):
            with self.assertRaises(ValueError):
                TaskPair(identity, pair.incumbent, pair.candidate)
        with self.assertRaises(ValueError):
            TaskPair("id", {}, pair.candidate)
        with self.assertRaises(ValueError):
            analyze([pair, {}])

    def test_invalid_numeric_and_boolean_inputs_are_rejected(self):
        for value in (-1, math.nan, math.inf, -math.inf, True, "0.3", 10**1000):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    Outcome(False, value, 1)
                with self.assertRaises(ValueError):
                    Outcome(False, .3, value)
        with self.assertRaises(ValueError):
            Outcome(True, .3, None)
        with self.assertRaises(ValueError):
            Outcome(1, .3, 1)
        with self.assertRaises(ValueError):
            Outcome(False, .3, 1, forbidden_effect="false")

    def test_aggregate_cost_overflow_is_rejected(self):
        pairs = tuple(TaskPair(str(i), Outcome(True, 1e308, 1), Outcome(True, .3, 1))
                      for i in range(2))
        with self.assertRaises(ValueError):
            analyze(pairs)


if __name__ == "__main__":
    unittest.main()
