"""Regression checks for Part I's worked calculations and counterexamples.

These are CPU semantic/arithmetic checks, not training or performance evidence.
"""
import math
import unittest

from examples.attention import attention
from examples.sequence_models import masked_token_loss


class FoundationChecks(unittest.TestCase):
    def test_cross_entropy_gradient(self):
        logits, target, step = [0.3, -0.2, 0.1], 1, 1e-5
        weights = [math.exp(x) for x in logits]
        for i in range(len(logits)):
            plus, minus = logits.copy(), logits.copy()
            plus[i] += step
            minus[i] -= step
            numeric = (masked_token_loss([plus], [target], [1]) -
                       masked_token_loss([minus], [target], [1])) / (2*step)
            analytic = weights[i] / sum(weights) - (i == target)
            self.assertAlmostEqual(numeric, analytic, places=8)
        self.assertAlmostEqual(1/(1+math.exp(-0.1)), 0.5249791875)

    def test_attention_row_masks_before_normalization(self):
        for shard in [1, 2, 3]:
            result = attention([1.], [[0.], [math.log(3.)], [100.]],
                               [[2.], [6.], [999.]], [True, True, False], shard)
            self.assertAlmostEqual(result[0], 5.)

    def test_permutation_requires_permuting_structural_mask(self):
        x, values, order = [[0.], [1.], [2.]], [[2.], [5.], [9.]], [2, 0, 1]
        permuted_x, permuted_values = [x[i] for i in order], [values[i] for i in order]
        full = [attention(q, x, values)[0] for q in x]
        permuted_full = [attention(q, permuted_x, permuted_values)[0] for q in permuted_x]
        for actual, i in zip(permuted_full, order):
            self.assertAlmostEqual(actual, full[i])
        causal = [attention(q, x, values, [j <= i for j in range(3)])[0]
                  for i, q in enumerate(x)]
        fixed_mask = [attention(q, permuted_x, permuted_values,
                                [j <= i for j in range(3)])[0]
                      for i, q in enumerate(permuted_x)]
        self.assertNotAlmostEqual(fixed_mask[0], causal[order[0]])
        for row, original_i in enumerate(order):
            moved_mask = [original_j <= original_i for original_j in order]
            result = attention(permuted_x[row], permuted_x, permuted_values, moved_mask)[0]
            self.assertAlmostEqual(result, causal[original_i])

    def test_token_mean_is_not_rank_mean(self):
        # At a point with per-target gradient contributions 4, 0, 0, 0.
        contributions = [[4.], [0., 0., 0.]]
        local_means = [sum(g)/len(g) for g in contributions]
        self.assertEqual(sum(local_means)/2, 2.)
        count = sum(map(len, contributions))
        scaled_local = [2*sum(g)/count for g in contributions]
        self.assertEqual(sum(scaled_local)/2, 1.)
        self.assertEqual(sum(map(sum, contributions))/count, 1.)

    def test_clipping_is_not_an_adam_update_bound(self):
        gradient, threshold = [3., 4.], 2.
        norm = math.hypot(*gradient)
        clipped = [g*min(1., threshold/norm) for g in gradient]
        for actual, expected in zip(clipped, [1.2, 1.6]):
            self.assertAlmostEqual(actual, expected)
        # One-coordinate, first-step Adam with a much smaller clipping cap.
        g, c, lr, beta1, beta2 = 2., 0.1, 0.1, 0.9, 0.99
        g = min(g, c)
        m, v = (1-beta1)*g, (1-beta2)*g*g
        update = lr*(m/(1-beta1))/(math.sqrt(v/(1-beta2))+1e-8)
        self.assertGreater(update, lr*c)
        self.assertAlmostEqual(math.sqrt(3**2+4**2), 5.)
        self.assertNotEqual(math.sqrt(2*(3**2+4**2)), 5.)  # Duplicate replicas.

    def test_cost_and_time_ratios(self):
        self.assertAlmostEqual(1/0.4, 2.5)
        self.assertAlmostEqual(1.2/0.55, 2.1818181818)
        self.assertAlmostEqual(0.8*1.3, 1.04)
        self.assertAlmostEqual(0.82*1.25, 1.025)

    def test_resource_floors_and_queue(self):
        self.assertEqual(max(1e12/1e14, 1e11/1e12), 0.1)
        self.assertEqual(max(1e12/2e14, 1e11/1e12), 0.1)
        self.assertEqual(1/(10-5), 0.2)
        self.assertEqual(1/(10-9), 1.)
        width, intermediate = 4096, 4*4096
        for sequence, pair_factor in [(8*width, 4), (16*width, 2)]:
            self.assertEqual(pair_factor*sequence**2*width,
                             8*sequence*width**2+6*sequence*width*intermediate)

    def test_normalized_scaling_optimum(self):
        def loss(n):
            return 4/n + n/16
        self.assertEqual(loss(8), 1.)
        self.assertEqual(loss(4), 1.25)
        self.assertGreater(loss(7.9), loss(8))
        self.assertGreater(loss(8.1), loss(8))
        self.assertAlmostEqual(-4/(8**2)+1/16, 0.)

    def test_paired_effect_uncertainty(self):
        differences = [1.]*15 + [-1.]*5 + [0.]*80
        mean = sum(differences)/len(differences)
        variance = sum((x-mean)**2 for x in differences)/(len(differences)-1)
        se = math.sqrt(variance/len(differences))
        self.assertAlmostEqual(mean, 0.1)
        self.assertAlmostEqual(variance, 19/99)
        self.assertAlmostEqual(se, 0.0438085827)
        self.assertLess(mean-1.96*se, 0.05)
        self.assertGreater(mean-1.96*se, 0.)


if __name__ == "__main__":
    unittest.main()
