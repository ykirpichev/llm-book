import math
import unittest
from examples.post_training import dpo_loss, group_advantages, clipped_surrogate, sequence_ratio, pass_at_k


class PostTrainingTests(unittest.TestCase):
    def test_dpo_reference_match_and_direction(self):
        self.assertAlmostEqual(dpo_loss(-2, -4, -2, -4), math.log(2))
        self.assertLess(dpo_loss(-1, -4, -2, -4), math.log(2))
        self.assertGreater(dpo_loss(-3, -4, -2, -4), math.log(2))

    def test_dpo_extremes(self):
        self.assertTrue(math.isfinite(dpo_loss(-10000, 0, 0, -10000)))

    def test_group_advantages(self):
        self.assertEqual(group_advantages([1., 1., 0., 0.]), [1., 1., -1., -1.])
        self.assertEqual(group_advantages([1., 1.]), [0., 0.])
        self.assertEqual(group_advantages([1., 1., 0., 0.], normalize=False), [.5, .5, -.5, -.5])

    def test_clipping_both_signs(self):
        self.assertAlmostEqual(clipped_surrogate(1.4, 2.), 2.4)
        self.assertAlmostEqual(clipped_surrogate(.6, -2.), -1.6)
        self.assertAlmostEqual(clipped_surrogate(1.4, -2.), -2.8)

    def test_sequence_ratio_is_geometric_not_arithmetic(self):
        ratio = sequence_ratio([math.log(.8), math.log(.2)], [math.log(.4), math.log(.4)])
        self.assertAlmostEqual(ratio, 1.)

    def test_pass_at_k(self):
        self.assertAlmostEqual(pass_at_k(10, 2, 3), 8/15)
        self.assertEqual(pass_at_k(10, 0, 3), 0)
        self.assertEqual(pass_at_k(10, 9, 3), 1)
        with self.assertRaises(ValueError):
            pass_at_k(2, 1, 3)
