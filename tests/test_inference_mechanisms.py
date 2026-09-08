import random
import unittest
from examples.inference_mechanisms import acceptance_and_residual, speculative_output_mass, symmetric_quantize


class InferenceMechanismTests(unittest.TestCase):
    def test_exact_mass_with_missing_support(self):
        for p, q in [([.5,.3,.2],[.2,.7,.1]), ([1.,0.],[0.,1.]), ([.5,.5],[.5,.5])]:
            for actual, expected in zip(speculative_output_mass(p,q), p):
                self.assertAlmostEqual(actual, expected)

    def test_random_distribution_mass(self):
        rng = random.Random(7)
        for _ in range(30):
            p, q = [rng.random() for _ in range(7)], [rng.random() for _ in range(7)]
            p, q = [x/sum(p) for x in p], [x/sum(q) for x in q]
            for actual, expected in zip(speculative_output_mass(p,q), p):
                self.assertAlmostEqual(actual, expected)

    def test_impossible_proposal_and_equal_distribution(self):
        with self.assertRaises(ValueError):
            acceptance_and_residual([1.,0.], [0.,1.], 0)
        self.assertEqual(acceptance_and_residual([.5,.5], [.5,.5], 1), (1.,None))

    def test_quantization_error_and_zero(self):
        values = [-3., -.13, 0., .2, 1.1, 3.]
        _, scale, restored = symmetric_quantize(values)
        self.assertLessEqual(max(abs(x-y) for x,y in zip(values,restored)), scale/2+1e-12)
        self.assertEqual(symmetric_quantize([0.,0.])[2], [0.,0.])

    def test_outlier_cost(self):
        self.assertEqual(symmetric_quantize([.1,.2,10.])[2][:2], [0.,0.])
        self.assertNotEqual(symmetric_quantize([.1,.2])[2], [0.,0.])
