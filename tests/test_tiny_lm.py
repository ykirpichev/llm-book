import math
import unittest
from examples.tiny_lm import Bigram


class TinyLMTests(unittest.TestCase):
    def test_training_and_generation(self):
        model = Bigram("ab")
        sequences = ["ababa", "babab"]
        self.assertAlmostEqual(model.loss_and_gradient(sequences)[0], math.log(2))
        model.fit(sequences)
        self.assertLess(model.loss_and_gradient(sequences)[0], 0.01)
        self.assertEqual(model.generate("a"), "abababa")
        self.assertEqual(model.generate("a", max_new_tokens=0), "a")

    def test_gradient_against_finite_difference(self):
        model = Bigram("ab")
        model.logits = [[0.3, -0.4], [0.1, 0.7]]
        sequences = ["aabab"]
        _, gradient = model.loss_and_gradient(sequences)
        epsilon = 1e-5
        for i in range(2):
            for j in range(2):
                original = model.logits[i][j]
                model.logits[i][j] = original + epsilon
                high = model.loss_and_gradient(sequences)[0]
                model.logits[i][j] = original - epsilon
                low = model.loss_and_gradient(sequences)[0]
                model.logits[i][j] = original
                self.assertAlmostEqual(gradient[i][j], (high-low)/(2*epsilon), places=8)

    def test_checkpoint_roundtrip(self):
        model = Bigram("ab").fit(["abab", "baba"])
        restored = Bigram.from_json(model.to_json())
        self.assertEqual(model.probabilities("a"), restored.probabilities("a"))
        with self.assertRaises(ValueError):
            Bigram.from_json('{"format": 1, "vocabulary": ["a"], "logits": [[NaN]]}')

    def test_invalid_data(self):
        model = Bigram("ab")
        with self.assertRaises(KeyError):
            model.loss_and_gradient(["ac"])
        with self.assertRaises(ValueError):
            model.loss_and_gradient(["a", "b"])
        with self.assertRaises(ValueError):
            model.generate("")
