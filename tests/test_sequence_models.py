import math
import unittest
from examples.sequence_models import bpe_pieces, masked_token_loss, delta_step


class SequenceMechanismTests(unittest.TestCase):
    def test_ranked_merges(self):
        self.assertEqual(bpe_pieces("low low", [(b"l", b"o"), (b"lo", b"w")]),
                         [b"low", b" ", b"low"])

    def test_bytes_roundtrip_and_empty(self):
        for text in ["", "café", "猫🙂", "  x\n", "e\u0301"]:
            self.assertEqual(b"".join(bpe_pieces(text, [])).decode("utf-8"), text)

    def test_masked_loss_and_numerical_stability(self):
        self.assertAlmostEqual(masked_token_loss([[1000, 1000], [0, 100]], [0, 0], [1, 0]), math.log(2))
        with self.assertRaises(ValueError):
            masked_token_loss([[0, 0]], [0], [0])

    def test_delta_selective_overwrite(self):
        state = [[2., 9.]]
        updated, out = delta_step(state, [1., 0.], [5.], [1., 0.])
        self.assertEqual(updated, [[5., 9.]])
        self.assertEqual(out, [5.])
        self.assertEqual(state, [[2., 9.]])

    def test_ignored_target_sentinel(self):
        self.assertAlmostEqual(
            masked_token_loss([[0, 0], [1, 2]], [0, -100], [1, 0]),
            math.log(2))
        for target in [-100, 2, 0.5]:
            with self.assertRaises(ValueError):
                masked_token_loss([[0, 0]], [target], [1])
        with self.assertRaises(ValueError):
            masked_token_loss([[0, 0], [float('nan'), 0]], [0, -100], [1, 0])

    def test_decay_then_correction(self):
        updated, _ = delta_step([[2., 9.]], [1., 0.], [5.], [0., 1.], alpha=.5, beta=.5)
        self.assertEqual(updated, [[3., 4.5]])

    def test_delta_replay_and_invalid_input(self):
        state = [[0., 0.]]
        saved = state
        state, _ = delta_step(state, [1., 0.], [3.], [1., 0.])
        replay, _ = delta_step(saved, [1., 0.], [3.], [1., 0.])
        self.assertEqual(state, replay)
        with self.assertRaises(ValueError):
            delta_step(state, [1.], [3.], [1.])
