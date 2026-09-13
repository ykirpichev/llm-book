"""Independent arithmetic checks for the book's declared running workload."""
import unittest


class ResourceModelTests(unittest.TestCase):
    def test_running_model_kv_budget(self):
        per_token = 2 * 32 * 8 * 128 * 2
        self.assertEqual(per_token, 128 * 1024)
        self.assertEqual(2000 * per_token / 2**20, 250)
        self.assertEqual(2000 * per_token / 4 / 2**20, 62.5)
        self.assertEqual(2000 * per_token / 2 / 2**20, 125)

    def test_weight_and_transfer_floors(self):
        self.assertAlmostEqual(14e9 / 3e12 * 1000, 4.6666666667, places=8)
        self.assertAlmostEqual(2000 * 131072 / 50e9 * 1000, 5.24288)

    def test_decode_attention_kv_read_ledger(self):
        # Eight sequences, eight KV heads, 4K retained tokens, 128-wide
        # heads, K and V, and two bytes per scalar.
        per_layer = 8 * 8 * 4096 * 128 * 2 * 2
        self.assertEqual(per_layer, 128 * 2**20)
        self.assertEqual(32 * per_layer, 4 * 2**30)
        self.assertAlmostEqual(per_layer / 3e12 * 1000, 0.0447392427)

    def test_dense_training_storage_convention(self):
        # Weights + grads FP16, master weights and two moments FP32.
        self.assertEqual(7e9 * (2 + 2 + 4 + 4 + 4) / 1e9, 112)
