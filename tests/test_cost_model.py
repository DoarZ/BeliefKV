import unittest

from beliefkv.policy.cost_model import PCIeCostModel, estimate_kv_bytes_per_token


class CostModelTest(unittest.TestCase):
    def test_qwen_like_kv_size_per_token(self):
        value = estimate_kv_bytes_per_token(
            num_layers=28,
            hidden_size=3584,
            num_attention_heads=28,
            num_kv_heads=4,
            dtype_bytes=2,
        )
        self.assertEqual(value, 57344)

    def test_transfer_time_is_positive(self):
        model = PCIeCostModel(bandwidth_gbps=24.0, overhead_ms=0.08)
        self.assertGreater(model.transfer_ms(57344 * 4096), 0.08)


if __name__ == "__main__":
    unittest.main()
