import unittest

from beliefkv.policy.transfer_cost import PCIeCostModel


class PCIeCostModelTest(unittest.TestCase):
    def test_transfer_time_uses_decimal_gigabytes_per_second(self):
        model = PCIeCostModel(bandwidth_gbps=24.0, overhead_ms=0.08)
        self.assertAlmostEqual(model.transfer_ms(240_000_000), 10.08)

    def test_empty_transfer_has_no_cost(self):
        self.assertEqual(PCIeCostModel().transfer_ms(0), 0.0)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            PCIeCostModel(bandwidth_gbps=0)
        with self.assertRaises(ValueError):
            PCIeCostModel(overhead_ms=-1)


if __name__ == "__main__":
    unittest.main()
