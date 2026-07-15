import unittest

from beliefkv.metrics.summary import (
    bootstrap_mean_ci,
    jain_fairness,
    mean,
    percentile,
)


class SummaryMetricsTest(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([0, 10], 25), 2.5)
        self.assertEqual(percentile([0, 10], 50), 5.0)
        self.assertEqual(percentile([0, 10], 100), 10.0)

    def test_bootstrap_interval_is_deterministic_and_contains_mean(self):
        first = bootstrap_mean_ci([1, 2, 3, 4], resamples=200, seed=9)
        second = bootstrap_mean_ci([1, 2, 3, 4], resamples=200, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(first.estimate, mean([1, 2, 3, 4]))
        self.assertLessEqual(first.lower, first.estimate)
        self.assertGreaterEqual(first.upper, first.estimate)

    def test_jain_fairness_handles_equal_and_skewed_allocations(self):
        self.assertEqual(jain_fairness([2, 2, 2]), 1.0)
        self.assertLess(jain_fairness([1, 1, 10]), 0.6)
        with self.assertRaises(ValueError):
            jain_fairness([1, -1])

    def test_non_finite_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            percentile([1, float("inf")], 50)


if __name__ == "__main__":
    unittest.main()
