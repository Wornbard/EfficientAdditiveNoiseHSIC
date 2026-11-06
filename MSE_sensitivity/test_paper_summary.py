"""Checks for paired Monte Carlo uncertainty and RTI reconstruction."""
import unittest

import numpy as np

from make_paper_summary import paired_mse_summary, rti_interval_from_width


class PaperSummaryTests(unittest.TestCase):
    def test_gain_is_ratio_of_mean_losses(self):
        result = paired_mse_summary(np.array([1., 9.]), np.array([1., 1.]), np.array([[.5, .5]]))
        self.assertEqual(result['gain_percent'], 80.)
        self.assertEqual(result['plugin_mse'], 5.)
        self.assertEqual(result['debiased_mse'], 1.)

    def test_paired_resampling_preserves_exact_relative_gain(self):
        rng = np.random.default_rng(10)
        plugin = np.exp(rng.normal(size=100))
        weights = rng.multinomial(100, np.full(100, .01), size=500) / 100
        result = paired_mse_summary(plugin, .5 * plugin, weights)
        np.testing.assert_allclose([result['gain_percent'], result['gain_percent_lower'],
                                   result['gain_percent_upper']], 50., atol=1e-12)
        self.assertLess(result['plugin_mse_lower'], result['plugin_mse_upper'])

    def test_invalid_losses_fail_instead_of_dropping_repetitions(self):
        for values in ([1., np.nan], [1., -1.], [0., 0.]):
            with self.assertRaises(ValueError):
                paired_mse_summary(np.array(values), np.ones(2), np.array([[.5, .5]]))

    def test_rti_endpoints_recovered_with_zero_lower_bound(self):
        centers = np.array([1., 2., 0.])
        radii = np.array([2., .5, .3])
        lower = np.maximum(centers - radii, 0.)
        upper = centers + radii
        lo, hi = rti_interval_from_width(centers ** 2, upper - lower)
        np.testing.assert_allclose(lo, lower)
        np.testing.assert_allclose(hi, upper)


if __name__ == '__main__':
    unittest.main()
