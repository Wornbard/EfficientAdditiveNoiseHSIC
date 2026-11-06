"""Focused checks for cross-U projections, splitting, and confidence-set handling."""
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from run_cross_u_benchmark import (balanced_split, cross_u_statistics, cross_u_interval,
                                  interval_metrics, evaluate_run, file_hash, delta_squared_u_interval,
                                  delta_squared_u_from_gram, DELTA_SQUARED_U)


class CrossUBenchmarkTests(unittest.TestCase):
    def test_negative_u_is_not_clipped_before_constructing_the_set(self):
        negative = delta_squared_u_interval(-2., 9., 100, 0.)
        zero = delta_squared_u_interval(0., 9., 100, 0.)
        self.assertEqual(negative["estimate"], -2.)
        self.assertAlmostEqual(negative["se_hsic"], .6)
        self.assertTrue(negative["empty"])
        self.assertFalse(negative["coverage"])
        self.assertTrue(negative["ci_excludes_zero"])
        self.assertEqual(negative["width_sqrt_hsic"], 0.)
        self.assertTrue(zero["coverage"])
        self.assertGreater(zero["width_sqrt_hsic"], 0.)
        singleton = delta_squared_u_interval(0., 0., 100, 0.)
        self.assertFalse(singleton["empty"])
        self.assertTrue(singleton["coverage"])
        self.assertEqual(singleton["width_sqrt_hsic"], 0.)

    def test_delta_is_signed_u_wald_then_sqrt_not_sqrt_scale_wald(self):
        result = delta_squared_u_interval(4., 9., 100, 4.)
        radius = 1.959963984540054 * .6
        self.assertAlmostEqual(result["lower_hsic"], 4. - radius)
        self.assertAlmostEqual(result["upper_hsic"], 4. + radius)
        self.assertAlmostEqual(result["width_sqrt_hsic"], math.sqrt(4. + radius)-math.sqrt(4. - radius))
        self.assertTrue(result["coverage"])
        self.assertTrue(result["ci_excludes_zero"])
        with self.assertRaisesRegex(ValueError, "nonnegative score variance"):
            delta_squared_u_interval(1., -1., 100, 1.)

    def test_fresh_gram_delta_matches_scalar_projection_variance(self):
        # Two folds give U=-7/4. Score projections are features times their mean.
        features = np.array([3., -2., 1., -1.])
        result = delta_squared_u_from_gram(np.outer(features, features), 2, 0.)
        self.assertAlmostEqual(result["estimate"], -1.75)
        self.assertAlmostEqual(result["score_variance"], features.mean()**2*features.var(ddof=0))
        self.assertAlmostEqual(result["se_hsic"]**2, 4*features.mean()**2*features.var(ddof=0)/4)
        self.assertTrue(result["empty"])
        self.assertFalse(result["coverage"])

    def test_gram_statistics_match_explicit_hilbert_features(self):
        rng = np.random.default_rng(19)
        features = rng.normal(size=(40, 7)) + np.arange(7) / 10
        masks = np.repeat(np.eye(5, dtype=bool), 8, axis=1)
        left, right = balanced_split(masks, 6)
        result = cross_u_statistics(features @ features.T, left, right)
        mean_a, mean_b = features[left].mean(axis=0), features[right].mean(axis=0)
        expected_se = np.linalg.norm((features[left] - mean_a) @ mean_b) / math.sqrt(20 * 19)
        self.assertAlmostEqual(result["estimate"], float(mean_a @ mean_b))
        self.assertAlmostEqual(result["se_a"], expected_se)
        inflated = cross_u_interval(result, 0.1, math.sqrt(2))
        self.assertAlmostEqual(inflated["se"] ** 2, 2 * expected_se ** 2)

    def test_split_is_balanced_disjoint_complete_and_reproducible(self):
        masks = np.repeat(np.eye(5, dtype=bool), 10, axis=1)
        left, right = balanced_split(masks, 42)
        np.testing.assert_array_equal(np.sort(np.concatenate((left, right))), np.arange(50))
        np.testing.assert_array_equal(masks[:, left].sum(axis=1), np.full(5, 5))
        np.testing.assert_array_equal(masks[:, right].sum(axis=1), np.full(5, 5))
        np.testing.assert_array_equal(left, balanced_split(masks, 42)[0])
        self.assertFalse(np.array_equal(left, balanced_split(masks, 43)[0]))

    def test_empty_confidence_set_does_not_artificially_cover_zero(self):
        result = interval_metrics(-2, -1, 0)
        self.assertTrue(result["empty"])
        self.assertFalse(result["coverage"])
        self.assertTrue(result["ci_excludes_zero"])
        self.assertEqual(result["width_hsic"], 0)
        self.assertEqual(result["width_sqrt_hsic"], 0)
        nonempty = interval_metrics(-1, 4, 0)
        self.assertTrue(nonempty["coverage"])
        self.assertEqual(nonempty["width_sqrt_hsic"], 2)

    def test_degenerate_studentizer_and_invalid_groups_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "studentizer"):
            cross_u_statistics(np.ones((4, 4)), np.array([0, 1]), np.array([2, 3]))
        with self.assertRaisesRegex(ValueError, "partition"):
            cross_u_statistics(np.eye(4), np.array([0, 1]), np.array([1, 3]))
        with self.assertRaisesRegex(ValueError, "even size"):
            balanced_split(np.repeat(np.eye(2, dtype=bool), 5, axis=1), 0)

    def test_rank_one_oracle_null_has_expected_calibration(self):
        # For Gaussian scalar scores, the uninflated pivot is exactly signed t.
        # This independently checks null versus alternative studentizer scaling.
        rng = np.random.default_rng(704)
        features = rng.normal(size=(6000, 80))
        a, b = features[:, :40], features[:, 40:]
        estimates = a.mean(axis=1) * b.mean(axis=1)
        se = a.std(axis=1, ddof=1) * np.abs(b.mean(axis=1)) / math.sqrt(40)
        pivots = estimates / se
        self.assertAlmostEqual(float(np.mean(np.abs(pivots) <= 1.95996398454)), .95, delta=.015)
        self.assertGreater(float(np.mean(np.abs(pivots) <= math.sqrt(2) * 1.95996398454)), .985)
        # Validate a subset through the actual Gram implementation as well.
        for i in range(3):
            stats = cross_u_statistics(np.outer(features[i], features[i]), np.arange(40), np.arange(40, 80))
            self.assertAlmostEqual(stats["pivot_null"], pivots[i])

    def test_older_bootstrap_schema_and_undefined_delta_interval(self):
        rng = np.random.default_rng(9)
        features = rng.normal(size=(20, 3))
        gram = features @ features.T
        raw = {
            "config": {"N": 20, "num_folds": 5, "rho": 0., "random_key": 10,
                       "f_smoothness": 1., "eps_smoothness": .75},
            "bandwidths": {"alpha_x": 1., "alpha_xi": 1.},
            "intervals": {"bootstrap_debiased": {"ci_lower": 0., "ci_upper": .3}},
        }
        for kind in ("u", "v"):
            raw["intervals"][f"delta_debiased_{kind}"] = {
                "hsic_lower": 0., "hsic_upper": .2, "sqrt_lower": None, "sqrt_upper": None}
        with TemporaryDirectory() as folder:
            source = Path(folder) / "results.yaml"
            source.write_text(yaml.safe_dump(raw))
            with patch("run_cross_u_benchmark.restore_score_grams",
                       return_value={name: gram for name in ("plugin", "debiased", "true_residual")}):
                result = evaluate_run({"source": str(source), "sha256": file_hash(source)})
        self.assertIsNone(result["methods"]["rti_debiased"]["p_value_right_tail"])
        self.assertFalse(result["methods"]["rti_debiased"]["reject_right_tail"])
        self.assertTrue(result["methods"]["dm_debiased_u_sqrt"]["undefined"])
        self.assertFalse(result["methods"][DELTA_SQUARED_U]["undefined"])


if __name__ == "__main__":
    unittest.main()
