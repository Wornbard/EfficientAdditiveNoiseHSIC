"""Check archived-statistic reconstruction independently of table aggregation."""
import unittest
import numpy as np
import pandas as pd
from paper_plot_utils import reconstruct_robust_delta_intervals, Z975

def example_rows():
    rows = []
    for u in (-.2, -.04, 0., .09):
        row = {}
        for estimator in ("naive", "debiased"):
            row[f"estimators.{estimator}_hsic_u"] = u
            row[f"estimators.{estimator}_hsic_v"] = .25
            for suffix in ("u", "u_loo"):
                prefix = f"intervals.delta_{estimator}_{suffix}"
                row[f"{prefix}.score_var"] = .04
                row[f"{prefix}.n"] = 100.0
                row[f"{prefix}.sqrt_lower"] = np.nan
                row[f"{prefix}.sqrt_upper"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)

class RobustDeltaReportingTests(unittest.TestCase):
    def test_negative_u_and_empty_sets_are_retained(self):
        source = example_rows()
        result = reconstruct_robust_delta_intervals(source)
        self.assertEqual(len(result), 4)
        u = np.array([-.2, -.04, 0., .09])
        half_width = Z975 * .04
        nonempty = u + half_width >= 0
        for estimator in ("naive", "debiased"):
            for suffix in ("u", "u_loo"):
                prefix = f"intervals.delta_{estimator}_{suffix}"
                np.testing.assert_allclose(result[f"{prefix}.sqrt_lower"], np.where(nonempty, np.sqrt(np.maximum(u-half_width, 0)), 1))
                np.testing.assert_allclose(result[f"{prefix}.sqrt_upper"], np.where(nonempty, np.sqrt(np.maximum(u+half_width, 0)), 0))
                np.testing.assert_array_equal(result[f"{prefix}.empty"], [True, False, False, False])
                np.testing.assert_array_equal(result[f"estimators.{estimator}_hsic_u"], source[f"estimators.{estimator}_hsic_u"])
        self.assertTrue(source["intervals.delta_debiased_u.sqrt_lower"].isna().all())
        from paper_plot_utils import build_ci_frame
        result["population_oracle.sqrt_hsic"] = 0.0
        result["N"] = 100
        ci = build_ci_frame(result)
        self.assertEqual(len(ci), 16)
        empty = ci.loc[ci["lower"] > ci["upper"]]
        self.assertEqual(len(empty), 4)
        self.assertFalse(empty["coverage"].any())
        self.assertFalse(empty["contains_zero"].any())
        self.assertTrue(empty["width"].eq(0).all())

    def test_bad_inputs_fail_instead_of_dropping_rows(self):
        for column, value in [("estimators.debiased_hsic_u",np.nan), ("intervals.delta_naive_u.score_var",-1.),
                              ("intervals.delta_debiased_u.n",np.nan)]:
            source=example_rows()
            source.loc[0,column]=value
            with self.assertRaisesRegex(ValueError,"no rows were discarded"):
                reconstruct_robust_delta_intervals(source)

    def test_v_is_not_needed_for_squared_wald_interval(self):
        source=example_rows()
        columns=[column for column in source if column.endswith("_hsic_v")]
        with_v=reconstruct_robust_delta_intervals(source)
        without_v=reconstruct_robust_delta_intervals(source.drop(columns=columns))
        for endpoint in ("sqrt_lower", "sqrt_upper"):
            np.testing.assert_array_equal(with_v[f"intervals.delta_debiased_u.{endpoint}"], without_v[f"intervals.delta_debiased_u.{endpoint}"])

    def test_matches_fresh_core_intervals(self):
        from pathlib import Path
        from multivariate_crossfit_core import RunConfig, run_experiment
        from sweep_config import load_spec, expand_sweep_grid
        from load_experiment_results import _flatten_scalars
        config = expand_sweep_grid(load_spec(Path("configs/smoke.yaml"))["sweep"])[0]
        config.update(compute_bootstrap_intervals=False, compute_permutation_test=False,
                      compute_x_permutation_test=False, compute_population_oracle=False,
                      delta_u_variance_reference="v")
        result = run_experiment(RunConfig(**config))
        reconstructed = reconstruct_robust_delta_intervals(pd.DataFrame([_flatten_scalars(result)]))
        for estimator in ("naive", "debiased"):
            for suffix in ("u", "u_loo"):
                method=f"delta_{estimator}_{suffix}"
                for endpoint in ("sqrt_lower", "sqrt_upper"):
                    self.assertAlmostEqual(reconstructed[f"intervals.{method}.{endpoint}"].iloc[0],
                                           result["intervals"][method][endpoint], places=12)

    def test_overflow_fails_explicitly(self):
        source=example_rows()
        source.loc[0,"intervals.delta_naive_u.n"]=np.nextafter(0.,1.)
        with np.errstate(over="ignore",divide="ignore"):
            with self.assertRaisesRegex(ValueError,"no rows were discarded"):
                reconstruct_robust_delta_intervals(source)

if __name__ == "__main__":
    unittest.main()
