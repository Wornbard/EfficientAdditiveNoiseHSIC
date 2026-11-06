"""Focused checks for squared-Wald intervals and safe local reruns."""
import dataclasses
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import yaml

from multivariate_crossfit_core import RunConfig, run_experiment, delta_hsic_ci_from_center
from sweep_config import load_spec, expand_sweep_grid

class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = RunConfig(
            N=50, num_folds=5, dimx=1, dimy=1, random_key=10, rho=.3,
            f_smoothness=1., eps_smoothness=.5, f_omega0=1., eps_omega0=1.,
            probe_size=100, mean_cv_folds=2, mean_alphas=(.01,),
            mean_length_scales=(1.,), mean_nus=(1.5,),
            v_bandwidth_multipliers=(1.,), v_etas=(.01,), v_nus=(1.5,),
            compute_bootstrap_intervals=False, compute_permutation_test=False,
            compute_x_permutation_test=False, compute_population_oracle=False,
            compute_v_diagnostics=False, compute_crossfit_rate_diagnostics=False)
        cls.current = run_experiment(config)
        cls.u_reference = run_experiment(dataclasses.replace(config, delta_u_variance_reference="u"))

    def test_negative_u_retained_independently_of_legacy_norm_divisor(self):
        for method in ("naive", "debiased"):
            self.assertLess(self.current["estimators"][f"{method}_hsic_u"], 0)
            for suffix in ("u", "u_loo"):
                reference_u = self.u_reference["intervals"][f"delta_{method}_{suffix}"]
                current = self.current["intervals"][f"delta_{method}_{suffix}"]
                self.assertTrue(math.isnan(reference_u["se_sqrt"]))
                for endpoint in ("sqrt_lower", "sqrt_upper", "sqrt_width"):
                    self.assertTrue(math.isfinite(current[endpoint]))
                    self.assertEqual(current[endpoint], reference_u[endpoint])
                self.assertEqual(current["sqrt_protocol"], "sqrt_squared_wald")
                self.assertEqual(current["sqrt_hsic_center"], 0.)

    def test_empty_singleton_and_positive_squared_intervals(self):
        widths = {"hsic_half_width": 1., "sqrt_half_width": float("nan")}
        empty = delta_hsic_ci_from_center(-2., widths)
        self.assertTrue(bool(empty["empty"]))
        self.assertGreater(float(empty["sqrt_lower"]), float(empty["sqrt_upper"]))
        self.assertEqual(float(empty["sqrt_width"]), 0.)
        singleton = delta_hsic_ci_from_center(-1., widths)
        self.assertFalse(bool(singleton["empty"]))
        self.assertEqual(float(singleton["sqrt_lower"]), 0.)
        self.assertEqual(float(singleton["sqrt_upper"]), 0.)
        positive = delta_hsic_ci_from_center(2., widths)
        self.assertFalse(bool(positive["empty"]))
        self.assertEqual(float(positive["sqrt_lower"]), 1.)
        self.assertAlmostEqual(float(positive["sqrt_upper"]), math.sqrt(3.))

    def test_v_centered_results_and_point_estimates_do_not_change(self):
        self.assertEqual(self.current["estimators"], self.u_reference["estimators"])
        for method in ("naive", "debiased"):
            a=self.current["intervals"][f"delta_{method}_v"]
            b=self.u_reference["intervals"][f"delta_{method}_v"]
            for key in ("hsic_ref","sqrt_lower","sqrt_upper"):
                self.assertEqual(a[key],b[key])

    def test_plan_rejects_conflicting_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd=[sys.executable,"run_synthetic.py","--config","configs/smoke.yaml",
                 "--output",tmp,"--dry-run"]
            subprocess.run(cmd,check=True,capture_output=True)
            path=Path(tmp)/"planned_jobs.yaml"
            plan=yaml.safe_load(path.read_text())
            plan["jobs"][0]["config"]["random_key"]=999
            path.write_text(yaml.safe_dump(plan))
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("different plan",result.stderr)

    def test_paper_grid_sizes(self):
        self.assertEqual(len(expand_sweep_grid(load_spec(Path("configs/synthetic.yaml"))["sweep"])),16200)
        self.assertEqual(len(expand_sweep_grid(load_spec(Path("configs/split.yaml"))["sweep"])),720)

if __name__ == "__main__":
    unittest.main()
