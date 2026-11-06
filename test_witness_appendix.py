"""Small independent checks of the plotted witness evaluations."""
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import yaml
import unittest

import numpy as np

from make_witness_appendix import empirical_witnesses, population_witness, witness_source
from make_witness_slices import density_slices, normal_densities


class WitnessEvaluationTests(unittest.TestCase):
    def test_manifest_selection_does_not_depend_on_run_number(self) -> None:
        config = {"N": 500, "random_key": 10, "rho": .5, "f_smoothness": 1.,
                  "eps_smoothness": .75, "num_folds": 5, "dimx": 1, "dimy": 1}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "arbitrary_number" / "results.yaml"
            path.parent.mkdir()
            path.write_text("config: {}")
            manifest = {"jobs": [{"run_name": "arbitrary_number", "config": config}]}
            (root / "planned_jobs.yaml").write_text(yaml.safe_dump(manifest))
            self.assertEqual(witness_source(root, .75), path)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                witness_source(root, .25)
            path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Generate"):
                witness_source(root, .75)

    def test_homoskedastic_slices_all_equal_marginal(self) -> None:
        geometry = {"u": np.array([-1., 0., 1.]), "w": np.array([0.2, 0.3, 0.5]),
                    "sigma": np.full(3, 0.4)}
        x, r = np.array([-0.8, 0.7]), np.linspace(-2, 2, 51)
        values = density_slices({"alpha_x": 1.2, "alpha_r": 0.5}, geometry,
                                x, r, np.full(2, 0.4))
        expected = np.repeat(normal_densities(r, np.array([0.4])), 2, axis=0)
        np.testing.assert_allclose(values["conditional"], expected)
        np.testing.assert_allclose(values["local"], expected)
        np.testing.assert_allclose(values["marginal"], expected[0])
        np.testing.assert_allclose(values["witness"], 0, atol=1e-15)

    def test_slice_contrast_equals_operator_for_unequal_noise_scales(self) -> None:
        geometry = {"u": np.array([-1., 0.2, 1.]), "w": np.array([0.2, 0.3, 0.5]),
                    "sigma": np.array([0.2, 0.4, 0.9])}
        run = {"alpha_x": 0.7, "alpha_r": 0.3}
        x, r = np.array([-0.8, 0.7]), np.linspace(-2, 2, 51)
        values = density_slices(run, geometry, x, r, np.array([0.25, 0.7]))
        np.testing.assert_allclose(values["witness"], population_witness(run, geometry, x, r),
                                   rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(values["local_weights"].sum(axis=1), 1, atol=1e-15)

    def test_constant_conditional_noise_has_zero_population_witness(self) -> None:
        geometry = {"u": np.array([-1., 0., 1.]), "w": np.array([0.2, 0.3, 0.5]),
                    "sigma": np.full(3, 0.4)}
        values = population_witness({"alpha_x": 1.2, "alpha_r": 0.5}, geometry,
                                    np.array([-0.4, 0.3]), np.array([-1., 0., 1.]))
        np.testing.assert_allclose(values, 0, atol=1e-15)

    def test_empirical_scores_match_scalar_evaluation_and_finite_difference(self) -> None:
        rng = np.random.default_rng(23)
        run = {"x": rng.normal(size=6), "eps": rng.normal(size=6),
               "residuals": rng.normal(size=(2, 6)), "weights": rng.normal(size=(2, 3, 3)),
               "masks": np.array([[True]*3 + [False]*3, [False]*3 + [True]*3]),
               "alpha_x": 0.8, "alpha_r": 1.3, "config": SimpleNamespace(N=6)}
        x, r = np.array([-0.5, 0.7]), np.array([-0.6, 0.2, 1.])
        actual = empirical_witnesses(run, x, r)
        expected = {method: np.zeros((len(x), len(r))) for method in actual}
        h = 1e-5
        for a, xx in enumerate(x):
            k = lambda z: np.exp(-(xx-z)**2 / run["alpha_x"])
            for b, rr in enumerate(r):
                ell = lambda z: np.exp(-(rr-z)**2 / run["alpha_r"])
                for fold, mask in enumerate(run["masks"]):
                    train = np.flatnonzero(~mask)
                    for row, i in enumerate(np.flatnonzero(mask)):
                        centered_x = k(run["x"][i]) - np.mean([k(run["x"][j]) for j in train])
                        oracle = ell(run["eps"][i]) - np.mean([ell(run["eps"][j]) for j in train])
                        residual = run["residuals"][fold]
                        derivative_fit = sum(run["weights"][fold, row, t]
                                             * (ell(residual[j]+h)-ell(residual[j]-h))/(2*h)
                                             for t, j in enumerate(train))
                        corrected = (ell(residual[i]) - np.mean([ell(residual[j]) for j in train])
                                     - residual[i]*derivative_fit)
                        expected["oracle"][a, b] += centered_x * oracle / 6
                        expected["debiased"][a, b] += centered_x * corrected / 6
        for method in actual:
            np.testing.assert_allclose(actual[method], expected[method], rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
