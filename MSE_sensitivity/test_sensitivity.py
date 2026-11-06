"""Small independent checks for operator errors, resampling, pairing and coverage."""
from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np
import pandas as pd

from run_sensitivity import (bootstrap_weights, delta_intervals, interval_record, rti_from_gram,
                             scaled_rows, write_json, read_json, checkpoint_path, FACTORS)
from summarize import aggregate, flatten, wilson_interval, summarize, save_heatmaps
from multivariate_crossfit_core import (
    RunConfig, bootstrap_sqrt_hsic_ci_crossfit, blocks_to_full_matrix,
    build_fold_masks, gaussian_gram, center_cross_block,
    centered_blocks_from_full_gram, compute_xi_debias_term,
    simulate_data,
)
from run_witness_n_rho_diagnostics import operator_errors, gaussian_numpy


class SensitivityTests(unittest.TestCase):
    def test_count_bootstrap_reproduces_original_resampling(self):
        rng = np.random.default_rng(5)
        features = rng.normal(size=(20, 6))
        gram = features @ features.T
        blocks = jnp.asarray(gram.reshape(5, 4, 5, 4).transpose(0, 2, 1, 3))
        original = bootstrap_sqrt_hsic_ci_crossfit(blocks, 5, 4, 64, .05, 42)
        weights = bootstrap_weights(20, 5, 64, 42)
        np.testing.assert_array_equal(np.asarray(weights).reshape(64, 5, 4).sum(axis=2), 0)
        lower, upper, radius = rti_from_gram(gram, weights, .05)
        np.testing.assert_allclose([lower, upper, radius],
                                   [original['ci_lower'], original['ci_upper'], original['radius']],
                                   rtol=1e-12, atol=1e-12)

    def test_coverage_checks_population_target_not_zero(self):
        interval = interval_record(.2, .4, .3)
        self.assertTrue(interval['coverage'])
        self.assertFalse(interval['zero_inclusion'])
        interval = interval_record(0., .1, .3)
        self.assertFalse(interval['coverage'])
        self.assertTrue(interval['zero_inclusion'])

    def test_delta_clipping_and_null_exclusion(self):
        result = delta_intervals(-.1, .4, .09, 100, .01, .3, .05)
        self.assertEqual(result['delta_u']['lower'], 0.)
        self.assertTrue(result['delta_u']['zero_inclusion'])
        self.assertIsNone(delta_intervals(0., .4, .09, 100, 0., 0., .05)['delta_u'])
        with self.assertRaises(ValueError):
            delta_intervals(.1, 0., .09, 100, .01, .3, .05)

    def test_operator_error_matches_independent_noise_quadrature(self):
        # Integrate Gaussian noise explicitly with Hermite quadrature, independently
        # of the analytic noise embeddings used in the production error formula.
        rng = np.random.default_rng(13)
        config = RunConfig(N=20, num_folds=5, dimx=1, dimy=1, rho=.3)
        masks = np.asarray(build_fold_masks(20, 5))
        x = rng.uniform(-1, 1, 20)
        residuals = rng.normal(scale=.3, size=(5, 20))
        eps = rng.normal(scale=.3, size=20)
        weights = rng.normal(scale=.05, size=(5, 4, 16))
        ax, ar = .7, .8
        X, R, M = jnp.asarray(x[:, None]), jnp.asarray(residuals[:, :, None]), jnp.asarray(masks)
        wx = centered_blocks_from_full_gram(gaussian_gram(X, X, ax), M)
        wr = jnp.asarray([[center_cross_block(gaussian_gram(R[k], R[l], ar), mk, ml)
                          for l, ml in enumerate(M)] for k, mk in enumerate(M)])
        correction = compute_xi_debias_term(config, R, M, jnp.asarray(weights[:, None]), ar)
        norms = {'plugin': float(jnp.mean(wx * wr)), 'debiased': float(jnp.mean(wx * (wr + correction))),
                 'oracle': float(jnp.mean(wx * centered_blocks_from_full_gram(
                     gaussian_gram(jnp.asarray(eps[:, None]), jnp.asarray(eps[:, None]), ar), M)))}
        u, p = np.polynomial.legendre.leggauss(8)
        p = p / 2
        sigma = .2 * np.exp(.4 * u)
        K = gaussian_numpy(u, u, ax)
        Kc = K - (K @ p)[:, None] - (K @ p)[None, :] + p @ K @ p
        Q = np.sqrt(ar / (ar + 2 * sigma[:, None] ** 2 + 2 * sigma[None, :] ** 2))
        population_norm = float(p @ (Kc * Q) @ p)
        run = {'config': config, 'x': x, 'eps': eps, 'residuals': residuals, 'masks': masks,
               'weights': weights, 'alpha_x': ax, 'alpha_r': ar,
               'raw': {'estimators': {'naive_hsic_v': norms['plugin'], 'debiased_hsic_v': norms['debiased']},
                       'oracle': {'crossfit_hsic_v': norms['oracle']}}}
        errors = operator_errors(run, {'u': u, 'w': p, 'sigma': sigma, 'population_norm_sq': population_norm})
        h, hp = np.polynomial.hermite.hermgauss(32)
        pop_x = np.repeat(u, len(h))
        pop_r = (np.sqrt(2) * sigma[:, None] * h).ravel()
        prob = (p[:, None] * hp[None, :] / np.sqrt(np.pi)).ravel()
        KP, LP = gaussian_numpy(pop_x, pop_x, ax), gaussian_numpy(pop_r, pop_r, ar)
        KP -= (KP @ prob)[:, None] + (KP @ prob)[None, :] - prob @ KP @ prob
        numeric_target = float(prob @ (KP * LP) @ prob)
        np.testing.assert_allclose(population_norm, numeric_target, atol=1e-12)
        K = gaussian_numpy(x, pop_x, ax)
        for method in ('plugin', 'debiased', 'oracle'):
            inner = 0.
            for k, mask in enumerate(masks):
                A = K[mask] - K[~mask].mean(axis=0)
                A -= (A @ prob)[:, None]
                residual = eps if method == 'oracle' else residuals[k]
                L = gaussian_numpy(residual, pop_r, ar)
                B = L[mask] - L[~mask].mean(axis=0)
                if method == 'debiased':
                    derivative = -2 * (residual[~mask, None] - pop_r[None, :]) / ar * L[~mask]
                    B -= residual[mask, None] * (weights[k] @ derivative)
                inner += float(np.sum((A * B) @ prob)) / 20
            expected = norms[method] + numeric_target - 2 * inner
            self.assertAlmostEqual(errors[method], expected, places=11)

    def test_paired_gain_and_complete_denominators(self):
        rows = []
        for seed, plugin, debiased in ((10, 2., 1.), (11, 4., 2.)):
            intervals = {'rti': interval_record(0., 1., .5),
                         **delta_intervals(.2, .3, .01, 250, .25, .3, .05)}
            row = {'N': 250, 'f_smoothness': 1., 'eps_smoothness': .25, 'rho': .3,
                   'factor_x': 1., 'factor_residual': 1., 'random_key': seed,
                   'plugin_operator_error_sq': plugin, 'debiased_operator_error_sq': debiased,
                   'oracle_operator_error_sq': .5, 'operator_error_sq_gain': plugin - debiased,
                   'operator_error_gain': np.sqrt(plugin) - np.sqrt(debiased), 'population_hsic': .25,
                   'methods': {method: {'hsic_u': .2, 'hsic_v': .3, 'center_clipped': False, **intervals}
                               for method in ('plugin', 'debiased')}}
            rows.append(flatten(row))
        frame = pd.DataFrame(rows)
        result = aggregate(frame, expected_repetitions=2).iloc[0]
        self.assertEqual(result.operator_mse_gain, 1.5)
        self.assertAlmostEqual(result.operator_mse_gain_mcse, .5)
        self.assertEqual(result.relative_mse_gain_percent, 50.)
        self.assertEqual(result.debiased_rti_n, 2)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            aggregate(frame)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate(pd.concat([frame, frame]), expected_repetitions=4)
        frame['debiased_rti_coverage'] = frame['debiased_rti_coverage'].astype(float)
        frame.loc[0, 'debiased_rti_coverage'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Undefined'):
            aggregate(frame, expected_repetitions=2)

    def test_wilson_uncertainty_does_not_vanish_at_full_coverage(self):
        lower, upper = wilson_interval(100, 100)
        self.assertLess(lower, .97)
        self.assertAlmostEqual(upper, 1.)

    def test_scaled_evaluation_retunes_and_squares_length_factors(self):
        config = RunConfig(N=25, num_folds=5, dimx=1, dimy=1, rho=.2,
                           probe_size=40, f_modes=2, eps_modes=2, bootstrap_B=16,
                           v_bandwidth_multipliers=(1.,), v_etas=(.01,), v_nus=(1.5,))
        state, X, _, eps, _ = simulate_data(config)
        residuals = np.repeat(np.asarray(eps)[:, 0][None, :], 5, axis=0)
        residuals += np.arange(5)[:, None] * .01
        run = {'config': config, 'state': state, 'x': np.asarray(X)[:, 0],
               'eps': np.asarray(eps)[:, 0], 'residuals': residuals,
               'masks': np.asarray(build_fold_masks(25, 5)),
               'alpha_x': .7, 'alpha_r': .8}
        with patch('run_sensitivity.FACTORS', (.1, 5.)):
            rows, params = scaled_rows(run, .5, {})
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(params), 5)
        self.assertAlmostEqual(rows[0]['alpha_x'], .7 * .1**2)
        self.assertAlmostEqual(rows[1]['alpha_x'], .7 * 5.**2)
        self.assertAlmostEqual(rows[0]['alpha_residual'], .8 * .5**2)
        for row in rows:
            self.assertGreaterEqual(row['plugin_operator_error_sq'], 0.)
            self.assertGreaterEqual(row['debiased_operator_error_sq'], 0.)
            self.assertGreaterEqual(row['quadrature_order'], 512)
            self.assertIsInstance(row['methods']['debiased']['rti']['coverage'], bool)

    def test_report_outputs_and_incomplete_run_page(self):
        # Exercise reporting end to end, including 100% coverage error bars.
        with TemporaryDirectory() as directory:
            output = Path(directory)
            entries = []
            for rho in (0., .3):
                for seed in range(10, 110):
                    config = {'N': 250, 'f_smoothness': 1., 'eps_smoothness': .25,
                              'rho': rho, 'random_key': seed}
                    entry = {'run_name': f'rho{rho}_seed{seed}', 'config': config}
                    entries.append(entry)
                    target = .01 if rho else 0.
                    row = {**config, 'factor_x': 1., 'factor_residual': 1.,
                           'plugin_operator_error_sq': .02, 'debiased_operator_error_sq': .01,
                           'oracle_operator_error_sq': .005, 'operator_error_sq_gain': .01,
                           'operator_error_gain': .04, 'population_hsic': target,
                           'methods': {method: {'hsic_u': .01, 'hsic_v': .02, 'center_clipped': False,
                                               'rti': interval_record(0., .2, np.sqrt(target)),
                                               **delta_intervals(.01, .02, .001, 250, target, rho, .05)}
                                       for method in ('plugin', 'debiased')}}
                    write_json(checkpoint_path(output, entry, 'baseline'), {'row': row, 'replay_verified': True})
            write_json(output / 'manifest.json', {'entries': entries})
            summarize(output, baseline_only=True)
            table = pd.read_csv(output / 'tables/baseline/mse_and_coverage.csv')
            self.assertEqual(len(table), 2)
            self.assertTrue((table.n_repetitions == 100).all())
            self.assertTrue((output / 'report_baseline.html').exists())
            summary = pd.read_csv(output / 'tables/baseline/all_metrics.csv')
            grid = pd.concat([summary.assign(factor_x=x, factor_residual=r) for x in FACTORS for r in FACTORS])
            self.assertEqual(len(save_heatmaps(grid, output / 'figures/baseline')), 2)
            checkpoint_path(output, entries[-1], 'baseline').unlink()
            with self.assertRaisesRegex(RuntimeError, 'Incomplete cohort'):
                summarize(output, baseline_only=True)
            self.assertFalse(read_json(output / 'status_baseline.json')['complete'])
            self.assertIn('Incomplete baseline results', (output / 'report_baseline.html').read_text())


if __name__ == '__main__':
    unittest.main()
