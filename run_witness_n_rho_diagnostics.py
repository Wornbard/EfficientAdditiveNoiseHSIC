"""Compare fitted and exact-nuisance witness corrections over saved N/rho runs.

The fixed input cohort varies only N, rho, and seed. Kernels and fitted regression
models are held fixed within each comparison. This replays saved hyperparameters;
it does not repeat the original cross-validation searches.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.polynomial.legendre import leggauss
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.kernel_ridge import KernelRidge

from multivariate_crossfit_core import (
    RunConfig, simulate_data, eps_std_fn, build_fold_masks, gaussian_gram,
    median_heuristic, matern_from_dists, residualized_krr_weight_matrix,
    standard_krr_weight_matrix, center_cross_block, centered_blocks_from_full_gram,
    compute_xi_debias_term, crossfit_v_from_blocks, crossfit_u_from_blocks,
)

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / 'tables/synthetic/witness_n_rho_diagnostics/fixed_runs.csv'
RESULT_DIR = ROOT / 'results/synthetic_witness_n_rho_diagnostics'
FIGURE_DIR = ROOT / 'figures/synthetic/witness_n_rho_diagnostics'
METHODS = ('population', 'oracle', 'plugin', 'debiased')
LABELS = {'plugin': 'Plug-in', 'debiased': 'Fitted correction',
          'exact_v': 'Correction using exact v₀'}
COLORS = {'plugin': '#666666', 'debiased': '#d66032', 'exact_v': '#2765b0'}

# Reconstruction functions retained from synthetic_witness_additional_configurations.ipynb.
def gaussian_numpy(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return np.exp(-(a[:, None] - b[None, :])**2 / alpha)


def restore_run(results_path: str) -> dict:
    """Replay a saved one-dimensional run without repeating its CV searches."""
    raw = yaml.safe_load(Path(results_path).read_text())
    config = RunConfig(**raw['config'])
    if config.dimx != 1 or config.dimy != 1:
        raise ValueError('This witness notebook requires scalar covariates and residuals.')
    state, X, signal, eps, Y = simulate_data(config)
    masks = np.asarray(build_fold_masks(config.N, config.num_folds))
    X_np, Y_np = np.asarray(X), np.asarray(Y)
    residuals, weights = [], []
    distances = jnp.abs(X[:, 0, None] - X[None, :, 0])
    for fold, mask in enumerate(masks):
        params = raw['mean_fit'][fold]
        if config.mean_kernel in ('gaussian', 'rbf'):
            kernel = RBF(length_scale=params['length_scale'])
        else:
            kernel = Matern(length_scale=params['length_scale'], nu=params['nu'])
        model = KernelRidge(alpha=params['alpha'], kernel=kernel)
        model.fit(X_np[~mask], Y_np[~mask])
        forecast = model.predict(X_np)
        heldout_mse = float(np.mean((forecast[mask] - np.asarray(signal)[mask])**2))
        np.testing.assert_allclose(heldout_mse, params['heldout_signal_mse'], rtol=2e-5, atol=2e-9)
        residuals.append((Y_np - forecast)[:, 0])
        params_v = raw['v_fit_hyperparams'][fold][0]
        Q = matern_from_dists(distances, params_v['length_scale'], params_v['nu'])
        train_kernel, eval_kernel = Q[~mask][:, ~mask], Q[mask][:, ~mask]
        weight_function = (residualized_krr_weight_matrix if config.v_fit_mode == 'residualized_centered'
                           else standard_krr_weight_matrix)
        weights.append(np.asarray(weight_function(train_kernel, eval_kernel, params_v['eta'])))
    residuals = np.asarray(residuals)
    oof_residuals = np.concatenate([residuals[k, mask] for k, mask in enumerate(masks)])
    alpha_x = float(raw['bandwidths']['alpha_x'])
    alpha_r = float(raw['bandwidths']['alpha_xi'])
    np.testing.assert_allclose(float(median_heuristic(X)), alpha_x, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(float(median_heuristic(jnp.asarray(oof_residuals[:, None]))),
                               alpha_r, rtol=2e-5, atol=2e-9)
    return {'raw': raw, 'config': config, 'state': state, 'x': X_np[:, 0],
            'eps': np.asarray(eps)[:, 0], 'masks': masks, 'residuals': residuals,
            'weights': np.asarray(weights), 'alpha_x': alpha_x, 'alpha_r': alpha_r,
            'results_path': results_path}


def verify_saved_statistics(run: dict) -> dict:
    """Check that the replay recovers the saved core operators and U/V summaries."""
    config, masks = run['config'], jnp.asarray(run['masks'])
    X = jnp.asarray(run['x'][:, None])
    residuals = jnp.asarray(run['residuals'][:, :, None])
    weights = jnp.asarray(run['weights'][:, None, :, :])
    WX = centered_blocks_from_full_gram(gaussian_gram(X, X, run['alpha_x']), masks)
    WXi = jnp.asarray([[center_cross_block(gaussian_gram(residuals[k], residuals[l], run['alpha_r']), mk, ml)
                        for l, ml in enumerate(masks)] for k, mk in enumerate(masks)])
    correction = compute_xi_debias_term(config, residuals, masks, weights, run['alpha_r'])
    stats = {}
    for method, blocks in [('naive', WX * WXi), ('debiased', WX * (WXi + correction))]:
        stats[f'{method}_hsic_v'] = float(crossfit_v_from_blocks(blocks))
        stats[f'{method}_hsic_u'] = float(crossfit_u_from_blocks(blocks))
    for name, value in stats.items():
        np.testing.assert_allclose(value, run['raw']['estimators'][name], rtol=3e-4, atol=2e-8)
    eps = jnp.asarray(run['eps'][:, None])
    oracle_blocks = WX * centered_blocks_from_full_gram(gaussian_gram(eps, eps, run['alpha_r']), masks)
    np.testing.assert_allclose(float(crossfit_v_from_blocks(oracle_blocks)),
                               run['raw']['oracle']['crossfit_hsic_v'], rtol=3e-4, atol=2e-8)
    return stats


def quadrature_geometry(run: dict, order: int) -> dict:
    """Population Gaussian kernel means and HSIC using probability quadrature."""
    config = run['config']
    nodes, weights = leggauss(order)
    u = (config.xmax - config.xmin) * nodes / 2.0 + (config.xmax + config.xmin) / 2.0
    weights = weights / 2.0
    sigma = np.asarray(eps_std_fn(run['state'], config, jnp.asarray(u[:, None])))[:, 0]
    K = gaussian_numpy(u, u, run['alpha_x'])
    row_mean = K @ weights
    Kc = K - row_mean[:, None] - row_mean[None, :] + weights @ row_mean
    Q = np.sqrt(run['alpha_r'] / (run['alpha_r'] + 2.0 * sigma[:, None]**2 + 2.0 * sigma[None, :]**2))
    norm_sq = float(weights @ ((Kc * Q) @ weights))
    return {'u': u, 'w': weights, 'sigma': sigma, 'population_norm_sq': norm_sq}


def gaussian_embedding(residuals: np.ndarray, geometry: dict, alpha_r: float) -> np.ndarray:
    denominator = alpha_r + 2.0 * geometry['sigma']**2
    return np.sqrt(alpha_r / denominator[None, :]) * np.exp(-residuals[:, None]**2 / denominator[None, :])


def operator_errors(run: dict, geometry: dict) -> dict:
    """Full-RKHS squared errors, not errors of the scalar HSIC estimates."""
    u, probabilities = geometry['u'], geometry['w']
    K = gaussian_numpy(run['x'], u, run['alpha_x'])
    oracle_embedding = gaussian_embedding(run['eps'], geometry, run['alpha_r'])
    inner = {method: 0.0 for method in METHODS[1:]}
    for fold, mask in enumerate(run['masks']):
        A = K[mask] - K[~mask].mean(axis=0)
        # The target is E[(phi_X(U)-mu_X) tensor mu_{R|U}].
        A -= (A @ probabilities)[:, None]
        residuals = run['residuals'][fold]
        embedding = gaussian_embedding(residuals, geometry, run['alpha_r'])
        derivative = (-2.0 * residuals[~mask, None]
                      / (run['alpha_r'] + 2.0 * geometry['sigma'][None, :]**2)) * embedding[~mask]
        plugin = embedding[mask] - embedding[~mask].mean(axis=0)
        debiased = plugin - residuals[mask, None] * (run['weights'][fold] @ derivative)
        oracle = oracle_embedding[mask] - oracle_embedding[~mask].mean(axis=0)
        for method, B in [('oracle', oracle), ('plugin', plugin), ('debiased', debiased)]:
            inner[method] += float(np.sum((A * B) @ probabilities)) / run['config'].N
    norms = {'oracle': run['raw']['oracle']['crossfit_hsic_v'],
             'plugin': run['raw']['estimators']['naive_hsic_v'],
             'debiased': run['raw']['estimators']['debiased_hsic_v']}
    errors = {method: norms[method] + geometry['population_norm_sq'] - 2.0 * inner[method]
              for method in METHODS[1:]}
    if min(errors.values()) < -2e-8:
        raise AssertionError(f'Negative squared operator error: {errors}')
    return {method: max(0.0, error) for method, error in errors.items()}


def checked_errors(run: dict) -> tuple[dict, dict, int]:
    """Check population and operator-error quadrature against finer resolutions."""
    previous = None
    for order in (256, 512, 1024, 2048):
        geometry = quadrature_geometry(run, order)
        if run['config'].rho == 0:
            np.testing.assert_allclose(geometry['population_norm_sq'], 0, atol=2e-10)
            geometry['population_norm_sq'] = 0.0
        errors = operator_errors(run, geometry)
        if previous is not None and np.allclose(list(errors.values()), list(previous.values()),
                                                rtol=2e-6, atol=2e-10):
            np.testing.assert_allclose(geometry['population_norm_sq'],
                run['raw']['population_oracle']['hsic'], rtol=2e-6, atol=2e-10)
            return geometry, errors, order
        previous = errors
    raise RuntimeError(f'Quadrature failed to converge: {run["results_path"]}')


def exact_v_correction(run: dict, plugin_error: float) -> dict:
    """Compute full operator error with the true Gaussian derivative nuisance.

    Its residual-coordinate witness is odd, whereas the population witness is
    even. Their RKHS inner product is therefore zero. The remaining correction
    norm and cross term are evaluated analytically using Gaussian kernel means.
    """
    x, masks, alpha = run['x'], run['masks'], run['alpha_r']
    n = len(x)
    sigma = np.asarray(eps_std_fn(run['state'], run['config'], jnp.asarray(x[:, None])))[:, 0]
    residual = np.empty(n)
    kernel = gaussian_numpy(x, x, run['alpha_x'])
    left_centered = kernel.copy()
    for fold, mask in enumerate(masks):
        residual[mask] = run['residuals'][fold, mask]
        left_centered[mask] -= kernel[~mask].mean(axis=0)
    gram_x = left_centered.copy()
    for mask in masks:
        gram_x[:, mask] -= left_centered[:, ~mask].mean(axis=1)[:, None]
    np.testing.assert_allclose(gram_x, gram_x.T, atol=1e-12)

    denominator = alpha + 2 * sigma[:, None]**2 + 2 * sigma[None, :]**2
    v_gram = 2 / denominator * np.sqrt(alpha / denominator)
    correction_norm_sq = float(np.mean(gram_x * np.outer(residual, residual) * v_gram))
    plugin_correction_inner = 0.0
    denominator = alpha + 2 * sigma[None, :]**2
    for fold, mask in enumerate(masks):
        r = run['residuals'][fold, :, None]
        values = 2 * r / denominator * np.sqrt(alpha / denominator) * np.exp(-r**2 / denominator)
        cross = values[mask] - values[~mask].mean(axis=0)
        plugin_correction_inner += float(np.sum(gram_x[mask] * cross * residual[None, :])) / n**2
    norm_sq = run['raw']['estimators']['naive_hsic_v'] + correction_norm_sq - 2 * plugin_correction_inner
    error_sq = plugin_error + correction_norm_sq - 2 * plugin_correction_inner
    if min(norm_sq, error_sq, correction_norm_sq) < -2e-10:
        raise AssertionError('Negative squared norm in the exact-nuisance diagnostic.')
    return {'operator_error_sq': max(0.0, error_sq), 'estimate_norm_sq': max(0.0, norm_sq),
            'correction_norm_sq': max(0.0, correction_norm_sq),
            'plugin_correction_inner': plugin_correction_inner}


def analyze_run(row: pd.Series, verify: bool = False) -> tuple[list[dict], dict]:
    run = restore_run(str(ROOT / row.results_path))
    for key in ('N', 'rho', 'f_smoothness', 'eps_smoothness', 'random_key'):
        assert getattr(run['config'], key) == row[key]
    replayed = verify_saved_statistics(run) if verify else None
    geometry, errors, order = checked_errors(run)
    exact = exact_v_correction(run, errors['plugin'])
    target = geometry['population_norm_sq']
    metrics = []
    for method in ('plugin', 'debiased', 'exact_v'):
        if method == 'exact_v':
            error_sq, norm_sq = exact['operator_error_sq'], exact['estimate_norm_sq']
        else:
            error_sq = errors[method]
            name = 'naive' if method == 'plugin' else 'debiased'
            norm_sq = run['raw']['estimators'][f'{name}_hsic_v']
        cosine = ((norm_sq + target - error_sq) / (2 * np.sqrt(norm_sq * target))
                  if norm_sq > 0 and target > 0 else np.nan)
        if np.isfinite(cosine) and abs(cosine) > 1 + 1e-5:
            raise AssertionError('Invalid operator alignment.')
        metrics.append({'N': int(row.N), 'rho': float(row.rho), 'random_key': int(row.random_key),
            'method': method, 'operator_error_sq': error_sq, 'population_hsic': target,
            'estimate_norm_sq': norm_sq, 'rkhs_alignment': float(np.clip(cosine, -1, 1)),
            'relative_rkhs_error': np.sqrt(error_sq / target) if target > 0 else np.nan,
            'error_ratio_to_plugin': error_sq / errors['plugin'],
            'improved_over_plugin': error_sq < errors['plugin'],
            'alpha_x': run['alpha_x'], 'alpha_r': run['alpha_r'],
            'mean_fit_mse': float(np.mean([f['heldout_signal_mse'] for f in run['raw']['mean_fit']])),
            'quadrature_order': order})
    check = {'N': int(row.N), 'rho': float(row.rho), 'random_key': int(row.random_key),
             'source_results': row.results_path,
             'source_sha256': hashlib.sha256((ROOT / row.results_path).read_bytes()).hexdigest(),
             'replayed_statistics': replayed, 'exact_v': exact}
    return metrics, check


def summarize_and_plot(frame: pd.DataFrame) -> None:
    """Show all fixed draws, paired loss ratios, and absolute operator errors."""
    summary = frame.groupby(['N', 'rho', 'method'], sort=True).agg(
        draws=('random_key', 'size'), mean_error_sq=('operator_error_sq', 'mean'),
        median_error_sq=('operator_error_sq', 'median'),
        median_paired_error_ratio=('error_ratio_to_plugin', 'median'),
        fraction_improved=('improved_over_plugin', 'mean'),
        median_alignment=('rkhs_alignment', 'median'),
    ).reset_index()
    plugin_means = summary[summary.method == 'plugin'].set_index(['N', 'rho']).mean_error_sq
    summary['ratio_of_mean_errors'] = [r.mean_error_sq / plugin_means.loc[(r.N, r.rho)]
                                       for r in summary.itertuples()]
    summary.to_csv(RESULT_DIR / 'configuration_summary.csv', index=False)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    for axis_name, fixed_name in [('N', 'rho'), ('rho', 'N')]:
        fixed_values = sorted(summary[fixed_name].unique())
        for metric, ylabel, stem in [
            ('ratio_of_mean_errors', 'Mean squared operator error / plug-in mean error', 'mean_error_ratio'),
            ('mean_error_sq', 'Mean squared operator error', 'absolute_error'),
        ]:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
            for ax, fixed in zip(axes.flat, fixed_values):
                subset = summary[summary[fixed_name] == fixed]
                methods = ('debiased', 'exact_v') if metric == 'ratio_of_mean_errors' else tuple(LABELS)
                for method in methods:
                    points = subset[subset.method == method].sort_values(axis_name)
                    ax.plot(points[axis_name], points[metric], marker='o', color=COLORS[method], label=LABELS[method])
                    if metric == 'ratio_of_mean_errors':
                        # Thin curves retain every seed; bold curves are ratios of means.
                        draws = frame[(frame[fixed_name] == fixed) & (frame.method == method)]
                        for _, draw in draws.groupby('random_key'):
                            draw = draw.sort_values(axis_name)
                            ax.plot(draw[axis_name], draw.error_ratio_to_plugin, color=COLORS[method], alpha=.15, linewidth=.7)
                if metric == 'ratio_of_mean_errors':
                    ax.axhline(1, color='0.4', linestyle=':', linewidth=1)
                ax.set_yscale('log')
                ax.set(xlabel='Sample size n' if axis_name == 'N' else 'Dependence parameter ρ', ylabel=ylabel,
                       title=f'{fixed_name} = {fixed:g}', xticks=sorted(subset[axis_name].unique()))
                ax.legend(fontsize=8)
            fig.suptitle('Fitted and exact-nuisance corrections: ten matched seeds per setting')
            for extension in ('pdf', 'png'):
                fig.savefig(FIGURE_DIR / f'{stem}_by_{axis_name}.{extension}', bbox_inches='tight')
            plt.close(fig)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    os.chdir(ROOT)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(INPUT_PATH)
    assert len(cohort) == 160
    assert not cohort.duplicated(['N', 'rho', 'random_key']).any()
    seeds = sorted(cohort.random_key.unique())
    assert len(seeds) == 10
    for _, block in cohort.groupby(['N', 'rho']):
        assert sorted(block.random_key) == seeds
    metadata = {
        'status': 'running', 'seeds': [int(s) for s in seeds],
        'selection': 'First ten seeds >=10 available at every N/rho setting; no outcome filtering.',
        'mean_smoothness': 1.0, 'noise_smoothness': .75,
        'kernel': 'Original per-run exact Gaussian kernel bandwidths, fixed within comparisons.',
        'exact_v': 'True conditional derivative feature mean; fitted m and centering unchanged.',
        'null_handling': 'Population operator is zero at rho=0; relative target error and alignment are undefined.',
        'interpretation': 'Seeds vary the Fourier functions and observations. Sample-size datasets are not nested. '
                          'Ten realizations are exploratory. Ratios compare methods within the same setting; '
                          'data-dependent bandwidths can change across settings.',
        'core_sha256': hashlib.sha256((ROOT / 'multivariate_crossfit_core.py').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest(), 'checks': [],
    }
    metrics = []
    for position, (_, row) in enumerate(cohort.iterrows(), start=1):
        # Replay full Gram statistics once for every shape and dependence setting.
        new_rows, check = analyze_run(row, verify=int(row.random_key) == seeds[0])
        metrics.extend(new_rows)
        metadata['checks'].append(check)
        pd.DataFrame(metrics).to_csv(RESULT_DIR / 'draw_metrics.csv', index=False)
        (RESULT_DIR / 'manifest.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        print(f'{position}/{len(cohort)}: n={int(row.N)}, rho={row.rho:g}, seed={int(row.random_key)}; '
              f'error ratios fitted={new_rows[1]["error_ratio_to_plugin"]:.3f}, '
              f'exact-v={new_rows[2]["error_ratio_to_plugin"]:.3f}', flush=True)
        if position % 10 == 0:
            jax.clear_caches()
            gc.collect()
    summarize_and_plot(pd.DataFrame(metrics))
    metadata['status'] = 'complete'
    (RESULT_DIR / 'manifest.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print('Completed all comparisons. Figures:', FIGURE_DIR, flush=True)


if __name__ == '__main__':
    main()
