from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

# Match the notebook's numerical setup.
jax.config.update("jax_enable_x64", True)


@dataclass
class RunConfig:
    N: int = 1000
    num_folds: int = 4
    dimx: int = 1
    dimy: int = 2
    random_key: int = 0
    rho: float = 0.9
    base_std: float = 0.35
    signal_std: float = 1.0
    f_smoothness: float = 1.1
    eps_smoothness: float = 0.2
    xmin: float = -3.0
    xmax: float = 3.0
    f_modes: int = 20
    eps_modes: int = 20
    f_omega0: float = 1.4
    eps_omega0: float = 1.8
    probe_size: int = 4000
    mean_cv_folds: int = 5
    v_cv_folds: int = 5
    mean_alphas: tuple[float, ...] = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    mean_length_scales: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0)
    mean_nus: tuple[float, ...] = (0.5, 1.5, 2.5, float("inf"))
    mean_kernel: str = "matern"
    v_fit_mode: str = "residualized_centered"
    v_bandwidth_multipliers: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    v_etas: tuple[float, ...] = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    v_nus: tuple[float, ...] = (0.5, 1.5, 2.5)
    oracle_mc_samples: int = 100
    bootstrap_B: int = 1000
    ci_alpha: float = 0.05
    bootstrap_seed: int = 42
    oracle_max_points: int = 1600
    compute_v_diagnostics: bool = True
    compute_crossfit_rate_diagnostics: bool = True
    compute_plugin_bias_diagnostic: bool = False
    compute_bootstrap_intervals: bool = True
    compute_delta_intervals: bool = True
    # Reference for legacy direct-norm diagnostics; squared-Wald endpoints do not use it.
    delta_u_variance_reference: str = "v"
    compute_permutation_test: bool = True
    compute_population_oracle: bool = True
    compute_debiased_objects: bool = True
    compute_x_permutation_test: bool = True
    compute_standard_residual_hsic_tests: bool = False
    standard_residual_hsic_fit_prefixes: tuple[int, ...] | None = None


@dataclass
class ToyState:
    basis: Any
    f_mode_idx: Any
    f_coef: Any
    f_phase: Any
    f_probe_scale: Any
    eps_mode_idx: Any
    eps_coef: Any
    eps_phase: Any
    eps_probe_scale: Any


def standard_residual_hsic_prefixes(config: RunConfig) -> tuple[int, ...]:
    prefixes = config.standard_residual_hsic_fit_prefixes
    if prefixes is None:
        return tuple(range(1, config.num_folds))
    if isinstance(prefixes, (int, np.integer)):
        return (int(prefixes),)
    return tuple(int(prefix) for prefix in prefixes)


def validate_config(config: RunConfig) -> None:
    if config.delta_u_variance_reference not in {"u", "v"}:
        raise ValueError("delta_u_variance_reference must be u (historical) or v (current).")
    if config.N % config.num_folds != 0:
        raise ValueError("Need N divisible by num_folds.")
    fit_size = config.N - config.N // config.num_folds
    if fit_size % config.v_cv_folds != 0:
        raise ValueError("Need fold training size divisible by v_cv_folds.")
    if config.dimx < 1 or config.dimy < 1:
        raise ValueError("Need dimx >= 1 and dimy >= 1.")
    if config.mean_kernel not in {"matern", "gaussian", "rbf"}:
        raise ValueError("Need mean_kernel in {'matern', 'gaussian', 'rbf'}.")
    if config.v_fit_mode not in {"standard", "residualized_centered"}:
        raise ValueError("Need v_fit_mode in {'standard', 'residualized_centered'}.")
    if (
        config.compute_bootstrap_intervals
        or config.compute_permutation_test
        or config.compute_standard_residual_hsic_tests
    ) and config.bootstrap_B < 1:
        raise ValueError("Need bootstrap_B >= 1 when bootstrap intervals or permutation tests are enabled.")
    for prefix in standard_residual_hsic_prefixes(config):
        if prefix < 1 or prefix >= config.num_folds:
            raise ValueError(
                "standard_residual_hsic_fit_prefixes entries must be between 1 and num_folds - 1."
            )
    if (
        config.compute_v_diagnostics
        or config.compute_crossfit_rate_diagnostics
        or config.compute_plugin_bias_diagnostic
    ) and config.oracle_mc_samples < 1:
        raise ValueError("Need oracle_mc_samples >= 1 when oracle-MC diagnostics are enabled.")


# --- Basic kernels and HSIC helpers ---

@jax.jit
def pairwise_sq_dists(A: Any, B: Any) -> Any:
    diff = A[:, None, :] - B[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


@jax.jit
def gaussian_gram(A: Any, B: Any, alpha: Any) -> Any:
    alpha = jnp.maximum(alpha, 1e-12)
    return jnp.exp(-pairwise_sq_dists(A, B) / alpha)


@jax.jit
def gaussian_grad1(A: Any, B: Any, alpha: Any) -> Any:
    alpha = jnp.maximum(alpha, 1e-12)
    diff = A[:, None, :] - B[None, :, :]
    K = jnp.exp(-jnp.sum(diff * diff, axis=-1) / alpha)
    return -2.0 * diff / alpha * K[..., None]


@jax.jit
def gaussian_cross_laplacian(A: Any, B: Any, alpha: Any) -> Any:
    alpha = jnp.maximum(alpha, 1e-12)
    diff = A[:, None, :] - B[None, :, :]
    K = jnp.exp(-jnp.sum(diff * diff, axis=-1) / alpha)
    eye = jnp.eye(diff.shape[-1], dtype=diff.dtype)
    outer = diff[..., :, None] * diff[..., None, :]
    return K[..., None, None] * (2.0 * eye / alpha - 4.0 * outer / (alpha**2))


@jax.jit
def gaussian_cross_laplacian_coord(A: Any, B: Any, alpha: Any, coord: Any) -> Any:
    alpha = jnp.maximum(alpha, 1e-12)
    diff = A[:, None, :] - B[None, :, :]
    K = jnp.exp(-jnp.sum(diff * diff, axis=-1) / alpha)
    coord_diff = jnp.take(diff, coord, axis=-1)
    return K * (2.0 / alpha - 4.0 * coord_diff * coord_diff / (alpha**2))


# def median_heuristic(Z1: Any, Z2: Any) -> Any:
#     Z = jnp.concatenate([Z1, Z2], axis=0)
#     sq = pairwise_sq_dists(Z, Z)
#     tri = sq[jnp.triu_indices(Z.shape[0], k=1)]
#     return jnp.median(tri)

def median_heuristic(Z1 : Any, Z2 : Any =None) -> Any:
    if Z2 is None or Z1 is Z2:
        Z = Z1
    else:
        Z = jnp.concatenate([Z1, Z2], axis=0)
    sq_dists = jnp.linalg.norm(Z[:, None] - Z[None, :], axis=-1) ** 2
    return jnp.median(sq_dists[jnp.triu_indices(Z.shape[0], k=1)])

def center_gram(G: Any) -> Any:
    return G - jnp.mean(G, axis=0)[None, :] - jnp.mean(G, axis=1)[:, None] + jnp.mean(G)


@jax.jit
def center_eval_kernel(Q_eval_train: Any, Q_train_train: Any) -> Any:
    return (
        Q_eval_train
        - jnp.mean(Q_eval_train, axis=1)[:, None]
        - jnp.mean(Q_train_train, axis=0)[None, :]
        + jnp.mean(Q_train_train)
    )


def hsic_unbiased(K: Any, L: Any) -> Any: # K\in R^{n\times n} L \in \R^{n\times n}. This is a standard plug-in non-cross-fitted formula
    n = K.shape[0]
    if n < 4:
        raise ValueError("Need n >= 4 for unbiased HSIC.")
    Kt = K - jnp.diag(jnp.diag(K))
    Lt = L - jnp.diag(jnp.diag(L))
    term1 = jnp.sum(Kt * Lt)
    term2 = jnp.sum(Kt) * jnp.sum(Lt) / ((n - 1) * (n - 2))
    term3 = 2.0 * jnp.sum(Kt @ Lt) / (n - 2)
    return (term1 + term2 - term3) / (n * (n - 3))


def matern_from_dists(dists: Any, length_scale: Any, nu: Any) -> Any:
    ls = jnp.maximum(length_scale, 1e-12)

    r05 = dists / ls
    k05 = jnp.exp(-r05)

    r15 = jnp.sqrt(3.0) * dists / ls
    k15 = (1.0 + r15) * jnp.exp(-r15)

    r25 = jnp.sqrt(5.0) * dists / ls
    k25 = (1.0 + r25 + (r25**2) / 3.0) * jnp.exp(-r25)

    return jnp.where(nu == 0.5, k05, jnp.where(nu == 1.5, k15, k25))


def build_fold_masks(N: int, num_folds: int) -> Any:
    size_fold = N // num_folds
    return jnp.repeat(jnp.eye(num_folds, dtype=bool), size_fold, axis=1)


def center_cross_block(G: Any, mask1: Any, mask2: Any) -> Any:
    return (
        G[mask1][:, mask2]
        - jnp.mean(G[mask1][:, ~mask2], axis=1)[:, None]
        - jnp.mean(G[~mask1][:, mask2], axis=0)[None, :]
        + jnp.mean(G[~mask1][:, ~mask2])[None, None]
    )


def centered_blocks_from_full_gram(G: Any, fold_masks: Any) -> Any:
    # Returns the tensor with (k, l) entry equal to the submatrix G_{I_k, I_l}
    # centered by row means over I_k x I_l^c, column means over I_k^c x I_l,
    # and the grand mean over I_k^c x I_l^c.
    rows = []
    for mask1 in fold_masks:
        row = []
        for mask2 in fold_masks:
            row.append(center_cross_block(G, mask1, mask2))
        rows.append(row)
    return jnp.array(rows)


def fold_indices_from_masks(fold_masks: Any, size_fold: int) -> Any:
    return jnp.stack([jnp.where(mask, size=size_fold)[0] for mask in fold_masks])


def center_cross_block_by_indices(G: Any, idx1: Any, idx2: Any, comp1: Any, comp2: Any) -> Any:
    return (
        G[jnp.ix_(idx1, idx2)]
        - jnp.mean(G[jnp.ix_(idx1, comp2)], axis=1)[:, None]
        - jnp.mean(G[jnp.ix_(comp1, idx2)], axis=0)[None, :]
        + jnp.mean(G[jnp.ix_(comp1, comp2)])[None, None]
    )


def centered_blocks_from_full_gram_indices(G: Any, fold_indices: Any) -> Any:
    num_folds = fold_indices.shape[0]
    rows = []
    for k in range(num_folds):
        row = []
        idx1 = fold_indices[k]
        comp1 = jnp.concatenate([fold_indices[a] for a in range(num_folds) if a != k])
        for l in range(num_folds):
            idx2 = fold_indices[l]
            comp2 = jnp.concatenate([fold_indices[a] for a in range(num_folds) if a != l])
            row.append(center_cross_block_by_indices(G, idx1, idx2, comp1, comp2))
        rows.append(row)
    return jnp.array(rows)


@jax.jit
def standard_krr_weight_matrix(Q_train_train: Any, Q_eval_train: Any, eta: Any) -> Any:
    n_train = Q_train_train.shape[0]
    reg = Q_train_train + eta * n_train * jnp.eye(n_train, dtype=Q_train_train.dtype)
    return jnp.linalg.solve(reg, Q_eval_train.T).T


@jax.jit
def residualized_krr_weight_matrix(Q_train_train: Any, Q_eval_train: Any, eta: Any) -> Any:
    n_train = Q_train_train.shape[0]
    dtype = Q_train_train.dtype
    Qc_train = center_gram(Q_train_train)
    qc_eval = center_eval_kernel(Q_eval_train, Q_train_train)
    # Fit only the centered residual part, but keep an explicit intercept term in
    # the forecast weights so constant targets are carried through unchanged.
    intercept_weights = jnp.full((Q_eval_train.shape[0], n_train), 1.0 / n_train, dtype=dtype)
    C = jnp.eye(n_train, dtype=dtype) - jnp.full((n_train, n_train), 1.0 / n_train, dtype=dtype)
    reg = Qc_train + eta * n_train * jnp.eye(n_train, dtype=dtype)
    residual_weights = jnp.linalg.solve(reg, qc_eval.T).T
    weights = intercept_weights + residual_weights @ C
    # Re-normalize rows to sum to one so the foldwise intercept is preserved
    # exactly even under small linear-solve numerical error.
    return weights - (jnp.sum(weights, axis=1, keepdims=True) - 1.0) / n_train


def orthonormal_basis(key: Any, d: int) -> Any:
    raw = jax.random.normal(key, shape=(d, d))
    q, _ = jnp.linalg.qr(raw)
    return q.T


def product_fourier_values(
    X: Any,
    mode_idx: Any,
    coef: Any,
    phase: Any,
    smoothness: float,
    omega0: float,
    basis: Any,
) -> Any:
    # Each coordinate is a short product-Fourier series on a random orthonormal basis.
    X = jnp.atleast_2d(jnp.asarray(X, dtype=jnp.float64))
    proj = X @ basis.T
    angles = omega0 * proj[:, None, None, :] * mode_idx[None, None, :, None] + phase[None, :, :, :]
    amps = coef[None, :, :] / (mode_idx[None, None, :] ** smoothness)
    return jnp.sum(amps * jnp.prod(jnp.sin(angles), axis=-1), axis=-1)


def build_toy_state(config: RunConfig, key: Any) -> ToyState:
    # Sample one fixed signal/noise environment and reuse it for the whole run.
    basis = orthonormal_basis(jax.random.fold_in(key, 0), config.dimx)
    probe_x = jax.random.uniform(
        jax.random.fold_in(key, 1),
        shape=(config.probe_size, config.dimx),
        minval=config.xmin,
        maxval=config.xmax,
    )

    f_mode_idx = jnp.arange(1, config.f_modes + 1, dtype=jnp.float64)
    f_coef = jax.random.normal(jax.random.fold_in(key, 2), shape=(config.dimy, config.f_modes))
    f_phase = jax.random.uniform(
        jax.random.fold_in(key, 3),
        shape=(config.dimy, config.f_modes, config.dimx),
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )
    f_probe = product_fourier_values(
        probe_x,
        f_mode_idx,
        f_coef,
        f_phase,
        config.f_smoothness,
        config.f_omega0,
        basis,
    )
    f_probe_scale = jnp.maximum(jnp.std(f_probe, axis=0), 1e-12)

    eps_mode_idx = jnp.arange(1, config.eps_modes + 1, dtype=jnp.float64)
    eps_coef = jax.random.normal(jax.random.fold_in(key, 4), shape=(config.dimy, config.eps_modes))
    eps_phase = jax.random.uniform(
        jax.random.fold_in(key, 5),
        shape=(config.dimy, config.eps_modes, config.dimx),
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )
    eps_probe = product_fourier_values(
        probe_x,
        eps_mode_idx,
        eps_coef,
        eps_phase,
        config.eps_smoothness,
        config.eps_omega0,
        basis,
    )
    eps_probe_scale = jnp.maximum(jnp.std(eps_probe, axis=0), 1e-12)

    return ToyState(
        basis=basis,
        f_mode_idx=f_mode_idx,
        f_coef=f_coef,
        f_phase=f_phase,
        f_probe_scale=f_probe_scale,
        eps_mode_idx=eps_mode_idx,
        eps_coef=eps_coef,
        eps_phase=eps_phase,
        eps_probe_scale=eps_probe_scale,
    )


def f_true(state: ToyState, config: RunConfig, X: Any) -> Any:
    return config.signal_std * product_fourier_values(
        X,
        state.f_mode_idx,
        state.f_coef,
        state.f_phase,
        config.f_smoothness,
        config.f_omega0,
        state.basis,
    ) / state.f_probe_scale


def eps_shape(state: ToyState, config: RunConfig, X: Any) -> Any:
    return product_fourier_values(
        X,
        state.eps_mode_idx,
        state.eps_coef,
        state.eps_phase,
        config.eps_smoothness,
        config.eps_omega0,
        state.basis,
    ) / state.eps_probe_scale


def eps_std_fn(state: ToyState, config: RunConfig, X: Any) -> Any:
    return config.base_std * jnp.exp(jnp.maximum(config.rho, 0.0) * eps_shape(state, config, X))


def sample_noise_at_x(key: Any, x: Any, state: ToyState, config: RunConfig) -> Any:
    sigma = eps_std_fn(state, config, x[None, :])[0]
    return sigma * jax.random.normal(key, shape=(config.dimy,))


def sample_noise_mc_at_x(
    key: Any,
    x: Any,
    basis: Any,
    eps_mode_idx: Any,
    eps_coef: Any,
    eps_phase: Any,
    eps_probe_scale: Any,
    eps_smoothness: Any,
    eps_omega0: Any,
    base_std: Any,
    rho: Any,
    oracle_mc_samples: int,
) -> Any:
    shape = product_fourier_values(
        x[None, :],
        eps_mode_idx,
        eps_coef,
        eps_phase,
        eps_smoothness,
        eps_omega0,
        basis,
    ) / eps_probe_scale
    sigma = base_std * jnp.exp(jnp.maximum(rho, 0.0) * shape[0])
    keys = jax.random.split(key, oracle_mc_samples)
    noise = jax.vmap(lambda subkey: jax.random.normal(subkey, shape=(eps_coef.shape[0],), dtype=x.dtype))(keys)
    return sigma[None, :] * noise


def evaluate_v_risk_means_for_coord(
    key: Any,
    fold_idx: Any,
    coord: Any,
    X_eval: Any,
    Xi_train: Any,
    Lap_train_j: Any,
    weights: Any,
    basis: Any,
    eps_mode_idx: Any,
    eps_coef: Any,
    eps_phase: Any,
    eps_probe_scale: Any,
    eps_smoothness: Any,
    eps_omega0: Any,
    base_std: Any,
    rho: Any,
    alpha_xi: Any,
    oracle_mc_samples: int,
    key_offset: int,
) -> tuple[Any, Any, Any]:
    point_ids = jnp.arange(X_eval.shape[0], dtype=jnp.uint32)
    coord_u32 = jnp.asarray(coord, dtype=jnp.uint32)
    fold_u32 = jnp.asarray(fold_idx, dtype=jnp.uint32)
    base_idx = (
        jnp.asarray(key_offset, dtype=jnp.uint32)
        + jnp.asarray(1_000_000, dtype=jnp.uint32) * fold_u32
        + jnp.asarray(10_000, dtype=jnp.uint32) * coord_u32
    )
    point_keys = jax.vmap(lambda idx: jax.random.fold_in(key, idx))(base_idx + point_ids)
    Xi_mc = jax.vmap(
        lambda subkey, x: sample_noise_mc_at_x(
            subkey,
            x,
            basis,
            eps_mode_idx,
            eps_coef,
            eps_phase,
            eps_probe_scale,
            eps_smoothness,
            eps_omega0,
            base_std,
            rho,
            oracle_mc_samples,
        )
    )(point_keys, X_eval)

    Lap_mc_j = jax.vmap(
        lambda Xi_local: gaussian_cross_laplacian_coord(Xi_local, Xi_local, alpha_xi, coord)
    )(Xi_mc)
    Lap_train_mc_j = jax.vmap(
        lambda Xi_local: gaussian_cross_laplacian_coord(Xi_train, Xi_local, alpha_xi, coord)
    )(Xi_mc)

    mc_diag_mean = jnp.mean(jnp.diagonal(Lap_mc_j, axis1=1, axis2=2), axis=1)
    oracle_risks = mc_diag_mean - jnp.mean(Lap_mc_j, axis=(1, 2))

    const_weights = jnp.full((Xi_train.shape[0],), 1.0 / Xi_train.shape[0], dtype=Lap_train_j.dtype)
    const_norm = const_weights @ Lap_train_j @ const_weights
    const_cross = jnp.mean(jnp.einsum("n,enb->eb", const_weights, Lap_train_mc_j), axis=1)
    const_risks = const_norm - 2.0 * const_cross + mc_diag_mean

    fitted_norms = jnp.einsum("en,nm,em->e", weights, Lap_train_j, weights)
    fitted_cross = jnp.mean(jnp.einsum("en,enb->eb", weights, Lap_train_mc_j), axis=1)
    fitted_risks = fitted_norms - 2.0 * fitted_cross + mc_diag_mean

    return jnp.mean(oracle_risks), jnp.mean(fitted_risks), jnp.mean(const_risks)


evaluate_v_risk_means_for_coord = partial(
    jax.jit,
    static_argnames=("oracle_mc_samples", "key_offset"),
)(evaluate_v_risk_means_for_coord)


def simulate_data(config: RunConfig) -> tuple[ToyState, Any, Any, Any, Any]:
    key = jax.random.PRNGKey(config.random_key)
    state = build_toy_state(config, key)
    X = jax.random.uniform(
        jax.random.fold_in(key, 6),
        shape=(config.N, config.dimx),
        minval=config.xmin,
        maxval=config.xmax,
    )
    signal = f_true(state, config, X)
    eps_keys = jax.random.split(jax.random.fold_in(key, 7), config.N)
    eps = jax.vmap(lambda k, x: sample_noise_at_x(k, x, state, config))(eps_keys, X)
    Y = signal + eps
    return state, X, signal, eps, Y


# --- First-stage mean fit ---


def make_mean_krr_cv(config: RunConfig, cv_folds: int) -> GridSearchCV:
    if config.mean_kernel == "matern":
        estimator = KernelRidge(kernel=Matern())
        param_grid = {
            "alpha": list(config.mean_alphas),
            "kernel__length_scale": list(config.mean_length_scales),
            "kernel__nu": list(config.mean_nus),
        }
    else:
        estimator = KernelRidge(kernel=RBF())
        param_grid = {
            "alpha": list(config.mean_alphas),
            "kernel__length_scale": list(config.mean_length_scales),
        }

    return GridSearchCV(
        estimator,
        param_grid,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
    )


def summarize_mean_krr_cv(config: RunConfig, model: GridSearchCV) -> dict[str, Any]:
    best_params = model.best_params_
    out = {
        "alpha": float(best_params["alpha"]),
        "kernel": "gaussian" if config.mean_kernel == "rbf" else config.mean_kernel,
        "length_scale": float(best_params["kernel__length_scale"]),
    }
    if config.mean_kernel == "matern":
        out["nu"] = float(best_params["kernel__nu"])
    else:
        out["nu"] = None
    return out

def fit_mean_forecasts(
    config: RunConfig,
    X: Any,
    Y: Any,
    signal: Any,
    fold_masks: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    forecasts = []
    diagnostics = []

    for fold_idx, mask in enumerate(np.asarray(fold_masks)):
        model = make_mean_krr_cv(config, config.mean_cv_folds)
        X_train = np.asarray(X[~mask])
        Y_train = np.asarray(Y[~mask])
        model.fit(X_train, Y_train)
        pred = model.best_estimator_.predict(np.asarray(X))
        forecasts.append(jnp.asarray(pred))

        y_holdout = np.asarray(Y[mask])
        signal_holdout = np.asarray(signal[mask])
        heldout_response_mse = float(np.mean((pred[mask] - y_holdout) ** 2))
        oracle_response_mse = float(np.mean((signal_holdout - y_holdout) ** 2))
        heldout_signal_mse = float(np.mean((pred[mask] - signal_holdout) ** 2))
        decomposition_cross_term = heldout_response_mse - oracle_response_mse - heldout_signal_mse

        diagnostics.append(
            {
                "fold": fold_idx,
                "cv_mse": float(-model.best_score_),
                "heldout_response_mse": heldout_response_mse,
                "oracle_response_mse": oracle_response_mse,
                "noise_variance_term": oracle_response_mse,
                "signal_mse_term": heldout_signal_mse,
                "decomposition_cross_term": decomposition_cross_term,
                "response_mse_minus_oracle": heldout_response_mse - oracle_response_mse,
                "heldout_signal_mse": heldout_signal_mse,
                **summarize_mean_krr_cv(config, model),
            }
        )

    return jnp.stack(forecasts), diagnostics


def fit_mean_on_split(
    config: RunConfig,
    X: Any,
    Y: Any,
    signal: Any,
    train_mask: Any,
    test_mask: Any,
    split_id: int,
) -> tuple[Any, Any, dict[str, Any]]:
    cv_folds = min(config.mean_cv_folds, int(np.sum(train_mask)))
    if cv_folds < 2:
        raise ValueError("Need at least two training points for split mean-fit cross-validation.")
    model = make_mean_krr_cv(config, cv_folds)
    X_train = np.asarray(X[train_mask])
    Y_train = np.asarray(Y[train_mask])
    X_test = np.asarray(X[test_mask])
    model.fit(X_train, Y_train)
    pred_train = model.best_estimator_.predict(X_train)
    pred_test = model.best_estimator_.predict(X_test)

    y_test = np.asarray(Y[test_mask])
    signal_test = np.asarray(signal[test_mask])
    heldout_response_mse = float(np.mean((pred_test - y_test) ** 2))
    oracle_response_mse = float(np.mean((signal_test - y_test) ** 2))
    heldout_signal_mse = float(np.mean((pred_test - signal_test) ** 2))
    decomposition_cross_term = heldout_response_mse - oracle_response_mse - heldout_signal_mse

    return jnp.asarray(pred_train), jnp.asarray(pred_test), {
        "split": int(split_id),
        "cv_mse": float(-model.best_score_),
        "heldout_response_mse": heldout_response_mse,
        "oracle_response_mse": oracle_response_mse,
        "noise_variance_term": oracle_response_mse,
        "signal_mse_term": heldout_signal_mse,
        "decomposition_cross_term": decomposition_cross_term,
        "response_mse_minus_oracle": heldout_response_mse - oracle_response_mse,
        "heldout_signal_mse": heldout_signal_mse,
        **summarize_mean_krr_cv(config, model),
    }


def evaluate_mean_oos_diagnostics(
    Y: Any,
    signal: Any,
    mean_forecasts: Any,
    fold_masks: Any,
) -> list[dict[str, Any]]:
    diagnostics = []

    for fold_idx, mask in enumerate(np.asarray(fold_masks)):
        pred = np.asarray(mean_forecasts[fold_idx])
        y_holdout = np.asarray(Y[mask])
        signal_holdout = np.asarray(signal[mask])
        heldout_response_mse = float(np.mean((pred[mask] - y_holdout) ** 2))
        oracle_response_mse = float(np.mean((signal_holdout - y_holdout) ** 2))
        heldout_signal_mse = float(np.mean((pred[mask] - signal_holdout) ** 2))
        decomposition_cross_term = heldout_response_mse - oracle_response_mse - heldout_signal_mse
        diagnostics.append(
            {
                "fold": fold_idx,
                "heldout_response_mse": heldout_response_mse,
                "oracle_response_mse": oracle_response_mse,
                "noise_variance_term": oracle_response_mse,
                "signal_mse_term": heldout_signal_mse,
                "decomposition_cross_term": decomposition_cross_term,
                "response_mse_minus_oracle": heldout_response_mse - oracle_response_mse,
                "heldout_signal_mse": heldout_signal_mse,
            }
        )

    return diagnostics


def assemble_oos_predictions(mean_forecasts: Any, fold_masks: Any) -> Any:
    pred_oos = jnp.zeros_like(mean_forecasts[0])
    for fold_idx, mask in enumerate(fold_masks):
        pred_oos = pred_oos.at[mask].set(mean_forecasts[fold_idx][mask])
    return pred_oos


def estimate_oracle_v_norm_mean(
    config: RunConfig,
    state: ToyState,
    X_eval: Any,
    alpha_xi: Any,
) -> dict[str, Any]:
    point_keys = jax.random.split(jax.random.PRNGKey(config.random_key + 314159), X_eval.shape[0])

    def coord_norms_at_point(subkey: Any, x: Any) -> Any:
        Xi_mc = sample_noise_mc_at_x(
            subkey,
            x,
            state.basis,
            state.eps_mode_idx,
            state.eps_coef,
            state.eps_phase,
            state.eps_probe_scale,
            config.eps_smoothness,
            config.eps_omega0,
            config.base_std,
            config.rho,
            config.oracle_mc_samples,
        )
        Lap_mc = gaussian_cross_laplacian(Xi_mc, Xi_mc, alpha_xi)
        return jnp.mean(jnp.diagonal(Lap_mc, axis1=-2, axis2=-1), axis=(0, 1))

    coord_norms = jax.vmap(coord_norms_at_point)(point_keys, X_eval)
    coord_mean = jnp.mean(coord_norms, axis=0)
    return {
        "by_coord": coord_mean,
        "total": jnp.sum(coord_mean),
    }


def summarize_plugin_bias_diagnostic(
    config: RunConfig,
    state: ToyState,
    X: Any,
    signal: Any,
    mean_forecasts: Any,
    fold_masks: Any,
    alpha_x: Any,
    alpha_xi: Any,
) -> dict[str, Any]:
    pred_oos = assemble_oos_predictions(mean_forecasts, fold_masks)
    mean_error = pred_oos - signal

    K = gaussian_gram(X, X, alpha_x)
    W_x = center_gram(K)

    centered_error_norm_sq_by_coord = jnp.array(
        [jnp.mean(W_x * jnp.outer(mean_error[:, j], mean_error[:, j])) for j in range(config.dimy)]
    )
    centered_error_norm_sq = jnp.sum(centered_error_norm_sq_by_coord)

    oracle_v_norm = estimate_oracle_v_norm_mean(config, state, X, alpha_xi)
    oracle_v_norm_sq_mean_by_coord = oracle_v_norm["by_coord"]
    oracle_v_norm_sq_mean = oracle_v_norm["total"]

    return {
        "centered_mean_error_norm_sq": centered_error_norm_sq,
        "centered_mean_error_norm_sq_by_coord": centered_error_norm_sq_by_coord,
        "oracle_v_norm_sq_mean": oracle_v_norm_sq_mean,
        "oracle_v_norm_sq_mean_by_coord": oracle_v_norm_sq_mean_by_coord,
        "plugin_bias_diagnostic": jnp.sqrt(jnp.maximum(centered_error_norm_sq, 0.0))
        * jnp.sqrt(jnp.maximum(oracle_v_norm_sq_mean, 0.0)),
        "plugin_bias_diagnostic_by_coord": jnp.sqrt(jnp.maximum(centered_error_norm_sq_by_coord, 0.0))
        * jnp.sqrt(jnp.maximum(oracle_v_norm_sq_mean_by_coord, 0.0)),
    }


# --- Second-stage v_j fit ---

def rkhs_kfold_cv_from_q(
    Q_local: Any,
    LapL_local: Any,
    eta: float,
    cv_fold_ids: Any,
    cv_fold_size: int,
    k_cv_folds: int,
    residualized_centered: bool,
) -> Any:
    def evaluate_fold(fold_idx: Any) -> Any:
        n_local = Q_local.shape[0]
        val_idx = jnp.where(cv_fold_ids == fold_idx, size=cv_fold_size)[0]
        train_idx = jnp.where(cv_fold_ids != fold_idx, size=n_local - cv_fold_size)[0]

        Q_tr_tr = Q_local[jnp.ix_(train_idx, train_idx)]
        Q_val_tr = Q_local[jnp.ix_(val_idx, train_idx)]

        Lap_tr_tr = LapL_local[jnp.ix_(train_idx, train_idx)]
        Lap_tr_val = LapL_local[jnp.ix_(train_idx, val_idx)]
        Lap_val_val = LapL_local[jnp.ix_(val_idx, val_idx)]

        w_val = jax.lax.cond(
            residualized_centered,
            lambda _: residualized_krr_weight_matrix(Q_tr_tr, Q_val_tr, eta),
            lambda _: standard_krr_weight_matrix(Q_tr_tr, Q_val_tr, eta),
            operand=None,
        )

        vhat_norm_sq = jnp.einsum("ij,ik,jk->i", w_val, w_val, Lap_tr_tr)
        cross_term = jnp.einsum("ij,ji->i", w_val, Lap_tr_val)
        true_norm_sq = jnp.diag(Lap_val_val)
        return jnp.mean(vhat_norm_sq - 2.0 * cross_term + true_norm_sq)

    return jnp.mean(jax.vmap(evaluate_fold)(jnp.arange(k_cv_folds)))


rkhs_kfold_cv_from_q = partial(
    jax.jit,
    static_argnames=("cv_fold_size", "k_cv_folds", "residualized_centered"),
)(rkhs_kfold_cv_from_q)


def v_weight_cv_grid(
    dists_local: Any,
    lap_fit_j: Any,
    base_length_scale: Any,
    bandwidth_multipliers: Any,
    etas: Any,
    nus: Any,
    cv_fold_ids: Any,
    cv_fold_size: int,
    k_cv_folds: int,
    residualized_centered: bool,
) -> Any:
    def mse_for_nu(nu: Any) -> Any:
        def mse_for_multiplier(multiplier: Any) -> Any:
            length_scale = multiplier * base_length_scale
            Q_local = matern_from_dists(dists_local, length_scale, nu)
            return jax.vmap(
                lambda eta: rkhs_kfold_cv_from_q(
                    Q_local,
                    lap_fit_j,
                    eta,
                    cv_fold_ids,
                    cv_fold_size,
                    k_cv_folds,
                    residualized_centered,
                )
            )(etas)

        return jax.vmap(mse_for_multiplier)(bandwidth_multipliers)

    return jax.vmap(mse_for_nu)(nus)


v_weight_cv_grid = partial(
    jax.jit,
    static_argnames=("cv_fold_size", "k_cv_folds", "residualized_centered"),
)(v_weight_cv_grid)


def fit_v_weights(
    config: RunConfig,
    X: Any,
    Xi_hat_folds: Any,
    fold_masks: Any,
    alpha_xi: Any,
) -> tuple[Any, Any, list[list[dict[str, Any]]]]:
    size_fold = config.N // config.num_folds
    fit_size = config.N - size_fold
    cv_fold_size = fit_size // config.v_cv_folds
    cv_fold_ids = jnp.arange(fit_size) // cv_fold_size

    dl = gaussian_grad1
    lap_l = gaussian_cross_laplacian

    base_length_scale = jnp.sqrt(jnp.maximum(median_heuristic(X, X), 1e-12))
    dists = jnp.sqrt(jnp.maximum(pairwise_sq_dists(X, X), 0.0))
    bandwidth_multipliers = jnp.asarray(config.v_bandwidth_multipliers, dtype=jnp.float64)
    etas = jnp.asarray(config.v_etas, dtype=jnp.float64)
    nus = jnp.asarray(config.v_nus, dtype=jnp.float64)
    residualized_centered = config.v_fit_mode == "residualized_centered"

    W_is_folds = []
    W_oos_folds = []
    best_v_params = []

    for fold_idx, mask in enumerate(fold_masks):
        # Fit each coordinate j separately, exactly as in the notebook.
        Xi_train = Xi_hat_folds[fold_idx][~mask]
        Lap_fit = lap_l(Xi_train, Xi_train, alpha_xi)
        dists_local = dists[~mask][:, ~mask]

        W_is_row = []
        W_oos_row = []
        params_row = []

        for j in range(config.dimy):
            Lap_fit_j = Lap_fit[..., j, j]
            mses = v_weight_cv_grid(
                dists_local,
                Lap_fit_j,
                base_length_scale,
                bandwidth_multipliers,
                etas,
                nus,
                cv_fold_ids,
                cv_fold_size,
                config.v_cv_folds,
                residualized_centered,
            )

            idxmin = int(jnp.argmin(mses))
            n_mult = len(config.v_bandwidth_multipliers)
            n_eta = len(config.v_etas)

            idx_nu = idxmin // (n_mult * n_eta)
            rem = idxmin % (n_mult * n_eta)
            idx_mult = rem // n_eta
            idx_eta = rem % n_eta

            nu_best = float(nus[idx_nu])
            scale_best = float(bandwidth_multipliers[idx_mult])
            eta_best = float(etas[idx_eta])
            length_scale_best = scale_best * float(base_length_scale)
            cv_mse_best = float(mses[idx_nu, idx_mult, idx_eta])

            Q_full = matern_from_dists(dists, length_scale_best, nu_best)
            Q_train_train = Q_full[~mask][:, ~mask]
            Q_oos_train = Q_full[mask][:, ~mask]

            if residualized_centered:
                W_is_row.append(residualized_krr_weight_matrix(Q_train_train, Q_train_train, eta_best))
                W_oos_row.append(residualized_krr_weight_matrix(Q_train_train, Q_oos_train, eta_best))
            else:
                W_is_row.append(standard_krr_weight_matrix(Q_train_train, Q_train_train, eta_best))
                W_oos_row.append(standard_krr_weight_matrix(Q_train_train, Q_oos_train, eta_best))
            params_row.append(
                {
                    "nu": nu_best,
                    "scale_multiplier": scale_best,
                    "length_scale": length_scale_best,
                    "eta": eta_best,
                    "cv_mse": cv_mse_best,
                }
            )

        W_is_folds.append(W_is_row)
        W_oos_folds.append(W_oos_row)
        best_v_params.append(params_row)

    return jnp.array(W_is_folds), jnp.array(W_oos_folds), best_v_params


def evaluate_v_regression_diagnostics(
    config: RunConfig,
    state: ToyState,
    X: Any,
    Xi_hat_folds: Any,
    fold_masks: Any,
    W_is_folds: Any,
    best_v_params: list[list[dict[str, Any]]],
    alpha_xi: Any,
) -> list[dict[str, Any]]:
    key = jax.random.PRNGKey(config.random_key)
    diagnostics = []

    for fold_idx, mask in enumerate(fold_masks):
        # Compare the attained fit to two baselines:
        # 1. the oracle Monte Carlo risk for the true conditional target
        # 2. the constant residual baseline used in the notebook
        X_train = X[~mask]
        Xi_train = Xi_hat_folds[fold_idx][~mask]

        for j in range(config.dimy):
            Lap_train_j = gaussian_cross_laplacian_coord(Xi_train, Xi_train, alpha_xi, j)
            surrogate_const_risk = float(jnp.mean(jnp.diag(Lap_train_j)) - jnp.mean(Lap_train_j))
            oracle_risk_mean, fitted_risk_mean, const_risk_mean = evaluate_v_risk_means_for_coord(
                key,
                fold_idx,
                j,
                X_train,
                Xi_train,
                Lap_train_j,
                W_is_folds[fold_idx, j],
                state.basis,
                state.eps_mode_idx,
                state.eps_coef,
                state.eps_phase,
                state.eps_probe_scale,
                config.eps_smoothness,
                config.eps_omega0,
                config.base_std,
                config.rho,
                alpha_xi,
                oracle_mc_samples=config.oracle_mc_samples,
                key_offset=0,
            )
            oracle_risk_mean = float(oracle_risk_mean)
            const_risk_mean = float(const_risk_mean)
            fitted_risk_mean = float(fitted_risk_mean)
            diagnostics.append(
                {
                    "fold": fold_idx,
                    "coord": j,
                    "oracle_risk": oracle_risk_mean,
                    "surrogate_const_risk": surrogate_const_risk,
                    "const_risk": const_risk_mean,
                    "fitted_risk": fitted_risk_mean,
                    "cv_mse": best_v_params[fold_idx][j]["cv_mse"],
                    "excess_over_oracle": fitted_risk_mean - oracle_risk_mean,
                    "improvement_over_const": const_risk_mean - fitted_risk_mean,
                }
            )

    return diagnostics


def evaluate_v_oos_regression_diagnostics(
    config: RunConfig,
    state: ToyState,
    X: Any,
    Xi_hat_folds: Any,
    fold_masks: Any,
    W_oos_folds: Any,
    best_v_params: list[list[dict[str, Any]]],
    alpha_xi: Any,
) -> list[dict[str, Any]]:
    key = jax.random.PRNGKey(config.random_key)
    diagnostics = []

    for fold_idx, mask in enumerate(fold_masks):
        # Evaluate the fold-k fit on the held-out fold-k points to proxy ||v_hat^{-k} - v_0||^2.
        X_eval = X[mask]
        Xi_train = Xi_hat_folds[fold_idx][~mask]

        for j in range(config.dimy):
            Lap_train_j = gaussian_cross_laplacian_coord(Xi_train, Xi_train, alpha_xi, j)
            oracle_risk_mean, fitted_risk_mean, _ = evaluate_v_risk_means_for_coord(
                key,
                fold_idx,
                j,
                X_eval,
                Xi_train,
                Lap_train_j,
                W_oos_folds[fold_idx, j],
                state.basis,
                state.eps_mode_idx,
                state.eps_coef,
                state.eps_phase,
                state.eps_probe_scale,
                config.eps_smoothness,
                config.eps_omega0,
                config.base_std,
                config.rho,
                alpha_xi,
                oracle_mc_samples=config.oracle_mc_samples,
                key_offset=100_000_000,
            )
            oracle_risk_mean = float(oracle_risk_mean)
            fitted_risk_mean = float(fitted_risk_mean)
            diagnostics.append(
                {
                    "fold": fold_idx,
                    "coord": j,
                    "oracle_risk": oracle_risk_mean,
                    "fitted_risk": fitted_risk_mean,
                    "cv_mse": best_v_params[fold_idx][j]["cv_mse"],
                    "excess_over_oracle": fitted_risk_mean - oracle_risk_mean,
                }
            )

    return diagnostics


def summarize_crossfit_rate_diagnostics(
    N: int,
    mean_oos_diagnostics: list[dict[str, Any]],
    v_oos_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    if N < 1:
        raise ValueError("Need N >= 1 to summarize rate diagnostics.")
    if not mean_oos_diagnostics:
        raise ValueError("Need at least one mean OOS diagnostic entry.")

    sqrt_N = math.sqrt(N)

    a_n_sq_by_fold = np.asarray(
        [entry["heldout_signal_mse"] for entry in mean_oos_diagnostics],
        dtype=np.float64,
    )
    a_n_sq = float(np.mean(a_n_sq_by_fold))
    a_n = math.sqrt(max(a_n_sq, 0.0))

    summary = {
        "N": int(N),
        "a_n_sq_by_fold": a_n_sq_by_fold,
        "a_n_by_fold": np.sqrt(np.maximum(a_n_sq_by_fold, 0.0)),
        "a_n_sq": a_n_sq,
        "a_n": a_n,
        "sqrt_N_a_n": sqrt_N * a_n,
        "sqrt_N_a_n_sq": sqrt_N * a_n_sq,
        "aggregation": (
            "a_n_sq averages foldwise held-out signal MSEs; "
            "b_n_sq averages held-out v excess risks over fold-coordinate fits."
        ),
    }

    if not v_oos_diagnostics:
        summary.update(
            {
                "b_n_sq_raw": None,
                "b_n_sq": None,
                "b_n": None,
                "sqrt_N_a_n_b_n": None,
                "b_n_sq_raw_by_fold": [],
                "b_n_sq_by_fold": [],
                "b_n_by_fold": [],
                "b_n_sq_raw_by_coord": [],
                "b_n_sq_by_coord": [],
                "b_n_by_coord": [],
            }
        )
        return summary

    b_n_sq_raw_entries = np.asarray(
        [entry["excess_over_oracle"] for entry in v_oos_diagnostics],
        dtype=np.float64,
    )
    b_n_sq_raw = float(np.mean(b_n_sq_raw_entries))
    b_n_sq = max(b_n_sq_raw, 0.0)
    b_n = math.sqrt(b_n_sq)

    fold_ids = sorted({entry["fold"] for entry in v_oos_diagnostics})
    coord_ids = sorted({entry["coord"] for entry in v_oos_diagnostics})

    b_n_sq_raw_by_fold = np.asarray(
        [
            np.mean([entry["excess_over_oracle"] for entry in v_oos_diagnostics if entry["fold"] == fold_idx])
            for fold_idx in fold_ids
        ],
        dtype=np.float64,
    )
    b_n_sq_raw_by_coord = np.asarray(
        [
            np.mean([entry["excess_over_oracle"] for entry in v_oos_diagnostics if entry["coord"] == coord_idx])
            for coord_idx in coord_ids
        ],
        dtype=np.float64,
    )

    summary.update(
        {
            "b_n_sq_raw": b_n_sq_raw,
            "b_n_sq": b_n_sq,
            "b_n": b_n,
            "sqrt_N_a_n_b_n": sqrt_N * a_n * b_n,
            "b_n_sq_raw_by_fold": b_n_sq_raw_by_fold,
            "b_n_sq_by_fold": np.maximum(b_n_sq_raw_by_fold, 0.0),
            "b_n_by_fold": np.sqrt(np.maximum(b_n_sq_raw_by_fold, 0.0)),
            "b_n_sq_raw_by_coord": b_n_sq_raw_by_coord,
            "b_n_sq_by_coord": np.maximum(b_n_sq_raw_by_coord, 0.0),
            "b_n_by_coord": np.sqrt(np.maximum(b_n_sq_raw_by_coord, 0.0)),
        }
    )
    return summary


# --- Debiasing corrections ---

def compute_xi_debias_term(
    config: RunConfig,
    Xi_hat_folds: Any,
    fold_masks: Any,
    W_oos_folds: Any,
    alpha_xi: Any,
) -> Any:
    v_xi_correction_folds = []
    v_correction_folds = []

    for idx1, mask1 in enumerate(fold_masks):
        Xi1 = Xi_hat_folds[idx1]
        v_xi_row = []
        v_row = []

        for idx2, mask2 in enumerate(fold_masks):
            Xi2 = Xi_hat_folds[idx2]
            DL = gaussian_grad1(Xi1[~mask1], Xi2, alpha_xi)
            LapL = gaussian_cross_laplacian(Xi1[~mask1], Xi2[~mask2], alpha_xi)

            # r_{v,xi}: sum_j xi_j * <v_j, d l(., xi)>
            v_xi_term = 0.0
            for j in range(config.dimy):
                centered_DL = DL[:, mask2, j] - jnp.mean(DL[:, ~mask2, j], axis=1)[:, None]
                v_xi_term = v_xi_term + (Xi1[mask1, j, None] * W_oos_folds[idx1, j]) @ centered_DL

            # r_v: sum_{j1,j2} xi_{j1} xi_{j2} <v_{j1}, v_{j2}>
            v_term = 0.0
            for j1 in range(config.dimy):
                for j2 in range(config.dimy):
                    kernel_part = (
                        W_oos_folds[idx1, j1]
                        @ LapL[:, :, j1, j2]
                        @ W_oos_folds[idx2, j2].T
                    )
                    v_term = v_term + jnp.einsum("i,j,ij->ij", Xi1[mask1, j1], Xi2[mask2, j2], kernel_part)

            v_xi_row.append(v_xi_term)
            v_row.append(v_term)

        v_xi_correction_folds.append(v_xi_row)
        v_correction_folds.append(v_row)

    v_xi_correction_folds = jnp.array(v_xi_correction_folds)
    v_correction_folds = jnp.array(v_correction_folds)
    # The notebook computes r_{xi,v} by transposing r_{v,xi}.
    return v_correction_folds - v_xi_correction_folds - jnp.transpose(v_xi_correction_folds, (1, 0, 3, 2))


# --- HSIC summaries and intervals ---

def offdiag_average(M: Any) -> Any:
    n = M.shape[0]
    return (jnp.sum(M) - jnp.trace(M)) / (n * (n - 1))


def crossfit_v_from_blocks(H_folds: Any) -> Any:
    return jnp.mean(H_folds)


def crossfit_u_from_blocks(H_folds: Any) -> Any:
    num_folds = H_folds.shape[0]
    diag_sum = sum(offdiag_average(H_folds[i, i]) for i in range(num_folds))
    offdiag_sum = sum(
        jnp.mean(H_folds[i, j])
        for i in range(num_folds)
        for j in range(num_folds)
        if i != j
    )
    return (diag_sum + offdiag_sum) / (num_folds**2)


def bootstrap_sqrt_hsic_ci_crossfit(
    H_folds: Any,
    num_folds: int,
    size_fold: int,
    n_boot: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    hsic_est = crossfit_v_from_blocks(H_folds)

    def bootstrap_distance(subkey: Any) -> Any:
        # Resample within each fold so the cross-fit geometry stays intact.
        keys = jax.random.split(subkey, num_folds)
        idx = jnp.stack(
            [jax.random.randint(keys[k], shape=(size_fold,), minval=0, maxval=size_fold) for k in range(num_folds)]
        )

        cross_term = 0.0
        boot_term = 0.0
        for k in range(num_folds):
            for l in range(num_folds):
                cross_term = cross_term + jnp.mean(H_folds[k, l][:, idx[l]])
                boot_term = boot_term + jnp.mean(H_folds[k, l][idx[k][:, None], idx[l][None, :]])

        scale = num_folds**2
        return hsic_est - 2.0 * cross_term / scale + boot_term / scale

    keys = jax.random.split(jax.random.PRNGKey(seed), n_boot)
    boot_dist = jax.lax.map(bootstrap_distance, keys)
    q_val = jnp.quantile(boot_dist, 1.0 - alpha)
    center_sq = jnp.maximum(hsic_est, 0.0)
    boot_dist_nonnegative = jnp.maximum(boot_dist, 0.0)
    sqrt_hsic = jnp.sqrt(jnp.maximum(hsic_est, 0.0))
    radius = jnp.sqrt(jnp.maximum(q_val, 0.0))
    # Invert the reverse-triangle confidence sets. Zero is excluded whenever the
    # bootstrap radius quantile is smaller than ||hat psi||, so the p-value is the
    # upper-tail mass of bootstrap squared distances beyond ||hat psi||^2.
    p_value = (1.0 + jnp.sum(boot_dist_nonnegative >= center_sq)) / (n_boot + 1.0)
    zero_inclusion_quantile = jnp.mean(boot_dist_nonnegative < center_sq)

    return {
        "hsic_est": hsic_est,
        "sqrt_hsic": sqrt_hsic,
        "radius": radius,
        "q_val": q_val,
        "p_value": p_value,
        "zero_inclusion_quantile": zero_inclusion_quantile,
        "ci_lower": jnp.maximum(0.0, sqrt_hsic - radius),
        "ci_upper": sqrt_hsic + radius,
    }


bootstrap_sqrt_hsic_ci_crossfit = partial(
    jax.jit,
    static_argnames=("num_folds", "size_fold", "n_boot"),
)(bootstrap_sqrt_hsic_ci_crossfit)


def blocks_to_full_matrix(H_folds: Any) -> Any:
    num_folds, _, size_fold, _ = H_folds.shape
    return jnp.transpose(H_folds, (0, 2, 1, 3)).reshape(num_folds * size_fold, num_folds * size_fold)


def crossfit_u_point_scores(H_folds: Any) -> Any:
    num_folds, _, size_fold, _ = H_folds.shape
    scores_by_fold = []

    for k in range(num_folds):
        same_block = H_folds[k, k]
        same_scores = (jnp.sum(same_block, axis=1) - jnp.diag(same_block)) / jnp.maximum(size_fold - 1, 1)

        total = same_scores
        for l in range(num_folds):
            if l != k:
                total = total + jnp.mean(H_folds[k, l], axis=1)

        # This fold-aware Hajek score averages back to crossfit_u_from_blocks.
        scores_by_fold.append(total / num_folds)

    return jnp.concatenate(scores_by_fold, axis=0)


def crossfit_point_scores(H_folds: Any, score_mode: str = "plugin") -> Any:
    if score_mode == "plugin":
        H_full = blocks_to_full_matrix(H_folds)
        row_sums = jnp.sum(H_full, axis=1)
        n = H_full.shape[0]
        return row_sums / n
    if score_mode == "loo":
        return crossfit_u_point_scores(H_folds)
    raise ValueError("Need score_mode in {'plugin', 'loo'}.")


def delta_hsic_ci_widths_crossfit(
    H_folds: Any,
    alpha: float,
    score_mode: str = "plugin",
    hsic_ref: Any | None = None,
) -> dict[str, Any]:
    scores = crossfit_point_scores(H_folds, score_mode=score_mode)
    n = scores.shape[0]
    hsic_v = crossfit_v_from_blocks(H_folds)
    hsic_u = crossfit_u_from_blocks(H_folds)
    score_mean = jnp.mean(scores)

    if hsic_ref is None:
        hsic_ref = hsic_v if score_mode == "plugin" else hsic_u

    if score_mode == "plugin":
        score_var = jnp.maximum(0.0, jnp.mean((scores - hsic_v) ** 2))
    elif score_mode == "loo":
        score_var = jnp.maximum(0.0, jnp.sum((scores - score_mean) ** 2) / jnp.maximum(n - 1, 1))
    else:
        raise ValueError("Need score_mode in {'plugin', 'loo'}.")

    z = jax.scipy.stats.norm.ppf(1.0 - alpha / 2.0)
    se_hsic = 2.0 * jnp.sqrt(score_var / n)
    hsic_half_width = z * se_hsic
    se_sqrt = jnp.where(hsic_ref > 0.0, jnp.sqrt(score_var / (n * hsic_ref)), jnp.nan)
    sqrt_half_width = z * se_sqrt

    return {
        "n": n,
        "hsic_v": hsic_v,
        "hsic_ref": hsic_ref,
        "score_mode": score_mode,
        "score_mean": score_mean,
        "score_var": score_var,
        "se_hsic": se_hsic,
        "se_sqrt": se_sqrt,
        "hsic_half_width": hsic_half_width,
        "sqrt_half_width": sqrt_half_width,
    }


def delta_hsic_ci_from_center(hsic_center: Any, widths: dict[str, Any]) -> dict[str, Any]:
    """Intersect the squared-HSIC Wald interval with [0, infinity), then square-root.

    Keep the signed center on the squared scale. The legacy se_sqrt and
    sqrt_half_width entries describe the direct norm delta approximation only;
    they are not used for these endpoints. An empty set has reversed finite
    norm endpoints (1, 0), zero width, and cannot cover any target.
    """
    hsic_lower = jnp.maximum(0.0, hsic_center - widths["hsic_half_width"])
    hsic_upper = hsic_center + widths["hsic_half_width"]
    empty = hsic_upper < 0.0
    sqrt_lower = jnp.where(empty, 1.0, jnp.sqrt(hsic_lower))
    sqrt_upper = jnp.sqrt(jnp.maximum(hsic_upper, 0.0))
    return {
        **widths,
        "hsic_center": hsic_center,
        "sqrt_hsic_center": jnp.sqrt(jnp.maximum(hsic_center, 0.0)),
        "hsic_lower": hsic_lower,
        "hsic_upper": hsic_upper,
        "sqrt_lower": sqrt_lower,
        "sqrt_upper": sqrt_upper,
        "sqrt_width": jnp.maximum(0.0, sqrt_upper - sqrt_lower),
        "empty": empty,
        "sqrt_protocol": "sqrt_squared_wald",
    }


def permuted_naive_hsic_u(H_folds: Any, WX_folds: Any, WXi_folds: Any, size_fold: int, subkey: Any) -> Any:
    num_folds = H_folds.shape[0]
    keys = jax.random.split(subkey, num_folds)
    perms = jnp.stack([jax.random.permutation(keys[k], size_fold) for k in range(num_folds)])
    permuted_blocks = []

    for k in range(num_folds):
        row = []
        for l in range(num_folds):
            WXi_perm = WXi_folds[k, l][perms[k]][:, perms[l]]
            row.append(WX_folds[k, l] * WXi_perm)
        permuted_blocks.append(row)

    return crossfit_u_from_blocks(jnp.array(permuted_blocks))


def permute_full_gram_within_folds(G: Any, fold_indices: Any, perms: Any) -> Any:
    out = jnp.zeros_like(G)
    num_folds, size_fold = perms.shape

    for k in range(num_folds):
        idx_k = fold_indices[k]
        src_k = idx_k[perms[k]]
        for l in range(num_folds):
            idx_l = fold_indices[l]
            src_l = idx_l[perms[l]]
            out = out.at[idx_k[:, None], idx_l[None, :]].set(G[src_k[:, None], src_l[None, :]])

    return out


def permuted_naive_hsic_u_by_x(K: Any, WXi_folds: Any, fold_indices: Any, size_fold: int, subkey: Any) -> Any:
    num_folds = WXi_folds.shape[0]
    keys = jax.random.split(subkey, num_folds)
    perms = jnp.stack([jax.random.permutation(keys[k], size_fold) for k in range(num_folds)])
    K_perm = permute_full_gram_within_folds(K, fold_indices, perms)
    WX_perm_folds = centered_blocks_from_full_gram_indices(K_perm, fold_indices)
    return crossfit_u_from_blocks(WX_perm_folds * WXi_folds)


def permuted_hsic_u_from_grams(K: Any, L: Any, subkey: Any) -> Any:
    perm = jax.random.permutation(subkey, K.shape[0])
    return hsic_unbiased(K, L[perm][:, perm])


def x_permuted_hsic_u_from_grams(K: Any, L: Any, subkey: Any) -> Any:
    perm = jax.random.permutation(subkey, K.shape[0])
    return hsic_unbiased(K[perm][:, perm], L)


def summarize_permutation_distribution(observed: Any, null_dist: Any) -> dict[str, Any]:
    return {
        "statistic": observed,
        "p_value": jnp.mean(null_dist >= observed),
        "null_mean": jnp.mean(null_dist),
        "null_q05": jnp.quantile(null_dist, 0.05),
        "null_q50": jnp.quantile(null_dist, 0.50),
        "null_q95": jnp.quantile(null_dist, 0.95),
    }


def fit_v_oos_weights_for_split(
    config: RunConfig,
    X_train: Any,
    X_test: Any,
    Xi_train: Any,
    alpha_xi: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    n_train = X_train.shape[0]
    k_cv_folds = min(config.v_cv_folds, int(n_train))
    if k_cv_folds < 2 or n_train % k_cv_folds != 0:
        raise ValueError("Split v-fit needs at least two equal-size CV folds in the prefix training sample.")

    cv_fold_size = n_train // k_cv_folds
    cv_fold_ids = jnp.arange(n_train) // cv_fold_size
    base_length_scale = jnp.sqrt(jnp.maximum(median_heuristic(X_train, X_train), 1e-12))
    dists_train = jnp.sqrt(jnp.maximum(pairwise_sq_dists(X_train, X_train), 0.0))
    dists_test_train = jnp.sqrt(jnp.maximum(pairwise_sq_dists(X_test, X_train), 0.0))
    bandwidth_multipliers = jnp.asarray(config.v_bandwidth_multipliers, dtype=jnp.float64)
    etas = jnp.asarray(config.v_etas, dtype=jnp.float64)
    nus = jnp.asarray(config.v_nus, dtype=jnp.float64)
    residualized_centered = config.v_fit_mode == "residualized_centered"

    Lap_fit = gaussian_cross_laplacian(Xi_train, Xi_train, alpha_xi)
    weights = []
    params = []
    for j in range(config.dimy):
        Lap_fit_j = Lap_fit[..., j, j]
        mses = v_weight_cv_grid(
            dists_train,
            Lap_fit_j,
            base_length_scale,
            bandwidth_multipliers,
            etas,
            nus,
            cv_fold_ids,
            cv_fold_size,
            k_cv_folds,
            residualized_centered,
        )

        idxmin = int(jnp.argmin(mses))
        n_mult = len(config.v_bandwidth_multipliers)
        n_eta = len(config.v_etas)
        idx_nu = idxmin // (n_mult * n_eta)
        rem = idxmin % (n_mult * n_eta)
        idx_mult = rem // n_eta
        idx_eta = rem % n_eta

        nu_best = float(nus[idx_nu])
        scale_best = float(bandwidth_multipliers[idx_mult])
        eta_best = float(etas[idx_eta])
        length_scale_best = scale_best * float(base_length_scale)
        Q_train_train = matern_from_dists(dists_train, length_scale_best, nu_best)
        Q_test_train = matern_from_dists(dists_test_train, length_scale_best, nu_best)
        if residualized_centered:
            weights.append(residualized_krr_weight_matrix(Q_train_train, Q_test_train, eta_best))
        else:
            weights.append(standard_krr_weight_matrix(Q_train_train, Q_test_train, eta_best))
        params.append(
            {
                "coord": int(j),
                "nu": nu_best,
                "scale_multiplier": scale_best,
                "length_scale": length_scale_best,
                "eta": eta_best,
                "cv_mse": float(mses[idx_nu, idx_mult, idx_eta]),
            }
        )

    return jnp.array(weights), params


def debiased_residual_kernel_for_split(
    config: RunConfig,
    X_train: Any,
    X_test: Any,
    Xi_train: Any,
    Xi_test: Any,
    alpha_xi: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    W_test, params = fit_v_oos_weights_for_split(config, X_train, X_test, Xi_train, alpha_xi)
    L_test = gaussian_gram(Xi_test, Xi_test, alpha_xi)
    DL_train_test = gaussian_grad1(Xi_train, Xi_test, alpha_xi)
    Lap_train = gaussian_cross_laplacian(Xi_train, Xi_train, alpha_xi)

    v_xi_term = 0.0
    for j in range(config.dimy):
        v_xi_term = v_xi_term + (Xi_test[:, j, None] * W_test[j]) @ DL_train_test[:, :, j]

    v_term = 0.0
    for j1 in range(config.dimy):
        for j2 in range(config.dimy):
            kernel_part = W_test[j1] @ Lap_train[:, :, j1, j2] @ W_test[j2].T
            v_term = v_term + jnp.einsum("i,j,ij->ij", Xi_test[:, j1], Xi_test[:, j2], kernel_part)

    return L_test + v_term - v_xi_term - v_xi_term.T, params


def standard_residual_hsic_tests(
    config: RunConfig,
    X: Any,
    Y: Any,
    signal: Any,
    fold_masks: Any,
) -> list[dict[str, Any]]:
    prefixes = standard_residual_hsic_prefixes(config)

    fold_masks_np = np.asarray(fold_masks)
    rows = []
    for prefix in prefixes:
        train_mask_np = np.any(fold_masks_np[:prefix], axis=0)
        test_mask_np = ~train_mask_np
        train_mask = jnp.asarray(train_mask_np)
        test_mask = jnp.asarray(test_mask_np)

        pred_train, pred_test, mean_diag = fit_mean_on_split(
            config,
            X,
            Y,
            signal,
            train_mask_np,
            test_mask_np,
            split_id=int(prefix),
        )
        X_train = X[train_mask]
        Y_train = Y[train_mask]
        X_test = X[test_mask]
        Y_test = Y[test_mask]
        Xi_train = Y_train - pred_train
        Xi_test = Y_test - pred_test

        alpha_x_test = jnp.maximum(median_heuristic(X_test, X_test), 1e-12)
        alpha_xi_test = jnp.maximum(median_heuristic(Xi_test, Xi_test), 1e-12)
        K_test = gaussian_gram(X_test, X_test, alpha_x_test)
        L_test = gaussian_gram(Xi_test, Xi_test, alpha_xi_test)
        hsic_u = hsic_unbiased(K_test, L_test)

        perm_keys = jax.random.split(
            jax.random.fold_in(jax.random.PRNGKey(config.bootstrap_seed + 10_000), int(prefix)),
            config.bootstrap_B,
        )
        null_dist = jax.lax.map(lambda subkey: x_permuted_hsic_u_from_grams(K_test, L_test, subkey), perm_keys)
        perm_summary = summarize_permutation_distribution(hsic_u, null_dist)
        debiased_hsic_u = None
        debiased_p_value = None
        debiased_null_mean = None
        debiased_null_q05 = None
        debiased_null_q50 = None
        debiased_null_q95 = None
        v_fit = []
        if config.compute_debiased_objects:
            L_debiased, v_fit = debiased_residual_kernel_for_split(
                config,
                X_train,
                X_test,
                Xi_train,
                Xi_test,
                alpha_xi_test,
            )
            debiased_hsic_u = hsic_unbiased(K_test, L_debiased)
            debiased_null_dist = jax.lax.map(
                lambda subkey: x_permuted_hsic_u_from_grams(K_test, L_debiased, subkey),
                perm_keys,
            )
            debiased_summary = summarize_permutation_distribution(debiased_hsic_u, debiased_null_dist)
            debiased_p_value = debiased_summary["p_value"]
            debiased_null_mean = debiased_summary["null_mean"]
            debiased_null_q05 = debiased_summary["null_q05"]
            debiased_null_q50 = debiased_summary["null_q50"]
            debiased_null_q95 = debiased_summary["null_q95"]

        rows.append(
            {
                "fit_prefix_folds": int(prefix),
                "fit_folds": list(range(int(prefix))),
                "test_folds": list(range(int(prefix), config.num_folds)),
                "n_fit": int(np.sum(train_mask_np)),
                "n_test": int(np.sum(test_mask_np)),
                "alpha_x": alpha_x_test,
                "alpha_xi": alpha_xi_test,
                "naive_hsic_u": hsic_u,
                "naive_sqrt_hsic_u": jnp.sqrt(jnp.maximum(hsic_u, 0.0)),
                "naive_hsic_u_p_value": perm_summary["p_value"],
                "null_mean": perm_summary["null_mean"],
                "null_q05": perm_summary["null_q05"],
                "null_q50": perm_summary["null_q50"],
                "null_q95": perm_summary["null_q95"],
                "mean_fit": mean_diag,
                "v_fit": v_fit,
                "debiased_hsic_u": debiased_hsic_u,
                "debiased_sqrt_hsic_u": jnp.sqrt(jnp.maximum(debiased_hsic_u, 0.0))
                if debiased_hsic_u is not None
                else None,
                "debiased_hsic_u_p_value": debiased_p_value,
                "debiased_null_mean": debiased_null_mean,
                "debiased_null_q05": debiased_null_q05,
                "debiased_null_q50": debiased_null_q50,
                "debiased_null_q95": debiased_null_q95,
            }
        )

    return rows


# --- Population oracle for general dimensions ---

def oracle_residual_kernel_mean(
    state: ToyState,
    config: RunConfig,
    X_left: np.ndarray,
    X_right: np.ndarray,
    alpha_xi: float,
) -> np.ndarray:
    # If eps|X=x is diagonal Gaussian, then
    # E exp(-||eps_x - eps_x'||^2 / alpha)
    # has a closed form coordinatewise product.
    sigma_left = np.asarray(eps_std_fn(state, config, jnp.asarray(X_left)))
    sigma_right = np.asarray(eps_std_fn(state, config, jnp.asarray(X_right)))
    var_sum = sigma_left[:, None, :] ** 2 + sigma_right[None, :, :] ** 2
    return np.exp(-0.5 * np.sum(np.log1p(2.0 * var_sum / float(alpha_xi)), axis=-1))


def weighted_center(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    mean_A = A @ w
    grand_A = w @ mean_A
    return A - mean_A[:, None] - mean_A[None, :] + grand_A


def population_oracle_hsic(
    state: ToyState,
    config: RunConfig,
    alpha_x: float,
    alpha_xi: float,
) -> dict[str, Any]:
    # Use a tensor-product Legendre grid on [xmin, xmax]^dimx.
    # We cap the total number of quadrature points for sanity.
    m = max(2, int(math.floor(config.oracle_max_points ** (1.0 / config.dimx))))
    while m**config.dimx > config.oracle_max_points and m > 2:
        m -= 1

    nodes, weights = np.polynomial.legendre.leggauss(m)
    x_1d = 0.5 * ((config.xmax - config.xmin) * nodes + (config.xmax + config.xmin))
    w_1d = 0.5 * weights

    mesh_x = np.meshgrid(*([x_1d] * config.dimx), indexing="ij")
    mesh_w = np.meshgrid(*([w_1d] * config.dimx), indexing="ij")
    x_grid = np.stack([g.reshape(-1) for g in mesh_x], axis=1)
    w_grid = np.prod(np.stack(mesh_w, axis=0), axis=0).reshape(-1)

    sq_x = np.sum((x_grid[:, None, :] - x_grid[None, :, :]) ** 2, axis=-1)
    K_grid = np.exp(-sq_x / float(alpha_x))
    Q_grid = oracle_residual_kernel_mean(state, config, x_grid, x_grid, alpha_xi)

    Kc_grid = weighted_center(K_grid, w_grid)
    Qc_grid = weighted_center(Q_grid, w_grid)
    hsic = float(w_grid @ ((Kc_grid * Qc_grid) @ w_grid))

    return {
        "hsic": hsic,
        "sqrt_hsic": math.sqrt(max(hsic, 0.0)),
        "grid_points_per_dim": int(m),
        "grid_points_total": int(w_grid.shape[0]),
    }

def summarize_bootstrap_ci(interval: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "hsic_est",
        "sqrt_hsic",
        "radius",
        "q_val",
        "p_value",
        "zero_inclusion_quantile",
        "ci_lower",
        "ci_upper",
    ]
    return {key: interval[key] for key in keys}


def summarize_delta_ci(interval: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "n",
        "hsic_v",
        "hsic_ref",
        "score_mode",
        "score_mean",
        "score_var",
        "se_hsic",
        "se_sqrt",
        "hsic_half_width",
        "sqrt_half_width",
        "hsic_center",
        "sqrt_hsic_center",
        "hsic_lower",
        "hsic_upper",
        "sqrt_lower",
        "sqrt_upper",
        "sqrt_width",
        "empty",
        "sqrt_protocol",
    ]
    return {key: interval[key] for key in keys}


def get_backend_info() -> dict[str, Any]:
    try:
        devices = jax.devices()
    except RuntimeError as exc:
        raise RuntimeError(
            "JAX backend initialization failed. "
            "If you want a quick workaround, rerun with `--jax-platform cpu` or set "
            "`JAX_PLATFORMS=cpu` before importing this module. "
            f"Original error: {exc}"
        ) from exc

    return {
        "default_backend": jax.default_backend(),
        "devices": [f"{device.platform}:{device.id}:{device.device_kind}" for device in devices],
        "device_count": len(devices),
    }


# --- Logging and YAML output ---

def write_scalar(writer: Any, name: str, value: Any, step: int = 0) -> None:
    if writer is None:
        return
    try:
        writer.add_scalar(name, float(np.asarray(value)), step)
    except (TypeError, ValueError):
        if hasattr(writer, "add_text"):
            writer.add_text(name, str(value), step)


def log_results(writer: Any, results: dict[str, Any]) -> None:
    if writer is None:
        return

    for section_name in ["oracle", "estimators", "sqrt_hsic"]:
        for key, value in results[section_name].items():
            write_scalar(writer, f"{section_name}/{key}", value)

    for key, value in results["population_oracle"].items():
        if "grid" not in key:
            write_scalar(writer, f"population_oracle/{key}", value)

    for interval_name, interval in results["intervals"].items():
        for key, value in interval.items():
            if key not in {"n"}:
                write_scalar(writer, f"intervals/{interval_name}/{key}", value)

    for entry in results["mean_fit"]:
        step = entry["fold"]
        write_scalar(writer, "mean_fit/cv_mse", entry["cv_mse"], step)
        write_scalar(writer, "mean_fit/heldout_response_mse", entry["heldout_response_mse"], step)
        write_scalar(writer, "mean_fit/oracle_response_mse", entry["oracle_response_mse"], step)
        write_scalar(writer, "mean_fit/noise_variance_term", entry["noise_variance_term"], step)
        write_scalar(writer, "mean_fit/signal_mse_term", entry["signal_mse_term"], step)
        write_scalar(writer, "mean_fit/decomposition_cross_term", entry["decomposition_cross_term"], step)
        write_scalar(writer, "mean_fit/response_mse_minus_oracle", entry["response_mse_minus_oracle"], step)
        write_scalar(writer, "mean_fit/heldout_signal_mse", entry["heldout_signal_mse"], step)

    for entry in results["v_fit"]:
        step = entry["fold"] * results["config"]["dimy"] + entry["coord"]
        prefix = f"v_fit/fold_{entry['fold']}_coord_{entry['coord']}"
        write_scalar(writer, f"{prefix}/oracle_risk", entry["oracle_risk"], step)
        write_scalar(writer, f"{prefix}/fitted_risk", entry["fitted_risk"], step)
        write_scalar(writer, f"{prefix}/const_risk", entry["const_risk"], step)
        write_scalar(writer, f"{prefix}/cv_mse", entry["cv_mse"], step)
        write_scalar(writer, f"{prefix}/excess_over_oracle", entry["excess_over_oracle"], step)
        write_scalar(writer, f"{prefix}/improvement_over_const", entry["improvement_over_const"], step)

    for entry in results.get("v_fit_oos", []):
        step = entry["fold"] * results["config"]["dimy"] + entry["coord"]
        prefix = f"v_fit_oos/fold_{entry['fold']}_coord_{entry['coord']}"
        write_scalar(writer, f"{prefix}/oracle_risk", entry["oracle_risk"], step)
        write_scalar(writer, f"{prefix}/fitted_risk", entry["fitted_risk"], step)
        write_scalar(writer, f"{prefix}/cv_mse", entry["cv_mse"], step)
        write_scalar(writer, f"{prefix}/excess_over_oracle", entry["excess_over_oracle"], step)

    for key, value in results.get("rate_diagnostics", {}).items():
        if isinstance(value, (str, list)) or value is None:
            continue
        if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0:
            continue
        write_scalar(writer, f"rate_diagnostics/{key}", value)

    for key, value in results.get("plugin_bias_diagnostic", {}).items():
        if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0:
            for coord_idx, coord_value in enumerate(np.asarray(value)):
                write_scalar(writer, f"plugin_bias_diagnostic/{key}/coord_{coord_idx}", coord_value, coord_idx)
            continue
        write_scalar(writer, f"plugin_bias_diagnostic/{key}", value)


def to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_python(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()
    return value


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_python(payload), handle, sort_keys=False)


def run_experiment(config: RunConfig, writer: Any | None = None) -> dict[str, Any]:
    validate_config(config)
    backend_info = get_backend_info()

    def sqrt_nonnegative_or_none(value: Any) -> Any:
        if value is None:
            return None
        return jnp.sqrt(jnp.maximum(value, 0.0))

    # 1. Simulate one dataset from the toy DGP.
    state, X, signal, eps, Y = simulate_data(config)
    size_fold = config.N // config.num_folds
    fold_masks = build_fold_masks(config.N, config.num_folds)
    fold_indices = fold_indices_from_masks(fold_masks, size_fold)

    # 2. Fit m^{-k} and build residuals.
    mean_forecasts, mean_diagnostics = fit_mean_forecasts(config, X, Y, signal, fold_masks)
    Xi_hat_folds = Y[None, :, :] - mean_forecasts
    Xi_hat_oos = jnp.concatenate([Xi_hat_folds[idx][mask] for idx, mask in enumerate(fold_masks)], axis=0)

    # 3. Build the X and residual kernel blocks with the notebook centering.
    alpha_x = jnp.maximum(median_heuristic(X, X), 1e-12)
    K = gaussian_gram(X, X, alpha_x)
    WX_folds = centered_blocks_from_full_gram(K, fold_masks)

    alpha_xi = jnp.maximum(median_heuristic(Xi_hat_oos, Xi_hat_oos), 1e-12)
    L_folds = []
    WXi_folds = []
    for idx1, mask1 in enumerate(fold_masks):
        L_row = []
        WXi_row = []
        Xi1 = Xi_hat_folds[idx1]
        for idx2, mask2 in enumerate(fold_masks):
            Xi2 = Xi_hat_folds[idx2]
            L = gaussian_gram(Xi1, Xi2, alpha_xi)
            L_row.append(L)
            WXi_row.append(center_cross_block(L, mask1, mask2))
        L_folds.append(L_row)
        WXi_folds.append(WXi_row)
    L_folds = jnp.array(L_folds)
    WXi_folds = jnp.array(WXi_folds)

    # 4. Optionally fit v_j^{-k} and assemble the debiasing term.
    if config.compute_debiased_objects:
        W_is_folds, W_oos_folds, best_v_params = fit_v_weights(config, X, Xi_hat_folds, fold_masks, alpha_xi)
        compute_v_oos_diagnostics = config.compute_v_diagnostics or config.compute_crossfit_rate_diagnostics
        if config.compute_v_diagnostics:
            v_diagnostics = evaluate_v_regression_diagnostics(
                config,
                state,
                X,
                Xi_hat_folds,
                fold_masks,
                W_is_folds,
                best_v_params,
                alpha_xi,
            )
        else:
            v_diagnostics = []

        if compute_v_oos_diagnostics:
            v_oos_diagnostics = evaluate_v_oos_regression_diagnostics(
                config,
                state,
                X,
                Xi_hat_folds,
                fold_masks,
                W_oos_folds,
                best_v_params,
                alpha_xi,
            )
        else:
            v_oos_diagnostics = []

        if config.compute_crossfit_rate_diagnostics:
            rate_diagnostics = summarize_crossfit_rate_diagnostics(config.N, mean_diagnostics, v_oos_diagnostics)
        else:
            rate_diagnostics = {}
    else:
        W_oos_folds = None
        best_v_params = []
        v_diagnostics = []
        v_oos_diagnostics = []
        rate_diagnostics = {}

    if config.compute_plugin_bias_diagnostic:
        plugin_bias_diagnostic = summarize_plugin_bias_diagnostic(
            config,
            state,
            X,
            signal,
            mean_forecasts,
            fold_masks,
            alpha_x,
            alpha_xi,
        )
    else:
        plugin_bias_diagnostic = {}

    if config.compute_debiased_objects:
        xi_debias_term = compute_xi_debias_term(config, Xi_hat_folds, fold_masks, W_oos_folds, alpha_xi)
    else:
        xi_debias_term = None

    # 5. Compare oracle, naive, and debiased HSIC estimators.
    E_full = gaussian_gram(eps, eps, alpha_xi)
    W_eps_full = center_gram(E_full)
    W_x_full = center_gram(K)

    oracle_hsic_v = jnp.mean(W_eps_full * W_x_full)
    oracle_hsic_u = hsic_unbiased(K, E_full)
    W_eps_folds = centered_blocks_from_full_gram(E_full, fold_masks)

    oracle_crossfit_hsic_v = crossfit_v_from_blocks(WX_folds * W_eps_folds)
    oracle_crossfit_hsic_u = crossfit_u_from_blocks(WX_folds * W_eps_folds)

    naive_D_folds = WX_folds * WXi_folds

    naive_hsic_v = crossfit_v_from_blocks(naive_D_folds)
    naive_hsic_u = crossfit_u_from_blocks(naive_D_folds)
    if config.compute_debiased_objects:
        debiased_D_folds = WX_folds * (WXi_folds + xi_debias_term)
        debiased_hsic_v = crossfit_v_from_blocks(debiased_D_folds)
        debiased_hsic_u = crossfit_u_from_blocks(debiased_D_folds)
    else:
        debiased_D_folds = None
        debiased_hsic_v = None
        debiased_hsic_u = None

    intervals = {}
    if config.compute_bootstrap_intervals:
        boot_naive = bootstrap_sqrt_hsic_ci_crossfit(
            naive_D_folds,
            config.num_folds,
            size_fold,
            config.bootstrap_B,
            config.ci_alpha,
            config.bootstrap_seed,
        )
        intervals["bootstrap_naive"] = summarize_bootstrap_ci(boot_naive)
        if config.compute_debiased_objects:
            boot_debiased = bootstrap_sqrt_hsic_ci_crossfit(
                debiased_D_folds,
                config.num_folds,
                size_fold,
                config.bootstrap_B,
                config.ci_alpha,
                config.bootstrap_seed,
            )
            intervals["bootstrap_debiased"] = summarize_bootstrap_ci(boot_debiased)

    if config.compute_delta_intervals:
        delta_naive_widths_v = delta_hsic_ci_widths_crossfit(
            naive_D_folds,
            config.ci_alpha,
            score_mode="plugin",
            hsic_ref=naive_hsic_v,
        )
        delta_naive_widths_u = delta_hsic_ci_widths_crossfit(
            naive_D_folds,
            config.ci_alpha,
            score_mode="plugin",
            hsic_ref=(naive_hsic_u if config.delta_u_variance_reference == "u" else naive_hsic_v),
        )
        delta_naive_widths_v_loo = delta_hsic_ci_widths_crossfit(
            naive_D_folds,
            config.ci_alpha,
            score_mode="loo",
            hsic_ref=naive_hsic_v,
        )
        delta_naive_widths_u_loo = delta_hsic_ci_widths_crossfit(
            naive_D_folds,
            config.ci_alpha,
            score_mode="loo",
            hsic_ref=(naive_hsic_u if config.delta_u_variance_reference == "u" else naive_hsic_v),
        )

        delta_naive_v = delta_hsic_ci_from_center(naive_hsic_v, delta_naive_widths_v)
        delta_naive_u = delta_hsic_ci_from_center(naive_hsic_u, delta_naive_widths_u)
        delta_naive_v_loo = delta_hsic_ci_from_center(naive_hsic_v, delta_naive_widths_v_loo)
        delta_naive_u_loo = delta_hsic_ci_from_center(naive_hsic_u, delta_naive_widths_u_loo)

        intervals.update(
            {
                "delta_naive_v": summarize_delta_ci(delta_naive_v),
                "delta_naive_u": summarize_delta_ci(delta_naive_u),
                "delta_naive_v_loo": summarize_delta_ci(delta_naive_v_loo),
                "delta_naive_u_loo": summarize_delta_ci(delta_naive_u_loo),
            }
        )
        if config.compute_debiased_objects:
            delta_debiased_widths_v = delta_hsic_ci_widths_crossfit(
                debiased_D_folds,
                config.ci_alpha,
                score_mode="plugin",
                hsic_ref=debiased_hsic_v,
            )
            delta_debiased_widths_u = delta_hsic_ci_widths_crossfit(
                debiased_D_folds,
                config.ci_alpha,
                score_mode="plugin",
                # Retained direct-norm diagnostic; the current squared-Wald interval uses neither divisor.
                hsic_ref=(debiased_hsic_u if config.delta_u_variance_reference == "u" else debiased_hsic_v),
            )
            delta_debiased_widths_v_loo = delta_hsic_ci_widths_crossfit(
                debiased_D_folds,
                config.ci_alpha,
                score_mode="loo",
                hsic_ref=debiased_hsic_v,
            )
            delta_debiased_widths_u_loo = delta_hsic_ci_widths_crossfit(
                debiased_D_folds,
                config.ci_alpha,
                score_mode="loo",
                hsic_ref=(debiased_hsic_u if config.delta_u_variance_reference == "u" else debiased_hsic_v),
            )
            delta_debiased_v = delta_hsic_ci_from_center(debiased_hsic_v, delta_debiased_widths_v)
            delta_debiased_u = delta_hsic_ci_from_center(debiased_hsic_u, delta_debiased_widths_u)
            delta_debiased_v_loo = delta_hsic_ci_from_center(debiased_hsic_v, delta_debiased_widths_v_loo)
            delta_debiased_u_loo = delta_hsic_ci_from_center(debiased_hsic_u, delta_debiased_widths_u_loo)
            intervals.update(
                {
                    "delta_debiased_v": summarize_delta_ci(delta_debiased_v),
                    "delta_debiased_u": summarize_delta_ci(delta_debiased_u),
                    "delta_debiased_v_loo": summarize_delta_ci(delta_debiased_v_loo),
                    "delta_debiased_u_loo": summarize_delta_ci(delta_debiased_u_loo),
                }
            )

    if config.compute_permutation_test:
        perm_keys = jax.random.split(jax.random.PRNGKey(config.bootstrap_seed + 1), config.bootstrap_B)
        null_dist = jax.lax.map(
            lambda subkey: permuted_naive_hsic_u(naive_D_folds, WX_folds, WXi_folds, size_fold, subkey),
            perm_keys,
        )
        block_summary = summarize_permutation_distribution(naive_hsic_u, null_dist)
        permutation = {
            "naive_hsic_u": naive_hsic_u,
            "naive_hsic_u_p_value": block_summary["p_value"],
            "null_mean": block_summary["null_mean"],
            "null_q05": block_summary["null_q05"],
            "null_q50": block_summary["null_q50"],
            "null_q95": block_summary["null_q95"],
        }
        if config.compute_x_permutation_test:
            x_null_dist = jax.lax.map(
                lambda subkey: permuted_naive_hsic_u_by_x(K, WXi_folds, fold_indices, size_fold, subkey),
                perm_keys,
            )
            x_summary = summarize_permutation_distribution(naive_hsic_u, x_null_dist)
            permutation.update(
                {
                    "x_permuted_naive_hsic_u": naive_hsic_u,
                    "x_permuted_naive_hsic_u_p_value": x_summary["p_value"],
                    "x_permuted_null_mean": x_summary["null_mean"],
                    "x_permuted_null_q05": x_summary["null_q05"],
                    "x_permuted_null_q50": x_summary["null_q50"],
                    "x_permuted_null_q95": x_summary["null_q95"],
                }
            )
    else:
        permutation = {}

    if config.compute_standard_residual_hsic_tests:
        standard_residual_hsic = standard_residual_hsic_tests(config, X, Y, signal, fold_masks)
    else:
        standard_residual_hsic = []

    # 6. Compute the population oracle on a deterministic quadrature grid.
    if config.compute_population_oracle:
        population_oracle = population_oracle_hsic(state, config, float(alpha_x), float(alpha_xi))
    else:
        population_oracle = {}

    results = {
        "config": asdict(config),
        "backend": backend_info,
        "bandwidths": {
            "alpha_x": alpha_x,
            "alpha_xi": alpha_xi,
        },
        "oracle": {
            "hsic_v": oracle_hsic_v,
            "hsic_u": oracle_hsic_u,
            "crossfit_hsic_v": oracle_crossfit_hsic_v,
            "crossfit_hsic_u": oracle_crossfit_hsic_u,
        },
        "estimators": {
            "naive_hsic_v": naive_hsic_v,
            "naive_hsic_u": naive_hsic_u,
            "debiased_hsic_v": debiased_hsic_v,
            "debiased_hsic_u": debiased_hsic_u,
        },
        "sqrt_hsic": {
            "oracle_hsic_v": sqrt_nonnegative_or_none(oracle_hsic_v),
            "oracle_hsic_u": sqrt_nonnegative_or_none(oracle_hsic_u),
            "naive_hsic_v": sqrt_nonnegative_or_none(naive_hsic_v),
            "naive_hsic_u": sqrt_nonnegative_or_none(naive_hsic_u),
            "debiased_hsic_v": sqrt_nonnegative_or_none(debiased_hsic_v),
            "debiased_hsic_u": sqrt_nonnegative_or_none(debiased_hsic_u),
        },
        "population_oracle": population_oracle,
        "intervals": intervals,
        "permutation": permutation,
        "standard_residual_hsic": standard_residual_hsic,
        "mean_fit": mean_diagnostics,
        "v_fit": v_diagnostics,
        "v_fit_oos": v_oos_diagnostics,
        "v_fit_hyperparams": best_v_params,
        "rate_diagnostics": rate_diagnostics,
        "plugin_bias_diagnostic": plugin_bias_diagnostic,
    }

    log_results(writer, results)
    return to_python(results)
