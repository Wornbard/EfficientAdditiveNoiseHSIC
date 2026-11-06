from __future__ import annotations

import argparse
from collections import OrderedDict
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
import warnings

import jax
import jax.numpy as jnp
import nbformat
import numpy as np
import pandas as pd
import scipy
from scipy import linalg, stats
from scipy.stats import chi2
import sklearn
import statsmodels
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    IterationLimitWarning,
)

from multivariate_crossfit_core import (
    RunConfig,
    bootstrap_sqrt_hsic_ci_crossfit,
    build_fold_masks,
    center_cross_block,
    compute_xi_debias_term,
    crossfit_u_from_blocks,
    fit_mean_forecasts,
    fit_v_weights,
    fold_indices_from_masks,
    gaussian_gram,
    median_heuristic,
    permuted_naive_hsic_u_by_x,
    validate_config,
)


ROOT = Path(__file__).resolve().parent
GOF_NOTEBOOK = ROOT / "submission_toy_additive_noise_gof_1d.ipynb"
CLASSICAL_NOTEBOOK = ROOT / "toy_additive_noise_classical_diagnostics.ipynb"
RESULT_DIR = ROOT / "results/toy_additive_noise_selected_1000"
REPETITION_DIR = RESULT_DIR / "raw_repetitions"
PROGRESS_DIR = RESULT_DIR / "progress"
BASELINE_HSIC_CSV = (
    ROOT
    / "results/toy_additive_noise_gof_1d"
    / "toy_additive_noise_gof_1d_20260502_233746_repetition_records.csv"
)
BASELINE_CLASSICAL_CSV = (
    ROOT
    / "results/toy_additive_noise_classical_diagnostics"
    / "classical_diagnostics_long.csv"
)

ALPHA = 0.05
TARGET_REPETITIONS = 1_000
HOLDOUT_START = 100
HOLDOUT_STOP = HOLDOUT_START + TARGET_REPETITIONS
CENTRAL_QUANTILES = (0.25, 0.5, 0.75)
SELECTED_METHOD_KEYS = (
    "old_hsic_cf",
    "old_hsic_debiased",
    "bp_linear",
    "bp_fourier2",
    "bp_fourier4",
    "levene_b10",
    "brown_forsythe_b10",
    "quantile_central_fourier4",
)
METHOD_TITLES = {
    "old_hsic_cf": (
        "Gaussian kernel HSIC, cross-fitted plug-in permutation test"
    ),
    "old_hsic_debiased": (
        "Gaussian kernel HSIC, cross-fitted debiased test (ours)"
    ),
    "bp_linear": "BP/White, linear $X$ feature",
    "bp_fourier2": "BP/White, Fourier $X$ features ($K=2$)",
    "bp_fourier4": "BP/White, Fourier $X$ features ($K=4$)",
    "levene_b10": "Levene, $B=10$",
    "brown_forsythe_b10": "Brown-Forsythe, $B=10$",
    "quantile_central_fourier4": (
        r"Quantile regression, $\tau\in\{0.25,0.5,0.75\}$, "
        r"Fourier $X$ features ($K=4$)"
    ),
}
SCENARIOS = OrderedDict(
    [
        ((0.0, "X->Y"), "rho=0, X->Y (size)"),
        ((0.0, "Y->X"), "rho=0, Y->X (power)"),
        ((0.35, "X->Y"), "rho=0.35, X->Y (power)"),
        ((0.35, "Y->X"), "rho=0.35, Y->X (power)"),
    ]
)
REPO_DIRECTION_MAP = {"X->Y": "Y on X", "Y->X": "X on Y"}


def source_cell(path: Path, cell_id: str) -> str:
    notebook = nbformat.read(path, as_version=4)
    matches = [
        cell.source for cell in notebook.cells if cell.get("id") == cell_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one cell {cell_id!r} in {path}; found {len(matches)}."
        )
    return matches[0]


def load_notebook_definitions() -> dict:
    namespace = globals()

    dgp_source = source_cell(GOF_NOTEBOOK, "e5b540f5")
    exec(
        compile(dgp_source, f"{GOF_NOTEBOOK}#e5b540f5", "exec"),
        namespace,
    )
    config_source = source_cell(GOF_NOTEBOOK, "6d59aeca")
    exec(
        compile(config_source, f"{GOF_NOTEBOOK}#6d59aeca", "exec"),
        namespace,
    )
    hsic_source = source_cell(GOF_NOTEBOOK, "052e0f98")
    exec(
        compile(hsic_source, f"{GOF_NOTEBOOK}#052e0f98", "exec"),
        namespace,
    )

    for cell_id in ("d05644ef", "8464e6a7", "f8e6a237", "aad4f60d"):
        cell_source = source_cell(CLASSICAL_NOTEBOOK, cell_id)
        exec(
            compile(
                cell_source,
                f"{CLASSICAL_NOTEBOOK}#{cell_id}",
                "exec",
            ),
            namespace,
        )

    return {
        "dgp_sha256": hashlib.sha256(dgp_source.encode()).hexdigest(),
        "config_sha256": hashlib.sha256(config_source.encode()).hexdigest(),
        "hsic_functions_sha256": hashlib.sha256(
            hsic_source.encode()
        ).hexdigest(),
    }


SOURCE_HASHES = load_notebook_definitions()
DIRECTION_MAP = {"Y on X": "X->Y", "X on Y": "Y->X"}
REPO_DIRECTIONS = tuple(DIRECTIONS)
DATA_SEED_BASE = int(SEED) + 70_000

_BASELINE_HSIC: pd.DataFrame | None = None
_BASELINE_CLASSICAL: pd.DataFrame | None = None


def selected_crossfit_hsic(
    predictor: np.ndarray,
    response: np.ndarray,
    signal_for_diagnostics: np.ndarray,
    config: RunConfig,
    seed: int,
) -> dict:
    """Compute only the two retained HSIC outputs and shared OOF residuals."""
    X = _col(predictor)
    Y = _col(response)
    signal = _col(signal_for_diagnostics)
    fold_masks = build_fold_masks(config.N, config.num_folds)
    size_fold = config.N // config.num_folds
    fold_indices = fold_indices_from_masks(fold_masks, size_fold)

    mean_forecasts, mean_diagnostics = fit_mean_forecasts(
        config,
        X,
        Y,
        signal,
        fold_masks,
    )
    Xi_hat_folds = Y[None, :, :] - mean_forecasts
    Xi_hat_oos = jnp.concatenate(
        [
            Xi_hat_folds[k][mask]
            for k, mask in enumerate(fold_masks)
        ],
        axis=0,
    )

    alpha_x = jnp.maximum(median_heuristic(X, X), 1e-12)
    K = gaussian_gram(X, X, alpha_x)
    alpha_xi = jnp.maximum(
        median_heuristic(Xi_hat_oos, Xi_hat_oos),
        1e-12,
    )

    WX_folds = []
    WXi_folds = []
    for k, mask_k in enumerate(fold_masks):
        wx_row = []
        wxi_row = []
        Xi_k = Xi_hat_folds[k]
        for l, mask_l in enumerate(fold_masks):
            Xi_l = Xi_hat_folds[l]
            wx_row.append(center_cross_block(K, mask_k, mask_l))
            L_kl = gaussian_gram(Xi_k, Xi_l, alpha_xi)
            wxi_row.append(center_cross_block(L_kl, mask_k, mask_l))
        WX_folds.append(wx_row)
        WXi_folds.append(wxi_row)
    WX_folds = jnp.asarray(WX_folds)
    WXi_folds = jnp.asarray(WXi_folds)

    _, W_oos_folds, v_params = fit_v_weights(
        config,
        X,
        Xi_hat_folds,
        fold_masks,
        alpha_xi,
    )
    xi_debias_term = compute_xi_debias_term(
        config,
        Xi_hat_folds,
        fold_masks,
        W_oos_folds,
        alpha_xi,
    )
    naive_D = WX_folds * WXi_folds
    debiased_D = WX_folds * (WXi_folds + xi_debias_term)

    naive_permutation = permutation_pvalue_crossfit_by_x(
        naive_D,
        K,
        WXi_folds,
        fold_indices,
        size_fold,
        config.bootstrap_B,
        seed,
    )
    debiased_bootstrap = _bootstrap_summary(
        debiased_D,
        config,
        size_fold,
        seed + 3_000,
    )

    fold_fitted_values = np.asarray(mean_forecasts[:, :, 0])
    fitted_oos = np.empty(config.N)
    residual_oos = np.empty(config.N)
    for k, mask in enumerate(np.asarray(fold_masks)):
        fitted_oos[mask] = fold_fitted_values[k][mask]
        residual_oos[mask] = np.asarray(Xi_hat_folds[k][mask, 0])

    return {
        "plugin_p_value": _float(naive_permutation["p_value"]),
        "plugin_statistic": _float(naive_permutation["statistic_u"]),
        "debiased_p_value": float(debiased_bootstrap["bootstrap_p_value"]),
        "debiased_sqrt_hsic": float(debiased_bootstrap["sqrt_hsic_v"]),
        "residual_oos": residual_oos,
        "fitted_oos": fitted_oos,
        "fold_fitted_values": fold_fitted_values,
        "fold_residuals": np.asarray(Xi_hat_folds[:, :, 0]),
        "alpha_x": _float(alpha_x),
        "alpha_xi": _float(alpha_xi),
        "mean_diagnostics": mean_diagnostics,
        "v_params": v_params,
    }


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def central_quantile_test(u: np.ndarray, e: np.ndarray, basis: str) -> dict:
    """Exact central-grid calculation without fitting unused tail quantiles."""
    prepared = prepare_feature_design(u, basis)
    selected = {
        tau: quantile_wald_test(e, prepared, tau)
        for tau in CENTRAL_QUANTILES
    }
    valid = prepared["valid"] and all(
        item["valid"] for item in selected.values()
    )
    if valid:
        p_values = {
            tau: float(item["p_value"])
            for tau, item in selected.items()
        }
        global_p = float(
            min(1.0, len(CENTRAL_QUANTILES) * min(p_values.values()))
        )
        failure_reason = ""
    else:
        p_values = {
            tau: (float(item["p_value"]) if item.get("valid") else None)
            for tau, item in selected.items()
        }
        global_p = np.nan
        failure_reason = " | ".join(
            f"tau={tau:g}: {item['failure_reason']}"
            for tau, item in selected.items()
            if not item["valid"]
        )
    return {
        "valid": bool(valid),
        "failure_reason": failure_reason,
        "p_value": global_p,
        "q": prepared.get("q", 0),
        "df": prepared.get("q", 0),
        "component_p_values": p_values,
        "component_iterations": {
            tau: item.get("iterations") for tau, item in selected.items()
        },
        "component_converged": {
            tau: bool(item.get("converged", False))
            for tau, item in selected.items()
        },
        "component_warnings": {
            tau: item.get("warning_text", "") for tau, item in selected.items()
        },
        "max_wald_agreement": max(
            (
                item.get("wald_agreement", 0.0)
                for item in selected.values()
                if item.get("valid")
            ),
            default=np.nan,
        ),
    }


def base_record(
    *,
    repetition: int,
    rho: float,
    report_direction: str,
    method_key: str,
    data_seed: int,
    hsic_seed: int,
    residual_hash: str,
    p_value: float,
    statistic: float | None,
    valid: bool,
    failure_reason: str,
    rejection_rule: str,
) -> dict:
    if valid:
        reject = (
            bool(p_value <= ALPHA)
            if rejection_rule == "p <= 0.05"
            else bool(p_value < ALPHA)
        )
    else:
        reject = None
    return {
        "repetition": int(repetition),
        "rho": float(rho),
        "direction": report_direction,
        "scenario": SCENARIOS[(float(rho), report_direction)],
        "method_key": method_key,
        "column_title": METHOD_TITLES[method_key],
        "data_seed": int(data_seed),
        "hsic_seed": int(hsic_seed),
        "residual_scheme": "five-fold cross-fitted",
        "n_folds": int(CORE_CONFIG.num_folds),
        "n_train_per_fold": int(
            CORE_CONFIG.N * (CORE_CONFIG.num_folds - 1)
            // CORE_CONFIG.num_folds
        ),
        "n_eval": int(CORE_CONFIG.N),
        "residual_sha256": residual_hash,
        "fold_residual_sha256": "",
        "jax_backend": jax.default_backend(),
        "jax_device_kind": jax.devices()[0].device_kind,
        "p_value": finite_or_none(p_value),
        "statistic": finite_or_none(statistic),
        "valid": bool(valid),
        "failure_reason": failure_reason,
        "rejection_rule": rejection_rule,
        "reject": reject,
        "feature_basis": "",
        "effective_rank": None,
        "n_bins": None,
        "quantile_grid": "",
        "component_p_values": "",
        "component_iterations": "",
        "component_converged": "",
        "mean_diagnostics": "",
        "alpha_x": None,
        "alpha_residual": None,
    }


def selected_records_for_direction(
    *,
    repetition: int,
    rho: float,
    rho_index: int,
    repo_direction: str,
    direction_index: int,
    data: dict,
) -> list[dict]:
    report_direction = DIRECTION_MAP[repo_direction]
    predictor, response, signal = direction_arrays(data, repo_direction)
    predictor = np.asarray(predictor, dtype=float)
    response = np.asarray(response, dtype=float)
    signal = np.asarray(signal, dtype=float)

    data_seed = DATA_SEED_BASE + 1_000 * repetition + 100 * rho_index
    hsic_seed = (
        DATA_SEED_BASE
        + 1_000_000
        + 1_000 * repetition
        + 100 * rho_index
        + direction_index
    )
    hsic_result = selected_crossfit_hsic(
        predictor,
        response,
        signal,
        CORE_CONFIG,
        hsic_seed,
    )
    residual = np.asarray(hsic_result["residual_oos"], dtype=float)
    if len(residual) != CORE_CONFIG.N or not np.isfinite(residual).all():
        raise RuntimeError("Invalid shared cross-fitted residual vector.")
    residual_hash = array_sha256(residual)
    fold_residual_hash = array_sha256(hsic_result["fold_residuals"])
    u = (predictor - predictor.mean()) / predictor.std(ddof=1)

    records = []
    plugin = base_record(
        repetition=repetition,
        rho=rho,
        report_direction=report_direction,
        method_key="old_hsic_cf",
        data_seed=data_seed,
        hsic_seed=hsic_seed,
        residual_hash=residual_hash,
        p_value=hsic_result["plugin_p_value"],
        statistic=hsic_result["plugin_statistic"],
        valid=True,
        failure_reason="",
        rejection_rule="p <= 0.05",
    )
    plugin.update(
        {
            "mean_diagnostics": json.dumps(
                hsic_result["mean_diagnostics"],
                sort_keys=True,
            ),
            "alpha_x": float(hsic_result["alpha_x"]),
            "alpha_residual": float(hsic_result["alpha_xi"]),
        }
    )
    records.append(plugin)

    debiased = base_record(
        repetition=repetition,
        rho=rho,
        report_direction=report_direction,
        method_key="old_hsic_debiased",
        data_seed=data_seed,
        hsic_seed=hsic_seed,
        residual_hash=residual_hash,
        p_value=hsic_result["debiased_p_value"],
        statistic=hsic_result["debiased_sqrt_hsic"],
        valid=True,
        failure_reason="",
        rejection_rule="p <= 0.05",
    )
    records.append(debiased)

    for method_key, basis in (
        ("bp_linear", "linear"),
        ("bp_fourier2", "fourier2"),
        ("bp_fourier4", "fourier4"),
    ):
        result = variance_score_test(u, residual, basis)
        record = base_record(
            repetition=repetition,
            rho=rho,
            report_direction=report_direction,
            method_key=method_key,
            data_seed=data_seed,
            hsic_seed=hsic_seed,
            residual_hash=residual_hash,
            p_value=result.get("p_value", np.nan),
            statistic=result.get("statistic"),
            valid=bool(result.get("valid", False)),
            failure_reason=result.get("failure_reason", ""),
            rejection_rule="p < 0.05",
        )
        record["feature_basis"] = basis
        record["effective_rank"] = result.get("q")
        records.append(record)

    for method_key, center in (
        ("levene_b10", "mean"),
        ("brown_forsythe_b10", "median"),
    ):
        result = binned_dispersion_test(
            predictor,
            residual,
            n_bins=10,
            center=center,
        )
        record = base_record(
            repetition=repetition,
            rho=rho,
            report_direction=report_direction,
            method_key=method_key,
            data_seed=data_seed,
            hsic_seed=hsic_seed,
            residual_hash=residual_hash,
            p_value=result.get("p_value", np.nan),
            statistic=result.get("statistic"),
            valid=bool(result.get("valid", False)),
            failure_reason=result.get("failure_reason", ""),
            rejection_rule="p < 0.05",
        )
        record["n_bins"] = 10
        records.append(record)

    quantile = central_quantile_test(u, residual, "fourier4")
    quantile_record = base_record(
        repetition=repetition,
        rho=rho,
        report_direction=report_direction,
        method_key="quantile_central_fourier4",
        data_seed=data_seed,
        hsic_seed=hsic_seed,
        residual_hash=residual_hash,
        p_value=quantile.get("p_value", np.nan),
        statistic=None,
        valid=bool(quantile.get("valid", False)),
        failure_reason=quantile.get("failure_reason", ""),
        rejection_rule="p < 0.05",
    )
    quantile_record.update(
        {
            "feature_basis": "fourier4",
            "effective_rank": quantile.get("q"),
            "quantile_grid": "central",
            "component_p_values": json.dumps(
                quantile.get("component_p_values", {}),
                sort_keys=True,
            ),
            "component_iterations": json.dumps(
                quantile.get("component_iterations", {}),
                sort_keys=True,
            ),
            "component_converged": json.dumps(
                quantile.get("component_converged", {}),
                sort_keys=True,
            ),
        }
    )
    records.append(quantile_record)

    for record in records:
        record["fold_residual_sha256"] = fold_residual_hash
    if len(records) != len(SELECTED_METHOD_KEYS):
        raise AssertionError("Wrong number of selected method records.")
    if {row["method_key"] for row in records} != set(SELECTED_METHOD_KEYS):
        raise AssertionError("Selected method keys are incomplete.")
    if {row["residual_sha256"] for row in records} != {residual_hash}:
        raise AssertionError("Methods did not share one residual vector.")
    if {row["fold_residual_sha256"] for row in records} != {
        fold_residual_hash
    }:
        raise AssertionError("Methods did not share one fold-residual tensor.")
    return records


def records_for_repetition(repetition: int) -> pd.DataFrame:
    records = []
    for rho_index, rho in enumerate(RHO_VALUES):
        data_seed = DATA_SEED_BASE + 1_000 * repetition + 100 * rho_index
        data = simulate_anm(CORE_N, rho, seed=data_seed)
        for direction_index, repo_direction in enumerate(REPO_DIRECTIONS):
            records.extend(
                selected_records_for_direction(
                    repetition=repetition,
                    rho=float(rho),
                    rho_index=rho_index,
                    repo_direction=repo_direction,
                    direction_index=direction_index,
                    data=data,
                )
            )
    frame = pd.DataFrame(records)
    expected_rows = len(SCENARIOS) * len(SELECTED_METHOD_KEYS)
    if len(frame) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} rows; obtained {len(frame)}."
        )
    if frame.duplicated(["scenario", "method_key"]).any():
        raise AssertionError("Duplicate method/scenario rows in repetition.")
    if frame.groupby("scenario")["residual_sha256"].nunique().ne(1).any():
        raise AssertionError("A scenario used more than one residual vector.")
    if (
        frame.groupby("scenario")["fold_residual_sha256"]
        .nunique()
        .ne(1)
        .any()
    ):
        raise AssertionError("A scenario used more than one residual tensor.")
    return frame


def repetition_path(repetition: int) -> Path:
    return REPETITION_DIR / f"rep_{repetition:04d}.csv"


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    destination_mode = (
        path.stat().st_mode & 0o777 if path.exists() else 0o664
    )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.chmod(temporary, destination_mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_repetition_file(path: Path, repetition: int) -> bool:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    required = {
        "repetition",
        "scenario",
        "method_key",
        "residual_sha256",
        "fold_residual_sha256",
        "p_value",
        "valid",
        "rejection_rule",
        "reject",
        "effective_rank",
        "n_bins",
        "component_p_values",
    }
    if not required.issubset(frame.columns):
        return False
    if not (
        len(frame) == len(SCENARIOS) * len(SELECTED_METHOD_KEYS)
        and frame["repetition"].eq(repetition).all()
        and not frame.duplicated(["scenario", "method_key"]).any()
        and set(frame["scenario"]) == set(SCENARIOS.values())
        and set(frame["method_key"]) == set(SELECTED_METHOD_KEYS)
        and frame.groupby("scenario")["residual_sha256"].nunique().eq(1).all()
        and frame.groupby("scenario")["fold_residual_sha256"].nunique().eq(1).all()
    ):
        return False
    valid = frame["valid"].eq(True)
    valid_p = pd.to_numeric(frame.loc[valid, "p_value"], errors="coerce")
    if not (np.isfinite(valid_p).all() and valid_p.between(0.0, 1.0).all()):
        return False
    if frame.loc[~valid, "p_value"].notna().any():
        return False
    expected_reject = np.where(
        frame.loc[valid, "rejection_rule"].eq("p <= 0.05"),
        valid_p <= ALPHA,
        valid_p < ALPHA,
    )
    actual_reject = frame.loc[valid, "reject"].astype(bool).to_numpy()
    if not np.array_equal(expected_reject, actual_reject):
        return False
    legacy = frame["method_key"].isin(("old_hsic_cf", "old_hsic_debiased"))
    legacy_scaled = pd.to_numeric(frame.loc[legacy, "p_value"]) * (
        CORE_CONFIG.bootstrap_B + 1
    )
    if not np.allclose(legacy_scaled, np.round(legacy_scaled), atol=1e-9):
        return False
    for method_key, expected_rank in (
        ("bp_linear", 1),
        ("bp_fourier2", 4),
        ("bp_fourier4", 8),
        ("quantile_central_fourier4", 8),
    ):
        ranks = pd.to_numeric(
            frame.loc[frame["method_key"].eq(method_key), "effective_rank"]
        )
        if not ranks.eq(expected_rank).all():
            return False
    binned = frame["method_key"].isin(("levene_b10", "brown_forsythe_b10"))
    if not pd.to_numeric(frame.loc[binned, "n_bins"]).eq(10).all():
        return False
    quantile_rows = frame.loc[
        frame["method_key"].eq("quantile_central_fourier4") & valid
    ]
    for row in quantile_rows.itertuples(index=False):
        component = json.loads(row.component_p_values)
        if set(component) != {"0.25", "0.5", "0.75"}:
            return False
        expected_p = min(1.0, 3.0 * min(float(value) for value in component.values()))
        if not np.isclose(float(row.p_value), expected_p, rtol=0.0, atol=1e-12):
            return False
    return True


def validate_against_existing(frame: pd.DataFrame) -> None:
    global _BASELINE_HSIC, _BASELINE_CLASSICAL

    repetition = int(frame["repetition"].iloc[0])
    if repetition >= 100:
        return
    if _BASELINE_HSIC is None:
        _BASELINE_HSIC = pd.read_csv(BASELINE_HSIC_CSV)
    if _BASELINE_CLASSICAL is None:
        _BASELINE_CLASSICAL = pd.read_csv(BASELINE_CLASSICAL_CSV)
    baseline_hsic = _BASELINE_HSIC
    baseline_classical = _BASELINE_CLASSICAL
    for row in frame.itertuples(index=False):
        repo_direction = REPO_DIRECTION_MAP[row.direction]
        if row.method_key == "old_hsic_cf":
            match = baseline_hsic.loc[
                (baseline_hsic["rep"] == repetition)
                & np.isclose(baseline_hsic["rho"], row.rho)
                & (baseline_hsic["direction"] == repo_direction)
                & (baseline_hsic["method"] == "core full naive")
            ]
            expected = float(match.iloc[0]["p_value"])
        elif row.method_key == "old_hsic_debiased":
            match = baseline_hsic.loc[
                (baseline_hsic["rep"] == repetition)
                & np.isclose(baseline_hsic["rho"], row.rho)
                & (baseline_hsic["direction"] == repo_direction)
                & (baseline_hsic["method"] == "core full debiased")
            ]
            expected = float(match.iloc[0]["bootstrap_p_value"])
        else:
            method_filter = {
                "bp_linear": ("variance_score", "linear", None, None),
                "bp_fourier2": (
                    "variance_score",
                    "fourier2",
                    None,
                    None,
                ),
                "bp_fourier4": (
                    "variance_score",
                    "fourier4",
                    None,
                    None,
                ),
                "levene_b10": ("levene", None, 10, None),
                "brown_forsythe_b10": (
                    "brown_forsythe",
                    None,
                    10,
                    None,
                ),
                "quantile_central_fourier4": (
                    "quantile_regression",
                    "fourier4",
                    None,
                    "central",
                ),
            }[row.method_key]
            method, basis, n_bins, grid = method_filter
            match = baseline_classical.loc[
                (baseline_classical["repetition"] == repetition)
                & np.isclose(baseline_classical["rho"], row.rho)
                & (baseline_classical["direction"] == row.direction)
                & (baseline_classical["method"] == method)
                & (
                    baseline_classical["residual_scheme"]
                    == "cross_fitted_variant"
                )
            ]
            if basis is not None:
                match = match.loc[match["feature_basis"] == basis]
            if n_bins is not None:
                match = match.loc[np.isclose(match["n_bins"], n_bins)]
            if grid is not None:
                match = match.loc[match["quantile_grid"] == grid]
            expected = float(match.iloc[0]["p_value"])
        if len(match) != 1:
            raise AssertionError(
                f"Baseline match failed for {row.scenario}, {row.method_key}."
            )
        if not np.isclose(
            float(row.p_value),
            expected,
            rtol=0.0,
            atol=5e-10,
        ):
            raise AssertionError(
                f"p-value mismatch for rep={repetition}, "
                f"{row.scenario}, {row.method_key}: "
                f"{row.p_value} versus {expected}."
            )
        expected_reject = (
            expected <= ALPHA
            if row.method_key in ("old_hsic_cf", "old_hsic_debiased")
            else expected < ALPHA
        )
        if bool(row.reject) != bool(expected_reject):
            raise AssertionError(
                f"rejection mismatch for rep={repetition}, "
                f"{row.scenario}, {row.method_key}."
            )


def write_progress(
    *,
    shard_index: int,
    shard_count: int,
    completed: int,
    assigned: int,
    started: float,
    last_repetition: int | None,
) -> None:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROGRESS_DIR / f"shard_{shard_index:02d}_of_{shard_count:02d}.json"
    payload = {
        "status": "complete" if completed == assigned else "running",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "completed": completed,
        "assigned": assigned,
        "last_repetition": last_repetition,
        "elapsed_seconds": time.perf_counter() - started,
        "updated_at_unix": time.time(),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run_shard(
    *,
    shard_index: int,
    shard_count: int,
    start: int,
    stop: int,
    validate_existing: bool,
) -> None:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count).")
    if start < 0 or stop <= start:
        raise ValueError("Require 0 <= start < stop.")
    print(
        f"jax_backend={jax.default_backend()} | "
        f"jax_device_kind={jax.devices()[0].device_kind} | "
        f"range=[{start}, {stop}) | "
        f"shard={shard_index + 1}/{shard_count}",
        flush=True,
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPETITION_DIR.mkdir(parents=True, exist_ok=True)
    assigned_repetitions = [
        repetition
        for repetition in range(start, stop)
        if repetition % shard_count == shard_index
    ]
    started = time.perf_counter()
    completed = 0
    for repetition in assigned_repetitions:
        path = repetition_path(repetition)
        if path.exists() and validate_repetition_file(path, repetition):
            completed += 1
            continue
        repetition_started = time.perf_counter()
        frame = records_for_repetition(repetition)
        if validate_existing:
            validate_against_existing(frame)
        atomic_write_csv(frame, path)
        if not validate_repetition_file(path, repetition):
            raise RuntimeError(f"Written repetition file failed validation: {path}")
        completed += 1
        write_progress(
            shard_index=shard_index,
            shard_count=shard_count,
            completed=completed,
            assigned=len(assigned_repetitions),
            started=started,
            last_repetition=repetition,
        )
        print(
            f"shard {shard_index + 1}/{shard_count} | "
            f"rep {repetition:04d} | {completed}/{len(assigned_repetitions)} | "
            f"rep_seconds={time.perf_counter() - repetition_started:.1f} | "
            f"elapsed_minutes={(time.perf_counter() - started) / 60:.1f}",
            flush=True,
        )
    write_progress(
        shard_index=shard_index,
        shard_count=shard_count,
        completed=completed,
        assigned=len(assigned_repetitions),
        started=started,
        last_repetition=(
            assigned_repetitions[-1] if assigned_repetitions else None
        ),
    )
    print(
        f"shard {shard_index + 1}/{shard_count} complete: "
        f"{completed}/{len(assigned_repetitions)} repetitions in "
        f"{(time.perf_counter() - started) / 60:.1f} minutes.",
        flush=True,
    )


def configuration_report() -> dict:
    return {
        "target_repetitions": TARGET_REPETITIONS,
        "holdout_repetition_start_inclusive": HOLDOUT_START,
        "holdout_repetition_stop_exclusive": HOLDOUT_STOP,
        "holdout_is_fresh_after_selection_runs": True,
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_device_kind": jax.devices()[0].device_kind,
        },
        "sample_size": int(CORE_N),
        "rho_values": list(RHO_VALUES),
        "repo_directions": list(REPO_DIRECTIONS),
        "data_seed_base": int(DATA_SEED_BASE),
        "first_stage": {
            "kernel": CORE_CONFIG.mean_kernel,
            "cv_folds": int(CORE_CONFIG.mean_cv_folds),
            "alphas": list(CORE_CONFIG.mean_alphas),
            "length_scales": list(CORE_CONFIG.mean_length_scales),
            "outer_folds": int(CORE_CONFIG.num_folds),
        },
        "hsic_resamples": int(CORE_CONFIG.bootstrap_B),
        "selected_method_keys": list(SELECTED_METHOD_KEYS),
        "method_titles": METHOD_TITLES,
        "source_hashes": SOURCE_HASHES,
        "versions": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--start", type=int, default=HOLDOUT_START)
    parser.add_argument("--stop", type=int, default=HOLDOUT_STOP)
    parser.add_argument(
        "--validate-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_config:
        print(json.dumps(configuration_report(), indent=2))
        return
    run_shard(
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        start=args.start,
        stop=args.stop,
        validate_existing=args.validate_existing,
    )


if __name__ == "__main__":
    main()
