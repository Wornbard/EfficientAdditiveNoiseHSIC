"""Replay saved five-fold synthetic runs for a studentized cross-U comparison.

Keep the original nuisance fits, bandwidths, and population targets. Split each
evaluation fold evenly between A and B, then use only the A-by-B score Gram block.
The primary interval inflates the directional cross-U standard error by sqrt(2).
This benchmarks the proposed cross-fitted extension; it does not prove coverage.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import time
import traceback
from typing import Any

import numpy as np
import yaml


ROOT = Path(os.environ.get("HSIC_PROJECT_ROOT", Path(__file__).resolve().parent))
SOURCE = ROOT / "runs/multivariate_sweep/favourability_search_updated_paper"
SPLIT_SEED = 20261029
ALPHA = 0.05
DELTA_SQUARED_U = "dm_debiased_squared_u"
LEGACY_DELTA = "dm_debiased_u_sqrt_v_variance"
ORIGINAL_RUNS = [ROOT / "runs/cross_u_benchmark_20261029",
                 ROOT / "runs/cross_u_n250_smoothness05_20261101"]


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.load(path.read_text(), Loader=yaml.CSafeLoader)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically checkpoint one small, strictly JSON-compatible result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def balanced_split(masks: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Randomize membership within each nuisance fold without using data values."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 2 or not np.all(masks.sum(axis=0) == 1):
        raise ValueError("Nuisance folds must partition all observations.")
    rng = np.random.default_rng(seed)
    left, right = [], []
    for mask in masks:
        indices = rng.permutation(np.flatnonzero(mask))
        if len(indices) < 4 or len(indices) % 2:
            raise ValueError("Every nuisance fold must have even size, at least four.")
        midpoint = len(indices) // 2
        left.extend(indices[:midpoint])
        right.extend(indices[midpoint:])
    return np.asarray(left), np.asarray(right)


def cross_u_statistics(gram: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Use the sample variance of A-side projections, conditional on the B mean.

    The B-side variance is recorded only as a diagnostic. Adding the two
    variances or averaging the directional pivots changes the null calibration.
    """
    gram = np.asarray(gram, dtype=float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not np.isfinite(gram).all():
        raise ValueError("Expected a finite square score Gram matrix.")
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("The sqrt(2) inflation requires equal cross-U groups.")
    joined = np.concatenate((left, right))
    if not np.array_equal(np.sort(joined), np.arange(len(gram))):
        raise ValueError("Cross-U groups must partition the score matrix.")
    block = gram[np.ix_(left, right)]
    rows, columns = block.mean(axis=1), block.mean(axis=0)
    estimate = float(rows.mean())
    se_a = float(np.sqrt(rows.var(ddof=1) / len(left)))
    se_b = float(np.sqrt(columns.var(ddof=1) / len(right)))
    if not np.isfinite(se_a) or se_a <= 0:
        raise ValueError("The directional cross-U studentizer is zero or nonfinite.")
    return {"estimate": estimate, "se_a": se_a, "se_b": se_b,
            "pivot_null": estimate / se_a}


def interval_metrics(lower: float, upper: float, truth: float) -> dict[str, Any]:
    """Intersect with [0,infinity); preserve an empty set when upper < 0."""
    if not all(math.isfinite(x) for x in (lower, upper, truth)) or truth < 0:
        raise ValueError("Interval endpoints and nonnegative truth must be finite.")
    lower = max(0.0, lower)
    empty = upper < lower
    return {
        "lower_hsic": lower, "upper_hsic": upper, "empty": empty,
        "coverage": bool(not empty and lower <= truth <= upper),
        "ci_excludes_zero": bool(empty or lower > 0),
        "width_hsic": max(0.0, upper - lower),
        "width_sqrt_hsic": 0.0 if empty else math.sqrt(upper) - math.sqrt(lower),
    }


def cross_u_interval(stats: dict[str, float], truth: float, inflation: float) -> dict[str, Any]:
    se = inflation * stats["se_a"]
    radius = NormalDist().inv_cdf(1.0 - ALPHA / 2.0) * se
    pivot = stats["estimate"] / se
    return {
        **interval_metrics(stats["estimate"] - radius, stats["estimate"] + radius, truth),
        **stats, "se": se, "inflation": inflation,
        "p_value_right_tail": NormalDist().cdf(-pivot),
        "reject_right_tail": bool(pivot > NormalDist().inv_cdf(1.0 - ALPHA)),
        "pivot_centered": (stats["estimate"] - truth) / se,
    }


def delta_squared_u_interval(u: float, score_variance: float, n: int,
                             truth: float, alpha: float = ALPHA) -> dict[str, Any]:
    """Build a signed-U Wald set for HSIC, intersect with [0,∞), then take sqrt.

    score_variance is the population-normalized variance of score-Gram row
    means, so the HSIC standard error is 2*sqrt(score_variance/n). An entirely
    negative Wald interval stays empty, rather than becoming the singleton {0}.
    """
    if (not all(math.isfinite(x) for x in (u, score_variance, alpha))
            or score_variance < 0 or n < 2 or not 0 < alpha < 1):
        raise ValueError("Delta requires finite U, nonnegative score variance, n>=2 and 0<alpha<1.")
    se = 2 * math.sqrt(score_variance / n)
    radius = NormalDist().inv_cdf(1 - alpha / 2) * se
    return {**interval_metrics(u - radius, u + radius, truth),
            "estimate": u, "se_hsic": se, "score_variance": score_variance,
            "wald_lower_hsic": u - radius, "wald_upper_hsic": u + radius,
            "undefined": False}


def delta_squared_u_from_gram(gram: np.ndarray, num_folds: int, truth: float,
                             alpha: float = ALPHA) -> dict[str, Any]:
    """Recompute the fold-weighted U center and score variance from fresh scores."""
    gram = np.asarray(gram, dtype=float)
    n = len(gram)
    if (gram.shape != (n, n) or not np.isfinite(gram).all() or num_folds < 1
            or n % num_folds or n // num_folds < 2):
        raise ValueError("Expected a finite score Gram matrix with equal nuisance folds.")
    size = n // num_folds
    v = float(gram.mean())
    u = v
    for fold in range(num_folds):
        block = gram[fold * size:(fold + 1) * size, fold * size:(fold + 1) * size]
        offdiag = (block.sum() - np.trace(block)) / (size * (size - 1))
        u += (offdiag - block.mean()) / num_folds ** 2
    score_variance = float(np.mean((gram.mean(axis=1) - v) ** 2))
    return delta_squared_u_interval(float(u), score_variance, n, truth, alpha)


def delta_squared_u_from_saved(raw: dict[str, Any], truth: float) -> dict[str, Any]:
    """Reconstruct the signed-U Wald set from saved U and score variance."""
    return delta_squared_u_interval(
        float(raw["estimators"]["debiased_hsic_u"]),
        float(raw["intervals"]["delta_debiased_v"]["score_var"]),
        raw["config"]["N"], truth, raw["config"].get("ci_alpha", ALPHA))


def restore_score_grams(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    """Refit saved nuisance choices and verify every replay against saved U/V values.

    This uses the reconstruction already checked in synthetic_witness_comparison.
    No hyperparameter search or new bootstrap is performed.
    """
    import jax.numpy as jnp
    from sklearn.gaussian_process.kernels import Matern, RBF
    from sklearn.kernel_ridge import KernelRidge
    from multivariate_crossfit_core import (
        RunConfig, simulate_data, build_fold_masks, gaussian_gram, median_heuristic,
        matern_from_dists, residualized_krr_weight_matrix, standard_krr_weight_matrix,
        center_cross_block, centered_blocks_from_full_gram, compute_xi_debias_term,
        crossfit_v_from_blocks, crossfit_u_from_blocks, blocks_to_full_matrix,
    )

    config = RunConfig(**raw["config"])
    if config.dimx != 1 or config.dimy != 1 or config.num_folds != 5:
        raise ValueError("This replay requires the original scalar five-fold setup.")
    _, X, signal, eps, Y = simulate_data(config)
    masks = np.asarray(build_fold_masks(config.N, config.num_folds))
    X_np, Y_np = np.asarray(X), np.asarray(Y)
    residuals, weights = [], []
    distances = jnp.abs(X[:, 0, None] - X[None, :, 0])
    for fold, mask in enumerate(masks):
        params = raw["mean_fit"][fold]
        kernel = (RBF(length_scale=params["length_scale"])
                  if config.mean_kernel in ("gaussian", "rbf")
                  else Matern(length_scale=params["length_scale"], nu=params["nu"]))
        model = KernelRidge(alpha=params["alpha"], kernel=kernel)
        model.fit(X_np[~mask], Y_np[~mask])
        forecast = model.predict(X_np)
        heldout_mse = float(np.mean((forecast[mask] - np.asarray(signal)[mask]) ** 2))
        np.testing.assert_allclose(heldout_mse, params["heldout_signal_mse"], rtol=2e-5, atol=2e-9)
        residuals.append(Y_np - forecast)
        params_v = raw["v_fit_hyperparams"][fold][0]
        Q = matern_from_dists(distances, params_v["length_scale"], params_v["nu"])
        weight_function = (residualized_krr_weight_matrix
                           if config.v_fit_mode == "residualized_centered"
                           else standard_krr_weight_matrix)
        weights.append(np.asarray(weight_function(Q[~mask][:, ~mask], Q[mask][:, ~mask], params_v["eta"])))
    residuals = jnp.asarray(np.asarray(residuals))
    weights = jnp.asarray(np.asarray(weights)[:, None, :, :])
    masks = jnp.asarray(masks)
    alpha_x, alpha_r = raw["bandwidths"]["alpha_x"], raw["bandwidths"]["alpha_xi"]
    oof_residuals = jnp.concatenate([residuals[k, mask] for k, mask in enumerate(masks)])
    np.testing.assert_allclose(float(median_heuristic(X)), alpha_x, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(float(median_heuristic(oof_residuals)), alpha_r, rtol=2e-5, atol=2e-9)
    WX = centered_blocks_from_full_gram(gaussian_gram(X, X, alpha_x), masks)
    WXi = jnp.asarray([[center_cross_block(gaussian_gram(residuals[k], residuals[l], alpha_r), mk, ml)
                       for l, ml in enumerate(masks)] for k, mk in enumerate(masks)])
    correction = compute_xi_debias_term(config, residuals, masks, weights, alpha_r)
    blocks = {
        "plugin": WX * WXi,
        "debiased": WX * (WXi + correction),
        "true_residual": WX * centered_blocks_from_full_gram(gaussian_gram(eps, eps, alpha_r), masks),
    }
    grams = {}
    for method, H in blocks.items():
        for kind, statistic in (("v", crossfit_v_from_blocks), ("u", crossfit_u_from_blocks)):
            if method == "true_residual":
                expected = raw["oracle"][f"crossfit_hsic_{kind}"]
            else:
                prefix = "naive" if method == "plugin" else "debiased"
                expected = raw["estimators"][f"{prefix}_hsic_{kind}"]
            np.testing.assert_allclose(float(statistic(H)), expected, rtol=3e-4, atol=2e-8,
                                       err_msg=f"Saved {method} {kind} was not reproduced")
        gram = np.asarray(blocks_to_full_matrix(H))
        np.testing.assert_allclose(gram, gram.T, rtol=1e-7, atol=1e-9)
        grams[method] = gram
    return grams


def evaluate_run(entry: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    path = ROOT / entry["source"]
    if file_hash(path) != entry["sha256"]:
        raise ValueError(f"Source result changed after planning: {path}")
    raw = read_yaml(path)
    config = raw["config"]
    # Homogeneous noise gives exactly zero; do not treat quadrature roundoff as signal.
    truth = 0.0 if config["rho"] == 0 else float(raw["population_oracle"]["hsic"])
    grams = restore_score_grams(raw)
    fold_size = config["N"] // config["num_folds"]
    masks = np.repeat(np.eye(config["num_folds"], dtype=bool), fold_size, axis=1)
    split_seed = SPLIT_SEED + config["random_key"]
    left, right = balanced_split(masks, split_seed)
    methods = {}
    for name, gram in grams.items():
        stats = cross_u_statistics(gram, left, right)
        methods[f"cross_u_{name}_inflated"] = cross_u_interval(stats, truth, math.sqrt(2.0))
        methods[f"cross_u_{name}_uninflated"] = cross_u_interval(stats, truth, 1.0)
    methods[DELTA_SQUARED_U] = delta_squared_u_from_gram(
        grams["debiased"], config["num_folds"], truth, config.get("ci_alpha", ALPHA))
    boot = raw["intervals"]["bootstrap_debiased"]
    methods["rti_debiased"] = {
        **interval_metrics(boot["ci_lower"] ** 2, boot["ci_upper"] ** 2, truth),
        "p_value_right_tail": boot.get("p_value"),
        # Older saved runs contain the calibrated radius but no bootstrap p-value.
        "reject_right_tail": bool(boot["ci_lower"] > 0),
    }
    for kind in ("u", "v"):
        dm = raw["intervals"][f"delta_debiased_{kind}"]
        methods[f"dm_debiased_{kind}_hsic"] = interval_metrics(dm["hsic_lower"], dm["hsic_upper"], truth)
        # The original sqrt delta interval may be undefined for a nonpositive U center.
        if all(value is not None and math.isfinite(value)
               for value in (dm["sqrt_lower"], dm["sqrt_upper"])):
            methods[f"dm_debiased_{kind}_sqrt"] = interval_metrics(dm["sqrt_lower"] ** 2, dm["sqrt_upper"] ** 2, truth)
        else:
            methods[f"dm_debiased_{kind}_sqrt"] = {"undefined": True}
    return {
        "source": entry["source"], "source_sha256": entry["sha256"],
        "config": {k: config[k] for k in ("N", "num_folds", "rho", "random_key", "f_smoothness", "eps_smoothness")},
        "population_hsic": truth, "bandwidths": raw["bandwidths"],
        "split_seed": split_seed, "split_a_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
        "group_size": len(left), "methods": methods,
        "replay_verified": True, "seconds": time.perf_counter() - start,
        "evaluation_mode": "regenerated_data_refit_saved_hyperparameters",
    }


def refresh_run(entry: dict[str, Any]) -> dict[str, Any]:
    """Update the delta comparator using saved fits; preserve original checkpoints."""
    path = ROOT / entry["source"]
    if file_hash(path) != entry["sha256"]:
        raise ValueError(f"Source result changed: {path}")
    previous = ROOT / entry["previous_checkpoint"]
    result = json.loads(previous.read_text())
    if not result["replay_verified"] or result["source_sha256"] != entry["sha256"]:
        raise ValueError(f"Incompatible original checkpoint: {previous}")
    raw = read_yaml(path)
    result["methods"][DELTA_SQUARED_U] = delta_squared_u_from_saved(raw, result["population_hsic"])
    result["evaluation_mode"] = "saved_fit_delta_refresh"
    return result


def recompute_paper_delta(input_run: Path, destination: Path) -> None:
    """Refresh the 1800 paper comparators from archived scalars, without fitting."""
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Choose a new empty output directory; archived checkpoints are never overwritten.")
    original = json.loads((input_run / "manifest.json").read_text())
    expected = {(250, smoothness, rho, seed)
                for smoothness in (.25, .5, .75)
                for rho in (0., .1, .2, .3, .4, .5) for seed in range(10, 110)}
    tasks, seen = [], set()
    for task in original["tasks"]:
        if task["N"] != 250 or task["eps_smoothness"] not in (.25, .5, .75):
            continue
        entries = []
        for entry in task["entries"]:
            path = input_run / "replicates" / (entry["run_name"] + ".json")
            result = json.loads(path.read_text())
            c = result["config"]
            key = (c["N"], c["eps_smoothness"], c["rho"], c["random_key"])
            if (key not in expected or key in seen or c["f_smoothness"] != 1 or c["num_folds"] != 5
                    or key[:3] != (task["N"], task["eps_smoothness"], task["rho"])
                    or c["random_key"] != entry["random_key"]):
                raise ValueError(f"Wrong or duplicate paper configuration: {path}")
            if (not result["replay_verified"] or result["source_sha256"] != entry["sha256"]
                    or result["source"] != entry["source"]):
                raise ValueError(f"Unverified or mismatched source provenance: {path}")
            old = result["methods"][LEGACY_DELTA]
            current = delta_squared_u_interval(float(old["estimate"]), float(old["score_variance"]),
                                               c["N"], result["population_hsic"])
            # All non-delta methods, bootstrap draws, targets and splits stay unchanged.
            result["methods"] = {name: value for name, value in result["methods"].items()
                                 if not name.startswith("dm_debiased_")}
            result["methods"][DELTA_SQUARED_U] = current
            checksum = file_hash(path)
            result["delta_recomputed_from"] = {"checkpoint": str(path.resolve()), "sha256": checksum,
                                                "method": LEGACY_DELTA}
            result["evaluation_mode"] = "saved_U_and_score_variance_squared_Wald"
            write_json(destination / "replicates" / path.name, result)
            entries.append({**entry, "input_checkpoint": str(path.resolve()),
                            "input_checkpoint_sha256": checksum})
            seen.add(key)
        tasks.append({**task, "entries": entries})
    if seen != expected:
        raise ValueError(f"Incomplete paper cohort: missing {len(expected - seen)} of 1800 repetitions.")
    write_json(destination / "manifest.json", {
        "alpha": ALPHA, "split_seed_base": original["split_seed_base"],
        "planned_source_runs": 1800, "missing_source_runs": [], "tasks": tasks,
        "method": "Cross-U and RTI unchanged; signed-U HSIC Wald set intersected with nonnegative HSIC then sqrt",
        "input_run": str(input_run.resolve()),
        "input_manifest_sha256": file_hash(input_run / "manifest.json"),
        "code_sha256": {"run_cross_u_benchmark.py": file_hash(Path(__file__))},
        "notes": ["No fitted model, score Gram, bootstrap sample or split was regenerated.",
                  "Empty sets have zero width and do not include zero or the target.",
                  "All 18 paper settings retain seeds 10--109."],
    })
    print(f"Recomputed all 1800 paper delta intervals from saved scalars in {destination}", flush=True)


def make_corrected_plan(destination: Path) -> None:
    """Freeze the exact union of the two completed benchmark grids (4,180 runs)."""
    import shutil
    if (destination / "manifest.json").exists():
        raise FileExistsError("A benchmark manifest already exists.")
    tasks, missing = [], []
    planned = 0
    for previous in ORIGINAL_RUNS:
        manifest = json.loads((previous / "manifest.json").read_text())
        planned += manifest["planned_source_runs"]
        missing.extend(manifest["missing_source_runs"])
        for task in manifest["tasks"]:
            for entry in task["entries"]:
                entry["previous_checkpoint"] = str(
                    (previous / "replicates" / (entry["run_name"] + ".json")).relative_to(ROOT))
            tasks.append(task)
    tasks.sort(key=lambda task: (task["N"], task["eps_smoothness"], task["rho"]))
    sources = [entry["source"] for task in tasks for entry in task["entries"]]
    if len(sources) != len(set(sources)) or len(sources) + len(missing) != planned:
        raise ValueError("Original benchmark manifests contain duplicate or inconsistent runs.")
    code = ("run_cross_u_benchmark.py", "multivariate_crossfit_core.py",
            "summarize_cross_u_benchmark.py", "make_cross_u_paper_tables.py")
    write_json(destination / "manifest.json", {
        "alpha": ALPHA, "split_seed_base": SPLIT_SEED, "planned_source_runs": planned,
        "missing_source_runs": missing, "tasks": tasks,
        "method": "Cross-U unchanged; delta uses a signed-U HSIC Wald set followed by square-root mapping",
        "code_sha256": {name: file_hash(ROOT / name) for name in code},
        "notes": ["Original seeds and selected nuisance hyperparameters are retained.",
                  "Fresh replay regenerates data, refits nuisances and recomputes score Grams and delta intervals.",
                  "Original RTI bootstrap and population targets are reused.",
                  "Negative U centers are retained; undefined primary intervals cause reporting to fail."],
    })
    snapshot = destination / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in code + ("cross_u_intervals_appendix.tex", "test_cross_u_benchmark.py", "test_cross_u_reporting.py"):
        shutil.copy2(ROOT / name, snapshot / name)
    print(f"Planned {len(tasks)} settings and {len(sources)} replays; {len(missing)} original sources missing.", flush=True)


def make_plan(destination: Path) -> None:
    if (destination / "manifest.json").exists():
        raise FileExistsError("A benchmark manifest already exists; use its run/resume commands.")
    source_manifest = read_yaml(SOURCE / "planned_jobs.yaml")
    groups = defaultdict(list)
    missing_sources = []
    for job in source_manifest["jobs"]:
        c = job["config"]
        if c["N"] != 250 or c["f_smoothness"] != 1.0 or c["eps_smoothness"] not in (0.25, 0.5, 0.75):
            continue
        if not 10 <= c["random_key"] <= 109:
            continue
        path = SOURCE / job["run_name"] / "results.yaml"
        if not path.is_file():
            missing_sources.append(str(path.resolve()))
            continue
        key = (c["N"], c["eps_smoothness"], c["rho"])
        groups[key].append({"source": str(path.resolve()), "sha256": file_hash(path),
                            "random_key": c["random_key"], "run_name": job["run_name"]})
    if len(groups) != 18 or sum(map(len, groups.values())) + len(missing_sources) != 1800:
        raise ValueError("Expected the paper grid: n=250, three noise smoothnesses, six rho values, seeds 10--109.")
    if missing_sources:
        raise FileNotFoundError(f"Generate all paper sources before planning: {len(missing_sources)} missing results.yaml files.")
    tasks = [{"N": key[0], "eps_smoothness": key[1], "rho": key[2],
              "entries": sorted(entries, key=lambda x: x["random_key"])}
             for key, entries in sorted(groups.items())]
    manifest = {
        "alpha": ALPHA, "split_seed_base": SPLIT_SEED,
        "planned_source_runs": 1800, "missing_source_runs": missing_sources,
        "method": "Five-fold nuisance fits; balanced within-fold A/B split; directional sample variance; sqrt(2) SE inflation",
        "notes": ["Uninflated intervals are diagnostics, not the proposed all-regime CI.",
                  "An entirely negative Gaussian interval has an empty intersection with HSIC >= 0.",
                  "Saved data-adaptive bandwidths and quadrature targets are shared by every method.",
                  "This fixed-rho grid does not assess uniform coverage along shrinking local alternatives."],
        "code_sha256": {name: file_hash(ROOT / name) for name in ("run_cross_u_benchmark.py", "multivariate_crossfit_core.py")},
        "tasks": tasks,
    }
    write_json(destination / "manifest.json", manifest)
    print(f"Planned {len(tasks)} tasks and {sum(len(t['entries']) for t in tasks)} replays in {destination}", flush=True)


def run_task(destination: Path, task_index: int | None, pilot: bool, refresh: bool = False) -> None:
    manifest = json.loads((destination / "manifest.json").read_text())
    for name, expected in manifest["code_sha256"].items():
        if file_hash(Path(__file__).resolve().parent / name) != expected:
            raise ValueError(f"Code changed after planning: {name}")
    if pilot:
        entries = [task["entries"][0] for task in manifest["tasks"]
                   if task["eps_smoothness"] == 0.75 and task["rho"] in (0.0, 0.3)]
    elif refresh and task_index is None:
        entries = [entry for task in manifest["tasks"] for entry in task["entries"]]
    else:
        if task_index is None:
            raise ValueError("Supply --task or --pilot.")
        entries = manifest["tasks"][task_index]["entries"]
    errors = 0
    for entry in entries:
        output = destination / "replicates" / (entry["run_name"] + ".json")
        if output.exists():
            saved = json.loads(output.read_text())
            if saved["source_sha256"] != entry["sha256"] or not saved["replay_verified"]:
                raise ValueError(f"Incompatible existing checkpoint: {output}")
            if "previous_checkpoint" in entry and DELTA_SQUARED_U not in saved["methods"]:
                raise ValueError(f"Checkpoint lacks the corrected delta method: {output}")
            continue
        failure = destination / "failures" / output.name
        try:
            result = refresh_run(entry) if refresh else evaluate_run(entry)
            write_json(output, result)
            if failure.exists():
                failure.unlink()
            print(f"OK {entry['run_name']} {result['seconds']:.2f}s", flush=True)
        except Exception:
            # Preserve each failed replicate and continue the array task; never drop it silently.
            errors += 1
            detail = traceback.format_exc()
            write_json(failure, {"source": entry["source"], "traceback": detail})
            print(detail, flush=True)
    if errors:
        raise RuntimeError(f"{errors} replay(s) failed; see {destination / 'failures'}")


def summarize(destination: Path, metrics_only: bool = False) -> None:
    """Summarize completed paired runs, including missing/undefined outcomes."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((destination / "manifest.json").read_text())
    rows, missing = [], []
    for task in manifest["tasks"]:
        for entry in task["entries"]:
            path = destination / "replicates" / (entry["run_name"] + ".json")
            if not path.exists():
                missing.append(entry["source"])
                continue
            result = json.loads(path.read_text())
            for method, metrics in result["methods"].items():
                rows.append({**result["config"], "method": method, "population_hsic": result["population_hsic"],
                             "source": result["source"], "seconds": result["seconds"], **metrics})
    expected = sum(len(task["entries"]) for task in manifest["tasks"])
    write_json(destination / "completion.json", {"expected": expected, "completed": expected - len(missing),
                                                "missing": missing, "missing_source_runs": manifest["missing_source_runs"]})
    if not rows:
        raise RuntimeError("No completed replicates to summarize.")
    if missing:
        raise RuntimeError(f"{len(missing)} planned replays are unfinished; refusing a partial report.")
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "replicate_metrics.csv", index=False)
    if metrics_only:
        print(f"Saved {len(rows)} metric records for {expected} completed replays.", flush=True)
        return
    keys = ["N", "eps_smoothness", "rho", "method"]
    summaries = []
    for key, block in frame.groupby(keys):
        row = dict(zip(keys, key))
        row["completed_replicates"] = len(block)
        row["defined_intervals"] = int(block.coverage.notna().sum())
        for metric in ("coverage", "ci_excludes_zero", "reject_right_tail", "empty"):
            values = block[metric].dropna().astype(float)
            n = len(values)
            row[metric] = float(values.mean()) if n else np.nan
            if n:
                # Wilson bands remain informative when all repetitions cover or reject.
                z = NormalDist().inv_cdf(0.975)
                p = float(values.mean())
                center = (p + z*z/(2*n)) / (1 + z*z/n)
                radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
                row[f"{metric}_lower"] = center - radius
                row[f"{metric}_upper"] = center + radius
        for metric in ("width_hsic", "width_sqrt_hsic"):
            row[f"mean_{metric}"] = block[metric].mean()
            row[f"median_{metric}"] = block[metric].median()
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(destination / "summary.csv", index=False)
    primary = ["cross_u_debiased_inflated", "rti_debiased", DELTA_SQUARED_U]
    labels = ["Cross-U (inflated)", "RTI", "Delta (squared-U then sqrt)"]
    for metric, label, reference in (("coverage", "95% interval coverage", .95),
                                     ("mean_width_sqrt_hsic", "Mean interval length (sqrt HSIC)", None),
                                     ("ci_excludes_zero", "95% interval excludes zero", .05)):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        for row_idx, smoothness in enumerate((.25, .75)):
            for col_idx, n in enumerate((250, 500, 1000)):
                ax = axes[row_idx, col_idx]
                for method, method_label in zip(primary, labels):
                    values = summary.loc[(summary.N == n) & (summary.eps_smoothness == smoothness)
                                         & (summary.method == method)].sort_values("rho")
                    if metric == "mean_width_sqrt_hsic":
                        ax.plot(values.rho, values[metric], marker="o", label=method_label)
                    else:
                        ax.errorbar(values.rho, values[metric],
                                    yerr=np.maximum(0, np.stack((values[metric] - values[f"{metric}_lower"],
                                                               values[f"{metric}_upper"] - values[metric]))),
                                    marker="o", capsize=2, label=method_label)
                if reference is not None:
                    ax.axhline(reference, color="gray", linestyle="--", linewidth=1)
                    ax.set_ylim(-.02, 1.02)
                ax.set_title(f"n={n}, noise smoothness={smoothness}")
                ax.set_xlabel("Heterogeneity rho")
                if col_idx == 0:
                    ax.set_ylabel(label)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(destination / f"{metric}.{extension}", dpi=160)
        plt.close(fig)
    # Paired differences make use of exactly the same simulated runs and targets.
    differences = []
    ids = ["N", "eps_smoothness", "rho", "random_key"]
    for metric in ("coverage", "width_sqrt_hsic"):
        pivot = frame.pivot(index=ids, columns="method", values=metric)
        for baseline in primary[1:]:
            difference = (pivot[primary[0]].astype(float) - pivot[baseline].astype(float)).rename("difference").reset_index()
            for key, block in difference.groupby(ids[:-1]):
                values = block.difference.dropna()
                differences.append({**dict(zip(ids[:-1], key)), "metric": metric, "baseline": baseline,
                                    "paired_runs": len(values), "cross_u_minus_baseline": values.mean(),
                                    "mc_standard_error": values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else np.nan})
    pd.DataFrame(differences).to_csv(destination / "paired_differences.csv", index=False)
    print(f"Summarized {expected-len(missing)}/{expected} available runs; outputs in {destination}", flush=True)


def main() -> None:
    global SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "plan-corrected", "run", "refresh", "summarize", "recompute-paper-delta"))
    parser.add_argument("--input", type=Path, help="Archived completed cross-U run for recompute-paper-delta")
    parser.add_argument("--source", type=Path, default=SOURCE, help="Synthetic sweep directory containing planned_jobs.yaml and per-run results.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    destination = args.output.resolve()
    if args.command == "recompute-paper-delta":
        if args.input is None:
            parser.error("recompute-paper-delta requires --input")
        recompute_paper_delta(args.input.resolve(), destination)
    elif args.command == "plan":
        make_plan(destination)
    elif args.command == "plan-corrected":
        make_corrected_plan(destination)
    elif args.command == "run":
        run_task(destination, args.task, args.pilot)
    elif args.command == "refresh":
        run_task(destination, args.task, False, refresh=True)
    else:
        summarize(destination, metrics_only=args.metrics_only)


if __name__ == "__main__":
    main()
