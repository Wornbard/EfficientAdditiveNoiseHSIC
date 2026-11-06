"""Paired operator-risk and CI-coverage experiment, with length-scale sensitivity."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from statistics import NormalDist
import sys
import time
import traceback

import numpy as np
import yaml

ROOT = Path(os.environ.get("HSIC_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
# Frozen dependencies live beside this script; the repository is only a fallback.
sys.path.append(str(ROOT))
FACTORS = (0.1, 0.5, 1.0, 2.0, 5.0)
SETTING = ("N", "f_smoothness", "eps_smoothness", "rho")
SOURCE = ROOT / "runs/multivariate_sweep/favourability_search_updated_paper"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    """Atomically save a checkpoint, rejecting nonfinite scientific results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix == ".gz":
        with gzip.open(temporary, "wt") as handle:
            handle.write(text)
    else:
        temporary.write_text(text)
    temporary.replace(path)


def checkpoint_path(output: Path, entry: dict, kind: str) -> Path:
    return output / "checkpoints" / kind / (entry["run_name"] + ".json.gz")


def plan(output: Path) -> None:
    """Freeze a balanced cohort, including explicit jobs to replace missing runs."""
    import shutil

    if (output / "manifest.json").exists():
        raise FileExistsError("A frozen manifest already exists; use run/resume.")
    planned = yaml.load((SOURCE / "planned_jobs.yaml").read_text(), Loader=yaml.CSafeLoader)
    entries = []
    for job in planned["jobs"]:
        c = job["config"]
        if (c["N"] not in (250, 500, 1000) or c["f_smoothness"] != 1
                or c["eps_smoothness"] not in (.25, .5, .75)
                or c["rho"] not in (0., .1, .2, .3, .4, .5)
                or not 10 <= c["random_key"] < 110):
            continue
        path = SOURCE / job["run_name"] / "results.yaml"
        entries.append({"run_name": job["run_name"], "config": c,
                        "source": str(path.resolve()) if path.exists() else None,
                        "source_sha256": digest(path) if path.exists() else None})
    entries.sort(key=lambda e: tuple(e["config"][k] for k in SETTING) + (e["config"]["random_key"],))
    keys = [tuple(e["config"][k] for k in SETTING) + (e["config"]["random_key"],) for e in entries]
    if len(entries) != 5400 or len(set(keys)) != 5400:
        raise ValueError("Expected exactly 54 settings with seeds 10--109.")
    code = [Path(__file__), Path(__file__).with_name("summarize.py"),
            Path(__file__).with_name("test_sensitivity.py"),
            Path(__file__).with_name("README.md"),
            ROOT / "multivariate_crossfit_core.py", ROOT / "run_witness_n_rho_diagnostics.py"]
    snapshot = output / "code_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for path in code:
        shutil.copy2(path, snapshot / path.name)
        hashes[path.name] = digest(path)
    missing = [i for i, e in enumerate(entries) if e["source"] is None]
    pilots = [next(i for i, e in enumerate(entries) if e["config"]["N"] == n
                   and e["config"]["rho"] == rho and e["config"]["eps_smoothness"] == smooth)
              for n, rho, smooth in ((250, 0., .25), (500, .3, .5), (1000, .5, .75))]
    manifest = {"entries": entries, "factors": FACTORS, "factor_applies_to": "sqrt(alpha)",
                "alpha": .05, "bootstrap_B": 1000, "missing_indices": missing,
                "pilot_indices": pilots, "baseline_chunk": 20, "sweep_chunk": 5,
                "code_sha256": hashes, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_manifest_sha256": digest(SOURCE / "planned_jobs.yaml")}
    write_json(output / "manifest.json", manifest)
    print(f"Frozen {len(entries)} repetitions; {len(missing)} require new fits; 135000 bandwidth pairs.", flush=True)


def prepare_missing(output: Path, entry: dict) -> None:
    """Generate missing seeded runs with the original tuning grids and intervals."""
    from multivariate_crossfit_core import RunConfig, run_experiment, save_yaml

    target = output / "checkpoints" / "new_sources" / entry["run_name"] / "results.yaml"
    if target.exists():
        return
    config = RunConfig(**entry["config"])
    config.compute_v_diagnostics = False
    config.compute_crossfit_rate_diagnostics = False
    config.compute_plugin_bias_diagnostic = False
    config.compute_permutation_test = False
    config.compute_x_permutation_test = False
    config.compute_standard_residual_hsic_tests = False
    config.compute_debiased_objects = True
    config.compute_bootstrap_intervals = True
    config.compute_delta_intervals = True
    config.compute_population_oracle = True
    config.bootstrap_B = 1000
    result = run_experiment(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.yaml")
    save_yaml(temporary, result)
    temporary.replace(target)


def restore(output: Path, entry: dict) -> dict:
    from run_witness_n_rho_diagnostics import restore_run, verify_saved_statistics

    path = (ROOT / entry["source"] if entry["source"] else
            output / "checkpoints" / "new_sources" / entry["run_name"] / "results.yaml")
    if entry["source"] and digest(path) != entry["source_sha256"]:
        raise ValueError(f"Source changed after planning: {path}")
    run = restore_run(str(path))
    for key in (*SETTING, "random_key", "num_folds", "dimx", "dimy"):
        if getattr(run["config"], key) != entry["config"][key]:
            raise ValueError(f"Source configuration mismatch: {key}")
    if run["config"].ci_alpha != .05 or run["config"].bootstrap_B != 1000:
        raise ValueError("The experiment requires 95% CIs and 1000 bootstrap draws.")
    verify_saved_statistics(run)
    return run


def checked_operator_errors(run: dict, check_saved_target: bool = False) -> tuple[dict, float, int]:
    """Integrate the true operator analytically in noise and numerically in X.

    Require population norm and all three squared operator errors to agree at
    successive quadrature orders. This is numerical evidence, not exact truth.
    """
    from run_witness_n_rho_diagnostics import quadrature_geometry, operator_errors

    previous = None
    for order in (256, 512, 1024, 2048):
        geometry = quadrature_geometry(run, order)
        if run["config"].rho == 0:
            np.testing.assert_allclose(geometry["population_norm_sq"], 0., atol=2e-10)
            geometry["population_norm_sq"] = 0.
        target = float(geometry["population_norm_sq"])
        if target < -2e-10 or not np.isfinite(target):
            raise ValueError("Invalid quadrature target.")
        target = max(target, 0.)
        geometry["population_norm_sq"] = target
        errors = operator_errors(run, geometry)
        values = np.array([target, *errors.values()])
        if previous is not None and np.allclose(values, previous, rtol=2e-6, atol=2e-10):
            if check_saved_target:
                np.testing.assert_allclose(target, run["raw"]["population_oracle"]["hsic"],
                                           rtol=2e-6, atol=2e-10)
            return errors, target, order
        previous = values
    raise RuntimeError("Population/operator quadrature did not converge by order 2048.")


def bootstrap_weights(n: int, folds: int, draws: int, seed: int):
    """The original within-fold bootstrap represented as centered count weights."""
    import jax
    import jax.numpy as jnp

    size = n // folds
    if n % folds:
        raise ValueError("Equal folds are required.")

    def counts(key):
        keys = jax.random.split(key, folds)
        return jnp.concatenate([jnp.bincount(jax.random.randint(k, (size,), 0, size),
                                            length=size) - 1 for k in keys])

    return jax.vmap(counts)(jax.random.split(jax.random.PRNGKey(seed), draws)).astype(jnp.float64)


def rti_from_gram(gram, weights, alpha: float) -> tuple[float, float, float]:
    """Exact bootstrap-distance identity a'Ga/n², using the existing resamples."""
    import jax.numpy as jnp

    gram = jnp.asarray(gram)
    distances = jnp.sum((weights @ gram) * weights, axis=1) / gram.shape[0] ** 2
    if float(jnp.min(distances)) < -2e-9:
        raise ValueError("Negative squared bootstrap distance.")
    radius = float(jnp.sqrt(jnp.maximum(jnp.quantile(distances, 1 - alpha), 0.)))
    center = np.sqrt(max(float(jnp.mean(gram)), 0.))
    return max(0., center - radius), center + radius, radius


def interval_record(lower: float, upper: float, target: float) -> dict:
    if not np.isfinite([lower, upper, target]).all() or upper < lower or lower < 0:
        raise ValueError("Invalid interval endpoints or population target.")
    return {"lower": lower, "upper": upper, "width": upper - lower,
            "coverage": bool(lower <= target <= upper), "zero_inclusion": bool(lower == 0.)}


def delta_intervals(u: float, v: float, score_var: float, n: int,
                    target: float, rho: float, alpha: float) -> dict:
    """Primary clipped-U delta uses V variance; retain V-centered diagnostics."""
    if rho == 0:
        return {name: None for name in ("delta_u", "delta_v", "delta_hsic_u")}
    if not np.isfinite([u, v, score_var]).all() or v <= 0 or score_var < 0:
        raise ValueError("Delta requires finite inputs, positive V, nonnegative score variance.")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    half = z * np.sqrt(score_var / (n * v))
    result = {}
    for name, center in (("delta_u", np.sqrt(max(u, 0.))), ("delta_v", np.sqrt(v))):
        result[name] = interval_record(max(0., center - half), center + half, np.sqrt(target))
    lower, upper = max(0., u - 2 * z * np.sqrt(score_var / n)), u + 2 * z * np.sqrt(score_var / n)
    # The HSIC-scale interval can have empty intersection with the parameter space.
    result["delta_hsic_u"] = ({"lower": None, "upper": None, "width": 0.,
                               "coverage": False, "zero_inclusion": False, "empty": True}
                              if upper < 0 else {**interval_record(lower, upper, target), "empty": False})
    return result


def pair_record(run: dict, factor_x: float, factor_r: float,
                errors: dict, target: float, order: int, statistics: dict) -> dict:
    config = run["config"]
    row = {key: getattr(config, key) for key in (*SETTING, "random_key")}
    row.update({"factor_x": factor_x, "factor_residual": factor_r,
                "alpha_x": run["alpha_x"], "alpha_residual": run["alpha_r"],
                "population_hsic": target, "quadrature_order": order,
                "plugin_operator_error_sq": errors["plugin"],
                "debiased_operator_error_sq": errors["debiased"],
                "oracle_operator_error_sq": errors["oracle"],
                "operator_error_sq_gain": errors["plugin"] - errors["debiased"],
                "operator_error_gain": np.sqrt(errors["plugin"]) - np.sqrt(errors["debiased"]),
                "methods": statistics})
    return row


def baseline(output: Path, entry: dict) -> None:
    target_path = checkpoint_path(output, entry, "baseline")
    if target_path.exists():
        return
    run = restore(output, entry)
    errors, target, order = checked_operator_errors(run, check_saved_target=True)
    raw, config = run["raw"], run["config"]
    methods = {}
    for method, prefix in (("plugin", "naive"), ("debiased", "debiased")):
        u, v = (raw["estimators"][f"{prefix}_hsic_{kind}"] for kind in ("u", "v"))
        var = raw["intervals"][f"delta_{prefix}_v"]["score_var"]
        boot = raw["intervals"][f"bootstrap_{prefix}"]
        methods[method] = {"hsic_u": u, "hsic_v": v, "score_var": var,
                           "center_clipped": bool(u <= 0.),
                           "rti": interval_record(boot["ci_lower"], boot["ci_upper"], np.sqrt(target)),
                           **delta_intervals(u, v, var, config.N, target, config.rho, config.ci_alpha)}
    record = pair_record(run, 1., 1., errors, target, order, methods)
    write_json(target_path, {"row": record, "replay_verified": True,
                            "source_sha256": digest(Path(run["results_path"])),
                            "reused_saved_intervals": True})


def scaled_rows(run: dict, factor_r: float, base_row: dict) -> tuple[list[dict], list]:
    """Retune v once per residual bandwidth; share it across the five X kernels."""
    import jax.numpy as jnp
    from multivariate_crossfit_core import (
        fit_v_weights, compute_xi_debias_term, gaussian_gram, center_cross_block,
        centered_blocks_from_full_gram, blocks_to_full_matrix, crossfit_u_from_blocks,
    )

    config = run["config"]
    masks = jnp.asarray(run["masks"])
    X = jnp.asarray(run["x"][:, None])
    residuals = jnp.asarray(run["residuals"][:, :, None])
    alpha_r = run["alpha_r"] * factor_r ** 2
    if factor_r == 1.:
        weights = jnp.asarray(run["weights"][:, None, :, :])
        params = run["raw"]["v_fit_hyperparams"]
    else:
        _, weights, params = fit_v_weights(config, X, residuals, masks, alpha_r)
    residual_blocks = jnp.asarray([[center_cross_block(gaussian_gram(residuals[k], residuals[l], alpha_r), mk, ml)
                                   for l, ml in enumerate(masks)] for k, mk in enumerate(masks)])
    correction = compute_xi_debias_term(config, residuals, masks, weights, alpha_r)
    eps = jnp.asarray(run["eps"][:, None])
    oracle_blocks = centered_blocks_from_full_gram(gaussian_gram(eps, eps, alpha_r), masks)
    draws = bootstrap_weights(config.N, config.num_folds, config.bootstrap_B, config.bootstrap_seed)
    rows = []
    for factor_x in FACTORS:
        if factor_x == factor_r == 1.:
            # Check the faster bootstrap against the actual saved intervals too,
            # including the source's numerical reconstruction tolerances.
            wx = centered_blocks_from_full_gram(gaussian_gram(X, X, run["alpha_x"]), masks)
            for method, h in (("plugin", wx * residual_blocks),
                              ("debiased", wx * (residual_blocks + correction))):
                lower, upper, _ = rti_from_gram(blocks_to_full_matrix(h), draws, config.ci_alpha)
                saved = base_row["methods"][method]["rti"]
                np.testing.assert_allclose([lower, upper], [saved["lower"], saved["upper"]],
                                           rtol=3e-4, atol=2e-8)
            rows.append(base_row)
            continue
        alpha_x = run["alpha_x"] * factor_x ** 2
        wx = centered_blocks_from_full_gram(gaussian_gram(X, X, alpha_x), masks)
        blocks = {"plugin": wx * residual_blocks, "debiased": wx * (residual_blocks + correction)}
        norms = {method: float(jnp.mean(h)) for method, h in blocks.items()}
        scaled = {**run, "alpha_x": alpha_x, "alpha_r": alpha_r,
                  "weights": np.asarray(weights)[:, 0],
                  "raw": {"estimators": {"naive_hsic_v": norms["plugin"], "debiased_hsic_v": norms["debiased"]},
                          "oracle": {"crossfit_hsic_v": float(jnp.mean(wx * oracle_blocks))}}}
        errors, target, order = checked_operator_errors(scaled)
        methods = {}
        for method, h in blocks.items():
            gram = blocks_to_full_matrix(h)
            v, u = norms[method], float(crossfit_u_from_blocks(h))
            var = float(jnp.mean((jnp.mean(gram, axis=1) - v) ** 2))
            lower, upper, radius = rti_from_gram(gram, draws, config.ci_alpha)
            methods[method] = {"hsic_u": u, "hsic_v": v, "score_var": var,
                               "center_clipped": bool(u <= 0.), "bootstrap_radius": radius,
                               "rti": interval_record(lower, upper, np.sqrt(target)),
                               **delta_intervals(u, v, var, config.N, target, config.rho, config.ci_alpha)}
        rows.append(pair_record(scaled, factor_x, factor_r, errors, target, order, methods))
    return rows, params


def sweep(output: Path, entry: dict) -> None:
    done = [checkpoint_path(output, entry, f"residual_{factor:g}") for factor in FACTORS]
    if all(path.exists() for path in done):
        return
    run = restore(output, entry)
    base = read_json(checkpoint_path(output, entry, "baseline"))
    if base["source_sha256"] != digest(Path(run["results_path"])):
        raise ValueError("Baseline/source mismatch.")
    # Run the original bandwidth first; each residual factor has its own checkpoint.
    for factor in (1., .1, .5, 2., 5.):
        path = checkpoint_path(output, entry, f"residual_{factor:g}")
        if path.exists():
            continue
        start = time.monotonic()
        rows, params = scaled_rows(run, factor, base["row"])
        write_json(path, {"rows": rows, "v_hyperparameters": params,
                         "replay_verified": True, "source_sha256": base["source_sha256"],
                         "v_retuned": factor != 1., "seconds": time.monotonic() - start})
        print(f"OK {entry['run_name']} residual factor {factor:g}: {time.monotonic() - start:.1f}s", flush=True)


def run_task(output: Path, mode: str, task: int) -> None:
    manifest = read_json(output / "manifest.json")
    for name, expected in manifest["code_sha256"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError(f"Frozen code changed: {name}")
    if mode == "prepare":
        indices = [manifest["missing_indices"][task]]
    elif mode == "pilot":
        indices = [manifest["pilot_indices"][task]]
    else:
        chunk = manifest["baseline_chunk" if mode == "baseline" else "sweep_chunk"]
        indices = range(task * chunk, min((task + 1) * chunk, len(manifest["entries"])))
    failures = 0
    for index in indices:
        entry = manifest["entries"][index]
        failure = checkpoint_path(output, entry, f"failures_{mode}")
        try:
            if mode == "prepare":
                prepare_missing(output, entry)
            elif mode == "baseline":
                baseline(output, entry)
            else:
                sweep(output, entry)
            if failure.exists():
                failure.unlink()
            print(f"OK {mode} {entry['run_name']}", flush=True)
        except Exception:
            # Record every failed repetition, continue independent repetitions,
            # and fail the array element so downstream success gates cannot pass.
            failures += 1
            detail = traceback.format_exc()
            write_json(failure, {"run_name": entry["run_name"], "traceback": detail})
            print(detail, flush=True)
    if failures:
        raise RuntimeError(f"{failures} failed repetitions; see checkpoints/failures_{mode}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "prepare", "baseline", "pilot", "sweep"))
    parser.add_argument("--source", type=Path, default=SOURCE, help="Synthetic sweep with planned_jobs.yaml and saved results")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    if args.mode == "plan":
        plan(args.output.resolve())
    else:
        run_task(args.output.resolve(), args.mode, args.task)
