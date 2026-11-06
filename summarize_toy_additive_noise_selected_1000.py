from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import run_toy_additive_noise_selected_1000 as runner


ROOT = Path(__file__).resolve().parent
RESULT_DIR = runner.RESULT_DIR
CANONICAL_RESULT_DIR = (
    ROOT / "results/toy_additive_noise_classical_diagnostics"
)
CANONICAL_TABLE_DIR = (
    ROOT / "tables/toy_additive_noise_classical_diagnostics"
)

COMBINED_RAW_CSV = RESULT_DIR / "selected_tests_raw_repetitions_100_1099.csv"
INPUT_INVENTORY_CSV = (
    RESULT_DIR / "selected_tests_input_repetition_file_hashes.csv"
)
LOCAL_SUMMARY_CSV = (
    RESULT_DIR / "selected_tests_rejection_probabilities_long.csv"
)
LOCAL_WIDE_CSV = (
    RESULT_DIR / "selected_tests_rejection_probabilities_wide.csv"
)
LOCAL_PUBLICATION_AUDIT_CSV = (
    RESULT_DIR / "selected_tests_publication_audit.csv"
)
LOCAL_CALIBRATION_AUDIT_CSV = (
    RESULT_DIR / "selected_tests_null_calibration_audit.csv"
)
LOCAL_MARKDOWN = RESULT_DIR / "selected_tests_rejection_probabilities.md"
LOCAL_LATEX = RESULT_DIR / "selected_tests_rejection_probabilities.tex"
MANIFEST_JSON = RESULT_DIR / "selected_tests_manifest.json"

CANONICAL_SUMMARY_CSV = (
    CANONICAL_RESULT_DIR / "selected_tests_rejection_probabilities_long.csv"
)
CANONICAL_WIDE_CSV = (
    CANONICAL_RESULT_DIR / "selected_tests_rejection_probabilities_wide.csv"
)
CANONICAL_PUBLICATION_AUDIT_CSV = (
    CANONICAL_RESULT_DIR / "selected_tests_publication_audit.csv"
)
CANONICAL_CALIBRATION_AUDIT_CSV = (
    CANONICAL_RESULT_DIR / "selected_tests_null_calibration_audit.csv"
)
CANONICAL_MARKDOWN = (
    CANONICAL_TABLE_DIR / "selected_tests_rejection_probabilities.md"
)
CANONICAL_LATEX = (
    CANONICAL_TABLE_DIR / "selected_tests_rejection_probabilities.tex"
)

CONFIDENCE_LEVEL = 0.95
NOMINAL_LEVEL = runner.ALPHA
EXPECTED_REPETITIONS = tuple(
    range(runner.HOLDOUT_START, runner.HOLDOUT_STOP)
)
EXPECTED_SCENARIOS = tuple(runner.SCENARIOS.values())
EXPECTED_METHODS = tuple(runner.SELECTED_METHOD_KEYS)
EXPECTED_ROWS = (
    len(EXPECTED_REPETITIONS)
    * len(EXPECTED_SCENARIOS)
    * len(EXPECTED_METHODS)
)
NULL_SCENARIO = runner.SCENARIOS[(0.0, "X->Y")]


PUBLICATION_SPECS = OrderedDict(
    [
        (
            "old_hsic_cf",
            {
                "group": "Gaussian kernel HSIC",
                "label": "cross-fitted plug-in permutation test",
                "test_family": "HSIC",
                "configuration": "Gaussian kernels",
                "procedure": "cross-fit permutation",
                "source_method": "core full naive",
                "technical_definition": (
                    "Gaussian predictor and residual kernels; five-fold "
                    "out-of-fold plug-in residuals; permutation p-value"
                ),
            },
        ),
        (
            "old_hsic_debiased",
            {
                "group": "Gaussian kernel HSIC",
                "label": "cross-fitted debiased test (ours)",
                "test_family": "HSIC",
                "configuration": "Gaussian kernels",
                "procedure": "debiased RTI bootstrap",
                "source_method": "core full debiased",
                "technical_definition": (
                    "Gaussian predictor and debiased residual kernels; "
                    "five-fold out-of-fold residuals; reverse-triangle "
                    "bootstrap p-value"
                ),
            },
        ),
        (
            "bp_linear",
            {
                "group": "BP/White",
                "label": "linear X feature",
                "test_family": "Variance score",
                "configuration": "linear",
                "procedure": "cross-fitted diagnostic variant",
                "source_method": "variance_score",
                "technical_definition": (
                    "n*R^2 chi-square score/LM test of squared out-of-fold "
                    "residuals on an intercept plus one linear X feature"
                ),
            },
        ),
        (
            "bp_fourier2",
            {
                "group": "BP/White",
                "label": "Fourier X features (K=2)",
                "test_family": "Variance score",
                "configuration": "fourier2",
                "procedure": "cross-fitted diagnostic variant",
                "source_method": "variance_score",
                "technical_definition": (
                    "n*R^2 chi-square score/LM test of squared out-of-fold "
                    "residuals on an intercept plus Fourier harmonics 1--2"
                ),
            },
        ),
        (
            "bp_fourier4",
            {
                "group": "BP/White",
                "label": "Fourier X features (K=4)",
                "test_family": "Variance score",
                "configuration": "fourier4",
                "procedure": "cross-fitted diagnostic variant",
                "source_method": "variance_score",
                "technical_definition": (
                    "n*R^2 chi-square score/LM test of squared out-of-fold "
                    "residuals on an intercept plus Fourier harmonics 1--4"
                ),
            },
        ),
        (
            "levene_b10",
            {
                "group": "Levene",
                "label": "B=10",
                "test_family": "Levene",
                "configuration": "B=10",
                "procedure": "cross-fitted diagnostic variant",
                "source_method": "levene",
                "technical_definition": (
                    "Levene F test; 10 equal-frequency X bins; "
                    "mean-centered absolute deviations of out-of-fold "
                    "residuals"
                ),
            },
        ),
        (
            "brown_forsythe_b10",
            {
                "group": "Brown-Forsythe",
                "label": "B=10",
                "test_family": "Brown-Forsythe",
                "configuration": "B=10",
                "procedure": "cross-fitted diagnostic variant",
                "source_method": "brown_forsythe",
                "technical_definition": (
                    "Brown-Forsythe F test; 10 equal-frequency X bins; "
                    "median-centered absolute deviations of out-of-fold "
                    "residuals"
                ),
            },
        ),
        (
            "quantile_central_fourier4",
            {
                "group": "Quantile regression",
                "label": "central grid, Fourier X features (K=4)",
                "test_family": "Quantile regression",
                "configuration": "central, fourier4",
                "procedure": "cross-fitted diagnostic variant",
                "source_method": "quantile_regression",
                "technical_definition": (
                    "Separate intercept-inclusive robust Wald tests of all "
                    "eight Fourier slopes at tau=0.25,0.5,0.75; three-test "
                    "Bonferroni global p-value"
                ),
            },
        ),
    ]
)


LATEX_TITLES = {
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
    "brown_forsythe_b10": "Brown--Forsythe, $B=10$",
    "quantile_central_fourier4": (
        r"Quantile regression, $\tau\in\{0.25,0.5,0.75\}$, "
        r"Fourier $X$ features ($K=4$)"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    destination_mode = (
        stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o664
    )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.chmod(destination_mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    destination_mode = (
        stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o664
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
        temporary.chmod(destination_mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def backup_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_n100_selection{path.suffix}")


def preserve_n100_canonical_artifacts() -> dict[str, dict[str, object]]:
    canonical = (
        CANONICAL_SUMMARY_CSV,
        CANONICAL_WIDE_CSV,
        CANONICAL_PUBLICATION_AUDIT_CSV,
        CANONICAL_MARKDOWN,
        CANONICAL_LATEX,
    )
    if not any(path.exists() for path in canonical):
        return {}
    if not CANONICAL_SUMMARY_CSV.exists():
        raise RuntimeError(
            "Canonical artifacts exist without the canonical long CSV; "
            "refusing to overwrite an unidentifiable table state."
        )

    current = pd.read_csv(CANONICAL_SUMMARY_CSV)
    is_n100 = (
        len(current) == len(EXPECTED_SCENARIOS) * len(EXPECTED_METHODS)
        and "n_valid" in current
        and pd.to_numeric(current["n_valid"], errors="coerce").eq(100).all()
    )
    is_fresh_n1000 = (
        len(current) == len(EXPECTED_SCENARIOS) * len(EXPECTED_METHODS)
        and "source_repetition_start" in current
        and "source_repetition_end" in current
        and pd.to_numeric(
            current["source_repetition_start"], errors="coerce"
        ).eq(runner.HOLDOUT_START).all()
        and pd.to_numeric(
            current["source_repetition_end"], errors="coerce"
        ).eq(runner.HOLDOUT_STOP - 1).all()
    )
    if not (is_n100 or is_fresh_n1000):
        raise RuntimeError(
            "The current canonical table is neither the recognized n=100 "
            "selection table nor this fresh n=1000 table. Refusing to "
            "overwrite it."
        )

    preserved: dict[str, dict[str, object]] = {}
    if is_n100:
        missing = [str(path) for path in canonical if not path.exists()]
        if missing:
            raise RuntimeError(
                "The n=100 canonical artifact set is incomplete: "
                + ", ".join(missing)
            )
        for source in canonical:
            destination = backup_path(source)
            if destination.exists():
                if sha256_file(destination) != sha256_file(source):
                    raise RuntimeError(
                        f"Existing selection backup differs from {source}: "
                        f"{destination}. Refusing to overwrite either file."
                    )
            else:
                atomic_copy(source, destination)
            preserved[str(destination.relative_to(ROOT))] = {
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
                "source_before_replacement": str(source.relative_to(ROOT)),
            }
    else:
        for source in canonical:
            destination = backup_path(source)
            # A fresh checkout can generate the 1000-run table directly,
            # without ever having generated preliminary selection tables.
            if not destination.exists():
                continue
            preserved[str(destination.relative_to(ROOT))] = {
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
                "source_before_replacement": str(source.relative_to(ROOT)),
            }
    return preserved


def load_and_validate_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        runner.HOLDOUT_START != 100
        or runner.HOLDOUT_STOP != 1100
        or runner.TARGET_REPETITIONS != 1000
    ):
        raise AssertionError(
            "This summarizer is locked to fresh repetitions 100,...,1099."
        )
    if tuple(PUBLICATION_SPECS) != EXPECTED_METHODS:
        raise AssertionError(
            "Publication specifications do not match the runner's selected "
            "method order."
        )

    paths = [
        (repetition, runner.repetition_path(repetition))
        for repetition in EXPECTED_REPETITIONS
    ]
    missing = [str(path) for _, path in paths if not path.is_file()]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f" ... ({len(missing)} missing)"
        raise FileNotFoundError(
            "The fresh 1000-run batch is incomplete; no outputs were "
            f"written. Missing: {preview}{suffix}"
        )

    invalid = [
        str(path)
        for repetition, path in paths
        if not runner.validate_repetition_file(path, repetition)
    ]
    if invalid:
        preview = ", ".join(invalid[:8])
        suffix = "" if len(invalid) <= 8 else f" ... ({len(invalid)} invalid)"
        raise RuntimeError(
            "One or more repetition files failed the runner's source "
            f"validator; no outputs were written: {preview}{suffix}"
        )

    frames = []
    inventory_rows = []
    for repetition, path in paths:
        frame = pd.read_csv(path)
        frames.append(frame)
        inventory_rows.append(
            {
                "repetition": repetition,
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    raw = pd.concat(frames, ignore_index=True)
    inventory = pd.DataFrame(inventory_rows)
    validate_combined_raw(raw)
    return raw, inventory


def validate_combined_raw(raw: pd.DataFrame) -> None:
    key = ["repetition", "scenario", "method_key"]
    if len(raw) != EXPECTED_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_ROWS:,} raw rows; found {len(raw):,}."
        )
    if raw.duplicated(key).any() or raw.duplicated().any():
        raise AssertionError("The 32,000 raw rows are not unique.")
    if set(raw["repetition"]) != set(EXPECTED_REPETITIONS):
        raise AssertionError("Raw repetition IDs are not exactly 100,...,1099.")
    if set(raw["scenario"]) != set(EXPECTED_SCENARIOS):
        raise AssertionError("Raw scenario labels differ from the runner.")
    if set(raw["method_key"]) != set(EXPECTED_METHODS):
        raise AssertionError("Raw method keys differ from the runner.")
    for (rho, direction), scenario in runner.SCENARIOS.items():
        rows = raw.loc[raw["scenario"].eq(scenario)]
        if (
            not np.isclose(
                pd.to_numeric(rows["rho"]).to_numpy(dtype=float),
                float(rho),
            ).all()
            or not rows["direction"].eq(direction).all()
        ):
            raise AssertionError(
                f"Scenario metadata is mislabeled for {scenario}."
            )

    per_repetition = raw.groupby("repetition", sort=False).size()
    if not per_repetition.eq(
        len(EXPECTED_SCENARIOS) * len(EXPECTED_METHODS)
    ).all():
        raise AssertionError("A repetition does not contain exactly 32 rows.")
    per_cell = raw.groupby(
        ["scenario", "method_key"], sort=False
    ).size()
    if not per_cell.eq(len(EXPECTED_REPETITIONS)).all():
        raise AssertionError("A scenario/method cell does not have 1000 rows.")

    expected_title = raw["method_key"].map(runner.METHOD_TITLES)
    if not raw["column_title"].eq(expected_title).all():
        raise AssertionError("A method title does not match the runner title.")
    if not raw["residual_scheme"].eq("five-fold cross-fitted").all():
        raise AssertionError("A row is not labelled five-fold cross-fitted.")
    if not pd.to_numeric(raw["n_folds"]).eq(
        int(runner.CORE_CONFIG.num_folds)
    ).all():
        raise AssertionError("Unexpected number of cross-fitting folds.")
    if not pd.to_numeric(raw["n_eval"]).eq(int(runner.CORE_CONFIG.N)).all():
        raise AssertionError("Unexpected evaluation sample size.")

    group_key = ["repetition", "scenario"]
    oof_counts = raw.groupby(group_key)[
        "residual_sha256"
    ].nunique(dropna=False)
    fold_counts = raw.groupby(group_key)[
        "fold_residual_sha256"
    ].nunique(dropna=False)
    expected_first_stage_groups = (
        len(EXPECTED_REPETITIONS) * len(EXPECTED_SCENARIOS)
    )
    if (
        len(oof_counts) != expected_first_stage_groups
        or not oof_counts.eq(1).all()
        or len(fold_counts) != expected_first_stage_groups
        or not fold_counts.eq(1).all()
    ):
        raise AssertionError(
            "The eight methods do not share one OOF residual vector and one "
            "fold-residual tensor in every repetition/scenario."
        )
    hash_pattern = r"^[0-9a-f]{64}$"
    if (
        not raw["residual_sha256"]
        .astype(str)
        .str.fullmatch(hash_pattern)
        .all()
        or not raw["fold_residual_sha256"]
        .astype(str)
        .str.fullmatch(hash_pattern)
        .all()
    ):
        raise AssertionError("A first-stage hash is missing or malformed.")

    valid = raw["valid"].eq(True)
    p_values = pd.to_numeric(raw["p_value"], errors="coerce")
    if (
        not np.isfinite(p_values.loc[valid]).all()
        or not p_values.loc[valid].between(0.0, 1.0).all()
        or p_values.loc[~valid].notna().any()
    ):
        raise AssertionError("The valid/invalid p-value encoding is wrong.")
    actual_reject = raw.loc[valid, "reject"].astype(bool).to_numpy()
    expected_reject = np.where(
        raw.loc[valid, "rejection_rule"].eq("p <= 0.05"),
        p_values.loc[valid] <= NOMINAL_LEVEL,
        p_values.loc[valid] < NOMINAL_LEVEL,
    )
    if not np.array_equal(actual_reject, expected_reject):
        raise AssertionError("A rejection indicator disagrees with its p-value.")
    if raw.loc[~valid, "reject"].notna().any():
        raise AssertionError("An invalid test row has a rejection indicator.")

    expected_rules = {
        "old_hsic_cf": "p <= 0.05",
        "old_hsic_debiased": "p <= 0.05",
        "bp_linear": "p < 0.05",
        "bp_fourier2": "p < 0.05",
        "bp_fourier4": "p < 0.05",
        "levene_b10": "p < 0.05",
        "brown_forsythe_b10": "p < 0.05",
        "quantile_central_fourier4": "p < 0.05",
    }
    if not raw["rejection_rule"].eq(
        raw["method_key"].map(expected_rules)
    ).all():
        raise AssertionError("A method has the wrong rejection convention.")

    rho_indices = {
        float(rho): index for index, rho in enumerate(runner.RHO_VALUES)
    }
    direction_indices = {"X->Y": 0, "Y->X": 1}
    repetitions = pd.to_numeric(raw["repetition"]).to_numpy(dtype=np.int64)
    rhos = pd.to_numeric(raw["rho"]).to_numpy(dtype=float)
    rho_index = np.array(
        [
            next(
                index
                for value, index in rho_indices.items()
                if np.isclose(rho, value)
            )
            for rho in rhos
        ],
        dtype=np.int64,
    )
    expected_data_seeds = (
        int(runner.DATA_SEED_BASE)
        + 1_000 * repetitions
        + 100 * rho_index
    )
    direction_index = raw["direction"].map(direction_indices).to_numpy()
    if pd.isna(direction_index).any():
        raise AssertionError("Unexpected direction label.")
    expected_hsic_seeds = (
        expected_data_seeds
        + 1_000_000
        + direction_index.astype(np.int64)
    )
    if not np.array_equal(
        pd.to_numeric(raw["data_seed"]).to_numpy(dtype=np.int64),
        expected_data_seeds,
    ):
        raise AssertionError("A data seed differs from the locked seed rule.")
    if not np.array_equal(
        pd.to_numeric(raw["hsic_seed"]).to_numpy(dtype=np.int64),
        expected_hsic_seeds,
    ):
        raise AssertionError("An HSIC/resampling seed differs from its rule.")

    expected_rank = {
        "bp_linear": 1,
        "bp_fourier2": 4,
        "bp_fourier4": 8,
        "quantile_central_fourier4": 8,
    }
    for method_key, rank in expected_rank.items():
        rows = raw.loc[raw["method_key"].eq(method_key)]
        if not pd.to_numeric(rows["effective_rank"]).eq(rank).all():
            raise AssertionError(f"Wrong effective rank for {method_key}.")
    expected_basis = {
        "bp_linear": "linear",
        "bp_fourier2": "fourier2",
        "bp_fourier4": "fourier4",
        "quantile_central_fourier4": "fourier4",
    }
    for method_key, basis in expected_basis.items():
        rows = raw.loc[raw["method_key"].eq(method_key)]
        if not rows["feature_basis"].eq(basis).all():
            raise AssertionError(f"Wrong feature basis for {method_key}.")
    binned = raw["method_key"].isin(
        ("levene_b10", "brown_forsythe_b10")
    )
    if not pd.to_numeric(raw.loc[binned, "n_bins"]).eq(10).all():
        raise AssertionError("A retained binned test does not use B=10.")
    quantile = raw["method_key"].eq("quantile_central_fourier4")
    if not raw.loc[quantile, "quantile_grid"].eq("central").all():
        raise AssertionError("The retained quantile test is not central-grid.")


def exact_interval(rejections: int, n_valid: int) -> tuple[float, float]:
    if n_valid <= 0:
        raise RuntimeError("Cannot construct a binomial interval with n=0.")
    interval = binomtest(rejections, n_valid).proportion_ci(
        confidence_level=CONFIDENCE_LEVEL,
        method="exact",
    )
    return float(interval.low), float(interval.high)


def unique_nonmissing(frame: pd.DataFrame, column: str):
    values = frame[column].dropna().unique()
    if len(values) > 1:
        raise AssertionError(
            f"Expected at most one {column} value; found {values!r}."
        )
    return values[0] if len(values) == 1 else np.nan


def format_interval(
    estimate: float,
    lower: float,
    upper: float,
    n_valid: int,
) -> str:
    text = f"{estimate:.3f} [{lower:.3f}, {upper:.3f}]"
    if n_valid != len(EXPECTED_REPETITIONS):
        text += f"; n={n_valid}"
    return text


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario_order, scenario in enumerate(EXPECTED_SCENARIOS):
        for method_order, method_key in enumerate(EXPECTED_METHODS):
            frame = raw.loc[
                raw["scenario"].eq(scenario)
                & raw["method_key"].eq(method_key)
            ].sort_values("repetition", kind="stable")
            if len(frame) != len(EXPECTED_REPETITIONS):
                raise AssertionError("A summary cell is incomplete.")
            valid = frame["valid"].eq(True)
            n_valid = int(valid.sum())
            n_invalid = int((~valid).sum())
            rejections = int(frame.loc[valid, "reject"].astype(bool).sum())
            estimate = rejections / n_valid
            lower, upper = exact_interval(rejections, n_valid)
            mcse = math.sqrt(estimate * (1.0 - estimate) / n_valid)
            spec = PUBLICATION_SPECS[method_key]
            rho = float(unique_nonmissing(frame, "rho"))
            direction = str(unique_nonmissing(frame, "direction"))
            null_p = (
                float(
                    binomtest(
                        rejections,
                        n_valid,
                        p=NOMINAL_LEVEL,
                        alternative="two-sided",
                    ).pvalue
                )
                if scenario == NULL_SCENARIO
                else np.nan
            )
            formatted = format_interval(
                estimate, lower, upper, n_valid
            )
            row_label = (
                f"{spec['test_family']} | {spec['configuration']} | "
                f"{spec['procedure']} | train 80%, OOF test 100%"
            )
            rows.append(
                {
                    "scenario": scenario,
                    "test_family": spec["test_family"],
                    "configuration": spec["configuration"],
                    "procedure": spec["procedure"],
                    "source_method": spec["source_method"],
                    "effective_rank": unique_nonmissing(
                        frame, "effective_rank"
                    ),
                    "n_bins": unique_nonmissing(frame, "n_bins"),
                    "n_resamples": (
                        int(runner.CORE_CONFIG.bootstrap_B)
                        if method_key
                        in ("old_hsic_cf", "old_hsic_debiased")
                        else np.nan
                    ),
                    "training_fraction": (
                        runner.CORE_CONFIG.num_folds - 1
                    )
                    / runner.CORE_CONFIG.num_folds,
                    "testing_fraction": 1.0,
                    "n_attempted": len(frame),
                    "n_valid": n_valid,
                    "n_invalid": n_invalid,
                    "rejections": rejections,
                    "rejection_probability": estimate,
                    "mcse": mcse,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": CONFIDENCE_LEVEL,
                    "confidence_interval_method": (
                        "exact Clopper-Pearson binomial"
                    ),
                    "null_calibration_p_value": null_p,
                    "selection_note": (
                        "Method/configuration retained after the separate "
                        "n=100 selection-stage comparison; this estimate uses "
                        "fresh repetitions 100--1099."
                    ),
                    "sort_group": method_order,
                    "sort_configuration": spec["configuration"],
                    "row_label": row_label,
                    "formatted_value": (
                        f"{estimate:.3f} "
                        f"[{lower:.3f}, {upper:.3f}]; n={n_valid}"
                    ),
                    "rho": rho,
                    "direction": direction,
                    "estimand_role": (
                        "size" if scenario == NULL_SCENARIO else "power"
                    ),
                    "scenario_order": scenario_order,
                    "method_order": method_order,
                    "method_key": method_key,
                    "column_group": spec["group"],
                    "column_label": spec["label"],
                    "column_title": runner.METHOD_TITLES[method_key],
                    "estimate_and_95_ci": formatted,
                    "source_repetition_start": runner.HOLDOUT_START,
                    "source_repetition_end": runner.HOLDOUT_STOP - 1,
                    "fresh_after_selection": True,
                    "raw_source_path": str(
                        COMBINED_RAW_CSV.relative_to(ROOT)
                    ),
                }
            )
    summary = pd.DataFrame(rows).sort_values(
        ["scenario_order", "method_order"], kind="stable"
    ).reset_index(drop=True)
    if (
        len(summary) != len(EXPECTED_SCENARIOS) * len(EXPECTED_METHODS)
        or summary[["scenario", "method_key"]].duplicated().any()
    ):
        raise AssertionError("The summary is not a unique 4 x 8 table.")
    for row in summary.itertuples(index=False):
        lower, upper = exact_interval(row.rejections, row.n_valid)
        if not (
            np.isclose(row.rejection_probability, row.rejections / row.n_valid)
            and np.isclose(row.ci_lower, lower)
            and np.isclose(row.ci_upper, upper)
        ):
            raise AssertionError("A summary row failed exact reproduction.")
    return summary


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or not ((0.0 <= values) & (values <= 1.0)).all()
    ):
        raise ValueError("Holm adjustment requires finite p-values in [0,1].")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    factors = len(values) - np.arange(len(values))
    adjusted_sorted = np.minimum(
        1.0,
        np.maximum.accumulate(sorted_values * factors),
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def build_calibration_audit(summary: pd.DataFrame) -> pd.DataFrame:
    null = (
        summary.loc[summary["scenario"].eq(NULL_SCENARIO)]
        .sort_values("method_order", kind="stable")
        .copy()
    )
    if len(null) != len(EXPECTED_METHODS):
        raise AssertionError("The null calibration audit must have eight rows.")

    audit_rows = []
    for row in null.itertuples(index=False):
        two_sided = float(
            binomtest(
                row.rejections,
                row.n_valid,
                p=NOMINAL_LEVEL,
                alternative="two-sided",
            ).pvalue
        )
        liberal = float(
            binomtest(
                row.rejections,
                row.n_valid,
                p=NOMINAL_LEVEL,
                alternative="greater",
            ).pvalue
        )
        conservative = float(
            binomtest(
                row.rejections,
                row.n_valid,
                p=NOMINAL_LEVEL,
                alternative="less",
            ).pvalue
        )
        if row.rejection_probability > NOMINAL_LEVEL:
            direction = "liberal"
            directional = liberal
        elif row.rejection_probability < NOMINAL_LEVEL:
            direction = "conservative"
            directional = conservative
        else:
            direction = "at nominal"
            directional = 1.0
        audit_rows.append(
            {
                "method_order": row.method_order,
                "method_key": row.method_key,
                "column_title": row.column_title,
                "scenario": row.scenario,
                "nominal_level": NOMINAL_LEVEL,
                "n_attempted": row.n_attempted,
                "n_valid": row.n_valid,
                "n_invalid": row.n_invalid,
                "rejections": row.rejections,
                "rejection_probability": row.rejection_probability,
                "mcse": row.mcse,
                "size_ratio_to_nominal": (
                    row.rejection_probability / NOMINAL_LEVEL
                ),
                "ci_lower": row.ci_lower,
                "ci_upper": row.ci_upper,
                "deviation_from_nominal": (
                    row.rejection_probability - NOMINAL_LEVEL
                ),
                "calibration_direction": direction,
                "exact_two_sided_p_value": two_sided,
                "exact_liberal_one_sided_p_value": liberal,
                "exact_conservative_one_sided_p_value": conservative,
                "directional_one_sided_p_value": directional,
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit["holm_family_size"] = len(audit)
    audit["exact_two_sided_holm_p_value"] = holm_adjust(
        audit["exact_two_sided_p_value"].to_numpy()
    )
    audit["exact_liberal_one_sided_holm_p_value"] = holm_adjust(
        audit["exact_liberal_one_sided_p_value"].to_numpy()
    )
    audit["exact_conservative_one_sided_holm_p_value"] = holm_adjust(
        audit["exact_conservative_one_sided_p_value"].to_numpy()
    )
    audit["directional_one_sided_holm_p_value"] = holm_adjust(
        audit["directional_one_sided_p_value"].to_numpy()
    )
    audit["two_sided_holm_significant"] = (
        audit["exact_two_sided_holm_p_value"] <= NOMINAL_LEVEL
    )
    audit["liberal_holm_significant"] = (
        audit["exact_liberal_one_sided_holm_p_value"] <= NOMINAL_LEVEL
    )
    audit["conservative_holm_significant"] = (
        audit["exact_conservative_one_sided_holm_p_value"] <= NOMINAL_LEVEL
    )
    audit["directional_holm_significant"] = (
        audit["directional_one_sided_holm_p_value"] <= NOMINAL_LEVEL
    )
    audit["statistically_inconsistent_with_5pct"] = audit[
        "two_sided_holm_significant"
    ]
    audit["anti_conservative_5pct_control_flag"] = audit[
        "liberal_holm_significant"
    ]
    audit["gross_departure_rate_threshold"] = (
        "rejection_probability >= 0.10 or <= 0.025"
    )
    audit["gross_departure_flag"] = (
        audit["two_sided_holm_significant"]
        & (
            (audit["rejection_probability"] >= 0.10)
            | (audit["rejection_probability"] <= 0.025)
        )
    )
    audit["wildly_inconsistent_flag"] = audit["gross_departure_flag"]
    return audit


def hash_ordered_p_values(frame: pd.DataFrame) -> str:
    lines = []
    for row in frame.sort_values("repetition").itertuples(index=False):
        token = (
            float(row.p_value).hex()
            if bool(row.valid)
            else "INVALID"
        )
        lines.append(f"{int(row.repetition)}\t{token}\n")
    return sha256_text("".join(lines))


def hash_ordered_rejections(frame: pd.DataFrame) -> str:
    lines = []
    for row in frame.sort_values("repetition").itertuples(index=False):
        token = (
            ("1" if bool(row.reject) else "0")
            if bool(row.valid)
            else "INVALID"
        )
        lines.append(f"{int(row.repetition)}\t{token}\n")
    return sha256_text("".join(lines))


def hash_ordered_strings(
    frame: pd.DataFrame,
    column: str,
) -> str:
    lines = [
        f"{int(row.repetition)}\t{getattr(row, column)}\n"
        for row in frame.sort_values("repetition").itertuples(index=False)
    ]
    return sha256_text("".join(lines))


def build_publication_audit(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for selected in summary.itertuples(index=False):
        frame = raw.loc[
            raw["scenario"].eq(selected.scenario)
            & raw["method_key"].eq(selected.method_key)
        ].sort_values("repetition", kind="stable")
        valid = frame["valid"].eq(True)
        direct_rejections = int(
            frame.loc[valid, "reject"].astype(bool).sum()
        )
        direct_n_valid = int(valid.sum())
        direct_lower, direct_upper = exact_interval(
            direct_rejections, direct_n_valid
        )
        raw_record_match = (
            direct_n_valid == selected.n_valid
            and direct_rejections == selected.rejections
            and np.isclose(
                direct_rejections / direct_n_valid,
                selected.rejection_probability,
            )
            and np.isclose(direct_lower, selected.ci_lower)
            and np.isclose(direct_upper, selected.ci_upper)
        )
        spec = PUBLICATION_SPECS[selected.method_key]
        rows.append(
            {
                "scenario_order": selected.scenario_order,
                "method_order": selected.method_order,
                "scenario": selected.scenario,
                "method_key": selected.method_key,
                "column_title": selected.column_title,
                "technical_definition": spec["technical_definition"],
                "source_path": str(COMBINED_RAW_CSV.relative_to(ROOT)),
                "source_method": spec["source_method"],
                "source_repetition_start": runner.HOLDOUT_START,
                "source_repetition_end": runner.HOLDOUT_STOP - 1,
                "residual_scheme": "five-fold cross-fitted",
                "p_value_field": "p_value",
                "rejection_rule": unique_nonmissing(
                    frame, "rejection_rule"
                ),
                "n_attempted": len(frame),
                "n_valid": direct_n_valid,
                "n_invalid": int((~valid).sum()),
                "rejections": direct_rejections,
                "rejection_probability": (
                    direct_rejections / direct_n_valid
                ),
                "ci_lower": direct_lower,
                "ci_upper": direct_upper,
                "p_value_vector_sha256": hash_ordered_p_values(frame),
                "rejection_vector_sha256": hash_ordered_rejections(frame),
                "residual_hash_vector_sha256": hash_ordered_strings(
                    frame, "residual_sha256"
                ),
                "fold_residual_hash_vector_sha256": hash_ordered_strings(
                    frame, "fold_residual_sha256"
                ),
                "vector_hash_encoding": (
                    "UTF-8 lines ordered by repetition; float.hex p-values; "
                    "0/1 rejection tokens; explicit INVALID token"
                ),
                "raw_record_match": raw_record_match,
                "title_definition_match": (
                    selected.column_title
                    == runner.METHOD_TITLES[selected.method_key]
                ),
                "shared_first_stage_hashes_verified": True,
                "fresh_after_selection": True,
            }
        )
    audit = pd.DataFrame(rows).sort_values(
        ["scenario_order", "method_order"], kind="stable"
    ).reset_index(drop=True)
    if (
        len(audit) != len(EXPECTED_SCENARIOS) * len(EXPECTED_METHODS)
        or not audit["raw_record_match"].all()
        or not audit["title_definition_match"].all()
        or not audit["shared_first_stage_hashes_verified"].all()
    ):
        raise AssertionError("The 32-row publication audit failed.")
    return audit


def build_wide(summary: pd.DataFrame) -> pd.DataFrame:
    display = (
        summary.pivot(
            index="scenario",
            columns="method_key",
            values="estimate_and_95_ci",
        )
        .reindex(index=EXPECTED_SCENARIOS, columns=EXPECTED_METHODS)
    )
    if display.isna().any().any():
        raise AssertionError("The publication table has a missing cell.")
    display.index.name = "Regime"
    display.columns = [
        (
            f"{PUBLICATION_SPECS[key]['group']} | "
            f"{PUBLICATION_SPECS[key]['label']}"
        )
        for key in display.columns
    ]
    return display.reset_index()


def build_markdown(summary: pd.DataFrame) -> str:
    headers = ["Regime"] + [
        runner.METHOD_TITLES[key] for key in EXPECTED_METHODS
    ]
    labels = {
        "rho=0, X->Y (size)": "rho=0, X→Y (size)",
        "rho=0, Y->X (power)": "rho=0, Y→X (power)",
        "rho=0.35, X->Y (power)": "rho=0.35, X→Y (power)",
        "rho=0.35, Y->X (power)": "rho=0.35, Y→X (power)",
    }
    n_invalid = int(summary["n_invalid"].sum())
    if n_invalid == 0:
        sample_note = (
            "Each entry is `rejection probability [95% exact confidence "
            "interval]`; `n=1000` fresh Monte Carlo data sets per cell "
            "(repetitions 100--1099)."
        )
    else:
        sample_note = (
            "Each entry is `rejection probability [95% exact confidence "
            "interval]`; 1,000 fresh Monte Carlo data sets were attempted "
            "per cell, and cells with failures display their valid `n`."
        )
    lines = [
        sample_note,
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for scenario in EXPECTED_SCENARIOS:
        frame = (
            summary.loc[summary["scenario"].eq(scenario)]
            .set_index("method_key")
            .reindex(EXPECTED_METHODS)
        )
        row = [labels[scenario]] + frame["estimate_and_95_ci"].tolist()
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "Only `rho=0, X→Y` is a null-size row. All methods use the same "
            "five-fold cross-fitted kernel-ridge residual vector within each "
            "Monte Carlo data set and direction. The quantile test uses "
            "`tau={0.25, 0.50, 0.75}` with a Bonferroni joint p-value. In "
            "method titles, `X` denotes the generic stage-two predictor or "
            "covariate; in the `Y→X` rows, that predictor is the original "
            "`Y` variable.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_latex(summary: pd.DataFrame) -> str:
    scenario_labels = {
        "rho=0, X->Y (size)": r"$\rho=0$, $X\to Y$ (size)",
        "rho=0, Y->X (power)": r"$\rho=0$, $Y\to X$ (power)",
        "rho=0.35, X->Y (power)": r"$\rho=0.35$, $X\to Y$ (power)",
        "rho=0.35, Y->X (power)": r"$\rho=0.35$, $Y\to X$ (power)",
    }
    lines = [
        r"% Each cell is estimate [95% exact Clopper--Pearson interval].",
        r"% Fresh Monte Carlo repetitions 100--1099; nominal level 5\%.",
        r"\begin{tabular}{l" + "c" * len(EXPECTED_METHODS) + "}",
        "Regime & "
        + " & ".join(LATEX_TITLES[key] for key in EXPECTED_METHODS)
        + r" \\",
        r"\hline",
    ]
    for scenario in EXPECTED_SCENARIOS:
        frame = (
            summary.loc[summary["scenario"].eq(scenario)]
            .set_index("method_key")
            .reindex(EXPECTED_METHODS)
        )
        lines.append(
            scenario_labels[scenario]
            + " & "
            + " & ".join(frame["estimate_and_95_ci"].tolist())
            + r" \\"
        )
    lines.extend(
        [
            r"\end{tabular}",
            (
                r"% Only the first row is a null-size experiment. All "
                r"methods use the same five-fold cross-fitted kernel-ridge "
                r"residuals within each data set and direction."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_local_and_canonical(
    raw: pd.DataFrame,
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
    wide: pd.DataFrame,
    publication_audit: pd.DataFrame,
    calibration_audit: pd.DataFrame,
    markdown: str,
    latex: str,
) -> dict[str, dict[str, object]]:
    preserved = preserve_n100_canonical_artifacts()

    csv_outputs = {
        COMBINED_RAW_CSV: raw,
        INPUT_INVENTORY_CSV: inventory,
        LOCAL_SUMMARY_CSV: summary,
        LOCAL_WIDE_CSV: wide,
        LOCAL_PUBLICATION_AUDIT_CSV: publication_audit,
        LOCAL_CALIBRATION_AUDIT_CSV: calibration_audit,
        CANONICAL_SUMMARY_CSV: summary,
        CANONICAL_WIDE_CSV: wide,
        CANONICAL_PUBLICATION_AUDIT_CSV: publication_audit,
        CANONICAL_CALIBRATION_AUDIT_CSV: calibration_audit,
    }
    text_outputs = {
        LOCAL_MARKDOWN: markdown,
        LOCAL_LATEX: latex,
        CANONICAL_MARKDOWN: markdown,
        CANONICAL_LATEX: latex,
    }
    for path, frame in csv_outputs.items():
        atomic_write_csv(frame, path)
    for path, text in text_outputs.items():
        atomic_write_text(text, path)

    outputs: dict[str, dict[str, object]] = {}
    for path in (*csv_outputs, *text_outputs):
        outputs[str(path.relative_to(ROOT))] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": (
                int(len(csv_outputs[path]))
                if path in csv_outputs
                else None
            ),
        }
    return {
        "outputs": outputs,
        "preserved_n100_selection_artifacts": preserved,
    }


def build_manifest(
    raw: pd.DataFrame,
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
    calibration_audit: pd.DataFrame,
    write_metadata: dict[str, dict[str, object]],
) -> dict:
    inventory_digest_payload = "".join(
        f"{row.repetition}\t{row.path}\t{row.sha256}\t{row.bytes}\n"
        for row in inventory.sort_values("repetition").itertuples(index=False)
    )
    gross = calibration_audit.loc[
        calibration_audit["gross_departure_flag"]
    ]
    invalid = summary.loc[summary["n_invalid"] > 0]
    runner_source = ROOT / "run_toy_additive_noise_selected_1000.py"
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Fresh 1000-repetition estimates for only the eight procedures "
            "retained in the final selected-test table."
        ),
        "simulation_batch": {
            "repetition_start_inclusive": runner.HOLDOUT_START,
            "repetition_stop_exclusive": runner.HOLDOUT_STOP,
            "repetitions": len(EXPECTED_REPETITIONS),
            "scenarios": list(EXPECTED_SCENARIOS),
            "methods": list(EXPECTED_METHODS),
            "expected_raw_rows": EXPECTED_ROWS,
            "observed_raw_rows": len(raw),
            "unique_raw_keys": int(
                raw[
                    ["repetition", "scenario", "method_key"]
                ].drop_duplicates().shape[0]
            ),
            "valid_test_rows": int(raw["valid"].eq(True).sum()),
            "invalid_test_rows": int(raw["valid"].ne(True).sum()),
            "invalid_cells": invalid[
                ["scenario", "method_key", "n_invalid"]
            ].to_dict(orient="records"),
        },
        "fresh_holdout_and_selection_notes": {
            "selection_stage": (
                "The final eight methods/configurations were chosen using "
                "the earlier repetitions 0--99 (n=100 per cell)."
            ),
            "estimation_stage": (
                "All estimates in this manifest use only fresh repetitions "
                "100--1099, generated after the table specification was "
                "fixed."
            ),
            "interpretation": (
                "Because the new Monte Carlo data sets are independent of "
                "the selection batch, selection does not reuse the outcomes "
                "entering these estimates. The intervals are nevertheless "
                "pointwise across table cells, not simultaneous confidence "
                "sets over methods and regimes."
            ),
            "binned_test_note": (
                "B=10 was retained based on the earlier n=100 descriptive "
                "comparison; it was not re-selected on repetitions 100--1099."
            ),
            "title_notation": (
                "X in method titles denotes the generic stage-two predictor "
                "or covariate. In Y->X rows, this stage-two predictor is the "
                "original Y variable."
            ),
        },
        "shared_first_stage_validation": {
            "description": (
                "Within each repetition and scenario, all eight methods must "
                "have identical SHA-256 values for both the common OOF "
                "residual vector and the full fold-residual tensor."
            ),
            "groups_checked": (
                len(EXPECTED_REPETITIONS) * len(EXPECTED_SCENARIOS)
            ),
            "methods_per_group": len(EXPECTED_METHODS),
            "all_groups_passed": True,
            "first_stage": {
                "model": "Gaussian kernel ridge regression",
                "outer_crossfit_folds": int(
                    runner.CORE_CONFIG.num_folds
                ),
                "internal_cv_folds": int(
                    runner.CORE_CONFIG.mean_cv_folds
                ),
                "alphas": list(runner.CORE_CONFIG.mean_alphas),
                "length_scales": list(
                    runner.CORE_CONFIG.mean_length_scales
                ),
            },
        },
        "confidence_intervals": {
            "confidence_level": CONFIDENCE_LEVEL,
            "method": "exact Clopper-Pearson binomial",
            "estimand": (
                "Monte Carlo rejection probability at nominal level 0.05"
            ),
            "denominator": "valid repetitions in each method/scenario cell",
            "scope": "pointwise, not simultaneous",
        },
        "calibration_audit": {
            "null_scenario": NULL_SCENARIO,
            "nominal_rejection_probability": NOMINAL_LEVEL,
            "tests": (
                "Exact binomial two-sided, liberal one-sided, and "
                "conservative one-sided tests for each of eight methods."
            ),
            "multiplicity": (
                "Holm adjustment across the eight methods is reported for "
                "each p-value family."
            ),
            "gross_departure_definition": (
                "Two-sided Holm-adjusted p <= 0.05 and observed null "
                "rejection probability >= 0.10 or <= 0.025."
            ),
            "gross_departure_methods": gross[
                [
                    "method_key",
                    "column_title",
                    "rejection_probability",
                    "calibration_direction",
                    "exact_two_sided_holm_p_value",
                ]
            ].to_dict(orient="records"),
        },
        "source_hashes": {
            "runner_script": {
                "path": str(runner_source.relative_to(ROOT)),
                "sha256": sha256_file(runner_source),
            },
            "runner_notebook_source_cells": runner.SOURCE_HASHES,
            "input_file_inventory_digest_sha256": sha256_text(
                inventory_digest_payload
            ),
            "input_file_inventory_digest_encoding": (
                "UTF-8 lines: repetition, relative path, file SHA-256, bytes"
            ),
            "input_repetition_file_count": len(inventory),
        },
        "runner_configuration": runner.configuration_report(),
        "raw_runtime_provenance": {
            "jax_backends": sorted(
                raw["jax_backend"].dropna().astype(str).unique().tolist()
            ),
            "jax_device_kinds": sorted(
                raw["jax_device_kind"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
        },
        **write_metadata,
        "manifest_hash_note": (
            "The manifest does not include its own SHA-256 in order to avoid "
            "a circular self-hash."
        ),
    }


def aggregate(*, validate_only: bool = False) -> None:
    raw, inventory = load_and_validate_raw()
    raw = raw.sort_values(
        ["repetition", "scenario", "method_key"], kind="stable"
    ).reset_index(drop=True)
    summary = build_summary(raw)
    calibration_audit = build_calibration_audit(summary)
    publication_audit = build_publication_audit(raw, summary)
    wide = build_wide(summary)
    markdown = build_markdown(summary)
    latex = build_latex(summary)

    if validate_only:
        print(
            "Validated 1,000 repetition files, 32,000 unique raw rows, "
            "4,000 shared-first-stage groups, 32 summary cells, and 8 null "
            "calibration rows. No files were written."
        )
        return

    write_metadata = write_local_and_canonical(
        raw=raw,
        inventory=inventory,
        summary=summary,
        wide=wide,
        publication_audit=publication_audit,
        calibration_audit=calibration_audit,
        markdown=markdown,
        latex=latex,
    )
    manifest = build_manifest(
        raw=raw,
        inventory=inventory,
        summary=summary,
        calibration_audit=calibration_audit,
        write_metadata=write_metadata,
    )
    atomic_write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        MANIFEST_JSON,
    )
    gross = calibration_audit.loc[
        calibration_audit["gross_departure_flag"], "method_key"
    ].tolist()
    print(
        f"Wrote fresh selected-test aggregation to {RESULT_DIR}: "
        f"{len(raw):,} raw rows, {len(summary)} table cells, "
        f"{len(calibration_audit)} null calibration checks."
    )
    print(
        "Gross nominal-calibration departures: "
        + (", ".join(gross) if gross else "none")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and aggregate fresh repetitions 100--1099 for the "
            "locked eight-method selected-test table."
        )
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all inputs and construct outputs in memory without writing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(validate_only=args.validate_only)


if __name__ == "__main__":
    main()
