from __future__ import annotations

from collections.abc import Callable, Iterable
import math
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Z975 = 1.959963984540054

METHOD_ORDER = [
    "bootstrap_naive",
    "bootstrap_naive_u_centered",
    "bootstrap_debiased",
    "bootstrap_debiased_u_centered",
    "delta_naive_v",
    "delta_naive_v_loo",
    "delta_naive_u",
    "delta_naive_u_loo",
    "delta_debiased_v",
    "delta_debiased_v_loo",
    "delta_debiased_u",
    "delta_debiased_u_loo",
]
METHOD_LABELS = {method: method.replace("_", " ") for method in METHOD_ORDER}
METHOD_LABELS.update(
    {
        "bootstrap_naive_u_centered": "bootstrap naive U-centered",
        "bootstrap_debiased_u_centered": "bootstrap debiased U-centered",
    }
)


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace(".", "p")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "value"


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def ordered_methods(methods: pd.Series | list[str]) -> list[str]:
    seen = list(pd.Series(methods).dropna().unique())
    seen_set = set(seen)
    return [method for method in METHOD_ORDER if method in seen_set] + sorted(seen_set - set(METHOD_ORDER))


def ci_specs_from_frame(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    specs = []
    prefixes = sorted({col.rsplit(".", 1)[0] for col in df.columns if col.startswith("intervals.")})
    for prefix in prefixes:
        if f"{prefix}.ci_lower" in df.columns and f"{prefix}.ci_upper" in df.columns:
            specs.append((prefix.split("intervals.", 1)[1], f"{prefix}.ci_lower", f"{prefix}.ci_upper"))
        elif f"{prefix}.sqrt_lower" in df.columns and f"{prefix}.sqrt_upper" in df.columns:
            specs.append((prefix.split("intervals.", 1)[1], f"{prefix}.sqrt_lower", f"{prefix}.sqrt_upper"))
    return specs



def reconstruct_robust_delta_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Invert U-centered squared-HSIC Wald intervals onto the norm scale.

    The squared-scale standard error is 2*sqrt(score_var/n). Intersect the
    interval with [0, infinity) before taking square roots. An empty set uses
    reversed endpoints (1, 0), giving zero width and false coverage without
    dropping its repetition. Missing or invalid inputs fail explicitly.
    """
    out = df.copy()
    for estimator in ("naive", "debiased"):
        for suffix in ("u", "u_loo"):
            prefix = f"intervals.delta_{estimator}_{suffix}"
            columns = [f"estimators.{estimator}_hsic_u", f"{prefix}.score_var", f"{prefix}.n"]
            missing = [column for column in columns if column not in out]
            if missing:
                raise ValueError(f"Squared delta reconstruction needs columns: {missing}")
            values = out[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            u, variance, n = values.T
            valid = np.isfinite(values).all(axis=1) & (variance >= 0) & (n > 0)
            if not valid.all():
                raise ValueError(f"{prefix}: {int((~valid).sum())} rows have invalid U/variance/n; no rows were discarded")
            se_hsic = 2.0 * np.sqrt(variance / n)
            half_width = Z975 * se_hsic
            lower, upper = u - half_width, u + half_width
            if not (np.isfinite(lower) & np.isfinite(upper)).all():
                raise ValueError(f"{prefix}: nonfinite squared interval; no rows were discarded")
            nonempty = upper >= 0.0
            out[f"{prefix}.se_hsic"] = se_hsic
            out[f"{prefix}.hsic_lower"] = np.maximum(lower, 0.0)
            out[f"{prefix}.hsic_upper"] = upper
            out[f"{prefix}.empty"] = ~nonempty
            out[f"{prefix}.sqrt_protocol"] = "squared_wald_then_sqrt"
            out[f"{prefix}.sqrt_lower"] = np.where(nonempty, np.sqrt(np.maximum(lower, 0.0)), 1.0)
            out[f"{prefix}.sqrt_upper"] = np.where(nonempty, np.sqrt(np.maximum(upper, 0.0)), 0.0)
            out[f"{prefix}.sqrt_width"] = np.maximum(out[f"{prefix}.sqrt_upper"] - out[f"{prefix}.sqrt_lower"], 0.0)
    return out

def build_ci_frame(df: pd.DataFrame) -> pd.DataFrame:
    truth_col = "population_oracle.sqrt_hsic"
    if df.empty or truth_col not in df.columns:
        return pd.DataFrame()

    id_cols = [
        col
        for col in [
            "run_name",
            "N",
            "f_smoothness",
            "eps_smoothness",
            "rho",
            "dimx",
            "dimy",
            "random_key",
            truth_col,
        ]
        if col in df.columns
    ]

    def finalize_ci_block(block: pd.DataFrame) -> pd.DataFrame:
        block = block.dropna(subset=[truth_col, "lower", "upper"]).copy()
        block["coverage"] = (block["lower"] <= block[truth_col]) & (block[truth_col] <= block["upper"])
        block["contains_zero"] = (block["lower"] <= 0.0) & (0.0 <= block["upper"])
        block["rejects_zero"] = ~block["contains_zero"]
        block["width"] = np.maximum(block["upper"] - block["lower"], 0.0)
        return block

    frames = []
    for method, lower_col, upper_col in ci_specs_from_frame(df):
        block = df[id_cols].copy()
        block[truth_col] = pd.to_numeric(block[truth_col], errors="coerce")
        block["method"] = method
        block["method_label"] = method_label(method)
        block["lower"] = pd.to_numeric(df[lower_col], errors="coerce")
        block["upper"] = pd.to_numeric(df[upper_col], errors="coerce")
        block = finalize_ci_block(block)
        frames.append(block)

    u_centered_specs = [
        ("bootstrap_naive_u_centered", "bootstrap_naive", "estimators.naive_hsic_u"),
        ("bootstrap_debiased_u_centered", "bootstrap_debiased", "estimators.debiased_hsic_u"),
    ]
    for method, bootstrap_method, center_col in u_centered_specs:
        q_col = f"intervals.{bootstrap_method}.q_val"
        radius_col = f"intervals.{bootstrap_method}.radius"
        if center_col not in df.columns or (q_col not in df.columns and radius_col not in df.columns):
            continue

        raw_center = pd.to_numeric(df[center_col], errors="coerce")
        if q_col in df.columns:
            raw_half_width = pd.to_numeric(df[q_col], errors="coerce")
        else:
            raw_half_width = pd.to_numeric(df[radius_col], errors="coerce") ** 2
        raw_half_width = np.maximum(raw_half_width, 0.0)

        raw_lower = raw_center - raw_half_width
        raw_upper = raw_center + raw_half_width
        valid = raw_center.notna() & pd.Series(raw_half_width, index=df.index).notna()
        nonempty = valid & (raw_upper >= 0.0)

        sqrt_lower = np.where(
            valid,
            np.where(nonempty, np.sqrt(np.maximum(raw_lower, 0.0)), 1.0),
            np.nan,
        )
        sqrt_upper = np.where(
            valid,
            np.where(nonempty, np.sqrt(np.maximum(raw_upper, 0.0)), 0.0),
            np.nan,
        )

        block = df[id_cols].copy()
        block[truth_col] = pd.to_numeric(block[truth_col], errors="coerce")
        block["method"] = method
        block["method_label"] = method_label(method)
        block["lower"] = sqrt_lower
        block["upper"] = sqrt_upper
        block = finalize_ci_block(block)
        frames.append(block)

    if not frames:
        return pd.DataFrame()

    ci = pd.concat(frames, ignore_index=True)
    order = ordered_methods(ci["method"])
    ci["method"] = pd.Categorical(ci["method"], categories=order, ordered=True)
    ci["method_label"] = pd.Categorical(
        ci["method_label"],
        categories=[method_label(method) for method in order],
        ordered=True,
    )
    sort_cols = [
        col
        for col in ["method", "N", "f_smoothness", "eps_smoothness", "rho", "dimx", "dimy", "random_key"]
        if col in ci.columns
    ]
    return ci.sort_values(sort_cols).reset_index(drop=True)


def bernoulli_interval(successes: int, total: int, z: float = Z975) -> tuple[float, float, float]:
    if total <= 0:
        return np.nan, np.nan, np.nan

    p_hat = successes / total
    se = math.sqrt(max(p_hat * (1.0 - p_hat) / total, 0.0))
    lower = max(0.0, p_hat - z * se)
    upper = min(1.0, p_hat + z * se)
    return p_hat, lower, upper


def summarize_binary(df: pd.DataFrame, group_cols: list[str], value_col: str, summary_col: str) -> pd.DataFrame:
    rows = []
    for key, block in df.groupby(group_cols, dropna=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        value, lower, upper = bernoulli_interval(int(block[value_col].sum()), len(block))
        row = dict(zip(group_cols, key))
        row["runs"] = len(block)
        row[summary_col] = value
        row[f"{summary_col}_lower"] = lower
        row[f"{summary_col}_upper"] = upper
        row[f"{summary_col}_err_low"] = value - lower if pd.notna(value) else np.nan
        row[f"{summary_col}_err_high"] = upper - value if pd.notna(value) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def figure_variant_suffix(*, include_legend: bool, include_x_axis: bool, include_y_axis: bool) -> str:
    parts = [
        "legend" if include_legend else "no_legend",
        "xaxis" if include_x_axis else "no_xaxis",
        "yaxis" if include_y_axis else "no_yaxis",
    ]
    return "_".join(parts)


def iter_figure_variants() -> list[dict[str, bool]]:
    variants = []
    for include_legend in [True, False]:
        for include_x_axis in [True, False]:
            for include_y_axis in [True, False]:
                variants.append(
                    {
                        "include_legend": include_legend,
                        "include_x_axis": include_x_axis,
                        "include_y_axis": include_y_axis,
                    }
                )
    return variants


def apply_axis_visibility(ax, *, include_x_axis: bool, include_y_axis: bool) -> None:
    if not include_x_axis:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    else:
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)

    if not include_y_axis:
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    else:
        ax.tick_params(axis="y", which="both", left=True, labelleft=True)


def remove_legends(fig) -> None:
    for legend in list(fig.legends):
        legend.remove()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def save_figure_variants(
    plot_factory: Callable[..., Any],
    *,
    output_dir: str | Path,
    stem: str,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for variant in iter_figure_variants():
        fig = plot_factory(**variant)
        suffix = figure_variant_suffix(**variant)
        for fmt in formats:
            path = output_dir / f"{stem}_{suffix}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
            saved_paths.append(path)
        plt.close(fig)

    return saved_paths
