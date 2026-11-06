"""Generate complete-cohort tables, figures, and an HTML index automatically."""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

from run_sensitivity import FACTORS, SETTING, checkpoint_path, read_json, write_json

GROUP = [*SETTING, "factor_x", "factor_residual"]


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    """Pointwise binomial interval for Monte Carlo coverage uncertainty."""
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def flatten(row: dict) -> dict:
    result = {k: v for k, v in row.items() if k != "methods"}
    for method, values in row["methods"].items():
        for name, value in values.items():
            if name in ("rti", "delta_u", "delta_v", "delta_hsic_u"):
                for field in ("coverage", "width", "zero_inclusion"):
                    result[f"{method}_{name}_{field}"] = value[field] if value is not None else np.nan
            else:
                result[f"{method}_{name}"] = value
    return result


def aggregate(frame: pd.DataFrame, expected_repetitions: int = 100) -> pd.DataFrame:
    """One paired row per setting/bandwidth, with an explicit complete denominator."""
    rows = []
    if frame.duplicated(GROUP + ["random_key"]).any():
        raise ValueError("Duplicate paired repetitions.")
    for key, group in frame.groupby(GROUP, sort=True):
        if len(group) != expected_repetitions or group.random_key.nunique() != expected_repetitions:
            raise ValueError(f"Incomplete group {key}: {len(group)} repetitions.")
        result = dict(zip(GROUP, key))
        result["n_repetitions"] = len(group)
        for method in ("plugin", "debiased", "oracle"):
            values = group[f"{method}_operator_error_sq"]
            result[f"{method}_operator_mse"] = values.mean()
            result[f"{method}_operator_mse_mcse"] = values.std(ddof=1) / np.sqrt(len(values))
        gain = group.operator_error_sq_gain
        result["operator_mse_gain"] = gain.mean()
        result["operator_mse_gain_mcse"] = gain.std(ddof=1) / np.sqrt(len(gain))
        result["operator_mse_gain_ci_lower"] = gain.mean() - 1.95996398454 * result["operator_mse_gain_mcse"]
        result["operator_mse_gain_ci_upper"] = gain.mean() + 1.95996398454 * result["operator_mse_gain_mcse"]
        result["relative_mse_gain_percent"] = 100 * gain.mean() / result["plugin_operator_mse"]
        result["debiased_better_fraction"] = (gain > 0).mean()
        result["mean_norm_error_gain"] = group.operator_error_gain.mean()
        result["mean_population_hsic"] = group.population_hsic.mean()
        for method in ("plugin", "debiased"):
            for interval in ("rti", "delta_u", "delta_v", "delta_hsic_u"):
                prefix = f"{method}_{interval}"
                values = group[f"{prefix}_coverage"]
                if interval.startswith("delta") and result["rho"] == 0:
                    if values.notna().any():
                        raise ValueError("Null delta coverage must be marked inapplicable.")
                    result[f"{prefix}_coverage"] = np.nan
                    result[f"{prefix}_n"] = 0
                    continue
                if values.isna().any():
                    raise ValueError(f"Undefined primary interval: {key}, {prefix}")
                hits = int(values.sum())
                result[f"{prefix}_n"] = len(values)
                result[f"{prefix}_coverage"] = hits / len(values)
                result[f"{prefix}_coverage_mcse"] = np.sqrt(values.mean() * (1 - values.mean()) / len(values))
                lo, hi = wilson_interval(hits, len(values))
                result[f"{prefix}_coverage_ci_lower"] = lo
                result[f"{prefix}_coverage_ci_upper"] = hi
                result[f"{prefix}_mean_width"] = group[f"{prefix}_width"].mean()
                result[f"{prefix}_zero_inclusion"] = group[f"{prefix}_zero_inclusion"].mean()
            result[f"{method}_clipped_u_fraction"] = group[f"{method}_center_clipped"].mean()
            for kind in ("u", "v"):
                result[f"{method}_hsic_{kind}_mse"] = ((group[f"{method}_hsic_{kind}"] - group.population_hsic) ** 2).mean()
                result[f"{method}_hsic_{kind}_bias"] = (group[f"{method}_hsic_{kind}"] - group.population_hsic).mean()
            for interval in ("rti", "delta_u"):
                if interval == "delta_u" and result["rho"] == 0:
                    continue
                paired = (group[f"debiased_{interval}_coverage"].astype(float)
                          - group[f"plugin_{interval}_coverage"].astype(float))
                result[f"{interval}_coverage_gain"] = paired.mean()
                result[f"{interval}_coverage_gain_mcse"] = paired.std(ddof=1) / np.sqrt(len(paired))
        rows.append(result)
    return pd.DataFrame(rows)


def compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [*GROUP, "n_repetitions", "plugin_operator_mse", "debiased_operator_mse",
               "operator_mse_gain", "operator_mse_gain_mcse", "relative_mse_gain_percent",
               "plugin_delta_u_coverage", "debiased_delta_u_coverage",
               "plugin_rti_coverage", "debiased_rti_coverage"]
    return summary[columns].copy()


def save_baseline_figures(summary: pd.DataFrame, folder: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    selected = summary[(summary.factor_x == 1) & (summary.factor_residual == 1)]
    paths = []
    for n, group in selected.groupby("N"):
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), squeeze=False)
        for column, (smoothness, cells) in enumerate(group.groupby("eps_smoothness")):
            cells = cells.sort_values("rho")
            ax = axes[0, column]
            ax.errorbar(cells.rho, cells.operator_mse_gain,
                        yerr=1.95996398454 * cells.operator_mse_gain_mcse, marker="o", capsize=3)
            ax.axhline(0, color="gray", lw=1)
            ax.set_title(f"Noise smoothness {smoothness:g}")
            ax.set_ylabel("Plug-in MSE − debiased MSE")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            for row, interval, title in ((1, "delta_u", "Delta coverage (ρ > 0)"), (2, "rti", "RTI coverage")):
                ax = axes[row, column]
                for method, label in (("plugin", "Plug-in"), ("debiased", "Debiased")):
                    subset = cells[cells.rho != 0] if interval == "delta_u" else cells
                    key = f"{method}_{interval}_coverage"
                    y = subset[key].to_numpy(float)
                    errors = np.maximum(0., np.vstack((y - subset[key + "_ci_lower"], subset[key + "_ci_upper"] - y)))
                    ax.errorbar(subset.rho, y, yerr=errors, marker="o", capsize=2, label=label)
                ax.axhline(.95, color="gray", linestyle="--", lw=1)
                ax.set_ylim(0, 1.03)
                ax.set_ylabel(title)
                ax.set_xlabel("ρ")
                ax.legend(fontsize=8)
        fig.suptitle(f"Median length scales, n = {n}; 100 paired repetitions per setting")
        fig.tight_layout()
        stem = folder / f"median_n{n}"
        for extension in ("png", "pdf"):
            fig.savefig(stem.with_suffix("." + extension), dpi=150, bbox_inches="tight")
        paths.append(stem.with_suffix(".png"))
        plt.close(fig)
    return paths


def save_heatmaps(summary: pd.DataFrame, folder: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    paths = []
    specs = [("relative_mse_gain_percent", "Relative operator-MSE reduction (%)", "RdBu", None),
             ("operator_mse_gain", "Plug-in MSE − debiased MSE", "RdBu", None),
             ("plugin_delta_u_coverage", "Plug-in delta coverage", "viridis", (0, 1)),
             ("debiased_delta_u_coverage", "Debiased delta coverage", "viridis", (0, 1)),
             ("plugin_rti_coverage", "Plug-in RTI coverage", "viridis", (0, 1)),
             ("debiased_rti_coverage", "Debiased RTI coverage", "viridis", (0, 1))]
    for key, group in summary.groupby(list(SETTING), sort=True):
        n, mean_smoothness, smoothness, rho = key
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for ax, (metric, label, cmap, limits) in zip(axes.flat, specs):
            grid = group.pivot(index="factor_residual", columns="factor_x", values=metric).reindex(index=FACTORS, columns=FACTORS)
            if rho == 0 and "delta" in metric:
                ax.text(.5, .5, "Not applicable at ρ = 0", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(label)
                ax.set_axis_off()
                continue
            array = grid.to_numpy(float)
            if limits is None:
                bound = max(float(np.nanmax(np.abs(array))), 1e-14)
                limits = (-bound, bound)
            im = ax.imshow(array, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1])
            ax.set_xticks(range(5), [f"{x:g}" for x in FACTORS])
            ax.set_yticks(range(5), [f"{x:g}" for x in FACTORS])
            ax.set_xlabel("X length-scale factor")
            ax.set_ylabel("Residual length-scale factor")
            ax.set_title(label, fontsize=10)
            for i in range(5):
                for j in range(5):
                    value = array[i, j]
                    text = f"{value:.2f}" if limits == (0, 1) else f"{value:.2g}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=7,
                            bbox={"facecolor": "white", "alpha": .65, "edgecolor": "none", "pad": .5})
            fig.colorbar(im, ax=ax, shrink=.8)
        fig.suptitle(f"n = {n}, mean smoothness = {mean_smoothness:g}, noise smoothness = {smoothness:g}, ρ = {rho:g}\n"
                     "Positive MSE reduction favors debiasing; pointwise coverage target = 0.95")
        fig.tight_layout()
        stem = folder / f"bandwidth_n{n}_noise{smoothness:g}_rho{rho:g}"
        for extension in ("png", "pdf"):
            fig.savefig(stem.with_suffix("." + extension), dpi=150, bbox_inches="tight")
        paths.append(stem.with_suffix(".png"))
        plt.close(fig)
    return paths


def summarize(output: Path, baseline_only: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")

    name = "baseline" if baseline_only else "sensitivity"
    manifest = read_json(output / "manifest.json")
    rows, missing = [], []
    for entry in manifest["entries"]:
        kinds = ["baseline"] if baseline_only else [f"residual_{factor:g}" for factor in FACTORS]
        for kind in kinds:
            path = checkpoint_path(output, entry, kind)
            if not path.exists():
                missing.append({"run_name": entry["run_name"], "checkpoint": kind})
                continue
            saved = read_json(path)
            if saved.get("replay_verified") is not True:
                raise ValueError(f"Unverified checkpoint: {path}")
            records = [saved["row"]] if baseline_only else saved["rows"]
            expected_pairs = {(1., 1.)} if baseline_only else {(x, float(kind.split("_")[1])) for x in FACTORS}
            pairs = {(row["factor_x"], row["factor_residual"]) for row in records}
            if pairs != expected_pairs or len(records) != len(expected_pairs):
                raise ValueError(f"Wrong bandwidth grid in {path}")
            for row in records:
                if any(row[k] != entry["config"][k] for k in (*SETTING, "random_key")):
                    raise ValueError(f"Mismatched checkpoint setting: {path}")
            rows.extend(flatten(row) for row in records)
    expected = len(manifest["entries"]) * (1 if baseline_only else 25)
    status = {"complete": not missing and len(rows) == expected, "expected_rows": expected,
              "completed_rows": len(rows), "missing_checkpoints": missing}
    write_json(output / f"status_{name}.json", status)
    report_path = output / f"report_{name}.html"
    if not status["complete"]:
        report_path.write_text(f"<!doctype html><meta charset='utf-8'><h1>Incomplete {name} results</h1>"
                              f"<p>{len(rows)} of {expected} paired rows are available. No complete-cohort estimates are reported.</p>"
                              f"<p>See <a href='status_{name}.json'>missing checkpoints</a> and checkpoints/failures_*.</p>")
        raise RuntimeError(f"Incomplete cohort: {len(missing)} missing checkpoints; see {report_path}")
    frame = pd.DataFrame(rows)
    summary = aggregate(frame)
    tables, figures = output / "tables" / name, output / "figures" / name
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "checkpoints" / f"repetition_metrics_{name}.csv.gz", index=False)
    summary.to_csv(tables / "all_metrics.csv", index=False)
    compact = compact_table(summary)
    compact.to_csv(tables / "mse_and_coverage.csv", index=False)
    compact.to_latex(tables / "mse_and_coverage.tex", index=False, float_format="%.5g", na_rep="--")
    median = summary[(summary.factor_x == 1) & (summary.factor_residual == 1)]
    compact_table(median).to_csv(tables / "median_bandwidth.csv", index=False)
    for key, group in compact.groupby(list(SETTING)):
        n, _, smoothness, rho = key
        group.to_csv(tables / f"n{n}_noise{smoothness:g}_rho{rho:g}.csv", index=False)
    images = save_baseline_figures(summary, figures)
    if not baseline_only:
        images.extend(save_heatmaps(summary, figures))
    content = ["<!doctype html><html><head><meta charset='utf-8'><title>Operator MSE and coverage</title>",
               "<style>body{font-family:system-ui;margin:2em;max-width:1500px}table{border-collapse:collapse;font-size:12px}"
               "td,th{padding:6px;border:1px solid #ddd}img{max-width:100%}.table{overflow:auto}summary{cursor:pointer}</style></head><body>",
               f"<h1>{name.capitalize()}: operator MSE and confidence-interval coverage</h1>",
               f"<p>Complete: {len(frame):,} paired bandwidth evaluations; 100 independent seeded repetitions per setting. "
               "Mean smoothness 1; n = 250, 500, 1000; noise smoothness 0.25, 0.5, 0.75; ρ = 0, 0.1, …, 0.5.</p>",
               "<p>MSE is mean squared Hilbert–Schmidt error of the cross-covariance operator. "
               "Gain = plug-in MSE − debiased MSE: positive favors debiasing. Errors compare against the population "
               "operator at each repetition's selected and scaled bandwidths. These targets and RKHSs change across bandwidths; "
               "raw MSE alone should not be used to select a bandwidth.</p>",
               "<p>Coverage means inclusion of the true population √HSIC, not inclusion of zero. "
               "The primary delta CI has center √max(U,0) and variance score_var/(nV); RTI is V-centered. "
               "Delta coverage is inapplicable at ρ=0. Full CSVs also contain V-centered delta and HSIC-scale delta diagnostics, "
               "interval widths, Wilson coverage intervals, paired gain uncertainty, oracle-residual MSE and scalar HSIC errors. "
               "Monte Carlo intervals are pointwise and do not adjust for examining the grid.</p>",
               f"<p><a href='tables/{name}/mse_and_coverage.csv'>Combined MSE/coverage table</a> · "
               f"<a href='tables/{name}/all_metrics.csv'>All metrics and uncertainty</a> · "
               f"<a href='tables/{name}/median_bandwidth.csv'>Median bandwidth table</a> · "
               "<a href='code_snapshot/README.md'>Methods and reproduction</a></p>",
               "<h2>Median bandwidths</h2><div class='table'>",
               compact_table(median).to_html(index=False, float_format=lambda x: f"{x:.5g}", na_rep="—"), "</div>"]
    for image_path in images:
        relative = image_path.relative_to(output)
        content.append(f"<details><summary>{html.escape(image_path.stem)}</summary><img src='{relative}'></details>")
    if not baseline_only:
        content.extend(["<h2>Every bandwidth pair</h2><div class='table'>",
                        compact.to_html(index=False, float_format=lambda x: f"{x:.5g}", na_rep="—"), "</div>"])
    content.append("</body></html>")
    report_path.write_text("\n".join(content))
    print(f"Wrote {report_path}; {len(summary)} complete summary rows.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    summarize(args.output.resolve(), args.baseline_only)
