"""Build appendix tables and bandwidth figures from the completed paired runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from summarize import wilson_interval

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "MSE_sensitivity"
OUTPUT = DATA / "appendix"
TABLES = ROOT / "tables/synthetic/debiasing_mse"
FIGURES = ROOT / "figures/synthetic/debiasing_mse"
KEYS = ["N", "f_smoothness", "eps_smoothness", "rho", "factor_x", "factor_residual"]
FACTORS = (.1, .5, 1., 2., 5.)
SMOOTHNESS = .5
FIGURE_N = 500
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261101


def paired_mse_summary(plugin: np.ndarray, debiased: np.ndarray,
                       weights: np.ndarray) -> dict[str, float]:
    """Bootstrap paired repetitions; relative gain is a ratio of mean losses."""
    plugin, debiased = np.asarray(plugin), np.asarray(debiased)
    if (plugin.ndim != 1 or plugin.shape != debiased.shape or len(plugin) < 2
            or weights.shape[1] != len(plugin)
            or not np.isfinite([plugin, debiased]).all()
            or min(plugin.min(), debiased.min()) < 0 or plugin.mean() <= 0):
        raise ValueError("Expected paired finite nonnegative losses and a positive plug-in MSE.")
    means = np.array([plugin.mean(), debiased.mean()])
    bootstrap = weights @ np.column_stack((plugin, debiased))
    if np.any(bootstrap[:, 0] <= 0):
        raise ValueError("A resampled plug-in MSE is zero; relative improvement is undefined.")
    values = {"plugin_mse": means[0], "debiased_mse": means[1],
              "gain_percent": 100 * (1 - means[1] / means[0]),
              "gain_absolute": means[0] - means[1]}
    draws = {"plugin_mse": bootstrap[:, 0], "debiased_mse": bootstrap[:, 1],
             "gain_percent": 100 * (1 - bootstrap[:, 1] / bootstrap[:, 0]),
             "gain_absolute": bootstrap[:, 0] - bootstrap[:, 1]}
    for name, samples in draws.items():
        values[name + "_lower"], values[name + "_upper"] = np.quantile(samples, [.025, .975])
    return values


def rti_interval_from_width(center_sq: np.ndarray, width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover the standard RTI endpoints on the sqrt(HSIC) scale."""
    lower = np.maximum(np.sqrt(center_sq) - width / 2, 0.)
    return lower, lower + width


def load_repetitions() -> pd.DataFrame:
    status = json.loads((DATA / "status_sensitivity.json").read_text())
    if not status["complete"] or status["completed_rows"] != 135000:
        raise ValueError("The complete 135000-row sensitivity cohort is required.")
    columns = KEYS + ["random_key", "population_hsic"]
    for method in ("plugin", "debiased"):
        columns.extend(f"{method}_{name}" for name in
                       ("operator_error_sq", "hsic_u", "hsic_v", "rti_coverage", "rti_width"))
    frame = pd.read_csv(DATA / "checkpoints/repetition_metrics_sensitivity.csv.gz", usecols=columns)
    if len(frame) != 135000 or frame.duplicated(KEYS + ["random_key"]).any():
        raise ValueError("Wrong cohort size or duplicate paired observations.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite scientific values.")
    expected = pd.MultiIndex.from_product(
        [[250, 500, 1000], [1.], [.25, .5, .75], [0., .1, .2, .3, .4, .5], FACTORS, FACTORS], names=KEYS)
    counts = frame.groupby(KEYS).size().reindex(expected)
    if len(counts) != 1350 or not counts.eq(100).all():
        raise ValueError("Each of the 1350 configurations must contain 100 repetitions.")
    for method in ("plugin", "debiased"):
        lo, hi = rti_interval_from_width(frame[f"{method}_hsic_v"].to_numpy(),
                                         frame[f"{method}_rti_width"].to_numpy())
        target = np.sqrt(frame.population_hsic)
        covers = (lo <= target) & (target <= hi)
        np.testing.assert_array_equal(covers, frame[f"{method}_rti_coverage"].astype(bool))
        # Preserve signed U estimates: negative estimates must not be clipped.
        frame[f"{method}_hsic_loss"] = (frame[f"{method}_hsic_u"] - frame.population_hsic) ** 2
    return frame


def summarize_repetitions(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(100, np.full(100, .01), size=BOOTSTRAP_DRAWS).astype(float) / 100
    records = []
    for key, group in frame.groupby(KEYS, sort=True):
        group = group.sort_values("random_key")
        np.testing.assert_array_equal(group.random_key, np.arange(10, 110))
        record = dict(zip(KEYS, key))
        record["repetitions"] = 100
        for metric, loss in (("operator", "operator_error_sq"), ("hsic", "hsic_loss")):
            estimates = paired_mse_summary(group[f"plugin_{loss}"].to_numpy(),
                                            group[f"debiased_{loss}"].to_numpy(), weights)
            record.update({metric + "_" + name: value for name, value in estimates.items()})
            for method in ("plugin", "debiased"):
                values = group[f"{method}_{loss}"].to_numpy()
                record[f"{metric}_{method}_largest_loss_share"] = values.max() / values.sum()
        for method in ("plugin", "debiased"):
            hits = int(group[f"{method}_rti_coverage"].sum())
            record[f"{method}_rti_coverage"] = hits / 100
            lo, hi = wilson_interval(hits, 100)
            record[f"{method}_rti_coverage_lower"] = lo
            record[f"{method}_rti_coverage_upper"] = hi
        records.append(record)
    result = pd.DataFrame(records)
    old = pd.read_csv(DATA / "tables/sensitivity/all_metrics.csv").set_index(KEYS).sort_index()
    current = result.set_index(KEYS).sort_index()
    for method in ("plugin", "debiased"):
        for metric, old_name in (("operator", f"{method}_operator_mse"), ("hsic", f"{method}_hsic_u_mse")):
            np.testing.assert_allclose(current[f"{metric}_{method}_mse"], old[old_name], rtol=2e-9, atol=1e-15)
        np.testing.assert_array_equal(current[f"{method}_rti_coverage"], old[f"{method}_rti_coverage"])
    return result


def slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def tex_cell(point: float, lower: float, upper: float, scale: float = 1.) -> str:
    return (r"\shortstack{$" + f"{point * scale:.2f}" + r"$\\{\scriptsize $["
            + f"{lower * scale:.2f},{upper * scale:.2f}" + r"]$}}")



def save_combined_median_table(median: pd.DataFrame) -> None:
    """Write the six-panel table included by the NeurIPS manuscript."""
    expected = {(n, round(rho, 1)) for n in (250, 500, 1000)
                for rho in np.arange(0., .6, .1)}
    actual = {(int(row.N), round(float(row.rho), 1)) for row in median.itertuples()}
    if len(median) != 18 or actual != expected or not (median.repetitions == 100).all():
        raise ValueError("The combined median table requires all 18 settings with 100 repetitions.")
    lines = [
        r"\begin{table}[p]",
        r"\blue{",
        r"\centering\small",
        r"\captionsetup{labelfont={color=blue}}",
        r"\caption{\red{MSE of the operator estimate and HSIC $U$-statistic, and coverage of nominal $95\%$ RTI intervals for $\sqrt{\mathrm{HSIC}}$, at median bandwidths, $s_m=1$, and $s_\epsilon=0.5$. Entries are estimates with $95\%$ Monte Carlo intervals from 100 paired repetitions.}}",
        r"\label{tab:debiasing-mse-median}",
        r"\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{@{}rcccccc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\red{\shortstack{Operator MSE\\$(\times10^4)$}}}",
        r"& \multicolumn{2}{c}{\red{\shortstack{HSIC $U$-statistic MSE\\$(\times10^7)$}}}",
        r"& \multicolumn{2}{c}{\red{\shortstack{RTI coverage\\(\%)}}}\\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r"$n$ & Plug-in & Debiased & Plug-in & Debiased & Plug-in & Debiased\\",
    ]
    for panel, (rho, group) in enumerate(median.groupby("rho", sort=True)):
        lines.extend([r"\midrule",
                      rf"\multicolumn{{7}}{{c}}{{\textbf{{({chr(97 + panel)})}} $\rho={rho:g}$}}\\[2pt]"])
        for _, row in group.sort_values("N").iterrows():
            cells = [str(int(row.N))]
            for metric, scale in (("operator", 1e4), ("hsic", 1e7)):
                for method in ("plugin", "debiased"):
                    key = f"{metric}_{method}_mse"
                    cells.append(tex_cell(row[key], row[key + "_lower"], row[key + "_upper"], scale))
            for method in ("plugin", "debiased"):
                key = f"{method}_rti_coverage"
                cells.append(tex_cell(row[key], row[key + "_lower"], row[key + "_upper"], 100.))
            lines.append(" & ".join(cells) + r"\\[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}\par", "}", r"\end{table}"])
    (TABLES / "median_results.tex").write_text("\n".join(lines) + "\n")


def save_median_tables(summary: pd.DataFrame) -> None:
    median = summary[(summary.eps_smoothness == SMOOTHNESS) & (summary.factor_x == 1) & (summary.factor_residual == 1)]
    median.to_csv(TABLES / "median_summary.csv", index=False)
    save_combined_median_table(median)
    for rho, group in median.groupby("rho", sort=True):
        lines = [r"\begin{table}[htbp]", r"\centering\small",
                 rf"\caption{{Operator and HSIC U-statistic MSE, and 95\% RTI coverage for $\sqrt{{\mathrm{{HSIC}}}}$, at $\rho={rho:g}$, $s_m=1$, $s_\sigma=0.5$, and median bandwidths. Each cell gives an estimate and a 95\% Monte Carlo interval from 100 paired repetitions.}}",
                 rf"\label{{tab:debiasing-mse-rho{slug(rho)}}}",
                 r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.3}",
                 r"\begin{tabular}{@{}rcccccc@{}}\toprule",
                 r"& \multicolumn{2}{c}{Operator MSE ($\times 10^{4}$)} & \multicolumn{2}{c}{HSIC U MSE ($\times 10^{7}$)} & \multicolumn{2}{c}{RTI coverage (\%)}\\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
                 r"$N$ & Plug-in & Debiased & Plug-in & Debiased & Plug-in & Debiased\\\midrule"]
        for _, row in group.sort_values("N").iterrows():
            cells = [str(int(row.N))]
            for metric, scale in (("operator", 1e4), ("hsic", 1e7)):
                for method in ("plugin", "debiased"):
                    key = f"{metric}_{method}_mse"
                    cells.append(tex_cell(row[key], row[key + "_lower"], row[key + "_upper"], scale))
            for method in ("plugin", "debiased"):
                key = f"{method}_rti_coverage"
                cells.append(tex_cell(row[key], row[key + "_lower"], row[key + "_upper"], 100.))
            lines.append(" & ".join(cells) + r"\\[3pt]")
        lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
        (TABLES / f"median_rho{slug(rho)}.tex").write_text("\n".join(lines) + "\n")


def compact_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.1e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    return f"{value:.1f}" if abs(value) < 100 else f"{value:.0f}"


def save_bandwidth_figures(summary: pd.DataFrame) -> None:
    # The published heatmaps use Liberation Sans; a shared Matplotlib cache can
    # otherwise silently substitute DejaVu Sans and change their tight bounds.
    for font in font_manager.findSystemFonts():
        if Path(font).name in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
            font_manager.fontManager.addfont(font)
    for weight in ("normal", "bold"):
        font_manager.findfont(font_manager.FontProperties(family="Liberation Sans", weight=weight),
                              fallback_to_default=False)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "dejavusans",
                         "figure.dpi": 140, "savefig.dpi": 300, "axes.titleweight": "semibold",
                         "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    selected = summary[(summary.N == FIGURE_N) & (summary.eps_smoothness == SMOOTHNESS)]
    # Keep a common scale across rho for each loss, with zero at the neutral color.
    norms = {}
    for metric, step in (("operator", 10), ("hsic", 100)):
        lower = step * np.floor(selected[metric + "_gain_percent"].min() / step)
        norms[metric] = TwoSlopeNorm(vmin=min(lower, -step), vcenter=0, vmax=100)
    cmap = plt.get_cmap("RdBu")
    figure_tex = []
    for rho, group in selected.groupby("rho", sort=True):
        fig, axes = plt.subplots(2, 1, figsize=(7.6, 9.2))
        fig.subplots_adjust(left=.15, right=.82, bottom=.14, top=.90, hspace=.43)
        for ax, metric, title in zip(axes, ("operator", "hsic"),
                                    (r"Operator MSE: $\mathbb{E}\|\widehat\Psi-\Psi\|_{\mathrm{HS}}^2$",
                                     r"HSIC U-statistic MSE: $\mathbb{E}(U-\theta)^2$")):
            norm = norms[metric]
            indexed = group.set_index(["factor_residual", "factor_x"])
            grid = group.pivot(index="factor_residual", columns="factor_x", values=metric + "_gain_percent")
            grid = grid.reindex(index=FACTORS, columns=FACTORS)
            im = ax.imshow(grid, origin="lower", aspect="auto", norm=norm, cmap=cmap)
            ax.grid(False)
            ax.set_xticks(range(5), [f"{x:g}" for x in FACTORS], fontsize=10)
            ax.set_yticks(range(5), [f"{x:g}" for x in FACTORS], fontsize=10)
            ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=.8)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_xlabel(r"Covariate length-scale factor $c_X$", fontsize=11)
            ax.set_ylabel(r"Residual length-scale factor $c_\varepsilon$", fontsize=11)
            ax.set_title(title, fontsize=12, pad=9)
            for i, factor_r in enumerate(FACTORS):
                for j, factor_x in enumerate(FACTORS):
                    row = indexed.loc[(factor_r, factor_x)]
                    key = metric + "_gain_percent"
                    value, lo, hi = row[key], row[key + "_lower"], row[key + "_upper"]
                    red, green, blue, _ = cmap(norm(value))
                    luminance = .2126 * red + .7152 * green + .0722 * blue
                    color = "white" if luminance < .5 else "#171717"
                    ax.text(j, i + .13, compact_number(value), ha="center", va="center", color=color,
                            fontsize=10, fontweight="semibold")
                    ax.text(j, i - .19, f"[{compact_number(lo)}, {compact_number(hi)}]",
                            ha="center", va="center", color=color, fontsize=7.5)
            ax.add_patch(Rectangle((1.5, 1.5), 1, 1, fill=False, edgecolor="black", linewidth=1.8))
            position = ax.get_position()
            color_axis = fig.add_axes([.865, position.y0, .025, position.height])
            bar = fig.colorbar(im, cax=color_axis, ticks=[norm.vmin, norm.vmin / 2, 0, 50, 100])
            bar.set_label("MSE reduction (%)", fontsize=10)
        fig.suptitle(rf"$N=500,\ s_m=1,\ s_\sigma=0.5,\ \rho={rho:g}$", y=.97, fontsize=14, fontweight="semibold")
        fig.text(.5, .068, "At each bandwidth pair: debiased MSE compared with plug-in MSE.",
                 ha="center", fontsize=8)
        fig.text(.5, .043, "Blue favors debiasing; red favors plug-in. Brackets: 95% Monte Carlo intervals.",
                 ha="center", fontsize=8)
        stem = FIGURES / f"bandwidth_gain_rho{slug(rho)}"
        for extension in ("pdf", "png"):
            fig.savefig(stem.with_suffix("." + extension), dpi=300, bbox_inches="tight")
        plt.close(fig)
        group.to_csv(TABLES / f"bandwidth_gain_rho{slug(rho)}.csv", index=False)
        figure_tex.extend([r"\begin{figure}[p]\centering",
                           r"\includegraphics[width=\linewidth]{\DebiasingResultsRoot/figures/synthetic/debiasing_mse/"
                           + stem.name + ".pdf}",
                           rf"\caption{{Debiased versus plug-in MSE at each bandwidth pair, for $\rho={rho:g}$, $N=500$, and $s_\sigma=0.5$. Cells show $100(1-\widehat{{\mathrm{{MSE}}}}_{{\mathrm{{D}}}}/\widehat{{\mathrm{{MSE}}}}_{{\mathrm{{P}}}})$ and paired 95\% Monte Carlo intervals for operator MSE (top) and HSIC $U$-statistic MSE (bottom). Blue favors debiasing; red favors plug-in. The outlined $(1,1)$ cell compares the two methods at the median bandwidths.}}",
                           rf"\label{{fig:debiasing-bandwidth-rho{slug(rho)}}}",
                           r"\end{figure}\clearpage"])
    (TABLES / "bandwidth_figures.tex").write_text("\n".join(figure_tex) + "\n")


def save_smoothness_check(summary: pd.DataFrame) -> None:
    """Keep the noise-smoothness audit in the companion CSVs."""
    selected = summary[(summary.N == 250) & (summary.rho == .5)
                       & (summary.factor_x == .1) & (summary.factor_residual == .1)]
    selected.to_csv(TABLES / "narrow_kernel_smoothness.csv", index=False)
    audit = []
    for metric in ("operator", "hsic"):
        pivot = summary.pivot(index=["N", "rho", "factor_x", "factor_residual"],
                              columns="eps_smoothness", values=metric + "_gain_percent")
        spread = pivot.max(axis=1) - pivot.min(axis=1)
        audit.append({"metric": metric, "configurations": len(pivot),
                      "sign_changes": int(((pivot.max(axis=1) > 0) & (pivot.min(axis=1) < 0)).sum()),
                      "median_spread_pp": spread.median(), "maximum_spread_pp": spread.max()})
    pd.DataFrame(audit).to_csv(TABLES / "smoothness_audit.csv", index=False)


def main() -> None:
    global DATA, OUTPUT, TABLES, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA, help="Completed sensitivity output directory")
    parser.add_argument("--tables-dir", type=Path, default=TABLES)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    DATA = args.data.resolve()
    TABLES, FIGURES = args.tables_dir.resolve(), args.figures_dir.resolve()
    OUTPUT = TABLES
    for folder in (OUTPUT, TABLES, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)
    frame = load_repetitions()
    print(f"Validated {len(frame)} paired observations and sqrt-HSIC RTI coverage.", flush=True)
    summary = summarize_repetitions(frame)
    summary.to_csv(TABLES / "all_metrics_with_mc_intervals.csv", index=False)
    print("Computed paired Monte Carlo intervals and matched the original summaries.", flush=True)
    save_median_tables(summary)
    save_bandwidth_figures(summary)
    save_smoothness_check(summary)
    source = DATA / "checkpoints/repetition_metrics_sensitivity.csv.gz"
    manifest = {"source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "rows": len(frame), "settings": len(summary), "repetitions_per_setting": 100,
                "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_method": "paired percentile over whole seeded repetitions; common resampling weights across settings",
                "coverage_interval": "pointwise Wilson 95%", "noise_smoothness": SMOOTHNESS,
                "bandwidth_figure_N": FIGURE_N, "rti_scale": "sqrt_hsic", "hsic_estimates_clipped": False,
                "validation": ["complete balanced grid", "paired seeds 10--109", "finite inputs",
                               "sqrt-HSIC RTI endpoints reproduce all coverage flags", "all MSEs match original summaries"]}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote tables to {TABLES} and six bandwidth figures to {FIGURES}.", flush=True)


if __name__ == "__main__":
    main()
