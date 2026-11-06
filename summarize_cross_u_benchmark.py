"""Report the paper's U-delta/V-RTI comparison and paired cross-U debiasing effects.

Read completed replay metrics only. Delta uses a signed-U HSIC Wald set, intersected with nonnegative HSIC
and mapped through the square root. All primary methods must retain every available repetition.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("HSIC_PROJECT_ROOT", Path(__file__).resolve().parent))
KEYS = ["N", "eps_smoothness", "rho", "random_key"]
SETTINGS = KEYS[:-1]
DEBIASED = "cross_u_debiased_inflated"
PLUGIN = "cross_u_plugin_inflated"
TRUE_RESIDUAL = "cross_u_true_residual_inflated"
RTI = "rti_debiased"
DELTA = "dm_debiased_squared_u"
PAPER_METHODS = [DEBIASED, RTI, DELTA]
CROSS_METHODS = [PLUGIN, DEBIASED, TRUE_RESIDUAL]
LABELS = {
    DEBIASED: "Cross-U: debiased",
    PLUGIN: "Cross-U: plug-in",
    TRUE_RESIDUAL: "Cross-U: true residuals",
    RTI: "RTI (V center)",
    DELTA: "Delta (squared-U then sqrt)",
}
COLORS = {DEBIASED: "#1f77b4", PLUGIN: "#d62728", TRUE_RESIDUAL: "#777777",
          RTI: "#ff7f0e", DELTA: "#2ca02c"}
BINARY_METRICS = ["coverage", "ci_excludes_zero", "reject_right_tail", "empty"]
PAIRED_METRICS = ["coverage", "ci_excludes_zero", "reject_right_tail", "empty",
                  "width_hsic", "width_sqrt_hsic", "point_error", "squared_error"]


def load_metrics(path: Path) -> pd.DataFrame:
    """Select the requested methods and validate matching targets and replicate IDs."""
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame.method.isin(PAPER_METHODS + CROSS_METHODS)].copy()
    if frame.duplicated(KEYS + ["method"]).any():
        raise ValueError("Duplicate method/replicate records.")
    if not frame.groupby(KEYS).method.nunique().eq(5).all():
        raise ValueError("Every replicate must contain all five reporting methods.")
    if not frame.groupby(KEYS).population_hsic.nunique().eq(1).all():
        raise ValueError("Methods do not share the same population target.")
    if frame.coverage.isna().any() or not np.isfinite(frame.width_sqrt_hsic).all():
        raise ValueError("A reported interval is undefined; refusing to discard repetitions.")
    frame["point_error"] = frame.estimate - frame.population_hsic
    frame["squared_error"] = frame.point_error ** 2
    return frame


def common_defined(frame: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Keep identical replicate IDs for all methods, without filling undefined CIs."""
    subset = frame[frame.method.isin(methods)].copy()
    available = subset.pivot(index=KEYS, columns="method", values="coverage")
    retained = available.dropna(subset=methods).reset_index()[KEYS]
    return subset.merge(retained, on=KEYS, how="inner", validate="many_to_one")


def summarize_methods(frame: pd.DataFrame) -> pd.DataFrame:
    """Give method-specific denominators; binary uncertainty uses Wilson intervals."""
    rows = []
    z = NormalDist().inv_cdf(.975)
    for key, block in frame.groupby(SETTINGS + ["method"]):
        row = dict(zip(SETTINGS + ["method"], key))
        row["available_runs"] = len(block)
        row["defined_intervals"] = int(block.coverage.notna().sum())
        row["undefined_intervals"] = len(block) - row["defined_intervals"]
        row["undefined_fraction"] = row["undefined_intervals"] / len(block)
        for metric in BINARY_METRICS:
            values = block[metric].dropna().astype(float)
            n = len(values)
            row[f"{metric}_runs"] = n
            row[f"{metric}_successes"] = int(values.sum())
            row[metric] = values.mean()
            if n:
                p = float(values.mean())
                center = (p + z*z/(2*n)) / (1 + z*z/n)
                radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
                row[f"{metric}_lower"] = center - radius
                row[f"{metric}_upper"] = center + radius
        for metric in ("width_hsic", "width_sqrt_hsic", "point_error", "squared_error"):
            row[f"mean_{metric}"] = block[metric].mean()
            row[f"median_{metric}"] = block[metric].median()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(frame: pd.DataFrame, first: str, second: str,
                       metrics: list[str]) -> pd.DataFrame:
    """Use within-replicate differences, not independent-method standard errors."""
    rows = []
    for metric in metrics:
        wide = frame[frame.method.isin([first, second])].pivot(index=KEYS, columns="method", values=metric)
        for key, block in wide.groupby(level=SETTINGS):
            paired = block[[first, second]].dropna().astype(float)
            if paired.empty:
                continue
            differences = paired[first] - paired[second]
            n = len(differences)
            mean = float(differences.mean())
            se = float(differences.std(ddof=1) / math.sqrt(n)) if n > 1 else np.nan
            row = {**dict(zip(SETTINGS, key)), "first_method": first, "second_method": second,
                   "metric": metric, "available_pairs": len(block), "paired_runs": n,
                   "undefined_pairs": len(block)-n, "first_mean": paired[first].mean(),
                   "second_mean": paired[second].mean(), "difference": mean,
                   "mc_standard_error": se,
                   "difference_lower": mean - 1.959963984540054*se,
                   "difference_upper": mean + 1.959963984540054*se}
            if metric in BINARY_METRICS:
                row["first_only_successes"] = int(((paired[first]==1)&(paired[second]==0)).sum())
                row["second_only_successes"] = int(((paired[first]==0)&(paired[second]==1)).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def debiasing_settings(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize signed changes and distance from nominal coverage separately."""
    selected = summary[summary.method.isin(CROSS_METHODS)]
    rows = []
    for key, block in selected.groupby(SETTINGS):
        block = block.set_index("method")
        row = dict(zip(SETTINGS, key))
        row["runs"] = int(block.loc[DEBIASED, "available_runs"])
        for method, short in ((PLUGIN, "plugin"), (DEBIASED, "debiased"), (TRUE_RESIDUAL, "true_residual")):
            for metric in ("coverage", "ci_excludes_zero", "reject_right_tail", "mean_width_sqrt_hsic", "mean_squared_error"):
                row[f"{short}_{metric}"] = block.loc[method, metric]
        row["coverage_change_pp"] = 100*(row["debiased_coverage"]-row["plugin_coverage"])
        row["zero_exclusion_change_pp"] = 100*(row["debiased_ci_excludes_zero"]-row["plugin_ci_excludes_zero"])
        row["width_change_percent"] = 100*(row["debiased_mean_width_sqrt_hsic"]/row["plugin_mean_width_sqrt_hsic"]-1)
        row["mse_change_percent"] = 100*(row["debiased_mean_squared_error"]/row["plugin_mean_squared_error"]-1)
        row["coverage_error_reduction_pp"] = 100*(abs(row["plugin_coverage"]-.95)-abs(row["debiased_coverage"]-.95))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_grid(summary: pd.DataFrame, methods: list[str], metric: str, label: str,
              path: Path, cohort_note: str, show_delta_null: bool = False) -> None:
    """Plot already aggregated metrics; explicitly label the conditioning convention."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    smoothnesses = sorted(summary.eps_smoothness.unique())
    fig, axes = plt.subplots(len(smoothnesses), 3, figsize=(12.5, 3*len(smoothnesses)+.8), sharex=True, squeeze=False)
    for row_idx, smoothness in enumerate(smoothnesses):
        for col_idx, n in enumerate((250, 500, 1000)):
            ax = axes[row_idx, col_idx]
            panel = summary[(summary.N==n)&(summary.eps_smoothness==smoothness)]
            if panel.empty:
                ax.text(.5, .5, "Not in the evaluated grid", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"n={n}, noise smoothness={smoothness}", fontsize=11)
                continue
            for method in methods:
                values = panel[panel.method==method].sort_values("rho")
                if method==DELTA and not show_delta_null:
                    values = values[values.rho>0]
                kwargs = {"color": COLORS[method], "label": LABELS[method], "marker": "o",
                          "linestyle": "--" if method in (DELTA,TRUE_RESIDUAL) else "-"}
                if metric in BINARY_METRICS:
                    errors = np.maximum(0, np.stack((values[metric]-values[f"{metric}_lower"],
                                                    values[f"{metric}_upper"]-values[metric])))
                    ax.errorbar(values.rho, values[metric], yerr=errors, capsize=2, **kwargs)
                    ax.set_ylim(-.025, 1.025)
                else:
                    ax.plot(values.rho, values[metric], **kwargs)
            if metric=="coverage":
                ax.axhline(.95, color="gray", linestyle=":", linewidth=1)
            if metric=="ci_excludes_zero":
                # The 5% reference is relevant at rho=0, not over the power curve.
                ax.plot(0, .05, marker="_", markersize=18, color="black")
            ax.set_title(f"n={n}, noise smoothness={smoothness}", fontsize=11)
            ax.set_xlabel("rho (0: null; positive: alternative)")
            if col_idx==0:
                ax.set_ylabel(label)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods), bbox_to_anchor=(.5,.045), fontsize=9)
    fig.text(.5,.018,cohort_note,ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.12,1,1))
    for extension in ("png","pdf"):
        fig.savefig(path.with_suffix("."+extension),dpi=160)
    plt.close(fig)


def plot_debiasing_difference(differences: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5),sharex=True,sharey=True)
    coverage = differences[differences.metric=="coverage"]
    for i,smoothness in enumerate((.25,.75)):
        for j,n in enumerate((250,500,1000)):
            ax=axes[i,j]
            values=coverage[(coverage.N==n)&(coverage.eps_smoothness==smoothness)].sort_values("rho")
            ax.errorbar(values.rho,100*values.difference,yerr=100*1.959963984540054*values.mc_standard_error,
                        marker="o",capsize=3,color=COLORS[DEBIASED])
            ax.axhline(0,color="gray",linestyle=":")
            ax.set_title(f"n={n}, noise smoothness={smoothness}")
            ax.set_xlabel("Heterogeneity rho")
            if j==0:
                ax.set_ylabel("Coverage change: debiased - plug-in (pp)")
    fig.text(.5,.015,"Paired differences, same runs and splits; bars: ±1.96 Monte Carlo SE. Higher coverage is not always closer to the 95% target.",
             ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1))
    for extension in ("png","pdf"):
        fig.savefig(path.with_suffix("."+extension),dpi=160)
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"runs/paper_cross_u_squared_u")
    parser.add_argument("--tables-dir", type=Path, default=ROOT/"tables/synthetic/cross_u_comparison")
    parser.add_argument("--figures-dir", type=Path, default=ROOT/"figures/synthetic/cross_u_comparison")
    args=parser.parse_args()
    frame=load_metrics(args.run/"replicate_metrics.csv")
    tables=args.tables_dir
    figures=args.figures_dir
    tables.mkdir(parents=True,exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    summary=summarize_methods(frame)
    summary.to_csv(tables/"summary.csv",index=False)
    common=common_defined(frame,PAPER_METHODS)
    common_summary=summarize_methods(common)
    common_summary.to_csv(tables/"paper_common_defined_summary.csv",index=False)
    summary[summary.method==DELTA].to_csv(tables/"delta_availability.csv",index=False)
    cross_differences=paired_differences(frame,DEBIASED,PLUGIN,PAIRED_METRICS)
    cross_differences.to_csv(tables/"cross_u_debiasing_paired.csv",index=False)
    debiasing_settings(summary).to_csv(tables/"cross_u_debiasing_by_setting.csv",index=False)
    paper_differences=pd.concat([paired_differences(frame,DEBIASED,baseline,
                                                   ["coverage","ci_excludes_zero","width_hsic","width_sqrt_hsic"])
                                 for baseline in (RTI,DELTA)],ignore_index=True)
    paper_differences.to_csv(tables/"paired_differences.csv",index=False)
    metrics=[("coverage","95% interval coverage","coverage"),
             ("mean_width_sqrt_hsic","Mean interval length (sqrt HSIC)","mean_width_sqrt_hsic"),
             ("ci_excludes_zero","Fraction excluding zero","ci_excludes_zero")]
    for metric,label,stem in metrics:
        plot_grid(summary,PAPER_METHODS,metric,label,figures/stem,
                  "All available runs for every method. Delta uses a signed-U HSIC Wald set, then sqrt; omitted at rho=0.")
        plot_grid(common_summary,PAPER_METHODS,metric,label,figures/("paper_common_defined_"+stem),
                  "All methods use the same full set of available runs; no conditioning on positive U. Delta is omitted at rho=0.")
        plot_grid(summary,CROSS_METHODS,metric,label,figures/("debiasing_"+stem),
                  "All available runs; identical data, bandwidths, five nuisance folds, A/B split, and sqrt(2) SE inflation for all three cross-U versions.")
    # Also refresh the previously shared HSIC-scale length figure, preserving the CI definitions.
    plot_grid(summary,PAPER_METHODS,"mean_width_hsic","Mean interval length (HSIC)",figures/"mean_width_hsic",
              "Square-root interval endpoints are squared for display only. Every method retains all available runs.")
    plot_debiasing_difference(cross_differences,figures/"debiasing_coverage_difference")
    print(f"Reported {frame.source.nunique()} paired runs; all have a defined signed-U HSIC Wald set, possibly empty.")
    print(f"Tables: {tables}\nFigures: {figures}")


if __name__=="__main__":
    main()
