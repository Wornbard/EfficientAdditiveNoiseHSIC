"""Compare whole-setting norm intervals with the square root of a squared-HSIC Wald set.

Input is analysis_input_rows.csv.gz exported by synthetic_paper_plots.ipynb.
No setting cutoff or per-repetition formula switch is selected here. Null rows
are descriptive: the fixed-alternative normal approximation is not null-valid.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd

KEYS = ["N", "f_smoothness", "eps_smoothness", "rho"]
Z = NormalDist().inv_cdf(.975)

def squared_norm_set(u: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect the squared-scale Wald set with [0,infinity), then square-root."""
    if not (np.isfinite(u).all() and np.isfinite(radius).all() and (radius>=0).all()):
        raise ValueError("Squared-norm intervals require finite U and nonnegative finite radius")
    low, high = u-radius, u+radius
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError("Squared-norm interval endpoints overflowed")
    empty = high < 0
    # An empty set is represented by reversed endpoints, not the singleton zero.
    return np.where(empty, 1., np.sqrt(np.maximum(low, 0.))), np.where(empty, 0., np.sqrt(np.maximum(high, 0.))), empty

def summarize(block: pd.DataFrame, estimator: str) -> dict:
    prefix = f"intervals.delta_{estimator}_u"
    u = block[f"estimators.{estimator}_hsic_u"].to_numpy(float)
    v = block[f"estimators.{estimator}_hsic_v"].to_numpy(float)
    variance = block[f"{prefix}.score_var"].to_numpy(float)
    n = block[f"{prefix}.n"].to_numpy(float)
    theta = block["population_oracle.hsic"].to_numpy(float)
    if np.isclose(block.rho.iloc[0], 0.):
        # Independence gives an exactly zero target; quadrature leaves ~1e-31 noise.
        if not np.all(np.abs(theta)<1e-12):
            raise ValueError("Unexpected nonzero population target at rho=0")
        theta = np.zeros_like(theta)
    values = np.column_stack([u, v, variance, n, theta])
    if not (np.isfinite(values).all() and (v>0).all() and (variance>=0).all() and (n>1).all() and (theta>=0).all()):
        raise ValueError("Invalid inputs; repetitions must not be silently excluded")
    truth = np.sqrt(theta)
    radius = Z*2*np.sqrt(variance/n)
    low, high, empty = squared_norm_set(u, radius)
    covered = (~empty) & (low<=truth) & (truth<=high)
    squared_covered = (u-radius<=theta) & (theta<=u+radius)
    np.testing.assert_array_equal(covered, squared_covered)
    width = np.where(empty, 0., high-low)
    zero_included = (~empty) & (low<=0) & (high>=0)
    positive = u>0

    center = np.sqrt(np.maximum(u,0.))
    current_radius = Z*np.sqrt(variance/(n*v))
    current_low = np.maximum(0.,center-current_radius)
    current_high = center+current_radius
    current_covered = (current_low<=truth) & (truth<=current_high)
    historical_radius = Z*np.sqrt(variance[positive]/(n[positive]*u[positive]))
    historical_low = np.maximum(0.,center[positive]-historical_radius)
    historical_high = center[positive]+historical_radius
    historical_covered = (historical_low<=truth[positive]) & (truth[positive]<=historical_high)
    difference = covered.astype(int)-current_covered.astype(int)

    result = dict(
        estimator=estimator, repetitions=len(block),
        nonpositive_u_count=int((~positive).sum()), nonpositive_u_rate=float((~positive).mean()),
        transformed_coverage=float(covered.mean()), transformed_covered=int(covered.sum()),
        transformed_coverage_mcse=float(np.sqrt(covered.mean()*(1-covered.mean())/len(block))),
        transformed_mean_width=float(width.mean()),
        transformed_mean_width_nonempty=float(width[~empty].mean()) if (~empty).any() else np.nan,
        transformed_empty_count=int(empty.sum()), transformed_empty_rate=float(empty.mean()),
        transformed_zero_inclusion=float(zero_included.mean()),
        current_norm_coverage=float(current_covered.mean()), current_norm_mean_width=float((current_high-current_low).mean()),
        historical_coverage_undefined_as_failure=float(historical_covered.sum()/len(block)),
        historical_defined_count=int(positive.sum()), historical_conditional_coverage=float(historical_covered.mean()) if positive.any() else np.nan,
        historical_conditional_mean_width=float((historical_high-historical_low).mean()) if positive.any() else np.nan,
        paired_coverage_gain=float(difference.mean()), paired_gain_mcse=float(difference.std(ddof=1)/np.sqrt(len(block))),
        transformed_coverage_positive_u=float(covered[positive].mean()) if positive.any() else np.nan,
        transformed_coverage_nonpositive_u=float(covered[~positive].mean()) if (~positive).any() else np.nan,
        null_descriptive=bool(np.isclose(block.rho.iloc[0],0.)),
    )
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tables/synthetic/squared_interval_comparison"))
    args = parser.parse_args()
    # Boundary checks: entirely negative -> empty; touching zero -> singleton zero.
    lo, hi, empty = squared_norm_set(np.array([-2.,-1.,2.]),np.array([1.,1.,1.]))
    np.testing.assert_array_equal(empty,[True,False,False])
    assert lo[0]>hi[0] and lo[1]==hi[1]==0 and lo[2]==1
    data = pd.read_csv(args.input)
    data = data[data.N.isin([250,500,1000]) & data.f_smoothness.isin([1.,1.5,2.]) &
                data.eps_smoothness.isin([.25,.5,.75]) & data.rho.isin([0.,.1,.2,.3,.4,.5])].copy()
    if data.duplicated(KEYS+["random_key"]).any():
        raise ValueError("Input must use the plotting notebook's deduplicated source cohort")
    if len(data.groupby(KEYS)) != 162:
        raise ValueError("Expected all 162 reported sample-size/smoothness/rho settings")
    rows=[]
    for key,block in data.groupby(KEYS,sort=True):
        for estimator in ("naive","debiased"):
            rows.append(dict(zip(KEYS,key)) | summarize(block,estimator))
    summary=pd.DataFrame(rows)
    args.output.mkdir(parents=True,exist_ok=True)
    summary.to_csv(args.output/"all_settings.csv",index=False)
    main=summary[(summary.estimator=="debiased") & (summary.f_smoothness==1) & (summary.eps_smoothness==.75)]
    main.to_csv(args.output/"main_setting.csv",index=False)
    for metric in ("nonpositive_u_rate","transformed_coverage","transformed_mean_width","transformed_empty_rate","current_norm_coverage","historical_coverage_undefined_as_failure"):
        main.pivot(index="rho",columns="N",values=metric).to_csv(args.output/f"main_{metric}.csv")
    record=dict(input=str(args.input.resolve()),sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                input_rows=len(data),settings=162,nominal_level=.95,variance="saved delta_debiased_u / delta_naive_u plugin point-score variance",
                method="sqrt(([U-z*2*sqrt(score_var/n),U+z*2*sqrt(score_var/n)]) intersect [0,infinity))",
                empty_set="upper endpoint below zero; width0, no coverage, no zero inclusion",
                selection="same method for every repetition within each setting; no cutoff chosen",
                null_rows="descriptive only; population target set to exact model value zero after tolerance check",coverage_invariance_checked=True,boundary_checks_passed=True)
    (args.output/"provenance.json").write_text(json.dumps(record,indent=2)+"\n")
    print(main[["N","rho","repetitions","nonpositive_u_count","nonpositive_u_rate","transformed_coverage","transformed_mean_width","transformed_empty_rate","current_norm_coverage"]].to_string(index=False))
    for estimator,block in summary[summary.rho>0].groupby("estimator"):
        print(estimator, "positive-rho ranges", block[["nonpositive_u_rate","transformed_coverage","transformed_empty_rate","paired_coverage_gain"]].agg(["min","max"]).to_string())

if __name__=="__main__":
    main()
