"""Compare U- and V-centering with exactly the same squared-HSIC standard error.

All repetitions are retained. Intersect each squared-scale Wald interval with
[0,infinity), then take square roots; an empty set stays empty. This changes
only the center, not the variance-score estimator or the cohort.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from compare_squared_synthetic import KEYS, Z, squared_norm_set


def summarize(block: pd.DataFrame, estimator: str) -> dict:
    prefix = f"intervals.delta_{estimator}_u"
    u = block[f"estimators.{estimator}_hsic_u"].to_numpy(float)
    v = block[f"estimators.{estimator}_hsic_v"].to_numpy(float)
    variance = block[f"{prefix}.score_var"].to_numpy(float)
    n = block[f"{prefix}.n"].to_numpy(float)
    theta = block["population_oracle.hsic"].to_numpy(float)
    null = bool(np.isclose(block.rho.iloc[0], 0.))
    if null:
        if not np.all(np.abs(theta) < 1e-12):
            raise ValueError("Unexpected nonzero null population target")
        theta = np.zeros_like(theta)
    inputs = np.column_stack([u, v, variance, n, theta])
    if not (np.isfinite(inputs).all() and (variance >= 0).all()
            and (n > 1).all() and (theta >= 0).all() and (v >= 0).all()):
        raise ValueError("Invalid numerical inputs; no rows may be dropped")
    np.testing.assert_allclose(n, block.N.to_numpy(float), atol=0, rtol=0)
    if np.min(v-u) < -1e-12:
        raise ValueError("Expected V>=U for positive-semidefinite score Grams")
    radius = Z * 2 * np.sqrt(variance/n)
    truth = np.sqrt(theta)
    result = dict(estimator=estimator, repetitions=len(block), null_descriptive=null,
                  nonpositive_u_count=int((u<=0).sum()), nonpositive_u_rate=float((u<=0).mean()),
                  mean_v_minus_u=float((v-u).mean()), min_v_minus_u=float((v-u).min()),
                  mean_squared_scale_se=float((radius/Z).mean()))
    covers = {}
    for name, center in (("u",u),("v",v)):
        low, high, empty = squared_norm_set(center, radius)
        covered = (~empty) & (low<=truth) & (truth<=high)
        np.testing.assert_array_equal(covered, (center-radius<=theta)&(theta<=center+radius))
        width = np.where(empty, 0., high-low)
        error = center-theta
        metrics = dict(coverage=float(covered.mean()), covered=int(covered.sum()),
                       coverage_mcse=float(np.sqrt(covered.mean()*(1-covered.mean())/len(block))),
                       mean_width=float(width.mean()),
                       mean_width_nonempty=float(width[~empty].mean()) if (~empty).any() else np.nan,
                       empty_count=int(empty.sum()), empty_rate=float(empty.mean()),
                       zero_inclusion=float(((~empty)&(low<=0)&(high>=0)).mean()),
                       interval_below_truth=int(((~empty)&(high<truth)).sum()),
                       interval_above_truth=int(((~empty)&(low>truth)).sum()),
                       mean_error_squared_scale=float(error.mean()),
                       sd_error_squared_scale=float(error.std(ddof=1)),
                       mse_squared_scale=float(np.mean(error**2)))
        assert metrics["covered"] + metrics["empty_count"] + metrics["interval_below_truth"] + metrics["interval_above_truth"] == len(block)
        result.update({name+"_"+key:value for key,value in metrics.items()})
        covers[name] = covered
    difference = covers["v"].astype(int)-covers["u"].astype(int)
    result.update(paired_coverage_gain=float(difference.mean()),
                  paired_gain_mcse=float(difference.std(ddof=1)/np.sqrt(len(block))),
                  v_only_covered=int(((~covers["u"])&covers["v"]).sum()),
                  u_only_covered=int((covers["u"]&(~covers["v"])).sum()),
                  v_closer_to_nominal=bool(abs(result["v_coverage"]-.95)<abs(result["u_coverage"]-.95)-1e-12))
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("tables/synthetic/centering_comparison"))
    args=parser.parse_args()
    # The same transform is used for both centers, including empty/boundary sets.
    low,high,empty=squared_norm_set(np.array([-2.,-1.,0.,2.]),np.array([1.,1.,1.,1.]))
    np.testing.assert_array_equal(empty,[True,False,False,False])
    assert low[1]==high[1]==0
    data=pd.read_csv(args.input)
    data=data[data.N.isin([250,500,1000]) & data.f_smoothness.isin([1.,1.5,2.])
              & data.eps_smoothness.isin([.25,.5,.75]) & data.rho.isin([0.,.1,.2,.3,.4,.5])].copy()
    if data.duplicated(KEYS+["random_key"]).any() or len(data.groupby(KEYS))!=162:
        raise ValueError("Expected the same deduplicated 162-setting paper cohort")
    rows=[]
    for key,block in data.groupby(KEYS,sort=True):
        for estimator in ("naive","debiased"):
            rows.append(dict(zip(KEYS,key)) | summarize(block,estimator))
    summary=pd.DataFrame(rows)
    args.output.mkdir(parents=True,exist_ok=True)
    summary.to_csv(args.output/"all_settings.csv",index=False)
    main=summary[(summary.estimator=="debiased")&(summary.f_smoothness==1)&(summary.eps_smoothness==.75)]
    main.to_csv(args.output/"main_setting.csv",index=False)
    for metric in ("u_coverage","v_coverage","u_mean_width","v_mean_width","paired_coverage_gain","v_zero_inclusion"):
        main.pivot(index="rho",columns="N",values=metric).to_csv(args.output/f"main_{metric}.csv")
    overview={}
    for estimator,block in summary.groupby("estimator"):
        alt=block[block.rho>0]
        delta=alt.v_coverage-alt.u_coverage
        closeness=np.abs(alt.u_coverage-.95)-np.abs(alt.v_coverage-.95)
        overview[estimator]=dict(positive_settings=len(alt),
            u_coverage_range=[float(alt.u_coverage.min()),float(alt.u_coverage.max())],
            v_coverage_range=[float(alt.v_coverage.min()),float(alt.v_coverage.max())],
            v_higher_count=int((delta>1e-12).sum()),u_higher_count=int((delta< -1e-12).sum()),equal_coverage_count=int((np.abs(delta)<=1e-12).sum()),
            v_closer_to_95_count=int((closeness>1e-12).sum()),u_closer_to_95_count=int((closeness< -1e-12).sum()),equal_distance_to_95_count=int((np.abs(closeness)<=1e-12).sum()),
            null_u_range=[float(block[block.rho==0].u_coverage.min()),float(block[block.rho==0].u_coverage.max())],
            null_v_range=[float(block[block.rho==0].v_coverage.min()),float(block[block.rho==0].v_coverage.max())])
    record=dict(input=str(args.input.resolve()),sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        input_rows=len(data),settings=162,nominal_level=.95,
        standard_error="same saved delta_{naive,debiased}_u plugin point-score variance: 2*sqrt(score_var/n) for both centers",
        centers=["raw U","V"],variance_changed=False,rows_dropped=0,
        transform="nonnegative intersection, then square root; empty remains empty",
        null_rows="descriptive only; known null target replaces negligible quadrature roundoff",
        overview=overview,checks=["same sample size and radius", "V>=U", "coverage invariance", "complete coverage/miss accounting", "empty boundary"])
    (args.output/"provenance.json").write_text(json.dumps(record,indent=2)+"\n")
    print(main[["N","rho","repetitions","u_coverage","v_coverage","u_mean_width","v_mean_width","u_interval_below_truth","u_interval_above_truth","v_interval_below_truth","v_interval_above_truth","paired_coverage_gain","paired_gain_mcse"]].to_string(index=False))
    print(json.dumps(overview,indent=2))

if __name__=="__main__":
    main()
