"""Compare U and V centering with an identical squared-HSIC standard error.

This is a controlled, descriptive comparison. Both squared-scale Wald intervals
use 2*sqrt(score_var/n), then intersection with [0, infinity) and square root.
No production procedure is selected or changed.
"""
from pathlib import Path
from statistics import NormalDist
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from compare_covariate_ci_scales import sqrt_squared_interval



def wilson_interval(covered: np.ndarray) -> tuple[float, float]:
    """95% Monte Carlo interval for a coverage proportion."""
    n = len(covered)
    if n == 0:
        return np.nan, np.nan
    z = NormalDist().inv_cdf(.975)
    p = float(covered.mean())
    denominator = 1 + z*z/n
    middle = (p + z*z/(2*n))/denominator
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return middle-radius, middle+radius

def oracle_records() -> pd.DataFrame:
    """Same 500 seeds with true nuisances; no regression fits are performed."""
    import nbformat
    notebook = nbformat.read("submission_with_covariate.ipynb", 4)
    ns = {"np": np, "pd": pd, "Path": Path, "NormalDist": NormalDist}
    for index in (1, 2, 3, 8):
        exec(notebook.cells[index].source, ns)
    scale, omega, phase = [ns[k] for k in ("RFF_SCALE", "RFF_OMEGA", "RFF_PHASE")]
    mu = [scale*np.cos(phase)*np.exp(-.5*(s*omega)**2) for s in (.35, .55)]
    truth = ns["population_oracle_hsic"]("x_scale")
    rows = []
    for rep in range(500):
        x, t, y, mean, sigma = ns["simulate"](120, regime="x_scale", seed=50000+rep)
        residual = y-mean
        a = ns["one_hot_binary"](x)-.5
        b = ns["residual_feature_map"](residual)-.5*(mu[0]+mu[1])
        v_true = -scale*omega[None,:]*np.sin(phase)[None,:]*np.exp(-.5*(sigma[:,None]*omega[None,:])**2)
        folds = ns["make_folds"](120, 4, np.random.default_rng(60000+rep))
        for estimator, contribution in (("plug-in", b), ("debiased", b-residual[:,None]*v_true)):
            gram = ns["score_gram_blocks"](a, contribution, folds)
            ci = ns["delta_sqrt_hsic_ci_from_blocks"](gram, NormalDist().inv_cdf(.975))
            rows.append(dict(study="oracle", N=120, estimator=estimator, rep=rep,
                             nominal_level=.95, true_hsic=truth, **ci))
    return pd.DataFrame(rows)


def compare(data: pd.DataFrame, expected_repetitions: int = 500) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep every repetition and compare only the center."""
    summaries, paired, bias = [], [], []
    for (study, estimator, level), block in data.groupby(["study", "estimator", "nominal_level"]):
        block = block.sort_values("rep")
        if len(block) != expected_repetitions or block.rep.tolist() != list(range(expected_repetitions)):
            raise ValueError(f"Expected repetitions 0 through {expected_repetitions-1}, without duplicates.")
        n_sample = int(block["N"].iloc[0])
        if block["N"].nunique() != 1 or block["N"].isna().any() or n_sample < 2:
            raise ValueError("Each study must have one valid sample size.")
        u, v, var, truth = [block[k].to_numpy(float) for k in ("hsic_center", "hsic_v", "score_var", "true_hsic")]
        if not np.all(np.isfinite([u,v,var,truth])) or np.any(var<0) or np.any(truth<0):
            raise ValueError("Finite inputs and nonnegative variance/target are required.")
        if np.any(v < u-1e-13):
            raise ValueError("Expected nonnegative V-U diagonal correction.")
        se = 2*np.sqrt(var/n_sample)
        if "se_hsic" in block:
            np.testing.assert_allclose(se, block.se_hsic, atol=1e-14, rtol=1e-10)
        half = NormalDist().inv_cdf((1+level)/2)*se
        covered_by_center = {}
        for center_name, center in (("U",u), ("V",v)):
            lo, hi, nonempty = sqrt_squared_interval(center, half)
            covered = nonempty & (lo<=np.sqrt(truth)) & (np.sqrt(truth)<=hi)
            raw_lo, raw_hi = center-half, center+half
            np.testing.assert_array_equal(covered, (raw_lo<=truth)&(truth<=raw_hi))
            width = np.where(nonempty, hi-lo, 0.)
            lower_miss = raw_lo>truth
            upper_miss = raw_hi<truth
            np.testing.assert_array_equal(covered.astype(int)+lower_miss+upper_miss, np.ones(len(u), int))
            covered_by_center[center_name] = covered
            wilson_lo, wilson_hi = wilson_interval(covered)
            summaries.append(dict(study=study, estimator=estimator, nominal_level=level,
                center=center_name, N=n_sample, n=len(u), covered=int(covered.sum()),
                coverage=float(covered.mean()), coverage_mcse=float(np.sqrt(covered.mean()*(1-covered.mean())/len(u))),
                coverage_wilson95_lower=wilson_lo, coverage_wilson95_upper=wilson_hi,
                lower_endpoint_above_truth=int(lower_miss.sum()),
                upper_endpoint_below_truth=int(upper_miss.sum()),
                empty=int((~nonempty).sum()), empty_rate=float((~nonempty).mean()),
                zero_inclusion=int(((raw_lo<=0)&(0<=raw_hi)).sum()),
                zero_inclusion_rate=float(((raw_lo<=0)&(0<=raw_hi)).mean()),
                mean_width=float(width.mean()),
                mean_width_nonempty=float(width[nonempty].mean()) if nonempty.any() else np.nan,
                nonpositive_u=int((u<=0).sum())))
            if np.isclose(level,.95):
                bias.append(dict(study=study, estimator=estimator, center=center_name, N=n_sample,
                    target=float(truth.mean()), mean_estimate=float(center.mean()),
                    bias=float((center-truth).mean()), empirical_sd=float(center.std(ddof=1)),
                    rmse=float(np.sqrt(np.mean((center-truth)**2))),
                    rms_held_fixed_se=float(np.sqrt(np.mean(se**2))),
                    mean_v_minus_u=float((v-u).mean())))
        cu, cv = covered_by_center["U"], covered_by_center["V"]
        change = cv.astype(int)-cu.astype(int)
        paired.append(dict(study=study, estimator=estimator, nominal_level=level, N=n_sample, n=len(u),
            coverage_gain_v_minus_u=float(change.mean()),
            paired_mcse=float(change.std(ddof=1)/np.sqrt(len(u))),
            newly_covered=int((cv&~cu).sum()), newly_missed=int((~cv&cu).sum())))
    return pd.DataFrame(summaries), pd.DataFrame(paired), pd.DataFrame(bias)



def compare_norm_intervals(data: pd.DataFrame) -> pd.DataFrame:
    """Secondary norm-scale intervals, including the historical undefined cases."""
    summaries = []
    for (study, estimator, level), block in data.groupby(["study", "estimator", "nominal_level"]):
        block = block.sort_values("rep")
        n_sample = int(block["N"].iloc[0])
        u, v, var, theta = [block[k].to_numpy(float) for k in ("hsic_center", "hsic_v", "score_var", "true_hsic")]
        if np.any(v <= 0):
            raise ValueError("The norm comparison requires positive V denominators.")
        truth = np.sqrt(theta)
        z = NormalDist().inv_cdf((1+level)/2)
        se_v = np.sqrt(var/(n_sample*v))
        positive_u = u > 0
        se_u = np.sqrt(np.divide(var, n_sample*u, out=np.full_like(var, np.nan), where=positive_u))
        variants = [
            ("norm_U_Vvariance", np.sqrt(np.maximum(u, 0)), se_v, np.ones(len(u), bool)),
            ("norm_V_Vvariance", np.sqrt(v), se_v, np.ones(len(u), bool)),
            ("historical_norm_U_Uvariance", np.sqrt(np.maximum(u, 0)), se_u, positive_u),
        ]
        for method, center, se, defined in variants:
            lo, hi = np.maximum(center-z*se, 0), center+z*se
            covered = defined & (lo <= truth) & (truth <= hi)
            width = hi-lo
            wilson_lo, wilson_hi = wilson_interval(covered)
            conditional_lo, conditional_hi = wilson_interval(covered[defined])
            summaries.append(dict(
                study=study, estimator=estimator, nominal_level=level, N=n_sample,
                method=method, repetitions=len(u), defined=int(defined.sum()),
                undefined=int((~defined).sum()), covered=int(covered.sum()),
                coverage_undefined_as_failure=float(covered.mean()),
                coverage_defined=float(covered[defined].mean()) if defined.any() else np.nan,
                coverage_all_wilson95_lower=wilson_lo, coverage_all_wilson95_upper=wilson_hi,
                coverage_defined_wilson95_lower=conditional_lo, coverage_defined_wilson95_upper=conditional_hi,
                mean_width_all=float(width.mean()) if defined.all() else np.nan,
                mean_width_defined=float(width[defined].mean()) if defined.any() else np.nan,
                lower_endpoint_above_truth=int((defined & (lo > truth)).sum()),
                upper_endpoint_below_truth=int((defined & (hi < truth)).sum()),
                zero_inclusion=int((defined & (lo <= 0) & (0 <= hi)).sum()),
                nonpositive_u=int((u <= 0).sum()),
            ))
    return pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/covariate/delta_repetitions.csv"))
    parser.add_argument("--output", type=Path, default=Path("tables/covariate/centering_comparison"))
    parser.add_argument("--expected-repetitions", type=int, default=500)
    parser.add_argument("--prefixes", type=int, nargs="+", help="Summarize these initial repetition counts separately.")
    parser.add_argument("--oracle", action="store_true", help="Also evaluate true nuisances on the original 500 seeds.")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    if "study" not in data:
        data["study"] = "fitted"
    if "nominal_level" not in data:
        data["nominal_level"] = .95
    if "N" not in data:
        data["N"] = data["n"] if "n" in data else 120
    if args.oracle:
        if args.expected_repetitions != 500:
            parser.error("--oracle applies only to the original 500-repetition cohort.")
        data = pd.concat([data, oracle_records()], ignore_index=True)
    # Validate the full input cohort before inspecting smaller prefixes.
    compare(data, args.expected_repetitions)
    prefixes = args.prefixes or [args.expected_repetitions]
    if any(count < 2 or count > args.expected_repetitions for count in prefixes):
        parser.error("Every prefix must be between 2 and --expected-repetitions.")
    frames = [[], [], [], []]
    for count in prefixes:
        subset = data[data.rep < count]
        outputs = (*compare(subset, count), compare_norm_intervals(subset))
        for destination, frame in zip(frames, outputs):
            frame["prefix_repetitions"] = count
            destination.append(frame)
    summary, paired, bias, norm = [pd.concat(parts, ignore_index=True) for parts in frames]
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output/"summary.csv", index=False)
    paired.to_csv(args.output/"paired_changes.csv", index=False)
    bias.to_csv(args.output/"bias_and_variability.csv", index=False)
    norm.to_csv(args.output/"norm_summary.csv", index=False)
    (args.output/"provenance.json").write_text(json.dumps(dict(
        input=str(args.input), sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        sample_sizes=sorted(int(value) for value in data.N.unique()),
        repetitions=args.expected_repetitions, prefixes=prefixes,
        change="Center only: U versus V; identical LOO squared-scale SE",
        se="2*sqrt(score_var/N)", oracle=args.oracle,
        empty="negative upper endpoint stays empty, width zero and no coverage",
        coverage_equivalence_checked=True, diagonal_correction_nonnegative_checked=True,
        norm_secondary="U center with V variance; V center with V variance; historical U center with U variance",
        historical_undefined="U<=0 is undefined; both conditional and undefined-as-failure coverage reported"
    ), indent=2)+"\n")
    print(summary[np.isclose(summary.nominal_level,.95)].to_string(index=False))
    print(paired[np.isclose(paired.nominal_level,.95)].to_string(index=False))
    print(bias.to_string(index=False))
    print(norm[np.isclose(norm.nominal_level,.95)].to_string(index=False))


if __name__ == "__main__":
    main()
