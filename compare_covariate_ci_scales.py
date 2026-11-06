"""Compare interval scales on the same binary-covariate repetitions.

Run after submission_with_covariate.ipynb. This is a descriptive comparison:
no frequency cutoff or per-repetition selection of methods is applied.
Squared-HSIC Wald sets are intersected with [0, infinity), then square-rooted.
An empty intersection stays empty, has width zero and does not cover zero.
Historical norm intervals have undefined cases; conditional summaries label them.
"""
from pathlib import Path
from statistics import NormalDist
import argparse
import numpy as np
import pandas as pd

def sqrt_squared_interval(u: np.ndarray, half_width: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform the nonnegative intersection, preserving empty sets."""
    upper_squared = u + half_width
    nonempty = upper_squared >= 0
    lower = np.where(nonempty, np.sqrt(np.maximum(u-half_width, 0)), np.nan)
    upper = np.where(nonempty, np.sqrt(np.maximum(upper_squared, 0)), np.nan)
    return lower, upper, nonempty

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/covariate/delta_repetitions.csv"))
    parser.add_argument("--output", type=Path, default=Path("tables/covariate/squared_norm_comparison"))
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    n_sample = 120  # The fixed sample size in the reported binary-covariate experiment.
    summary, paired = [], []
    for (estimator, level), block in data.groupby(["estimator", "nominal_level"], sort=True):
        block = block.sort_values("rep")
        if len(block) != 500 or block.rep.tolist() != list(range(500)):
            raise ValueError("Expected each of the original 500 repetitions exactly once.")
        u, v, score_var, truth = [block[c].to_numpy(float) for c in ("hsic_center", "hsic_v", "score_var", "true_hsic")]
        if not np.all(np.isfinite([u, v, score_var, truth])) or np.any(v <= 0) or np.any(score_var < 0):
            raise ValueError("Inputs must be finite, with positive V and nonnegative variance.")
        z = NormalDist().inv_cdf((1+level)/2)
        positive = u > 0
        center = np.sqrt(np.maximum(u, 0))
        se_old = np.sqrt(np.divide(score_var, n_sample*u, out=np.full_like(u, np.nan), where=positive))
        se_new = np.sqrt(score_var/(n_sample*v))
        old_lower, old_upper = np.maximum(center-z*se_old, 0), center+z*se_old
        new_lower, new_upper = np.maximum(center-z*se_new, 0), center+z*se_new
        np.testing.assert_allclose(new_lower, block.sqrt_lower, atol=1e-14, rtol=1e-10)
        np.testing.assert_allclose(new_upper, block.sqrt_upper, atol=1e-14, rtol=1e-10)
        half_squared = z*2*np.sqrt(score_var/n_sample)
        transformed_lower, transformed_upper, nonempty = sqrt_squared_interval(u, half_squared)
        sqrt_truth = np.sqrt(truth)
        transformed_covered = nonempty & (transformed_lower <= sqrt_truth) & (sqrt_truth <= transformed_upper)
        squared_covered = (u-half_squared <= truth) & (truth <= u+half_squared)
        np.testing.assert_array_equal(transformed_covered, squared_covered)
        current_covered = (new_lower <= sqrt_truth) & (sqrt_truth <= new_upper)
        difference = transformed_covered.astype(int)-current_covered.astype(int)
        paired.append(dict(estimator=estimator, nominal_level=level, n=len(u),
            coverage_difference=float(difference.mean()), paired_mc_se=float(difference.std(ddof=1)/np.sqrt(len(u))),
            newly_covered=int(np.sum(transformed_covered & ~current_covered)),
            newly_missed=int(np.sum(~transformed_covered & current_covered))))
        variants = [
            ("historical_norm_u_conditional", old_lower, old_upper, positive, positive),
            ("norm_v_clipped_u", new_lower, new_upper, np.ones(len(u), bool), np.ones(len(u), bool)),
            ("sqrt_of_squared_wald", transformed_lower, transformed_upper, np.ones(len(u), bool), nonempty),
        ]
        for method, lower, upper, defined, nonempty in variants:
            covered = defined & nonempty & (lower <= sqrt_truth) & (sqrt_truth <= upper)
            zero = defined & nonempty & (lower <= 0) & (0 <= upper)
            widths = np.where(defined & nonempty, upper-lower, np.where(defined, 0, np.nan))
            for subset, mask in (("all", np.ones(len(u), bool)), ("u_positive", positive), ("u_nonpositive", ~positive)):
                count = int(mask.sum())
                if count == 0:
                    continue
                valid = mask & defined
                present = valid & nonempty
                all_defined = np.all(defined[mask])
                summary.append(dict(estimator=estimator, nominal_level=level, method=method, subset=subset,
                    n=count, n_nonpositive_u=int(np.sum(mask & ~positive)), nonpositive_u_rate=float(np.mean(~positive[mask])),
                    n_defined=int(valid.sum()), n_undefined=int(np.sum(mask & ~defined)), n_empty=int(np.sum(valid & ~nonempty)),
                    empty_rate=float(np.mean((defined & ~nonempty)[mask])),
                    n_covered=int(covered[mask].sum()),
                    coverage_all=float(covered[mask].mean()) if all_defined else np.nan,
                    coverage_defined=float(covered[valid].mean()) if valid.any() else np.nan,
                    coverage_undefined_as_misses=float(covered[mask].mean()),
                    mean_width_all=float(widths[mask].mean()) if all_defined else np.nan,
                    mean_width_defined=float(widths[valid].mean()) if valid.any() else np.nan,
                    mean_width_nonempty=float(widths[present].mean()) if present.any() else np.nan,
                    n_zero_inclusion=int(zero[mask].sum()),
                    zero_inclusion_all=float(zero[mask].mean()) if all_defined else np.nan,
                    zero_inclusion_defined=float(zero[valid].mean()) if valid.any() else np.nan))
    args.output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(summary)
    result.to_csv(args.output/"summary.csv", index=False)
    pd.DataFrame(paired).to_csv(args.output/"paired_coverage_changes.csv", index=False)
    print(result[(result.subset=="all") & np.isclose(result.nominal_level,.95)].to_string(index=False))
    print(pd.DataFrame(paired)[np.isclose(pd.DataFrame(paired).nominal_level,.95)].to_string(index=False))

if __name__ == "__main__":
    main()
