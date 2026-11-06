"""Diagnose norm-CI coverage with true-nuisance controls on the same 500 seeds.

Run from the repository root after submission_with_covariate.ipynb. This writes
small diagnostic summaries only; it does not change the production intervals.
The HSIC-scale Wald intervals are a diagnostic comparison, not a selected
replacement procedure. No additional large experiment is run.
"""
from pathlib import Path
from statistics import NormalDist
import json
import nbformat
import numpy as np
import pandas as pd

ns = {"np": np, "pd": pd, "Path": Path, "NormalDist": NormalDist}
nb = nbformat.read("submission_with_covariate.ipynb", 4)
for index in (1, 2, 3, 5, 8):
    exec(nb.cells[index].source, ns)
scale, omega, phase = [ns[name] for name in ("RFF_SCALE", "RFF_OMEGA", "RFF_PHASE")]
mu = [scale * np.cos(phase) * np.exp(-.5 * (sigma * omega)**2) for sigma in (.35, .55)]
truth = ns["population_oracle_hsic"]("x_scale")
nodes, weights = np.polynomial.hermite.hermgauss(80)
quad_mu = [weights @ ns["residual_feature_map"](np.sqrt(2) * sigma * nodes) / np.sqrt(np.pi) for sigma in (.35, .55)]
np.testing.assert_allclose(quad_mu, mu, atol=2e-15, rtol=2e-15)
quad_delta = quad_mu[0] - quad_mu[1]
np.testing.assert_allclose(np.dot(quad_delta, quad_delta) / 8, truth, atol=1e-16, rtol=1e-12)

# Independent vector/Gram and diagonal-correction checks on fitted repetitions.
captured = []
original_gram = ns["score_gram_blocks"]
def capture(a, b, folds):
    captured.append((a.copy(), b.copy(), folds))
    return original_gram(a, b, folds)
ns["score_gram_blocks"] = capture
for rep in (0, 1, 2):
    x, z, y, mean, sigma = ns["simulate"](120, regime="x_scale", seed=50000 + rep)
    estimates = ns["estimate_hsic_u"](x, z, y, mean, sigma, seed=60000 + rep)
    for name, (a, b, folds) in zip(("naive", "debiased"), captured[-2:]):
        d = np.einsum("ia,ib->iab", a, b).reshape(120, -1)
        h = estimates[name + "_H_folds"]
        v = sum(np.mean(block) for row in h for block in row) / 16
        u = ns["crossfit_u_from_blocks"](h)
        gap = sum(np.trace(np.cov(d[f], rowvar=False, ddof=1)) / 30 for f in folds) / 16
        np.testing.assert_allclose(v, np.dot(d.mean(0), d.mean(0)), atol=1e-14, rtol=1e-12)
        np.testing.assert_allclose(v-u, gap, atol=1e-14, rtol=1e-12)
        np.testing.assert_allclose(d.mean(0).reshape(2, 16), estimates["psi_" + name], atol=1e-14, rtol=1e-12)

records = []
for rep in range(500):
    x, z, y, mean, sigma = ns["simulate"](120, regime="x_scale", seed=50000 + rep)
    residual = y-mean
    a = ns["one_hot_binary"](x)-.5
    b = ns["residual_feature_map"](residual)-.5*(mu[0]+mu[1])
    v_true = -scale * omega[None,:] * np.sin(phase)[None,:] * np.exp(-.5*(sigma[:,None]*omega[None,:])**2)
    folds = ns["make_folds"](120, 4, np.random.default_rng(60000 + rep))
    for method, contribution in (("oracle_plugin", b), ("oracle_debiased", b-residual[:,None]*v_true)):
        ci = ns["delta_sqrt_hsic_ci_from_blocks"](original_gram(a, contribution, folds), NormalDist().inv_cdf(.975))
        records.append({"method": method, "rep": rep, **ci})
oracle = pd.DataFrame(records)
fitted = pd.read_csv("results/covariate/delta_repetitions.csv")
fitted = fitted[np.isclose(fitted.nominal_level, .95)].copy()
fitted["method"] = fitted.estimator.map({"plug-in": "fitted_plugin", "debiased": "fitted_debiased"})
all_records = pd.concat([oracle, fitted], ignore_index=True)
all_records["norm_covered"] = (all_records.sqrt_lower <= np.sqrt(truth)) & (np.sqrt(truth) <= all_records.sqrt_upper)
all_records["hsic_covered"] = (all_records.hsic_lower <= truth) & (truth <= all_records.hsic_upper)
summary = {}
for method, b in all_records.groupby("method"):
    u, v, var = [b[name].to_numpy() for name in ("hsic_center", "hsic_v", "score_var")]
    pos = u > 0
    z_hsic = (u-truth) / (2*np.sqrt(var/120))
    old_center = np.sqrt(u[pos])
    old_radius = NormalDist().inv_cdf(.975) * np.sqrt(var[pos] / (120*u[pos]))
    old_covered = (np.maximum(0, old_center-old_radius) <= np.sqrt(truth)) & (np.sqrt(truth) <= old_center+old_radius)
    summary[method] = dict(n=len(b), covered_norm=int(b.norm_covered.sum()), covered_hsic=int(b.hsic_covered.sum()), coverage_norm=b.norm_covered.mean(), coverage_hsic=b.hsic_covered.mean(),
        norm_upper_below_truth=int((b.sqrt_upper < np.sqrt(truth)).sum()), norm_lower_above_truth=int((b.sqrt_lower > np.sqrt(truth)).sum()),
        hsic_upper_below_truth=int((b.hsic_upper < truth).sum()), hsic_lower_above_truth=int((b.hsic_lower > truth).sum()),
        historical_positive_u_covered=int(old_covered.sum()), positive_u_count=int(pos.sum()),
        updated_positive_u_covered=int(b.loc[pos, "norm_covered"].sum()),
        updated_nonpositive_u_covered=int(b.loc[~pos, "norm_covered"].sum()),
        nonpositive_u=int((u<=0).sum()), mean_u=u.mean(), mean_v=v.mean(), mean_v_minus_u=(v-u).mean(),
        empirical_u_sd=u.std(ddof=1), sqrt_mean_u_variance=np.sqrt(np.mean(4*var/120)),
        median_se_ratio_new_old_positive_u=np.median(np.sqrt(u[pos]/v[pos])),
        z_hsic_quantiles=np.quantile(z_hsic,[.025,.5,.975]).tolist(),
        correlation_u_hsic_se=np.corrcoef(u,2*np.sqrt(var/120))[0,1])
paired = all_records.pivot(index="rep", columns="method", values="norm_covered")
difference = paired.oracle_debiased.astype(int)-paired.fitted_debiased.astype(int)
gaussian_target = (1/np.sqrt(1+2*.35**2)+1/np.sqrt(1+2*.55**2)-2/np.sqrt(1+.35**2+.55**2))/8
report = dict(target=truth, target_norm=np.sqrt(truth), full_gaussian_target_lengthscale1=gaussian_target, rff_to_gaussian_target_ratio=truth/gaussian_target, quadrature_check=True, fitted_vector_gram_checks=[0,1,2],
    summaries=summary, paired_oracle_minus_fitted_debiased_coverage=difference.mean(), paired_mc_se=difference.std(ddof=1)/np.sqrt(500))
output = Path("tables/covariate/diagnosis")
output.mkdir(parents=True, exist_ok=True)
(output / "oracle_controls.json").write_text(json.dumps(report, indent=2) + "\n")
pd.DataFrame.from_dict(summary, orient="index").rename_axis("method").to_csv(output / "oracle_controls.csv")
print(json.dumps(report, indent=2))
