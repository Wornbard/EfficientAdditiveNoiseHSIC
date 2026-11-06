# Paired MSE and bandwidth sensitivity

See [the appendix reproduction guide](../APPENDIX_REPRODUCTION.md) for commands.
`run_sensitivity.py` replays saved nuisance fits and shares the observations,
folds and mean fits across bandwidth pairs. A changed residual bandwidth causes
its derivative nuisance to be retuned. The Gaussian kernel is
`exp(-distance_squared / alpha)`; length-scale factor `c` multiplies alpha by
`c**2`. Each bandwidth has its own population target.

Operator squared error is the actual squared RKHS error, using analytic Gaussian
noise expectations and converged quadrature over X. HSIC squared error uses the
signed U-statistic. RTI coverage concerns the population square root of HSIC,
not zero inclusion. Delta results at rho=0 are descriptive only.

The retained historical cohort has 5400 paired repetitions (100 seeds in each
of 54 settings) and 25 bandwidth pairs. The paper uses noise smoothness 0.5:
median-bandwidth results at N=250,500,1000 and bandwidth heatmaps at N=500.
The broader cohort is retained so the paper summary can check against the
original complete analysis. This is substantial computation; use scheduled
resources for full reruns. No jobs are submitted automatically by these scripts.

`plan` freezes source checksums and code in the selected output directory.
Run the frozen copy for baseline/sweep tasks. Existing valid checkpoints are
reused. Missing planned source runs have explicit `prepare` tasks. Reports fail
on incomplete cohorts rather than silently discarding failed repetitions.

Paper uncertainty: 10000 paired percentile-bootstrap samples (seed 20261101)
for losses and relative gains, pointwise Wilson intervals for coverage.
Relative gain is 100*(1 - mean(debiased loss)/mean(plugin loss)), not a mean of
per-repetition ratios. Full results include denominators and provenance.

The paper heatmaps require Liberation Sans regular/bold text fonts and DejaVu
Sans math. The plotter registers system Liberation Sans files explicitly because
shared Matplotlib caches can otherwise choose DejaVu Sans silently, changing the
figure bounding box. See `APPENDIX_REPRODUCTION.md` for exact-layout regeneration
and validation commands.
