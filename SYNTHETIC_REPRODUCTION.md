# Synthetic experiments

Use the environment in [README.md](README.md) and run from the repository root.

## Generate repetitions

`configs/synthetic.yaml` specifies the scalar Fourier model: n=250/500/1000,
five folds, mean smoothness 1/1.5/2, noise smoothness .25/.5/.75,
rho=0,.1,...,.5, and seeds 10--109. There are 16,200 fits and 1,000
bootstrap/permutation draws per fit. Both Fourier base frequencies are 1.
The remaining tuning grids are explicit in `RunConfig` in
`multivariate_crossfit_core.py`; every result saves its resolved configuration.

`configs/split.yaml` specifies n=250/500/1000/1500, mean smoothness 1,
noise smoothness .25, rho=0/.3, seeds 10--99, and training fractions .2/.4/.6/.8:
720 datasets with four fit/test choices per dataset.

```bash
python run_synthetic.py --config configs/synthetic.yaml \
  --output runs/multivariate_sweep/favourability_search_updated_paper --dry-run
python run_synthetic.py --config configs/synthetic.yaml \
  --output runs/multivariate_sweep/favourability_search_updated_paper --index 0 --limit 1
python run_synthetic.py --config configs/split.yaml \
  --output runs/multivariate_sweep/better_permutation_calibration_paper --index 0 --limit 1
```

For complete grids use `--limit 16200` and `--limit 720`, respectively.
Independent contiguous slices can run in separate processes using the same plan.
Completed results with matching configurations are skipped; conflicting plans
are rejected. Each output directory contains `planned_jobs.yaml` and one
`results.yaml` per completed fit, including estimates, intervals, tuning choices,
kernel bandwidths and population targets.

## Summarize and plot

After the required runs finish, execute `synthetic_paper_plots.ipynb` in order:

```bash
export HSIC_RUNS=runs/multivariate_sweep
export HSIC_OUTPUT_DIR=results/synthetic_analysis
export HSIC_SYNTHETIC_PATTERN=favourability_search_updated_paper
export HSIC_SPLIT_PATTERN=better_permutation_calibration_paper
python -m jupyter execute synthetic_paper_plots.ipynb --timeout=-1
```

The notebook writes figures, coverage tables, rejection summaries and a
compressed record of the analysed inputs under `HSIC_OUTPUT_DIR`. It does not
refit nuisance models. Set explicit input patterns to avoid pooling unrelated
campaigns. The included paper summaries pool historical campaigns and retain
100--110 repetitions per positive-rho coverage setting, 110--210 for the main
rejection/QQ settings, and 90--100 for split-test settings. Their raw fit archives
are not included. New balanced runs have the counts specified above.

## Statistical conventions

The reported squared-HSIC interval is
`[U - 2*z*sqrt(score_var/n), U + 2*z*sqrt(score_var/n)]`. Intersect with
`[0,infinity)` and square-root the endpoints. A negative upper bound produces
an empty set with zero width and noncoverage. All repetitions, including negative
U estimates, remain in the denominator. Missing or invalid inputs raise an error.
Only positive-rho delta coverage is reported as an alternative-coverage check;
null delta results do not establish a calibrated Gaussian approximation.

Coverage error bars use clipped normal/Wald binomial intervals. Rejection plots
use Wilson intervals. QQ plots centre the squared-HSIC U-statistic at population
HSIC and divide by the empirical standard deviation of these errors (ddof=1).

The illustrative DGP previews retain mean/noise frequencies 1.5/.5; the simulation
grids use 1/1. Population targets use numerical integration. A smoke run's small
integration grid is not a paper-quality target.

The current paper tables are `tables/synthetic/paper_table_*.tex`.
`tables/synthetic/historical_delta/` contains the earlier conditional-coverage
summaries for comparison only. Current results retain all repetitions.
