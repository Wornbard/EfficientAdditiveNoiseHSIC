# Causal-direction and covariate-adjusted experiments

Use the environment in [README.md](README.md). Run from the repository root.

## Causal-direction figures

Execute `submission_toy_additive_noise_gof_1d.ipynb` in order, with
`RUN_CORE_REPETITIONS=True` and `RUN_FAST_REPETITIONS=False`. The configuration
cell specifies n=250, five outer folds, 100 repetitions, rho=(0,.35),
399 permutations/bootstrap draws and seed 12345. The fast path is a different
experiment. The DGP draws X uniformly on [-2,2]; both directions use each dataset.
Split baselines train on 20%, 40%, 60% and 80% of observations.

The notebook writes a timestamped payload, JSON and repetition CSVs under
`results/toy_additive_noise_gof_1d/`, and figures under `figures/causal_arrow/`.
To redraw an existing run, execute the imports and the two code cells under
**Load And Visualize Saved Runs**, setting `SAVED_RUN_PATH` to your payload.
The one-draw illustration also uses the **Data** and **Forward And Reverse Fits**
cells. A `None` path means the run saved earlier in the notebook, not the latest
file on disk. The original paper payload is not bundled.

## Covariate-adjusted figures

Execute `submission_with_covariate.ipynb` in order. This CPU experiment generates
all three covariate figures: fold-specific mean fits, feasible/oracle U-statistics,
and confidence-interval coverage/width. It uses n=120, four outer folds,
120 repetitions per regime for the U-statistic comparison, and 500 alternative
repetitions for the confidence intervals.

Figures are written under `figures/toy_residual_hsic_with_covariate/`.
Per-repetition CSVs and summaries are written under `results/covariate/`.
Set `LOAD_SAVED_RESULTS=True` to redraw those CSVs after generating them once.
Changing model, sample-size, feature or seed settings requires regeneration.

The mean and derivative nuisance use a 22-column trigonometric feature design,
with ridge penalties selected by three-fold training-set cross-validation.
The residual kernel has 16 fixed random Fourier features with Gaussian length
scale 1; both estimators and the analytic oracle use the same feature kernel.

Confidence sets use the raw U-statistic and squared-norm standard error
`2*sqrt(score_var/n)`. Intersect the squared-norm Wald interval with `[0,infinity)`
and take square roots. Empty sets have zero width and noncoverage; every
repetition is retained. At nominal 95%, the paper's 500-repetition coverages are
0.956 (plug-in) and 0.886 (debiased), with mean widths 0.0728610 and 0.0430726.

`run_covariate_trials.py --help` documents optional independent sample-size and
repetition checks. These are separate experiments from the paper figure.

## Classical diagnostic comparison

The final table uses 1,000 fresh repetitions, IDs 100--1099, with n=250,
five folds, rho=(0,.35) and 399 HSIC resamples:

```bash
python run_toy_additive_noise_selected_1000.py --print-config
python run_toy_additive_noise_selected_1000.py --start 100 --stop 1100
python summarize_toy_additive_noise_selected_1000.py
```

The runner reads the DGP and test functions from stable notebook cell IDs;
keep those IDs unchanged. All eight methods share fitted residuals within each
scenario. It checkpoints repetitions and skips valid existing results.
For parallel execution, use consistent `--shard-index i --shard-count k`
arguments or disjoint start/stop ranges. The full experiment is substantial.

The summarizer checks completeness, pairing, configuration and p-values and
writes CSV, Markdown and LaTeX summaries under
`tables/toy_additive_noise_classical_diagnostics/`.
HSIC rejects at p<=.05; classical diagnostics use p<.05. Error bars are pointwise
Clopper--Pearson intervals. The debiased null rejection estimate is .086;
this finite-sample departure from .05 is retained.

`toy_additive_noise_classical_diagnostics.ipynb` contains the preliminary
100-repetition configuration search. It is not needed for the fixed final table.
To repeat it, generate the causal run, set `BASELINE_RUN_JSON` in its configuration
cell to the resulting JSON, and run in order.
