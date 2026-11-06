# Efficient inference for additive-noise HSIC

Code and reproduction instructions for the paper's synthetic experiments.
All data are simulated; there is no external dataset to download.

| Results | Generation and analysis guide |
|---|---|
| Synthetic rejection, split tests, QQ plots, coverage, example datasets | [Synthetic experiments](SYNTHETIC_REPRODUCTION.md) |
| Causal direction, classical diagnostics, covariate adjustment | [Causal and covariate experiments](CAUSAL_COVARIATE_REPRODUCTION.md) |
| Cross-U intervals, operator/HSIC MSE, bandwidth sensitivity, witnesses | [Appendix experiments](APPENDIX_REPRODUCTION.md) |

[PAPER_RESULTS.md](PAPER_RESULTS.md) maps the paper results to their entry points.
Run commands from the repository root. Each guide separates generating data from
summarizing it; no single command launches all experiments.

## Environment

Use Python 3.10 in an isolated environment, for example:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
```

The requirements support CPU execution. For a GPU, install the JAX CUDA packages
appropriate to your system. The code enables float64. Exact figure styling also
requires the Liberation Sans font; the scripts report an error if it is missing.

Open notebooks in an editor using this Python environment, or execute one with:

```bash
python -m jupyter execute submission_with_covariate.ipynb --timeout=-1
```

## Quick check

```bash
python run_synthetic.py --config configs/smoke.yaml --output runs/smoke --platform cpu
JAX_PLATFORMS=cpu python -m unittest test_synthetic_reproduction test_result_loading test_causal_covariate test_robust_delta_reporting test_cross_u_benchmark test_cross_u_reporting test_witness_appendix
JAX_PLATFORMS=cpu PYTHONPATH=.:MSE_sensitivity python -m unittest discover -s MSE_sensitivity -p 'test_*.py'
```

Smoke settings reduce sample size, tuning, integration and resampling. They are
not paper settings. Full simulation grids require substantially more computation.
Run individual tasks or disjoint index ranges as described in the guides.

## Included results and conventions

`figures/` and `tables/` contain the paper outputs and numerical summaries.
The original full simulation checkpoints are not bundled. The guides explain
how to generate fresh repetitions and analyse those results. Some reported
synthetic summaries pool incomplete historical campaigns, so a fresh balanced
campaign need not reproduce their exact finite-run counts or estimates.
Numerical-library and hardware differences can also affect fitted models.

Reported delta confidence sets use U-centered squared-norm Wald intervals,
intersect them with the nonnegative half-line, and take square roots. Empty sets
remain in every denominator, contribute zero width and never cover the target.
The binary example has nominal-95% coverage 0.956 (plug-in) and 0.886 (debiased)
over 500 repetitions. The reported undercoverage is retained.

Use a new output directory for a changed configuration or implementation.
Saved pickle inputs should only be loaded from sources you trust.
