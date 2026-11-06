# Paper result map

Use artifact names and LaTeX labels to identify results; table numbering can
change as the manuscript is edited. All data-generating mechanisms are synthetic.

| Paper result | Entry point | Saved paper outputs |
|---|---|---|
| Main rejection curves, split fit-test comparison, QQ plots, coverage tables, DGP previews | `run_synthetic.py`, then `synthetic_paper_plots.ipynb` | `figures/synthetic/`, `tables/synthetic/paper_table_*.tex` and coverage/rejection CSVs |
| Covariate-adjusted coverage, mean fits and oracle comparison | `submission_with_covariate.ipynb` | `figures/toy_residual_hsic_with_covariate/`, `tables/covariate/delta_summary.csv` |
| Causal-direction rejection, example fits and p-value distributions | `submission_toy_additive_noise_gof_1d.ipynb` | `figures/causal_arrow/`, `tables/causal_arrow/` |
| Classical diagnostic comparison (`tab:classical-baselines`) | `run_toy_additive_noise_selected_1000.py`, then `summarize_toy_additive_noise_selected_1000.py` | `tables/toy_additive_noise_classical_diagnostics/` |
| Cross-U width and zero inclusion | `run_cross_u_benchmark.py`, then `make_cross_u_paper_tables.py` | `tables/synthetic/cross_u_comparison/n250/` |
| Operator/HSIC MSE and bandwidth sensitivity (`tab:debiasing-mse-median`, `fig:debiasing-bandwidth`) | `MSE_sensitivity/run_sensitivity.py`, `summarize.py`, `make_paper_summary.py` | `tables/synthetic/debiasing_mse/`, `figures/synthetic/debiasing_mse/` |
| Witness slices and full maps | `make_witness_appendix.py`, then `make_witness_slices.py` | `figures/synthetic/witness_slices/` |

Generation commands, sample sizes, repetition counts and analysis conventions
are given in the three reproduction guides linked from [README.md](README.md).
The included figure/table outputs do not include the full historical simulation
checkpoints. Additional diagnostic comparisons are retained under
`tables/covariate/` and `tables/synthetic/`; they are not replacements for the
paper's reported U-centered squared-norm intervals.
