# Appendix experiments

Use the environment in [README.md](README.md). First generate the synthetic
sweep described in [SYNTHETIC_REPRODUCTION.md](SYNTHETIC_REPRODUCTION.md), then set:

```bash
export HSIC_PROJECT_ROOT="$PWD"
SOURCE=runs/multivariate_sweep/favourability_search_updated_paper
```

The source must contain `planned_jobs.yaml` and the corresponding `results.yaml`
files. The scripts replay observations and selected nuisance hyperparameters;
they do not treat residuals as observed. Use new output directories for new runs.

## Cross-U intervals

The comparison uses n=250, five folds, mean smoothness 1, noise smoothness
.25/.5/.75, rho=0,.1,...,.5 and seeds 10--109: 1,800 paired repetitions.

```bash
python run_cross_u_benchmark.py plan --source "$SOURCE" --output runs/paper_cross_u
for task in $(seq 0 17); do
  python run_cross_u_benchmark.py run --output runs/paper_cross_u --task "$task"
done
python run_cross_u_benchmark.py summarize --output runs/paper_cross_u --metrics-only
python make_cross_u_paper_tables.py --runs runs/paper_cross_u
```

There are 18 tasks, one per (noise smoothness, rho), each with 100 repetitions.
Task commands can be run independently. Plans reject missing source runs;
reports reject missing settings/seeds or undefined primary intervals.
The outputs include `paper_table.tex`, separate noise panels, a summary CSV,
replicate metrics and a width/zero-inclusion figure.

A/B splits are balanced within nuisance folds. The delta comparator uses the
U-centered squared-norm interval followed by intersection and square root;
RTI uses the V center. The reported probability is **zero inclusion**, not target
coverage. Empty sets have zero width and exclude zero. The normal approximation
at the null is descriptive. All 1,800 repetitions remain in the summaries.
Legacy refresh commands are only needed for older checkpoint formats.

## MSE and bandwidth sensitivity

The full cohort uses n=250/500/1000, mean smoothness 1, noise smoothness
.25/.5/.75, rho=0,.1,...,.5 and seeds 10--109. It has 5,400 paired repetitions
and 25 bandwidth pairs. The paper selects noise smoothness .5, with median
results at all three sample sizes and heatmaps at n=500.

```bash
python MSE_sensitivity/run_sensitivity.py plan --source "$SOURCE" --output runs/paper_mse
```

The manifest lists missing source indices and task counts. If source runs are
missing, complete every `prepare` task first. Then complete baseline tasks
0--269, followed by sweep tasks 0--1079. Run the frozen script recorded in the
output directory, for example:

```bash
python runs/paper_mse/code_snapshot/run_sensitivity.py prepare --output runs/paper_mse --task 0
python runs/paper_mse/code_snapshot/run_sensitivity.py baseline --output runs/paper_mse --task 0
python runs/paper_mse/code_snapshot/run_sensitivity.py sweep --output runs/paper_mse --task 0
```

These are individual task examples, not commands for the entire grid. Omit
`prepare` if all source runs exist. Each baseline task handles 20 seeds and each
sweep task handles five. After all tasks complete:

```bash
python runs/paper_mse/code_snapshot/summarize.py --output runs/paper_mse
python MSE_sensitivity/make_paper_summary.py --data runs/paper_mse
```

This writes the median MSE/RTI-coverage table, six bandwidth heatmaps, their CSVs,
and provenance under `tables/synthetic/debiasing_mse/` and
`figures/synthetic/debiasing_mse/`. For saved complete runs, only these summary
commands are needed. Use `--tables-dir` and `--figures-dir` to preserve earlier
outputs. The LaTeX figure include file uses `\DebiasingResultsRoot`; define it
as the path to this repository (or `.` if copying the output folders).

See [MSE_sensitivity/README.md](MSE_sensitivity/README.md) for population-target,
paired-bootstrap and uncertainty conventions. Raw historical checkpoints are
not bundled; fresh numerical fits can differ from the included paper summaries.

## Witness illustrations

The fixed paper setting is seed 10, n=500, rho=.5, mean smoothness 1 and noise
smoothness .75. The first command also generates .25/.5 examples:

```bash
python make_witness_appendix.py --source "$SOURCE"
python make_witness_slices.py --source "$SOURCE"
```

The first writes fitted witness arrays and a manifest under
`results/witness_appendix/`. The second uses those arrays and writes figures
under `figures/synthetic/witness_slices/`. Both select the scientific settings
from the source manifest, independently of directory numbering. The second also
accepts an explicit `results.yaml` path.

Both accept `--results-dir` and `--figures-dir`. When changing the first result
directory, pass its `noise_0p75.npz` to the second command with `--estimates`.
Checks cover quadrature convergence, symmetry, density normalization and an
independent density convolution. The bandwidth figures require Liberation Sans;
the witness figures use DejaVu Sans.
