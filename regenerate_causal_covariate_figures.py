"""Redraw the six NeurIPS causal/covariate figures from their saved results.

Uses the readable submission notebooks as the canonical plotting definitions.
The small structural-fit illustration is deterministic; Monte Carlo fits are loaded.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="figure root; creates causal_arrow/ and toy_residual_hsic_with_covariate/")
    parser.add_argument("--covariate-results", type=Path, default=ROOT / "results/covariate")
    parser.add_argument("--causal-run", type=Path, default=ROOT / "results/toy_additive_noise_gof_1d/toy_additive_noise_gof_1d_20260502_233746.pkl.gz")
    args = parser.parse_args()
    output = args.output_dir.resolve()

    notebook = nbformat.read(ROOT / "submission_with_covariate.ipynb", as_version=4)
    namespace = {}
    plt.rcdefaults()
    exec(notebook.cells[1].source, namespace)
    namespace.update(LOAD_SAVED_RESULTS=True,
                     RESULT_DIR=args.covariate_results.resolve(),
                     FIGURE_DIR=output / "toy_residual_hsic_with_covariate")
    for index in (2, 3, 4, 5, 6, 8, 9, 10):
        if index == 9:
            frame = namespace["pd"].read_csv(namespace["RESULT_DIR"] / "delta_repetitions.csv")
            count = namespace["DELTA_N_REPS"]
            groups = frame.groupby(["estimator", "nominal_level"]).size()
            if (len(frame) != count * 2 * len(namespace["DELTA_NOMINAL_LEVELS"])
                    or not groups.eq(count).all()
                    or frame.duplicated(["rep", "estimator", "nominal_level"]).any()
                    or set(frame["rep"]) != set(range(count))):
                raise ValueError("Figure 3 requires the complete saved repetition cohort.")
        exec(compile(notebook.cells[index].source, f"covariate cell {index}", "exec"), namespace)
    plt.close("all")

    notebook = nbformat.read(ROOT / "submission_toy_additive_noise_gof_1d.ipynb", as_version=4)
    namespace = {}
    plt.rcdefaults()
    for index in (1, 3):
        exec(notebook.cells[index].source, namespace)
    namespace["CAUSAL_FIGURE_DIR"] = output / "causal_arrow"
    source = notebook.cells[23].source.replace("SAVED_RUN_PATH = None", f"SAVED_RUN_PATH = {str(args.causal_run.resolve())!r}")
    exec(source, namespace)
    for index in (11, 24):
        exec(compile(notebook.cells[index].source, f"causal cell {index}", "exec"), namespace)
    plt.close("all")
    print(f"Regenerated six paper figures under {output}")


if __name__ == "__main__":
    main()
