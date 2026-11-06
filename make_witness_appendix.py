"""Reproduce three fixed-seed appendix witness examples (no estimation losses).

Saved nuisance
hyperparameters are replayed; no CV, bootstrap, or seed selection is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from run_witness_n_rho_diagnostics import (
    gaussian_embedding, gaussian_numpy, quadrature_geometry, restore_run,
    verify_saved_statistics,
)

ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures/synthetic/witness_appendix"
RESULT_DIR = ROOT / "results/witness_appendix"
SOURCE_DIR = ROOT / "runs/multivariate_sweep/favourability_search_updated_paper"
SMOOTHNESS = (0.25, 0.5, 0.75)
METHODS = ("population", "oracle", "debiased")
LABELS = ("Population", "Oracle mean", "Debiased")
X_GRID = np.linspace(-3.0, 3.0, 121)
R_GRID = np.linspace(-2.0, 2.0, 161)
STEM = "witness_m_smoothness_1_rho_0p5_N500"


def witness_source(directory: Path, smoothness: float) -> Path:
    """Select the paper's prespecified seed and setting from a sweep manifest."""
    planned = yaml.load((directory / "planned_jobs.yaml").read_text(), Loader=yaml.CSafeLoader)
    selected = []
    for job in planned["jobs"]:
        config = job["config"]
        if all(config[key] == value for key, value in {
            "N": 500, "random_key": 10, "rho": .5, "f_smoothness": 1.,
            "eps_smoothness": smoothness, "num_folds": 5, "dimx": 1, "dimy": 1,
        }.items()):
            selected.append(directory / job["run_name"] / "results.yaml")
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one seed-10 paper witness configuration; found {len(selected)}")
    if not selected[0].is_file():
        raise FileNotFoundError(f"Generate the planned witness run first: {selected[0]}")
    return selected[0]


def population_witness(run: dict, geometry: dict, x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Integrate the Gaussian residual analytically, and X by quadrature."""
    kernel = gaussian_numpy(x, geometry["u"], run["alpha_x"])
    embedding = gaussian_embedding(r, geometry, run["alpha_r"]).T
    weights = geometry["w"]
    return ((kernel * weights) @ embedding
            - (kernel @ weights)[:, None] * (weights @ embedding)[None, :])


def checked_population(run: dict) -> tuple[np.ndarray, int]:
    """Check numerical integration, without computing estimation error."""
    previous = None
    for order in (256, 512, 1024, 2048):
        geometry = quadrature_geometry(run, order)
        witness = population_witness(run, geometry, X_GRID, R_GRID)
        if previous is not None and np.allclose(witness, previous, rtol=1e-5, atol=2e-8):
            np.testing.assert_allclose(witness, witness[:, ::-1], atol=1e-12)
            np.testing.assert_allclose(geometry["population_norm_sq"],
                                       run["raw"]["population_oracle"]["hsic"],
                                       rtol=2e-6, atol=2e-10)
            return witness, order
        previous = witness
    raise RuntimeError(f"Population quadrature did not converge: {run['results_path']}")


def empirical_witnesses(run: dict, x: np.ndarray, r: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate the oracle and debiased cross-fitted operators on test features.

    Both feature means use the training complement of the evaluation fold.
    The derivative is with respect to the training residual (first argument).
    """
    kernel_x = gaussian_numpy(run["x"], x, run["alpha_x"])
    kernel_oracle = gaussian_numpy(run["eps"], r, run["alpha_r"])
    witnesses = {method: np.zeros((len(x), len(r))) for method in METHODS[1:]}
    for fold, mask in enumerate(run["masks"]):
        centered_x = kernel_x[mask] - kernel_x[~mask].mean(axis=0)
        residual = run["residuals"][fold]
        kernel_r = gaussian_numpy(residual, r, run["alpha_r"])
        derivative = (-2 * (residual[~mask, None] - r[None, :])
                      / run["alpha_r"] * kernel_r[~mask])
        corrected = (kernel_r[mask] - kernel_r[~mask].mean(axis=0)
                     - residual[mask, None] * (run["weights"][fold] @ derivative))
        oracle = kernel_oracle[mask] - kernel_oracle[~mask].mean(axis=0)
        witnesses["oracle"] += centered_x.T @ oracle / run["config"].N
        witnesses["debiased"] += centered_x.T @ corrected / run["config"].N
    return witnesses


def plot_witnesses(examples: list[dict], stem: str) -> None:
    """Match the paper typography and use one symmetric colour scale per row."""
    rows = len(examples)
    fig, axes = plt.subplots(rows, 3, figsize=(7.0, 2.05 * rows + 0.35),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    for row, example in enumerate(examples):
        limit = max(float(np.max(np.abs(example[m]))) for m in METHODS)
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            mesh = ax.pcolormesh(X_GRID, R_GRID, example[method].T, shading="auto",
                                 cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
            ax.set(xticks=[-3, 0, 3], yticks=[-2, 0, 2], xlim=(-3, 3), ylim=(-2, 2))
            ax.grid(False)
            if row == 0:
                ax.set_title(LABELS[col], pad=6)
            if col == 0:
                ax.set_ylabel(rf"$s_\varepsilon={example['smoothness']:g}$" + "\n" + r"residual $r$")
            if row == rows - 1:
                ax.set_xlabel(r"$x$")
        colorbar = fig.colorbar(mesh, ax=axes[row].tolist(), fraction=0.025, pad=0.025)
        colorbar.set_label(r"$w(x,r)$")
        colorbar.ax.tick_params(labelsize=8)
        colorbar.formatter.set_powerlimits((-2, 2))
        colorbar.update_ticks()
    for extension in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{stem}.{extension}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    global SOURCE_DIR, RESULT_DIR, FIGURE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    SOURCE_DIR, RESULT_DIR, FIGURE_DIR = args.source.resolve(), args.results_dir.resolve(), args.figures_dir.resolve()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "figure.dpi": 140, "savefig.dpi": 300, "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "semibold",
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    examples, provenance = [], []
    baseline_config = None
    baseline_x = None
    for smoothness in SMOOTHNESS:
        slug = str(smoothness).replace(".", "p")
        path = witness_source(SOURCE_DIR, smoothness)
        run = restore_run(str(path))
        config = vars(run["config"]).copy()
        assert config.pop("eps_smoothness") == smoothness
        if baseline_config is None:
            baseline_config, baseline_x = config, run["x"].copy()
        assert config == baseline_config, "Only noise smoothness should change."
        np.testing.assert_array_equal(run["x"], baseline_x)
        verify_saved_statistics(run)
        population, order = checked_population(run)
        witnesses = {"population": population, **empirical_witnesses(run, X_GRID, R_GRID)}
        for witness in witnesses.values():
            assert np.isfinite(witness).all()
        examples.append({"smoothness": smoothness, **witnesses})
        np.savez_compressed(RESULT_DIR / f"noise_{slug}.npz", x=X_GRID, r=R_GRID, **witnesses)
        provenance.append({"eps_smoothness": smoothness, "source": str(path.resolve()),
                           "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "alpha_x": run["alpha_x"], "alpha_r": run["alpha_r"],
                           "quadrature_order": order})
        plot_witnesses(examples[-1:], f"{STEM}_eps_smoothness_{slug}")
        print(f"Validated and plotted noise smoothness {smoothness}", flush=True)
    plot_witnesses(examples, f"{STEM}_eps_smoothness_0p25_0p5_0p75_combined")
    manifest = {"N": 500, "rho": 0.5, "f_smoothness": 1, "random_key": 10,
                "num_folds": 5, "methods": METHODS, "normalization": "none",
                "selection": "Fixed seed 10 from the previous gallery; no error-based selection.",
                "kernel": "exp(-(a-b)^2/alpha); saved median bandwidths, shared within each row",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "runs": provenance}
    (RESULT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
