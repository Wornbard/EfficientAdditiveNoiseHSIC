"""Illustrate full witness estimates and the population slices' density meaning.

No models are fitted:
the saved N=500 run supplies the DGP and the original median kernel bandwidths,
and the previously validated witness arrays supply the two empirical maps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.integrate import trapezoid
from scipy.special import ndtr

from make_witness_appendix import population_witness, witness_source
from multivariate_crossfit_core import RunConfig, build_toy_state, eps_std_fn
from run_witness_n_rho_diagnostics import gaussian_numpy, quadrature_geometry

ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures/synthetic/witness_slices"
RESULT_DIR = ROOT / "results/witness_slices"
SOURCE = ROOT / "runs/multivariate_sweep/favourability_search_updated_paper"
STEM = "witness_slices_m_smoothness_1_eps_smoothness_0p75_rho_0p5_N500"
ESTIMATES = ROOT / "results/witness_appendix/noise_0p75.npz"
X_SLICES = np.array([-1.5, 0.5, 2.5])
R_GRID = np.linspace(-2.0, 2.0, 801)


def normal_densities(r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Zero-mean Gaussian densities; rows index scales, columns residuals."""
    return (np.exp(-0.5 * (r[None, :] / sigma[:, None])**2)
            / (np.sqrt(2 * np.pi) * sigma[:, None]))


def density_slices(run: dict, geometry: dict, x: np.ndarray, r: np.ndarray,
                   conditional_sigma: np.ndarray) -> dict[str, np.ndarray]:
    """Compare marginal, point-conditional, and kernel-local noise laws.

    For positive Gaussian k, local_weights define a probability mixture.
    The witness is a_x times the residual-kernel embedding of local minus
    marginal noise. Gaussian convolution verifies this identity independently
    of the existing operator-evaluation implementation.
    """
    weights = geometry["w"]
    joint_weights = gaussian_numpy(x, geometry["u"], run["alpha_x"]) * weights
    mass = joint_weights.sum(axis=1)
    local_weights = joint_weights / mass[:, None]
    densities = normal_densities(r, geometry["sigma"])
    # ell(r,t)=exp(-(r-t)^2/alpha_r) is sqrt(pi*alpha_r) times a
    # Gaussian density with variance alpha_r/2.
    smoothed = normal_densities(r, np.sqrt(geometry["sigma"]**2 + run["alpha_r"] / 2))
    marginal_smoothed = weights @ smoothed
    local_smoothed = local_weights @ smoothed
    witness = (mass[:, None] * np.sqrt(np.pi * run["alpha_r"])
               * (local_smoothed - marginal_smoothed))
    return {"marginal": weights @ densities,
            "conditional": normal_densities(r, conditional_sigma),
            "local": local_weights @ densities,
            "marginal_smoothed": marginal_smoothed, "local_smoothed": local_smoothed,
            "witness": witness, "kernel_mass": mass, "local_weights": local_weights}


def plot_slices(values: dict[str, np.ndarray]) -> None:
    """Show marginal and kernel-averaged noise above each population slice."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "figure.dpi": 140, "savefig.dpi": 300, "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "semibold",
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.65), sharex=True, sharey="row")
    for col, x in enumerate(X_SLICES):
        ax = axes[0, col]
        ax.plot(R_GRID, values["marginal"], color="#555555", ls="--", lw=1.6,
                label=r"Marginal $p_R$")
        ax.plot(R_GRID, values["local"][col], color="#d66032", lw=1.8,
                label=r"Kernel-averaged $p_{R;k,x}$")
        ax.set_title(rf"$x={x:g}$", pad=5)
        ax.set_ylim(bottom=0)
        ax = axes[1, col]
        witness = values["witness"][col]
        ax.axhline(0, color="#888888", lw=0.7)
        ax.fill_between(R_GRID, 0, witness, where=witness >= 0, interpolate=True,
                        color="#d66032", alpha=0.3, linewidth=0)
        ax.fill_between(R_GRID, 0, witness, where=witness <= 0, interpolate=True,
                        color="#2765b0", alpha=0.3, linewidth=0)
        ax.plot(R_GRID, witness, color="#252525", lw=1.5)
        ax.set_xlabel(r"residual $r$")
        ax.set(xlim=(-2, 2), xticks=[-2, -1, 0, 1, 2])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        for row in range(2):
            axes[row, col].grid(alpha=0.25)
    limit = 1.1 * np.max(np.abs(values["witness"]))
    axes[1, 0].set_ylim(-limit, limit)
    axes[0, 0].set_ylabel("noise density")
    axes[1, 0].set_ylabel(r"witness $w(x,r)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.54, 1.02),
               fontsize=8.5, columnspacing=1.5, handlelength=2.5)
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.13, top=0.85,
                        wspace=0.18, hspace=0.32)
    for extension in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{STEM}.{extension}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_full_maps(x: np.ndarray, r: np.ndarray, witnesses: dict[str, np.ndarray]) -> None:
    """Compare all three operators with a common scale and marked slice locations."""
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65), sharex=True, sharey=True,
                             layout="constrained")
    limit = max(np.max(np.abs(witness)) for witness in witnesses.values())
    for ax, method, title in zip(axes, ("population", "oracle", "debiased"),
                                ("Population", "Oracle mean", "Debiased")):
        mesh = ax.pcolormesh(x, r, witnesses[method].T, shading="auto", cmap="RdBu_r",
                             vmin=-limit, vmax=limit, rasterized=True)
        for location in X_SLICES:
            ax.axvline(location, color="#333333", ls="--", lw=0.7)
            if method == "population":
                ax.text(location, 1.7, rf"${location:g}$", ha="center", va="center",
                        fontsize=7, bbox={"facecolor": "white", "edgecolor": "none",
                                          "alpha": 0.85, "pad": 1})
        ax.set(xlabel=r"covariate $x$", title=title,
               xlim=(-3, 3), ylim=(-2, 2), xticks=[-3, 0, 3], yticks=[-2, 0, 2])
        ax.grid(False)
    axes[0].set_ylabel(r"residual $r$")
    colorbar = fig.colorbar(mesh, ax=axes.tolist(), pad=0.02, fraction=0.025)
    colorbar.set_label("witness value")
    colorbar.formatter.set_powerlimits((-2, 2))
    colorbar.formatter.set_useMathText(True)
    colorbar.update_ticks()
    for extension in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{STEM}_full_2d.{extension}",
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    global SOURCE, ESTIMATES, RESULT_DIR, FIGURE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help="Synthetic sweep directory, or seed 10 n=500 rho=.5 noise-smoothness .75 results.yaml")
    parser.add_argument("--estimates", type=Path, default=ESTIMATES)
    parser.add_argument("--results-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    SOURCE, ESTIMATES = args.source.resolve(), args.estimates.resolve()
    if SOURCE.is_dir():
        SOURCE = witness_source(SOURCE, .75)
    RESULT_DIR, FIGURE_DIR = args.results_dir.resolve(), args.figures_dir.resolve()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(SOURCE.read_text())
    config = RunConfig(**raw["config"])
    state = build_toy_state(config, jax.random.PRNGKey(config.random_key))
    run = {"config": config, "state": state, "alpha_x": raw["bandwidths"]["alpha_x"],
           "alpha_r": raw["bandwidths"]["alpha_xi"]}
    sigma = np.asarray(eps_std_fn(state, config, jnp.asarray(X_SLICES[:, None])))[:, 0]
    previous = None
    for order in (256, 512, 1024, 2048):
        geometry = quadrature_geometry(run, order)
        values = density_slices(run, geometry, X_SLICES, R_GRID, sigma)
        checked_keys = ("marginal", "local", "witness")
        if previous is not None and all(np.allclose(values[k], previous[k], rtol=1e-6, atol=1e-9)
                                        for k in checked_keys):
            break
        previous = values
    else:
        raise RuntimeError("Noise-density and witness quadrature did not converge.")
    np.testing.assert_allclose(values["witness"],
        population_witness(run, geometry, X_SLICES, R_GRID), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(values["local_weights"].sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(values["witness"], values["witness"][:, ::-1], atol=1e-14)
    # Check these are slices of the existing plotted operator, with the same kernels.
    saved_manifest = json.loads((ESTIMATES.parent / "manifest.json").read_text())
    saved_run = next(row for row in saved_manifest["runs"] if row["eps_smoothness"] == config.eps_smoothness)
    assert saved_run["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with np.load(ESTIMATES) as old:
        empirical_maps = {method: old[method].copy() for method in ("oracle", "debiased")}
        saved_x, saved_r = old["x"].copy(), old["r"].copy()
        indices = [int(np.argmin(np.abs(old["x"]-x))) for x in X_SLICES]
        np.testing.assert_allclose(old["x"][indices], X_SLICES, atol=1e-14)
        np.testing.assert_allclose(population_witness(run, geometry, X_SLICES, old["r"]),
                                   old["population"][indices], rtol=1e-5, atol=2e-8)
    # Independent numerical convolution of the actual plotted density contrast.
    t = np.linspace(-12 * geometry["sigma"].max(), 12 * geometry["sigma"].max(), 12001)
    dense = density_slices(run, geometry, X_SLICES, t, sigma)
    for key in ("marginal", "conditional", "local"):
        np.testing.assert_allclose(trapezoid(dense[key], t, axis=-1), 1, atol=1e-10)
    r_check = np.array([-1.0, 0.0, 0.5])
    convolution = np.array([
        trapezoid((dense["local"]-dense["marginal"])
                     * np.exp(-(r-t)**2 / run["alpha_r"]), t, axis=-1)
        for r in r_check]).T * values["kernel_mass"][:, None]
    np.testing.assert_allclose(convolution,
        population_witness(run, geometry, X_SLICES, r_check), rtol=1e-9, atol=1e-12)
    map_x, map_r = np.linspace(-3, 3, 121), R_GRID[::5]
    population_map = population_witness(run, geometry, map_x, map_r)
    np.testing.assert_allclose(map_x, saved_x, atol=1e-14)
    np.testing.assert_allclose(map_r, saved_r, atol=1e-14)
    for witness in empirical_maps.values():
        assert witness.shape == population_map.shape and np.isfinite(witness).all()
    slice_indices = [int(np.argmin(np.abs(map_x-x))) for x in X_SLICES]
    np.testing.assert_allclose(map_x[slice_indices], X_SLICES, atol=1e-14)
    np.testing.assert_allclose(population_map[slice_indices], values["witness"][:, ::5],
                               rtol=1e-10, atol=1e-14)
    plot_slices(values)
    plot_full_maps(map_x, map_r, {"population": population_map, **empirical_maps})
    np.savez_compressed(RESULT_DIR / "slices.npz", x=X_SLICES, r=R_GRID,
                        sigma=sigma, map_x=map_x, map_r=map_r, population_map=population_map,
                        oracle_map=empirical_maps["oracle"], debiased_map=empirical_maps["debiased"],
                        **{k: v for k, v in values.items() if k != "local_weights"})
    window_mass = 2 * ndtr(2 / geometry["sigma"]) - 1
    manifest = {"N": config.N, "rho": config.rho, "f_smoothness": config.f_smoothness,
                "eps_smoothness": config.eps_smoothness, "random_key": config.random_key,
                "x_slices": X_SLICES.tolist(), "conditional_sigma": sigma.tolist(),
                "alpha_x": run["alpha_x"], "alpha_r": run["alpha_r"],
                "quadrature_order": order, "slices_population_only": True,
                "full_map_methods": ["population", "oracle", "debiased"],
                "estimates_source": str(ESTIMATES.resolve()),
                "estimates_sha256": hashlib.sha256(ESTIMATES.read_bytes()).hexdigest(),
                "displayed_marginal_probability": float(geometry["w"] @ window_mass),
                "source": str(SOURCE.resolve()),
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "checks": ["quadrature convergence", "density normalization",
                           "independent density convolution equals witness",
                           "agreement with existing population map", "residual symmetry",
                           "full 2D map agrees with plotted slices"]}
    (RESULT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
