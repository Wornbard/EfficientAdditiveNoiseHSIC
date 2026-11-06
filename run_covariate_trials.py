"""Generate explicit independent binary-covariate validation cohorts.

Run from the repository root in the project environment. Scalar checkpoints are
appended every 100 repetitions; matching, complete checkpoints can be resumed.
The DGP, fixed feature map, tuning grids and fold count come from the notebook.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import nbformat
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--data-seed-base", type=int, required=True)
    parser.add_argument("--fold-seed-base", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n < 8 or args.n % 4 or args.repetitions < 1:
        parser.error("Use n divisible by four and a positive repetition count.")
    if min(args.data_seed_base, args.fold_seed_base) < 0:
        parser.error("Seed bases must be nonnegative.")
    data_seeds = set(range(args.data_seed_base, args.data_seed_base + args.repetitions))
    fold_seeds = set(range(args.fold_seed_base, args.fold_seed_base + args.repetitions))
    if data_seeds & fold_seeds:
        parser.error("Data and fold seed ranges must be disjoint.")

    notebook_path = Path("submission_with_covariate.ipynb")
    notebook = nbformat.read(notebook_path, 4)
    namespace = {}
    for index in (1, 2, 3, 5, 8):
        exec(notebook.cells[index].source, namespace)
    namespace["N"] = args.n
    if namespace["N_FOLDS"] != 4:
        raise ValueError("This validation protocol requires four outer folds.")
    config = {
        "study": args.output.name, "n": args.n, "repetitions": args.repetitions,
        "data_seed_base": args.data_seed_base, "fold_seed_base": args.fold_seed_base,
        "regime": "x_scale", "num_folds": 4, "fold_size": args.n // 4,
        "training_size": 3 * args.n // 4,
        "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "feature_dimension": namespace["RFF_DIM"],
        "feature_lengthscale": namespace["RFF_LENGTH_SCALE"],
        "feature_seed": namespace["RFF_SEED"],
        "mean_max_frequency": namespace["MAX_FREQ"],
        "mean_amplitude": namespace["MEAN_AMPLITUDE"],
        "noise_scales": namespace["sigma_values_for_regime"]("x_scale"),
    }
    # JSON round-trip normalizes tuples before comparing a resumed configuration.
    config = json.loads(json.dumps(config))
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output has a different configuration; use a new directory.")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    output = args.output / "repetitions.csv"
    fields = ["study", "N", "n", "rep", "data_seed", "fold_seed", "estimator",
              "hsic_center", "hsic_v", "score_var", "true_hsic",
              "mean_fit_signal_mse", "se_hsic"]
    start = 0
    if output.exists():
        saved = pd.read_csv(output)
        start = len(saved) // 2
        if len(saved) % 2 or start > args.repetitions:
            raise ValueError("Incomplete or oversized repetition checkpoint.")
        expected = [(rep, method) for rep in range(start) for method in ("plug-in", "debiased")]
        if list(zip(saved.rep, saved.estimator)) != expected:
            raise ValueError("Checkpoint does not contain consecutive paired repetitions.")
        for column, value in (("N", args.n), ("n", args.n), ("study", args.output.name)):
            if not (saved[column] == value).all():
                raise ValueError(f"Checkpoint mismatch in {column}.")
        if not np.array_equal(saved.data_seed, args.data_seed_base + saved.rep):
            raise ValueError("Checkpoint data seeds differ from configuration.")
        if not np.array_equal(saved.fold_seed, args.fold_seed_base + saved.rep):
            raise ValueError("Checkpoint fold seeds differ from configuration.")
    truth = namespace["population_oracle_hsic"]("x_scale")
    z = NormalDist().inv_cdf(.975)
    pending = []
    for rep in range(start, args.repetitions):
        data_seed, fold_seed = args.data_seed_base + rep, args.fold_seed_base + rep
        x, t, y, mean, sigma = namespace["simulate"](args.n, regime="x_scale", seed=data_seed)
        fitted = namespace["estimate_hsic_u"](x, t, y, mean, sigma, seed=fold_seed)
        for estimator, key in (("plug-in", "naive_H_folds"), ("debiased", "debiased_H_folds")):
            blocks = fitted[key]
            if len(blocks) != 4 or any(block.shape != (args.n//4, args.n//4)
                                       for row in blocks for block in row):
                raise ValueError("Unexpected fold Gram dimensions.")
            ci = namespace["delta_sqrt_hsic_ci_from_blocks"](blocks, z)
            pending.append({
                "study": args.output.name, "N": args.n, "n": args.n, "rep": rep,
                "data_seed": data_seed, "fold_seed": fold_seed, "estimator": estimator,
                "hsic_center": ci["hsic_center"], "hsic_v": ci["hsic_v"],
                "score_var": ci["score_var"], "true_hsic": truth,
                "mean_fit_signal_mse": fitted["mean_fit_signal_mse"],
                "se_hsic": ci["se_hsic"],
            })
        if (rep + 1) % 100 == 0 or rep + 1 == args.repetitions:
            exists = output.exists()
            with output.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if not exists:
                    writer.writeheader()
                writer.writerows(pending)
            pending.clear()
            print(f"{args.output.name}: {rep+1}/{args.repetitions} paired repetitions", flush=True)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
