"""Run an explicit slice of a synthetic sweep on one machine; no scheduler required."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import yaml
from sweep_config import load_spec, expand_sweep_grid, build_run_name, save_yaml

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0, help="First zero-based job index")
    parser.add_argument("--limit", type=int, default=1, help="Number of jobs to run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--platform", choices=["cpu", "gpu"], default=None)
    args = parser.parse_args()
    spec = load_spec(args.config)
    configs = expand_sweep_grid(spec["sweep"])
    if args.index < 0 or args.limit < 1 or args.index + args.limit > len(configs):
        parser.error(f"Requested slice must lie within {len(configs)} jobs")
    varying = [k for k in spec["sweep"] if len({str(c[k]) for c in configs}) > 1]
    jobs = [{"run_name": build_run_name(i,c,varying), "config": c} for i,c in enumerate(configs)]
    plan = {"config_path": str(args.config), "num_jobs": len(jobs), "jobs": jobs}
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "planned_jobs.yaml"
    if plan_path.exists() and yaml.safe_load(plan_path.read_text()) != plan:
        raise ValueError("Output directory has a different plan; choose a new output directory.")
    save_yaml(plan_path, plan)
    print(f"{len(jobs)} planned jobs; requested indices {args.index}:{args.index+args.limit}", flush=True)
    if args.dry_run:
        return
    if args.platform:
        os.environ["JAX_PLATFORMS"] = args.platform
    from multivariate_crossfit_core import RunConfig, run_experiment
    for job in jobs[args.index:args.index+args.limit]:
        path = args.output / job["run_name"] / "results.yaml"
        if path.exists():
            saved = yaml.safe_load(path.read_text())
            if any(saved.get("config", {}).get(k) != v for k,v in job["config"].items()):
                raise ValueError(f"Existing result has different config: {path}")
            print(f"Already present: {path}", flush=True)
            continue
        print(f"Running {job['run_name']}", flush=True)
        results = run_experiment(RunConfig(**job["config"]))
        save_yaml(path, results)

if __name__ == "__main__":
    main()
