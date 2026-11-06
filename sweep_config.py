"""Read experiment grids and name runs for the sequential simulation runner."""
from __future__ import annotations
import ast
import itertools
import re
from pathlib import Path
from typing import Any
import yaml

RANGE_EXPR = re.compile(r"^\(?\s*range\((.*)\)\s*\)?$")
ATOMIC_LIST_SWEEP_KEYS = {"standard_residual_hsic_fit_prefixes"}


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def sweep_values_for_key(key: str, value: Any) -> list[Any]:
    if key in ATOMIC_LIST_SWEEP_KEYS and isinstance(value, list):
        return [value]
    return as_list(value)


def expand_sweep_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    match = RANGE_EXPR.fullmatch(value.strip())
    if match is None:
        return value

    args = ast.literal_eval("[" + match.group(1) + "]")
    if not isinstance(args, list) or not 1 <= len(args) <= 3:
        raise ValueError(f"range(...) expects 1 to 3 integer arguments, got: {value}")
    if not all(isinstance(arg, int) for arg in args):
        raise ValueError(f"range(...) expects integer arguments, got: {value}")
    return list(range(*args))


def expand_sweep_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    values = [sweep_values_for_key(key, grid[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def format_tag_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.4g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p").replace("/", "_")


def build_run_name(index: int, config: dict[str, Any], varying_keys: list[str]) -> str:
    # Keep names short and readable. Include only parameters that actually vary.
    tags = [f"{key}{format_tag_value(config[key])}" for key in varying_keys]
    suffix = "_".join(tags) if tags else "default"
    return f"cfg_{index:04d}_{suffix}"


def load_spec(path: Path) -> dict[str, Any]:
    """Read a YAML sweep, expanding explicit integer ranges."""
    with path.open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    spec["sweep"] = {key: expand_sweep_value(value)
                     for key, value in spec.get("sweep", {}).items()}
    return spec
