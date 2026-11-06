from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import pickle

import yaml


_CACHE_SCHEMA_VERSION = 1


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _flatten_scalars(obj: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    if out is None:
        out = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_scalars(value, next_prefix, out)
    elif _is_scalar(obj):
        out[prefix] = obj

    return out


def _results_fingerprint(yaml_paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in yaml_paths
    )


def _cache_path(experiment_dir: Path, include_raw: bool, keep_columns: tuple[str, ...] | None = None) -> Path:
    suffix = "raw" if include_raw else "flat"
    if keep_columns:
        digest = hashlib.sha1("\0".join(keep_columns).encode("utf-8")).hexdigest()[:12]
        suffix = f"{suffix}_{digest}"
    return experiment_dir / f".results_cache_{suffix}.pkl"


def _load_cached_rows(
    cache_path: Path,
    fingerprint: tuple[tuple[str, int, int], ...],
    include_raw: bool,
    keep_columns: tuple[str, ...] | None,
) -> list[dict[str, Any]] | None:
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
        return None

    if not isinstance(cached, dict):
        return None
    if cached.get("schema_version") != _CACHE_SCHEMA_VERSION:
        return None
    if cached.get("include_raw") != include_raw:
        return None
    if cached.get("keep_columns") != keep_columns:
        return None
    # A checkout may be addressed through a different relative/absolute path.
    # Within this experiment directory the job name identifies each result.
    def local_fingerprint(entries):
        return tuple((Path(name).parent.name, mtime, size) for name, mtime, size in entries)

    if local_fingerprint(cached.get("fingerprint", ())) != local_fingerprint(fingerprint):
        return None

    rows = cached.get("rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        run_dir = cache_path.parent / row["run_name"]
        row["run_dir"] = str(run_dir)
        row["results_path"] = str(run_dir / "results.yaml")
    return rows


def _write_cached_rows(
    cache_path: Path,
    fingerprint: tuple[tuple[str, int, int], ...],
    include_raw: bool,
    keep_columns: tuple[str, ...] | None,
    rows: list[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "include_raw": include_raw,
        "keep_columns": keep_columns,
        "fingerprint": fingerprint,
        "rows": rows,
    }
    try:
        with cache_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:
        pass


def load_experiment_rows(
    experiment_dir: str | Path,
    include_raw: bool = True,
    use_cache: bool = True,
    write_cache: bool = True,
    keep_columns: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    # One row per results.yaml file. Rows are flattened on scalar leaves so
    # they are easy to filter, but the raw nested object is kept too by default.
    experiment_dir = Path(experiment_dir)
    yaml_paths = sorted(experiment_dir.glob("**/results.yaml"))
    keep_columns_tuple = tuple(sorted(set(keep_columns))) if keep_columns else None
    base_columns = ("results_path", "run_dir", "run_name")
    retained_columns = set(base_columns)
    if keep_columns_tuple:
        retained_columns.update(keep_columns_tuple)

    if use_cache:
        fingerprint = _results_fingerprint(yaml_paths)
        cache_path = _cache_path(experiment_dir, include_raw, keep_columns_tuple)
        cached_rows = _load_cached_rows(cache_path, fingerprint, include_raw, keep_columns_tuple)
        if cached_rows is not None:
            return cached_rows
        # Extra summary columns need not force a reread of every YAML result.
        # Reuse only an unfiltered cache whose full fingerprint still matches.
        if keep_columns_tuple:
            full_cache = _cache_path(experiment_dir, include_raw)
            cached_rows = _load_cached_rows(full_cache, fingerprint, include_raw, None)
            if cached_rows is not None:
                return [
                    {key: value for key, value in row.items() if key in retained_columns}
                    for row in cached_rows
                ]
    else:
        fingerprint = ()

    rows = []

    for yaml_path in yaml_paths:
        raw = yaml.safe_load(yaml_path.read_text()) or {}
        flat = _flatten_scalars(raw)

        row = {
            "results_path": str(yaml_path),
            "run_dir": str(yaml_path.parent),
            "run_name": yaml_path.parent.name,
        }

        # Make config parameters easy to access with bare names like row["eps_smoothness"].
        config = raw.get("config", {})
        if isinstance(config, dict):
            for key, value in config.items():
                if _is_scalar(value):
                    row[key] = value

        row.update(flat)

        if include_raw:
            row["raw"] = raw

        if keep_columns_tuple:
            row = {key: value for key, value in row.items() if key in retained_columns}

        rows.append(row)

    if use_cache and write_cache:
        _write_cached_rows(cache_path, fingerprint, include_raw, keep_columns_tuple, rows)

    return rows


def select_rows(rows: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    # Example: select_rows(rows, eps_smoothness=0.1, dimy=2)
    selected = []

    for row in rows:
        keep = True
        for key, target in filters.items():
            value = row.get(key, row.get(f"config.{key}"))
            if value != target:
                keep = False
                break
        if keep:
            selected.append(row)

    return selected
