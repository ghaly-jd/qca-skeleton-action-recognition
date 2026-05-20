"""Small IO helpers used by scripts and experiment code."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.utils.paths import ensure_parent_dir


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    path = ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, sort_keys=True)
        handle.write("\n")
    return path


def read_yaml(path: str | Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML config files.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_csv_rows(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> Path:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []

    path = ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path

