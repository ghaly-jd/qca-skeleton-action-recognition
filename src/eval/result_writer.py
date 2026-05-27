"""Result writing utilities with reproducibility metadata."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.paths import ensure_parent_dir


RESULT_FIELDNAMES = [
    "dataset",
    "method",
    "feature_mode",
    "seed",
    "parameters",
    "accuracy",
    "macro_f1",
    "runtime_sec",
    "git_commit",
    "timestamp",
]

#: Valid values for ``feature_mode``.
#: - ``"position"``         — raw joint positions (T × D); default and backward-compatible.
#: - ``"velocity"``         — first-order temporal differences (T × D).
#: - ``"acceleration"``     — second-order temporal differences (T × D).
#: - ``"position_velocity"``— position concatenated with velocity (T × 2D).
#: - ``"bone_vectors"``     — child-minus-parent joint offsets (T × n_bones*3).
#: - ``"bone_velocity"``    — bone vectors concatenated with their velocity.
VALID_FEATURE_MODES = (
    "position",
    "velocity",
    "acceleration",
    "position_velocity",
    "bone_vectors",
    "bone_velocity",
)


@dataclass(frozen=True)
class ResultRecord:
    """One experiment result row in the standard project schema."""

    dataset: str
    method: str
    seed: int | None
    #: Which feature representation was used. Default ``"position"`` for backward
    #: compatibility with existing result CSVs produced before this field existed.
    feature_mode: str = "position"
    parameters: dict[str, Any] = field(default_factory=dict)
    accuracy: float | None = None
    macro_f1: float | None = None
    runtime_sec: float | None = None
    git_commit: str | None = None
    timestamp: str | None = None

    def to_csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["parameters"] = json.dumps(row["parameters"], sort_keys=True)
        row["git_commit"] = row["git_commit"] or get_git_commit()
        row["timestamp"] = row["timestamp"] or utc_timestamp()
        return row


def utc_timestamp() -> str:
    """Return a compact UTC timestamp for result rows."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_git_commit() -> str:
    """Return the current git commit hash, or a stable placeholder."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    commit = result.stdout.strip()
    if not commit or commit == "HEAD":
        return "unknown"
    return commit


def append_result(path: str | Path, record: ResultRecord) -> Path:
    """Append one result row, creating the CSV header if needed."""
    path = ensure_parent_dir(path)
    file_exists = path.exists() and path.stat().st_size > 0

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record.to_csv_row())

    return path

