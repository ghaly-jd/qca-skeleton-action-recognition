"""Project-relative path helpers."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str | Path) -> Path:
    """Return an absolute path under the project root."""
    return PROJECT_ROOT.joinpath(*map(Path, parts))


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path and return the file path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

