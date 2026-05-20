"""MSR Action3D skeleton loading utilities."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MSR_FILENAME_RE = re.compile(
    r"^a(?P<action>\d{2})_s(?P<subject>\d{2})_e(?P<repetition>\d{2})"
    r"_skeleton(?P<real3d>3D)?\.txt$"
)


@dataclass(frozen=True)
class MSRSequence:
    """One raw MSR Action3D skeleton sequence."""

    sequence_id: str
    action_id: int
    subject_id: int
    repetition_id: int
    path: Path
    joints: np.ndarray
    confidence: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return int(self.joints.shape[0])


def parse_msr_filename(path: str | Path) -> dict[str, int | str | bool]:
    """Parse action, subject, repetition, and sequence id from an MSR filename."""
    name = Path(path).name
    match = MSR_FILENAME_RE.match(name)
    if not match:
        raise ValueError(f"Not an MSR Action3D skeleton filename: {name}")

    action_id = int(match.group("action"))
    subject_id = int(match.group("subject"))
    repetition_id = int(match.group("repetition"))
    sequence_id = f"a{action_id:02d}_s{subject_id:02d}_e{repetition_id:02d}"

    return {
        "sequence_id": sequence_id,
        "action_id": action_id,
        "subject_id": subject_id,
        "repetition_id": repetition_id,
        "is_real3d": bool(match.group("real3d")),
    }


def load_skeleton_file(path: str | Path, *, num_joints: int = 20) -> MSRSequence:
    """Load one MSR skeleton text file as ``T x J x 3`` coordinates."""
    path = Path(path)
    parsed = parse_msr_filename(path)
    raw = np.loadtxt(path, dtype=np.float64)

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"{path} must contain at least three coordinate columns.")
    if raw.shape[0] % num_joints != 0:
        raise ValueError(
            f"{path} has {raw.shape[0]} rows, not divisible by {num_joints} joints."
        )

    num_frames = raw.shape[0] // num_joints
    joints = raw[:, :3].reshape(num_frames, num_joints, 3)
    confidence = None
    if raw.shape[1] >= 4:
        confidence = raw[:, 3].reshape(num_frames, num_joints)

    return MSRSequence(
        sequence_id=str(parsed["sequence_id"]),
        action_id=int(parsed["action_id"]),
        subject_id=int(parsed["subject_id"]),
        repetition_id=int(parsed["repetition_id"]),
        path=path,
        joints=joints,
        confidence=confidence,
    )


def find_skeleton_files(raw_dir: str | Path, *, prefer_real3d: bool = True) -> list[Path]:
    """Find extracted MSR skeleton files under ``raw_dir``."""
    raw_dir = Path(raw_dir)
    patterns = ["*_skeleton3D.txt", "*_skeleton.txt"]
    if not prefer_real3d:
        patterns.reverse()

    for pattern in patterns:
        files = [path for path in raw_dir.rglob(pattern) if path.is_file()]
        if files:
            return sorted(files, key=_sequence_sort_key)
    return []


def ensure_skeleton_files(
    raw_dir: str | Path,
    *,
    prefer_real3d: bool = True,
) -> tuple[list[Path], str]:
    """Return skeleton files, extracting bundled RAR archives with ``bsdtar`` if needed."""
    raw_dir = Path(raw_dir)
    files = find_skeleton_files(raw_dir, prefer_real3d=prefer_real3d)
    if files:
        return files, "extracted"

    archive_options = [
        (
            raw_dir / "MSRAction3DSkeletonReal3D.rar",
            raw_dir / "MSRAction3DSkeletonReal3D",
        ),
        (
            raw_dir / "MSRAction3DSkeleton(20joints).rar",
            raw_dir / "MSRAction3DSkeleton20Joints",
        ),
    ]
    if not prefer_real3d:
        archive_options.reverse()

    for archive_path, extract_dir in archive_options:
        if not archive_path.exists():
            continue
        _extract_rar(archive_path, extract_dir)
        files = find_skeleton_files(raw_dir, prefer_real3d=prefer_real3d)
        if files:
            return files, archive_path.name

    expected = ", ".join(path.name for path, _ in archive_options)
    raise FileNotFoundError(
        f"No MSR skeleton text files found in {raw_dir}. "
        f"Expected extracted skeleton files or one of: {expected}."
    )


def load_msr_action3d(
    raw_dir: str | Path,
    *,
    num_joints: int = 20,
    prefer_real3d: bool = True,
) -> tuple[list[MSRSequence], str]:
    """Load all MSR Action3D skeleton sequences found in ``raw_dir``."""
    files, source = ensure_skeleton_files(raw_dir, prefer_real3d=prefer_real3d)
    sequences = [load_skeleton_file(path, num_joints=num_joints) for path in files]
    return sequences, source


def _extract_rar(archive_path: Path, extract_dir: Path) -> None:
    bsdtar = shutil.which("bsdtar")
    if bsdtar is None:
        raise RuntimeError(
            "bsdtar is required to extract the MSR Action3D RAR archive. "
            "Install libarchive/bsdtar or extract the archive manually."
        )

    extract_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [bsdtar, "-xf", str(archive_path), "-C", str(extract_dir)],
        check=True,
    )


def _sequence_sort_key(path: Path) -> tuple[int, int, int, str]:
    parsed = parse_msr_filename(path)
    return (
        int(parsed["action_id"]),
        int(parsed["subject_id"]),
        int(parsed["repetition_id"]),
        path.name,
    )
