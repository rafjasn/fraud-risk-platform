"""JSON, state paths, and integrity checks shared by jobs and serving."""

import hashlib
import json
import os
import re
from pathlib import Path


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def identity(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def state_root() -> Path:
    return Path(os.environ.get("STATE_DIR", "state"))


def work_root(state_dir: Path) -> Path:
    """Scratch space for snapshot building: `WORK_DIR`, else inside the state directory.

    On AWS the state directory is a shared network file system, which is slow for
    DuckDB scratch files, so tasks point `WORK_DIR` at local disk.
    """
    return Path(os.environ.get("WORK_DIR") or state_dir / "work")


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}", value):
        raise ValueError("Invalid artifact identifier")
    return value


def verify_files(directory: Path, checksums: dict) -> None:
    for name, digest in checksums.items():
        if Path(name).name != name or sha256(directory / name) != digest:
            raise ValueError(f"Checksum mismatch or unsafe filename: {name}")
