"""Write results and a run manifest.

The CSV header is derived from :class:`~septosympto.measure.LeafMeasurement`, so a
column can never be declared without a value behind it. v1 wrote a 23-column
header and 19 values per row on the ``--import`` path; deriving both from one
source makes that class of bug unrepresentable.

The manifest answers "which version produced this CSV", which v1 could not. It
records the code version, the git commit, the weights and their hash, the
parameters, and the run time. A results file without provenance is not
reproducible, and reproducibility is the point of a scientific tool.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

from septosympto import __version__
from septosympto.measure import LeafMeasurement


def write_csv(measurements: Iterable[LeafMeasurement], path: str | Path) -> int:
    """Write measurements to ``path``. Returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = LeafMeasurement.columns()
    n = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for measurement in measurements:
            writer.writerow(measurement.as_row())
            n += 1
    return n


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    weights: dict[str, str | Path],
    parameters: dict[str, object],
    n_scans: int,
    n_leaves: int,
    timestamp: str,
) -> dict[str, object]:
    """Assemble a run manifest. ``timestamp`` is passed in, never read from the clock here."""
    return {
        "septosympto_version": __version__,
        "git_commit": _git_commit(),
        "timestamp": timestamp,
        "weights": {
            role: {"path": str(path), "sha256": file_sha256(path)}
            for role, path in weights.items()
        },
        "parameters": parameters,
        "n_scans": n_scans,
        "n_leaves": n_leaves,
    }


def write_manifest(manifest: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
