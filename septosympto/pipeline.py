"""Orchestration: scans in, per-leaf measurements out.

Written entirely against :mod:`septosympto.ports`. It never imports a model, so
swapping the U-Net for a YOLO segmentation head, or plugging in the pycnidia
counter once it exists, changes a constructor argument and nothing here.

The counter is optional. The necrosis segmenter is ported and available today;
the pycnidia point counter is being trained. With no counter, pycnidia columns
are zero and the rest of the pipeline is unaffected, so the tool is usable for
necrosis now rather than blocked on both models at once.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from septosympto.leaf import Scan, crop, find_leaves, load_scan
from septosympto.measure import (
    MIN_CIRCULARITY,
    MIN_LESION_AREA_MM2,
    LeafMeasurement,
    measure_leaf,
)
from septosympto.ports import PointCounter, Segmenter


class MissingScaleError(ValueError):
    """Raised when a scan carries no resolution and none was supplied."""


def analyze_scan(
    scan: Scan,
    segmenter: Segmenter,
    counter: PointCounter | None = None,
    *,
    px_per_cm: float | None = None,
    min_lesion_area_mm2: float = MIN_LESION_AREA_MM2,
    min_circularity: float = MIN_CIRCULARITY,
    min_leaf_area_px: int | None = None,
) -> list[LeafMeasurement]:
    """Measure every leaf on one scan.

    Scale is taken from the scan metadata; ``px_per_cm`` overrides it, and is
    required when the scan carries none. Refusing to guess a scale keeps a silent
    unit error out of the results, which is where v1's magic ``472`` default hid.
    """
    scale = px_per_cm if px_per_cm is not None else scan.px_per_cm
    if scale is None:
        raise MissingScaleError(
            f"{scan.image}: no resolution in metadata; pass px_per_cm explicitly"
        )

    kwargs = {} if min_leaf_area_px is None else {"min_area_px": min_leaf_area_px}
    measurements = []
    for leaf in find_leaves(scan, **kwargs):
        patch = crop(scan, leaf)
        necrosis_full = _place(segmenter.segment(patch), leaf, scan.bgr.shape[:2])
        points = None if counter is None else counter.count(patch)
        measurements.append(
            measure_leaf(
                leaf,
                necrosis_full,
                points,
                scale,
                min_lesion_area_mm2=min_lesion_area_mm2,
                min_circularity=min_circularity,
            )
        )
    return measurements


def _place(patch_mask: np.ndarray, leaf, scan_shape: tuple[int, int]) -> np.ndarray:
    """Lift a leaf-patch mask back into full-scan coordinates."""
    x, y, w, h = leaf.bbox
    full = np.zeros(scan_shape, bool)
    full[y : y + h, x : x + w] = patch_mask
    return full


def analyze_paths(
    paths: Iterable[str | Path],
    segmenter: Segmenter,
    counter: PointCounter | None = None,
    **kwargs,
) -> Iterator[LeafMeasurement]:
    """Measure a sequence of scan files, yielding measurements as they are produced."""
    for path in paths:
        scan = load_scan(path)
        yield from analyze_scan(scan, segmenter, counter, **kwargs)


def iter_scan_files(directory: str | Path, extension: str = ".tif") -> list[Path]:
    """Scan files in a directory, sorted, matching ``extension`` case-insensitively."""
    directory = Path(directory)
    suffix = extension.lower()
    return sorted(p for p in directory.iterdir() if p.suffix.lower() == suffix)
