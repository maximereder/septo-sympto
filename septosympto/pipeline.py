"""Orchestration: scans in, per-leaf measurements out.

Written entirely against :mod:`septosympto.ports`. It never imports a model, so
swapping the U-Net for a YOLO segmentation head, or plugging in the pycnidia
counter once it exists, changes a constructor argument and nothing here.

The counter is optional. The necrosis segmenter is ported and available today;
the pycnidia point counter is being trained. With no counter, pycnidia columns
are zero and the rest of the pipeline is unaffected, so the tool is usable for
necrosis now rather than blocked on both models at once.

The unit of work is :func:`iter_analyses`, which yields a :class:`LeafAnalysis`
per leaf: the leaf, its crop, the necrosis mask, the pycnidia points, and the
measurement. Measuring is one consumer of that stream; rendering masks is
another. :func:`analyze_scan` is the measure-only shortcut.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from septosympto.leaf import Leaf, Scan, crop, find_leaves, load_scan
from septosympto.measure import (
    MIN_CIRCULARITY,
    MIN_LESION_AREA_MM2,
    LeafMeasurement,
    measure_leaf,
    points_in_necrosis,
)
from septosympto.ports import PointCounter, Segmenter


class MissingScaleError(ValueError):
    """Raised when a scan carries no resolution and none was supplied."""


@dataclass(frozen=True)
class LeafAnalysis:
    """Everything produced for one leaf: geometry, model outputs, and measurement.

    ``necrosis_patch`` is the mask at the crop's resolution, aligned with
    ``patch``, which is what a renderer wants. ``measurement`` is computed from
    the same mask lifted into full-scan coordinates.

    ``points`` holds every pycnidium the counter found, in patch coordinates.
    ``points_kept`` is ``None`` when all of them are counted, and otherwise a
    boolean per point saying which ones the measurement kept (those inside a
    necrotic lesion, under ``pycnidia_in_necrosis``).
    """

    leaf: Leaf
    patch: np.ndarray
    necrosis_patch: np.ndarray
    points: np.ndarray | None
    measurement: LeafMeasurement
    points_kept: np.ndarray | None = None


def _resolve_scale(scan: Scan, px_per_cm: float | None) -> float:
    scale = px_per_cm if px_per_cm is not None else scan.px_per_cm
    if scale is None:
        raise MissingScaleError(
            f"{scan.image}: no resolution in metadata; pass px_per_cm explicitly"
        )
    return scale


def iter_analyses(
    scan: Scan,
    segmenter: Segmenter,
    counter: PointCounter | None = None,
    *,
    px_per_cm: float | None = None,
    min_lesion_area_mm2: float = MIN_LESION_AREA_MM2,
    min_circularity: float = MIN_CIRCULARITY,
    min_leaf_area_px: int | None = None,
    pycnidia_in_necrosis: bool = False,
) -> Iterator[LeafAnalysis]:
    """Analyse every leaf on one scan, yielding a full record per leaf.

    Scale is taken from the scan metadata; ``px_per_cm`` overrides it, and is
    required when the scan carries none. Refusing to guess a scale keeps a silent
    unit error out of the results, which is where v1's magic ``472`` default hid.

    With ``pycnidia_in_necrosis``, only the pycnidia lying on a counted necrotic
    lesion enter the measurement; the others are kept in the record, flagged, so
    the overlay can show what was dropped.
    """
    scale = _resolve_scale(scan, px_per_cm)
    kwargs = {} if min_leaf_area_px is None else {"min_area_px": min_leaf_area_px}

    for leaf in find_leaves(scan, **kwargs):
        patch = crop(scan, leaf)
        necrosis_patch = segmenter.segment(patch)
        necrosis_full = _place(necrosis_patch, leaf, scan.bgr.shape[:2])
        points = None if counter is None else counter.count(patch)
        points_kept = None
        if pycnidia_in_necrosis and points is not None:
            x, y, w, h = leaf.bbox
            points_kept = points_in_necrosis(
                points,
                necrosis_patch & leaf.mask[y : y + h, x : x + w],
                scale,
                min_lesion_area_mm2=min_lesion_area_mm2,
                min_circularity=min_circularity,
            )
        counted = points if points_kept is None else np.asarray(points)[points_kept]
        measurement = measure_leaf(
            leaf,
            necrosis_full,
            counted,
            scale,
            min_lesion_area_mm2=min_lesion_area_mm2,
            min_circularity=min_circularity,
        )
        yield LeafAnalysis(leaf, patch, necrosis_patch, points, measurement, points_kept)


def analyze_scan(
    scan: Scan,
    segmenter: Segmenter,
    counter: PointCounter | None = None,
    **kwargs,
) -> list[LeafMeasurement]:
    """Measure every leaf on one scan. Measure-only shortcut over :func:`iter_analyses`."""
    return [analysis.measurement for analysis in iter_analyses(scan, segmenter, counter, **kwargs)]


def _place(patch_mask: np.ndarray, leaf: Leaf, scan_shape: tuple[int, int]) -> np.ndarray:
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
