"""Turn a necrosis mask and a set of pycnidia points into per-leaf measurements.

This is the biological core: everything a results row reports is computed here,
from a leaf, a boolean necrosis mask at the leaf's resolution, and an array of
pycnidia coordinates. Segmentation and counting happen upstream; this module
does not know which model produced them.

Two v1 lesion filters are replaced. v1 kept components with ``area > 300`` px and
``perimeter / area < 0.9``. Both are resolution-dependent: 300 px is a different
lesion at 600 dpi than at 1200, and ``perimeter / area`` carries the dimension of
an inverse length, so its numeric value changes with scale. Here the area floor
is given in **mm²** and converted through the scan's own ``px_per_cm``, and shape
is filtered by **circularity** ``4*pi*A / P**2``, which is dimensionless and
transfers across scanners.

Pycnidia are points, not boxes. v1's YOLOv5 produced boxes and v1 reported a
pycnidia *area*; a point-regression model does not, so ``pycnidia_area`` is gone.
The reported quantities are the count and its densities, which is what the
biology uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import cv2
import numpy as np

from septosympto.leaf import Leaf

MIN_LESION_AREA_MM2 = 0.135
MIN_CIRCULARITY = 0.0
MM_PER_CM = 10.0


@dataclass(frozen=True)
class LeafMeasurement:
    image: str
    leaf_index: int
    px_per_cm: float
    qc: str

    leaf_area_px: int
    leaf_area_cm2: float

    necrosis_count: int
    necrosis_area_px: int
    necrosis_area_cm2: float
    necrosis_area_ratio: float

    pycnidia_count: int
    pycnidia_per_leaf_cm2: float
    pycnidia_per_necrosis_cm2: float

    @classmethod
    def columns(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def as_row(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _count_lesions(
    necrosis_mask: np.ndarray, min_area_px: float, min_circularity: float
) -> tuple[int, int]:
    """Lesions passing the area and shape filters, and their total pixel area.

    Area is a pixel count, from the connected component, so it is the same kind
    of number as ``leaf.area_px`` and the two can be divided into a ratio. Shape
    uses the component's external contour: circularity ``4*pi*A / P**2`` is
    dimensionless, unlike v1's ``perimeter / area``, so a thin-artefact filter set
    on one scanner transfers to another. ``cv2.contourArea`` is deliberately not
    used for the area itself — it integrates the polygon between pixel centres and
    would disagree with the pixel count by a border's worth of area.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        necrosis_mask.astype(np.uint8), connectivity=8
    )
    kept = 0
    total_area = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        if min_circularity > 0:
            component = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
            circularity = 4 * math.pi * area / perimeter**2 if perimeter > 0 else 0.0
            if circularity < min_circularity:
                continue
        kept += 1
        total_area += area
    return kept, total_area


def measure_leaf(
    leaf: Leaf,
    necrosis_mask: np.ndarray,
    pycnidia_points: np.ndarray | None,
    px_per_cm: float,
    min_lesion_area_mm2: float = MIN_LESION_AREA_MM2,
    min_circularity: float = MIN_CIRCULARITY,
) -> LeafMeasurement:
    """Measure one leaf. ``necrosis_mask`` and ``leaf.mask`` must share a shape."""
    if necrosis_mask.shape != leaf.mask.shape:
        raise ValueError(
            f"necrosis mask {necrosis_mask.shape} does not match leaf {leaf.mask.shape}"
        )

    necrosis_in_leaf = necrosis_mask & leaf.mask
    px_per_mm = px_per_cm / MM_PER_CM
    min_area_px = min_lesion_area_mm2 * px_per_mm**2
    necrosis_count, necrosis_area_px = _count_lesions(
        necrosis_in_leaf, min_area_px, min_circularity
    )

    leaf_area_px = leaf.area_px
    leaf_area_cm2 = leaf.area_cm2(px_per_cm)
    necrosis_area_cm2 = necrosis_area_px / px_per_cm**2

    pycnidia_count = 0 if pycnidia_points is None else int(len(pycnidia_points))

    return LeafMeasurement(
        image=leaf.image,
        leaf_index=leaf.leaf_index,
        px_per_cm=round(px_per_cm, 2),
        qc="|".join(leaf.qc),
        leaf_area_px=leaf_area_px,
        leaf_area_cm2=round(leaf_area_cm2, 4),
        necrosis_count=necrosis_count,
        necrosis_area_px=necrosis_area_px,
        necrosis_area_cm2=round(necrosis_area_cm2, 4),
        necrosis_area_ratio=round(necrosis_area_px / leaf_area_px, 4) if leaf_area_px else 0.0,
        pycnidia_count=pycnidia_count,
        pycnidia_per_leaf_cm2=round(pycnidia_count / leaf_area_cm2, 3) if leaf_area_cm2 else 0.0,
        pycnidia_per_necrosis_cm2=(
            round(pycnidia_count / necrosis_area_cm2, 3) if necrosis_area_cm2 else 0.0
        ),
    )
