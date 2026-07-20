"""Reading scans, finding leaves on them, and measuring what was found.

Three things are fixed here relative to v1.

**Scale is read, not asserted.** v1 took ``--pixels_for_cm`` on the command line,
defaulting to 472 while its own README documented 145. The reference scans are
1200 dpi TIFFs, so 1200 / 2.54 = 472.44 px/cm. That number is in the file.

**Leaf area is counted, not summed over contours.** v1 summed ``contourArea``
over every contour returned by ``RETR_TREE``, which adds internal holes instead
of subtracting them.

**Clipped leaves are flagged.** On the reference scans every leaf spans the full
scan width and both tips fall outside the image. Measuring a standardised leaf
segment is the intended protocol, but a silent measurement is not the same thing
as a declared one, and scan widths vary by 13 % across the reference set.

A note on the tissue mask. v1 called it an HSV "green" threshold with bounds
``[0, 35, 65]`` to ``[255, 255, 255]``. OpenCV stores hue on 0-179 for 8-bit
images, so an upper bound of 255 accepts every hue: the hue channel was never
filtering anything. What v1 actually applied was ``saturation >= 35 and
value >= 65``. Since value is the channel maximum and saturation is
``(max - min) / max``, both are invariant to channel order, which is why v1's
``COLOR_RGB2HSV`` call on BGR data was harmless. The behaviour is preserved here
under its real name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MIN_LEAF_AREA_PX = 50_000
MIN_SATURATION = 35
MIN_VALUE = 65
CM_PER_INCH = 2.54

TIFF_X_RESOLUTION = 282
TIFF_RESOLUTION_UNIT = 296
UNIT_INCH = 2
UNIT_CM = 3


@dataclass(frozen=True)
class Leaf:
    """One leaf found on a scan, with its provenance and quality flags."""

    image: str
    leaf_index: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    qc: tuple[str, ...]

    @property
    def leaf_id(self) -> str:
        """For display only. Nothing parses this back: scan names contain underscores."""
        return f"{self.image}_{self.leaf_index}"

    @property
    def area_px(self) -> int:
        return int(self.mask.sum())

    def area_cm2(self, px_per_cm: float) -> float:
        return self.area_px / px_per_cm**2


@dataclass(frozen=True)
class Scan:
    image: str
    bgr: np.ndarray
    px_per_cm: float | None

    @property
    def height(self) -> int:
        return self.bgr.shape[0]

    @property
    def width(self) -> int:
        return self.bgr.shape[1]


def read_px_per_cm(path: str | Path) -> float | None:
    """Pixels per centimetre, from image metadata. ``None`` when not recorded."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            tags = getattr(im, "tag_v2", None)
            if tags is not None and TIFF_X_RESOLUTION in tags:
                resolution = float(tags[TIFF_X_RESOLUTION])
                unit = int(tags.get(TIFF_RESOLUTION_UNIT, UNIT_INCH))
                if unit == UNIT_CM:
                    return resolution
                if unit == UNIT_INCH:
                    return resolution / CM_PER_INCH
                return None
            dpi = im.info.get("dpi")
            if dpi:
                return float(dpi[0]) / CM_PER_INCH
    except (OSError, ValueError, TypeError):
        return None
    return None


def load_scan(path: str | Path) -> Scan:
    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image: {path}")
    return Scan(image=path.stem, bgr=bgr, px_per_cm=read_px_per_cm(path))


def tissue_mask(bgr: np.ndarray) -> np.ndarray:
    """Leaf tissue against the scanner background: saturated enough, bright enough."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    return (saturation >= MIN_SATURATION) & (value >= MIN_VALUE)


def _qc_flags(mask: np.ndarray, bbox: tuple[int, int, int, int], scan: Scan) -> tuple[str, ...]:
    x, y, w, h = bbox
    flags = []
    if mask[:, 0].any():
        flags.append("clipped-left")
    if mask[:, scan.width - 1].any():
        flags.append("clipped-right")
    if mask[0, :].any():
        flags.append("clipped-top")
    if mask[scan.height - 1, :].any():
        flags.append("clipped-bottom")
    if h > w:
        flags.append("taller-than-wide")
    return tuple(flags)


def find_leaves(scan: Scan, min_area_px: int = MIN_LEAF_AREA_PX) -> list[Leaf]:
    """Leaves on a scan, ordered top to bottom, indexed from 1.

    Connected components rather than contours. Filling an external contour would
    swallow the holes inside a leaf, and summing ``contourArea`` over ``RETR_TREE``
    contours — what v1 did — adds them. A labelled component is the set of tissue
    pixels itself, so a hole is simply absent from it.

    Ordering by vertical position makes ``leaf_index`` reproducible across runs.
    OpenCV's contour order is an implementation detail and must not leak into an
    identifier that ends up in a results table.
    """
    tissue = tissue_mask(scan.bgr).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(tissue, connectivity=8)

    found = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] <= min_area_px:
            continue
        bbox = (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        )
        found.append((bbox, labels == label))

    found.sort(key=lambda item: item[0][1])
    return [
        Leaf(
            image=scan.image,
            leaf_index=i,
            bbox=bbox,
            mask=mask,
            qc=_qc_flags(mask, bbox, scan),
        )
        for i, (bbox, mask) in enumerate(found, start=1)
    ]


def crop(scan: Scan, leaf: Leaf, background: int = 255) -> np.ndarray:
    """The leaf at native resolution, everything else set to ``background``.

    Interior holes in the mask — specular glare on the blade, small debris — are
    filled first, so those pixels keep their real values instead of being punched
    to ``background``. This matches how the training crops are cut.
    """
    from scipy import ndimage

    x, y, w, h = leaf.bbox
    patch = scan.bgr[y : y + h, x : x + w].copy()
    solid = ndimage.binary_fill_holes(leaf.mask[y : y + h, x : x + w])
    patch[~solid] = background
    return patch


def resize_anisotropy(leaf: Leaf, target_h: int, target_w: int) -> float:
    """How much a fixed-size resize would distort this leaf, as an axis ratio.

    1.0 means the leaf already has the target proportions. On the reference
    scans the median is 1.23 and the maximum 1.94, so a circular pycnidium can
    come out nearly twice as tall as it is wide. The distortion depends only on
    leaf thickness, which is a varietal trait.
    """
    _, _, w, h = leaf.bbox
    stretch_x, stretch_y = target_w / w, target_h / h
    return max(stretch_x, stretch_y) / min(stretch_x, stretch_y)
