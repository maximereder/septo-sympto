"""Turn a leaf analysis into images a human can check.

Two outputs per leaf: the raw binary necrosis mask, and an overlay that draws the
necrosis outline (and pycnidia points, once a counter is wired) on the leaf crop.
The overlay is what you look at to decide whether the model is sensible; the mask
is what you keep for the record.

Necrosis is shown only where it falls inside the leaf silhouette, matching what
the measurement counts. Drawing the raw model output instead would show necrosis
over the blanked-out background and disagree with the numbers.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from septosympto.pipeline import LeafAnalysis

NECROSIS_COLOR = (0, 255, 0)
PYCNIDIA_COLOR = (255, 0, 255)


def _leaf_silhouette_patch(analysis: LeafAnalysis) -> np.ndarray:
    x, y, w, h = analysis.leaf.bbox
    return analysis.leaf.mask[y : y + h, x : x + w]


def necrosis_mask_image(analysis: LeafAnalysis) -> np.ndarray:
    """Binary necrosis mask (uint8, 0 or 255), clipped to the leaf."""
    necrosis = analysis.necrosis_patch & _leaf_silhouette_patch(analysis)
    return (necrosis.astype(np.uint8)) * 255


def overlay_image(analysis: LeafAnalysis, thickness: int = 2) -> np.ndarray:
    """The leaf crop with necrosis outlined and pycnidia marked."""
    canvas = analysis.patch.copy()
    necrosis = (analysis.necrosis_patch & _leaf_silhouette_patch(analysis)).astype(np.uint8)
    contours, _ = cv2.findContours(necrosis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, NECROSIS_COLOR, thickness)

    if analysis.points is not None:
        for x, y in np.asarray(analysis.points, dtype=int):
            cv2.circle(canvas, (int(x), int(y)), 3, PYCNIDIA_COLOR, 1)

    return canvas


def save_analysis(analysis: LeafAnalysis, directory: str | Path, *, masks: bool = True) -> None:
    """Write ``<image>_<index>_overlay.jpg`` and, if ``masks``, ``_mask.png``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{analysis.leaf.image}_{analysis.leaf.leaf_index}"
    cv2.imwrite(str(directory / f"{stem}_overlay.jpg"), overlay_image(analysis))
    if masks:
        cv2.imwrite(str(directory / f"{stem}_mask.png"), necrosis_mask_image(analysis))
