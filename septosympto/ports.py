"""The two contracts the pipeline depends on.

Everything downstream of a model — measurement, CSV export, evaluation — is
written against these, never against a concrete architecture. That is what makes
a U-Net, a YOLO segmentation head and a point-regression network comparable:
the same evaluation code drives all of them.

Deliberately not an ``nn.Module`` base class. Ultralytics owns its training loop,
its data format and its checkpoints; it is a framework, not an architecture. It
satisfies :class:`Segmenter` through an adapter, like anything else.

Each implementation owns its own preprocessing and must declare it. The v1
pipeline ran a U-Net expecting **BGR** (``cv2.imread``, never swapped) next to a
YOLOv5 expecting **RGB** (converted by ``AutoShape``), in the same script. Put
those two behind a shared interface without care and you will silently feed one
of them the wrong channel order.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Segmenter(Protocol):
    """Turns a leaf image into a boolean mask of necrotic tissue."""

    def segment(self, leaf_bgr: np.ndarray) -> np.ndarray:
        """Take an ``(H, W, 3)`` uint8 BGR leaf, return an ``(H, W)`` bool mask."""
        ...


@runtime_checkable
class PointCounter(Protocol):
    """Locates pycnidia as points rather than boxes.

    The median annotated bounding box in the reference dataset is 4 x 4 px, so a
    box encodes a point and three noisy numbers. Counting models regress points.
    """

    def count(self, leaf_bgr: np.ndarray) -> np.ndarray:
        """Take an ``(H, W, 3)`` uint8 BGR leaf, return an ``(N, 2)`` array of ``(x, y)``."""
        ...
