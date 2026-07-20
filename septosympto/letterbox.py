"""The one leaf-to-canvas transform, shared by dataset regeneration and inference.

Training images and inference must see a leaf in exactly the same geometry, or
the model meets a distribution at test time it never trained on. So the letterbox
lives here, once: a leaf is scaled 1:1 (never stretched) onto a fixed
``CANVAS_H x CANVAS_W`` canvas, centred, the rest left as background. Predictions
are mapped back to native pixels through the inverse of the very same placement.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CANVAS_W, CANVAS_H = 3072, 384
BACKGROUND = 255


@dataclass(frozen=True)
class Placement:
    """Where a ``w x h`` leaf sits on the canvas: uniform ``scale``, offset, size."""

    scale: float
    ox: int
    oy: int
    nw: int
    nh: int
    w: int
    h: int


def plan(w: int, h: int) -> Placement:
    """Uniform-scale placement of a ``w x h`` leaf onto the canvas, centred."""
    s = min(CANVAS_W / w, CANVAS_H / h, 1.0)
    nw, nh = int(round(w * s)), int(round(h * s))
    return Placement(s, (CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2, nw, nh, w, h)


def letterbox(img: np.ndarray, background: int = BACKGROUND):
    """Place an already-backgrounded leaf crop on the canvas; return canvas + placement."""
    h, w = img.shape[:2]
    p = plan(w, h)
    shape = (CANVAS_H, CANVAS_W, img.shape[2]) if img.ndim == 3 else (CANVAS_H, CANVAS_W)
    canvas = np.full(shape, background, img.dtype)
    canvas[p.oy : p.oy + p.nh, p.ox : p.ox + p.nw] = cv2.resize(img, (p.nw, p.nh))
    return canvas, p


def unletterbox(canvas: np.ndarray, p: Placement, interpolation: int) -> np.ndarray:
    """Inverse of :func:`letterbox` for a raster: canvas -> native ``h x w``."""
    region = canvas[p.oy : p.oy + p.nh, p.ox : p.ox + p.nw]
    return cv2.resize(region, (p.w, p.h), interpolation=interpolation)


def unletterbox_points(points: np.ndarray, p: Placement) -> np.ndarray:
    """Inverse of :func:`letterbox` for ``(N, 2)`` canvas-pixel points -> native pixels."""
    out = points.astype(np.float64).copy()
    out[:, 0] = (out[:, 0] - p.ox) / p.scale
    out[:, 1] = (out[:, 1] - p.oy) / p.scale
    return out
