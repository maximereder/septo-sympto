"""Wraps an Ultralytics YOLO semantic-segmentation model into a :class:`Segmenter`.

Ultralytics is a framework with its own trainer, data format and checkpoints, so
it does not go through the segmenter registry; it satisfies the port through this
adapter, exactly as :mod:`septosympto.ports` anticipates.

The preprocessing is the one Ultralytics trains with: **RGB**, ``[0, 1]``, no
mean/std normalisation. The leaf arrives BGR (``cv2.imread`` convention, what
every other model in the project eats), so the channel swap happens here, once,
in the open. The v1 pipeline ran a BGR U-Net beside an RGB YOLO and relied on
each script remembering which was which; the adapter is what makes forgetting
impossible.

Only the raw ``nn.Module`` is used at inference — not the Ultralytics predictor —
so that the leaf goes through the project's own letterbox and comes back through
its inverse, on the same canvas the U-Net sees. A binary (``nc == 1``) semantic
head emits one logit map; the threshold is applied to its sigmoid.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from septosympto.letterbox import letterbox, unletterbox

YOLO_THRESHOLD = 0.5


class YoloSegmenter:
    """A binary YOLO semantic-segmentation module presented as a ``Segmenter``.

    The module must accept ``(N, 3, H, W)`` float32 in ``[0, 1]``, RGB, and return
    ``(N, 1, h, w)`` logits (or a tuple whose first element is), at any stride.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = YOLO_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        self.model = model.float().eval().to(device)
        self.threshold = threshold
        self.device = device

    @classmethod
    def from_weights(
        cls,
        weights: str | Path,
        threshold: float = YOLO_THRESHOLD,
        device: str = "cpu",
    ) -> YoloSegmenter:
        """Load an Ultralytics ``.pt`` checkpoint trained with task ``semantic``."""
        from ultralytics import YOLO

        yolo = YOLO(str(weights))
        if yolo.task != "semantic":
            raise ValueError(
                f"{weights} is a {yolo.task!r} model; YoloSegmenter needs a semantic "
                "segmentation checkpoint (yolo26*-sem)"
            )
        return cls(yolo.model, threshold=threshold, device=device)

    def probabilities(self, leaf_bgr: np.ndarray) -> np.ndarray:
        """Per-pixel necrosis probability, at the leaf's own resolution."""
        if leaf_bgr.ndim != 3 or leaf_bgr.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) BGR image, got {leaf_bgr.shape}")

        canvas, place = letterbox(leaf_bgr)
        rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
        batch = torch.from_numpy(rgb.transpose(2, 0, 1)[None].copy()).to(self.device)

        with torch.inference_mode():
            logits = self.model(batch)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.ndim != 4 or logits.shape[1] != 1:
                raise ValueError(
                    f"expected (N, 1, h, w) logits from a binary semantic head, "
                    f"got {tuple(logits.shape)}"
                )
            if logits.shape[-2:] != batch.shape[-2:]:
                logits = F.interpolate(
                    logits, size=batch.shape[-2:], mode="bilinear", align_corners=False
                )
            probabilities = torch.sigmoid(logits)

        probabilities = probabilities.squeeze().cpu().numpy()
        return unletterbox(probabilities, place, cv2.INTER_LINEAR)

    def segment(self, leaf_bgr: np.ndarray) -> np.ndarray:
        return self.probabilities(leaf_bgr) > self.threshold
