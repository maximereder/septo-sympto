"""Wraps a plain ``nn.Module`` into a :class:`~septosympto.ports.Segmenter`.

The adapter owns the preprocessing, and states it. The necrosis U-Net was
trained on images read with ``cv2.imread``, which returns **BGR**, and no channel
swap was ever applied, so BGR is what it expects. A model trained on RGB needs a
different adapter, not a flag that someone will forget to set.

``segment`` returns a mask at the leaf's own resolution. The model works at a
fixed 304 x 3072, but the caller measures areas in native pixels, so the mask is
resized back on the way out. v1 instead measured in the distorted space and
rescaled the resulting area by the ratio of the two areas. The arithmetic is
equivalent; returning a native-resolution mask means nothing downstream has to
know the model's input size.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

NECROSIS_THRESHOLD = 0.8


class TorchSegmenter:
    """A binary segmentation ``nn.Module`` presented as a ``Segmenter``.

    The module must accept ``(N, 3, H, W)`` float32 in ``[0, 1]``, BGR, and return
    ``(N, 1, H, W)`` **logits**.
    """

    def __init__(
        self,
        model: nn.Module,
        imgsz: tuple[int, int] = (304, 3072),
        threshold: float = NECROSIS_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        self.model = model.eval().to(device)
        self.imgsz = imgsz
        self.threshold = threshold
        self.device = device

    @classmethod
    def from_safetensors(
        cls,
        weights: str | Path,
        model: nn.Module,
        imgsz: tuple[int, int] = (304, 3072),
        threshold: float = NECROSIS_THRESHOLD,
        device: str = "cpu",
    ) -> TorchSegmenter:
        from safetensors.torch import load_file

        model.load_state_dict(load_file(str(weights)))
        return cls(model, imgsz=imgsz, threshold=threshold, device=device)

    def probabilities(self, leaf_bgr: np.ndarray) -> np.ndarray:
        """Per-pixel necrosis probability, at the leaf's own resolution."""
        if leaf_bgr.ndim != 3 or leaf_bgr.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) BGR image, got {leaf_bgr.shape}")

        h, w = leaf_bgr.shape[:2]
        target_h, target_w = self.imgsz
        resized = cv2.resize(leaf_bgr, (target_w, target_h)).astype(np.float32) / 255.0
        batch = torch.from_numpy(resized.transpose(2, 0, 1)[None].copy()).to(self.device)

        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(batch))

        probabilities = probabilities.squeeze().cpu().numpy()
        return cv2.resize(probabilities, (w, h), interpolation=cv2.INTER_LINEAR)

    def segment(self, leaf_bgr: np.ndarray) -> np.ndarray:
        return self.probabilities(leaf_bgr) > self.threshold
