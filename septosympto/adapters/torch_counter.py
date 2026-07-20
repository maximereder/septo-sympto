"""Wraps a P2PNet point-regression model into a :class:`~septosympto.ports.PointCounter`.

Like the segmenter adapter, this owns its preprocessing and mirrors training: the
model was fed BGR in ``[0, 1]`` on the shared letterbox canvas, so the leaf is
letterboxed in, decoded to points on that canvas, and the points are mapped back
to native pixels through the inverse of the same placement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from septosympto.letterbox import letterbox, unletterbox_points

DECODE_THRESHOLD = 0.3


class TorchCounter:
    """A P2PNet-style counter presented as a ``PointCounter``.

    The module must accept ``(N, 3, H, W)`` float32 in ``[0, 1]``, BGR, and expose
    ``decode(output, threshold) -> list[np.ndarray]`` returning ``(M, 2)`` points
    in input-pixel coordinates.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = DECODE_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        self.model = model.eval().to(device)
        self.threshold = threshold
        self.device = device

    @classmethod
    def from_safetensors(
        cls,
        weights: str | Path,
        model: nn.Module,
        threshold: float = DECODE_THRESHOLD,
        device: str = "cpu",
    ) -> TorchCounter:
        from safetensors.torch import load_file

        model.load_state_dict(load_file(str(weights)))
        return cls(model, threshold=threshold, device=device)

    def count(self, leaf_bgr: np.ndarray) -> np.ndarray:
        """Pycnidia as ``(N, 2)`` ``(x, y)`` points, at the leaf's own resolution."""
        if leaf_bgr.ndim != 3 or leaf_bgr.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) BGR image, got {leaf_bgr.shape}")

        canvas, place = letterbox(leaf_bgr)
        batch = torch.from_numpy(
            (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None].copy()
        ).to(self.device)

        with torch.inference_mode():
            points = self.model.decode(self.model(batch), threshold=self.threshold)[0]

        return unletterbox_points(points, place)
