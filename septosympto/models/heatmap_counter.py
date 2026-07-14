"""A heatmap keypoint counter: the baseline point-counting architecture.

Pycnidia are ~4 px discs, hundreds per leaf. A box detector is the wrong tool;
the annotations are points in all but name. This model predicts a heatmap at
stride 4 with a Gaussian peak at each pycnidium, and decodes points by local-max
extraction. No boxes, no NMS over thousands of candidates, no confidence
threshold that shifts the biological result the way v1's ``-pt 0.3`` did.

It is a **baseline**, deliberately simple, meant to prove the counting harness
end to end. It owns the three methods the training loop calls — ``forward``,
``loss``, ``decode`` — so a P2P network with Hungarian matching replaces it by
registering under another name and implementing the same three, with no change
to the loop, the data, or the evaluation.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from septosympto.models.registry import register_counter

STRIDE = 4
GAUSSIAN_SIGMA = 2.0
DECODE_THRESHOLD = 0.3
DECODE_MAX_WINDOW = 2 * round(GAUSSIAN_SIGMA) + 1


def _conv_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _splat(points_hw: torch.Tensor, h: int, w: int, sigma: float) -> torch.Tensor:
    """A ``(h, w)`` heatmap with a Gaussian at each point, peaks stamped to 1.

    The exact peak cell is forced to 1 so the penalty-reduced focal loss has an
    unambiguous positive to key on; the Gaussian skirt only softens the penalty on
    near-miss negatives.
    """
    target = torch.zeros(h, w)
    radius = int(3 * sigma)
    for px, py in points_hw.tolist():
        cx, cy = int(round(px)), int(round(py))
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        ys = torch.arange(y0, y1).view(-1, 1)
        xs = torch.arange(x0, x1).view(1, -1)
        blob = torch.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma**2))
        target[y0:y1, x0:x1] = torch.maximum(target[y0:y1, x0:x1], blob)
        if 0 <= cy < h and 0 <= cx < w:
            target[cy, cx] = 1.0
    return target


def _focal_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss, normalised by the positive count.

    Plain MSE on a heatmap that is 99 % background collapses to predicting all
    zeros, which is what stalled the count at a near-total under-detection. This
    keys the gradient on the sparse peaks instead.
    """
    prob = prob.clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1.0).float()
    negative_weight = (1 - target) ** 4

    pos_loss = ((1 - prob) ** 2) * torch.log(prob) * positive
    neg_loss = negative_weight * (prob**2) * torch.log(1 - prob) * (1 - positive)

    n_pos = positive.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos


@register_counter("heatmap")
class HeatmapCounter(nn.Module):
    """Stride-4 heatmap keypoint detector for dense small points.

    Input ``(N, 3, H, W)`` float32 in ``[0, 1]``, BGR. ``forward`` returns heatmap
    logits ``(N, 1, H/4, W/4)``. Points are in full-resolution pixel coordinates
    everywhere the loop touches them.
    """

    INPUT_DIVISOR = STRIDE

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.stem = _conv_block(in_channels, c, stride=2)
        self.down = _conv_block(c, c * 2, stride=2)
        self.body = _conv_block(c * 2, c * 2, stride=1)
        self.head = nn.Conv2d(c * 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down(x)
        x = self.body(x)
        return self.head(x)

    def loss(self, logits: torch.Tensor, target_points: list[torch.Tensor]) -> torch.Tensor:
        _, _, h, w = logits.shape
        targets = torch.stack(
            [_splat(points / STRIDE, h, w, GAUSSIAN_SIGMA) for points in target_points]
        ).unsqueeze(1).to(logits.device)
        return _focal_loss(torch.sigmoid(logits), targets)

    @torch.inference_mode()
    def decode(
        self, logits: torch.Tensor, threshold: float = DECODE_THRESHOLD
    ) -> list[np.ndarray]:
        """Peaks above ``threshold``, as full-resolution ``(x, y)`` points.

        One point per connected region of local maxima, taken at its centroid.
        Clustering the peak mask rather than emitting every equal-to-pooled cell
        keeps a single point when a well-fit blob saturates into a flat plateau,
        where a bare ``heat == pooled`` test would fire on the whole top.
        """
        heat = torch.sigmoid(logits)
        pooled = F.max_pool2d(heat, DECODE_MAX_WINDOW, stride=1, padding=DECODE_MAX_WINDOW // 2)
        peaks = ((heat == pooled) & (heat > threshold)).cpu().numpy()
        out = []
        for i in range(peaks.shape[0]):
            mask = peaks[i, 0].astype(np.uint8)
            count, _, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            points = centroids[1:count] * STRIDE + STRIDE / 2 if count > 1 else np.empty((0, 2))
            out.append(points.astype(np.float32))
        return out
