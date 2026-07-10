import numpy as np
import pytest
import torch
import torch.nn as nn

from septosympto.adapters import TorchSegmenter
from septosympto.ports import Segmenter


class ConstantLogit(nn.Module):
    """Emits a fixed logit everywhere, so thresholds are testable without weights."""

    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.logit)


class LeftHalfPositive(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        out = torch.full((n, 1, h, w), -10.0)
        out[..., : w // 2] = 10.0
        return out


def leaf(h=120, w=1200):
    return np.full((h, w, 3), 128, np.uint8)


def test_it_satisfies_the_segmenter_protocol():
    seg = TorchSegmenter(ConstantLogit(10.0), imgsz=(64, 640))
    assert isinstance(seg, Segmenter)


def test_mask_comes_back_at_the_leaf_resolution_not_the_model_resolution():
    seg = TorchSegmenter(ConstantLogit(10.0), imgsz=(64, 640))
    mask = seg.segment(leaf(h=173, w=2011))
    assert mask.shape == (173, 2011)
    assert mask.dtype == np.bool_


def test_threshold_is_applied_to_probabilities():
    above = TorchSegmenter(ConstantLogit(1.0), imgsz=(32, 320), threshold=0.5)
    below = TorchSegmenter(ConstantLogit(1.0), imgsz=(32, 320), threshold=0.9)
    assert above.segment(leaf()).all()
    assert not below.segment(leaf()).any()


def test_spatial_layout_survives_the_round_trip():
    seg = TorchSegmenter(LeftHalfPositive(), imgsz=(32, 320), threshold=0.5)
    mask = seg.segment(leaf(h=100, w=1000))
    assert mask[:, :400].all()
    assert not mask[:, 600:].any()


def test_probabilities_stay_in_the_unit_interval():
    seg = TorchSegmenter(ConstantLogit(0.3), imgsz=(32, 320))
    p = seg.probabilities(leaf())
    assert p.min() >= 0.0
    assert p.max() <= 1.0
    assert p.mean() == pytest.approx(torch.sigmoid(torch.tensor(0.3)).item(), abs=1e-5)


def test_a_grayscale_image_is_rejected():
    seg = TorchSegmenter(ConstantLogit(0.0), imgsz=(32, 320))
    with pytest.raises(ValueError, match=r"\(H, W, 3\) BGR"):
        seg.segment(np.zeros((10, 10), np.uint8))
