import numpy as np
import torch
import torch.nn as nn

from septosympto.adapters import YoloSegmenter
from septosympto.ports import Segmenter


class Stride4Constant(nn.Module):
    """A binary semantic head: one logit map at stride 4, like Ultralytics emits."""

    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, x):
        n, _, h, w = x.shape
        return torch.full((n, 1, h // 4, w // 4), self.logit)


class RedChannelIsPositive(nn.Module):
    """Positive where the *first* input channel is bright: tells RGB from BGR."""

    def forward(self, x):
        return (x[:, :1] - 0.5) * 20.0


def leaf(h=120, w=1200):
    return np.full((h, w, 3), 128, np.uint8)


def test_it_satisfies_the_segmenter_protocol():
    assert isinstance(YoloSegmenter(Stride4Constant(10.0)), Segmenter)


def test_mask_comes_back_at_leaf_resolution_from_a_stride_4_head():
    mask = YoloSegmenter(Stride4Constant(10.0)).segment(leaf(h=173, w=2011))
    assert mask.shape == (173, 2011)
    assert mask.dtype == np.bool_
    assert mask.all()


def test_threshold_is_applied_to_the_sigmoid():
    assert YoloSegmenter(Stride4Constant(1.0), threshold=0.5).segment(leaf()).all()
    assert not YoloSegmenter(Stride4Constant(1.0), threshold=0.9).segment(leaf()).any()


def test_the_leaf_is_fed_as_rgb_not_bgr():
    red_leaf = np.zeros((64, 640, 3), np.uint8)
    red_leaf[..., 2] = 255  # BGR: red lives in channel 2
    blue_leaf = np.zeros((64, 640, 3), np.uint8)
    blue_leaf[..., 0] = 255
    seg = YoloSegmenter(RedChannelIsPositive(), threshold=0.5)
    assert seg.segment(red_leaf).all()
    assert not seg.segment(blue_leaf).any()


def test_a_tuple_output_uses_its_first_element():
    class Tupled(Stride4Constant):
        def forward(self, x):
            return super().forward(x), None

    assert YoloSegmenter(Tupled(10.0)).segment(leaf()).all()
