import pytest
import torch
import torch.nn as nn

from septosympto.models import (
    UNet,
    available_segmenters,
    build_segmenter,
    register_segmenter,
    segmenter_class,
)


def test_unet_is_registered():
    assert "unet" in available_segmenters()
    assert segmenter_class("unet") is UNet


def test_build_segmenter_returns_an_instance():
    model = build_segmenter("unet")
    assert isinstance(model, UNet)


def test_unknown_name_lists_the_available_ones():
    with pytest.raises(KeyError, match="unknown segmenter"):
        build_segmenter("does-not-exist")


def test_a_new_architecture_registers_and_builds_by_name():
    @register_segmenter("tiny-test-seg")
    class TinySeg(nn.Module):
        INPUT_DIVISOR = 1

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    try:
        assert "tiny-test-seg" in available_segmenters()
        model = build_segmenter("tiny-test-seg")
        out = model(torch.zeros(1, 3, 8, 8))
        assert out.shape == (1, 1, 8, 8)
    finally:
        from septosympto.models.registry import _SEGMENTERS

        _SEGMENTERS.pop("tiny-test-seg", None)


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):

        @register_segmenter("unet")
        class Clash(nn.Module):
            pass
