import pytest
import torch

from septosympto.models import build_segmenter


@pytest.mark.parametrize("arch", ["unet-resnet18", "unet-convnext-t"])
def test_logits_come_back_at_input_resolution(arch):
    model = build_segmenter(arch, pretrained=False).eval()
    assert model.INPUT_DIVISOR == 32
    with torch.inference_mode():
        out = model(torch.rand(1, 3, 64, 128))
    assert out.shape == (1, 1, 64, 128)


def test_backbone_unet_trains_end_to_end():
    model = build_segmenter("unet-resnet18", pretrained=False)
    x = torch.rand(2, 3, 32, 64)
    loss = model(x).mean()
    loss.backward()
    assert any(p.grad is not None for p in model.backbone.parameters())
    assert any(p.grad is not None for p in model.head.parameters())
