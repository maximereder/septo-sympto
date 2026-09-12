"""U-Net decoders over the shared pretrained backbones.

The original ``unet`` is trained from scratch, 31 M parameters, and pools to
stride 16. These variants swap its hand-built encoder for an ImageNet-pretrained
pyramid from :mod:`backbones` — the same ResNet-18 and ConvNeXt-Tiny trunks the
P2P counters use — and keep a light U-Net decoder over it. Pretrained features
converge in far fewer epochs on 250 training leaves than an encoder learning
from zero, which is the point of the comparison.

The backbones expose ``C2..C5`` (strides 4..32) and no full-resolution skip, so
the decoder fuses down to stride 4 and a two-step learned upsampling brings the
logits back to pixel resolution. Same contract as ``unet``: ``(N, 3, H, W)`` BGR
in ``[0, 1]`` to ``(N, 1, H, W)`` logits, and :class:`TorchSegmenter` wraps it
unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from septosympto.models.backbones import build_backbone
from septosympto.models.registry import register_segmenter


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    """Upsample x2, concatenate the skip, refine with two 3x3 convs."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(ConvBNReLU(in_ch + skip_ch, out_ch), ConvBNReLU(out_ch, out_ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class BackboneUNet(nn.Module):
    """A U-Net decoder over a pretrained ``C2..C5`` backbone.

    ``H`` and ``W`` must be divisible by ``INPUT_DIVISOR`` (the deepest stride).
    """

    INPUT_DIVISOR = 32

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        decoder_channels: tuple[int, int, int] = (256, 128, 64),
        head_channels: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained)
        ch = self.backbone.out_channels
        d4, d3, d2 = decoder_channels
        self.up4 = UpBlock(ch["C5"], ch["C4"], d4)
        self.up3 = UpBlock(d4, ch["C3"], d3)
        self.up2 = UpBlock(d3, ch["C2"], d2)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(d2, head_channels, 2, stride=2),
            ConvBNReLU(head_channels, head_channels),
            nn.ConvTranspose2d(head_channels, head_channels, 2, stride=2),
            ConvBNReLU(head_channels, head_channels),
            nn.Conv2d(head_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c2, c3, c4, c5 = self.backbone(x)
        d = self.up4(c5, c4)
        d = self.up3(d, c3)
        d = self.up2(d, c2)
        return self.head(d)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x))


@register_segmenter("unet-resnet18")
class UNetResNet18(BackboneUNet):
    """U-Net decoder over ResNet-18: the light, fast-converging variant."""

    def __init__(self, pretrained: bool = True, **kwargs) -> None:
        super().__init__(backbone="resnet18", pretrained=pretrained, **kwargs)


@register_segmenter("unet-convnext-t")
class UNetConvNeXtTiny(BackboneUNet):
    """U-Net decoder over ConvNeXt-Tiny: the modern, higher-capacity variant."""

    def __init__(self, pretrained: bool = True, **kwargs) -> None:
        super().__init__(backbone="convnext_tiny", pretrained=pretrained, **kwargs)
