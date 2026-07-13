"""PyTorch port of the SeptoSympto necrosis U-Net.

This is a faithful re-implementation of the original TensorFlow/Keras model
(``necrosis-model-375.h5``, 31 055 297 parameters). Layer for layer, it is the
same network, so the Keras weights can be transferred into it exactly rather
than approximately. ``tools/convert_keras_unet.py`` performs the transfer and
checks numerical parity against the Keras model.

Two Keras conventions matter and are reproduced here.

``BatchNormalization`` uses ``epsilon=1e-3`` where PyTorch defaults to ``1e-5``,
and ``momentum=0.99``, which is ``1 - 0.99 = 0.01`` in PyTorch's convention.

Every convolution carries a bias even though it is immediately followed by a
batch norm that cancels it. Mathematically redundant, but the trained values are
what they are, so the bias terms must exist in order to be loaded.

The decoder concatenates ``[upsampled, skip]`` in that order, matching the Keras
graph. Swapping the two silently produces a model that trains but cannot load
the original weights.

``forward`` returns **logits**, not probabilities, which makes the model usable
with ``BCEWithLogitsLoss`` for fine-tuning. Use :meth:`UNet.predict` to obtain
the sigmoid probabilities that the original Keras model emitted directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from septosympto.models.registry import register_segmenter

_BN_EPS = 1e-3
_BN_MOMENTUM = 0.01


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2, the repeating unit of the U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)))


@register_segmenter("unet")
class UNet(nn.Module):
    """Binary segmentation U-Net, 4-level encoder and a 1024-channel bottleneck.

    Input is expected as ``(N, 3, H, W)`` float32 in ``[0, 1]``, in **BGR**
    channel order: the original model was trained on images read with
    ``cv2.imread``, which returns BGR, and no channel swap was ever applied.
    ``H`` and ``W`` must be divisible by ``INPUT_DIVISOR``. The reference size is
    304x3072.
    """

    INPUT_DIVISOR = 16

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels

        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = DoubleConv(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        return self.head(d1)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Necrosis probability per pixel, matching the Keras model's output."""
        return torch.sigmoid(self(x))

    def _blocks(self) -> list[DoubleConv]:
        return [self.enc1, self.enc2, self.enc3, self.enc4, self.bottleneck,
                self.dec4, self.dec3, self.dec2, self.dec1]

    def ordered_convs(self) -> list[nn.Conv2d]:
        """The 19 Conv2d layers in Keras graph order, for weight transfer."""
        convs = [conv for blk in self._blocks() for conv in (blk.conv1, blk.conv2)]
        return [*convs, self.head]

    def ordered_bns(self) -> list[nn.BatchNorm2d]:
        """The 18 BatchNorm2d layers in Keras graph order."""
        return [bn for blk in self._blocks() for bn in (blk.bn1, blk.bn2)]

    def ordered_deconvs(self) -> list[nn.ConvTranspose2d]:
        """The 4 ConvTranspose2d layers in Keras graph order."""
        return [self.up4, self.up3, self.up2, self.up1]
