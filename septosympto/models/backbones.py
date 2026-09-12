"""ImageNet-pretrained feature pyramids shared by the counters and the segmenters.

Each backbone exposes the same thing: a ``C2..C5`` pyramid at strides 4/8/16/32
and an ``out_channels`` map, so a neck or decoder is written once against that
shape and any of them slots in. They are fed BGR in ``[0, 1]`` without ImageNet
normalisation — the pretrained weights adapt within the first epochs, and it keeps
every model in the project on the one input convention the adapters implement.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_BN_Weights, vgg16_bn


def _split_vgg_stages(features: nn.Sequential) -> list[nn.Sequential]:
    """Split ``vgg16_bn.features`` into five stages, one per MaxPool.

    Outputs C1(/2, 64), C2(/4, 128), C3(/8, 256), C4(/16, 512), C5(/32, 512).
    """
    stages, current = [], []
    for layer in features:
        current.append(layer)
        if isinstance(layer, nn.MaxPool2d):
            stages.append(nn.Sequential(*current))
            current = []
    if current:
        stages.append(nn.Sequential(*current))
    return stages


class VGGBackbone(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = VGG16_BN_Weights.DEFAULT if pretrained else None
        stages = _split_vgg_stages(vgg16_bn(weights=weights).features)
        self.stage1, self.stage2, self.stage3, self.stage4, self.stage5 = stages
        self.out_channels = {"C2": 128, "C3": 256, "C4": 512, "C5": 512}

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c2, c3, c4, c5


_RESNET = {
    "resnet18": (models.resnet18, models.ResNet18_Weights, (64, 128, 256, 512)),
    "resnet50": (models.resnet50, models.ResNet50_Weights, (256, 512, 1024, 2048)),
}


class ResNetBackbone(nn.Module):
    """A torchvision ResNet as a C2..C5 feature pyramid at strides 4/8/16/32.

    Lighter and lower in activation memory than VGG: the stem drops to stride 4
    before the first residual stage, where VGG still holds full-resolution feature
    maps. That is what lets a ResNet variant train at a larger batch or resolution.
    """

    def __init__(self, variant: str, pretrained: bool = True) -> None:
        super().__init__()
        builder, weights_enum, channels = _RESNET[variant]
        net = builder(weights=weights_enum.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            net.layer1, net.layer2, net.layer3, net.layer4
        )
        self.out_channels = dict(zip(("C2", "C3", "C4", "C5"), channels, strict=True))

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt-Tiny as a C2..C5 feature pyramid at strides 4/8/16/32.

    ``features`` alternates stage / downsample: a stride-4 patchify stem and stage
    (C2), then three (downsample, stage) pairs giving C3/C4/C5. Modern features at
    a higher parameter cost than ResNet, for when accuracy matters more than size.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        feats = models.convnext_tiny(weights=weights).features
        self.s2 = feats[0:2]
        self.s3 = feats[2:4]
        self.s4 = feats[4:6]
        self.s5 = feats[6:8]
        self.out_channels = {"C2": 96, "C3": 192, "C4": 384, "C5": 768}

    def forward(self, x):
        c2 = self.s2(x)
        c3 = self.s3(c2)
        c4 = self.s4(c3)
        c5 = self.s5(c4)
        return c2, c3, c4, c5


def build_backbone(name: str, pretrained: bool) -> nn.Module:
    if name == "vgg16":
        return VGGBackbone(pretrained)
    if name in _RESNET:
        return ResNetBackbone(name, pretrained)
    if name == "convnext_tiny":
        return ConvNeXtBackbone(pretrained)
    raise ValueError(f"unknown backbone {name!r}")
