"""P2PNet: a purely point-based counter for dense, tiny pycnidia.

After "Rethinking Counting and Localization in Crowds: A Purely Point-Based
Framework" (Song et al., ICCV 2021), adapted for ~4 px objects: the neck fuses
down to stride 4 (denser anchoring than the stride-8 original), a single
foreground class, and an exposed ``reg_scale`` for the anchor offset.

Unlike the heatmap baseline, points are **explicit**: each anchor predicts an
offset and a foreground probability, and the loss matches the predicted point set
to the ground-truth set with the Hungarian algorithm. There is no grid of
heatmap cells to merge touching pycnidia into one peak, which is the failure mode
that made the heatmap baseline under-count.

The architecture is unchanged from the reference implementation. What is added is
the three-method contract the training loop calls -- ``forward``, ``loss``,
``decode`` -- so P2PNet drops into the same loop, data, and evaluation as any
other registered counter. ``SetCriterion`` and the matcher stay as their own
classes; the model wraps them.

SciPy is imported lazily inside the matcher so that importing this module for
inference (which only needs ``decode``) does not require it. Training does.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_BN_Weights, vgg16_bn

from septosympto.models.registry import register_counter


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


def _build_backbone(name: str, pretrained: bool) -> nn.Module:
    if name == "vgg16":
        return VGGBackbone(pretrained)
    if name in _RESNET:
        return ResNetBackbone(name, pretrained)
    if name == "convnext_tiny":
        return ConvNeXtBackbone(pretrained)
    raise ValueError(f"unknown backbone {name!r}")


class FineDecoder(nn.Module):
    """Top-down FPN C5->C4->C3->C2, returning a map at C2's stride (/4)."""

    def __init__(self, c2=128, c3=256, c4=512, c5=512, feat=256) -> None:
        super().__init__()
        self.lat5 = nn.Conv2d(c5, feat, 1)
        self.lat4 = nn.Conv2d(c4, feat, 1)
        self.lat3 = nn.Conv2d(c3, feat, 1)
        self.lat2 = nn.Conv2d(c2, feat, 1)
        self.smooth = nn.Conv2d(feat, feat, 3, padding=1)

    @staticmethod
    def _up_add(top, lateral):
        up = F.interpolate(top, size=lateral.shape[-2:], mode="nearest")
        return up + lateral

    def forward(self, c2, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self._up_add(p5, self.lat4(c4))
        p3 = self._up_add(p4, self.lat3(c3))
        p2 = self._up_add(p3, self.lat2(c2))
        return self.smooth(p2)


def _anchor_offsets(stride: int, row: int, line: int, device) -> torch.Tensor:
    """A ``(row*line, 2)`` grid of ``(dx, dy)`` offsets centred in a stride cell."""
    ys = (torch.arange(1, row + 1, device=device) - 0.5) * (stride / row) - stride / 2
    xs = (torch.arange(1, line + 1, device=device) - 0.5) * (stride / line) - stride / 2
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


class AnchorPoints(nn.Module):
    def __init__(self, stride: int, row: int = 2, line: int = 2) -> None:
        super().__init__()
        self.stride, self.row, self.line = stride, row, line
        self.num = row * line

    def forward(self, feat):
        h, w = feat.shape[-2:]
        device = feat.device
        cx = (torch.arange(w, device=device) + 0.5) * self.stride
        cy = (torch.arange(h, device=device) + 0.5) * self.stride
        gy, gx = torch.meshgrid(cy, cx, indexing="ij")
        centers = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
        offsets = _anchor_offsets(self.stride, self.row, self.line, device)
        points = centers[:, None, :] + offsets[None, :, :]
        return points.reshape(-1, 2)


class _Head(nn.Module):
    def __init__(self, in_ch, num_anchor, out_per_anchor, feat=256, n_layers=2) -> None:
        super().__init__()
        layers = []
        channels = in_ch
        for _ in range(n_layers):
            layers += [nn.Conv2d(channels, feat, 3, padding=1), nn.ReLU(inplace=True)]
            channels = feat
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(feat, num_anchor * out_per_anchor, 3, padding=1)
        self.num_anchor, self.out_per_anchor = num_anchor, out_per_anchor

    def forward(self, x):
        x = self.out(self.body(x))
        b, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.view(b, h * w * self.num_anchor, self.out_per_anchor)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_point=0.05) -> None:
        super().__init__()
        self.cost_class, self.cost_point = cost_class, cost_point

    @torch.no_grad()
    def forward(self, outputs, targets):
        from scipy.optimize import linear_sum_assignment

        batch = outputs["pred_logits"].shape[0]
        prob = outputs["pred_logits"].softmax(-1)
        pts = outputs["pred_points"]
        indices = []
        for b in range(batch):
            tgt_pts = targets[b]["points"]
            if tgt_pts.numel() == 0:
                indices.append(
                    (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
                )
                continue
            cost_pt = torch.cdist(pts[b], tgt_pts, p=2)
            cost_cls = -prob[b][:, 1:2]
            cost = self.cost_point * cost_pt + self.cost_class * cost_cls
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
            indices.append(
                (torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long))
            )
        return indices


class SetCriterion(nn.Module):
    def __init__(
        self, num_classes=2, matcher=None, eos_coef=0.5, w_class=1.0, w_point=0.0002
    ) -> None:
        super().__init__()
        self.matcher = matcher or HungarianMatcher()
        self.num_classes = num_classes
        self.w_class, self.w_point = w_class, w_point
        weight = torch.ones(num_classes)
        weight[0] = eos_coef
        self.register_buffer("empty_weight", weight)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        logits = outputs["pred_logits"]
        b, n, _ = logits.shape

        target_classes = torch.zeros(b, n, dtype=torch.long, device=logits.device)
        for i, (src, _) in enumerate(indices):
            target_classes[i, src] = 1
        loss_ce = F.cross_entropy(
            logits.transpose(1, 2), target_classes, weight=self.empty_weight
        )

        src_pts, tgt_pts = [], []
        for i, (src, tgt) in enumerate(indices):
            if src.numel():
                src_pts.append(outputs["pred_points"][i, src])
                tgt_pts.append(targets[i]["points"][tgt])
        if src_pts:
            src_pts = torch.cat(src_pts)
            tgt_pts = torch.cat(tgt_pts)
            loss_pt = F.mse_loss(src_pts, tgt_pts, reduction="sum") / max(len(src_pts), 1)
        else:
            loss_pt = logits.sum() * 0.0

        loss = self.w_class * loss_ce + self.w_point * loss_pt
        return {"loss": loss, "loss_ce": loss_ce.detach(), "loss_point": loss_pt.detach()}


@register_counter("p2p")
class P2PNet(nn.Module):
    """Point-set counter. Input ``(N, 3, H, W)`` BGR in ``[0, 1]``, H and W /4.

    ``forward`` returns ``{"pred_logits", "pred_points"}`` with points in
    full-resolution pixel coordinates, matching the coordinate space of the
    dataset's target points, so no rescaling is needed on either side.
    """

    INPUT_DIVISOR = 4

    def __init__(
        self,
        backbone: str = "vgg16",
        pretrained: bool = True,
        feat: int = 256,
        row: int = 2,
        line: int = 2,
        stride: int = 4,
        num_classes: int = 2,
        reg_scale: float = 8.0,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.reg_scale = reg_scale
        self.num_classes = num_classes
        self.backbone = _build_backbone(backbone, pretrained)
        ch = self.backbone.out_channels
        self.neck = FineDecoder(ch["C2"], ch["C3"], ch["C4"], ch["C5"], feat)
        num_anchor = row * line
        self.reg_head = _Head(feat, num_anchor, 2, feat)
        self.cls_head = _Head(feat, num_anchor, num_classes, feat)
        self.anchors = AnchorPoints(stride, row, line)
        self.criterion = SetCriterion(num_classes=num_classes)

    def forward(self, x):
        c2, c3, c4, c5 = self.backbone(x)
        f = self.neck(c2, c3, c4, c5)
        reg = self.reg_head(f)
        logits = self.cls_head(f)
        anchors = self.anchors(f).to(x.device)
        points = anchors[None] + reg * self.reg_scale
        return {"pred_logits": logits, "pred_points": points}

    def loss(self, output: dict, target_points: list[torch.Tensor]) -> torch.Tensor:
        device = output["pred_points"].device
        targets = [{"points": p.to(device)} for p in target_points]
        return self.criterion(output, targets)["loss"]

    @torch.inference_mode()
    def decode(self, output: dict, threshold: float = 0.5) -> list[np.ndarray]:
        prob = output["pred_logits"].softmax(-1)[..., 1]
        pts = output["pred_points"]
        out = []
        for b in range(prob.shape[0]):
            keep = prob[b] > threshold
            out.append(pts[b][keep].cpu().numpy().astype(np.float32))
        return out


@register_counter("p2p-resnet18")
class P2PResNet18(P2PNet):
    """P2PNet on a ResNet-18 backbone: the efficient, low-memory variant."""

    def __init__(self, pretrained: bool = True, **kwargs) -> None:
        super().__init__(backbone="resnet18", pretrained=pretrained, **kwargs)


@register_counter("p2p-resnet50")
class P2PResNet50(P2PNet):
    """P2PNet on a ResNet-50 backbone: the intermediate variant."""

    def __init__(self, pretrained: bool = True, **kwargs) -> None:
        super().__init__(backbone="resnet50", pretrained=pretrained, **kwargs)


@register_counter("p2p-convnext-t")
class P2PConvNeXtTiny(P2PNet):
    """P2PNet on a ConvNeXt-Tiny backbone: the modern, higher-capacity variant."""

    def __init__(self, pretrained: bool = True, **kwargs) -> None:
        super().__init__(backbone="convnext_tiny", pretrained=pretrained, **kwargs)
