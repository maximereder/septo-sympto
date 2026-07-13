"""Segmentation losses.

A soft Dice term handles the strong class imbalance of thin lesions on a large
leaf, where pixelwise BCE alone drifts toward predicting all-background. BCE is
kept alongside it for stable gradients early in training. Both take **logits**,
matching what the models return, so the sigmoid lives in one place.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SMOOTH = 1.0


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + SMOOTH) / (union + SMOOTH)
    return (1 - dice).mean()


def bce_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, bce_weight: float, dice_weight: float
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(logits, target)
    return bce_weight * bce + dice_weight * dice
