"""Evaluation metrics, written once and driven through :mod:`septosympto.ports`.

This module exists because of two failures in v1 that a shared, explicit metric
would have caught.

The training logs reported ``val_dice_coef`` around 0.47 and it was taken at face
value. The Roboflow masks encode necrosis as magenta ``(255, 0, 124)``; read as
greyscale and divided by 255, the target becomes 0.354 rather than 1.0, which
caps the achievable Dice at ``2 * 0.354 / (1 + 0.354) = 0.523``. The model was
near its ceiling. The real binary Dice is around 0.78. **Binarise the ground
truth explicitly. Never compute Dice against a soft target.**

Model selection looked only at Dice. The shipped checkpoint recovers 81 % of the
annotated necrotic area while another checkpoint on the same disk recovers 98 %.
Dice is insensitive to a systematic under-segmentation that shrinks every lesion
slightly. Necrotic area is what gets published. **Report area bias alongside
overlap, always.**

For counting, the biological quantity is a count, so report the error on counts.
Localisation quality is reported separately, by matching predicted points to
annotated ones within a radius.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPSILON = 1e-9


def _check_binary(mask: np.ndarray, name: str) -> np.ndarray:
    if mask.dtype != np.bool_:
        unique = np.unique(mask)
        if not np.isin(unique, (0, 1)).all():
            raise ValueError(
                f"{name} is not binary (values {unique[:5]}). Threshold it explicitly: "
                "computing Dice against a soft or colour-encoded target silently caps the score."
            )
    return mask.astype(bool)


def dice(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = _check_binary(pred, "pred"), _check_binary(true, "true")
    total = pred.sum() + true.sum()
    if total == 0:
        return 1.0
    return float(2 * np.logical_and(pred, true).sum() / total)


def iou(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = _check_binary(pred, "pred"), _check_binary(true, "true")
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, true).sum() / union)


def area_ratio(pred: np.ndarray, true: np.ndarray) -> float:
    """Predicted area over annotated area. 1.0 is unbiased, below 1.0 under-segments."""
    pred, true = _check_binary(pred, "pred"), _check_binary(true, "true")
    return float(pred.sum() / (true.sum() + EPSILON))


@dataclass(frozen=True)
class SegmentationReport:
    n: int
    dice: float
    iou: float
    area_ratio: float
    area_bias_pct: float
    empty_truth_false_positives: int

    def __str__(self) -> str:
        return (
            f"n={self.n}  Dice={self.dice:.4f}  IoU={self.iou:.4f}  "
            f"area_ratio={self.area_ratio:.3f} ({self.area_bias_pct:+.1f} %)  "
            f"false positives on empty leaves={self.empty_truth_false_positives}"
        )


def segmentation_report(preds: list[np.ndarray], truths: list[np.ndarray]) -> SegmentationReport:
    """Overlap and area bias, averaged over leaves that carry a lesion.

    Leaves with an empty ground truth are excluded from the averages, where they
    would otherwise contribute a Dice of 1.0 or 0.0 depending on nothing. They
    are counted separately as false positives.
    """
    if len(preds) != len(truths):
        raise ValueError(f"{len(preds)} predictions for {len(truths)} ground truths")

    dices, ious, ratios, false_positives = [], [], [], 0
    for pred, true in zip(preds, truths, strict=True):
        true_bin = _check_binary(true, "true")
        if not true_bin.any():
            false_positives += int(_check_binary(pred, "pred").any())
            continue
        dices.append(dice(pred, true))
        ious.append(iou(pred, true))
        ratios.append(area_ratio(pred, true))

    if not dices:
        raise ValueError("no ground truth contains a lesion")

    mean_ratio = float(np.mean(ratios))
    return SegmentationReport(
        n=len(dices),
        dice=float(np.mean(dices)),
        iou=float(np.mean(ious)),
        area_ratio=mean_ratio,
        area_bias_pct=100 * (mean_ratio - 1),
        empty_truth_false_positives=false_positives,
    )


def match_points(
    pred: np.ndarray, true: np.ndarray, radius_px: float
) -> tuple[int, int, int]:
    """Greedy one-to-one matching within ``radius_px``. Returns (tp, fp, fn).

    Greedy by increasing distance rather than optimal assignment: with pycnidia
    several radii apart the two agree, and this keeps the package free of a
    SciPy dependency.
    """
    if len(pred) == 0 or len(true) == 0:
        return 0, len(pred), len(true)

    distances = np.linalg.norm(pred[:, None, :] - true[None, :, :], axis=2)
    order = np.dstack(np.unravel_index(np.argsort(distances, axis=None), distances.shape))[0]

    used_pred, used_true, tp = set(), set(), 0
    for i, j in order:
        if distances[i, j] > radius_px:
            break
        if i in used_pred or j in used_true:
            continue
        used_pred.add(int(i))
        used_true.add(int(j))
        tp += 1
    return tp, len(pred) - tp, len(true) - tp


@dataclass(frozen=True)
class CountingReport:
    n: int
    mae: float
    rmse: float
    bias: float
    relative_error_median_pct: float
    r2: float
    slope: float
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None

    def __str__(self) -> str:
        base = (
            f"n={self.n}  MAE={self.mae:.1f}  RMSE={self.rmse:.1f}  bias={self.bias:+.1f}  "
            f"median rel. err={self.relative_error_median_pct:.1f} %  "
            f"R²={self.r2:.4f}  slope={self.slope:.3f}"
        )
        if self.f1 is not None:
            base += f"  |  P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f}"
        return base


def counting_report(
    pred_points: list[np.ndarray],
    true_points: list[np.ndarray],
    radius_px: float | None = None,
) -> CountingReport:
    """Error on counts, and optionally on localisation.

    A slope below 1.0 means the model under-counts proportionally, which matters
    when comparing genotypes; a non-zero intercept shifts every leaf equally,
    which mostly cancels. Reporting only MAE hides which of the two is happening.
    """
    pred_counts = np.array([len(p) for p in pred_points], float)
    true_counts = np.array([len(t) for t in true_points], float)
    if len(pred_counts) != len(true_counts):
        raise ValueError("mismatched number of leaves")
    if len(pred_counts) < 2:
        raise ValueError("need at least two leaves to fit a slope")

    error = pred_counts - true_counts
    residual = float((error**2).sum())
    total = float(((true_counts - true_counts.mean()) ** 2).sum())
    slope = float(np.polyfit(true_counts, pred_counts, 1)[0])

    precision = recall = f1 = None
    if radius_px is not None:
        tp = fp = fn = 0
        for p, t in zip(pred_points, true_points, strict=True):
            a, b, c = match_points(np.asarray(p), np.asarray(t), radius_px)
            tp, fp, fn = tp + a, fp + b, fn + c
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return CountingReport(
        n=len(pred_counts),
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt((error**2).mean())),
        bias=float(error.mean()),
        relative_error_median_pct=float(
            np.median(np.abs(error) / np.maximum(true_counts, 1)) * 100
        ),
        r2=1 - residual / total if total else 0.0,
        slope=slope,
        precision=precision,
        recall=recall,
        f1=f1,
    )
