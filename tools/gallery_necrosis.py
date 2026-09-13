"""Render a checkpoint's predictions against the annotation on held-out leaves.

    poetry run python tools/gallery_necrosis.py \
        --weights runs/nec-yolo26m/weights/best.pt --threshold 0.3 \
        --out runs/nec-yolo26m/test-gallery.png

Leaves are picked at fixed quantiles of annotated necrosis fraction across the
fold, so two galleries of the same fold show the same leaves and are comparable
side by side. Blue is the prediction, red the annotation; each row is labelled
with the leaf's Dice and area ratio.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_necrosis import FOLDS, load_segmenter  # noqa: E402

from septosympto.eval import area_ratio, dice  # noqa: E402
from train.data import decode_image, decode_mask, load_pool, scan_grouped_split  # noqa: E402

QUANTILES = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0)
WIDTH = 1536


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/leaves-native"))
    parser.add_argument("--split", default="test", choices=FOLDS)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--arch", default="unet", help="Architecture of a .safetensors checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pool = load_pool(args.dataset)
    folds = dict(zip(FOLDS, scan_grouped_split(pool, args.val_fraction, args.test_fraction,
                                                args.seed), strict=True))
    samples = sorted(folds[args.split], key=lambda s: decode_mask(s.mask_bytes).mean())
    picks = [samples[int(round(q * (len(samples) - 1)))] for q in QUANTILES]

    segmenter = load_segmenter(args.weights, args.arch, args.threshold, args.device)
    label = (args.weights.parent.parent.name if args.weights.name == "best.pt"
             else args.weights.stem)

    rows = [_banner(f"{label}  thr {args.threshold:.2f}  [{args.split}]   "
                    "blue = prediction, red = annotation")]
    for sample in picks:
        image = decode_image(sample.image_bytes)
        truth = decode_mask(sample.mask_bytes)
        pred = segmenter.segment(image)
        rows.append(_banner(
            f"{sample.image}   necrosis {100 * truth.mean():.1f}% of canvas   "
            f"Dice {dice(pred, truth):.2f}   area pred/truth {area_ratio(pred, truth):.2f}"
        ))
        rows.append(_overlay(image, truth, pred))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows))
    print(f"gallery -> {args.out}")


def _banner(text: str) -> np.ndarray:
    bar = np.full((30, WIDTH, 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return bar


def _overlay(image: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    canvas = np.ascontiguousarray(image.copy())
    for mask, colour in ((pred, (255, 0, 0)), (truth, (0, 0, 255))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, colour, 4)
    rows = np.where((image < 250).any(2).any(1))[0]
    y0, y1 = max(0, rows.min() - 8), min(image.shape[0], rows.max() + 8)
    band = np.ascontiguousarray(canvas[y0:y1])
    scale = WIDTH / band.shape[1]
    return cv2.resize(band, (WIDTH, max(1, int(band.shape[0] * scale))))


if __name__ == "__main__":
    main()
