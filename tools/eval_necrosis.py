"""Evaluate necrosis checkpoints on a held-out split, with the correct metric.

    poetry run python tools/eval_necrosis.py \
        --dataset data/necrosis/dataset/300.zip --split valid \
        --weights data/necrosis/models/50.safetensors \
                  data/necrosis/models/100.safetensors \
                  data/necrosis/models/200.safetensors \
                  data/necrosis/models/300.safetensors \
                  data/necrosis-model-375.safetensors

Two things this fixes about how v1 chose a checkpoint.

The masks are Roboflow RGB PNGs with necrosis painted magenta, ``(255, 0, 124)``.
Converted to greyscale and divided by 255 they yield a target of 0.354, which
caps Dice at 0.523. Here the mask is binarised explicitly: a pixel is necrotic
when its strongest channel reaches half intensity. On the reference split that
agrees with the greyscale route to within 31 pixels out of 933 888, and unlike
it, it cannot silently rescale the positive class.

Selection looked only at Dice. Dice is nearly blind to a uniform erosion of every
lesion, and necrotic area is the published quantity. ``area_ratio`` is reported
next to it.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from septosympto.adapters import TorchSegmenter
from septosympto.eval import segmentation_report
from septosympto.models import UNet

MASK_CHANNEL_THRESHOLD = 128


def load_split(dataset: Path, split: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    leaves, truths = [], []
    with zipfile.ZipFile(dataset) as archive:
        images = sorted(
            n for n in archive.namelist()
            if f"/{split}/img/" in n and n.endswith(".jpg") and "__MACOSX" not in n
        )
        for name in images:
            mask_name = name.replace("/img/", "/mask/").replace(".jpg", ".png")
            if mask_name not in archive.namelist():
                continue
            leaf = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            mask = cv2.imdecode(
                np.frombuffer(archive.read(mask_name), np.uint8), cv2.IMREAD_UNCHANGED
            )
            leaves.append(leaf)
            truths.append(mask.max(axis=2) >= MASK_CHANNEL_THRESHOLD)
    if not leaves:
        raise SystemExit(f"no {split} images found in {dataset}")
    return leaves, truths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/necrosis/dataset/300.zip"))
    parser.add_argument("--split", default="valid")
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.8])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    leaves, truths = load_split(args.dataset, args.split)
    carrying = sum(t.any() for t in truths)
    print(f"{args.dataset.name} [{args.split}] : {len(leaves)} feuilles, {carrying} avec nécrose\n")

    header = (
        f"{'checkpoint':22s} {'seuil':>6s} {'Dice':>8s} {'IoU':>8s} "
        f"{'aire p/v':>9s} {'biais':>8s}"
    )
    print(header)
    print("-" * len(header))

    for weights in args.weights:
        for threshold in args.thresholds:
            segmenter = TorchSegmenter.from_safetensors(
                weights, UNet(), threshold=threshold, device=args.device
            )
            preds = [segmenter.segment(leaf) for leaf in leaves]
            report = segmentation_report(preds, truths)
            print(
                f"{weights.stem:22s} {threshold:6.2f} {report.dice:8.4f} {report.iou:8.4f} "
                f"{report.area_ratio:9.3f} {report.area_bias_pct:+7.1f} %"
            )
        print()


if __name__ == "__main__":
    main()
