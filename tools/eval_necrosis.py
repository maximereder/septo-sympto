"""Evaluate necrosis checkpoints on a held-out split, with the correct metric.

    # the project's test fold: same pool, split and seed as training
    poetry run python tools/eval_necrosis.py --split test \
        --weights runs/nec-r18/best.safetensors runs/nec-yolo26s/weights/best.pt \
        --arch unet-resnet18

    # the v1 baseline on the legacy Roboflow valid split
    poetry run python tools/eval_necrosis.py \
        --dataset data/necrosis/dataset/300.zip --split valid \
        --weights data/necrosis-model-375.safetensors --arch unet

Every checkpoint goes through its adapter — ``.safetensors`` through
:class:`TorchSegmenter` with the architecture named by ``--arch``, ``.pt``
through :class:`YoloSegmenter` — so a U-Net and a YOLO are scored on the very
same leaves, at the leaf's own resolution, by the same code. The split is the
project's scan-grouped one; ``test`` is the fold no training step ever touched.
The Roboflow ``train``/``valid`` folds of a legacy zip are still accepted.

Two things this fixes about how v1 chose a checkpoint.

The masks are Roboflow RGB PNGs with necrosis painted magenta, ``(255, 0, 124)``.
Converted to greyscale and divided by 255 they yield a target of 0.354, which
caps Dice at 0.523. Here the mask is binarised explicitly: a pixel is necrotic
when its strongest channel reaches half intensity.

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

from septosympto.eval import segmentation_report  # noqa: E402
from train.data import decode_image, decode_mask, load_pool, scan_grouped_split  # noqa: E402

MASK_CHANNEL_THRESHOLD = 128
FOLDS = ("train", "val", "test")


def load_roboflow_split(dataset: Path, split: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """A Roboflow fold as shipped (``train`` or ``valid``), for the v1 baseline."""
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


def load_project_split(
    dataset: Path, split: str, val_fraction: float, test_fraction: float, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """One fold of the project's scan-grouped split, as the training runs see it."""
    pool = load_pool(dataset)
    folds = dict(zip(FOLDS, scan_grouped_split(pool, val_fraction, test_fraction, seed),
                     strict=True))
    samples = folds[split]
    leaves = [decode_image(s.image_bytes) for s in samples]
    truths = [decode_mask(s.mask_bytes) for s in samples]
    return leaves, truths


def load_segmenter(weights: Path, arch: str, threshold: float, device: str):
    if weights.suffix == ".pt":
        from septosympto.adapters import YoloSegmenter

        return YoloSegmenter.from_weights(weights, threshold=threshold, device=device)
    from septosympto.adapters import TorchSegmenter
    from septosympto.models import build_segmenter

    return TorchSegmenter.from_safetensors(
        weights, build_segmenter(arch), threshold=threshold, device=device
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/leaves-native"),
                        help="Native letterbox directory or a legacy Roboflow zip.")
    parser.add_argument("--split", default="test",
                        help="train / val / test of the project's split, or train / valid "
                             "of a Roboflow zip as shipped.")
    parser.add_argument("--weights", type=Path, nargs="+", required=True,
                        help=".safetensors (PyTorch, see --arch) or .pt (YOLO semantic).")
    parser.add_argument("--arch", default="unet",
                        help="Architecture of the .safetensors checkpoints.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.8])
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.split in FOLDS:
        leaves, truths = load_project_split(
            args.dataset, args.split, args.val_fraction, args.test_fraction, args.seed
        )
    else:
        leaves, truths = load_roboflow_split(args.dataset, args.split)
    carrying = sum(t.any() for t in truths)
    print(f"{args.dataset.name} [{args.split}] : {len(leaves)} leaves, {carrying} with necrosis\n")

    header = (
        f"{'checkpoint':28s} {'thr':>5s} {'Dice':>8s} {'IoU':>8s} "
        f"{'area p/t':>9s} {'bias':>8s}"
    )
    print(header)
    print("-" * len(header))

    for weights in args.weights:
        label = weights.parent.parent.name if weights.name == "best.pt" else weights.stem
        for threshold in args.thresholds:
            segmenter = load_segmenter(weights, args.arch, threshold, args.device)
            preds = [segmenter.segment(leaf) for leaf in leaves]
            report = segmentation_report(preds, truths)
            print(
                f"{label[:28]:28s} {threshold:5.2f} {report.dice:8.4f} {report.iou:8.4f} "
                f"{report.area_ratio:9.3f} {report.area_bias_pct:+7.1f} %"
            )
        print()


if __name__ == "__main__":
    main()
