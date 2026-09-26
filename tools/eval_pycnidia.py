"""Evaluate pycnidia counters on a held-out fold of the project's split.

    # the test fold: the leaves no training step and no checkpoint selection saw
    poetry run python tools/eval_pycnidia.py --split test \
        --weights p2p-convnext-v2 runs/pyc-p2p-convnext-corrected/best.safetensors \
        --arch p2p-convnext-t --device mps

A counter is scored exactly as the training loop scores its validation fold —
same pool, same scan-grouped split, same canvas size, same decode — so a number
here and a ``val_mae`` in a run manifest mean the same thing. What changes is the
fold: ``val`` drives checkpoint selection and is therefore optimistic, ``test``
is untouched.

The count is the biological quantity, so selection reads **MAE**; the slope and
bias beside it say *how* a model is wrong. A slope below 1 means proportional
under-counting, which distorts comparisons between genotypes; a bias with slope
near 1 shifts every leaf alike and largely cancels. Localisation precision,
recall and F1 (within ``--match-radius-px``) catch a model that gets the count
right while placing its points badly.

``--weights`` takes a published name from the zoo, or a path to a checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from septosympto import zoo  # noqa: E402
from septosympto.eval import counting_report  # noqa: E402
from train.count_data import (  # noqa: E402
    PycnidiaPointDataset,
    collate_points,
    load_points_pool,
    split_pool,
)

FOLDS = ("train", "val", "test")


def load_counter(spec: str, arch: str, device: str):
    """A zoo name or a checkpoint path -> an evaluated model on ``device``."""
    from septosympto.models import build_counter

    resolved = zoo.resolve(spec, "pycnidia", arch=arch)
    from safetensors.torch import load_file

    model = build_counter(resolved.arch or arch)
    model.load_state_dict(load_file(str(resolved.path)))
    return resolved, model.eval().to(device)


def predict(model, loader, device: str, threshold: float):
    pred_points, true_points = [], []
    with torch.inference_mode():
        for images, points in loader:
            output = model(images.to(device))
            for pred, truth in zip(model.decode(output, threshold), points, strict=True):
                pred_points.append(np.asarray(pred))
                true_points.append(truth.numpy())
    return pred_points, true_points


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=Path, nargs="+",
                        default=[Path("data/leaves-native")])
    parser.add_argument("--split", default="test", choices=FOLDS)
    parser.add_argument("--weights", nargs="+", required=True,
                        help="Zoo names (e.g. p2p-convnext-v2) or .safetensors paths.")
    parser.add_argument("--arch", default="p2p-convnext-t",
                        help="Architecture of checkpoints given as a path.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--imgsz", type=int, nargs=2, default=[384, 3072], metavar=("H", "W"))
    parser.add_argument("--match-radius-px", type=float, default=8.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    pool = load_points_pool(list(args.dataset_dir))
    folds = dict(zip(FOLDS,
                     split_pool(pool, args.val_fraction, args.test_fraction, args.seed),
                     strict=True))
    samples = folds[args.split]
    counts = np.array([len(s.points_norm) for s in samples])
    print(f"{'+'.join(d.name for d in args.dataset_dir)} [{args.split}]: {len(samples)} leaves, "
          f"{counts.sum()} pycnidia, {int((counts == 0).sum())} leaf-level zeros, "
          f"{len({s.scan for s in samples})} scans\n")

    dataset = PycnidiaPointDataset(samples, imgsz=(args.imgsz[0], args.imgsz[1]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_points)

    header = (f"{'checkpoint':34s} {'thr':>4s} {'MAE':>7s} {'RMSE':>7s} {'bias':>8s} "
              f"{'rel%':>6s} {'R2':>7s} {'slope':>6s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    print(header)
    print("-" * len(header))
    for spec in args.weights:
        resolved, model = load_counter(spec, args.arch, args.device)
        label = Path(spec).parent.name if spec.endswith(".safetensors") else resolved.name
        for threshold in args.thresholds:
            preds, truths = predict(model, loader, args.device, threshold)
            r = counting_report(preds, truths, radius_px=args.match_radius_px)
            print(f"{label[:34]:34s} {threshold:4.2f} {r.mae:7.1f} {r.rmse:7.1f} "
                  f"{r.bias:+8.1f} {r.relative_error_median_pct:6.1f} {r.r2:7.3f} "
                  f"{r.slope:6.3f} {r.precision:6.3f} {r.recall:6.3f} {r.f1:6.3f}")


if __name__ == "__main__":
    main()
