"""Run the trained P2P counter on a held-out test leaf and draw the result.

Reproduces the exact scan-grouped split the model was trained with (same seed and
fractions), picks a leaf from the test fold the model never saw, runs inference,
and writes an overlay: predicted pycnidia as green circles, annotated ones as red
dots, with the counts and the localisation match in the title.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from septosympto.eval import match_points
from septosympto.models import build_counter
from train.count_data import load_points_pool, split_pool


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/pyc-p2p/best.safetensors")
    p.add_argument("--arch", default="p2p", help="Counter architecture the weights belong to.")
    p.add_argument("--dirs", nargs="+",
                   default=["data/pycnidia/train-200-aug-x3", "data/pycnidia/valid-40"])
    p.add_argument("--imgsz", type=int, nargs=2, default=[200, 2048], metavar=("H", "W"))
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--index", type=int, default=None,
                   help="Which test leaf; default = median count.")
    p.add_argument("--show-gt", action="store_true",
                   help="Also draw the annotations; off by default (detections only).")
    p.add_argument("--out", default="runs/pyc-p2p/inference.jpg")
    args = p.parse_args()

    _, _, test = split_pool(load_points_pool(args.dirs), 0.15, 0.15, args.seed)
    test = sorted(test, key=lambda s: len(s.points_norm))
    if not test:
        raise SystemExit("empty test split")
    sample = test[args.index] if args.index is not None else test[len(test) // 2]

    model = build_counter(args.arch, pretrained=False).eval()
    model.load_state_dict(load_file(args.weights))

    h, w = args.imgsz
    bgr = cv2.imread(sample.image_path)
    resized = cv2.resize(bgr, (w, h))
    x = torch.from_numpy((resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None].copy())
    with torch.inference_mode():
        pred = model.decode(model(x), threshold=args.threshold)[0]

    gt = sample.points_norm.copy()
    gt[:, 0] *= w
    gt[:, 1] *= h

    tp, fp, fn = match_points(pred, gt, args.radius)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    canvas = resized.copy()
    if args.show_gt:
        for x0, y0 in gt.astype(int):
            cv2.circle(canvas, (int(x0), int(y0)), 2, (0, 0, 255), -1)
    for x0, y0 in np.asarray(pred).astype(int):
        cv2.circle(canvas, (int(x0), int(y0)), 4, (0, 255, 0), 1)

    banner = np.full((44, w, 3), 30, np.uint8)
    text = f"{sample.image}  |  {len(pred)} detections"
    if args.show_gt:
        text += (f"  (GT {len(gt)}, err {len(pred) - len(gt):+d})  |  "
                 f"P {precision:.2f} R {recall:.2f} F1 {f1:.2f}")
    cv2.putText(banner, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    out = np.vstack([banner, canvas])

    cv2.imwrite(args.out, out)
    print(f"leaf {sample.image} (test fold, never seen)")
    print(f"  ground truth : {len(gt)} pycnidia")
    print(f"  predicted    : {len(pred)}  (error {len(pred) - len(gt):+d})")
    print(f"  localisation : P {precision:.3f}  R {recall:.3f}  F1 {f1:.3f}  (r={args.radius}px)")
    print(f"  overlay -> {args.out}")


if __name__ == "__main__":
    main()
