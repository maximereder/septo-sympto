"""Zoom into a leaf region to judge predictions against annotations by eye.

The full overlay is too small to tell a false positive from a real pycnidium the
annotator missed. This crops a horizontal band at working resolution, upscales
it, and draws predictions (green) and annotations (red) large enough to see the
underlying dark pycnidia, so green-without-red can be checked against the pixels.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from septosympto.models import build_counter
from train.count_data import load_points_pool, split_pool


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/pyc-p2p/best.safetensors")
    p.add_argument("--dirs", nargs="+",
                   default=["data/pycnidia/train-200-aug-x3", "data/pycnidia/valid-40"])
    p.add_argument("--index", type=int, default=30)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--x0", type=int, default=1450)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--zoom", type=int, default=6)
    p.add_argument("--out", default="runs/pyc-p2p/zoom.jpg")
    args = p.parse_args()

    _, _, test = split_pool(load_points_pool(args.dirs), 0.15, 0.15, 0)
    test = sorted(test, key=lambda s: len(s.points_norm))
    sample = test[args.index]

    h, w = 200, 2048
    model = build_counter("p2p", pretrained=False).eval()
    model.load_state_dict(load_file(args.weights))

    resized = cv2.resize(cv2.imread(sample.image_path), (w, h))
    x = torch.from_numpy((resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None].copy())
    with torch.inference_mode():
        pred = np.asarray(model.decode(model(x), threshold=args.threshold)[0])

    gt = sample.points_norm.copy()
    gt[:, 0] *= w
    gt[:, 1] *= h

    matched_pred = set()
    if len(pred) and len(gt):
        dists = np.linalg.norm(pred[:, None] - gt[None], axis=2)
        order = np.dstack(np.unravel_index(np.argsort(dists, axis=None), dists.shape))[0]
        used_gt = set()
        for i, j in order:
            if dists[i, j] > args.radius:
                break
            if i in matched_pred or j in used_gt:
                continue
            matched_pred.add(int(i))
            used_gt.add(int(j))

    x0, x1 = args.x0, args.x0 + args.width
    crop = resized[:, x0:x1]
    big = cv2.resize(crop, (crop.shape[1] * args.zoom, crop.shape[0] * args.zoom),
                     interpolation=cv2.INTER_NEAREST)

    for px, py in gt:
        if x0 <= px < x1:
            c = (int((px - x0) * args.zoom), int(py * args.zoom))
            cv2.circle(big, c, 5, (0, 0, 255), -1)
    for i, (px, py) in enumerate(pred):
        if x0 <= px < x1:
            c = (int((px - x0) * args.zoom), int(py * args.zoom))
            color = (0, 255, 0) if i in matched_pred else (0, 200, 255)
            cv2.circle(big, c, 9, color, 2)

    n_fp = sum(1 for i, (px, _) in enumerate(pred) if x0 <= px < x1 and i not in matched_pred)
    cv2.imwrite(args.out, big)
    print(f"{sample.image}  x[{x0}:{x1}] zoom x{args.zoom}")
    print("  green circle = matched prediction, orange circle = unmatched (green-without-red)")
    print("  red dot = annotation")
    print(f"  unmatched predictions in this crop: {n_fp}  -> {args.out}")


if __name__ == "__main__":
    main()
