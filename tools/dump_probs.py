"""Dump necrosis probability maps for a fixed set of leaves, from either backend.

Used to prove that the converted PyTorch model reproduces the original Keras
model, across two environments and two PyTorch versions::

    .venv-legacy/bin/python tools/dump_probs.py --backend keras \
        --weights data/necrosis-model-375.h5 --images <dir> --out /tmp/keras.npy

    .venv/bin/python tools/dump_probs.py --backend torch \
        --weights data/necrosis-model-375.safetensors --images <dir> --out /tmp/torch.npy

    python tools/dump_probs.py --compare /tmp/keras.npy /tmp/torch.npy

The comparison reports pixel-level deviation, mask disagreement at the two
thresholds that matter, and the relative error on necrosis area, which is the
quantity the tool actually publishes.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

PARITY_TOLERANCE = 1e-3


def load_images(directory: str, n: int, h: int, w: int) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(directory, "*.jpg")))[:n]
    if not files:
        raise SystemExit(f"aucune image dans {directory}")
    return np.stack([cv2.resize(cv2.imread(f), (w, h)).astype(np.float32) / 255.0 for f in files])


def run_keras(weights: str, x: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    from tensorflow.keras.utils import CustomObjectScope

    from tools.metrics import dice_coef, dice_loss, iou

    with CustomObjectScope({"iou": iou, "dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = tf.keras.models.load_model(weights)
    return np.stack([np.squeeze(model.predict(xi[None], verbose=0)) for xi in x])


def run_torch(weights: str, x: np.ndarray) -> np.ndarray:
    import torch

    from septosympto.models.unet import UNet

    model = UNet().eval()
    if weights.endswith(".safetensors"):
        from safetensors.torch import load_file

        model.load_state_dict(load_file(weights))
    else:
        model.load_state_dict(torch.load(weights, map_location="cpu"))

    xt = torch.from_numpy(np.transpose(x, (0, 3, 1, 2)).copy())
    with torch.inference_mode():
        return torch.sigmoid(model(xt)).squeeze(1).numpy()


def compare(a_path: str, b_path: str) -> None:
    a, b = np.load(a_path), np.load(b_path)
    if a.shape != b.shape:
        raise SystemExit(f"formes incompatibles: {a.shape} vs {b.shape}")

    d = np.abs(a - b)
    print(f"images comparées : {a.shape[0]}  ({a.shape[1]}x{a.shape[2]} px)")
    print(f"  écart absolu max   : {d.max():.3e}")
    print(f"  écart absolu moyen : {d.mean():.3e}")

    for th in (0.5, 0.8):
        ma, mb = a > th, b > th
        dis = int((ma != mb).sum())
        dice = 2 * (ma & mb).sum() / (ma.sum() + mb.sum()) if (ma.sum() + mb.sum()) else 1.0
        pct = 100 * dis / ma.size
        print(f"  seuil {th}: {dis} pixels en désaccord ({pct:.6f} %) | Dice = {dice:.6f}")

    area_a, area_b = (a > 0.8).sum(axis=(1, 2)), (b > 0.8).sum(axis=(1, 2))
    rel = np.abs(area_a - area_b) / np.maximum(area_a, 1)
    print(f"  aire de nécrose @ 0.8 : écart relatif max = {rel.max() * 100:.6f} %")
    print("\n  verdict :", "PARITÉ OK" if d.max() < PARITY_TOLERANCE else "ÉCART SUSPECT")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["keras", "torch"])
    p.add_argument("--weights")
    p.add_argument("--images")
    p.add_argument("--out")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--imgsz", type=int, nargs=2, default=[304, 3072], metavar=("H", "W"))
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = p.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    x = load_images(args.images, args.n, *args.imgsz)
    y = run_keras(args.weights, x) if args.backend == "keras" else run_torch(args.weights, x)
    np.save(args.out, y)
    print(f"{args.backend}: {y.shape} -> {args.out}  (min={y.min():.4f} max={y.max():.4f})")


if __name__ == "__main__":
    main()
