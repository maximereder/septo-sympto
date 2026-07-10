"""Transfer the Keras necrosis U-Net weights into the PyTorch port, and verify parity.

Run this inside an environment that still has TensorFlow 2.15, so Python 3.11 or
older. The artifact it produces is framework-agnostic and loads under any recent
PyTorch, which is the whole point: TensorFlow is needed once, never again.

    .venv-legacy/bin/python tools/convert_keras_unet.py \
        --keras data/necrosis-model-375.h5 \
        --output data/necrosis-model-375.safetensors \
        --validate-dir <folder of leaf .jpg>

Layout conversion::

    Conv2D           Keras (kh, kw, in,  out) -> Torch (out, in, kh, kw)
    Conv2DTranspose  Keras (kh, kw, out, in ) -> Torch (in,  out, kh, kw)

Both are a ``(3, 2, 0, 1)`` transpose. Neither needs a spatial flip: Keras and
PyTorch both implement cross-correlation, and Keras' ``conv2d_transpose`` is the
gradient of that same cross-correlation, exactly like ``nn.ConvTranspose2d``.

The script exits non-zero if the maximum absolute deviation between the two
backends exceeds 1e-3, so it can gate a release.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from septosympto.models.unet import UNet

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

PARITY_TOLERANCE = 1e-3


def load_keras(path: str):
    import tensorflow as tf
    from tensorflow.keras.utils import CustomObjectScope

    from tools.metrics import dice_coef, dice_loss, iou

    with CustomObjectScope({"iou": iou, "dice_coef": dice_coef, "dice_loss": dice_loss}):
        return tf.keras.models.load_model(path)


def transfer(keras_model, torch_model: UNet) -> None:
    def of_type(name: str) -> list:
        return [layer for layer in keras_model.layers if type(layer).__name__ == name]

    k_convs = of_type("Conv2D")
    k_deconv = of_type("Conv2DTranspose")
    k_bns = of_type("BatchNormalization")

    t_convs = torch_model.ordered_convs()
    t_deconv = torch_model.ordered_deconvs()
    t_bns = torch_model.ordered_bns()

    assert len(k_convs) == len(t_convs) == 19, (len(k_convs), len(t_convs))
    assert len(k_deconv) == len(t_deconv) == 4
    assert len(k_bns) == len(t_bns) == 18

    with torch.no_grad():
        for kl, tl in zip([*k_convs, *k_deconv], [*t_convs, *t_deconv], strict=True):
            weight, bias = kl.get_weights()
            weight = np.transpose(weight, (3, 2, 0, 1))
            assert weight.shape == tuple(tl.weight.shape), f"{kl.name}: {weight.shape}"
            tl.weight.copy_(torch.from_numpy(weight))
            tl.bias.copy_(torch.from_numpy(bias))

        for kl, tl in zip(k_bns, t_bns, strict=True):
            gamma, beta, mean, var = kl.get_weights()
            tl.weight.copy_(torch.from_numpy(gamma))
            tl.bias.copy_(torch.from_numpy(beta))
            tl.running_mean.copy_(torch.from_numpy(mean))
            tl.running_var.copy_(torch.from_numpy(var))
            tl.eps = float(kl.epsilon)

    print(f"transféré : {len(k_convs)} Conv2D, {len(k_deconv)} Conv2DTranspose, "
          f"{len(k_bns)} BatchNorm")


def leaf_batch(directory: str, n: int, h: int, w: int) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(directory, "*.jpg")))[:n]
    if not files:
        raise SystemExit(f"aucune image .jpg dans {directory}")
    imgs = [cv2.resize(cv2.imread(f), (w, h)).astype(np.float32) / 255.0 for f in files]
    print(f"{len(imgs)} feuilles de validation depuis {directory}")
    return np.stack(imgs)


def save(state: dict, path: str, source: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file

        save_file(
            {k: v.contiguous() for k, v in state.items()},
            path,
            metadata={"source": source, "arch": "septosympto.models.unet.UNet"},
        )
    else:
        torch.save(state, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--keras", default="data/necrosis-model-375.h5")
    p.add_argument("--output", default="data/necrosis-model-375.safetensors")
    p.add_argument("--validate-dir", default=None, help="dossier d'images .jpg de validation")
    p.add_argument("--n-validate", type=int, default=6)
    p.add_argument("--imgsz", type=int, nargs=2, default=[304, 3072], metavar=("H", "W"))
    args = p.parse_args()

    h, w = args.imgsz

    print(f"chargement de {args.keras}…")
    km = load_keras(args.keras)
    tm = UNet().eval()

    n_params = sum(t.numel() for t in tm.parameters())
    n_buffers = sum(b.numel() for b in tm.buffers())
    print(f"paramètres Keras : {km.count_params():,}")
    print(f"paramètres Torch : {n_params:,} (+ buffers BN, total {n_params + n_buffers:,})")

    transfer(km, tm)

    if args.validate_dir:
        x = leaf_batch(args.validate_dir, args.n_validate, h, w)
    else:
        x = np.random.default_rng(0).random((2, h, w, 3), dtype=np.float32)
        print("pas de --validate-dir : parité vérifiée sur du bruit uniforme")

    y_keras = np.stack([np.squeeze(km.predict(xi[None], verbose=0)) for xi in x])

    xt = torch.from_numpy(np.transpose(x, (0, 3, 1, 2)).copy())
    with torch.inference_mode():
        y_torch = torch.sigmoid(tm(xt)).squeeze(1).numpy()

    diff = np.abs(y_keras - y_torch)
    print("\n===== parité Keras vs PyTorch (probabilités) =====")
    print(f"  écart absolu max    : {diff.max():.3e}")
    print(f"  écart absolu moyen  : {diff.mean():.3e}")
    print(f"  pixels différant de plus de 1e-4 : {100 * (diff > 1e-4).mean():.4f} %")

    for th in (0.5, 0.8):
        mk, mt = y_keras > th, y_torch > th
        dice = 2 * (mk & mt).sum() / (mk.sum() + mt.sum()) if (mk.sum() + mt.sum()) else 1.0
        disagree = int((mk != mt).sum())
        print(f"  seuil {th}: pixels en désaccord = {disagree} "
              f"({100 * disagree / mk.size:.6f} %) | Dice entre les deux masques = {dice:.6f}")

    ok = diff.max() < PARITY_TOLERANCE
    print(f"\n  verdict : {'PARITÉ OK' if ok else 'ÉCART SUSPECT — ne pas livrer'}")

    save(tm.state_dict(), args.output, os.path.basename(args.keras))
    print(f"  écrit : {args.output} ({os.path.getsize(args.output) / 1e6:.1f} Mo)")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
