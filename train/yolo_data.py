"""Export the necrosis pool in the layout Ultralytics' semantic trainer reads.

The pool and the split are the project's — :func:`train.data.load_pool` and
:func:`train.data.scan_grouped_split` with the same seed and fractions as a
PyTorch run — so a YOLO run and a U-Net run train on the same leaves and are
validated on the same leaves. The export is the only thing Ultralytics-specific:

    <root>/data.yaml
    <root>/images/{train,val,test}/<leaf>.png
    <root>/masks/{train,val,test}/<leaf>.png      uint8, 0 = background, 1 = necrosis

``masks/`` mirrors ``images/`` by stem, which is how ``SemanticDataset`` pairs
them. Masks are written as class ids, not ``{0, 255}``: in a semantic mask 255 is
the *ignore* label, so a raw ``{0, 255}`` PNG would train on background only.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from train.data import Sample, decode_mask

CLASS_NAME = "necrosis"


def export_yolo_dataset(
    root: str | Path,
    train: list[Sample],
    val: list[Sample],
    test: list[Sample] = (),
) -> Path:
    """Write the three folds under ``root`` and return the path of ``data.yaml``.

    ``root`` is recreated from scratch, so a stale export never mixes with a new
    split. The test fold is exported too, so a held-out evaluation can be run
    with the Ultralytics CLI, but the trainer never reads it.
    """
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    for fold, samples in (("train", train), ("val", val), ("test", test)):
        _write_fold(root, fold, samples)

    yaml = root / "data.yaml"
    yaml.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: images/train",
                "val: images/val",
                *(["test: images/test"] if test else []),
                "masks_dir: masks",
                "nc: 1",
                f"names: {{0: {CLASS_NAME}}}",
                "",
            ]
        )
    )
    return yaml


def _write_fold(root: Path, fold: str, samples: list[Sample]) -> None:
    images = root / "images" / fold
    masks = root / "masks" / fold
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        (images / sample.image).write_bytes(sample.image_bytes)
        mask = decode_mask(sample.mask_bytes).astype(np.uint8)
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            raise RuntimeError(f"could not encode the mask of {sample.image}")
        (masks / (Path(sample.image).stem + ".png")).write_bytes(encoded.tobytes())
