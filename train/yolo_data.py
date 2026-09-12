"""Export the necrosis pool in the layout Ultralytics' semantic trainer reads.

The pool and the split are the project's — :func:`train.data.load_pool` and
:func:`train.data.scan_grouped_split` with the same seed and fractions as a
PyTorch run — so a YOLO run and a U-Net run train on the same leaves and are
validated on the same leaves. The export is the only thing Ultralytics-specific:

    <root>/data.yaml
    <root>/images/{train,val,test}/<leaf>__t<i>.png
    <root>/masks/{train,val,test}/<leaf>__t<i>.png   uint8, 0 = background, 1 = necrosis

``masks/`` mirrors ``images/`` by stem, which is how ``SemanticDataset`` pairs
them. Masks are written as class ids, not ``{0, 255}``: in a semantic mask 255 is
the *ignore* label, so a raw ``{0, 255}`` PNG would train on background only.

**Why tiles.** Ultralytics' semantic trainer scales the *short* side of every
training image to ``imgsz`` (Cityscapes-style scale-and-crop), while validation
scales the *long* side. Fed a 384x3072 canvas with ``imgsz=3072`` it trains on
x8 zooms of the leaf centre and validates on the whole strip — a distribution
gap that collapses validation after the first epochs. Square tiles of the canvas
height, with ``imgsz`` equal to the tile, make both paths the identity: scale 1
in training and in validation. The model is fully convolutional, so it is run on
the full canvas at inference, as semantic nets trained on crops always are.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from train.data import Sample, decode_image, decode_mask

CLASS_NAME = "necrosis"


def tile_spans(width: int, tile: int) -> list[tuple[int, int]]:
    """``(x0, x1)`` windows of ``tile`` px covering ``width``; the last one is
    pulled back to end at ``width`` when it does not divide evenly."""
    if tile <= 0 or tile >= width:
        return [(0, width)]
    starts = list(range(0, width - tile + 1, tile))
    if starts[-1] + tile < width:
        starts.append(width - tile)
    return [(x, x + tile) for x in starts]


def export_yolo_dataset(
    root: str | Path,
    train: list[Sample],
    val: list[Sample],
    test: list[Sample] = (),
    tile: int = 384,
) -> Path:
    """Write the three folds under ``root``, tiled, and return the path of ``data.yaml``.

    Every leaf is cut into ``tile``-wide windows over its full height (a 384x3072
    canvas at ``tile=384`` gives 8 square tiles). ``root`` is recreated from
    scratch, so a stale export never mixes with a new split. The test fold is
    exported too, so a held-out evaluation can be run with the Ultralytics CLI,
    but the trainer never reads it.
    """
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    for fold, samples in (("train", train), ("val", val), ("test", test)):
        _write_fold(root, fold, samples, tile)

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


def _write_fold(root: Path, fold: str, samples: list[Sample], tile: int) -> None:
    images = root / "images" / fold
    masks = root / "masks" / fold
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        image = decode_image(sample.image_bytes)
        mask = decode_mask(sample.mask_bytes).astype(np.uint8)
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"{sample.image}: mask {mask.shape} does not match image {image.shape[:2]}"
            )
        stem = Path(sample.image).stem
        for i, (x0, x1) in enumerate(tile_spans(image.shape[1], tile)):
            _write_png(images / f"{stem}__t{i}.png", image[:, x0:x1])
            _write_png(masks / f"{stem}__t{i}.png", mask[:, x0:x1])


def _write_png(path: Path, array: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", np.ascontiguousarray(array))
    if not ok:
        raise RuntimeError(f"could not encode {path.name}")
    path.write_bytes(encoded.tobytes())
