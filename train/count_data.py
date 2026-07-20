"""Loading, splitting, and augmenting the pycnidia point dataset.

The annotations are YOLO boxes with a median width of 4 px. A box that small
encodes a point and three noisy numbers, so only the centre is kept.

The images ship pre-augmented (``*-aug-x3``, flips baked in) and split by
Roboflow, which leaks: mirrored copies of the same leaf sit on both sides. Here
the leaves are pooled and **deduplicated by base id** — the filename before
Roboflow's ``_jpg.rf.<hash>`` suffix — so each physical leaf appears once, then
re-split grouped by scan so no scan spans two folds. Augmentation is applied on
the fly instead, after the split, which is the only order that cannot leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from train.data import scan_grouped_split


@dataclass(frozen=True)
class PointSample:
    image: str
    scan: str
    image_path: str
    points_norm: np.ndarray


def _base_id(path: Path) -> str:
    return path.name.split("_jpg.rf.")[0]


def _read_points(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.empty((0, 2), np.float32)
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            rows.append((float(parts[1]), float(parts[2])))
    return np.array(rows, np.float32) if rows else np.empty((0, 2), np.float32)


def load_points_pool(directories: list[str | Path]) -> list[PointSample]:
    """Every leaf across ``directories``, deduplicated by base id, one sample each.

    Handles both layouts: the Roboflow export (``images/*.jpg``) and the native
    letterbox set (``img/*.png``). An image is only taken when it has a label
    file, so the shared native ``img/`` — which also holds necrosis-only leaves —
    contributes exactly the pycnidia-annotated leaves, not phantom zero-count ones.
    """
    by_base: dict[str, PointSample] = {}
    for directory in directories:
        directory = Path(directory)
        images = sorted([*(directory / "images").glob("*.jpg"),
                         *(directory / "img").glob("*.png")])
        for image_path in images:
            label_path = directory / "labels" / (image_path.stem + ".txt")
            if not label_path.exists():
                continue
            base = _base_id(image_path)
            if base in by_base:
                continue
            by_base[base] = PointSample(
                image=base,
                scan=base.split("__")[0],
                image_path=str(image_path),
                points_norm=_read_points(label_path),
            )
    if not by_base:
        raise ValueError(f"no image/label pairs found in {directories}")
    return list(by_base.values())


class PycnidiaPointDataset(Dataset):
    """Leaf images and pycnidia points at a fixed size.

    Yields ``(image, points)``: image ``(3, H, W)`` float32 BGR in ``[0, 1]``,
    points ``(N, 2)`` in full-resolution pixel coordinates ``(x, y)``. N varies
    per leaf, so batching needs :func:`collate_points`.
    """

    def __init__(
        self,
        samples: list[PointSample],
        imgsz: tuple[int, int] = (384, 3072),
        hflip: bool = False,
        vflip: bool = False,
        seed: int = 0,
    ) -> None:
        self.samples = samples
        self.imgsz = imgsz
        self.hflip = hflip
        self.vflip = vflip
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        h, w = self.imgsz
        image = cv2.resize(cv2.imread(sample.image_path), (w, h)).astype(np.float32) / 255.0
        points = sample.points_norm.copy()
        points[:, 0] *= w
        points[:, 1] *= h

        if self.hflip and self._rng.random() < 0.5:
            image = image[:, ::-1]
            points[:, 0] = w - 1 - points[:, 0]
        if self.vflip and self._rng.random() < 0.5:
            image = image[::-1]
            points[:, 1] = h - 1 - points[:, 1]

        image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        points_t = torch.from_numpy(np.ascontiguousarray(points)).float()
        return image_t, points_t


def collate_points(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Stack images; keep points as a list, since their counts differ."""
    images = torch.stack([item[0] for item in batch])
    points = [item[1] for item in batch]
    return images, points


def split_pool(
    pool: list[PointSample], val_fraction: float, test_fraction: float, seed: int
) -> tuple[list[PointSample], list[PointSample], list[PointSample]]:
    return scan_grouped_split(pool, val_fraction, test_fraction, seed)
