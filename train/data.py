"""Loading, splitting, and augmenting the necrosis dataset.

Two things are fixed relative to how v1 was trained.

**Masks are binarised, not rescaled.** The Roboflow PNGs paint necrosis magenta,
``(255, 0, 124)``. v1's training read them as greyscale and divided by 255,
turning the positive class into a target of 0.354 and capping Dice at 0.523.
Here a pixel is necrotic when its strongest channel reaches half intensity.

**The split is grouped by scan.** The Roboflow train/valid split leaks: 24 of the
305 scans have leaves on both sides. Necrosis is a local, dense task so the
measured effect is small, but a clean split costs nothing to do right. All 375
images are pooled and re-split so that every leaf of a scan lands in the same
fold, seeded for reproducibility, into train / val / test.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MASK_CHANNEL_THRESHOLD = 128


def seed_worker(_worker_id: int) -> None:
    """Give each DataLoader worker a fresh augmentation RNG every epoch.

    The datasets draw flips from their own ``np.random.default_rng`` instance, which
    PyTorch's automatic per-worker seeding leaves untouched — it reseeds only the
    global numpy/torch/random state. So without this every worker forks the same
    dataset RNG state: flips correlate across workers and repeat each epoch.
    ``torch.initial_seed()`` is unique per worker and per epoch, so reseeding from
    it restores independent augmentation. Pair it with a seeded DataLoader
    ``generator`` to keep the whole thing reproducible.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._rng = np.random.default_rng(torch.initial_seed() % 2**32)


@dataclass(frozen=True)
class Sample:
    image: str
    scan: str
    image_bytes: bytes
    mask_bytes: bytes


def _scan_of(filename: str) -> str:
    stem = Path(filename).name
    stem = stem.split("_jpg.rf.")[0]
    return stem.split("__")[0]


def load_pool(dataset: str | Path) -> list[Sample]:
    """Every image/mask pair in a Roboflow zip, both splits pooled."""
    samples = []
    with zipfile.ZipFile(dataset) as archive:
        images = sorted(
            n for n in archive.namelist()
            if "/img/" in n and n.endswith(".jpg") and "__MACOSX" not in n
        )
        for name in images:
            mask_name = name.replace("/img/", "/mask/").replace(".jpg", ".png")
            if mask_name not in archive.namelist():
                continue
            samples.append(
                Sample(
                    image=Path(name).name,
                    scan=_scan_of(name),
                    image_bytes=archive.read(name),
                    mask_bytes=archive.read(mask_name),
                )
            )
    if not samples:
        raise ValueError(f"no image/mask pairs found in {dataset}")
    return samples


def scan_grouped_split(
    samples: list[Sample], val_fraction: float, test_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Partition samples so that no scan spans two folds."""
    scans = sorted({s.scan for s in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(scans)

    n_test = int(round(len(scans) * test_fraction))
    n_val = int(round(len(scans) * val_fraction))
    test_scans = set(scans[:n_test])
    val_scans = set(scans[n_test : n_test + n_val])

    train, val, test = [], [], []
    for sample in samples:
        if sample.scan in test_scans:
            test.append(sample)
        elif sample.scan in val_scans:
            val.append(sample)
        else:
            train.append(sample)
    return train, val, test


def decode_mask(mask_bytes: bytes) -> np.ndarray:
    array = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if array.ndim == 3:
        return array.max(axis=2) >= MASK_CHANNEL_THRESHOLD
    return array >= MASK_CHANNEL_THRESHOLD


def decode_image(image_bytes: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)


class NecrosisDataset(Dataset):
    """Leaf images and binary necrosis masks at a fixed size.

    Images are BGR in ``[0, 1]``, the order the segmenter expects. Augmentation is
    flips only: leaves are scanned horizontally and are symmetric enough that a
    horizontal or vertical flip is a valid leaf, while rotations or scale changes
    would fight the fixed-aspect resize the model already imposes.
    """

    def __init__(
        self,
        samples: list[Sample],
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
        image = cv2.resize(decode_image(sample.image_bytes), (w, h)).astype(np.float32) / 255.0
        mask = cv2.resize(
            decode_mask(sample.mask_bytes).astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

        if self.hflip and self._rng.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if self.vflip and self._rng.random() < 0.5:
            image, mask = image[::-1], mask[::-1]

        image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return image_t, mask_t
