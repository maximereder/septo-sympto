import zipfile

import cv2
import numpy as np
import pytest

from train.data import (
    NecrosisDataset,
    Sample,
    decode_mask,
    load_pool,
    scan_grouped_split,
)


def magenta_png(positive: np.ndarray) -> bytes:
    h, w = positive.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[positive] = (124, 0, 255)
    return cv2.imencode(".png", rgb)[1].tobytes()


def jpg(h=32, w=320) -> bytes:
    return cv2.imencode(".jpg", np.full((h, w, 3), 128, np.uint8))[1].tobytes()


def build_zip(path, leaves: list[tuple[str, int]]):
    """leaves: (scan, leaf_index) pairs, split arbitrarily into train/valid."""
    with zipfile.ZipFile(path, "w") as archive:
        for i, (scan, idx) in enumerate(leaves):
            split = "train" if i % 2 == 0 else "valid"
            name = f"ds/{split}/img/{scan}__{idx}.jpg"
            mask = name.replace("/img/", "/mask/").replace(".jpg", ".png")
            positive = np.zeros((32, 320), bool)
            positive[8:24, 80:240] = True
            archive.writestr(name, jpg())
            archive.writestr(mask, magenta_png(positive))


def test_magenta_mask_is_binarised_not_rescaled():
    positive = np.zeros((10, 10), bool)
    positive[2:8, 2:8] = True
    mask = decode_mask(magenta_png(positive))
    assert mask.dtype == bool
    assert mask.sum() == 36
    assert mask[5, 5]
    assert not mask[0, 0]


def test_load_pool_reads_both_splits(tmp_path):
    path = tmp_path / "ds.zip"
    build_zip(path, [("scanA", 1), ("scanA", 2), ("scanB", 1), ("scanC", 1)])
    pool = load_pool(path)
    assert len(pool) == 4
    assert {s.scan for s in pool} == {"scanA", "scanB", "scanC"}


def test_split_keeps_every_leaf_of_a_scan_together(tmp_path):
    path = tmp_path / "ds.zip"
    leaves = [(f"scan{n}", i) for n in range(20) for i in (1, 2)]
    build_zip(path, leaves)
    pool = load_pool(path)

    train, val, test = scan_grouped_split(pool, val_fraction=0.2, test_fraction=0.2, seed=0)
    fold_of = {}
    for name, fold in [("train", train), ("val", val), ("test", test)]:
        for s in fold:
            fold_of.setdefault(s.scan, set()).add(name)
    assert all(len(folds) == 1 for folds in fold_of.values())


def test_split_is_deterministic_for_a_seed(tmp_path):
    path = tmp_path / "ds.zip"
    build_zip(path, [(f"scan{n}", 1) for n in range(30)])
    pool = load_pool(path)
    a = scan_grouped_split(pool, 0.2, 0.2, seed=7)
    b = scan_grouped_split(pool, 0.2, 0.2, seed=7)
    assert [s.image for s in a[0]] == [s.image for s in b[0]]


def test_dataset_yields_chw_image_and_binary_mask():
    positive = np.zeros((32, 320), bool)
    positive[8:24, 80:240] = True
    sample = Sample(image="s__1.jpg", scan="s", image_bytes=jpg(), mask_bytes=magenta_png(positive))
    ds = NecrosisDataset([sample], imgsz=(32, 320))
    image, mask = ds[0]
    assert image.shape == (3, 32, 320)
    assert mask.shape == (1, 32, 320)
    assert set(np.unique(mask.numpy())).issubset({0.0, 1.0})


def test_flip_augmentation_preserves_image_mask_alignment():
    positive = np.zeros((32, 320), bool)
    positive[:, :160] = True
    sample = Sample(image="s__1.jpg", scan="s",
                    image_bytes=cv2.imencode(".jpg", _left_bright())[1].tobytes(),
                    mask_bytes=magenta_png(positive))
    ds = NecrosisDataset([sample] * 20, imgsz=(32, 320), hflip=True, seed=1)
    for i in range(len(ds)):
        image, mask = ds[i]
        bright_left = image[:, :, :160].mean() > image[:, :, 160:].mean()
        mask_left = mask[:, :, :160].mean() > mask[:, :, 160:].mean()
        assert bright_left == mask_left


def _left_bright(h=32, w=320) -> np.ndarray:
    img = np.full((h, w, 3), 40, np.uint8)
    img[:, : w // 2] = 220
    return img


def test_empty_dataset_raises(tmp_path):
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w"):
        pass
    with pytest.raises(ValueError, match="no image/mask pairs"):
        load_pool(path)
