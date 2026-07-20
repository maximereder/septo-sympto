import cv2
import numpy as np
import torch

from train.count_data import (
    PointSample,
    PycnidiaPointDataset,
    collate_points,
    load_points_pool,
)


def write_leaf(directory, base_id, rf_hash, points_norm):
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "labels").mkdir(parents=True, exist_ok=True)
    name = f"{base_id}_jpg.rf.{rf_hash}"
    cv2.imwrite(str(directory / "images" / f"{name}.jpg"), np.full((32, 128, 3), 128, np.uint8))
    lines = [f"0 {x} {y} 0.002 0.002" for x, y in points_norm]
    (directory / "labels" / f"{name}.txt").write_text("\n".join(lines))


def test_boxes_become_centre_points(tmp_path):
    write_leaf(tmp_path / "d", "scanA__1", "aaa", [(0.5, 0.25), (0.1, 0.9)])
    pool = load_points_pool([tmp_path / "d"])
    assert len(pool) == 1
    assert pool[0].points_norm.shape == (2, 2)
    assert pool[0].points_norm[0].tolist() == [0.5, 0.25]


def test_dedup_keeps_one_copy_per_base_leaf(tmp_path):
    write_leaf(tmp_path / "d", "scanA__1", "aaa", [(0.5, 0.5)])
    write_leaf(tmp_path / "d", "scanA__1", "bbb", [(0.5, 0.5)])
    write_leaf(tmp_path / "d", "scanA__2", "ccc", [(0.3, 0.3)])
    pool = load_points_pool([tmp_path / "d"])
    assert len(pool) == 2


def test_scan_is_the_prefix_before_double_underscore(tmp_path):
    write_leaf(tmp_path / "d", "Soi_LGA_2_Soi_1__3", "aaa", [(0.5, 0.5)])
    pool = load_points_pool([tmp_path / "d"])
    assert pool[0].scan == "Soi_LGA_2_Soi_1"


def test_an_empty_label_means_zero_points(tmp_path):
    (tmp_path / "d" / "images").mkdir(parents=True)
    (tmp_path / "d" / "labels").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "d" / "images" / "s__1_jpg.rf.x.jpg"),
                np.zeros((16, 16, 3), np.uint8))
    (tmp_path / "d" / "labels" / "s__1_jpg.rf.x.txt").write_text("")
    pool = load_points_pool([tmp_path / "d"])
    assert pool[0].points_norm.shape == (0, 2)


def test_an_image_without_a_label_is_skipped(tmp_path):
    (tmp_path / "d" / "img").mkdir(parents=True)
    (tmp_path / "d" / "labels").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "d" / "img" / "kept__1.png"), np.zeros((16, 16, 3), np.uint8))
    (tmp_path / "d" / "labels" / "kept__1.txt").write_text("0 0.5 0.5 0.01 0.01")
    cv2.imwrite(str(tmp_path / "d" / "img" / "skipped__1.png"), np.zeros((16, 16, 3), np.uint8))
    pool = load_points_pool([tmp_path / "d"])
    bases = {s.image for s in pool}
    assert "kept__1.png" in bases
    assert "skipped__1.png" not in bases


def test_dataset_scales_points_to_pixels():
    sample = PointSample(
        image="s__1", scan="s", image_path="",
        points_norm=np.array([[0.5, 0.5]], np.float32),
    )
    ds = PycnidiaPointDataset([sample], imgsz=(64, 256))
    ds.samples[0].points_norm  # noqa: B018
    import unittest.mock

    with unittest.mock.patch(
        "train.count_data.cv2.imread", return_value=np.zeros((10, 10, 3), np.uint8)
    ):
        _, points = ds[0]
    assert points.tolist() == [[128.0, 32.0]]


def test_hflip_moves_points_with_the_image():
    sample = PointSample(
        image="s__1", scan="s", image_path="",
        points_norm=np.array([[0.1, 0.5]], np.float32),
    )
    ds = PycnidiaPointDataset([sample], imgsz=(64, 256), hflip=True, seed=3)
    import unittest.mock

    left = np.zeros((10, 10, 3), np.uint8)
    with unittest.mock.patch("train.count_data.cv2.imread", return_value=left):
        seen_x = {round(ds[0][1][0, 0].item()) for _ in range(10)}
    assert len(seen_x) == 2


def test_collate_stacks_images_and_keeps_points_as_a_list():
    batch = [
        (torch.zeros(3, 8, 8), torch.tensor([[1.0, 2.0]])),
        (torch.zeros(3, 8, 8), torch.tensor([[3.0, 4.0], [5.0, 6.0]])),
    ]
    images, points = collate_points(batch)
    assert images.shape == (2, 3, 8, 8)
    assert [len(p) for p in points] == [1, 2]
