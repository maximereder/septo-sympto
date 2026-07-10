import cv2
import numpy as np
import pytest

from septosympto.leaf import (
    CM_PER_INCH,
    Scan,
    crop,
    find_leaves,
    read_px_per_cm,
    resize_anisotropy,
    tissue_mask,
)


def synthetic_scan(leaves: list[tuple[int, int, int, int]], size=(600, 4000)) -> Scan:
    """A white background with saturated green rectangles standing in for leaves."""
    bgr = np.full((*size, 3), 255, np.uint8)
    for x, y, w, h in leaves:
        bgr[y : y + h, x : x + w] = (40, 160, 60)
    return Scan(image="synthetic", bgr=bgr, px_per_cm=472.44)


def test_tissue_mask_ignores_the_white_background():
    scan = synthetic_scan([(100, 100, 3000, 300)])
    mask = tissue_mask(scan.bgr)
    assert mask[200, 500]
    assert not mask[10, 10]


def test_tissue_mask_keeps_dark_lesions_inside_the_leaf():
    """Necrosis and pycnidia are dark but saturated; they must not be dropped."""
    scan = synthetic_scan([(100, 100, 3000, 300)])
    scan.bgr[200:220, 500:520] = (10, 70, 30)
    assert tissue_mask(scan.bgr)[210, 510]


def test_find_leaves_indexes_from_one_top_to_bottom():
    scan = synthetic_scan([(0, 400, 3900, 120), (0, 100, 3900, 120)])
    leaves = find_leaves(scan)
    assert [leaf.leaf_index for leaf in leaves] == [1, 2]
    assert leaves[0].bbox[1] < leaves[1].bbox[1]


def test_find_leaves_ignores_specks_below_the_area_threshold():
    scan = synthetic_scan([(0, 100, 3900, 120), (10, 500, 20, 20)])
    assert len(find_leaves(scan)) == 1


def test_leaf_id_is_display_only():
    scan = synthetic_scan([(0, 100, 3900, 120)])
    leaf = find_leaves(scan)[0]
    assert leaf.leaf_id == "synthetic_1"


def test_area_counts_pixels_and_does_not_double_count_holes():
    """v1 summed contourArea over RETR_TREE contours, adding holes instead of subtracting."""
    scan = synthetic_scan([(0, 100, 3900, 200)])
    scan.bgr[150:250, 1000:2000] = 255

    leaf = find_leaves(scan)[0]
    solid = 3900 * 200
    assert leaf.area_px < solid
    assert leaf.area_cm2(472.44) == pytest.approx(leaf.area_px / 472.44**2)


def test_clipped_leaves_are_flagged_on_both_sides():
    scan = synthetic_scan([(0, 100, 4000, 200)])
    leaf = find_leaves(scan)[0]
    assert "clipped-left" in leaf.qc
    assert "clipped-right" in leaf.qc


def test_interior_leaf_carries_no_clipping_flag():
    scan = synthetic_scan([(100, 100, 3000, 200)])
    assert find_leaves(scan)[0].qc == ()


def test_crop_blanks_everything_outside_the_leaf():
    scan = synthetic_scan([(100, 100, 3000, 200)])
    leaf = find_leaves(scan)[0]
    patch = crop(scan, leaf)
    assert patch.shape[:2] == (200, 3000)
    assert (patch[100, 1500] != 255).any()


def test_resize_anisotropy_is_one_when_proportions_already_match():
    scan = synthetic_scan([(0, 100, 3040, 304)], size=(600, 3040))
    leaf = find_leaves(scan)[0]
    assert resize_anisotropy(leaf, 304, 3040) == pytest.approx(1.0, abs=1e-6)


def test_resize_anisotropy_grows_as_the_leaf_gets_thinner():
    thick = find_leaves(synthetic_scan([(0, 100, 3072, 300)]))[0]
    thin = find_leaves(synthetic_scan([(0, 100, 3072, 170)]))[0]
    assert resize_anisotropy(thin, 304, 3072) > resize_anisotropy(thick, 304, 3072)


def test_read_px_per_cm_from_a_tiff(tmp_path):
    from PIL import Image

    path = tmp_path / "scan.tif"
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(path, dpi=(1200, 1200))
    assert read_px_per_cm(path) == pytest.approx(1200 / CM_PER_INCH, abs=0.01)


def test_read_px_per_cm_returns_none_when_absent(tmp_path):
    path = tmp_path / "leaf.png"
    cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))
    assert read_px_per_cm(path) is None
