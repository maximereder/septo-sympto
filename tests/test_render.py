import cv2
import numpy as np

from septosympto.leaf import Leaf
from septosympto.measure import measure_leaf
from septosympto.pipeline import LeafAnalysis
from septosympto.render import necrosis_mask_image, overlay_image, save_analysis


def an_analysis(image="scan", index=1) -> LeafAnalysis:
    h, w = 120, 1000
    leaf_mask = np.zeros((h, w), bool)
    leaf_mask[10:110, 50:950] = True
    leaf = Leaf(image=image, leaf_index=index, bbox=(0, 0, w, h), mask=leaf_mask, qc=())

    patch = np.full((h, w, 3), 200, np.uint8)
    necrosis_patch = np.zeros((h, w), bool)
    necrosis_patch[30:80, 200:400] = True

    measurement = measure_leaf(leaf, necrosis_patch & leaf_mask, None, 472.44)
    return LeafAnalysis(leaf, patch, necrosis_patch, None, measurement)


def test_mask_image_is_binary_and_leaf_shaped():
    img = necrosis_mask_image(an_analysis())
    assert img.shape == (120, 1000)
    assert set(np.unique(img)).issubset({0, 255})
    assert img[50, 300] == 255


def test_mask_is_clipped_to_the_leaf_silhouette():
    h, w = 120, 1000
    leaf_mask = np.zeros((h, w), bool)
    leaf_mask[:, :500] = True
    leaf = Leaf(image="s", leaf_index=1, bbox=(0, 0, w, h), mask=leaf_mask, qc=())

    necrosis_patch = np.ones((h, w), bool)
    measurement = measure_leaf(leaf, necrosis_patch & leaf_mask, None, 472.44)
    analysis = LeafAnalysis(leaf, np.zeros((h, w, 3), np.uint8), necrosis_patch, None, measurement)

    img = necrosis_mask_image(analysis)
    assert img[60, 100] == 255
    assert img[60, 800] == 0


def test_overlay_keeps_the_crop_size_and_draws_something():
    analysis = an_analysis()
    out = overlay_image(analysis)
    assert out.shape == analysis.patch.shape
    assert (out != analysis.patch).any()


def test_overlay_marks_pycnidia_points_when_present():
    base = an_analysis()
    points = np.array([[300.0, 55.0], [320.0, 60.0]])
    analysis = LeafAnalysis(base.leaf, base.patch, base.necrosis_patch, points, base.measurement)
    out = overlay_image(analysis)
    assert (out[:, :, 0] == 255).any()


def test_save_analysis_writes_overlay_and_mask(tmp_path):
    save_analysis(an_analysis(image="CORS", index=3), tmp_path)
    assert (tmp_path / "CORS_3_overlay.jpg").exists()
    assert (tmp_path / "CORS_3_mask.png").exists()
    mask = cv2.imread(str(tmp_path / "CORS_3_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (120, 1000)


def test_save_analysis_can_skip_the_mask(tmp_path):
    save_analysis(an_analysis(), tmp_path, masks=False)
    assert (tmp_path / "scan_1_overlay.jpg").exists()
    assert not (tmp_path / "scan_1_mask.png").exists()
