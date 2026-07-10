import numpy as np
import pytest

from septosympto.leaf import Leaf
from septosympto.measure import LeafMeasurement, measure_leaf

PX_PER_CM = 472.44


def make_leaf(mask: np.ndarray, image="scan", index=1, qc=()) -> Leaf:
    ys, xs = np.where(mask)
    bbox = (
        int(xs.min()), int(ys.min()),
        int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1),
    )
    return Leaf(image=image, leaf_index=index, bbox=bbox, mask=mask, qc=qc)


def full_leaf(h=400, w=4000) -> Leaf:
    return make_leaf(np.ones((h, w), bool))


def test_leaf_area_in_cm2_uses_the_scale():
    leaf = full_leaf(400, 4000)
    m = measure_leaf(leaf, np.zeros((400, 4000), bool), None, PX_PER_CM)
    assert m.leaf_area_px == 400 * 4000
    assert m.leaf_area_cm2 == pytest.approx(400 * 4000 / PX_PER_CM**2, abs=1e-3)


def test_necrosis_outside_the_leaf_is_ignored():
    leaf_mask = np.zeros((400, 4000), bool)
    leaf_mask[:, :2000] = True
    leaf = make_leaf(leaf_mask)

    necrosis = np.zeros((400, 4000), bool)
    necrosis[100:200, 2500:3500] = True

    m = measure_leaf(leaf, necrosis, None, PX_PER_CM)
    assert m.necrosis_count == 0
    assert m.necrosis_area_px == 0


def test_two_separated_lesions_are_counted_separately():
    leaf = full_leaf()
    necrosis = np.zeros((400, 4000), bool)
    necrosis[50:150, 100:300] = True
    necrosis[50:150, 1000:1200] = True

    m = measure_leaf(leaf, necrosis, None, PX_PER_CM)
    assert m.necrosis_count == 2
    assert m.necrosis_area_px == 2 * 100 * 200


def test_small_specks_are_filtered_by_the_mm2_floor():
    leaf = full_leaf()
    necrosis = np.zeros((400, 4000), bool)
    necrosis[10:13, 10:13] = True

    m = measure_leaf(leaf, necrosis, None, PX_PER_CM, min_lesion_area_mm2=0.135)
    assert m.necrosis_count == 0


def test_the_area_floor_scales_with_resolution():
    """A fixed mm² floor keeps the same physical lesion at two resolutions."""
    leaf_hi = full_leaf(400, 4000)
    necrosis_hi = np.zeros((400, 4000), bool)
    necrosis_hi[50:70, 50:70] = True

    hi = measure_leaf(leaf_hi, necrosis_hi, None, 944.88, min_lesion_area_mm2=0.1)
    lo = measure_leaf(leaf_hi, necrosis_hi, None, 236.22, min_lesion_area_mm2=0.1)

    assert hi.necrosis_count == 0
    assert lo.necrosis_count == 1


def test_necrosis_ratio_is_area_over_leaf_area():
    leaf = full_leaf(400, 4000)
    necrosis = np.zeros((400, 4000), bool)
    necrosis[:, :400] = True
    m = measure_leaf(leaf, necrosis, None, PX_PER_CM, min_lesion_area_mm2=0.0)
    assert m.necrosis_area_ratio == pytest.approx(0.1, abs=1e-3)


def test_pycnidia_densities_use_leaf_and_necrosis_area():
    leaf = full_leaf(400, 4000)
    necrosis = np.zeros((400, 4000), bool)
    necrosis[:, :2000] = True
    points = np.array([[x, 10] for x in range(0, 1000, 10)], float)

    m = measure_leaf(leaf, necrosis, points, PX_PER_CM, min_lesion_area_mm2=0.0)
    assert m.pycnidia_count == 100
    assert m.pycnidia_per_leaf_cm2 == pytest.approx(100 / m.leaf_area_cm2, abs=1e-2)
    assert m.pycnidia_per_necrosis_cm2 == pytest.approx(100 / m.necrosis_area_cm2, abs=1e-2)


def test_no_counter_yields_zero_pycnidia_without_dividing_by_zero():
    leaf = full_leaf()
    m = measure_leaf(leaf, np.zeros((400, 4000), bool), None, PX_PER_CM)
    assert m.pycnidia_count == 0
    assert m.pycnidia_per_leaf_cm2 == 0.0
    assert m.pycnidia_per_necrosis_cm2 == 0.0


def test_mask_shape_must_match_the_leaf():
    leaf = full_leaf(400, 4000)
    with pytest.raises(ValueError, match="does not match"):
        measure_leaf(leaf, np.zeros((10, 10), bool), None, PX_PER_CM)


def test_columns_and_row_are_consistent():
    leaf = full_leaf()
    m = measure_leaf(leaf, np.zeros((400, 4000), bool), None, PX_PER_CM)
    assert set(m.as_row()) == set(LeafMeasurement.columns())
    assert "pycnidia_area" not in LeafMeasurement.columns()
