import numpy as np
import pytest

from septosympto.leaf import Scan
from septosympto.pipeline import MissingScaleError, analyze_scan, iter_scan_files


class HalfNecrotic:
    """Segmenter that marks the left half of any patch as necrotic."""

    def segment(self, leaf_bgr: np.ndarray) -> np.ndarray:
        h, w = leaf_bgr.shape[:2]
        mask = np.zeros((h, w), bool)
        mask[:, : w // 2] = True
        return mask


class FixedCounter:
    def __init__(self, n: int) -> None:
        self.n = n

    def count(self, leaf_bgr: np.ndarray) -> np.ndarray:
        return np.zeros((self.n, 2))


def scan_with_leaves(n_leaves=2, px_per_cm=472.44) -> Scan:
    bgr = np.full((600, 4000, 3), 255, np.uint8)
    for i in range(n_leaves):
        y = 100 + i * 200
        bgr[y : y + 120, 200:3800] = (40, 160, 60)
    return Scan(image="synthetic", bgr=bgr, px_per_cm=px_per_cm)


def test_one_measurement_per_leaf():
    scan = scan_with_leaves(3)
    out = analyze_scan(scan, HalfNecrotic())
    assert len(out) == 3
    assert [m.leaf_index for m in out] == [1, 2, 3]


def test_necrosis_is_measured_through_the_segmenter():
    scan = scan_with_leaves(1)
    m = analyze_scan(scan, HalfNecrotic(), min_lesion_area_mm2=0.0)[0]
    assert m.necrosis_area_px > 0
    assert m.necrosis_area_ratio == pytest.approx(0.5, abs=0.05)


def test_counter_is_optional_and_defaults_to_zero():
    scan = scan_with_leaves(1)
    without = analyze_scan(scan, HalfNecrotic())[0]
    assert without.pycnidia_count == 0

    with_counter = analyze_scan(scan, HalfNecrotic(), FixedCounter(42))[0]
    assert with_counter.pycnidia_count == 42


def test_missing_scale_is_refused_not_guessed():
    scan = scan_with_leaves(1, px_per_cm=None)
    with pytest.raises(MissingScaleError, match="no resolution"):
        analyze_scan(scan, HalfNecrotic())


def test_px_per_cm_override_is_used_when_metadata_is_absent():
    scan = scan_with_leaves(1, px_per_cm=None)
    m = analyze_scan(scan, HalfNecrotic(), px_per_cm=472.44)[0]
    assert m.px_per_cm == pytest.approx(472.44)


def test_necrosis_stays_within_leaf_bounds():
    scan = scan_with_leaves(2)
    out = analyze_scan(scan, HalfNecrotic(), min_lesion_area_mm2=0.0)
    for m in out:
        assert m.necrosis_area_px <= m.leaf_area_px


def test_iter_scan_files_is_sorted_and_extension_filtered(tmp_path):
    (tmp_path / "b.tif").touch()
    (tmp_path / "a.tif").touch()
    (tmp_path / "note.txt").touch()
    (tmp_path / "c.TIF").touch()
    files = iter_scan_files(tmp_path, ".tif")
    assert [f.name for f in files] == ["a.tif", "b.tif", "c.TIF"]
