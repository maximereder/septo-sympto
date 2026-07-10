import numpy as np
import pytest

from septosympto.eval import (
    area_ratio,
    counting_report,
    dice,
    iou,
    match_points,
    segmentation_report,
)


def test_dice_and_iou_on_perfect_overlap():
    m = np.zeros((10, 10), bool)
    m[2:6, 2:6] = True
    assert dice(m, m) == pytest.approx(1.0)
    assert iou(m, m) == pytest.approx(1.0)


def test_dice_and_iou_on_disjoint_masks():
    a, b = np.zeros((10, 10), bool), np.zeros((10, 10), bool)
    a[0:3, 0:3] = True
    b[7:10, 7:10] = True
    assert dice(a, b) == pytest.approx(0.0)
    assert iou(a, b) == pytest.approx(0.0)


def test_two_empty_masks_agree():
    empty = np.zeros((4, 4), bool)
    assert dice(empty, empty) == 1.0
    assert iou(empty, empty) == 1.0


def test_soft_target_is_rejected():
    """The v1 failure: a magenta mask read as greyscale/255 gives a target of 0.354.

    Dice against it silently caps at 0.523, and 0.47 then looks like a bad model
    rather than a near-optimal one.
    """
    pred = np.ones((4, 4), bool)
    soft = np.full((4, 4), 0.354)
    with pytest.raises(ValueError, match="not binary"):
        dice(pred, soft)


def test_area_ratio_detects_under_segmentation_that_dice_tolerates():
    """A lesion eroded by 3 px on every side keeps a Dice above 0.92 and loses 14 % of its area.

    This is the shape of the v1 model-selection failure: the shipped checkpoint
    was chosen on Dice while under-recovering necrotic area by 19 %.
    """
    true = np.zeros((100, 100), bool)
    true[10:90, 10:90] = True
    pred = np.zeros((100, 100), bool)
    pred[13:87, 13:87] = True

    assert dice(pred, true) > 0.92
    assert area_ratio(pred, true) == pytest.approx(0.856, abs=0.01)


def test_segmentation_report_excludes_empty_truths_and_counts_them():
    lesion = np.zeros((10, 10), bool)
    lesion[2:8, 2:8] = True
    empty = np.zeros((10, 10), bool)
    spurious = np.zeros((10, 10), bool)
    spurious[0, 0] = True

    report = segmentation_report([lesion, spurious, empty], [lesion, empty, empty])
    assert report.n == 1
    assert report.dice == pytest.approx(1.0)
    assert report.empty_truth_false_positives == 1


def test_segmentation_report_reports_area_bias_as_percent():
    true = np.zeros((100, 100), bool)
    true[0:100, 0:50] = True
    pred = np.zeros((100, 100), bool)
    pred[0:100, 0:40] = True

    report = segmentation_report([pred], [true])
    assert report.area_ratio == pytest.approx(0.8, abs=1e-6)
    assert report.area_bias_pct == pytest.approx(-20.0, abs=1e-4)


def test_mismatched_lengths_raise():
    m = np.zeros((4, 4), bool)
    with pytest.raises(ValueError, match="predictions for"):
        segmentation_report([m, m], [m])


def test_match_points_is_one_to_one():
    true = np.array([[0.0, 0.0], [10.0, 0.0]])
    pred = np.array([[0.5, 0.0], [1.0, 0.0], [10.2, 0.0]])
    tp, fp, fn = match_points(pred, true, radius_px=2.0)
    assert (tp, fp, fn) == (2, 1, 0)


def test_match_points_respects_the_radius():
    true = np.array([[0.0, 0.0]])
    pred = np.array([[5.0, 0.0]])
    assert match_points(pred, true, radius_px=2.0) == (0, 1, 1)


def test_match_points_handles_empty_sides():
    pts = np.array([[1.0, 1.0]])
    assert match_points(np.empty((0, 2)), pts, 3.0) == (0, 0, 1)
    assert match_points(pts, np.empty((0, 2)), 3.0) == (0, 1, 0)


def test_counting_report_separates_slope_from_intercept():
    true = [np.zeros((n, 2)) for n in (100, 200, 300, 400)]
    proportional = [np.zeros((int(0.9 * n), 2)) for n in (100, 200, 300, 400)]
    shifted = [np.zeros((n - 10, 2)) for n in (100, 200, 300, 400)]

    prop = counting_report(proportional, true)
    shift = counting_report(shifted, true)

    assert prop.slope == pytest.approx(0.9, abs=1e-6)
    assert shift.slope == pytest.approx(1.0, abs=1e-6)
    assert shift.bias == pytest.approx(-10.0)


def test_counting_report_localisation_is_optional():
    true = [np.array([[0.0, 0.0], [5.0, 5.0]]), np.array([[1.0, 1.0]])]
    pred = [np.array([[0.1, 0.0], [5.0, 5.2]]), np.array([[9.0, 9.0]])]

    without = counting_report(pred, true)
    assert without.f1 is None

    with_loc = counting_report(pred, true, radius_px=1.0)
    assert with_loc.precision == pytest.approx(2 / 3)
    assert with_loc.recall == pytest.approx(2 / 3)
