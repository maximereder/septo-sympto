"""The necrosis path of tools/regen_pycnidia_native.py, on synthetic strips."""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("scipy")

_spec = importlib.util.spec_from_file_location(
    "regen_native", Path(__file__).resolve().parents[1] / "tools" / "regen_pycnidia_native.py"
)
regen = importlib.util.module_from_spec(_spec)
sys.modules["regen_native"] = regen
_spec.loader.exec_module(regen)

from septosympto.letterbox import CANVAS_H, CANVAS_W, plan  # noqa: E402


def test_read_polygons_parses_yolo_seg_lines(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.1 0.2 0.5 0.2 0.5 0.8\n1 0 0 1 0 1 1 0 1\n\n")
    polys = regen.read_polygons(label)
    assert [c for c, _ in polys] == [0, 1]
    assert polys[0][1].shape == (3, 2)
    assert polys[0][1][2].tolist() == [0.5, 0.8]


def test_necrosis_class_ids_reads_inline_and_block_yaml(tmp_path):
    (tmp_path / "data.yaml").write_text("nc: 2\nnames: ['object', 'necrosis']\n")
    assert regen.necrosis_class_ids(tmp_path) == {1}
    (tmp_path / "data.yaml").write_text("names:\n  0: Necrosis\n  1: object\n")
    assert regen.necrosis_class_ids(tmp_path) == {0}
    (tmp_path / "data.yaml").unlink()
    assert regen.necrosis_class_ids(tmp_path) == {0}


def _export(root: Path, files: dict[str, str]) -> None:
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text("nc: 2\nnames: ['necrosis', 'object']\n")
    for stem, text in files.items():
        image = np.full((30, 300, 3), 200, np.uint8)
        cv2.imwrite(str(root / "train" / "images" / f"{stem}.jpg"), image)
        (root / "train" / "labels" / f"{stem}.txt").write_text(text)


def test_export_loader_drops_orphans_flags_unannotated_and_dedups(tmp_path, capsys):
    _export(tmp_path, {
        "S1__1__1_jpg.rf.aaa": "0 0.1 0.1 0.5 0.1 0.5 0.9 0.1 0.9\n1 0 0 0.1 0 0.1 0.01 0 0.01\n",
        "S1__2__1_jpg.rf.bbb": "1 0 0 0.1 0 0.1 0.01 0 0.01\n",       # only a slip
        "S2__1__1_jpg.rf.ccc": "",                                     # necrosis-free leaf
        "S3__1__1_jpg.rf.ddd": "0 0 0 0.2 0 0.2 1 0 1\n",             # duplicate, smaller
        "S3__1__1_jpg.rf.eee": "0 0 0 0.9 0 0.9 1 0 1\n",             # duplicate, larger
    })
    samples, flagged = regen.load_necrosis(str(tmp_path))
    assert set(samples) == {"S1__1__1", "S2__1__1", "S3__1__1"}
    assert flagged == {"S1__2__1": "unannotated"}
    assert len(samples["S1__1__1"][1].polygons) == 1
    assert samples["S2__1__1"][1].strip_fraction() == 0.0
    assert samples["S3__1__1"][1].strip_fraction() == pytest.approx(0.9, abs=0.02)
    out = capsys.readouterr().out
    assert "dropped 2 polygon(s)" in out
    assert "duplicate S3__1__1" in out


def test_polygon_and_mask_annotations_land_on_the_same_canvas_pixels():
    """Remapping vertices must agree with warping the rasterised strip mask."""
    w, h = 2400, 220                       # native leaf size
    p = plan(w, h)
    solid = np.ones((h, w), bool)
    solid[:, :50] = False                  # a bit of background to clip against
    poly = np.array([[0.02, 0.2], [0.4, 0.1], [0.6, 0.9], [0.1, 0.8]])
    strip_shape = (300, 3070)
    polygons = regen.PolygonAnnotation([poly], strip_shape)
    strip_mask = regen.rasterise([poly], strip_shape, lambda uv: uv * [3070, 300])
    pixels = regen.MaskAnnotation(strip_mask)

    for fx, fy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        a = polygons.to_canvas(fx, fy, solid, p) > 0
        b = pixels.to_canvas(fx, fy, solid, p) > 0
        assert a.shape == b.shape == (CANVAS_H, CANVAS_W)
        assert a.sum() > 0
        iou = (a & b).sum() / (a | b).sum()
        assert iou > 0.97, (fx, fy, iou)
        assert not a[p.oy : p.oy + p.nh, p.ox : p.ox + int(45 * p.scale)].any()


def test_rasterise_ignores_degenerate_polygons():
    segment = [np.array([[0.1, 0.1], [0.2, 0.2]])]
    assert not regen.rasterise(segment, (10, 10), lambda uv: uv * 10).any()
