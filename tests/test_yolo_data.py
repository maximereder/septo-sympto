import cv2
import numpy as np

from train.config import YoloConfig
from train.data import Sample
from train.yolo_data import export_yolo_dataset, tile_spans
from train.yolo_run import _best_epoch, resolve_model, yolo_train_kwargs


def png_mask_255(positive: np.ndarray) -> bytes:
    return cv2.imencode(".png", positive.astype(np.uint8) * 255)[1].tobytes()


def png_image(h=32, w=256) -> bytes:
    return cv2.imencode(".png", np.full((h, w, 3), 128, np.uint8))[1].tobytes()


def sample(name: str) -> Sample:
    positive = np.zeros((32, 256), bool)
    positive[8:24, 64:192] = True
    return Sample(image=name, scan=name.split("__")[0], image_bytes=png_image(),
                  mask_bytes=png_mask_255(positive))


def test_export_writes_class_id_mask_tiles_mirroring_image_tiles(tmp_path):
    root = tmp_path / "ds"
    yaml = export_yolo_dataset(root, [sample("a__1.png"), sample("a__2.png")],
                               [sample("b__1.png")], [sample("c__1.png")], tile=32)
    assert yaml == root / "data.yaml"
    names = sorted(p.name for p in (root / "images/train").iterdir())
    assert len(names) == 2 * 8 and names[0] == "a__1__t0.png" and names[-1] == "a__2__t7.png"
    assert (root / "masks/val/b__1__t0.png").exists()
    assert (root / "images/test/c__1__t7.png").exists()

    tile = cv2.imread(str(root / "images/train/a__1__t3.png"))
    assert tile.shape == (32, 32, 3)
    mask = cv2.imread(str(root / "masks/train/a__1__t3.png"), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (32, 32)
    assert set(np.unique(mask)) == {0, 1}, "255 is the ignore label in a semantic mask"
    assert mask.sum() == 16 * 32                      # x in [96, 128) is inside [64, 192)
    edge = cv2.imread(str(root / "masks/train/a__1__t0.png"), cv2.IMREAD_UNCHANGED)
    assert edge.sum() == 0

    text = yaml.read_text()
    assert "nc: 1" in text
    assert "masks_dir: masks" in text
    assert "train: images/train" in text
    assert "val: images/val" in text
    assert "test: images/test" in text


def test_tile_spans_cover_the_width_and_pull_the_last_tile_back():
    assert tile_spans(3072, 384) == [(i * 384, (i + 1) * 384) for i in range(8)]
    assert tile_spans(100, 40) == [(0, 40), (40, 80), (60, 100)]
    assert tile_spans(100, 0) == [(0, 100)]
    assert tile_spans(100, 200) == [(0, 100)]


def test_export_starts_from_a_clean_root(tmp_path):
    root = tmp_path / "ds"
    export_yolo_dataset(root, [sample("a__1.png")], [sample("b__1.png")], tile=0)
    export_yolo_dataset(root, [sample("z__1.png")], [sample("b__1.png")], tile=0)
    assert [p.name for p in (root / "images/train").iterdir()] == ["z__1__t0.png"]
    assert "test:" not in (root / "data.yaml").read_text()


def test_train_kwargs_train_on_square_tiles_and_honour_the_lr(tmp_path):
    config = YoloConfig(dataset="x", imgsz=(384, 3072), tile=384, learning_rate=3e-4,
                        hflip=True, vflip=False, extra={"hsv_h": 0.0})
    kwargs = yolo_train_kwargs(config, tmp_path / "data.yaml")
    assert kwargs["imgsz"] == 384, "imgsz is the tile, so Ultralytics scales train and val by 1"
    assert kwargs["rect"] is False
    assert kwargs["mosaic"] == 0.0
    assert kwargs["lr0"] == 3e-4
    assert kwargs["optimizer"] == "AdamW"
    assert (kwargs["fliplr"], kwargs["flipud"]) == (0.5, 0.0)
    assert kwargs["hsv_h"] == 0.0
    assert kwargs["project"] == "runs" and kwargs["name"] == "necrosis-yolo"


def test_best_epoch_is_the_miou_maximiser():
    history = [
        {"epoch": 1, "metrics/mIoU": 0.3},
        {"epoch": 2, "metrics/mIoU": 0.7},
        {"epoch": 3, "metrics/mIoU": 0.5},
    ]
    assert _best_epoch(history) == 2
    assert _best_epoch([]) == -1


def test_bare_asset_names_are_anchored_in_pretrained_dir(tmp_path):
    config = YoloConfig(dataset="x", model="yolo26n-sem.pt", pretrained_dir=str(tmp_path / "w"))
    assert resolve_model(config) == str(tmp_path / "w" / "yolo26n-sem.pt")
    assert (tmp_path / "w").is_dir()
    assert resolve_model(YoloConfig(dataset="x", model="yolo26n-sem.yaml")) == "yolo26n-sem.yaml"
    assert resolve_model(YoloConfig(dataset="x", model="runs/r/weights/best.pt")) == (
        "runs/r/weights/best.pt"
    )
