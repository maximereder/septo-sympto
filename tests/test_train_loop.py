import cv2
import numpy as np
import torch
import torch.nn as nn

from train.config import TrainConfig
from train.data import Sample
from train.loop import train_segmenter
from train.loss import bce_dice_loss, soft_dice_loss


class TinySegmenter(nn.Module):
    """A 1x1 conv: enough to overfit a trivial pattern, fast enough for a test."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def sample_where_necrosis_is_the_blue_channel() -> Sample:
    h, w = 16, 64
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = (255, 0, 0)
    positive = np.zeros((h, w), bool)
    positive[:, : w // 2] = True
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[positive] = (124, 0, 255)
    return Sample(
        image="s__1.jpg", scan="s",
        image_bytes=cv2.imencode(".jpg", img)[1].tobytes(),
        mask_bytes=cv2.imencode(".png", rgb)[1].tobytes(),
    )


def test_dice_loss_is_low_for_a_correct_prediction():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4] = 1.0
    confident = (target * 2 - 1) * 20
    assert soft_dice_loss(confident, target).item() < 0.05


def test_bce_dice_is_higher_when_wrong():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4] = 1.0
    right = (target * 2 - 1) * 20
    wrong = -right
    assert bce_dice_loss(wrong, target, 0.5, 0.5) > bce_dice_loss(right, target, 0.5, 0.5)


def test_training_improves_val_dice_and_writes_outputs(tmp_path):
    samples = [sample_where_necrosis_is_the_blue_channel() for _ in range(4)]
    config = TrainConfig(
        dataset="synthetic",
        output_dir=str(tmp_path),
        run_name="tiny",
        imgsz=(16, 64),
        epochs=30,
        batch_size=2,
        learning_rate=0.1,
        num_workers=0,
        early_stopping_patience=30,
    )
    summary = train_segmenter(
        TinySegmenter(), config, samples, samples,
        timestamp="2026-07-10T00:00:00+00:00", progress=False,
    )

    assert summary["best"]["val_dice"] > 0.9
    assert (tmp_path / "tiny" / "best.safetensors").exists()
    assert (tmp_path / "tiny" / "manifest.json").exists()
    assert summary["history"][-1]["val_dice"] >= summary["history"][0]["val_dice"]


def test_periodic_checkpoint_saves_last_and_fires_the_callback(tmp_path):
    samples = [sample_where_necrosis_is_the_blue_channel() for _ in range(2)]
    config = TrainConfig(
        dataset="synthetic", output_dir=str(tmp_path), run_name="ck",
        imgsz=(16, 64), epochs=6, batch_size=1, num_workers=0,
        early_stopping_patience=99, checkpoint_every=2,
    )
    fired = []
    train_segmenter(
        TinySegmenter(), config, samples, samples,
        timestamp="2026-07-10T00:00:00+00:00", progress=False,
        on_checkpoint=fired.append,
    )
    assert (tmp_path / "ck" / "last.safetensors").exists()
    assert {1, 3, 5}.issubset(set(fired))


def test_no_periodic_checkpoint_when_disabled(tmp_path):
    samples = [sample_where_necrosis_is_the_blue_channel() for _ in range(2)]
    config = TrainConfig(
        dataset="synthetic", output_dir=str(tmp_path), run_name="nock",
        imgsz=(16, 64), epochs=3, batch_size=1, num_workers=0, checkpoint_every=0,
    )
    train_segmenter(
        TinySegmenter(), config, samples, samples,
        timestamp="2026-07-10T00:00:00+00:00", progress=False,
    )
    assert not (tmp_path / "nock" / "last.safetensors").exists()
    assert (tmp_path / "nock" / "best.safetensors").exists()


def test_manifest_records_config_and_split_sizes(tmp_path):
    samples = [sample_where_necrosis_is_the_blue_channel() for _ in range(3)]
    config = TrainConfig(
        dataset="synthetic", output_dir=str(tmp_path), run_name="m",
        imgsz=(16, 64), epochs=2, batch_size=1, num_workers=0,
    )
    summary = train_segmenter(
        TinySegmenter(), config, samples, samples[:1],
        timestamp="2026-07-10T00:00:00+00:00", progress=False,
    )
    assert summary["n_train"] == 3
    assert summary["n_val"] == 1
    assert summary["timestamp"] == "2026-07-10T00:00:00+00:00"

    import json

    manifest = json.loads((tmp_path / "m" / "manifest.json").read_text())
    assert manifest["config"]["imgsz"] == [16, 64]
    assert manifest["n_train"] == 3
