"""Modal launcher. The training logic lives in train.run; this only sends it to a GPU.

    modal run train/modal_app.py --dataset data/necrosis/dataset/300.zip \
        --epochs 100 --run-name necrosis-v2 --gpu A10

The dataset zip is uploaded once into a Volume, not baked into the image, so a
new run reuses it. Checkpoints land in a second Volume that outlives the
container. Nothing here computes anything: if this file grew a training detail,
that detail would become impossible to run or test locally, which is the trap we
are avoiding.
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.13,<3.0",
        "numpy>=2.5,<3.0",
        "opencv-python-headless>=4.13,<5.0",
        "safetensors>=0.8,<1.0",
        "tqdm>=4.67,<5.0",
    )
    .add_local_python_source("septosympto", "train")
)

app = modal.App("septosympto-train", image=image)

data_volume = modal.Volume.from_name("septosympto-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("septosympto-runs", create_if_missing=True)

DATA_DIR = "/data"
RUNS_DIR = "/runs"


@app.function(
    gpu="A10",
    timeout=6 * 60 * 60,
    volumes={DATA_DIR: data_volume, RUNS_DIR: runs_volume},
)
def train_remote(config_dict: dict, timestamp: str) -> dict:
    from train.config import TrainConfig
    from train.run import run

    config = TrainConfig(**config_dict)
    summary = run(config, timestamp=timestamp, progress=True)
    runs_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    dataset: str,
    run_name: str = "necrosis",
    epochs: int = 100,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    height: int = 304,
    width: int = 3072,
    gpu: str = "A10",
) -> None:
    from datetime import UTC, datetime
    from pathlib import Path

    from train.config import TrainConfig

    remote_dataset = f"{DATA_DIR}/{Path(dataset).name}"
    with data_volume.batch_upload(force=True) as upload:
        upload.put_file(dataset, Path(remote_dataset).name)

    config = TrainConfig(
        dataset=remote_dataset,
        output_dir=RUNS_DIR,
        run_name=run_name,
        imgsz=(height, width),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device="cuda",
    )
    trainer = train_remote if gpu == "A10" else train_remote.with_options(gpu=gpu)
    summary = trainer.remote(config.as_dict(), datetime.now(UTC).isoformat())
    best = summary["best"]
    print(
        f"\nbest epoch {summary['best_epoch']}: Dice {best['val_dice']:.4f}  "
        f"area {best['val_area_ratio']:.3f} ({best['val_area_bias_pct']:+.1f} %)"
    )
    print(f"checkpoint in the septosympto-runs volume at {run_name}/best.safetensors")
