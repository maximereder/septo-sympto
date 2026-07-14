"""Modal launchers for training on a GPU. Logic lives in train.run / train.count_run.

Following the volume convention from the linear-transcription project: data is
pushed into a Modal volume once, out of band, with ``scripts/push_data.sh``. The
worker reads it from the mounted volume and fails fast if it is missing, so a run
never uploads on its hot path.

    scripts/push_data.sh                       # once: local data -> volume
    modal run train/modal_app.py::pycnidia --arch p2p --run-name pyc-p2p --gpu A100-40GB
    modal run train/modal_app.py::necrosis --run-name nec-v2 --gpu A10

Checkpoints land in a runs volume that outlives the container. Pretrained
backbones (P2PNet's VGG) download into a cache volume under TORCH_HOME the first
time and persist, so later runs do not re-download.
"""

from __future__ import annotations

import modal

APP_NAME = "septosympto-train"
DATA_DIR = "/data"
RUNS_DIR = "/runs"
CACHE_DIR = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.13,<3.0",
        "torchvision>=0.28,<1.0",
        "numpy>=2.5,<3.0",
        "opencv-python-headless>=4.13,<5.0",
        "safetensors>=0.8,<1.0",
        "scipy>=1.14,<2.0",
        "tqdm>=4.67,<5.0",
    )
    .env({"TORCH_HOME": f"{CACHE_DIR}/torch"})
    .add_local_python_source("septosympto", "train")
)

app = modal.App(APP_NAME, image=image)

data_volume = modal.Volume.from_name("septosympto-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("septosympto-runs", create_if_missing=True)
cache_volume = modal.Volume.from_name("septosympto-cache", create_if_missing=True)

VOLUMES = {DATA_DIR: data_volume, RUNS_DIR: runs_volume, CACHE_DIR: cache_volume}


def _require(path: str) -> None:
    import os

    if not os.path.exists(path):
        raise RuntimeError(
            f"{path} is not in the septosympto-data volume. Push it first: scripts/push_data.sh"
        )


def _commit_runs(_epoch: int) -> None:
    """Persist checkpoints to the volume as they are written, so a crash keeps them."""
    runs_volume.commit()


@app.function(gpu="A10", timeout=8 * 60 * 60, scaledown_window=1, volumes=VOLUMES)
def train_necrosis_remote(config_dict: dict, timestamp: str) -> dict:
    from train.config import TrainConfig
    from train.run import run

    config = TrainConfig(**config_dict)
    _require(config.dataset)
    summary = run(config, timestamp=timestamp, progress=True, on_checkpoint=_commit_runs)
    runs_volume.commit()
    return summary


@app.function(gpu="A10", timeout=8 * 60 * 60, scaledown_window=1, volumes=VOLUMES)
def train_pycnidia_remote(config_dict: dict, timestamp: str) -> dict:
    from train.config import CountConfig
    from train.count_run import run

    config = CountConfig(**config_dict)
    for directory in config.dataset_dirs:
        _require(directory)
    summary = run(config, timestamp=timestamp, progress=True, on_checkpoint=_commit_runs)
    runs_volume.commit()
    cache_volume.commit()
    return summary


@app.local_entrypoint()
def necrosis(
    dataset: str = "necrosis/dataset/300.zip",
    run_name: str = "necrosis",
    epochs: int = 100,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    height: int = 304,
    width: int = 3072,
    checkpoint_every: int = 25,
    gpu: str = "A10",
) -> None:
    from datetime import UTC, datetime

    from train.config import TrainConfig

    config = TrainConfig(
        dataset=f"{DATA_DIR}/{dataset}",
        output_dir=RUNS_DIR,
        run_name=run_name,
        imgsz=(height, width),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_every=checkpoint_every,
        device="cuda",
    )
    fn = train_necrosis_remote if gpu == "A10" else train_necrosis_remote.with_options(gpu=gpu)
    summary = fn.remote(config.as_dict(), datetime.now(UTC).isoformat())
    best = summary["best"]
    print(
        f"\nbest epoch {summary['best_epoch']}: Dice {best['val_dice']:.4f} "
        f"area {best['val_area_ratio']:.3f} ({best['val_area_bias_pct']:+.1f} %)"
    )
    print(f"checkpoint in septosympto-runs at {run_name}/best.safetensors")


@app.local_entrypoint()
def pycnidia(
    dataset_dirs: str = "pycnidia/train-200-aug-x3,pycnidia/valid-40",
    arch: str = "p2p",
    run_name: str = "pycnidia",
    epochs: int = 200,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    height: int = 200,
    width: int = 2048,
    match_radius_px: float = 8.0,
    checkpoint_every: int = 25,
    gpu: str = "A100-40GB",
) -> None:
    from datetime import UTC, datetime

    from train.config import CountConfig

    dirs = tuple(f"{DATA_DIR}/{d.strip()}" for d in dataset_dirs.split(","))
    config = CountConfig(
        dataset_dirs=dirs,
        arch=arch,
        output_dir=RUNS_DIR,
        run_name=run_name,
        imgsz=(height, width),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        match_radius_px=match_radius_px,
        checkpoint_every=checkpoint_every,
        device="cuda",
    )
    fn = train_pycnidia_remote if gpu == "A10" else train_pycnidia_remote.with_options(gpu=gpu)
    summary = fn.remote(config.as_dict(), datetime.now(UTC).isoformat())
    best = summary["best"]
    print(
        f"\nbest epoch {summary['best_epoch']}: MAE {best['val_mae']:.1f} "
        f"bias {best['val_bias']:+.1f} slope {best['val_slope']:.3f} F1 {best['val_f1']:.3f}"
    )
    print(f"checkpoint in septosympto-runs at {run_name}/best.safetensors")
