"""Training configuration as a plain dataclass.

One object carries everything a run needs, so a run is reproducible from it alone
and it can be serialised into the checkpoint manifest. No values are read from
the environment or the clock inside the training code; they come from here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TrainConfig:
    dataset: str
    arch: str = "unet"
    output_dir: str = "runs"
    run_name: str = "necrosis"

    imgsz: tuple[int, int] = (304, 3072)

    epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 0

    bce_weight: float = 0.5
    dice_weight: float = 0.5

    hflip: bool = True
    vflip: bool = True

    threshold: float = 0.5
    early_stopping_patience: int = 15
    num_workers: int = 4
    device: str = "cpu"

    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CountConfig:
    dataset_dirs: tuple[str, ...]
    arch: str = "heatmap"
    output_dir: str = "runs"
    run_name: str = "pycnidia"

    imgsz: tuple[int, int] = (304, 3072)

    epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 0

    hflip: bool = True
    vflip: bool = True

    decode_threshold: float = 0.3
    match_radius_px: float = 8.0
    early_stopping_patience: int = 20
    num_workers: int = 4
    device: str = "cpu"

    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)
