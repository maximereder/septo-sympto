"""Run a training locally: pool the dataset, split by scan, train, report.

    poetry run python -m train.run --dataset data/necrosis/dataset/300.zip \
        --epochs 100 --device mps --run-name necrosis-v2

This is the same function ``modal_app`` calls remotely. Keeping it runnable
locally means an architecture can be debugged on CPU or MPS before spending a
GPU-hour, and the training logic never lives inside a Modal decorator.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import torch

from septosympto.models import available_segmenters, build_segmenter
from train.config import TrainConfig
from train.data import load_pool, scan_grouped_split
from train.loop import train_segmenter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train.run", description="Train a necrosis segmenter.")
    parser.add_argument("--dataset", required=True, help="Roboflow necrosis zip.")
    parser.add_argument(
        "--arch", default="unet",
        help=f"Segmentation architecture to train. One of: {', '.join(available_segmenters())}.",
    )
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default="necrosis")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--imgsz", type=int, nargs=2, default=[304, 3072], metavar=("H", "W"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        dataset=args.dataset,
        arch=args.arch,
        output_dir=args.output_dir,
        run_name=args.run_name,
        imgsz=(args.imgsz[0], args.imgsz[1]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )


def _check_input_size(model: torch.nn.Module, imgsz: tuple[int, int]) -> None:
    """Validate imgsz against a constraint the architecture declares, if any."""
    divisor = getattr(type(model), "INPUT_DIVISOR", 1)
    h, w = imgsz
    if h % divisor or w % divisor:
        near_h = round(h / divisor) * divisor
        near_w = round(w / divisor) * divisor
        raise ValueError(
            f"imgsz {imgsz} must be divisible by {divisor} for {type(model).__name__}; "
            f"try {near_h} x {near_w}"
        )


def run(config: TrainConfig, *, timestamp: str, progress: bool = True) -> dict:
    model = build_segmenter(config.arch)
    _check_input_size(model, config.imgsz)

    pool = load_pool(config.dataset)
    train_samples, val_samples, test_samples = scan_grouped_split(
        pool, config.val_fraction, config.test_fraction, config.seed
    )
    print(
        f"arch {config.arch} | {len(pool)} images, {len({s.scan for s in pool})} scans -> "
        f"train {len(train_samples)} / val {len(val_samples)} / test {len(test_samples)}"
    )
    summary = train_segmenter(
        model, config, train_samples, val_samples, timestamp=timestamp, progress=progress
    )
    best = summary["best"]
    print(
        f"\nbest epoch {summary['best_epoch']}: "
        f"Dice {best['val_dice']:.4f}  IoU {best['val_iou']:.4f}  "
        f"area {best['val_area_ratio']:.3f} ({best['val_area_bias_pct']:+.1f} %)"
    )
    print(f"weights -> {summary['weights']}")
    print(f"test set held out: {len(test_samples)} leaves (never seen during training)")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(config_from_args(args), timestamp=datetime.now(UTC).isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
