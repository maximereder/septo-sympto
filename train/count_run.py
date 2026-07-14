"""Run a pycnidia counter training locally: pool, split by scan, train, report.

    poetry run python -m train.count_run \
        --dataset-dir data/pycnidia/train-200-aug-x3 data/pycnidia/valid-40 \
        --arch heatmap --epochs 100 --device mps

Same function modal_app calls remotely. Local-runnable so an architecture is
debugged on CPU or MPS before a GPU-hour.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import torch

from septosympto.models import available_counters, build_counter
from train.config import CountConfig
from train.count_data import load_points_pool, split_pool
from train.count_loop import train_counter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.count_run", description="Train a pycnidia counter."
    )
    parser.add_argument(
        "--dataset-dir", nargs="+", required=True, help="Pycnidia image/label dirs."
    )
    parser.add_argument(
        "--arch", default="heatmap",
        help=f"Counting architecture. One of: {', '.join(available_counters())}.",
    )
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default="pycnidia")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--imgsz", type=int, nargs=2, default=[304, 3072], metavar=("H", "W"))
    parser.add_argument("--match-radius-px", type=float, default=8.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def config_from_args(args: argparse.Namespace) -> CountConfig:
    return CountConfig(
        dataset_dirs=tuple(args.dataset_dir),
        arch=args.arch,
        output_dir=args.output_dir,
        run_name=args.run_name,
        imgsz=(args.imgsz[0], args.imgsz[1]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        match_radius_px=args.match_radius_px,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )


def _check_input_size(model: torch.nn.Module, imgsz: tuple[int, int]) -> None:
    divisor = getattr(type(model), "INPUT_DIVISOR", 1)
    h, w = imgsz
    if h % divisor or w % divisor:
        near_h = round(h / divisor) * divisor
        near_w = round(w / divisor) * divisor
        raise ValueError(
            f"imgsz {imgsz} must be divisible by {divisor} for {type(model).__name__}; "
            f"try {near_h} x {near_w}"
        )


def run(config: CountConfig, *, timestamp: str, progress: bool = True) -> dict:
    model = build_counter(config.arch)
    _check_input_size(model, config.imgsz)

    pool = load_points_pool(list(config.dataset_dirs))
    train_samples, val_samples, test_samples = split_pool(
        pool, config.val_fraction, config.test_fraction, config.seed
    )
    total_points = sum(len(s.points_norm) for s in pool)
    print(
        f"arch {config.arch} | {len(pool)} leaves, {len({s.scan for s in pool})} scans, "
        f"{total_points} points -> train {len(train_samples)} / val {len(val_samples)} / "
        f"test {len(test_samples)}"
    )
    summary = train_counter(
        model, config, train_samples, val_samples, timestamp=timestamp, progress=progress
    )
    best = summary["best"]
    print(
        f"\nbest epoch {summary['best_epoch']}: MAE {best['val_mae']:.1f}  "
        f"bias {best['val_bias']:+.1f}  slope {best['val_slope']:.3f}  F1 {best['val_f1']:.3f}"
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
