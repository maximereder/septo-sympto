"""Train a YOLO26 semantic segmenter on the necrosis pool, locally or on Modal.

    poetry run python -m train.yolo_run --dataset data/leaves-native \
        --model yolo26n-sem.pt --epochs 100 --device mps --run-name nec-yolo26n

Ultralytics owns the training loop, its progress bars and its checkpoint format;
this module owns everything around it so that a YOLO run is a peer of a PyTorch
run. The pool, the scan-grouped split and the seed are the project's, so the
folds are identical to ``train.run`` at the same settings. When Ultralytics is
done, the checkpoint it selected is re-evaluated on the same validation leaves
through :mod:`septosympto.eval` — real binary Dice and area bias, the numbers a
U-Net run reports — and a ``manifest.json`` of the same shape is written beside
Ultralytics' own ``results.csv``.

Ultralytics logs to the terminal as it goes (the per-epoch table); on Modal the
same output streams back through ``modal run``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from septosympto.eval import segmentation_report
from train.config import YoloConfig
from train.data import Sample, decode_image, decode_mask, load_pool, scan_grouped_split
from train.loop import _git_commit, _json_default
from train.yolo_data import export_yolo_dataset

DEFAULT_MODEL = "yolo26n-sem.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.yolo_run", description="Train a YOLO26 semantic necrosis segmenter."
    )
    parser.add_argument("--dataset", default="data/leaves-native",
                        help="Native letterbox directory (img/ + mask/) or a Roboflow zip.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Ultralytics weights or config: yolo26{n,s,m,l,x}-sem.pt (pretrained) or "
             "yolo26n-sem.yaml (from scratch).",
    )
    parser.add_argument("--pretrained-dir", default="data/pretrained",
                        help="Where pretrained Ultralytics weights are downloaded and kept.")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default="necrosis-yolo")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[384, 3072], metavar=("H", "W"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--extra", type=json.loads, default={},
        help='JSON of extra YOLO.train kwargs, e.g. \'{"hsv_h": 0.0, "scale": 0.0}\'.',
    )
    return parser


def config_from_args(args: argparse.Namespace) -> YoloConfig:
    return YoloConfig(
        dataset=args.dataset,
        model=args.model,
        pretrained_dir=args.pretrained_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        imgsz=(args.imgsz[0], args.imgsz[1]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        threshold=args.threshold,
        early_stopping_patience=args.patience,
        device=args.device,
        num_workers=args.num_workers,
        extra=args.extra,
    )


def yolo_train_kwargs(config: YoloConfig, data_yaml: Path) -> dict:
    """The ``YOLO.train`` call a config maps to. Pure, so it is testable without a GPU.

    ``imgsz`` is the long side and ``rect=True`` keeps the leaves at their own
    aspect ratio, so a 384x3072 canvas trains as 384x3072 rather than padded to a
    3072x3072 square. Every leaf shares that shape, so Ultralytics keeps shuffling
    (it only disables it when rectangular batches differ). Geometry augmentations
    other than flips are off: a leaf is already placed 1:1 on the canvas and
    inference never scales, translates or mosaics it. ``optimizer`` is set
    explicitly because Ultralytics' ``auto`` silently overrides ``lr0``.
    """
    h, w = config.imgsz
    kwargs = {
        "data": str(data_yaml),
        "imgsz": max(h, w),
        "rect": True,
        "epochs": config.epochs,
        "batch": config.batch_size,
        "optimizer": config.optimizer,
        "lr0": config.learning_rate,
        "weight_decay": config.weight_decay,
        "patience": config.early_stopping_patience,
        "device": config.device,
        "workers": config.num_workers,
        "project": config.output_dir,
        "name": config.run_name,
        "exist_ok": True,
        "seed": config.seed,
        "deterministic": True,
        "fliplr": 0.5 if config.hflip else 0.0,
        "flipud": 0.5 if config.vflip else 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "scale": 0.0,
        "translate": 0.0,
        "degrees": 0.0,
        "shear": 0.0,
        "erasing": 0.0,
        "plots": True,
    }
    kwargs.update(config.extra)
    return kwargs


def resolve_model(config: YoloConfig) -> str:
    """Where Ultralytics should look for ``config.model``.

    A bare asset name (``yolo26n-sem.pt``) is anchored in ``pretrained_dir`` so the
    download lands there once and is reused, rather than in whatever the current
    directory happens to be. Paths and ``.yaml`` configs pass through untouched.
    """
    model = config.model
    if "/" in model or model.endswith(".yaml") or Path(model).exists():
        return model
    Path(config.pretrained_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(config.pretrained_dir) / model)


def _evaluate(weights: Path, samples: list[Sample], threshold: float, device: str) -> dict:
    from septosympto.adapters import YoloSegmenter

    segmenter = YoloSegmenter.from_weights(weights, threshold=threshold, device=device)
    preds, truths = [], []
    for sample in samples:
        preds.append(segmenter.segment(decode_image(sample.image_bytes)))
        truths.append(decode_mask(sample.mask_bytes))
    report = segmentation_report(preds, truths)
    return {
        "val_dice": report.dice,
        "val_iou": report.iou,
        "val_area_ratio": report.area_ratio,
        "val_area_bias_pct": report.area_bias_pct,
    }


def _read_history(results_csv: Path) -> list[dict]:
    """Ultralytics' per-epoch table, with numeric columns parsed."""
    if not results_csv.exists():
        return []
    rows = []
    with results_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                key = key.strip()
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _best_epoch(history: list[dict]) -> int:
    """The epoch Ultralytics kept as ``best.pt``: the one maximising its fitness, mIoU."""
    if not history:
        return -1
    scores = [row.get("metrics/mIoU", float("-inf")) for row in history]
    index = max(range(len(scores)), key=scores.__getitem__)
    return int(history[index].get("epoch", index))


def run(
    config: YoloConfig,
    *,
    timestamp: str,
    on_checkpoint: Callable[[int], None] | None = None,
) -> dict:
    from ultralytics import YOLO

    pool = load_pool(config.dataset)
    train_samples, val_samples, test_samples = scan_grouped_split(
        pool, config.val_fraction, config.test_fraction, config.seed
    )
    print(
        f"model {config.model} | {len(pool)} images, {len({s.scan for s in pool})} scans -> "
        f"train {len(train_samples)} / val {len(val_samples)} / test {len(test_samples)}"
    )

    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = export_yolo_dataset(out_dir / "dataset", train_samples, val_samples, test_samples)

    model = YOLO(resolve_model(config))
    if on_checkpoint is not None:
        model.add_callback("on_model_save", lambda trainer: on_checkpoint(trainer.epoch))
    model.train(**yolo_train_kwargs(config, data_yaml))

    weights = out_dir / "weights" / "best.pt"
    if not weights.exists():
        raise RuntimeError(f"Ultralytics finished without writing {weights}")

    history = _read_history(out_dir / "results.csv")
    best = _evaluate(weights, val_samples, config.threshold, config.device)
    summary = {
        "run_name": config.run_name,
        "timestamp": timestamp,
        "git_commit": _git_commit(),
        "config": config.as_dict(),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "best_epoch": _best_epoch(history),
        "best": best,
        "history": history,
        "weights": str(weights),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    print(
        f"\nbest.pt (epoch {summary['best_epoch']}) on the project's val leaves: "
        f"Dice {best['val_dice']:.4f}  IoU {best['val_iou']:.4f}  "
        f"area {best['val_area_ratio']:.3f} ({best['val_area_bias_pct']:+.1f} %)"
    )
    print(f"weights -> {weights}")
    print(f"test set held out: {len(test_samples)} leaves (never seen during training)")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(config_from_args(args), timestamp=datetime.now(UTC).isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
