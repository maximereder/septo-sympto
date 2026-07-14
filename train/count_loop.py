"""The training loop for a point counter.

Task-generic, architecture-generic. It never builds a target or computes a loss
itself: the model owns ``loss(output, target_points)`` and ``decode(output)``, so
a heatmap counter and a P2P network with Hungarian matching run through the exact
same loop. The loop's only jobs are batching, optimisation, and evaluation.

Selection is by validation **MAE on the count**, the biological quantity, with
localisation precision/recall/F1 logged beside it through
:func:`septosympto.eval.counting_report`. A count model that gets the number
right by luck while placing points badly is visible, not hidden.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from septosympto.eval import counting_report
from train.config import CountConfig
from train.count_data import PointSample, PycnidiaPointDataset, collate_points


class CountingModel(Protocol):
    """The training contract a registered counter satisfies."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
    def loss(self, output: torch.Tensor, target_points: list[torch.Tensor]) -> torch.Tensor: ...
    def decode(self, output: torch.Tensor, threshold: float) -> list[np.ndarray]: ...


def _evaluate(model, loader, device, threshold, radius):
    model.eval()
    pred_points, true_points = [], []
    with torch.inference_mode():
        for images, points in loader:
            output = model(images.to(device))
            for pred, truth in zip(model.decode(output, threshold), points, strict=True):
                pred_points.append(np.asarray(pred))
                true_points.append(truth.numpy())
    return counting_report(pred_points, true_points, radius_px=radius)


def _git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def train_counter(
    model: torch.nn.Module,
    config: CountConfig,
    train_samples: list[PointSample],
    val_samples: list[PointSample],
    *,
    timestamp: str,
    progress: bool = True,
) -> dict:
    """Train ``model``, select the best epoch by validation MAE, save it."""
    device = torch.device(config.device)
    model = model.to(device)

    train_ds = PycnidiaPointDataset(
        train_samples, config.imgsz, hflip=config.hflip, vflip=config.vflip, seed=config.seed
    )
    val_ds = PycnidiaPointDataset(val_samples, config.imgsz)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, collate_fn=collate_points,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, num_workers=config.num_workers,
        collate_fn=collate_points,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_mae = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    epoch_iter = range(config.epochs)
    if progress:
        from tqdm import tqdm

        epoch_iter = tqdm(epoch_iter, desc=config.run_name)

    for epoch in epoch_iter:
        model.train()
        running = 0.0
        for images, points in train_loader:
            optimizer.zero_grad()
            output = model(images.to(device))
            loss = model.loss(output, points)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.shape[0]
        train_loss = running / len(train_ds)

        report = _evaluate(
            model, val_loader, device, config.decode_threshold, config.match_radius_px
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mae": report.mae,
                "val_rmse": report.rmse,
                "val_bias": report.bias,
                "val_slope": report.slope,
                "val_f1": report.f1,
            }
        )
        if progress:
            epoch_iter.set_postfix(loss=f"{train_loss:.4f}", mae=f"{report.mae:.1f}",
                                   f1=f"{report.f1:.3f}")

        if report.mae < best_mae:
            best_mae = report.mae
            best_epoch = epoch
            epochs_without_improvement = 0
            save_file(
                {k: v.contiguous() for k, v in model.state_dict().items()},
                str(out_dir / "best.safetensors"),
                metadata={"epoch": str(epoch), "val_mae": f"{report.mae:.4f}"},
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    summary = {
        "run_name": config.run_name,
        "task": "counting",
        "timestamp": timestamp,
        "git_commit": _git_commit(),
        "config": config.as_dict(),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "best_epoch": best_epoch,
        "best": history[best_epoch],
        "history": history,
        "weights": str(out_dir / "best.safetensors"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    return summary


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
