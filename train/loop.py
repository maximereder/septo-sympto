"""The training loop for a native PyTorch segmenter.

Architecture-agnostic: it trains anything mapping ``(N, 3, H, W)`` to
``(N, 1, H, W)`` logits, which is the contract the U-Net and any replacement
share. YOLO segmentation heads train through Ultralytics' own trainer and do not
come through here.

Validation each epoch goes through :mod:`septosympto.eval`, so the number that
drives early stopping and checkpoint selection is the real binary Dice, and the
area bias is logged next to it. That is the pair v1 lacked: it selected on a Dice
computed against a soft target, and never looked at area at all, which is how a
checkpoint under-recovering necrotic area by 19 % came to be shipped.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from septosympto.eval import segmentation_report
from train.config import TrainConfig
from train.data import NecrosisDataset, Sample
from train.loss import bce_dice_loss


def _evaluate(model, loader, device, threshold):
    model.eval()
    preds, truths = [], []
    with torch.inference_mode():
        for images, masks in loader:
            logits = model(images.to(device))
            batch = (torch.sigmoid(logits) > threshold).squeeze(1).cpu().numpy()
            for pred, truth in zip(batch, masks.squeeze(1).numpy().astype(bool), strict=True):
                preds.append(pred)
                truths.append(truth)
    return segmentation_report(preds, truths)


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


def train_segmenter(
    model: torch.nn.Module,
    config: TrainConfig,
    train_samples: list[Sample],
    val_samples: list[Sample],
    *,
    timestamp: str,
    progress: bool = True,
    on_checkpoint: Callable[[int], None] | None = None,
) -> dict:
    """Train ``model``, select the best epoch by validation Dice, save it.

    Writes ``best.safetensors`` (on every validation improvement) and, when
    ``config.checkpoint_every > 0``, ``last.safetensors`` every that many epochs,
    both under ``output_dir/run_name/``, plus ``manifest.json`` at the end.
    ``on_checkpoint(epoch)`` fires whenever a checkpoint is written, which the
    Modal launcher uses to commit the volume so a crash mid-run keeps its
    progress. ``timestamp`` is passed in, never read from the clock, so a run is
    reproducible.
    """
    device = torch.device(config.device)
    model = model.to(device)

    train_ds = NecrosisDataset(
        train_samples, config.imgsz, hflip=config.hflip, vflip=config.vflip, seed=config.seed
    )
    val_ds = NecrosisDataset(val_samples, config.imgsz)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, num_workers=config.num_workers)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_dice = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    epoch_iter = range(config.epochs)
    if progress:
        from tqdm import tqdm

        epoch_iter = tqdm(epoch_iter, desc=config.run_name)

    for epoch in epoch_iter:
        model.train()
        running = 0.0
        for images, masks in train_loader:
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = bce_dice_loss(logits, masks.to(device), config.bce_weight, config.dice_weight)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.shape[0]
        train_loss = running / len(train_ds)

        report = _evaluate(model, val_loader, device, config.threshold)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_dice": report.dice,
                "val_iou": report.iou,
                "val_area_ratio": report.area_ratio,
                "val_area_bias_pct": report.area_bias_pct,
            }
        )
        if progress:
            epoch_iter.set_postfix(loss=f"{train_loss:.3f}", dice=f"{report.dice:.4f}",
                                   area=f"{report.area_ratio:.3f}")

        saved = False
        if report.dice > best_dice:
            best_dice = report.dice
            best_epoch = epoch
            epochs_without_improvement = 0
            save_file(
                {k: v.contiguous() for k, v in model.state_dict().items()},
                str(out_dir / "best.safetensors"),
                metadata={"epoch": str(epoch), "val_dice": f"{report.dice:.6f}"},
            )
            saved = True
        else:
            epochs_without_improvement += 1

        if config.checkpoint_every and (epoch + 1) % config.checkpoint_every == 0:
            save_file(
                {k: v.contiguous() for k, v in model.state_dict().items()},
                str(out_dir / "last.safetensors"),
                metadata={"epoch": str(epoch), "val_dice": f"{report.dice:.6f}"},
            )
            saved = True

        if saved and on_checkpoint is not None:
            on_checkpoint(epoch)

        if epochs_without_improvement >= config.early_stopping_patience:
            break

    best = history[best_epoch]
    summary = {
        "run_name": config.run_name,
        "timestamp": timestamp,
        "git_commit": _git_commit(),
        "config": config.as_dict(),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "best_epoch": best_epoch,
        "best": best,
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
