# Necrosis segmentation leaderboard

Every row is scored by the same command on the same leaves:

```bash
poetry run python tools/eval_necrosis.py --split test --weights <checkpoint> [--arch <arch>]
```

- **Data**: `data/leaves-native` regenerated from the Roboflow YOLO-seg export
  `necrosis-dsjfsrgsrg` v1 (polygons, 278 leaves), see `report-necrosis.csv`.
- **Split**: scan-grouped, seed 0, 15 % / 15 % → train 194 / val 43 / **test 41**. The test
  fold is never seen by training or checkpoint selection. A run trained on any other pool or
  seed is not comparable and does not belong here.
- **Metric to rank on**: Dice at the run's best threshold. Read the **pooled** area ratio
  (Σ predicted / Σ annotated) and the **median** per-leaf ratio for bias; the mean per-leaf
  ratio is kept for continuity with v1 but is dominated by near-healthy leaves.
- **Threshold**: sweep 0.3 / 0.5 / 0.7 (`--thresholds`), report the best and say which.

| # | run | model | train | thr | Dice | IoU | area pooled | area median | area mean | weights | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nec-yolo26s` run 3 | YOLO26s-sem, pretrained, 384² tiles | batch 128, lr 1e-3 AdamW, best ep 6/26, 2026-09-13 | 0.3 | **0.694** | 0.613 | **1.02** | 1.01 | 1.90 | `runs/keep/nec-yolo26s-run3/weights/best.pt` · volume `keep/nec-yolo26s-run3/` · sha256 `e0b26d29de00d1d6…` | `663c0e5` |
| 2 | `nec-yolo26m` run 1 | YOLO26m-sem, pretrained, 384² tiles | batch 128, lr 1e-3 AdamW, best ep 29/49, 2026-09-13 | 0.3 | 0.619 | 0.557 | 0.90 | 0.93 | 1.48 | `runs/keep/nec-yolo26m-run1/weights/best.pt` · volume `keep/nec-yolo26m-run1/` · sha256 `2e8acb908cb7d92d…` | `663c0e5` |
| – | v1 baseline | U-Net (Keras port), 304×3072 strips | 2023, `data/necrosis-model-375.safetensors` | 0.5 | 0.620 | 0.536 | 0.78 | 0.80 | 1.25 | `data/necrosis-model-375.safetensors` | – |

Same checkpoints at other thresholds, for reference:

| run | thr | Dice | IoU | pooled | median | mean |
|---|---|---|---|---|---|---|
| run 3 | 0.5 | 0.681 | 0.605 | 0.96 | 0.99 | 1.58 |
| run 3 | 0.7 | 0.652 | 0.578 | 0.90 | 0.92 | 1.31 |
| 26m run 1 | 0.5 | 0.581 | 0.520 | 0.84 | 0.82 | 1.21 |
| 26m run 1 | 0.7 | 0.535 | 0.473 | 0.79 | 0.68 | 0.99 |
| v1 | 0.8 | 0.611 | 0.526 | 0.77 | 0.78 | 1.21 |

## Notes

- Batch 128 on 194 leaves is 12 iterations per epoch; validation mIoU then swings by
  ±0.1 between consecutive epochs and checkpoint selection on 43 leaves is close to a
  lottery (26m run 1: 0.69 → 0.82 → 0.78 over three epochs). Prefer batch 16–32 and
  `patience 40` so a run sees a few thousand iterations before it is judged.

## Adding a run

1. Train with the defaults for pool and split (do not touch `--seed`, `--val-fraction`,
   `--test-fraction`).
2. Pull `weights/best.pt` (or `best.safetensors`), `results.csv`, `manifest.json` into
   `runs/keep/<run-name>/` and copy them to `keep/<run-name>/` on the `septosympto-runs`
   volume, so a later run with the same name cannot overwrite them:
   `modal volume cp septosympto-runs <run>/weights/best.pt keep/<run>/weights/best.pt`.
3. Score with the command above, sweep the thresholds, add the row, note the commit.
