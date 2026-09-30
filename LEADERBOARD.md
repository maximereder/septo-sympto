# SeptoSympto leaderboard

Two tasks, two protocols, one rule: every row is scored by the same command on the same
held-out leaves as every other row in its section. A checkpoint enters the catalogue in
`septosympto/zoo.py` by being promoted from here.

## Necrosis segmentation

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

The 2026-09 annotation correction replaced `data/leaves-native/labels/` (pycnidia points) and
left `mask/` untouched, so it does not reach this fold: the rows below stand as scored.

| # | run | model | train | thr | Dice | IoU | area pooled | area median | area mean | weights | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nec-yolo26s` run 3 | YOLO26s-sem, pretrained, 384² tiles | batch 128, lr 1e-3 AdamW, best ep 6/26, 2026-09-13 | 0.3 | **0.694** | 0.613 | **1.02** | 1.01 | 1.90 | `runs/keep/nec-yolo26s-run3/weights/best.pt` · volume `keep/nec-yolo26s-run3/` · sha256 `e0b26d29de00d1d6…` | `663c0e5` |
| 2 | `nec-yolo26l` run 1 | YOLO26l-sem, pretrained, 384² tiles | batch 128, lr 1e-3 AdamW, best ep 20/40, 2026-09-13 | 0.3 | 0.656 | 0.589 | 1.07 | 1.01 | 2.15 | `runs/keep/nec-yolo26l-run1/weights/best.pt` · volume `keep/nec-yolo26l-run1/` · sha256 `8f13ae4d574cd8d0…` | `663c0e5` |
| 3 | `nec-yolo26m` run 1 | YOLO26m-sem, pretrained, 384² tiles | batch 128, lr 1e-3 AdamW, best ep 29/49, 2026-09-13 | 0.3 | 0.619 | 0.557 | 0.90 | 0.93 | 1.48 | `runs/keep/nec-yolo26m-run1/weights/best.pt` · volume `keep/nec-yolo26m-run1/` · sha256 `2e8acb908cb7d92d…` | `663c0e5` |
| – | v1 baseline | U-Net (Keras port), 304×3072 strips | 2023, `data/necrosis-model-375.safetensors` | 0.5 | 0.620 | 0.536 | 0.78 | 0.80 | 1.25 | `data/necrosis-model-375.safetensors` | – |

Same checkpoints at other thresholds, for reference:

| run | thr | Dice | IoU | pooled | median | mean |
|---|---|---|---|---|---|---|
| run 3 | 0.5 | 0.681 | 0.605 | 0.96 | 0.99 | 1.58 |
| run 3 | 0.7 | 0.652 | 0.578 | 0.90 | 0.92 | 1.31 |
| 26l run 1 | 0.5 | 0.647 | 0.578 | 1.00 | 0.99 | 1.82 |
| 26l run 1 | 0.7 | 0.625 | 0.550 | 0.93 | 0.92 | 1.54 |
| 26m run 1 | 0.5 | 0.581 | 0.520 | 0.84 | 0.82 | 1.21 |
| 26m run 1 | 0.7 | 0.535 | 0.473 | 0.79 | 0.68 | 0.99 |
| v1 | 0.8 | 0.611 | 0.526 | 0.77 | 0.78 | 1.21 |

### Notes

- Batch 128 on 194 leaves is 12 iterations per epoch; validation mIoU then swings by
  ±0.1 between consecutive epochs and checkpoint selection on 43 leaves is close to a
  lottery (26m run 1: 0.69 → 0.82 → 0.78 over three epochs; 26l run 1 swings 0.72–0.81 over its last eight). With s > l > m under this regime, the size order is within selection noise. Prefer batch 16–32 and
  `patience 40` so a run sees a few thousand iterations before it is judged.

### Test-fold annotations under review

Both YOLO runs miss the same five leaves in the same way, which points at the annotation
rather than the model (`runs/keep/nec-yolo26s-run3/test-both-fail.png`). Until the
researcher rules, they cap the reachable Dice on this fold equally for every row; the ranking
stands, the absolute numbers are a floor.

| leaf | annotated | what the models do | status |
|---|---|---|---|
| `Rec_Apa_4_Rec_1__1` | 20 % — brown core plus its pale halo | segment the brown core only | **annotation error suspected** (Maxime: would predict as the model does) — sent to the researcher, 2026-09-13 |
| `Cal_Fru_2_Cal_1__3` | 21 % — polygon extends over plain green | the one brown patch | sent to the researcher |
| `Tit_Acc_2_Acc_2__2` | 15 % — large triangle over green | the brown patch inside it | sent to the researcher |
| `Gen_Gen_11_Gen_2__3` | 8 % — brown zone plus halo | brown zone only | definition question: brown only, or brown + halo? |
| `Des_SYM_1_SYM_2__4` | 50 % — whole mottled leaf | yellow patches only | definition question |

When a ruling comes back, fix the polygons on Roboflow, re-export, rerun the regeneration
(`--fresh`), re-push the volume and re-score every kept checkpoint — the fold changes for all.

---

## Pycnidia counting

```bash
poetry run python tools/eval_pycnidia.py --split test --weights <name-or-checkpoint> \
    [--arch <arch>] --thresholds 0.15 0.2 0.3 0.5
```

- **Data**: `data/leaves-native`, pycnidia points from the `pycnidia_corrected` v1 export
  (202 leaves, 40 004 points), see `report-pycnidia-corrected.csv`. The correction removed
  19.4 % of the previous points, so **anything scored against the earlier labels is not
  comparable** and does not belong here.
- **Split**: scan-grouped, seed 0, 15 % / 15 % → train 140 / val 33 / **test 29**. Dropping
  the eight uncorrected leaves reshuffled the folds: this test fold shares only 4 leaves with
  the one used before the correction.
- **Metric to rank on**: **MAE on the count**, the biological quantity. Read **slope** beside
  it — a model can buy a low MAE by under-counting the loaded leaves — and F1 (within
  `--match-radius-px 8`) to catch a model that gets the number right while placing its points
  badly.
- **Threshold**: sweep 0.15 / 0.2 / 0.3 / 0.5, report the best and say which.

| # | run | model | train | thr | MAE | RMSE | bias | slope | R² | F1 | weights | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `pyc-p2p-convnext-corrected` | P2PNet/ConvNeXt-T, pretrained, 3072×384 | batch 4, lr 5e-5 AdamW, best ep 40/61, 2026-09-26 | 0.15 | **33.0** | 52.1 | +11.6 | **1.007** | 0.906 | 0.732 | published as `p2p-convnext-v3` · `runs/pyc-p2p-convnext-corrected/best.safetensors` · volume `pyc-p2p-convnext-corrected/` · sha256 `ade8ed0e271974e1…` | `3f11065` (data: `c89667c`) |
| – | baseline `p2p-convnext-v2` | P2PNet/ConvNeXt-T, same architecture | trained on the **pre-correction** labels, 2026-07-20 | 0.5 | 36.3 | 57.8 | −12.1 | 0.807 | 0.885 | 0.687 | zoo card `p2p-convnext-v2`, NOT UPLOADED | – |

The baseline is a baseline, not a run: it was trained on a different pool with different
labels, so it cannot be ranked against row 1. It is here because it is what row 1 replaced.

Same checkpoints at other thresholds:

| run | thr | MAE | RMSE | bias | slope | R² | F1 |
|---|---|---|---|---|---|---|---|
| row 1 | 0.10 | 48.3 | 70.9 | +37.9 | 1.107 | 0.826 | 0.714 |
| row 1 | 0.20 | 33.8 | **48.5** | −6.9 | 0.939 | **0.919** | **0.739** |
| row 1 | 0.25 | 38.5 | 54.5 | −22.1 | 0.875 | 0.897 | 0.733 |
| row 1 | 0.30 | 44.1 | 62.6 | −34.2 | 0.826 | 0.865 | 0.727 |
| row 1 | 0.50 | 72.6 | 100.7 | −72.6 | 0.648 | 0.650 | 0.662 |
| baseline | 0.30 | 65.2 | 85.6 | +55.9 | 1.058 | 0.747 | 0.676 |
| baseline | 0.40 | 41.7 | 58.4 | +17.6 | 0.919 | 0.882 | 0.685 |
| baseline | 0.60 | 49.0 | 76.3 | −40.9 | 0.699 | 0.799 | 0.666 |

### Notes

- **The zoo ships v3 at 0.20, not at the 0.15 that ranks here.** MAE ranks by protocol, and
  0.15 wins it — by 0.8 leaves-worth of error on a 29-leaf fold, which is noise. At 0.20 the
  same checkpoint has the lower RMSE, the higher R² and the higher F1, so that is the
  operating point the card carries. If you rerun this sweep, expect the two to keep trading
  places on MAE.
- **The gap to the baseline is not where it looks.** 36.3 → 33.0 on MAE is small; 0.807 →
  1.007 on slope is not. The baseline reaches its best MAE by proportionally under-counting
  the loaded leaves, which is exactly the failure that distorts a comparison between
  genotypes. Precision rises 0.712 → 0.754 at the shipped threshold, the phantom points the
  old labels taught it are gone, and on the two leaves the correction emptied it predicted 9
  and 3 pycnidia where row 1 predicts none.
- **This checkpoint is not frozen yet.** Unlike the necrosis rows it still lives under its run
  name, local and on the volume, so a rerun with the same name would overwrite it. Promote it
  to `runs/keep/` and `keep/` before it is cited anywhere.
- 29 leaves carrying 5 169 pycnidia is a small fold, and one leaf (611 annotated points)
  contributes a sixth of the error on its own. Treat differences under ~3 MAE as noise.

---

## Adding a run

1. Train with the defaults for pool and split (do not touch `--seed`, `--val-fraction`,
   `--test-fraction`). A run on any other pool or seed is not comparable and does not belong
   here.
2. Pull `weights/best.pt` (or `best.safetensors`), `results.csv`, `manifest.json` into
   `runs/keep/<run-name>/` and copy them to `keep/<run-name>/` on the `septosympto-runs`
   volume, so a later run with the same name cannot overwrite them:
   `modal volume cp septosympto-runs <run>/weights/best.pt keep/<run>/weights/best.pt`.
3. Score with the command for the task — `tools/eval_necrosis.py` or
   `tools/eval_pycnidia.py` — sweep the thresholds, add the row, note the commit. If the
   worker recorded no commit (Modal has no git tree), use the local `HEAD` at launch and say
   which commit carried the data.
