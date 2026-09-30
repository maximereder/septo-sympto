# Training

Retraining the models, scoring them, and rebuilding the datasets they learn from. None of this
is needed to *use* SeptoSympto — for that, read the [README](../README.md).

The `train/` package retrains the necrosis segmenter from scratch. It is kept out of the
shipped `septosympto` package — installing the tool for inference does not pull Modal or the
training loop:

```bash
poetry install --with train
```

Everything runs locally and is tested locally; Modal is a thin launcher over the same
functions, not where the logic lives.

Two things it does that v1's training did not. Masks are **binarised**, not read as
greyscale and divided by 255 — the bug that capped the reported Dice at 0.523. And the split
is **grouped by scan**: all 375 images are pooled and re-partitioned so no scan has leaves in
two folds, with a held-out test set that no training step touches. Validation each epoch goes
through `septosympto.eval`, so the metric that selects the checkpoint is the real binary Dice,
and the **area bias is logged next to it** — the quantity whose neglect shipped a model
under-recovering necrotic area by 19 %.

## Local

```bash
poetry run python -m train.run \
    --dataset data/leaves-native --arch unet-resnet18 \
    --run-name nec-r18 --epochs 100 --device mps
```

`--dataset` is the native letterbox set by default — `img/*.png` with a `mask/*.png` beside
it, 355 annotated leaves on the shared 384×3072 canvas, the exact geometry
`TorchSegmenter` presents at inference. A Roboflow zip (`data/necrosis/dataset/300.zip`,
stretched strips) is still accepted, for comparison against v1. Outputs land in
`runs/<run-name>/`: `best.safetensors` and a `manifest.json` recording the config, the git
commit, the split sizes, and the full per-epoch history.

Three segmentation architectures are registered; `--arch` picks one and nothing else changes:

| arch | encoder | params | note |
|---|---|---|---|
| `unet` | hand-built, from scratch | 31.1 M | the v1 network; `imgsz` divisible by 16 |
| `unet-resnet18` | ResNet-18, ImageNet | 14.3 M | light, converges fast on 250 leaves; `imgsz` divisible by 32 |
| `unet-convnext-t` | ConvNeXt-Tiny, ImageNet | 31.9 M | higher capacity; `imgsz` divisible by 32 |

The pretrained variants reuse the backbones the P2P counters are built on
(`septosympto/models/backbones.py`) under a light U-Net decoder. To add another: define it in
`septosympto/models/`, decorate it `@register_segmenter("name")`, return `(N, 1, H, W)`
logits, and train it with `--arch name`.

## Scoring checkpoints on the held-out test fold

```bash
poetry run python tools/eval_necrosis.py --split test \
    --weights runs/nec-r18/best.safetensors runs/nec-yolo26s/weights/best.pt \
    --arch unet-resnet18
```

`test` is the fold of the project's scan-grouped split that no training step or checkpoint
selection ever touched — 41 leaves on the current set, identical for every run at the same
seed. `.safetensors` checkpoints go through `TorchSegmenter` with `--arch`, `.pt` through
`YoloSegmenter`, so a U-Net and a YOLO are scored on the same leaves by the same code, at the
leaf's own resolution. The v1 U-Net (`data/necrosis-model-375.safetensors`, `--arch unet`)
is the baseline to beat; on the current test fold it scores Dice 0.62 with a +25 % area bias
at threshold 0.5.

## Regenerating the native necrosis set

`data/leaves-native/` is produced by `tools/regen_pycnidia_native.py` from the 1200 dpi
TIFF crops (`LM1__allcrop`): each annotated leaf is re-cut at native scale, letterboxed onto
the 3072×384 canvas, and its annotation carried into the new frame by the same transform.
The strip is matched to its native leaf by image cross-correlation over every (leaf ×
flip) — the shipped `__N` index is not trusted — and weak or ambiguous matches are flagged
in `report-necrosis.csv` rather than written.

Two necrosis sources are read. The legacy zip carries pixel masks, which are warped. A
Roboflow **YOLO-seg export** (`*/images/*.jpg` + `*/labels/*.txt` polygons, `data.yaml`)
carries polygons: their vertices are remapped and rasterised directly on the canvas, so the
mask is drawn once at output resolution. Only the class named `necrosis` is kept; polygons of
any other class are annotation slips and are dropped, and a leaf left with nothing but slips
is flagged `unannotated` — an empty label file, by contrast, is a genuine necrosis-free leaf.

A leaf that already has its canvas in `img/` — written by an earlier pass, pycnidia or
necrosis, and validated by that pass's oracle — needs **no TIFF**: the strip is matched to the
canvas over the four flips (the leaf is known, only its orientation is open) and the polygons
are rasterised straight onto it. On the current export that covers 278 of 297 leaves and
agrees with the TIFF-based orientation on every one of the 217 leaves both methods saw. TIFFs
are only consulted for leaves with no canvas yet; `--from-tiffs` forces the TIFF path for
every leaf, to regenerate the canvases themselves.

```bash
poetry run python tools/regen_pycnidia_native.py --task necrosis \
    --nec-src data/necrosis/roboflow-v1 --tiffs ~/Downloads/LM1__allcrop \
    --out data/leaves-native --fresh
```

`--fresh` wipes `mask/` first, so masks from an older source do not linger beside the new
ones. `img/` is shared with the pycnidia set and is left alone. `--tiffs ""` runs without
TIFFs at all; leaves that would need one are reported `no-tiff`.

## Importing corrected pycnidia annotations

The researcher re-annotates on our own canvases — `leaves-native/img/` re-uploaded to
Roboflow — so a corrected export needs no geometric transform, only the `_png.rf.<hash>`
suffix taken off. `tools/import_pycnidia_corrected.py` does that and, more to the point,
refuses to do it blindly: every exported image is compared pixel-wise against the canvas it
claims to be (a JPEG round-trip costs 1–2 grey levels, a re-cut leaf costs far more), and the
annotations are checked to be single-class points inside the frame.

```bash
poetry run python -m tools.import_pycnidia_corrected ~/Downloads/export.zip --dry-run
poetry run python -m tools.import_pycnidia_corrected ~/Downloads/export.zip
scripts/push_data.sh native        # the volume is not pruned by itself
```

The previous labels are archived to `data/leaves-native-labels-<date>.tar.gz` first, and
`report-pycnidia-corrected.csv` records the per-leaf before/after counts. An **empty** label
file is kept as written: it is a leaf judged to carry no pycnidium, which the loader reads as
a genuine zero-count sample. A leaf **missing** from the export has not been corrected, so its
old labels are moved to `labels-stale/` rather than left to pass as ground truth — the loader
takes a leaf only when it has a label file, so this drops it from the pycnidia pool while its
image stays available to the necrosis task.

The 2026-09-23 correction (`pycnidia_corrected` v1) was a pure deletion pass: 49 602 → 40 004
points (**−19.4 %**), 9 774 removed against 176 added, kept points untouched to the pixel. It
touched 194 of 202 leaves, emptied 12 outright, and bites hardest on lightly infected leaves
(−48 % on the lowest count quartile, −13 % on the highest) — the removed points sit on
distinctly fainter spots. **Counting scores from before it do not compare to scores after**,
and a counter trained on the old labels over-counts by about a fifth against this ground
truth. Eight leaves were never uploaded and are now in `labels-stale/`.

## On a GPU with Modal

```bash
scripts/push_data.sh native                  # once: data/leaves-native -> volume
modal run train/modal_app.py::necrosis --arch unet-resnet18 --run-name nec-r18 --gpu A10
```

Data is uploaded once into a Modal Volume; checkpoints are written to a second Volume that
outlives the container. Requires a configured Modal account.

## YOLO26 semantic segmentation

Necrosis is a binary, dense mask — semantic segmentation, not instance segmentation. So the
YOLO comparison is [YOLO26](https://docs.ultralytics.com/models/yolo26)'s **`-sem`** head
(PNG masks, `nc: 1`, BCE + Dice), not `-seg`, which would mean cutting every lesion into
polygons and merging instances back at inference. Ultralytics owns its training loop, data
format and checkpoints, so it does not go through the segmenter registry; `train.yolo_run`
wraps it so a YOLO run is a peer of a PyTorch run:

- the pool, the scan-grouped split and the seed are the project's, so at the same settings a
  YOLO run and a U-Net run train and validate on **the same leaves**;
- the export to Ultralytics' layout is written per run under `runs/<run-name>/dataset/`, masks
  as class ids `{0, 1}` (in a semantic mask 255 is the *ignore* label);
- training runs on **square 384×384 tiles** of the canvas (8 per leaf, `imgsz 384`). This is
  not a memory trick: Ultralytics' semantic trainer scales the *short* side of a training
  image to `imgsz` and the *long* side of a validation image, so a full 384×3072 strip at
  `imgsz 3072` trains on ×8 zooms of the leaf centre and validates on the whole strip — a gap
  that collapses validation after a few epochs. With square tiles both scalings are the
  identity. The network is fully convolutional and is run on the whole canvas at inference;
  geometry augmentations other than flips are off, since inference never scales a leaf.
  Ultralytics' colour jitter (`hsv_s 0.7`, `hsv_v 0.4`) stays on by default; necrosis is a
  colour-defined class, so `--extra '{"hsv_s": 0.3, "hsv_v": 0.2}'` is the first knob to try;
- when Ultralytics is done, its `best.pt` is re-evaluated on the project's validation leaves
  through `septosympto.eval` — binary Dice and area bias, the numbers the U-Net reports — and a
  `manifest.json` of the same shape is written next to Ultralytics' `results.csv`.

Ultralytics prints its per-epoch table as it trains, locally and — through `modal run` —
remotely.

```bash
poetry install --extras yolo
poetry run python -m train.yolo_run \
    --dataset data/leaves-native --model yolo26n-sem.pt \
    --run-name nec-yolo26n --epochs 100 --device mps

modal run train/modal_app.py::yolo --model yolo26s-sem.pt --run-name nec-yolo26s --gpu A10
```

`--model` takes any of `yolo26{n,s,m,l,x}-sem.pt` (pretrained, downloaded once into
`data/pretrained/`) or `yolo26n-sem.yaml` to train from scratch. Tiles are small, so the
default batch is 16; an A10 takes 64 for the `s` model. Anything Ultralytics'
`train` accepts and this CLI does not surface goes through `--extra '{"hsv_h": 0.0}'`.
At inference `YoloSegmenter` (`septosympto/adapters/yolo_segmenter.py`) wraps the checkpoint
as a `Segmenter`, feeding the same letterbox canvas as the U-Net — in **RGB**, the channel
order Ultralytics trains with, converted in the adapter and nowhere else.

## Pycnidia counting

Counting is a separate task from segmentation, so it has its own loop, data loader, and model
registry. Pycnidia are ~4 px discs, hundreds per leaf: the YOLO boxes are points in all but
name, and only their centres are used. There is no single output format across counting
architectures — a heatmap, a density map and a P2P point-set network differ in output, target
and loss — so a counter **owns its own `loss(output, points)` and `decode(output) -> points`**,
and the loop is identical across all of them. Two are registered:

- **`heatmap`** — a stride-4 keypoint detector. Simple and robust, but its heatmap grid merges
  pycnidia closer than the Gaussian window into one peak, so it under-counts dense leaves.
- **`p2p`** — a point-set network (P2PNet) with a Hungarian matcher. Points are explicit
  predictions, not grid peaks, so touching pycnidia stay separate. Slower (matching runs per
  image, ~0.2–0.4 s) but built for exactly this density. The backbone is swappable; four are
  registered, all sharing the neck, heads, matcher and loss, differing only in the feature
  extractor:

  | arch | backbone | params | note |
  |---|---|---|---|
  | `p2p` | VGG-16-BN | 18.1 M | the original; heavy activation memory at full resolution |
  | `p2p-resnet18` | ResNet-18 | 14.4 M | lightest, low memory — trains at a larger batch |
  | `p2p-resnet50` | ResNet-50 | 27.5 M | intermediate capacity |
  | `p2p-convnext-t` | ConvNeXt-Tiny | 31.2 M | modern backbone, higher capacity |

  Only the conv trunk of each backbone is used (VGG's fully-connected head, ~120 M params, is
  dropped). Pretrained ImageNet weights download on first use unless built with
  `pretrained=False`. Train any of them with `--arch <name>`; `counting_report` compares them
  on the same held-out leaves.

Both satisfy the same three-method contract, so `--arch heatmap` and `--arch p2p` train
through the same loop, data, and evaluation, and `counting_report` compares them on the same
held-out leaves.

```bash
poetry run python -m train.count_run \
    --dataset-dir data/pycnidia/train-200-aug-x3 data/pycnidia/valid-40 \
    --arch heatmap --run-name pycnidia-v2 --epochs 100 --device mps
```

> The `data/pycnidia/*-aug-x3` directories above are the original stretched 2048×200 strips
> and still carry the **uncorrected** annotations — they predate the 2026-09 correction and are
> kept only to reproduce older runs. Train on `data/leaves-native` for anything new.

The pre-augmented Roboflow directories are pooled, **deduplicated by leaf** (the mirror copies
that leaked in v1 collapse to one), and re-split grouped by scan. Each epoch reports MAE on the
count — the biological quantity that drives checkpoint selection — with localisation
precision/recall/F1 beside it. To add another counting architecture: define it in
`septosympto/models/`, decorate it `@register_counter("name")`, give it `forward` / `loss` /
`decode`, and train it with `--arch name`.

## Scoring counters on the held-out test fold

`tools/eval_pycnidia.py` is to counting what `tools/eval_necrosis.py` is to
segmentation: same pool, same scan-grouped split, same canvas and decode as the
training loop, so a number here and a `val_mae` in a manifest mean the same
thing. The fold is what differs — `val` drove checkpoint selection and flatters a
model, `test` was never looked at.

```bash
poetry run python tools/eval_pycnidia.py --split test \
    --weights p2p-convnext-v3 runs/my-run/best.safetensors \
    --arch p2p-convnext-t --device mps
```

MAE ranks, because the count is the published quantity; slope and bias say *how*
a model is wrong (slope under 1 is proportional under-counting, which distorts
comparisons between genotypes; a bias at slope 1 shifts every leaf alike and
largely cancels), and precision/recall/F1 within `--match-radius-px` catch a model
that gets the number right while placing its points badly.

What the annotation correction did to the published checkpoint, scored on the same
29 test leaves with only the ground truth swapped:

| truth | MAE | bias | slope | P | R | F1 |
|---|---|---|---|---|---|---|
| pre-correction labels | 51.1 | +8.7 | 1.03 | 0.676 | 0.703 | 0.689 |
| corrected labels | 65.2 | **+55.9** | 1.06 | 0.595 | 0.782 | 0.676 |

The model is not worse than it was; it was trained to reproduce annotations that
the correction removed. Recall *rises* (0.70 → 0.78 — it does find the real
pycnidia) while precision falls (0.68 → 0.60) and the bias goes from nearly
nothing to +56 per leaf. That is the signature of a counter that learned the
phantom points, and it is the number a retrain on the corrected labels has to beat.

## Training P2P on the largest pycnidia set, on a Modal GPU

The `p2p` architecture wants a GPU: a VGG backbone plus Hungarian matching is heavier than the
local smoke can show. The Modal setup follows a push-then-run convention — data is uploaded to
a volume once, out of band, and the worker reads it and fails fast if it is missing, so nothing
uploads on a run's hot path.

```bash
scripts/push_data.sh pycnidia
```

This uploads the pycnidia directories to the `septosympto-data` volume via `modal volume put`.
Then launch the run. The largest available set is `train-200-aug-x3` pooled with `valid-40`
(219 unique leaves after dedup), which is the default:

```bash
modal run train/modal_app.py::pycnidia \
    --arch p2p --run-name pyc-p2p \
    --epochs 200 --batch-size 4 --gpu A100-40GB
```

Checkpoints land in the `septosympto-runs` **volume**, not on your local disk:
`best.safetensors` (written every time validation improves) and, by default,
`last.safetensors` (every 25 epochs, `--checkpoint-every`), plus a manifest recording the
config, git commit, scan-grouped split sizes, and per-epoch MAE/F1. On Modal the volume is
committed on each checkpoint, so a crash at epoch 150/200 keeps its progress. Pull the weights
to your machine when the run is done:

```bash
modal volume get septosympto-runs pyc-p2p/best.safetensors ./
```

Local runs (`python -m train.count_run`) write the same files straight to `runs/<run-name>/`
on disk.

## Launch and walk away

A 200-epoch P2P run takes hours. To start it and close the terminal, combine Modal's `--detach`
(the app keeps running server-side after the client disconnects) with the entrypoint's own
`--detach` (it `spawn`s the job and returns at once instead of blocking on the result):

```bash
modal run --detach train/modal_app.py::pycnidia --detach --arch p2p --run-name pyc-p2p --gpu L40S
```

The first `--detach` is Modal's; the second is the entrypoint's. It prints a call id and exits;
the run continues on Modal. Checkpoints stream to the volume as they are written, so nothing is
tied to your machine staying on. Follow it in the Modal dashboard or with `modal app logs`, and
pull the weights when it is done. Without the entrypoint `--detach`, the run still survives a
dropped connection (thanks to Modal's `--detach`), but the terminal stays attached streaming
progress until you disconnect.
P2PNet's pretrained VGG downloads once into a `septosympto-cache` volume under `TORCH_HOME` and
persists. VGG at full resolution is memory-heavy: the default trains at 200×2048 with batch 4;
raise the batch or the resolution on an H100. Swap `--arch p2p` for `--arch heatmap` to train
the lighter baseline and compare — `counting_report` scores both on the same held-out leaves.

The same launcher trains necrosis: `modal run train/modal_app.py::necrosis --run-name nec-v2`.
Requires a configured Modal account.
