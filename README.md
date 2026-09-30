# SeptoSympto — quantification of Septoria tritici blotch symptoms

SeptoSympto is a deep learning tool that quantifies **necrosis** and **pycnidia** on scanned
wheat leaves infected by *Zymoseptoria tritici*. Leaf isolation and area measurement use
classical OpenCV image processing; the symptoms are measured by neural networks.

![With pycnidia](/pictures/Cad_Rub_3_Rub_2__1__1__1.webp)

If you use SeptoSympto in your research, please [cite the paper](#citation).

> ### ⚠ This branch is a rewrite in progress
>
> **v2 runs.** TensorFlow is gone, both models have been retrained and scored on a held-out
> test fold, and `septo-sympto` analyses a folder of scans end to end.
>
> The two best models are published and fetched on first use. Two older cards in the catalogue
> are not backed by a file yet — see [Pre-trained models](#pre-trained-models).
>
> **To reproduce the published results, use the v1 tag:**
>
> ```bash
> git checkout v1.0-legacy
> ```
>
> That tag is the implementation described in the paper. It runs `septo_sympto.py` on
> TensorFlow 2.15 (Python ≤ 3.11) and YOLOv5 v7.0. Read its
> [Known issues](#known-issues-and-limitations) before trusting its numbers: one CSV column is
> wrong, and necrosis area is underestimated by roughly 19 %.
>
> Everything below describes the v1 method, which v2 keeps, and the state of the migration.

---

## How it works

The v1 pipeline processes a folder of scanned images in three stages.

**1. Leaf isolation.** Each scan is thresholded in HSV space to separate leaf tissue from the
background. Contours larger than 50 000 px are treated as individual leaves, cropped to their
bounding box, and saved twice: once at native resolution (`cropped_not_resized`, used to
recover true areas) and once resized to 304 × 3072 px (`cropped`, fed to both models).

**2. Necrosis segmentation** (`predict_necrosis_mask`). The U-Net predicts, for every pixel,
the probability of belonging to a necrotic lesion. The probability map is binarised at the
`--necrosis_threshold` (default 0.8). Connected components are then filtered to keep only
those with an area above 300 px and a perimeter-to-area ratio below 0.9. The function returns
the total necrotic area and the number of lesions.

**3. Pycnidia detection** (`predict_pycnidia`). YOLOv5 predicts bounding boxes and confidence
scores. Boxes above `--pycnidia_threshold` (default 0.3) are retained, up to a maximum of
10 000 per leaf. The function returns the total pycnidia area and their count.

Areas measured in the 304 × 3072 working space are rescaled back to the native crop
resolution, then converted to cm² using `--pixels_for_cm` (default 472).

### Scale and leaf extent

Reference scans are TIFF at **1200 dpi**, which is 1200 / 2.54 = **472.44 px/cm**. That is
where the default of 472 comes from. v2 reads the resolution from the TIFF metadata instead
of taking it as an argument.

Leaves are laid horizontally and span the full width of the scan, so both tips fall outside
the image. `leaf_area_cm2` therefore measures **a standardised leaf segment**, not a whole
leaf. This is intentional and consistent across scans; all per-cm² densities are densities
over that segment. Two consequences worth keeping in mind when reporting: absolute leaf areas
are not whole-leaf areas, and scan widths vary (3078–3476 px in the reference set), so the
segment length is not identical from scan to scan.

---

## Results

### Leaf identity (v2)

v2 drops the `--import` metadata join. It was the source of two defects: the header declared
23 columns while each row wrote 19, so the four derived ratios were never emitted; and the
`leaf` column was overwritten with the scan name, making two leaves from the same scan
indistinguishable.

Instead, each leaf is identified by **two columns**, never by a parsed string:

| column | example | meaning |
|---|---|---|
| `image` | `Soi_LGA_2_Soi_1` | source scan, stem of the input filename |
| `leaf_index` | `2` | 1-based, in the order leaves are found on the scan |

A `leaf_id` of `Soi_LGA_2_Soi_1_2` can be composed for display, but nothing parses it back.
Scan names contain underscores, so a single-underscore identifier is ambiguous; v1 used a
double underscore (`Acc_Acc_4_Acc_1__2__1`) precisely to keep `split("__")[0]` working, and
the existing annotation filenames still rely on it. Two columns removes the problem rather
than encoding around it.

To attach experimental metadata, join on `image` downstream, in R or pandas, where a failed
join is visible instead of silently shifting columns.

### v1 output format

Results were written to `outputs/output_<n>/results.csv`:

```
leaf,leaf_area_px,leaf_area_cm2,necrosis_number,necrosis_area_ratio,necrosis_area_cm2,pycnidia_number,pycnidia_area_px,pycnidia_area_cm2,pycnidia_number_per_leaf_cm2,pycnidia_number_per_necrosis_cm2,pycnidia_area_cm2_per_necrosis_area_cm2,pycnidia_mean_area_cm2
9__1,680969.5,3.0181,2,0.707,2.1347,340,14938.5055,0.0662,112.65365627381465,159.27296575631235,0.03101138333255258,0.00019470588235294116
88__1,648293.5,2.8733,2,0.645,1.854,614,21092.5784,0.0935,213.691574148192,331.17583603020495,0.050431499460625674,0.00015228013029315962
14__1,638934.0,2.8318,1,0.855,2.4207,413,13490.1368,0.0598,145.84363302493114,170.6118065022514,0.024703598132771513,0.00014479418886198547
20__1,821680.5,3.6418,2,0.417,1.5194,13,570.0783,0.0025,3.569663353286836,8.55600895090167,0.00164538633671186,0.0001923076923076923
43__1,575263.5,2.5496,2,0.735,1.8736,438,18033.416,0.0799,171.79165359272042,233.7745516652434,0.04264517506404782,0.00018242009132420092
```

Alongside the CSV, the run directory contained `images_output/` (leaves with detections drawn)
and, when `--save-masks` was used, `masks/` (binary necrosis masks).

> **Note:** the example above was generated before a regression that currently affects the
> `pycnidia_number_per_leaf_cm2` column. See [Known issues](#known-issues-and-limitations).

---

## Installation

The project is managed with [Poetry](https://python-poetry.org/) and requires **Python 3.12
or 3.13**.

```bash
poetry install
```

This installs PyTorch, NumPy, OpenCV and pandas. It deliberately does **not** install
Ultralytics, which is AGPL-3.0: keeping it out of the default tree means a plain
`poetry install` yields an MIT-only dependency set. Install it explicitly when you need to
retrain or to experiment with YOLO segmentation heads:

```bash
poetry install --extras yolo
```

TensorFlow is gone. It is not an optional extra, not a legacy requirements file, not a
dependency of any kind. Everything at `v1.0-legacy` if you need to run the old pipeline.

### Pre-trained models

Models are published by **name** and fetched on first use. `septo-sympto --list-models` shows
what is available, what each was validated at, and what it scored on the held-out test fold:

| name | task | what it is | threshold | test score |
|---|---|---|---|---|
| `yolo26s-v2` (default) | necrosis | YOLO26s-sem trained on the 278-leaf native set | 0.3 | Dice 0.694, area ratio 1.02 |
| `unet-v1` | necrosis | the published 2023 U-Net, ported weight-for-weight | 0.8 | Dice 0.611, area ratio 0.77 — *not uploaded* |
| `p2p-convnext-v3` | pycnidia | P2PNet/ConvNeXt-T on the corrected labels | 0.2 | MAE 33.8, slope 0.94, F1 0.739 |
| `p2p-convnext-v2` | pycnidia | the same, on the pre-correction labels — *superseded* | 0.3 | MAE 65.2 there, 36.3 at 0.5 — *not uploaded* |

Weights live on the Hugging Face Hub at
[`maximereder/septo-sympto`](https://huggingface.co/maximereder/septo-sympto), downloaded once into
`~/.cache/septosympto/models/` (`SEPTOSYMPTO_HOME` overrides) and verified against their
SHA-256 on every load — a download whose digest does not match is refused, never used. The
catalogue is `septosympto/zoo.py`; a model gets in by being promoted from `LEADERBOARD.md`.
Training datasets remain on
[Google Drive](https://drive.google.com/drive/folders/1a2VhXy-sMx77-BOHEgP7jXdWoIJI20s4?usp=sharing).

**Two cards are not backed by a file yet.** `unet-v1` and `p2p-convnext-v2` are listed, and
`--list-models` marks them `NOT UPLOADED`, but asking for one 404s on any machine whose cache
is empty. Both are kept on purpose: `unet-v1` is the published model and belongs in the
catalogue, `p2p-convnext-v2` documents what the annotation correction replaced.

`p2p-convnext-v3` counts pycnidia at **threshold 0.20**, not the 0.3 its predecessor carried:
on the same 29 test leaves, v2 reaches MAE 36.3 at best (threshold 0.5) and gets there by
under-counting the loaded leaves — slope 0.807 against 0.939. See
[Pycnidia counting](#pycnidia-counting).

---

## Usage

Point it at a directory of scans. With no other option it uses the default models, reads the
scale from each TIFF's resolution tag, and writes one CSV row per leaf plus a manifest that
records exactly which weights (name and SHA-256) and parameters produced it:

```bash
septo-sympto scans/ -o results.csv
```

```
necrosis: yolo26s-v2 (threshold 0.3)
pycnidia: none
Acc_Acc_1_Acc_1.tif: 4 leaves
...
132 leaves from 33 scans -> results.csv
```

Choosing models — a published name or a checkpoint path, per task:

```bash
septo-sympto scans/ --necrosis unet-v1 -o v1.csv               # the 2023 model, for comparison
septo-sympto scans/ --necrosis runs/my-run/weights/best.pt -pn 0.3      # an unpublished YOLO run
septo-sympto scans/ --necrosis runs/r18/best.safetensors --necrosis-arch unet-resnet18 -pn 0.5
septo-sympto --list-models
```

Counting pycnidia too — the name carries the architecture and the threshold:

```bash
septo-sympto scans/ -o results.csv -d mps --pycnidia p2p-convnext-v3
```

A published name carries its own threshold; a path needs `--necrosis-threshold` /
`--pycnidia-threshold` (and, for `.safetensors`, the architecture) because nothing else knows
what the checkpoint was validated at. Other options: `-e .tiff` for the input extension,
`-pc 472.44` to force a scale when a scan carries none, `--min-lesion-area-mm2`, `--masks-dir`
to also write a necrosis mask and overlay per leaf, `-d mps` / `-d 0` for the device.

YOLO models need the `yolo` extra (`poetry install --extras yolo`); asking for one without it
says so and stops. The pycnidia columns stay empty unless `--pycnidia` names a counter, which
is off by default.

---

## Migration from TensorFlow to PyTorch

The necrosis U-Net was originally a Keras model (`necrosis-model-375.h5`, 31 055 297
parameters). It has been re-implemented layer for layer in PyTorch
(`septosympto/models/unet.py`) and the trained weights transferred into it, rather than
retrained. The port is therefore numerically equivalent, not merely comparable.

The conversion has been performed and its result committed as weights. The tooling that did it
lived under `tools/` and is preserved at `v1.0-legacy`, together with the TensorFlow
environment it needed. It is not carried into v2: TensorFlow will never be installed again,
and the five converted `.safetensors` files are the artifacts that matter.

Measured on six validation leaves at 304 × 3072, comparing Keras 2.15 on Python 3.11 against
PyTorch 2.13 on Python 3.12:

| quantity | deviation |
|---|---|
| max absolute difference on probabilities | 7.0 × 10⁻⁶ |
| mean absolute difference | 2.1 × 10⁻⁸ |
| pixels disagreeing at threshold 0.5 and 0.8 | 0 |
| **relative error on necrosis area** | **0.000000 %** |

The two Keras conventions that must be reproduced are `BatchNormalization(epsilon=1e-3)`
— PyTorch defaults to `1e-5` — and the decoder concatenation order `[upsampled, skip]`.
Both are enforced by the tests in `tests/test_unet.py`.

The safetensors artifact is 124 MB against 372 MB for the `.h5`, which carried the optimiser
state. It loads under any recent PyTorch, with no TensorFlow present.

### The pycnidia detector was retrained, not ported

`pycnidia-model.pt` was trained with the `ultralytics/yolov5` repository. It could not be
carried into the modern stack, and this was not a packaging detail that a flag could fix:

- PyTorch ≥ 2.6 defaults `torch.load` to `weights_only=True` and refuses to deserialise it.
- The current `ultralytics` package detects the checkpoint and rejects it outright: *"appears
  to be an Ultralytics YOLOv5 model originally trained with .../yolov5. This model is NOT
  forwards compatible."* The old detection head is anchor-based; the current one is
  anchor-free. The head weights have no counterpart. The `yolov5su.pt` models that Ultralytics
  ships are retrained re-implementations, not the same weights.

Rather than freeze a bridge around a model already slated for replacement, the pycnidia
detector was **retrained**. The annotations are points in all but name — the median bounding
box is 4 × 4 px — which is the real reason a plain object detector was the wrong tool, so what
replaced it is a point-set counter (P2PNet), trained on the undistorted `data/leaves-native/`
canvases rather than the stretched `data/pycnidia/` strips. See
[Pycnidia counting](#pycnidia-counting).

The v1 pipeline at `v1.0-legacy` remains the reference implementation for the published paper.

---

## Known issues and limitations

These are defects of the **published v1** (`v1.0-legacy`), the only released version. Each one
says whether v2 fixes it. None of them prevent v1 from running.

**The `--import` join drops four columns.** When a metadata CSV was supplied, `export_result`
wrote a header of 23 columns but only 19 values per row. `pycnidia_number_per_leaf_cm2`,
`pycnidia_number_per_necrosis_cm2`, `pycnidia_area_cm2_per_necrosis_area_cm2` and
`pycnidia_mean_area_cm2` were never written. The `leaf` column also received the scan name
stripped of its `__n` suffix, so leaves from the same scan produced identical, indistinguishable
rows. Any analysis run through the `--import` path is affected. v2 removes the join entirely.

**Wrong CSV column.** Without `--import`, since February 2023 the column labelled `pycnidia_number_per_leaf_cm2`
actually contains `necrosis_area_cm2 / leaf_area_cm2` — that is, a duplicate of
`necrosis_area_ratio`. The intended metric (pycnidia count per cm² of leaf) is not currently
emitted. Results produced since then should not rely on that column.

**Boolean flags behave inversely.** `--save-masks` and `--no-save` are parsed as strings, so
passing *any* value — including `False` — is interpreted as true. Omit the flag entirely to
get the default behaviour.

**Necrosis area is underestimated.** On the 67 validation leaves carrying necrosis, the
shipped `necrosis-model-375.h5` recovers about 81 % of the annotated necrotic area
(95 % CI [0.69, 0.95]). Binary Dice is 0.76 and IoU 0.67. Absolute necrotic areas are
therefore biased low; relative comparisons between treatments are much less affected.
A clean held-out test set is being assembled to select the operating point properly.

**Training-log metrics understate the models.** The Roboflow masks encode necrosis as magenta
`(255, 0, 124)`. Read as greyscale and divided by 255, the target becomes 0.354 rather than
1.0, which caps the achievable Dice at roughly 0.52. The `val_dice_coef` values in
`data/necrosis/results/*.csv` must be read with that ceiling in mind; the real binary Dice is
around 0.76–0.82.

**Leaf area is double-counted.** `get_leaf_area` sums the areas of all contours returned with
`RETR_TREE`, so internal holes are added rather than subtracted.

**Crops accumulate across runs.** Cropped leaves are written into `images/cropped/`, and the
inference loop iterates over everything found there. Running on a second batch without
clearing the folder will silently include leaves from the previous batch. **Delete
`images/cropped/` and `images/cropped_not_resized/` between runs.**

**Aspect ratio is not preserved, and the distortion tracks a genotypic trait.** Every leaf is
resized to a fixed 304 × 3072 regardless of its true proportions. Because leaves span the full
scan width, the horizontal scale barely changes (×0.88 to ×1.00); the entire distortion falls
on the vertical axis and is governed by one thing, leaf thickness.

Measured on 27 leaves from 7 reference scans, whose heights range from 169 to 398 px (a factor
of 2.4):

| | value |
|---|---|
| median anisotropy of the resize | 1.23 |
| 90th percentile | 1.64 |
| maximum | 1.94 |
| leaves distorted by more than 25 % | 12 / 27 |

A circular 6 px pycnidium becomes a 6 × 7.4 px ellipse on a median leaf, and 6 × 11.6 px on
the thinnest. Leaf thickness is a varietal trait, so the measurement distortion is confounded
with the genotype being compared.

The area arithmetic is sound: `convert_model_to_base_area` rescales areas exactly. What is
biased is the segmentation and detection performed upstream, in the distorted space. Working
at native resolution with tiling is the fix, and it is a v2 goal.

---

## Training (v2)

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

### Local

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

### Scoring checkpoints on the held-out test fold

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

### Regenerating the native necrosis set

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

### Importing corrected pycnidia annotations

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

### On a GPU with Modal

```bash
scripts/push_data.sh native                  # once: data/leaves-native -> volume
modal run train/modal_app.py::necrosis --arch unet-resnet18 --run-name nec-r18 --gpu A10
```

Data is uploaded once into a Modal Volume; checkpoints are written to a second Volume that
outlives the container. Requires a configured Modal account.

### YOLO26 semantic segmentation

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

### Pycnidia counting

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

### Scoring counters on the held-out test fold

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

### Training P2P on the largest pycnidia set, on a Modal GPU

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

### Launch and walk away

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

## Training the original weights (v1)

The models shipped with the paper were trained outside this repository, with the external
YOLOv5 and U-Net projects below. This is kept for provenance; new necrosis training goes
through `train/` above.

### Pycnidia — YOLOv5

Annotate with bounding boxes and export in *YOLOv5 PyTorch* format (e.g. via
[Roboflow](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/)). The dataset
layout is one `.txt` per image, sharing its basename:

```
dataset/
├── images/
│   ├── image1.jpg
│   └── ...
└── labels/
    ├── image1.txt
    └── ...
```

Each annotation line is `<class> <x_center> <y_center> <width> <height>`, with coordinates
normalised to [0, 1] and `class` = 0 for pycnidia.

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5 && pip install -r requirements.txt
python train.py --img <image_size> --batch <batch_size> --epochs <epochs> \
                --data <data.yaml> --weights yolov5s.pt
```

The shipped weights were trained from `yolov5x6.pt` for 400 epochs at `--img 3070`,
`--batch 12`, `--rect`. Larger batches and image sizes need proportionally more GPU memory.

Video tutorial: [YOLOv5 model training](https://www.youtube.com/watch?v=19VbN6IK1zM&ab_channel=LauraMATHIEU)
Reference: [YOLOv5](https://github.com/ultralytics/yolov5)

### Necrosis — U-Net

Annotate with semantic segmentation masks and export one `.png` per image. Masks must be
**binary** (0 or 255).

![Mask](pictures/mask.png)

> If you export from Roboflow, verify the encoding: Roboflow writes coloured masks, not
> binary ones. Converting a coloured mask to greyscale silently rescales the positive class
> and corrupts the training target. Binarise explicitly.

```
dataset/
├── images/
│   ├── image1.jpg
│   └── ...
└── masks/
    ├── image1.png
    └── ...
```

```bash
git clone https://github.com/maximereder/unet.git
cd unet
python train.py --data <data_folder> --csv <csv_output> --model <model_output> \
                --epochs <epochs> --batch-size <batch_size> --imgsz 304 3072
```

Arguments: `--data` (default `data`), `--csv` (default `results_unet_train.csv`), `--model`
(default `model.h5`), `--epochs` (default 100), `--batch-size` (default 2), `--img_ext`
(default `.jpg`), `--mask_ext` (default `.png`), `--imgsz` (default `304 3072`).

Video tutorial: [U-Net model training](https://www.youtube.com/watch?v=KhGBcwwc-zQ&ab_channel=LauraMATHIEU)
Reference: [U-Net](https://github.com/maximereder/unet)

---

## Citation

If you use SeptoSympto, please cite:

> Mathieu, L., Reder, M., Siah, A., Ducasse, A., Langlands-Perry, C., Marcel, T. C.,
> Morel, J.-B., Saintenac, C., & Ballini, E. (2024). SeptoSympto: a precise image analysis of
> Septoria tritici blotch disease symptoms using deep learning methods on scanned images.
> *Plant Methods*, 20(1), 18. https://doi.org/10.1186/s13007-024-01136-z

```bibtex
@article{mathieu2024septosympto,
  title    = {SeptoSympto: a precise image analysis of Septoria tritici blotch disease
              symptoms using deep learning methods on scanned images},
  author   = {Mathieu, Laura and Reder, Maxime and Siah, Ali and Ducasse, Aur{\'e}lie
              and Langlands-Perry, Camilla and Marcel, Thierry C. and Morel, Jean-Beno{\^i}t
              and Saintenac, Cyrille and Ballini, Elsa},
  journal  = {Plant Methods},
  volume   = {20},
  number   = {1},
  pages    = {18},
  year     = {2024},
  doi      = {10.1186/s13007-024-01136-z}
}
```

---

## Authors

- **Laura Mathieu**, PhD — laura.mathieu@supagro.fr — [LinkedIn](https://www.linkedin.com/in/laura-mathieu/)
- **Maxime Reder**, Deep Learning Engineer — maximereder@live.fr — [maximereder.fr](https://maximereder.fr)

See the [publication](https://doi.org/10.1186/s13007-024-01136-z) for the full author list.

## License

The SeptoSympto source code is released under the [MIT License](LICENSE).

The pycnidia weights (`pycnidia-model.pt`) were trained with Ultralytics YOLOv5 and are
distributed under **AGPL-3.0**, not MIT. See [NOTICE](NOTICE) for the full third-party
license breakdown and its practical consequences.

The associated article is published under CC BY 4.0.
