# SeptoSympto — quantification of Septoria tritici blotch symptoms

SeptoSympto is a deep learning tool that quantifies **necrosis** and **pycnidia** on scanned
wheat leaves infected by *Zymoseptoria tritici*. It combines two models: a **U-Net**
(TensorFlow/Keras) that segments necrotic tissue, and a **YOLOv5** detector (PyTorch) that
locates pycnidia. Leaf isolation and area measurement use classical OpenCV image processing.

![With pycnidia](/pictures/Cad_Rub_3_Rub_2__1__1__1.webp)

If you use SeptoSympto in your research, please [cite the paper](#citation).

---

## How it works

The script processes a folder of scanned images in three stages.

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
resolution, then converted to cm² using `--pixels_for_cm`.

---

## Results

Results are written to `outputs/output_<n>/results.csv`:

```
leaf,leaf_area_px,leaf_area_cm2,necrosis_number,necrosis_area_ratio,necrosis_area_cm2,pycnidia_number,pycnidia_area_px,pycnidia_area_cm2,pycnidia_number_per_leaf_cm2,pycnidia_number_per_necrosis_cm2,pycnidia_area_cm2_per_necrosis_area_cm2,pycnidia_mean_area_cm2
9__1,680969.5,3.0181,2,0.707,2.1347,340,14938.5055,0.0662,112.65365627381465,159.27296575631235,0.03101138333255258,0.00019470588235294116
88__1,648293.5,2.8733,2,0.645,1.854,614,21092.5784,0.0935,213.691574148192,331.17583603020495,0.050431499460625674,0.00015228013029315962
14__1,638934.0,2.8318,1,0.855,2.4207,413,13490.1368,0.0598,145.84363302493114,170.6118065022514,0.024703598132771513,0.00014479418886198547
20__1,821680.5,3.6418,2,0.417,1.5194,13,570.0783,0.0025,3.569663353286836,8.55600895090167,0.00164538633671186,0.0001923076923076923
43__1,575263.5,2.5496,2,0.735,1.8736,438,18033.416,0.0799,171.79165359272042,233.7745516652434,0.04264517506404782,0.00018242009132420092
```

Alongside the CSV, the run directory contains `images_output/` (leaves with detections drawn)
and, when `--save-masks` is used, `masks/` (binary necrosis masks).

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

### The legacy TensorFlow environment

TensorFlow is no longer a dependency. The necrosis U-Net has been ported to PyTorch and its
weights transferred exactly — see [Migration](#migration-from-tensorflow-to-pytorch).

`requirements-legacy.txt` exists solely to rebuild the TensorFlow 2.15 environment needed to
*re-run* that conversion from the original `.h5` file. It pins Python 3.9 – 3.11 and, because
of `tensorflow-macos` / `tensorflow-metal`, installs only on macOS with Apple Silicon. On
Linux or Windows, replace those two lines with `tensorflow==2.15.0`.

```bash
uv venv --python 3.11 .venv-legacy
uv pip install --python .venv-legacy/bin/python -r requirements-legacy.txt safetensors
```

### Pre-trained models

Download the weights and place them in a `models/` folder at the repository root:

- [pycnidia-model.pt](https://drive.google.com/file/d/1WLIej7263MieoIrfGBtN7ljiZpE4NZy1/view?usp=share_link) — YOLOv5x6, pycnidia detection
- [necrosis-model-375.h5](https://drive.google.com/file/d/1BPOsgdUjoA8uCGht4-kL2Er3SbB4JalR/view?usp=share_link) — U-Net, necrosis segmentation

Training datasets: [SeptoSympto Datasets](https://drive.google.com/drive/folders/1a2VhXy-sMx77-BOHEgP7jXdWoIJI20s4?usp=sharing)

> On first run, `torch.hub` downloads the YOLOv5 source from GitHub at the pinned tag `v7.0`.
> An internet connection is required for that first run. The tag is pinned deliberately:
> loading YOLOv5 from `master` breaks against the pinned `ultralytics` version.

---

## Usage

Put your scans in `images/` and, optionally, a metadata CSV in `import/`. Images must be
scanned with the leaves **horizontal**. TIFF at 1200 dpi is the reference format.

```bash
python3 septo_sympto.py -w images -o results.csv -e .tif -d cpu
```

| Flag | Long form | Default | Description |
|---|---|---|---|
| `-w` | `--images_input` | `images` | Folder containing the input scans. |
| `-i` | `--import` | `None` | Metadata CSV to join onto the results (`;` separated). |
| `-o` | `--output` | `results.csv` | Name of the output CSV. |
| `-nm` | `--necrosis_model` | `models/necrosis-model-375.h5` | Path to the U-Net weights. |
| `-pm` | `--pycnidia_model` | `models/pycnidia-model.pt` | Path to the YOLOv5 weights. |
| `-e` | `--extension` | `.tif` | Extension of the input images. |
| `-is` | `--imgsz` | `304 3072` | Working size, given as **height width**. |
| `-d` | `--device` | `cpu` | `cpu`, `mps` (Apple Silicon), or a GPU index. |
| `-pc` | `--pixels_for_cm` | `472` | Pixels per cm. For a scan at *D* dpi, use *D* / 2.54 (1200 dpi → 472). |
| `-pt` | `--pycnidia_threshold` | `0.3` | Confidence threshold for pycnidia. |
| `-pn` | `--necrosis_threshold` | `0.8` | Probability threshold for necrosis. |
| `-dm` | `--draw_mode` | `all` | What to draw: `pycnidia`, `necrosis`, or `all`. |
| `-sm` | `--save-masks` | `False` | Save the binary necrosis masks. |
| `-ns` | `--no-save` | `False` | Skip writing annotated images. |

The script creates `images/`, `import/`, `models/` and `outputs/` if they do not exist. Each
run writes to a fresh `outputs/output_<n>/` directory. The `tools/` folder holds `metrics.py`,
which defines the segmentation metrics (`dice_coef`, `dice_loss`, `iou`) needed to deserialise
the U-Net.

---

## Migration from TensorFlow to PyTorch

The necrosis U-Net was originally a Keras model (`necrosis-model-375.h5`, 31 055 297
parameters). It has been re-implemented layer for layer in PyTorch
(`septosympto/models/unet.py`) and the trained weights transferred into it, rather than
retrained. The port is therefore numerically equivalent, not merely comparable.

Reproduce the transfer with:

```bash
.venv-legacy/bin/python tools/convert_keras_unet.py \
    --keras data/necrosis-model-375.h5 \
    --output data/necrosis-model-375.safetensors \
    --validate-dir <folder of leaf .jpg>
```

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

### The pycnidia detector will be retrained, not ported

`pycnidia-model.pt` was trained with the `ultralytics/yolov5` repository. It cannot be
carried into the modern stack, and this is not a packaging detail that a flag can fix:

- PyTorch ≥ 2.6 defaults `torch.load` to `weights_only=True` and refuses to deserialise it.
- The current `ultralytics` package detects the checkpoint and rejects it outright: *"appears
  to be an Ultralytics YOLOv5 model originally trained with .../yolov5. This model is NOT
  forwards compatible."* The old detection head is anchor-based; the current one is
  anchor-free. The head weights have no counterpart. The `yolov5su.pt` models that Ultralytics
  ships are retrained re-implementations, not the same weights.

Rather than freeze a bridge around a model that is already slated for replacement, the
pycnidia detector will be **retrained** on `data/pycnidia/` with a current architecture. The
annotations are points in all but name — the median bounding box is 4 × 4 px — which is the
real reason a plain object detector is the wrong tool here.

Until that lands, `septo_sympto.py` remains the reference pipeline and must be run from the
legacy environment. It is frozen: no new features, bug fixes only.

---

## Known issues and limitations

These are open defects in the current release, documented here pending the ongoing rework.
None of them prevent the tool from running.

**Wrong CSV column.** Since February 2023, the column labelled `pycnidia_number_per_leaf_cm2`
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

**Aspect ratio is not preserved.** Every leaf is resized to a fixed 304 × 3072, regardless of
its true proportions, so pycnidia are deformed by an amount that depends on leaf geometry.

---

## Training your own weights

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
