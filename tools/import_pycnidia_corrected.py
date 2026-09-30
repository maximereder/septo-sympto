"""Import a corrected Roboflow pycnidia export over ``data/leaves-native/labels``.

The researcher re-annotates on our own letterbox canvases (``leaves-native/img``,
3072x384), re-uploaded to Roboflow, so a corrected export needs no geometric
transform at all — only Roboflow's ``_png.rf.<hash>`` suffix has to come off. The
work here is therefore verification, not conversion:

* every export image is compared pixel-wise to the canvas it claims to be, so a
  re-cut or re-scaled upload can never be written over our labels as if it were
  the same leaf (JPEG re-encoding alone lands around 1-2 grey levels);
* the annotations are checked to be single-class points inside the frame;
* a leaf that is *missing* from the export has not been corrected. Its old labels
  are moved to ``labels-stale/`` rather than left in place: the loader takes a
  leaf only when it has a label file, so moving them drops those leaves from the
  pycnidia pool while leaving their images for the necrosis task.

An **empty** label file is meaningful and is kept — it is a leaf the researcher
judged to carry no pycnidium at all, which the loader reads as a zero-count
sample.

The previous labels are archived next to the dataset before anything is written,
and ``--dry-run`` reports without touching the tree.

    poetry run python -m tools.import_pycnidia_corrected ~/Downloads/export.zip --dry-run
    poetry run python -m tools.import_pycnidia_corrected ~/Downloads/export.zip
    scripts/push_data.sh native        # then re-push: the volume is not pruned by itself
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import tarfile
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

CANVAS_W, CANVAS_H = 3072, 384
MAX_IMAGE_DIFF = 6.0  # mean |delta| in grey levels; a JPEG round-trip costs ~1-2
MATCH_TOL_PX = 6.0


def base_name(path: Path) -> str:
    """``Acc_Acc_2__1__1_png.rf.<hash>.txt`` -> ``Acc_Acc_2__1__1``."""
    return re.sub(r"_(jpg|png)\.rf\.[0-9a-f]+$", "", path.stem)


def read_boxes(path: Path) -> np.ndarray:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if any(len(r) != 5 for r in rows):
        widths = sorted({len(r) for r in rows})
        raise ValueError(f"{path}: expected 5 fields per line, got {widths}")
    return np.array([[float(v) for v in r] for r in rows], np.float64).reshape(-1, 5)


def check_boxes(name: str, boxes: np.ndarray) -> None:
    if len(boxes) == 0:
        return
    classes = set(boxes[:, 0].astype(int))
    if classes != {0}:
        raise ValueError(f"{name}: expected class 0 only, found {sorted(classes)}")
    if not ((boxes[:, 1:3] >= 0).all() and (boxes[:, 1:3] <= 1).all()):
        raise ValueError(f"{name}: centres outside the frame")


def count_edits(old: np.ndarray, new: np.ndarray) -> tuple[int, int]:
    """(added, removed) between two point sets, pairing centres within a few px.

    The assignment cost is *gated* at ``2 * MATCH_TOL_PX``: without it, a handful
    of unpairable points drags the optimum into a long cascade of reshuffles and
    every count comes out inflated.
    """
    po, pn = old[:, 1:3] * [CANVAS_W, CANVAS_H], new[:, 1:3] * [CANVAS_W, CANVAS_H]
    if not len(po) or not len(pn):
        return len(pn), len(po)
    dist = np.linalg.norm(pn[:, None] - po[None], axis=2)
    ri, ci = linear_sum_assignment(np.minimum(dist, 2 * MATCH_TOL_PX))
    same = int((dist[ri, ci] <= MATCH_TOL_PX).sum())
    return len(pn) - same, len(po) - same


def export_labels(export: Path) -> dict[str, Path]:
    """Leaf id -> label file, over whatever split layout the export happens to use."""
    labels: dict[str, Path] = {}
    for path in sorted(export.rglob("labels/*.txt")):
        name = base_name(path)
        if name in labels:
            raise ValueError(f"{name} appears twice in the export")
        labels[name] = path
    return labels


def export_image(label_path: Path) -> Path:
    """The image beside a label file, under any of Roboflow's extensions."""
    images = label_path.parent.parent / "images"
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = images / (label_path.stem + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no image for {label_path.name}")


def verify_image(name: str, exported: Path, canvas: Path) -> float:
    """Mean absolute difference between an exported image and our canvas."""
    a, b = cv2.imread(str(exported)), cv2.imread(str(canvas))
    if b is None:
        raise FileNotFoundError(f"{name}: no native canvas at {canvas}")
    if a.shape != b.shape:
        raise ValueError(
            f"{name}: export is {a.shape[1]}x{a.shape[0]}, "
            f"canvas is {b.shape[1]}x{b.shape[0]}"
        )
    diff = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
    if diff > MAX_IMAGE_DIFF:
        raise ValueError(
            f"{name}: image differs from the canvas "
            f"(mean |delta| {diff:.1f}) — not the same leaf"
        )
    return diff


def unpack(source: Path, stack) -> Path:
    if source.is_dir():
        return source
    tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
    with zipfile.ZipFile(source) as zf:
        zf.extractall(tmp)
    return tmp


def archive(labels_dir: Path, out: Path) -> None:
    with tarfile.open(out, "w:gz") as tar:
        tar.add(labels_dir, arcname=labels_dir.name)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("export", type=Path,
                    help="corrected Roboflow export (.zip or unpacked directory)")
    ap.add_argument("--dataset", type=Path, default=Path("data/leaves-native"))
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    labels_dir = args.dataset / "labels"
    stale_dir = args.dataset / "labels-stale"

    import contextlib

    with contextlib.ExitStack() as stack:
        export = unpack(args.export, stack)
        corrected = export_labels(export)
        if not corrected:
            raise SystemExit(f"no labels found under {export}")

        current = {p.stem: p for p in sorted(labels_dir.glob("*.txt"))}
        print(f"export: {len(corrected)} leaves | dataset: {len(current)} leaves")

        rows, totals = [], dict(old=0, new=0, added=0, removed=0, emptied=0, new_leaf=0)
        for name, label_path in corrected.items():
            boxes = read_boxes(label_path)
            check_boxes(name, boxes)
            canvas = args.dataset / "img" / f"{name}.png"
            diff = verify_image(name, export_image(label_path), canvas)
            old = read_boxes(current[name]) if name in current else np.empty((0, 5))
            added, removed = count_edits(old, boxes)
            status = ("new-leaf" if name not in current
                      else "emptied" if len(boxes) == 0 else "updated")
            totals["old"] += len(old)
            totals["new"] += len(boxes)
            totals["added"] += added
            totals["removed"] += removed
            totals["emptied"] += status == "emptied"
            totals["new_leaf"] += status == "new-leaf"
            rows.append((name, len(old), len(boxes), added, removed, round(diff, 2), status))

        dropped = sorted(set(current) - set(corrected))
        for name in dropped:
            old = read_boxes(current[name])
            rows.append((name, len(old), 0, 0, len(old), "", "stale-dropped"))

        print(f"points {totals['old']} -> {totals['new']} "
              f"({100 * (totals['new'] - totals['old']) / max(totals['old'], 1):+.1f} %): "
              f"{totals['removed']} removed, {totals['added']} added")
        print(f"{totals['emptied']} leaves emptied, {totals['new_leaf']} new, "
              f"{len(dropped)} not in the export -> labels-stale/")
        if args.dry_run:
            print("dry run — nothing written")
            return

        backup = args.dataset.parent / f"{args.dataset.name}-labels-{date.today():%Y%m%d}.tar.gz"
        archive(labels_dir, backup)
        print(f"previous labels archived -> {backup}")

        stale_dir.mkdir(exist_ok=True)
        for name in dropped:
            shutil.move(str(current[name]), stale_dir / f"{name}.txt")
        for name, label_path in corrected.items():
            boxes = read_boxes(label_path)
            np.savetxt(labels_dir / f"{name}.txt", boxes,
                       fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"])

        report = args.dataset / "report-pycnidia-corrected.csv"
        with open(report, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["leaf", "points_before", "points_after", "added", "removed",
                        "image_mean_abs_diff", "status"])
            w.writerows(sorted(rows))
        print(f"{len(corrected)} labels written -> {labels_dir}")
        print(f"report -> {report}")


if __name__ == "__main__":
    main()
