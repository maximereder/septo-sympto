"""Regenerate a dataset from the native TIFF crops, undistorted.

The shipped datasets stretch every leaf to a fixed strip (2048x200 for pycnidia,
3072x304 for necrosis), which destroys the aspect ratio and, for pycnidia, halves
the objects through a second JPEG pass. The native crops in ``LM1__allcrop`` are
1200 dpi TIFFs, so the resolution is still there — we just re-cut them.

For each annotated leaf we:

1. find the leaf on its native scan (``find_leaves``);
2. letterbox it 1:1 onto a fixed 3072x384 canvas (native scale, no stretch, white
   padding, interior mask holes filled so specular glare is not punched out);
3. carry the annotation into the new frame with the *same* transform — points are
   remapped (pycnidia), the mask is warped (necrosis).

The catch is matching: the shipped leaf index ``__N`` is NOT ``find_leaves``'s
top-to-bottom order, and some strips are flipped. So we never trust the name — we
try every (native leaf x orientation) and keep the best under a self-validating
oracle:

* pycnidia — the fraction of annotation points landing on a dark pycnidium
  (~99 % correct, ~70 % wrong);
* necrosis — the image cross-correlation between the strip and the native leaf
  (~0.98 correct, <0.8 wrong).

Anything below the threshold, or without a clear margin over the runner-up, is
flagged rather than written, so a bad match can never silently corrupt the set.

Augmentation is deliberately NOT reproduced: each real leaf is emitted once in
native orientation, and the training loop re-augments on the fly.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import os
import re
import zipfile
from collections import defaultdict
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from septosympto.leaf import Scan, find_leaves
from septosympto.letterbox import (
    BACKGROUND,
    CANVAS_H,
    CANVAS_W,
    letterbox,
)

MIN_SCORE = 0.85
CONTRAST = 8
MIN_NCC = 0.80
MASK_THRESHOLD = 128
MIN_MARGIN = 0.10


def base_name(path: str) -> str:
    """Real leaf id: drop Roboflow's ``_jpg.rf.<hash>`` suffix (pycnidia labels) or
    a plain ``.jpg``/``.png`` extension (necrosis strips)."""
    name = os.path.basename(path).split("_jpg.rf")[0]
    return re.sub(r"\.(jpg|png)$", "", name)


def scan_of(base: str) -> str:
    """``Acc_Acc_2_Acc_2__2__1`` -> ``Acc_Acc_2_Acc_2`` (drop the leaf/sub tail)."""
    return re.sub(r"(__\d+){1,2}$", "", base)


def orient(pts: np.ndarray, fx: int, fy: int) -> np.ndarray:
    """Flip normalised (x, y) points horizontally and/or vertically."""
    out = pts.copy()
    if fx:
        out[:, 0] = 1.0 - out[:, 0]
    if fy:
        out[:, 1] = 1.0 - out[:, 1]
    return out


def tiff_resolver(tiff_dir: str):
    """Map a scan name to its TIFF, tolerating Roboflow's truncated tokens
    (``Soi_LGA`` -> ``Soi_LG``)."""
    stems = {p.stem: p for p in Path(tiff_dir).glob("*.tif")}

    def resolve(scan_name: str):
        if scan_name in stems:
            return stems[scan_name]
        close = difflib.get_close_matches(scan_name, stems, n=1, cutoff=0.9)
        return stems[close[0]] if close else None

    return resolve


def load_leaves(bgr: np.ndarray, scan_name: str):
    """Return per-leaf ``(leaf_index, crop_bgr, solid_mask, (w, h))`` on a native scan.

    ``crop_bgr`` still has its raw pixels; ``solid_mask`` is the leaf silhouette
    with interior holes filled, used both to blank the background and to clip
    warped masks to the blade.
    """
    leaves = find_leaves(Scan(image=scan_name, bgr=bgr, px_per_cm=472.44))
    out = []
    for lf in leaves:
        x, y, w, h = lf.bbox
        solid = ndimage.binary_fill_holes(lf.mask[y : y + h, x : x + w])
        out.append((lf.leaf_index, bgr[y : y + h, x : x + w], solid, (w, h)))
    return out


def letterbox_leaf(crop: np.ndarray, solid: np.ndarray):
    """Blank the background outside the hole-filled silhouette, then letterbox.

    Filling the holes keeps specular glare and debris inside the blade as real
    pixels instead of white spots.
    """
    blanked = crop.copy()
    blanked[~solid] = BACKGROUND
    return letterbox(blanked)


def status_of(score: float, margin: float, floor: float) -> str:
    if score < floor:
        return "low-score"
    if margin < MIN_MARGIN:
        return "ambiguous"
    return "ok"


def write_report(out: Path, rows, extra_col: str, name: str):
    with open(out / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["leaf", "scan", "matched_leaf", "flip_xy",
                    "score", "margin", extra_col, "status"])
        w.writerows(rows)
    flags = [r for r in rows if r[-1] != "ok"]
    print(f"flagged {len(flags)}")
    for r in flags[:20]:
        print("  FLAG", r[0], r[-1], "score", r[4], "margin", r[5])
    print(f"report -> {out / name}")


def darkspot_hit(crop: np.ndarray, mask: np.ndarray, pts: np.ndarray) -> float:
    """Fraction of points that sit on a local dark spot inside the tissue."""
    if len(pts) == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    contrast = cv2.medianBlur(gray, 31).astype(np.float32) - gray.astype(np.float32)
    h, w = gray.shape
    xs = np.clip((pts[:, 0] * w).astype(int), 3, w - 4)
    ys = np.clip((pts[:, 1] * h).astype(int), 3, h - 4)
    hits = sum(
        mask[y, x] and contrast[y - 3 : y + 4, x - 3 : x + 4].max() > CONTRAST
        for x, y in zip(xs, ys, strict=True)
    )
    return hits / len(pts)


def match_points(boxes_by_copy, leaves):
    """Search (label copy x leaf x orientation); return best + margin over other leaves."""
    best, per_leaf = None, defaultdict(float)
    for ci, boxes in enumerate(boxes_by_copy):
        pts = boxes[:, 1:3]
        for fx in (0, 1):
            for fy in (0, 1):
                p = orient(pts, fx, fy)
                for li, (_, crop, solid, _) in enumerate(leaves):
                    s = darkspot_hit(crop, solid, p)
                    per_leaf[li] = max(per_leaf[li], s)
                    if best is None or s > best[0]:
                        best = (s, li, fx, fy, ci)
    other = max((v for k, v in per_leaf.items() if k != best[1]), default=0.0)
    return best, best[0] - other


def remap_points(boxes, fx, fy, p):
    """Roboflow-frame YOLO boxes -> canvas-frame YOLO boxes, aspect preserved."""
    cxy = orient(boxes[:, 1:3], fx, fy)
    ncx = (p.ox + cxy[:, 0] * p.w * p.scale) / CANVAS_W
    ncy = (p.oy + cxy[:, 1] * p.h * p.scale) / CANVAS_H
    nbw = boxes[:, 3] * p.w * p.scale / CANVAS_W
    nbh = boxes[:, 4] * p.h * p.scale / CANVAS_H
    return np.column_stack([boxes[:, 0:1], ncx, ncy, nbw, nbh])


def regen_pycnidia(args):
    copies = defaultdict(list)
    for f in glob(f"{args.src}/*/labels/*.txt"):
        copies[base_name(f)].append(f)
    by_scan = defaultdict(list)
    for b in copies:
        by_scan[scan_of(b)].append(b)
    print(f"{len(copies)} pycnidia leaves across {len(by_scan)} scans")

    resolve = tiff_resolver(args.tiffs)
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rows, written = [], 0

    for scan_name in sorted(by_scan):
        tif = resolve(scan_name)
        if tif is None:
            rows += [(b, scan_name, -1, 0, 0.0, 0.0, 0, "no-tiff") for b in by_scan[scan_name]]
            continue
        leaves = load_leaves(cv2.imread(str(tif), cv2.IMREAD_COLOR), scan_name)
        for b in sorted(by_scan[scan_name]):
            boxes = [np.loadtxt(f).reshape(-1, 5) for f in copies[b]]
            (score, li, fx, fy, ci), margin = match_points(boxes, leaves)
            st = status_of(score, margin, MIN_SCORE)
            if st == "ok":
                _, crop, solid, _ = leaves[li]
                canvas, place = letterbox_leaf(crop, solid)
                cv2.imwrite(str(out / "img" / f"{b}.png"), canvas)
                np.savetxt(out / "labels" / f"{b}.txt", remap_points(boxes[ci], fx, fy, place),
                           fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"])
                written += 1
            rows.append((b, scan_name, leaves[li][0], f"{fx}{fy}",
                         round(score, 3), round(margin, 3), len(boxes[0]), st))

    print(f"written {written}/{len(rows)} pycnidia leaves")
    write_report(out, rows, "n_points", "report-pycnidia.csv")


def load_necrosis(zip_path: str):
    """Every image/mask pair in a dataset zip, both splits pooled, one per leaf."""
    samples = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if "/img/" not in name or not name.endswith(".jpg") or "__MACOSX" in name:
                continue
            mask_name = name.replace("/img/", "/mask/").replace(".jpg", ".png")
            if mask_name not in z.namelist():
                continue
            b = base_name(name)
            if b in samples:
                continue
            img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            m = cv2.imdecode(np.frombuffer(z.read(mask_name), np.uint8), cv2.IMREAD_UNCHANGED)
            mask = (m.max(2) if m.ndim == 3 else m) >= MASK_THRESHOLD
            samples[b] = (img, mask)
    return samples


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def _native_gray(crop, solid, size):
    """Native leaf, background blanked, resized to the strip's dimensions."""
    blanked = np.where(solid[..., None], crop, BACKGROUND).astype(np.uint8)
    gray = cv2.cvtColor(blanked, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size).astype(np.float32)


def match_image(strip_gray, leaves):
    """Match a stretched strip to its native leaf by cross-correlation, over flips."""
    hs, ws = strip_gray.shape
    refs = [_native_gray(crop, solid, (ws, hs)) for _, crop, solid, _ in leaves]
    best, per_leaf = None, defaultdict(float)
    for fx in (0, 1):
        for fy in (0, 1):
            flipped = strip_gray[:: -1 if fy else 1, :: -1 if fx else 1]
            for pos, ref in enumerate(refs):
                s = _ncc(ref, flipped)
                per_leaf[pos] = max(per_leaf[pos], s)
                if best is None or s > best[0]:
                    best = (s, pos, fx, fy)
    other = max((v for k, v in per_leaf.items() if k != best[1]), default=0.0)
    return best, best[0] - other


def warp_mask(strip_mask, fx, fy, solid, p):
    """Strip-frame binary mask -> canvas-frame mask, same letterbox as the image."""
    m = strip_mask[:: -1 if fy else 1, :: -1 if fx else 1].astype(np.uint8)
    m = cv2.resize(m, (p.w, p.h), interpolation=cv2.INTER_NEAREST).astype(bool) & solid
    canvas = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    canvas[p.oy : p.oy + p.nh, p.ox : p.ox + p.nw] = cv2.resize(
        m.astype(np.uint8), (p.nw, p.nh), interpolation=cv2.INTER_NEAREST) * 255
    return canvas


def regen_necrosis(args):
    samples = load_necrosis(args.src)
    by_scan = defaultdict(list)
    for b in samples:
        by_scan[scan_of(b)].append(b)
    print(f"{len(samples)} necrosis leaves across {len(by_scan)} scans")

    resolve = tiff_resolver(args.tiffs)
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "mask").mkdir(parents=True, exist_ok=True)
    rows, written = [], 0

    for scan_name in sorted(by_scan):
        tif = resolve(scan_name)
        if tif is None:
            rows += [(b, scan_name, -1, 0, 0.0, 0.0, 0.0, "no-tiff") for b in by_scan[scan_name]]
            continue
        leaves = load_leaves(cv2.imread(str(tif), cv2.IMREAD_COLOR), scan_name)
        for b in sorted(by_scan[scan_name]):
            img, mask = samples[b]
            (score, li, fx, fy), margin = match_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                                       .astype(np.float32), leaves)
            st = status_of(score, margin, MIN_NCC)
            if st == "ok":
                _, crop, solid, _ = leaves[li]
                canvas, place = letterbox_leaf(crop, solid)
                cv2.imwrite(str(out / "img" / f"{b}.png"), canvas)
                cv2.imwrite(str(out / "mask" / f"{b}.png"), warp_mask(mask, fx, fy, solid, place))
                written += 1
            rows.append((b, scan_name, leaves[li][0], f"{fx}{fy}",
                         round(score, 3), round(margin, 3), round(float(mask.mean()), 3), st))

    print(f"written {written}/{len(rows)} necrosis leaves")
    write_report(out, rows, "necrosis_frac", "report-necrosis.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["pycnidia", "necrosis", "both"], default="both")
    ap.add_argument("--pyc-src", default="data/pycnidia",
                    help="Roboflow splits dir holding the pycnidia point labels.")
    ap.add_argument("--nec-src", default="data/necrosis/dataset/300.zip",
                    help="Necrosis dataset zip (img/mask pairs).")
    ap.add_argument("--tiffs", default=os.path.expanduser("~/Downloads/LM1__allcrop"))
    ap.add_argument("--out", default="data/leaves-native",
                    help="Shared dataset dir: img/ + labels/ (pycnidia) + mask/ (necrosis).")
    args = ap.parse_args()
    if args.task in ("pycnidia", "both"):
        regen_pycnidia(argparse.Namespace(src=args.pyc_src, tiffs=args.tiffs, out=args.out))
    if args.task in ("necrosis", "both"):
        regen_necrosis(argparse.Namespace(src=args.nec_src, tiffs=args.tiffs, out=args.out))


if __name__ == "__main__":
    main()
