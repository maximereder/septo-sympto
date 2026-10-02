"""Command-line wiring. No analysis logic lives here.

The CLI parses arguments, resolves the models by name or path through
:mod:`septosympto.zoo`, calls the pipeline, and writes the report. Everything it
orchestrates is importable and testable without it, which is what v1's
``septo_sympto.py`` — argument parsing and model loading at module import time —
made impossible.

    septo-sympto scans/                                   # default models, CSV next to you
    septo-sympto scans/ --necrosis unet-v1 -o v1.csv      # the 2023 model, for comparison
    septo-sympto scans/ --necrosis runs/x/weights/best.pt # an unpublished checkpoint
    septo-sympto --list-models
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from septosympto import __version__, zoo
from septosympto.leaf import load_scan
from septosympto.letterbox import CANVAS_H, CANVAS_W
from septosympto.pipeline import iter_analyses, iter_scan_files
from septosympto.render import save_analysis
from septosympto.report import build_manifest, write_csv, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="septo-sympto",
        description="Quantify Septoria tritici blotch symptoms on scanned wheat leaves.",
    )
    parser.add_argument("images", type=Path, nargs="?", help="Directory of scanned images.")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("results.csv"), help="Output CSV path."
    )
    parser.add_argument("-e", "--extension", default=".tif", help="Input image extension.")

    models = parser.add_argument_group("models (a published name or a checkpoint path)")
    models.add_argument(
        "-nm", "--necrosis", "--necrosis-weights", dest="necrosis", default=zoo.DEFAULT_NECROSIS,
        help=f"Necrosis segmenter. Default: {zoo.DEFAULT_NECROSIS}. See --list-models.",
    )
    models.add_argument(
        "--necrosis-arch", default=None,
        help="Architecture of a .safetensors necrosis checkpoint given by path.",
    )
    models.add_argument(
        "-pn", "--necrosis-threshold", type=float, default=None,
        help="Necrosis probability threshold. Default: the one the model was validated at.",
    )
    models.add_argument(
        "--pycnidia", default="none",
        help="Pycnidia counter, or 'none' to leave the pycnidia columns empty (default).",
    )
    models.add_argument(
        "--pycnidia-arch", default=None,
        help="Architecture of a .safetensors pycnidia checkpoint given by path.",
    )
    models.add_argument(
        "--pycnidia-threshold", type=float, default=None,
        help="Pycnidia detection threshold. Default: the one the model was validated at.",
    )
    models.add_argument(
        "--list-models", action="store_true", help="List the published models and exit."
    )

    parser.add_argument(
        "-pc", "--pixels-for-cm", type=float, default=None,
        help="Override the scale. By default it is read from each scan's metadata.",
    )
    parser.add_argument(
        "--min-lesion-area-mm2", type=float, default=0.135,
        help="Ignore necrosis components smaller than this.",
    )
    parser.add_argument(
        "-d", "--device", default="cpu", help="Torch device: cpu, mps, or a CUDA index."
    )
    parser.add_argument(
        "--masks-dir", type=Path, default=None,
        help="Also write, per leaf, an overlay (necrosis outlined, pycnidia circled) "
             "and the necrosis mask to this directory.",
    )
    parser.add_argument("--version", action="version", version=f"septo-sympto {__version__}")
    return parser


def _pick_threshold(explicit: float | None, resolved: zoo.Resolved, flag: str) -> float:
    if explicit is not None:
        return explicit
    if resolved.threshold is None:
        raise ValueError(f"{resolved.name} is not a published model; pass {flag}")
    return resolved.threshold


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        print(zoo.describe())
        return 0
    if args.images is None:
        parser.error("the following arguments are required: images")
    if not args.images.is_dir():
        print(f"error: {args.images} is not a directory", file=sys.stderr)
        return 2

    scan_files = iter_scan_files(args.images, args.extension)
    if not scan_files:
        print(f"error: no {args.extension} files in {args.images}", file=sys.stderr)
        return 1

    try:
        necrosis = zoo.resolve(args.necrosis, "necrosis", arch=args.necrosis_arch)
        necrosis_threshold = _pick_threshold(
            args.necrosis_threshold, necrosis, "--necrosis-threshold"
        )
        segmenter = zoo.load_segmenter(necrosis, necrosis_threshold, args.device)

        counter, pycnidia, pycnidia_threshold = None, None, None
        if args.pycnidia != "none":
            pycnidia = zoo.resolve(args.pycnidia, "pycnidia", arch=args.pycnidia_arch)
            pycnidia_threshold = _pick_threshold(
                args.pycnidia_threshold, pycnidia, "--pycnidia-threshold"
            )
            counter = zoo.load_counter(pycnidia, pycnidia_threshold, args.device)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"necrosis: {necrosis.name} (threshold {necrosis_threshold})")
    print(f"pycnidia: {pycnidia.name if pycnidia else 'none'}"
          + (f" (threshold {pycnidia_threshold})" if pycnidia else ""))

    measurements = []
    for path in scan_files:
        scan = load_scan(path)
        analyses = list(
            iter_analyses(
                scan, segmenter, counter,
                px_per_cm=args.pixels_for_cm,
                min_lesion_area_mm2=args.min_lesion_area_mm2,
            )
        )
        for analysis in analyses:
            measurements.append(analysis.measurement)
            if args.masks_dir is not None:
                save_analysis(analysis, args.masks_dir)
        print(f"{path.name}: {len(analyses)} leaves")

    n_rows = write_csv(measurements, args.output)

    weights = {"necrosis": necrosis.path}
    if pycnidia is not None:
        weights["pycnidia"] = pycnidia.path
    manifest = build_manifest(
        weights=weights,
        parameters={
            "necrosis_model": necrosis.name,
            "necrosis_threshold": necrosis_threshold,
            "pycnidia_model": pycnidia.name if pycnidia else None,
            "pycnidia_threshold": pycnidia_threshold,
            "imgsz": [CANVAS_H, CANVAS_W],
            "pixels_for_cm": args.pixels_for_cm,
            "min_lesion_area_mm2": args.min_lesion_area_mm2,
        },
        n_scans=len(scan_files),
        n_leaves=n_rows,
        timestamp=datetime.now(UTC).isoformat(),
    )
    write_manifest(manifest, args.output.with_suffix(".manifest.json"))

    print(f"\n{n_rows} leaves from {len(scan_files)} scans -> {args.output}")
    if args.masks_dir is not None:
        print(f"overlays and masks -> {args.masks_dir}")
    if counter is None:
        print("pycnidia not counted (--pycnidia none); those columns are empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
