"""Command-line wiring. No analysis logic lives here.

The CLI parses arguments, loads the models, calls the pipeline, and writes the
report. Everything it orchestrates is importable and testable without it, which
is what v1's ``septo_sympto.py`` — argument parsing and model loading at module
import time — made impossible.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from septosympto import __version__
from septosympto.pipeline import analyze_scan, iter_scan_files
from septosympto.report import build_manifest, write_csv, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="septo-sympto",
        description="Quantify Septoria tritici blotch symptoms on scanned wheat leaves.",
    )
    parser.add_argument("images", type=Path, help="Directory of scanned images.")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("results.csv"), help="Output CSV path."
    )
    parser.add_argument(
        "-nm", "--necrosis-weights", type=Path, required=True,
        help="Necrosis segmenter weights (.safetensors).",
    )
    parser.add_argument("-e", "--extension", default=".tif", help="Input image extension.")
    parser.add_argument(
        "-is", "--imgsz", type=int, nargs=2, default=[304, 3072], metavar=("H", "W"),
        help="Segmenter input size.",
    )
    parser.add_argument(
        "-pn", "--necrosis-threshold", type=float, default=0.8,
        help="Necrosis probability threshold.",
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
    parser.add_argument("--version", action="version", version=f"septo-sympto {__version__}")
    return parser


def _load_segmenter(weights: Path, imgsz: list[int], threshold: float, device: str):
    from septosympto.adapters import TorchSegmenter
    from septosympto.models import UNet

    return TorchSegmenter.from_safetensors(
        weights, UNet(), imgsz=(imgsz[0], imgsz[1]), threshold=threshold, device=device
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.images.is_dir():
        print(f"error: {args.images} is not a directory", file=sys.stderr)
        return 2

    scan_files = iter_scan_files(args.images, args.extension)
    if not scan_files:
        print(f"error: no {args.extension} files in {args.images}", file=sys.stderr)
        return 1

    segmenter = _load_segmenter(
        args.necrosis_weights, args.imgsz, args.necrosis_threshold, args.device
    )

    measurements = []
    for path in scan_files:
        from septosympto.leaf import load_scan

        scan = load_scan(path)
        measurements.extend(
            analyze_scan(
                scan, segmenter, None,
                px_per_cm=args.pixels_for_cm,
                min_lesion_area_mm2=args.min_lesion_area_mm2,
            )
        )
        print(f"{path.name}: {sum(m.image == scan.image for m in measurements)} leaves")

    n_rows = write_csv(measurements, args.output)

    manifest = build_manifest(
        weights={"necrosis": args.necrosis_weights},
        parameters={
            "necrosis_threshold": args.necrosis_threshold,
            "imgsz": args.imgsz,
            "pixels_for_cm": args.pixels_for_cm,
            "min_lesion_area_mm2": args.min_lesion_area_mm2,
        },
        n_scans=len(scan_files),
        n_leaves=n_rows,
        timestamp=datetime.now(UTC).isoformat(),
    )
    write_manifest(manifest, args.output.with_suffix(".manifest.json"))

    print(f"\n{n_rows} leaves from {len(scan_files)} scans -> {args.output}")
    print("pycnidia counting is not wired yet; those columns are zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
