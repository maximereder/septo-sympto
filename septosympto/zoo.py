"""Published models by name, so the CLI takes ``--necrosis yolo26s-v2``, not a path.

A model is more than a weights file: it has an architecture, a threshold it was
validated at, a task, and a score on the held-out test fold. Asking the user for
those separately is how the wrong threshold ends up on the right weights. Each
:class:`ModelCard` carries the lot, and the CLI resolves a name to a verified
local file — downloaded once into the cache from the project's GitHub release,
checked against its SHA-256 every time it is loaded.

A path is accepted wherever a name is, for a checkpoint that is not published
yet. Its kind is read from the suffix (``.pt`` is Ultralytics, ``.safetensors``
is a project architecture, which then has to be named).

The catalogue is code, not a config file: it changes when a model is promoted
from ``LEADERBOARD.md``, which is a reviewed change, and it needs no parser.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

RELEASE_URL = "https://github.com/maximereder/septo-sympto/releases/download/models-2026.09"


@dataclass(frozen=True)
class ModelCard:
    name: str
    task: str  # "necrosis" | "pycnidia"
    kind: str  # "torch" | "yolo"
    arch: str | None  # registry name of a torch architecture; None for yolo
    threshold: float  # the threshold it was validated at
    file: str
    sha256: str
    note: str  # score on the held-out test fold, one line
    experimental: bool = False

    @property
    def url(self) -> str:
        return f"{RELEASE_URL}/{self.file}"


NECROSIS: dict[str, ModelCard] = {
    "yolo26s-v2": ModelCard(
        name="yolo26s-v2", task="necrosis", kind="yolo", arch=None, threshold=0.3,
        file="necrosis-yolo26s-v2.pt",
        sha256="e0b26d29de00d1d6afd69a6ebe794f95382258da47ffb51274e8c29f681127e9",
        note="YOLO26s-sem on the 278-leaf native set; test Dice 0.694, area ratio 1.02",
    ),
    "unet-v1": ModelCard(
        name="unet-v1", task="necrosis", kind="torch", arch="unet", threshold=0.8,
        file="necrosis-unet-v1.safetensors",
        sha256="d71c2e87091568936744e51bdd0c0aca3d965bc54912e3de1823d19a12e96a68",
        note="the published 2023 U-Net (Keras port); test Dice 0.611, area ratio 0.77",
    ),
}

PYCNIDIA: dict[str, ModelCard] = {
    "p2p-convnext-v2": ModelCard(
        name="p2p-convnext-v2", task="pycnidia", kind="torch", arch="p2p-convnext-t",
        threshold=0.3, file="pycnidia-p2p-convnext-v2.safetensors",
        sha256="e80d45a9a078f3a881b704b7fedcb4640f0da8e5d2f43ffbf86c7d62b68a23a8",
        note="P2PNet/ConvNeXt-T on the native set; val MAE 64.7, bias +41 % — preview only",
        experimental=True,
    ),
}

DEFAULT_NECROSIS = "yolo26s-v2"
CATALOGUE = {"necrosis": NECROSIS, "pycnidia": PYCNIDIA}


def cache_dir() -> Path:
    """``$SEPTOSYMPTO_HOME`` or ``~/.cache/septosympto``."""
    return Path(os.environ.get("SEPTOSYMPTO_HOME") or Path.home() / ".cache" / "septosympto")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(card: ModelCard, *, quiet: bool = False) -> Path:
    """The verified local file for ``card``, downloading it into the cache if needed.

    The download lands in a temporary name and is renamed only after its digest
    matches, so an interrupted transfer never leaves a plausible-looking file.
    """
    target = cache_dir() / "models" / card.file
    if target.exists():
        if _sha256(target) == card.sha256:
            return target
        target.unlink()  # corrupt or superseded: fetch again
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if not quiet:
        print(f"downloading {card.name} ({card.file}) -> {target}", file=sys.stderr)
    try:
        urllib.request.urlretrieve(card.url, partial)
    except OSError as error:
        raise RuntimeError(f"could not download {card.name} from {card.url}: {error}") from None
    digest = _sha256(partial)
    if digest != card.sha256:
        partial.unlink()
        raise RuntimeError(
            f"{card.name}: downloaded file has sha256 {digest[:16]}…, "
            f"expected {card.sha256[:16]}…; refusing to use it"
        )
    partial.replace(target)
    return target


@dataclass(frozen=True)
class Resolved:
    """What a ``--necrosis`` / ``--pycnidia`` argument resolves to."""

    name: str  # the card name, or the path as given
    path: Path
    kind: str
    arch: str | None
    threshold: float | None  # None when a path was given and no card knows better


def resolve(spec: str, task: str, *, arch: str | None = None) -> Resolved:
    """A published name or a local path -> a verified file and how to load it."""
    table = CATALOGUE[task]
    if spec in table:
        card = table[spec]
        return Resolved(card.name, fetch(card), card.kind, card.arch, card.threshold)
    path = Path(spec).expanduser()
    if path.exists():
        if path.suffix == ".pt":
            return Resolved(str(path), path, "yolo", None, None)
        if path.suffix == ".safetensors":
            if arch is None:
                raise ValueError(
                    f"{path} is a project checkpoint; name its architecture with --{task}-arch"
                )
            return Resolved(str(path), path, "torch", arch, None)
        raise ValueError(f"{path}: expected a .pt (Ultralytics) or .safetensors checkpoint")
    raise ValueError(
        f"unknown {task} model {spec!r}: not a published name ({', '.join(table)}) "
        "and not an existing file"
    )


def load_segmenter(resolved: Resolved, threshold: float, device: str):
    if resolved.kind == "yolo":
        try:
            from septosympto.adapters import YoloSegmenter
            return YoloSegmenter.from_weights(resolved.path, threshold=threshold, device=device)
        except ModuleNotFoundError as error:
            if error.name and error.name.startswith("ultralytics"):
                raise RuntimeError(
                    f"{resolved.name} is an Ultralytics model; install the extra: "
                    "poetry install --extras yolo"
                ) from None
            raise
    from septosympto.adapters import TorchSegmenter
    from septosympto.models import build_segmenter

    return TorchSegmenter.from_safetensors(
        resolved.path, build_segmenter(resolved.arch), threshold=threshold, device=device
    )


def load_counter(resolved: Resolved, threshold: float, device: str):
    if resolved.kind != "torch":
        raise ValueError(f"{resolved.name}: pycnidia counters are project architectures")
    from septosympto.adapters import TorchCounter
    from septosympto.models import build_counter

    return TorchCounter.from_safetensors(
        resolved.path, build_counter(resolved.arch), threshold=threshold, device=device
    )


def describe() -> str:
    """The catalogue as a table, for ``--list-models``."""
    lines = []
    for task, table in CATALOGUE.items():
        lines.append(f"{task}:")
        for card in table.values():
            default = "  (default)" if card.name == DEFAULT_NECROSIS else ""
            flag = "  [experimental]" if card.experimental else ""
            cached = "cached" if (cache_dir() / "models" / card.file).exists() else "not downloaded"
            lines.append(
                f"  {card.name:18s} thr {card.threshold:<4}  {cached:14s} "
                f"{card.note}{default}{flag}"
            )
    lines.append(f"\ncache: {cache_dir() / 'models'}  (override with SEPTOSYMPTO_HOME)")
    return "\n".join(lines)
