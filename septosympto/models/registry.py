"""Name -> architecture, so a model can be built and trained by name.

Architectures live in this package because inference needs them too; the registry
lets both inference and :mod:`train` resolve one by a string, which is what makes
the training harness architecture-generic. Define a segmenter here, decorate it
with ``@register_segmenter("name")``, and ``train.run --arch name`` trains it with
no change to the loop.

Registration is per **task**. A segmenter maps an image to a ``(N, 1, H, W)``
logit map; that is the contract the training loop and :class:`TorchSegmenter`
depend on. A point counter is a different task — different output, loss, data and
metric — and gets its own registry rather than being forced through this one.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

_SEGMENTERS: dict[str, type[nn.Module]] = {}
_COUNTERS: dict[str, type[nn.Module]] = {}


def _register(table: dict, kind: str, name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        if name in table:
            raise ValueError(f"{kind} {name!r} is already registered")
        table[name] = cls
        return cls

    return decorator


def _build(table: dict, kind: str, name: str, **kwargs) -> nn.Module:
    if name not in table:
        raise KeyError(f"unknown {kind} {name!r}; available: {sorted(table)}")
    return table[name](**kwargs)


def register_segmenter(name: str) -> Callable[[type], type]:
    """Class decorator registering a segmentation architecture under ``name``.

    A segmenter maps ``(N, 3, H, W)`` to ``(N, 1, H, W)`` logits. One loss fits
    all of them, so the training loop owns the loss.
    """
    return _register(_SEGMENTERS, "segmenter", name)


def build_segmenter(name: str, **kwargs) -> nn.Module:
    return _build(_SEGMENTERS, "segmenter", name, **kwargs)


def segmenter_class(name: str) -> type[nn.Module]:
    if name not in _SEGMENTERS:
        raise KeyError(f"unknown segmenter {name!r}; available: {available_segmenters()}")
    return _SEGMENTERS[name]


def available_segmenters() -> list[str]:
    return sorted(_SEGMENTERS)


def register_counter(name: str) -> Callable[[type], type]:
    """Class decorator registering a point-counting architecture under ``name``.

    Counting has no single output format: a density map, a heatmap and a
    point-set network differ in output, target and loss. So a counter owns its
    own ``loss(output, target_points)`` and ``decode(output) -> points``, and the
    training loop stays identical across all of them. That is the contract a P2P
    network satisfies to drop in beside a heatmap counter.
    """
    return _register(_COUNTERS, "counter", name)


def build_counter(name: str, **kwargs) -> nn.Module:
    return _build(_COUNTERS, "counter", name, **kwargs)


def counter_class(name: str) -> type[nn.Module]:
    if name not in _COUNTERS:
        raise KeyError(f"unknown counter {name!r}; available: {available_counters()}")
    return _COUNTERS[name]


def available_counters() -> list[str]:
    return sorted(_COUNTERS)
