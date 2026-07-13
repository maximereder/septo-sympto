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


def register_segmenter(name: str) -> Callable[[type], type]:
    """Class decorator that registers a segmentation architecture under ``name``."""

    def decorator(cls: type) -> type:
        if name in _SEGMENTERS:
            raise ValueError(f"segmenter {name!r} is already registered")
        _SEGMENTERS[name] = cls
        return cls

    return decorator


def build_segmenter(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered segmenter."""
    if name not in _SEGMENTERS:
        raise KeyError(f"unknown segmenter {name!r}; available: {available_segmenters()}")
    return _SEGMENTERS[name](**kwargs)


def segmenter_class(name: str) -> type[nn.Module]:
    if name not in _SEGMENTERS:
        raise KeyError(f"unknown segmenter {name!r}; available: {available_segmenters()}")
    return _SEGMENTERS[name]


def available_segmenters() -> list[str]:
    return sorted(_SEGMENTERS)
