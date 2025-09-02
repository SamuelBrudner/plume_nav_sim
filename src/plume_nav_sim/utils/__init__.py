"""Centralized utilities for plume_nav_sim.

Each submodule is imported eagerly to ensure required dependencies are
available. If a core utility submodule is missing, an ``ImportError`` is
raised immediately rather than providing silent fallbacks.
"""

from __future__ import annotations

import logging
from importlib import import_module

logger = logging.getLogger(__name__)

CORE_MODULES = [
    "frame_cache",
    "logging_setup",
    "seed_manager",
    "visualization",
    "io",
    "navigator_utils",
]

__all__: list[str] = []


def _load_submodule(name: str) -> None:
    try:
        module = import_module(f"plume_nav_sim.utils.{name}")
    except ImportError as exc:  # pragma: no cover - handled by tests
        logger.error("Missing utils submodule %s: %s", name, exc)
        raise
    public = getattr(module, "__all__", [])
    globals().update({attr: getattr(module, attr) for attr in public})
    __all__.extend(public)


for _module in CORE_MODULES:
    _load_submodule(_module)
