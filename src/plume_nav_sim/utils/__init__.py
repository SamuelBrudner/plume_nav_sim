"""Centralized utilities for plume_nav_sim.

Each submodule is imported eagerly to ensure required dependencies are
available. If a core utility submodule is missing, an ``ImportError`` is
raised immediately rather than providing silent fallbacks.
"""

from __future__ import annotations
from loguru import logger
from importlib import import_module
CORE_MODULES = [
    "frame_cache",
    "logging_setup",
    "seed_manager",
    "io",
]

__all__: list[str] = []

# Optional submodules loaded on demand to avoid circular imports.
_OPTIONAL_MODULES = [
    "visualization",
    "navigator_utils",
]


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


def __getattr__(name: str):
    """Lazily import optional utility submodules."""
    for module_name in _OPTIONAL_MODULES:
        module = import_module(f"plume_nav_sim.utils.{module_name}")
        public = getattr(module, "__all__", [])
        if name in public:
            globals().update({attr: getattr(module, attr) for attr in public})
            __all__.extend(attr for attr in public if attr not in __all__)
            return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
