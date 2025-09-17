"""Bridge module that exposes the configuration package within the plume_nav_sim namespace."""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Iterable

logger = logging.getLogger("plume_nav_sim.config.bridge")

_config = importlib.import_module("config")

EnvironmentConfig = _config.EnvironmentConfig
CompleteConfig = _config.CompleteConfig
PerformanceConfig = _config.PerformanceConfig
get_default_environment_config = _config.get_default_environment_config
get_complete_default_config = _config.get_complete_default_config

__all__ = list(getattr(_config, "__all__", []))

_LAZY_NAMES = {
    name
    for name in __all__
    if name not in {"EnvironmentConfig", "CompleteConfig", "PerformanceConfig", "get_default_environment_config", "get_complete_default_config"}
}


def __getattr__(name: str):
    if name in _LAZY_NAMES:
        value = getattr(_config, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _expose_submodules(names: Iterable[str]) -> None:
    for name in names:
        try:
            module = importlib.import_module(f"config.{name}")
        except ImportError as exc:
            logger.debug("Skipping optional config submodule %s: %s", name, exc)
            continue
        globals()[name] = module
        sys.modules[f"{__name__}.{name}"] = module
        logger.debug("Registered plume_nav_sim.config.%s", name)


_expose_submodules(["default_config"])
