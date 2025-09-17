"""Canonical configuration entrypoint built on the shared core type system."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

from plume_nav_sim.core.types import EnvironmentConfig, create_environment_config
logger = logging.getLogger("plume_nav_sim.config")

_default_config = import_module(".default_config", __name__)
CompleteConfig = _default_config.CompleteConfig
PerformanceConfig = _default_config.PerformanceConfig
get_complete_default_config = _default_config.get_complete_default_config


def _environment_configs():
    return import_module(".environment_configs", __name__)


_LAZY_EXPORTS = {
    "ConfigurationRegistry",
    "ENVIRONMENT_REGISTRY",
    "PresetMetadata",
    "create_preset_config",
    "create_research_scenario",
    "create_benchmark_config",
    "create_custom_scenario",
    "get_available_presets",
    "validate_preset_name",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module = _environment_configs()
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_default_environment_config(**overrides: Any) -> EnvironmentConfig:
    """Return a validated EnvironmentConfig using canonical factories."""
    if overrides:
        logger.debug("Creating EnvironmentConfig with overrides: %s", overrides)
        return create_environment_config(overrides)
    logger.debug("Creating default EnvironmentConfig without overrides")
    return EnvironmentConfig()


__all__ = [
    "EnvironmentConfig",
    "CompleteConfig",
    "PerformanceConfig",
    "ConfigurationRegistry",
    "ENVIRONMENT_REGISTRY",
    "PresetMetadata",
    "create_preset_config",
    "create_research_scenario",
    "create_benchmark_config",
    "create_custom_scenario",
    "get_available_presets",
    "validate_preset_name",
    "get_default_environment_config",
    "get_complete_default_config",
]
