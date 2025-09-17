"""Minimal configuration utilities required by the utils package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from plume_nav_sim.core.types import EnvironmentConfig, create_environment_config


def create_quick_config(**overrides: Any) -> EnvironmentConfig:
    """Create a quick EnvironmentConfig using canonical factory helpers."""
    return create_environment_config(overrides)


def validate_config(config: EnvironmentConfig) -> bool:
    """Basic validation shim that leverages the EnvironmentConfig contract."""
    return isinstance(config, EnvironmentConfig) and config.validate()


@dataclass
class ConfigManager:
    """Placeholder configuration manager that stores the active configuration."""

    current_config: EnvironmentConfig

    def update(self, **overrides: Any) -> EnvironmentConfig:
        self.current_config = create_environment_config(
            self.current_config.clone_with_overrides(**overrides)
        )
        return self.current_config

    def to_dict(self) -> Dict[str, Any]:
        return self.current_config.to_dict()
