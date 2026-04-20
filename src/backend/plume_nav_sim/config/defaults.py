from ..envs.config_types import EnvironmentConfig
from .component_configs import ComponentEnvironmentConfig

__all__ = [
    "get_complete_default_config",
    "get_default_environment_config",
    "get_default_component_environment_config",
]


def get_complete_default_config() -> EnvironmentConfig:
    return EnvironmentConfig()


def get_default_environment_config() -> EnvironmentConfig:
    """Alias for get_complete_default_config for clarity."""
    return get_complete_default_config()


def get_default_component_environment_config() -> ComponentEnvironmentConfig:
    """Return the compatibility component-config defaults explicitly."""
    return ComponentEnvironmentConfig()
