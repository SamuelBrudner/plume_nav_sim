from .component_configs import EnvironmentConfig

__all__ = [
    "get_complete_default_config",
    "get_default_environment_config",
]


def get_complete_default_config() -> EnvironmentConfig:
    return EnvironmentConfig()


def get_default_environment_config() -> EnvironmentConfig:
    """Alias for get_complete_default_config for clarity."""
    return get_complete_default_config()
