"""Default configuration helpers for plume_nav_sim.config.

Provides simple factories that return validated default Pydantic
configuration objects for common use.
"""

from .component_configs import EnvironmentConfig

__all__ = [
    "get_complete_default_config",
    "get_default_environment_config",
]


def get_complete_default_config() -> EnvironmentConfig:
    """Return a default environment configuration.

    This mirrors the legacy helper name while returning the new
    Pydantic-based EnvironmentConfig used throughout the codebase.
    """

    return EnvironmentConfig()


def get_default_environment_config() -> EnvironmentConfig:
    """Alias for get_complete_default_config for clarity."""
    return get_complete_default_config()
