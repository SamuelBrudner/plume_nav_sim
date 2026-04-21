import warnings

from .component_configs import ComponentEnvironmentConfig, EnvironmentConfig
from .composition import (
    SimulationSpec,
    _build_selector_env_kwargs_from_spec,
    build_env,
    create_simulation_spec,
)

__all__ = [
    "component_environment_config_to_spec",
    "component_environment_config_to_kwargs",
    "create_component_environment_from_config",
    "create_environment_from_config",
]


def component_environment_config_to_spec(
    config: ComponentEnvironmentConfig | EnvironmentConfig,
) -> SimulationSpec:
    """Translate the compatibility config model into the canonical SimulationSpec."""
    return create_simulation_spec(config)


def component_environment_config_to_kwargs(
    config: ComponentEnvironmentConfig | EnvironmentConfig,
) -> dict[str, object]:
    """Translate the compatibility config model into selector kwargs via SimulationSpec."""
    return _build_selector_env_kwargs_from_spec(component_environment_config_to_spec(config))


def create_component_environment_from_config(
    config: ComponentEnvironmentConfig | EnvironmentConfig,
) -> object:
    """Build a PlumeEnv from the compatibility config model."""
    return build_env(component_environment_config_to_spec(config))


def create_environment_from_config(
    config: ComponentEnvironmentConfig | EnvironmentConfig,
) -> object:
    warnings.warn(
        "create_environment_from_config is deprecated; use "
        "create_component_environment_from_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_component_environment_from_config(config)
