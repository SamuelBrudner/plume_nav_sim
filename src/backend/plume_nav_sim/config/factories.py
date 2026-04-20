import warnings
from typing import Any

from ..actions import DiscreteGridActions, OrientedGridActions, OrientedRunTumbleActions
from ..core.geometry import Coordinates, GridSize
from ..envs.factory import _create_plume_env_from_selectors
from ..observations import AntennaeArraySensor, ConcentrationSensor, WindVectorSensor
from ..plume.gaussian import GaussianPlume
from ..rewards import SparseGoalReward, StepPenaltyReward
from ..wind_field import ConstantWindField
from .component_configs import (
    ActionConfig,
    ComponentEnvironmentConfig,
    EnvironmentConfig,
    ObservationConfig,
    PlumeConfig,
    RewardConfig,
    WindConfig,
)

__all__ = [
    "create_action_processor",
    "create_observation_model",
    "create_reward_function",
    "create_concentration_field",
    "create_wind_field",
    "component_environment_config_to_kwargs",
    "create_component_environment_from_config",
    "create_environment_from_config",
]


def create_action_processor(config: ActionConfig):
    if config.type == "discrete":
        return DiscreteGridActions(step_size=config.step_size)
    elif config.type == "oriented":
        return OrientedGridActions(step_size=config.step_size)
    elif config.type == "run_tumble":
        return OrientedRunTumbleActions(step_size=config.step_size)
    else:
        raise ValueError(
            f"Invalid action type: {config.type}. Must be 'discrete', 'oriented', or 'run_tumble'."
        )


def create_observation_model(config: ObservationConfig):
    if config.type == "concentration":
        return ConcentrationSensor()
    elif config.type == "antennae":
        return AntennaeArraySensor(
            n_sensors=config.n_sensors,
            sensor_angles=config.sensor_angles,
            sensor_distance=config.sensor_distance,
        )
    elif config.type == "wind_vector":
        return WindVectorSensor(noise_std=float(config.noise_std))
    else:
        raise ValueError(
            f"Invalid observation type: {config.type}. "
            f"Must be 'concentration', 'antennae', or 'wind_vector'."
        )


def create_reward_function(config: RewardConfig, goal_location: Coordinates):
    if config.type == "sparse":
        return SparseGoalReward(
            goal_position=goal_location,
            goal_radius=config.goal_radius,
        )
    elif config.type == "step_penalty":
        return StepPenaltyReward(
            goal_position=goal_location,
            goal_radius=config.goal_radius,
            goal_reward=config.goal_reward,
            step_penalty=config.step_penalty,
        )
    else:
        raise ValueError(
            f"Invalid reward type: {config.type}. Must be 'sparse' or 'step_penalty'."
        )


def create_concentration_field(
    config: PlumeConfig, grid_size: GridSize, goal_location: Coordinates
):
    """Create concentration field from configuration."""
    return GaussianPlume(
        grid_size=grid_size,
        source_location=goal_location,
        sigma=config.sigma,
    )


def create_wind_field(config: WindConfig | None):
    if config is None:
        return None

    if config.type != "constant":
        raise ValueError(
            f"Invalid wind type: {config.type}. Only 'constant' is supported."
        )

    return ConstantWindField(
        direction_deg=config.direction_deg,
        speed=config.speed,
        vector=config.vector,
    )


def component_environment_config_to_kwargs(
    config: ComponentEnvironmentConfig | EnvironmentConfig,
) -> dict[str, Any]:
    """Normalize the component-style config model into explicit env kwargs."""
    kwargs: dict[str, Any] = {
        "grid_size": tuple(config.grid_size),
        "goal_location": tuple(config.goal_location),
        "max_steps": config.max_steps,
        "goal_radius": config.reward.goal_radius,
        "action_type": config.action.type,
        "step_size": config.action.step_size,
        "observation_type": config.observation.type,
        "reward_type": config.reward.type,
        "plume_sigma": config.plume.sigma,
        "render_mode": config.render_mode,
    }
    if config.start_location is not None:
        kwargs["start_location"] = tuple(config.start_location)
    if config.observation.type == "wind_vector":
        kwargs["wind_noise_std"] = float(config.observation.noise_std)
    if config.wind is not None:
        kwargs["enable_wind"] = True
        kwargs["wind_direction_deg"] = config.wind.direction_deg
        kwargs["wind_speed"] = config.wind.speed
        if config.wind.vector is not None:
            kwargs["wind_vector"] = tuple(config.wind.vector)
    return kwargs


def create_component_environment_from_config(
    config: ComponentEnvironmentConfig | EnvironmentConfig,
) -> Any:
    """Build a PlumeEnv from the compatibility config model."""
    return _create_plume_env_from_selectors(**component_environment_config_to_kwargs(config))


def create_environment_from_config(
    config: ComponentEnvironmentConfig | EnvironmentConfig,
) -> Any:
    warnings.warn(
        "create_environment_from_config is deprecated; use "
        "create_component_environment_from_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_component_environment_from_config(config)
