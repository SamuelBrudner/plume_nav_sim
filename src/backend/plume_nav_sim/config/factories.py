"""
Factory functions for creating components and environments from configs.

This module provides functions to instantiate actual component objects
from Pydantic configuration models, enabling config-driven environment creation.

Example:
    >>> from plume_nav_sim.config import EnvironmentConfig, create_environment_from_config
    >>>
    >>> config = EnvironmentConfig.parse_file("conf/experiment.yaml")
    >>> env = create_environment_from_config(config)
"""

import numpy as np

from ..actions import DiscreteGridActions, OrientedGridActions, OrientedRunTumbleActions
from ..core.geometry import Coordinates, GridSize
from ..envs import ComponentBasedEnvironment
from ..observations import AntennaeArraySensor, ConcentrationSensor, WindVectorSensor
from ..plume.concentration_field import ConcentrationField
from ..plume.wind_field import ConstantWindField
from ..rewards import SparseGoalReward, StepPenaltyReward
from .component_configs import (
    ActionConfig,
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
    "create_environment_from_config",
]


def create_action_processor(config: ActionConfig):
    """Create action processor from configuration.

    Args:
        config: ActionConfig with type and parameters

    Returns:
        ActionProcessor instance (DiscreteGridActions, OrientedGridActions, or OrientedRunTumbleActions)

    Raises:
        ValueError: If config.type is invalid

    Example:
        >>> config = ActionConfig(type="discrete", step_size=2)
        >>> processor = create_action_processor(config)
    """
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
    """Create observation model from configuration.

    Args:
        config: ObservationConfig with type and parameters

    Returns:
        ObservationModel instance (ConcentrationSensor, AntennaeArraySensor, or WindVectorSensor)

    Raises:
        ValueError: If config.type is invalid

    Example:
        >>> config = ObservationConfig(type="concentration")
        >>> model = create_observation_model(config)
    """
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
    """Create reward function from configuration.

    Args:
        config: RewardConfig with type and parameters
        goal_location: Goal position (required for reward calculation)

    Returns:
        RewardFunction instance (SparseGoalReward or StepPenaltyReward)

    Raises:
        ValueError: If config.type is invalid

    Example:
        >>> config = RewardConfig(type="sparse", goal_radius=5.0)
        >>> reward_fn = create_reward_function(config, Coordinates(64, 64))
    """
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
    """Create concentration field from configuration.

    Args:
        config: PlumeConfig with dispersion parameters
        grid_size: Grid dimensions
        goal_location: Plume source location

    Returns:
        ConcentrationField instance with generated field

    Example:
        >>> config = PlumeConfig(sigma=20.0, normalize=True)
        >>> field = create_concentration_field(
        ...     config,
        ...     GridSize(128, 128),
        ...     Coordinates(64, 64)
        ... )
    """
    field = ConcentrationField(
        grid_size=grid_size, enable_caching=config.enable_caching
    )

    # Manually create Gaussian field
    x = np.arange(grid_size.width)
    y = np.arange(grid_size.height)
    xx, yy = np.meshgrid(x, y)
    dx = xx - goal_location.x
    dy = yy - goal_location.y
    field_array = np.exp(-(dx**2 + dy**2) / (2 * config.sigma**2))

    if config.normalize:
        # Already normalized by Gaussian, but ensure [0, 1]
        field_array = np.clip(field_array, 0.0, 1.0)

    field.field_array = field_array.astype(np.float32)
    field.is_generated = True

    return field


def create_wind_field(config: WindConfig | None):
    """Create wind field from configuration.

    Args:
        config: Optional WindConfig. If None, wind is disabled.

    Returns:
        ConstantWindField instance or None if no wind configured.
    """

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


def create_environment_from_config(
    config: EnvironmentConfig,
) -> ComponentBasedEnvironment:
    """Create complete environment from configuration.

    This is the main entry point for config-driven environment creation.
    It instantiates all components and assembles them into an environment.

    Args:
        config: EnvironmentConfig with all component configurations

    Returns:
        Fully configured ComponentBasedEnvironment

    Example:
        >>> # From Python
        >>> config = EnvironmentConfig(
        ...     grid_size=(128, 128),
        ...     goal_location=(64, 64),
        ...     action=ActionConfig(type="discrete"),
        ...     observation=ObservationConfig(type="concentration"),
        ...     reward=RewardConfig(type="sparse")
        ... )
        >>> env = create_environment_from_config(config)
        >>>
        >>> # From YAML
        >>> config = EnvironmentConfig.parse_file("conf/experiment.yaml")
        >>> env = create_environment_from_config(config)
    """
    # Convert tuples to proper types
    grid_size = GridSize(width=config.grid_size[0], height=config.grid_size[1])
    goal_location = Coordinates(x=config.goal_location[0], y=config.goal_location[1])

    start_location = None
    if config.start_location:
        start_location = Coordinates(
            x=config.start_location[0], y=config.start_location[1]
        )

    # Create components from configs
    action_processor = create_action_processor(config.action)
    observation_model = create_observation_model(config.observation)
    reward_function = create_reward_function(config.reward, goal_location)
    concentration_field = create_concentration_field(
        config.plume, grid_size, goal_location
    )
    wind_field = create_wind_field(config.wind)

    # Assemble environment
    return ComponentBasedEnvironment(
        action_processor=action_processor,
        observation_model=observation_model,
        reward_function=reward_function,
        concentration_field=concentration_field,
        wind_field=wind_field,
        grid_size=grid_size,
        max_steps=config.max_steps,
        goal_location=goal_location,
        goal_radius=config.reward.goal_radius,
        start_location=start_location,
        render_mode=config.render_mode,
    )
