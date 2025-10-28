"""
Factory functions for creating component-based environments.

This module provides convenience functions for assembling environments
from components without manually wiring dependencies.

Example:
    >>> from plume_nav_sim.envs.factory import create_component_environment
    >>>
    >>> env = create_component_environment(
    ...     grid_size=(128, 128),
    ...     goal_location=(64, 64),
    ...     action_type='discrete',
    ...     observation_type='concentration',
    ...     reward_type='sparse'
    ... )
    >>>
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from typing import Literal, Optional, Union

import numpy as np

from ..actions import DiscreteGridActions, OrientedGridActions
from ..actions.oriented_run_tumble import OrientedRunTumbleActions
from ..core.geometry import Coordinates, GridSize
from ..observations import AntennaeArraySensor, ConcentrationSensor
from ..plume.concentration_field import ConcentrationField
from ..rewards import SparseGoalReward, StepPenaltyReward
from .component_env import ComponentBasedEnvironment

__all__ = ["create_component_environment"]


def create_component_environment(  # noqa: C901
    *,
    grid_size: Union[tuple[int, int], GridSize] = (128, 128),
    goal_location: Union[tuple[int, int], Coordinates] = (64, 64),
    start_location: Optional[Union[tuple[int, int], Coordinates]] = None,
    max_steps: int = 1000,
    goal_radius: float = 5.0,
    action_type: Literal["discrete", "oriented"] = "discrete",
    observation_type: Literal["concentration", "antennae"] = "concentration",
    reward_type: Literal["sparse", "step_penalty"] = "sparse",
    plume_sigma: float = 20.0,
    step_size: int = 1,
    render_mode: Optional[str] = None,
) -> ComponentBasedEnvironment:
    """
    Create a fully-configured component-based environment.

    This factory function assembles an environment from sensible defaults,
    allowing you to quickly create environments for experiments without
    manual component wiring.

    Args:
        grid_size: Environment dimensions (width, height)
        goal_location: Target position (x, y)
        start_location: Initial agent position (default: grid center)
        max_steps: Episode step limit
        goal_radius: Success threshold distance
        action_type: Action processor type ('discrete' or 'oriented')
        observation_type: Observation model type ('concentration' or 'antennae')
        reward_type: Reward function type ('sparse' or 'step_penalty')
        plume_sigma: Gaussian plume dispersion parameter
        step_size: Movement step size in grid cells
        render_mode: Rendering mode ('rgb_array' or 'human')

    Returns:
        Configured ComponentBasedEnvironment ready to use

    Example:
        >>> # Simple sparse goal environment with discrete actions
        >>> env = create_component_environment(
        ...     grid_size=(64, 64),
        ...     goal_location=(50, 50),
        ...     action_type='discrete',
        ...     reward_type='sparse'
        ... )
        >>>
        >>> # Oriented navigation with dense reward
        >>> env = create_component_environment(
        ...     action_type='oriented',
        ...     reward_type='dense',
        ...     observation_type='antennae'
        ... )

    Raises:
        ValueError: If invalid type specified for any component
    """
    # Convert tuples to proper types
    if isinstance(grid_size, tuple):
        grid_size = GridSize(width=grid_size[0], height=grid_size[1])

    if isinstance(goal_location, tuple):
        goal_location = Coordinates(x=goal_location[0], y=goal_location[1])

    if start_location and isinstance(start_location, tuple):
        start_location = Coordinates(x=start_location[0], y=start_location[1])

    if goal_radius < 0:
        raise ValueError("goal_radius must be non-negative")
    if goal_radius == 0:
        goal_radius = float(np.finfo(np.float32).eps)

    # Create action processor
    if action_type == "discrete":
        action_processor = DiscreteGridActions(step_size=step_size)
    elif action_type == "oriented":
        action_processor = OrientedGridActions(step_size=step_size)
    elif action_type == "run_tumble":
        action_processor = OrientedRunTumbleActions(step_size=step_size)
    else:
        raise ValueError(
            f"Invalid action_type: {action_type}. Must be 'discrete' or 'oriented'."
        )

    # Create observation model
    if observation_type == "concentration":
        observation_model = ConcentrationSensor()
    elif observation_type == "antennae":
        observation_model = AntennaeArraySensor(n_sensors=2, sensor_distance=2.0)
    else:
        raise ValueError(
            f"Invalid observation_type: {observation_type}. "
            f"Must be 'concentration' or 'antennae'."
        )

    # Create reward function
    if reward_type == "sparse":
        reward_function = SparseGoalReward(
            goal_position=goal_location, goal_radius=goal_radius
        )
    elif reward_type == "step_penalty":
        reward_function = StepPenaltyReward(
            goal_position=goal_location,
            goal_radius=goal_radius,
            goal_reward=1.0,
            step_penalty=0.01,
        )
    else:
        raise ValueError(
            f"Invalid reward_type: {reward_type}. Must be 'sparse' or 'step_penalty'."
        )

    # Create concentration field
    concentration_field = ConcentrationField(grid_size=grid_size, enable_caching=True)
    # Manually create Gaussian field (generate_field has signature issues)

    x = np.arange(grid_size.width)
    y = np.arange(grid_size.height)
    xx, yy = np.meshgrid(x, y)
    dx = xx - goal_location.x
    dy = yy - goal_location.y
    field_array = np.exp(-(dx**2 + dy**2) / (2 * plume_sigma**2))
    concentration_field.field_array = field_array.astype(np.float32)
    concentration_field.is_generated = True

    # Assemble environment
    return ComponentBasedEnvironment(
        action_processor=action_processor,
        observation_model=observation_model,
        reward_function=reward_function,
        concentration_field=concentration_field,
        grid_size=grid_size,
        max_steps=max_steps,
        goal_location=goal_location,
        goal_radius=goal_radius,
        start_location=start_location,
        render_mode=render_mode,
    )
