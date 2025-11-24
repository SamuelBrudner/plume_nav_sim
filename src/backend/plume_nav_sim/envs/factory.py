"""Factory functions for creating component-based environments.

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

from pathlib import Path
from typing import Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np

from ..actions import DiscreteGridActions, OrientedGridActions
from ..actions.oriented_run_tumble import OrientedRunTumbleActions
from ..core.geometry import Coordinates, GridSize
from ..data_zoo import DEFAULT_CACHE_ROOT, ensure_dataset_available
from ..observations import AntennaeArraySensor, ConcentrationSensor, WindVectorSensor
from ..plume.concentration_field import ConcentrationField
from ..plume.movie_field import MovieConfig, MoviePlumeField, resolve_movie_dataset_path
from ..plume.wind_field import ConstantWindField
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
    action_type: Literal["discrete", "oriented", "run_tumble"] = "discrete",
    observation_type: Literal[
        "concentration", "antennae", "wind_vector"
    ] = "concentration",
    reward_type: Literal["sparse", "step_penalty"] = "sparse",
    plume_sigma: float = 20.0,
    step_size: int = 1,
    render_mode: Optional[str] = None,
    # New: select plume source and optional movie configuration
    plume: Literal["static", "movie"] = "static",
    movie_path: Optional[str] = None,
    movie_dataset_id: Optional[str] = None,
    movie_auto_download: bool = False,
    movie_cache_root: Optional[Union[str, Path]] = None,
    movie_fps: Optional[float] = None,
    movie_pixel_to_grid: Optional[tuple[float, float]] = None,
    movie_origin: Optional[tuple[float, float]] = None,
    movie_extent: Optional[tuple[float, float]] = None,
    movie_h5_dataset: Optional[str] = None,
    movie_step_policy: Literal["wrap", "clamp"] = "wrap",
    enable_wind: bool = False,
    wind_direction_deg: float = 0.0,
    wind_speed: float = 1.0,
    wind_vector: Optional[tuple[float, float]] = None,
    wind_noise_std: float = 0.0,
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
        action_type: Action processor type ('discrete', 'oriented', or 'run_tumble')
        observation_type: Observation model type ('concentration', 'antennae', or 'wind_vector')
        reward_type: Reward function type ('sparse' or 'step_penalty')
        plume_sigma: Gaussian plume dispersion parameter
        step_size: Movement step size in grid cells
        render_mode: Rendering mode ('rgb_array' or 'human')
        movie_dataset_id: Registry id for curated plume datasets; when set,
            movie_path is optional and will be resolved via the data zoo cache.
        movie_auto_download: Allow registry downloads when cache is missing.
        movie_cache_root: Override cache root for registry-backed datasets.
        enable_wind: Whether to instantiate a wind field (also enabled automatically when observation_type='wind_vector')
        wind_direction_deg: Wind direction in degrees when using constant wind
        wind_speed: Wind speed magnitude when using constant wind
        wind_vector: Optional explicit wind vector (overrides direction/speed)
        wind_noise_std: Gaussian noise stddev applied by WindVectorSensor

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
            f"Invalid action_type: {action_type}. Must be 'discrete', 'oriented', or 'run_tumble'."
        )

    # Create observation model
    if observation_type == "concentration":
        observation_model = ConcentrationSensor()
    elif observation_type == "antennae":
        observation_model = AntennaeArraySensor(n_sensors=2, sensor_distance=2.0)
    elif observation_type == "wind_vector":
        observation_model = WindVectorSensor(noise_std=wind_noise_std)
    else:
        raise ValueError(
            f"Invalid observation_type: {observation_type}. "
            f"Must be 'concentration', 'antennae', or 'wind_vector'."
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

    # Create plume/concentration field
    if plume == "movie":
        if not movie_path and not movie_dataset_id:
            raise ValueError(
                "env.plume=movie requires movie.path or movie.dataset_id to be provided"
            )

        dataset_path = _resolve_movie_dataset(
            movie_path=movie_path,
            movie_dataset_id=movie_dataset_id,
            movie_auto_download=movie_auto_download,
            movie_cache_root=movie_cache_root,
            movie_fps=movie_fps,
            movie_pixel_to_grid=movie_pixel_to_grid,
            movie_origin=movie_origin,
            movie_extent=movie_extent,
            movie_h5_dataset=movie_h5_dataset,
        )

        # For raw media sources (non-directory movie_path values), treat the
        # sidecar as the single source of truth for movie-level metadata. Once
        # ingestion has produced a dataset, rely on its attrs instead of
        # overriding via MovieConfig. For existing dataset directories, preserve
        # the prior behavior and allow explicit overrides.
        source_is_dir = Path(movie_path).is_dir()
        if source_is_dir:
            cfg_fps = movie_fps
            cfg_pixel_to_grid = movie_pixel_to_grid
            cfg_origin = movie_origin
            cfg_extent = movie_extent
        else:
            cfg_fps = None
            cfg_pixel_to_grid = None
            cfg_origin = None
            cfg_extent = None

        movie_cfg = MovieConfig(
            path=str(dataset_path),
            fps=cfg_fps,
            pixel_to_grid=cfg_pixel_to_grid,
            origin=cfg_origin,
            extent=cfg_extent,
            step_policy=movie_step_policy,
        )
        movie_field = MoviePlumeField(movie_cfg)
        concentration_field = movie_field  # type: ignore[assignment]

        # Override grid_size based on dataset to avoid mismatches
        grid_size = movie_field.grid_size
    else:
        concentration_field = ConcentrationField(
            grid_size=grid_size, enable_caching=True
        )
        # Manually create Gaussian field (generate_field has signature issues)
        x = np.arange(grid_size.width)
        y = np.arange(grid_size.height)
        xx, yy = np.meshgrid(x, y)
        dx = xx - goal_location.x
        dy = yy - goal_location.y
        field_array = np.exp(-(dx**2 + dy**2) / (2 * plume_sigma**2))
        concentration_field.field_array = field_array.astype(np.float32)
        concentration_field.is_generated = True

    wind_field = None
    if enable_wind or observation_type == "wind_vector":
        wind_field = ConstantWindField(
            direction_deg=wind_direction_deg,
            speed=wind_speed,
            vector=wind_vector,
        )

    # Assemble environment
    return ComponentBasedEnvironment(
        action_processor=action_processor,
        observation_model=observation_model,
        reward_function=reward_function,
        concentration_field=concentration_field,
        wind_field=wind_field,
        grid_size=grid_size,
        max_steps=max_steps,
        goal_location=goal_location,
        goal_radius=goal_radius,
        start_location=start_location,
        render_mode=render_mode,
    )


def _resolve_movie_dataset(
    *,
    movie_path: Optional[str],
    movie_dataset_id: Optional[str],
    movie_auto_download: bool,
    movie_cache_root: Optional[Union[str, Path]],
    movie_fps: Optional[float],
    movie_pixel_to_grid: Optional[tuple[float, float]],
    movie_origin: Optional[tuple[float, float]],
    movie_extent: Optional[tuple[float, float]],
    movie_h5_dataset: Optional[str],
) -> Path:
    """Resolve dataset path via registry or direct path/ingest."""

    if movie_dataset_id:
        cache_root = Path(movie_cache_root) if movie_cache_root else DEFAULT_CACHE_ROOT
        return ensure_dataset_available(
            movie_dataset_id,
            cache_root=cache_root,
            auto_download=movie_auto_download,
            verify_checksum=True,
        )

    if not movie_path:
        raise ValueError(
            "env.plume=movie requires movie.path when movie.dataset_id is not set"
        )

    return resolve_movie_dataset_path(
        movie_path,
        fps=movie_fps,
        pixel_to_grid=movie_pixel_to_grid,
        origin=movie_origin,
        extent=movie_extent,
        movie_h5_dataset=movie_h5_dataset,
    )
