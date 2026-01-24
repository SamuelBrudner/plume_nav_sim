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

import logging
from pathlib import Path
from typing import Any, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np

from ..actions import DiscreteGridActions, OrientedGridActions
from ..actions.oriented_run_tumble import OrientedRunTumbleActions
from ..constants import DEFAULT_SOURCE_LOCATION
from ..core.geometry import Coordinates, GridSize
from ..data_zoo import (
    DEFAULT_CACHE_ROOT,
    describe_dataset,
    ensure_dataset_available,
    get_dataset_registry,
    load_plume,
)
from ..observations import AntennaeArraySensor, ConcentrationSensor, WindVectorSensor
from ..plume.gaussian import GaussianPlume
from ..plume.video import VideoConfig, VideoPlume, resolve_movie_dataset_path
from ..rewards import SparseGoalReward, StepPenaltyReward
from ..wind_field import ConstantWindField
from .component_env import ComponentBasedEnvironment

LOG = logging.getLogger(__name__)

__all__ = ["create_component_environment"]


def _coerce_source_location_px(value: object) -> tuple[int, int] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except Exception:
        return None


def _source_location_px_from_dataset_dir(dataset_path: Path) -> tuple[int, int] | None:
    try:
        import zarr
    except Exception:
        return None
    try:
        root = zarr.open_group(str(dataset_path), mode="r")
        direct = _coerce_source_location_px(root.attrs.get("source_location_px"))
        if direct is not None:
            return direct
        ingest_args = root.attrs.get("ingest_args")
        if isinstance(ingest_args, dict):
            return _coerce_source_location_px(ingest_args.get("source_location_px"))

        source_format = root.attrs.get("source_format")
        dims = root.attrs.get("dims")
        looks_like_rigolli = (
            source_format == "matlab_v73"
            or dataset_path.name.startswith("rigolli_")
            or (dims == ["t", "y", "x"] and "x" in root and "y" in root)
        )
        looks_like_emonet = (
            source_format == "emonet_flywalk" or dataset_path.name.startswith("emonet_")
        )
        if looks_like_rigolli or looks_like_emonet:
            conc = root.get("concentration")
            if conc is not None and hasattr(conc, "shape") and len(conc.shape) >= 3:
                ny = int(conc.shape[1])
                resolved = (0, ny // 2)
                LOG.warning(
                    "Dataset missing source_location_px; using heuristic (x=%d, y=%d) for %s",
                    resolved[0],
                    resolved[1],
                    str(dataset_path),
                )
                return resolved
    except Exception:
        return None
    return None


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
    warn_deprecated: bool = True,
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
    movie_normalize: str | None = None,
    movie_chunks: Any = "auto",
    movie_data: Any = None,
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
        warn_deprecated: Emit a deprecation warning when constructing ComponentBasedEnvironment
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

    goal_location_is_default = (
        goal_location.x,
        goal_location.y,
    ) == DEFAULT_SOURCE_LOCATION

    # Clamp goal_location to grid bounds if default is outside smaller grid
    if goal_location.x >= grid_size.width or goal_location.y >= grid_size.height:
        goal_location = Coordinates(
            x=min(goal_location.x, grid_size.width - 1),
            y=min(goal_location.y, grid_size.height - 1),
        )

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

    # Create plume/concentration field
    if plume == "movie":
        if not movie_path and not movie_dataset_id:
            raise ValueError(
                "env.plume=movie requires movie.path or movie.dataset_id to be provided"
            )

        cache_root = Path(movie_cache_root) if movie_cache_root else DEFAULT_CACHE_ROOT
        dataset_path = _resolve_movie_dataset(
            movie_path=movie_path,
            movie_dataset_id=movie_dataset_id,
            movie_auto_download=movie_auto_download,
            movie_cache_root=cache_root,
            movie_fps=movie_fps,
            movie_pixel_to_grid=movie_pixel_to_grid,
            movie_origin=movie_origin,
            movie_extent=movie_extent,
            movie_h5_dataset=movie_h5_dataset,
        )

        if goal_location_is_default:
            resolved: tuple[int, int] | None = None
            if movie_dataset_id:
                entry = describe_dataset(movie_dataset_id)
                ingest = getattr(entry, "ingest", None)
                resolved = _coerce_source_location_px(
                    getattr(ingest, "source_location_px", None)
                )
            if resolved is None and Path(dataset_path).is_dir():
                resolved = _source_location_px_from_dataset_dir(Path(dataset_path))
            if resolved is not None:
                goal_location = Coordinates(x=resolved[0], y=resolved[1])

        data_array = movie_data
        if data_array is None and movie_dataset_id:
            data_array = load_plume(
                movie_dataset_id,
                normalize=movie_normalize,
                cache_root=cache_root,
                auto_download=movie_auto_download,
                chunks=movie_chunks,
            )

        # For raw media sources (non-directory movie_path values), treat the
        # sidecar as the single source of truth for movie-level metadata. Once
        # ingestion has produced a dataset, rely on its attrs instead of
        # overriding via VideoConfig. For existing dataset directories, preserve
        # the prior behavior and allow explicit overrides.
        source_root = Path(movie_path) if movie_path else Path(dataset_path)
        source_is_dir = source_root.is_dir()
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

        movie_cfg = VideoConfig(
            path=str(dataset_path),
            fps=cfg_fps,
            pixel_to_grid=cfg_pixel_to_grid,
            origin=cfg_origin,
            extent=cfg_extent,
            step_policy=movie_step_policy,
            data_array=data_array,
        )
        movie_field = VideoPlume(movie_cfg)
        concentration_field = movie_field  # type: ignore[assignment]

        # Override grid_size based on dataset to avoid mismatches
        grid_size = movie_field.grid_size
    else:
        concentration_field = _create_gaussian_plume_field(
            grid_size, goal_location, plume_sigma
        )
    # Create reward function (after movie goal override)
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
        _warn_deprecated=warn_deprecated,
    )


def _create_gaussian_plume_field(
    grid_size: GridSize, goal_location: Coordinates, plume_sigma: float
) -> GaussianPlume:
    return GaussianPlume(
        grid_size=grid_size, source_location=goal_location, sigma=plume_sigma
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

    # Explicit paths (including raw media) take precedence over registry ids so
    # callers can override a curated dataset with a local copy.
    if movie_path:
        return resolve_movie_dataset_path(
            movie_path,
            fps=movie_fps,
            pixel_to_grid=movie_pixel_to_grid,
            origin=movie_origin,
            extent=movie_extent,
            movie_h5_dataset=movie_h5_dataset,
        )

    if movie_dataset_id:
        cache_root = Path(movie_cache_root) if movie_cache_root else DEFAULT_CACHE_ROOT
        try:
            return ensure_dataset_available(
                movie_dataset_id,
                cache_root=cache_root,
                auto_download=movie_auto_download,
                verify_checksum=True,
            )
        except KeyError as exc:
            known = sorted(get_dataset_registry().keys())
            known_hint = f" Known ids: {', '.join(known)}." if known else ""
            raise ValueError(
                f"Unknown movie dataset id '{movie_dataset_id}'.{known_hint}"
            ) from exc

    raise ValueError(
        "env.plume=movie requires movie.path or movie.dataset_id to be provided"
    )
