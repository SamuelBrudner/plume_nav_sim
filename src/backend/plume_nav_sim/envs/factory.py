import logging
import warnings
from pathlib import Path
from typing import Any, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np

from .._compat import ValidationError, validate_coordinates, validate_grid_size
from ..actions import DiscreteGridActions, OrientedGridActions
from ..actions.oriented_run_tumble import OrientedRunTumbleActions
from ..constants import DEFAULT_SOURCE_LOCATION
from ..core.geometry import Coordinates, GridSize
from ..data_zoo.download import DatasetDownloadError, ensure_dataset_available
from ..data_zoo.loader import load_plume
from ..data_zoo.registry import DEFAULT_CACHE_ROOT, get_dataset_registry
from ..observations import AntennaeArraySensor, ConcentrationSensor, WindVectorSensor
from ..plume.gaussian import GaussianPlume
from ..plume.video import VideoConfig, VideoPlume, resolve_movie_dataset_path
from ..rewards import SparseGoalReward, StepPenaltyReward
from ..wind_field import ConstantWindField
from .plume_env import PlumeEnv

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


def _create_plume_env_from_selectors(  # noqa: C901
    *,
    grid_size: Union[tuple[int, int], GridSize] = (128, 128),
    goal_location: Optional[Union[tuple[int, int], Coordinates]] = None,
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
    warn_deprecated: bool = False,
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
    movie_chunks: Any = None,
    movie_data: Any = None,
    enable_wind: bool = False,
    wind_direction_deg: float = 0.0,
    wind_speed: float = 1.0,
    wind_vector: Optional[tuple[float, float]] = None,
    wind_noise_std: float = 0.0,
) -> PlumeEnv:
    # Convert tuples to proper types
    grid_size = validate_grid_size(grid_size)
    goal_location_was_implicit = goal_location is None
    goal_value = DEFAULT_SOURCE_LOCATION if goal_location is None else goal_location
    goal_location = (
        validate_coordinates(goal_value, grid_size)
        if not goal_location_was_implicit
        else Coordinates(x=int(goal_value[0]), y=int(goal_value[1]))
    )
    if start_location is not None:
        start_location = validate_coordinates(start_location, grid_size)

    if goal_radius < 0:
        raise ValueError("goal_radius must be non-negative")
    if goal_radius == 0:
        goal_radius = float(np.finfo(np.float32).eps)

    goal_location_is_default = goal_location_was_implicit

    # Preserve the long-standing small-grid fallback only for the implicit default.
    if goal_location_is_default and not grid_size.contains(goal_location):
        goal_location = Coordinates(
            x=min(max(goal_location.x, 0), grid_size.width - 1),
            y=min(max(goal_location.y, 0), grid_size.height - 1),
        )

    action_model = _create_action_model(action_type, step_size)
    observation_model = _create_observation_model(
        observation_type, wind_noise_std=wind_noise_std
    )

    # Create plume/concentration field
    if plume == "movie":
        if movie_data is None and not movie_path and not movie_dataset_id:
            raise ValueError(
                "env.plume=movie requires movie.path or movie.dataset_id to be provided"
            )

        cache_root = Path(movie_cache_root) if movie_cache_root else DEFAULT_CACHE_ROOT
        dataset_path: Path | None = None
        if movie_path or movie_dataset_id:
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
            if resolved is None and dataset_path is not None and dataset_path.is_dir():
                resolved = _source_location_px_from_dataset_dir(dataset_path)
            if resolved is not None:
                goal_location = Coordinates(x=resolved[0], y=resolved[1])

        data_array = movie_data
        if data_array is None and movie_dataset_id:
            if movie_normalize is not None or movie_chunks is not None:
                chunks = "auto" if movie_chunks is None else movie_chunks
                try:
                    data_array = load_plume(
                        movie_dataset_id,
                        normalize=movie_normalize,
                        cache_root=cache_root,
                        auto_download=movie_auto_download,
                        chunks=chunks,
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Failed loading registry dataset '{movie_dataset_id}' with normalization/chunking: {exc}"
                    ) from exc
        elif movie_normalize is not None or movie_chunks is not None:
            raise ValueError(
                "movie_normalize/movie_chunks are only supported with movie_dataset_id"
            )

        # For raw media sources (non-directory movie_path values), treat the
        # sidecar as the single source of truth for movie-level metadata. Once
        # ingestion has produced a dataset, rely on its attrs instead of
        # overriding via VideoConfig. For existing dataset directories, preserve
        # the prior behavior and allow explicit overrides.
        source_root = Path(movie_path) if movie_path else dataset_path
        source_is_dir = source_root.is_dir() if source_root is not None else False
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
            path="" if dataset_path is None else str(dataset_path),
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
        if goal_location_is_default and not grid_size.contains(goal_location):
            goal_location = grid_size.center()
    else:
        concentration_field = _create_gaussian_plume_field(
            grid_size, goal_location, plume_sigma
        )
    reward_function = _create_reward_function(
        reward_type, goal_location, goal_radius=goal_radius
    )

    wind_field = None
    if enable_wind or observation_type == "wind_vector":
        wind_field = ConstantWindField(
            direction_deg=wind_direction_deg,
            speed=wind_speed,
            vector=wind_vector,
        )

    plume_params = None
    if plume != "movie":
        plume_params = {"sigma": float(plume_sigma)}

    return PlumeEnv(
        grid_size=grid_size.to_tuple(),
        source_location=goal_location,
        start_location=start_location,
        max_steps=max_steps,
        goal_radius=goal_radius,
        plume=concentration_field,
        sensor_model=observation_model,
        action_model=action_model,
        reward_fn=reward_function,
        plume_params=plume_params,
        render_mode=render_mode,
        wind_field=wind_field,
    )


def create_component_environment(
    **kwargs: Any,
) -> PlumeEnv:
    warnings.warn(
        "create_component_environment is deprecated; use plume_nav_sim.make_env(...) "
        "with selector kwargs or plume_nav_sim.config.SimulationSpec instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs.pop("warn_deprecated", None)
    return _create_plume_env_from_selectors(**kwargs)


def _create_gaussian_plume_field(
    grid_size: GridSize, goal_location: Coordinates, plume_sigma: float
) -> GaussianPlume:
    return GaussianPlume(
        grid_size=grid_size, source_location=goal_location, sigma=plume_sigma
    )


def _create_action_model(action_type: str, step_size: int):
    if action_type == "discrete":
        return DiscreteGridActions(step_size=step_size)
    if action_type == "oriented":
        return OrientedGridActions(step_size=step_size)
    if action_type == "run_tumble":
        return OrientedRunTumbleActions(step_size=step_size)
    raise ValueError(
        f"Invalid action_type: {action_type}. Must be 'discrete', 'oriented', or 'run_tumble'."
    )


def _create_observation_model(observation_type: str, *, wind_noise_std: float):
    if observation_type == "concentration":
        return ConcentrationSensor()
    if observation_type == "antennae":
        return AntennaeArraySensor(n_sensors=2, sensor_distance=2.0)
    if observation_type == "wind_vector":
        return WindVectorSensor(noise_std=wind_noise_std)
    raise ValueError(
        f"Invalid observation_type: {observation_type}. "
        f"Must be 'concentration', 'antennae', or 'wind_vector'."
    )


def _create_reward_function(
    reward_type: str, goal_location: Coordinates, *, goal_radius: float
):
    if reward_type == "sparse":
        return SparseGoalReward(goal_position=goal_location, goal_radius=goal_radius)
    if reward_type == "step_penalty":
        return StepPenaltyReward(
            goal_position=goal_location,
            goal_radius=goal_radius,
            goal_reward=1.0,
            step_penalty=0.01,
        )
    raise ValueError(
        f"Invalid reward_type: {reward_type}. Must be 'sparse' or 'step_penalty'."
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
        except DatasetDownloadError as exc:
            raise ValueError(str(exc)) from exc

    raise ValueError(
        "env.plume=movie requires movie.path or movie.dataset_id to be provided"
    )
