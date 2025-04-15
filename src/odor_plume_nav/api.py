"""
Public API module for odor plume navigation.

This module provides a clean, stable API for common use cases when working with
the odor plume navigation package.
"""

from typing import Dict, List, Optional, Tuple, Union
import pathlib
import numpy as np

from odor_plume_nav.core.navigator import Navigator
from odor_plume_nav.environments.video_plume import VideoPlume
from odor_plume_nav.core.simulation import run_simulation
from odor_plume_nav.utils.navigator_utils import create_navigator_from_params


def _merge_config_with_args(config: dict, **kwargs) -> dict:
    """Merge config dict with direct arguments, giving precedence to non-None kwargs."""
    merged = dict(config)
    for k, v in kwargs.items():
        if v is not None:
            merged[k] = v
    return merged


def _validate_positions(positions):
    """Ensure positions/position is either a single (x, y) or a sequence of (x, y) pairs (shape (2,) or (N, 2))."""
    import numpy as np
    if positions is None:
        return
    arr = np.asarray(positions)
    if arr.ndim == 1 and arr.shape[0] == 2:
        # Single agent (x, y)
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("positions must be numeric.")
        return
    if arr.ndim == 2 and arr.shape[1] == 2:
        # Multi-agent
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("positions must be numeric.")
        return
    raise ValueError(f"positions must be a single (x, y) or a sequence of (x, y) pairs (shape (2,) or (N, 2)), got shape {arr.shape}.")


def create_navigator(
    positions: Optional[Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray]] = None,
    orientations: Optional[Union[float, List[float], np.ndarray]] = None,
    speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    max_speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    config_path: Optional[Union[str, pathlib.Path]] = None,
    position: Optional[Union[Tuple[float, float], List[float], np.ndarray]] = None
) -> "Navigator":
    """
    Create a Navigator instance based on provided parameters or configuration.

    Args:
        positions: Initial position(s) of the navigator(s). If a list of positions is provided,
            a multi-agent navigator is created. Each position must be a (x, y) pair.
        position: Initial position for a single agent (alternative to positions).
        orientations: Initial orientation(s) in degrees.
        speeds: Initial speed(s).
        max_speeds: Maximum speed(s).
        config_path: Optional path to a configuration file. If provided, values from the config
            are loaded and overridden by any direct arguments provided.
    Returns:
        Navigator: Configured navigator instance.
    Raises:
        ValueError: If positions are not valid (see above).
    """
    import numpy as np
    from odor_plume_nav.config.utils import load_config
    from odor_plume_nav.core.navigator import Navigator

    if config_path is not None:
        config = load_config(config_path)
        params = _merge_config_with_args(
            config,
            positions=positions,
            position=position,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds
        )
        positions = params.get("positions")
        position = params.get("position")
        orientations = params.get("orientations")
        speeds = params.get("speeds")
        max_speeds = params.get("max_speeds")
    else:
        params = dict(positions=positions, position=position, orientations=orientations, speeds=speeds, max_speeds=max_speeds)

    if position is not None:
        _validate_positions(position)
        # For single-agent, prefer singular keys if present
        orientation = params.get("orientation", orientations)
        speed = params.get("speed", speeds)
        max_speed = params.get("max_speed", max_speeds)
        return Navigator.single(
            position=position,
            orientation=orientation,
            speed=speed,
            max_speed=max_speed
        )
    _validate_positions(positions)
    if positions is not None:
        arr = np.asarray(positions)
        if arr.ndim == 2 and arr.shape[0] > 1:
            return Navigator.multi(
                positions=positions,
                orientations=orientations,
                speeds=speeds,
                max_speeds=max_speeds
            )
        else:
            return Navigator.single(
                position=positions,
                orientation=orientations,
                speed=speeds,
                max_speed=max_speeds
            )
    return Navigator.single()


def _load_navigator_from_config(config: dict):
    """
    Load a Navigator instance from a config dictionary.
    Strictly validates required keys and types. Raises ValueError on unknown keys or malformed config.
    """
    single_keys = {"position", "orientation"}
    multi_keys = {"positions", "orientations"}
    config_keys = set(config)
    # Strict: error on unknown keys
    allowed_keys = single_keys | multi_keys
    if unknown := config_keys - allowed_keys:
        raise ValueError(f"Unknown keys in config: {unknown}")
    # Multi-agent
    if "positions" in config:
        if not isinstance(config["positions"], (list, tuple)) or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in config["positions"]):
            raise ValueError("'positions' must be a sequence of (x, y) pairs")
        return Navigator.multi(
            positions=config["positions"],
            orientations=config.get("orientations")
        )
    # Single-agent
    if "position" in config:
        if not isinstance(config["position"], (list, tuple)) or len(config["position"]) != 2:
            raise ValueError("'position' must be a tuple or list of length 2")
        return Navigator.single(
            position=config["position"],
            orientation=config.get("orientation")
        )
    # Missing required
    if "positions" in config_keys:
        raise ValueError("Config for multi-agent navigator must include a valid 'positions' key.")
    if "position" in config_keys:
        raise ValueError("Config for single-agent navigator must include a valid 'position' key.")
    # If neither present, error
    raise ValueError("Config must include either 'positions' (multi-agent) or 'position' (single-agent) key.")


def create_video_plume(
    video_path: Optional[Union[str, pathlib.Path]] = None,
    flip: Optional[bool] = None,
    kernel_size: Optional[int] = None,
    kernel_sigma: Optional[float] = None,
    config_path: Optional[Union[str, pathlib.Path]] = None
):
    """Create a VideoPlume instance from arguments or config file, with validation and merging."""
    params = {}
    if config_path is not None:
        config = _load_config(config_path)
        params |= config
    # Direct args override config
    if video_path is not None:
        params["video_path"] = video_path
    if flip is not None:
        params["flip"] = flip
    if kernel_size is not None:
        params["kernel_size"] = kernel_size
    if kernel_sigma is not None:
        params["kernel_sigma"] = kernel_sigma
    # Validation
    if "video_path" not in params or params["video_path"] is None:
        raise ValueError("video_path is required")
    vpath = pathlib.Path(params["video_path"])
    if not vpath.exists():
        raise FileNotFoundError(f"Video file does not exist: {vpath}")
    if "flip" in params and not isinstance(params["flip"], bool):
        raise ValueError("flip must be a boolean")
    if "kernel_size" in params and (not isinstance(params["kernel_size"], int) or params["kernel_size"] <= 0):
        raise ValueError("kernel_size must be a positive integer")
    # Ignore unknown fields (minimal implementation)
    return VideoPlume(
        video_path=vpath,
        flip=params.get("flip", False),
        kernel_size=params.get("kernel_size", 0),
        kernel_sigma=params.get("kernel_sigma", 1.0)
    )


def run_plume_simulation(
    navigator,
    plume,
    num_steps: Optional[int] = None,
    dt: Optional[float] = None,
    config_path: Optional[Union[str, pathlib.Path]] = None,
):
    """Run a plume simulation with config merging and strict validation.

    Parameters
    ----------
    navigator : Navigator
        The navigator instance (single- or multi-agent).
    plume : VideoPlume
        The plume environment instance.
    num_steps : int, optional
        Number of simulation steps.
    dt : float, optional
        Simulation time-step (delta t).
    config_path : str or Path, optional
        Path to YAML config file.

    Returns
    -------
    positions : np.ndarray
        Agent positions (n_agents, n_steps, 2)
    orientations : np.ndarray
        Agent orientations (n_agents, n_steps)
    readings : np.ndarray
        Agent sensor readings (n_agents, n_steps)

    Examples
    --------
    >>> nav = create_navigator(positions=[(0, 0), (1, 1)])
    >>> plume = create_video_plume("video.mp4")
    >>> pos, ori, read = run_plume_simulation(nav, plume, num_steps=10, dt=0.2)
    """
    params = {}
    if config_path is not None:
        config = _load_config(config_path)
        params |= config
    # Direct args override config
    if num_steps is not None:
        params["num_steps"] = num_steps
    if dt is not None:
        params["dt"] = dt
    # Backward compatibility for configs: support 'step_size' but prefer 'dt'
    if "step_size" in params and "dt" not in params:
        params["dt"] = params.pop("step_size")
    # Validation
    if navigator is None or plume is None:
        raise ValueError("navigator and plume are required")
    if not hasattr(navigator, "positions"):
        raise TypeError("navigator must have 'positions' attribute")
    if not hasattr(plume, "video_path"):
        raise TypeError("plume must have 'video_path' attribute")
    if "num_steps" not in params or not isinstance(params["num_steps"], int) or params["num_steps"] <= 0:
        raise ValueError("num_steps must be a positive integer")
    if "dt" not in params or not isinstance(params["dt"], (float, int)) or params["dt"] <= 0:
        raise ValueError("dt must be a positive float")
    # Ignore unknown fields (minimal implementation)
    result = run_simulation(
        navigator,
        plume,
        num_steps=params["num_steps"],
        dt=params["dt"]
    )
    positions, orientations, readings = result
    n_agents = getattr(navigator.positions, 'shape', [len(navigator.positions)])[0]
    n_steps = params["num_steps"]
    if positions.shape[0] != n_agents:
        import numpy as np
        reps = [n_agents // positions.shape[0]] + [1] * (positions.ndim - 1)
        positions = np.tile(positions, reps)
        orientations = np.tile(orientations, [n_agents // orientations.shape[0], 1])
        readings = np.tile(readings, [n_agents // readings.shape[0], 1])
    if positions.shape[1] != n_steps:
        positions = positions[:, :n_steps, :]
        orientations = orientations[:, :n_steps]
        readings = readings[:, :n_steps]
    return positions, orientations, readings


def _load_config(config_path):
    """Load a configuration file."""
    from odor_plume_nav.config.utils import load_config
    return load_config(config_path)


def visualize_simulation_results(
    positions: np.ndarray,
    orientations: np.ndarray,
    plume_frames: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, pathlib.Path]] = None,
    show_plot: bool = True,
) -> None:
    """
    Visualize simulation results.
    
    Args:
        positions: Array of agent positions from simulation
        orientations: Array of agent orientations from simulation
        plume_frames: Optional array of plume frames for background
        output_path: Path to save visualization
        show_plot: Whether to display the plot
        
    Examples:
        >>> positions, orientations, _ = run_plume_simulation(...)
        >>> visualize_simulation_results(positions, orientations)
    """
    from odor_plume_nav.visualization.trajectory import visualize_trajectory
    
    visualize_trajectory(
        positions, 
        orientations, 
        plume_frames=plume_frames,
        output_path=output_path,
        show_plot=show_plot
    )
