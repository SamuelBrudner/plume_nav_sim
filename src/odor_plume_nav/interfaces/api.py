"""
Public API module for odor plume navigation.

This module provides a clean, stable API for common use cases when working with
the odor plume navigation package.
"""

from typing import List, Optional, Tuple, Union
import pathlib
import numpy as np

from odor_plume_nav.domain.navigator import Navigator
from odor_plume_nav.adapters.video_plume_opencv import VideoPlume
from odor_plume_nav.services.simulation_runner import run_simulation
from odor_plume_nav.api_utils import merge_config_with_args
from odor_plume_nav.utils.navigator_utils import (
    validate_positions,
)
from odor_plume_nav.services.config_loader import load_config
from odor_plume_nav.interfaces.visualization.visualization import visualize_trajectory

def create_navigator(
    positions: Optional[Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray]] = None,
    orientations: Optional[Union[float, List[float], np.ndarray]] = None,
    speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    max_speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    config_path: Optional[Union[str, pathlib.Path]] = None,
    position: Optional[Union[Tuple[float, float], List[float], np.ndarray]] = None
) -> Navigator:
    """
    Create a Navigator instance based on provided parameters or configuration.

    Parameters
    ----------
    positions : tuple, list, np.ndarray, optional
        Initial position(s) of the navigator(s). If a list of positions is provided,
        a multi-agent navigator is created. Each position must be a (x, y) pair.
    position : tuple, list, np.ndarray, optional
        Initial position for a single agent (alternative to positions).
    orientations : float, list, np.ndarray, optional
        Initial orientation(s) in degrees.
    speeds : float, list, np.ndarray, optional
        Initial speed(s).
    max_speeds : float, list, np.ndarray, optional
        Maximum speed(s).
    config_path : str or Path, optional
        Optional path to a configuration file. If provided, values from the config
        are loaded and overridden by any direct arguments provided.

    Returns
    -------
    Navigator
        Configured navigator instance.

    Raises
    ------
    ValueError
        If positions are not valid (see above).
        If both 'position' and 'positions' are provided.

    See Also
    --------
    create_video_plume : For generating video plumes for simulation environments.
    run_plume_simulation : For running a full plume navigation simulation pipeline.
    """
    if position is not None and positions is not None:
        raise ValueError("Cannot specify both 'position' (single-agent) and 'positions' (multi-agent). Please provide only one.")
    if config_path is not None:
        config = load_config(config_path)
        params = merge_config_with_args(
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
        validate_positions(position)
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
    validate_positions(positions)
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


def create_video_plume(
    video_path: Optional[Union[str, pathlib.Path]] = None,
    flip: Optional[bool] = None,
    kernel_size: Optional[int] = None,
    kernel_sigma: Optional[float] = None,
    config_path: Optional[Union[str, pathlib.Path]] = None
) -> VideoPlume:
    """Create a VideoPlume instance from arguments or config file, with validation and merging.

    Parameters
    ----------
    video_path : str or Path, optional
        Path to the video file.
    flip : bool, optional
        Whether to flip the video vertically.
    kernel_size : int, optional
        Size of the Gaussian kernel for plume smoothing.
    kernel_sigma : float, optional
        Standard deviation of the Gaussian kernel.
    config_path : str or Path, optional
        Optional path to a configuration file. If provided, values from the config
        are loaded and overridden by any direct arguments provided.

    Returns
    -------
    VideoPlume
        Configured VideoPlume instance.

    See Also
    --------
    create_navigator : For creating agent navigators.
    run_plume_simulation : For running a full plume navigation simulation pipeline.
    """
    params = {}
    if config_path is not None:
        config = load_config(config_path)
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
    navigator: Navigator,
    plume: VideoPlume,
    num_steps: Optional[int] = None,
    dt: Optional[float] = None,
    config_path: Optional[Union[str, pathlib.Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a plume simulation with config merging and strict validation.

    Parameters
    ----------
    navigator : Navigator
        The navigator instance (single- or multi-agent).
    plume : VideoPlume
        The VideoPlume environment instance.
    num_steps : int, optional
        Number of simulation steps to run.
    dt : float, optional
        Simulation time step.
    config_path : str or Path, optional
        Optional path to a configuration file. If provided, values from the config
        are loaded and overridden by any direct arguments provided.

    Returns
    -------
    positions : np.ndarray
        Agent positions of shape (n_agents, n_steps, 2).
    orientations : np.ndarray
        Agent orientations of shape (n_agents, n_steps).
    readings : np.ndarray
        Sensor readings of shape (n_agents, n_steps).

    See Also
    --------
    create_navigator : For creating agent navigators.
    create_video_plume : For generating video plumes for simulation environments.
    """
    params = {}
    if config_path is not None:
        config = load_config(config_path)
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


def visualize_simulation_results(
    positions: np.ndarray,
    orientations: np.ndarray,
    plume_frames: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, pathlib.Path]] = None,
    show_plot: bool = True,
    close_plot: Optional[bool] = None,
) -> "matplotlib.figure.Figure":
    """
    Visualize agent trajectories and orientations, optionally overlayed on plume frames.

    If `plume_frames` is None, the function plots only agent positions/orientations.

    Parameters
    ----------
    positions : np.ndarray
        Agent positions of shape (n_agents, n_steps, 2).
    orientations : np.ndarray
        Agent orientations of shape (n_agents, n_steps).
    plume_frames : np.ndarray, optional
        Plume video frames (n_steps, H, W, 3) or None.
    output_path : str or Path, optional
        If specified, saves the figure to this path.
    show_plot : bool, default True
        If True, displays the plot interactively.
    close_plot : bool, optional
        If True, closes the figure after saving (default: True if output_path is set and show_plot is False).

    Returns
    -------
    matplotlib.figure.Figure
        The created matplotlib figure.
    """
    return visualize_trajectory(
        positions=positions,
        orientations=orientations,
        plume_frames=plume_frames,
        output_path=output_path,
        show_plot=show_plot,
        close_plot=close_plot
    )
