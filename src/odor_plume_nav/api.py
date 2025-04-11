"""
Public API module for odor plume navigation.

This module provides a clean, stable API for common use cases when working with
the odor plume navigation package.
"""

from typing import Dict, List, Optional, Tuple, Union
import pathlib
import numpy as np

from odor_plume_nav.core.navigator import Navigator, SimpleNavigator, VectorizedNavigator
from odor_plume_nav.navigator_factory import create_navigator_from_config
from odor_plume_nav.environments.video_plume import VideoPlume
from odor_plume_nav.simulation import run_simulation
from odor_plume_nav.utils import normalize_array_parameter, create_navigator_from_params


def create_navigator(
    positions: Optional[Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray]] = None,
    orientations: Optional[Union[float, List[float], np.ndarray]] = None,
    speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    max_speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    config_path: Optional[Union[str, pathlib.Path]] = None
) -> Union[Navigator, VectorizedNavigator]:
    """
    Create a Navigator instance based on provided parameters or configuration.
    
    Args:
        positions: Initial position(s) of the navigator(s). If a list of positions is provided,
                  a multi-agent navigator will be created.
        orientations: Initial orientation(s) of the navigator(s) in degrees.
        speeds: Initial speed(s) of the navigator(s).
        max_speeds: Maximum speed(s) of the navigator(s).
        config_path: Path to a configuration file. If provided, other parameters will
                    be ignored and the configuration file will be used.
                    
    Returns:
        A Navigator or VectorizedNavigator instance.
    """
    # Configuration-based creation has priority
    if config_path is not None:
        return _extracted_from_create_navigator_25(config_path)
    # Parameter-based creation
    if positions is not None:
        return create_navigator_from_params(positions, orientations, speeds, max_speeds)

    # Default case: create a navigator with default parameters
    return Navigator()


# TODO Rename this here and in `create_navigator`
def _extracted_from_create_navigator_25(config_path):
    from odor_plume_nav.config.utils import load_config
    config = load_config(config_path)

    # Filter out keys that aren't relevant for Navigator initialization
    navigator_keys = [
        "position", "orientation", "speed", "max_speed", 
        "positions", "orientations", "speeds", "max_speeds",
        "angular_velocity", "angular_velocities", "num_agents"
    ]
    filtered_config = {k: v for k, v in config.items() if k in navigator_keys}

    # Create appropriate navigator based on whether it's multi-agent
    is_multi_agent = "positions" in filtered_config and isinstance(
        filtered_config["positions"], (list, np.ndarray)
    ) and len(filtered_config["positions"]) > 1

    return (
        VectorizedNavigator(**filtered_config)
        if is_multi_agent
        else Navigator(**filtered_config)
    )


def create_video_plume(
    video_path: Union[str, pathlib.Path],
    config_path: Optional[Union[str, pathlib.Path]] = None,
    config_dict: Optional[Dict] = None,
    **kwargs
) -> VideoPlume:
    """
    Create a VideoPlume instance with simplified interface.
    
    Args:
        video_path: Path to video file
        config_path: Path to a configuration file
        config_dict: Configuration dictionary
        **kwargs: Additional parameters to pass to VideoPlume
        
    Returns:
        Configured VideoPlume instance
        
    Examples:
        >>> # Basic usage
        >>> plume = create_video_plume("path/to/video.mp4")
        >>> 
        >>> # With additional options
        >>> plume = create_video_plume("path/to/video.mp4", flip=True, kernel_size=5)
        >>>
        >>> # From config
        >>> plume = create_video_plume("path/to/video.mp4", config_path="path/to/config.yaml")
    """
    # Handle configuration from file if provided
    if config_dict is None and config_path is not None:
        from odor_plume_nav.config.utils import load_config
        config_dict = load_config(config_path)
    
    # If config_dict is provided, use it
    if config_dict is not None:
        return VideoPlume.from_config(video_path=video_path, config_dict=config_dict)
    
    # Otherwise, use kwargs
    return VideoPlume(video_path=video_path, **kwargs)


def run_plume_simulation(
    navigator: Navigator,
    video_plume: VideoPlume,
    num_steps: int,
    step_size: float = 1.0,
    sensor_distance: float = 5.0,
    sensor_angle: float = 45.0,
    config_path: Optional[Union[str, pathlib.Path]] = None,
    config_dict: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a simulation of agent(s) navigating through a plume.
    
    This function simplifies running a simulation with agents navigating
    through a video plume environment.
    
    Args:
        navigator: Navigator instance controlling agent(s)
        video_plume: VideoPlume instance providing the environment
        num_steps: Number of simulation steps to run
        step_size: Size of each simulation step
        sensor_distance: Distance of the sensors from the agent
        sensor_angle: Angle of the sensors relative to the agent's orientation
        config_path: Path to a configuration file
        config_dict: Configuration dictionary
        
    Returns:
        Tuple of (positions_history, orientations_history, odor_readings)
        
    Examples:
        >>> navigator = create_navigator(positions=(10, 20), orientations=45)
        >>> plume = create_video_plume("path/to/video.mp4")
        >>> positions, orientations, readings = run_plume_simulation(
        ...     navigator, plume, num_steps=100, step_size=0.5
        ... )
    """
    # Handle configuration from file if provided
    if config_dict is None and config_path is not None:
        from odor_plume_nav.config.utils import load_config
        config_dict = load_config(config_path)

    # Prepare simulation parameters
    sim_params = {
        "num_steps": num_steps,
        "step_size": step_size,
        "sensor_distance": sensor_distance,
        "sensor_angle": sensor_angle,
    }

    # If config_dict is provided, update simulation parameters
    if config_dict is not None:
        sim_params |= config_dict

    # Run the simulation
    return run_simulation(navigator, video_plume, **sim_params)


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
