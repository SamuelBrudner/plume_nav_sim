"""
Public API module for odor plume navigation.

This module provides a clean, stable API for common use cases when working with
the odor plume navigation package.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pathlib

# Update imports to use the new module structure
from odor_plume_nav.core.navigator import Navigator
from odor_plume_nav.environments.video_plume import VideoPlume
from odor_plume_nav.core.simulation import run_simulation


def create_navigator(
    positions: Optional[Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray]] = None,
    orientations: Optional[Union[float, List[float], np.ndarray]] = None,
    speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    max_speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    config_path: Optional[Union[str, pathlib.Path]] = None,
    config_dict: Optional[Dict] = None,
) -> Navigator:
    """
    Create a Navigator instance with simplified interface.
    
    This function provides a simplified way to create a Navigator instance
    handling both single and multi-agent cases with a consistent interface.
    
    Args:
        positions: Agent position(s) as a tuple (x, y) for single agent or a list/array of positions
        orientations: Initial orientation(s) in degrees
        speeds: Initial movement speed(s)
        max_speeds: Maximum allowed movement speed(s)
        config_path: Path to a configuration file
        config_dict: Configuration dictionary
        
    Returns:
        Configured Navigator instance
        
    Examples:
        >>> # Create a single navigator
        >>> nav = create_navigator(positions=(10, 20), orientations=45, speeds=0.5)
        >>> 
        >>> # Create multiple navigators
        >>> positions = [(10, 20), (30, 40), (50, 60)]
        >>> orientations = [45, 90, 135]
        >>> nav = create_navigator(positions=positions, orientations=orientations)
        >>>
        >>> # Create from config
        >>> nav = create_navigator(config_path="path/to/config.yaml")
    """
    # Handle configuration from file if provided
    if config_dict is None and config_path is not None:
        from odor_plume_nav.config.utils import load_config
        config_dict = load_config(config_path)
    
    # If config_dict is provided, create navigator from it
    if config_dict is not None:
        return Navigator.from_config(config_dict)
    
    # Otherwise, prepare parameters for the navigator
    params = {}
    
    # Handle single agent case
    if positions is not None and isinstance(positions, tuple) and len(positions) == 2:
        params["position"] = positions
        if orientations is not None:
            params["orientation"] = orientations if isinstance(orientations, (int, float)) else orientations[0]
        if speeds is not None:
            params["speed"] = speeds if isinstance(speeds, (int, float)) else speeds[0]
        if max_speeds is not None:
            params["max_speed"] = max_speeds if isinstance(max_speeds, (int, float)) else max_speeds[0]
            
    # Handle multi-agent case
    elif positions is not None and isinstance(positions, (list, np.ndarray)):
        # Convert to numpy arrays as needed
        if isinstance(positions, list):
            positions = np.array(positions)
        params["positions"] = positions
        
        if orientations is not None:
            if isinstance(orientations, (int, float)):
                orientations = np.full(len(positions), orientations)
            elif isinstance(orientations, list):
                orientations = np.array(orientations)
            params["orientations"] = orientations
            
        if speeds is not None:
            if isinstance(speeds, (int, float)):
                speeds = np.full(len(positions), speeds)
            elif isinstance(speeds, list):
                speeds = np.array(speeds)
            params["speeds"] = speeds
            
        if max_speeds is not None:
            if isinstance(max_speeds, (int, float)):
                max_speeds = np.full(len(positions), max_speeds)
            elif isinstance(max_speeds, list):
                max_speeds = np.array(max_speeds)
            params["max_speeds"] = max_speeds
    
    return Navigator(**params)


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
