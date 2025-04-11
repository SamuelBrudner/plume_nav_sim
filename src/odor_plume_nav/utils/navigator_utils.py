"""
Utility functions for navigator creation and management.

This module provides helper functions for creating and manipulating
navigator instances.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from odor_plume_nav.core.navigator import Navigator, VectorizedNavigator


def normalize_array_parameter(
    param: Optional[Union[float, List[float], np.ndarray]],
    length: int
) -> Optional[np.ndarray]:
    """
    Convert a parameter that could be a single value, list, or array into a numpy array.
    
    Args:
        param: The parameter value to normalize
        length: Expected length for multi-agent case
        
    Returns:
        Normalized numpy array or None if param is None
    """
    if param is None:
        return None
        
    if isinstance(param, (int, float)):
        return np.full(length, param)
    
    if isinstance(param, list):
        return np.array(param)
        
    return param  # Already a numpy array


def create_navigator_from_params(
    positions: Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray],
    orientations: Optional[Union[float, List[float], np.ndarray]] = None,
    speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    max_speeds: Optional[Union[float, List[float], np.ndarray]] = None
) -> Union[Navigator, VectorizedNavigator]:
    """
    Create a navigator based on provided parameters.
    
    Args:
        positions: Initial position(s) of the navigator(s)
        orientations: Initial orientation(s) in degrees
        speeds: Initial speed(s)
        max_speeds: Maximum speed(s)
        
    Returns:
        A Navigator or VectorizedNavigator instance
    """
    is_multi_agent = isinstance(positions, (list, np.ndarray)) and len(positions) > 1
    
    if is_multi_agent:
        # Convert positions to numpy array if it's a list
        if isinstance(positions, list):
            positions = np.array(positions)
            
        num_agents = len(positions)
        
        # Normalize all parameters to numpy arrays with the correct length
        orientations = normalize_array_parameter(orientations, num_agents)
        speeds = normalize_array_parameter(speeds, num_agents)
        max_speeds = normalize_array_parameter(max_speeds, num_agents)
        
        # Create a VectorizedNavigator with multi-agent parameters
        return VectorizedNavigator(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds
        )
    else:
        # Single agent case
        return Navigator(
            position=positions,
            orientation=orientations,
            speed=speeds,
            max_speed=max_speeds
        )
