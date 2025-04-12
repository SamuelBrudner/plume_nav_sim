"""
Factory module for creating navigators with configuration.

This module provides functions to create navigator instances based on configuration settings.
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path

from odor_plume_nav.core.navigator import Navigator
from odor_plume_nav.config.utils import load_config


def create_navigator_from_config(config_path: Optional[Union[str, Path]] = None, **kwargs) -> Navigator:
    """
    Create a Navigator instance based on configuration settings.
    
    Args:
        config_path: Optional path to a configuration file.
                    If None, the default configuration will be used.
        **kwargs: Additional parameters to override configuration settings.
    
    Returns:
        A Navigator instance configured according to the provided settings.
    """
    # Load configuration
    config = load_config(config_path)

    # Extract navigator configuration section
    navigator_config = config.get("navigator", {})

    # Check if we need to create a multi-agent navigator
    is_multi_agent = "positions" in navigator_config

    # Set parameters from configuration with defaults
    params: Dict[str, Any] = {}

    if is_multi_agent:
        # Multi-agent parameters
        params["positions"] = navigator_config.get("positions", None)
        params["orientations"] = navigator_config.get("orientations", None)
        params["speeds"] = navigator_config.get("speeds", None)
        params["max_speeds"] = navigator_config.get("max_speeds", None)
        params["angular_velocities"] = navigator_config.get("angular_velocities", None)
        params["num_agents"] = len(params["positions"]) if params["positions"] is not None else 1
    else:
        # Single-agent parameters
        params["position"] = navigator_config.get("position", (0.0, 0.0))
        params["orientation"] = navigator_config.get("orientation", 0.0)
        params["speed"] = navigator_config.get("speed", 0.0)
        params["max_speed"] = navigator_config.get("max_speed", 1.0)
        params["angular_velocity"] = navigator_config.get("angular_velocity", 0.0)

    # Override with any explicitly provided parameters
    params |= kwargs

    # Create and return the navigator using the appropriate factory method
    return Navigator.from_config(params)
