"""
Factory module for creating navigators with configuration.

This module provides functions to create navigator instances based on configuration settings.
"""

from typing import Optional, Union
from pathlib import Path

from odor_plume_nav.navigator import SimpleNavigator
from odor_plume_nav.config_utils import load_config


def create_navigator_from_config(config_path: Optional[Union[str, Path]] = None, **kwargs) -> SimpleNavigator:
    """
    Create a SimpleNavigator instance based on configuration settings.
    
    Args:
        config_path: Optional path to a configuration file.
                    If None, the default configuration will be used.
        **kwargs: Additional parameters to override configuration settings.
    
    Returns:
        A SimpleNavigator instance configured according to the provided settings.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Extract navigator configuration section
    navigator_config = config.get("navigator", {})
    
    # Set parameters from configuration with defaults
    params = {
        "orientation": navigator_config.get("orientation", 0.0),
        "speed": navigator_config.get("speed", 0.0),
        "max_speed": navigator_config.get("max_speed", 1.0)
    }
    
    # Override with any explicitly provided parameters
    params.update(kwargs)
    
    # Create and return the navigator
    return SimpleNavigator(**params)
