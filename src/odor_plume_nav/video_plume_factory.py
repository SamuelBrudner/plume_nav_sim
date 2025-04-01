"""
Factory functions for creating VideoPlume instances from configuration.
"""

from pathlib import Path
from typing import Dict, Optional, Union, Any

from odor_plume_nav.video_plume import VideoPlume
from odor_plume_nav.config_utils import load_config


def create_video_plume_from_config(
    video_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> VideoPlume:
    """
    Create a VideoPlume instance using configuration settings.
    
    This function loads configuration from the default config file and any user-provided
    config file, then creates a VideoPlume with those settings. Any explicitly provided
    kwargs will override the config settings.
    
    Args:
        video_path: Path to the video file
        config_path: Optional path to a user configuration file
        **kwargs: Additional arguments to pass to VideoPlume constructor,
                 will override any values from config
    
    Returns:
        Configured VideoPlume instance
    """
    # Load configuration (merges default with user config if provided)
    config = load_config(config_path)

    # Extract VideoPlume settings from config
    video_plume_config = config.get("video_plume", {})

    params = {
        "flip": video_plume_config.get("flip", False),
        "kernel_size": video_plume_config.get("kernel_size", 0),
        "kernel_sigma": video_plume_config.get("kernel_sigma", 1.0),
    } | kwargs
    # Create and return the VideoPlume instance
    return VideoPlume(video_path, **params)
