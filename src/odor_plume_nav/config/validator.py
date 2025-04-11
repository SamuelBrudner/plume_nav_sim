"""
Configuration validation for odor plume navigation.

This module provides validation functions for configuration dictionaries.
"""

from typing import Dict, List, Optional, Any, Union


class ConfigValidationError(Exception):
    """Error raised when configuration validation fails."""
    pass


def validate_video_plume_config(config: Dict) -> None:
    """
    Validate the video_plume section of a configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ConfigValidationError: If the configuration is invalid
    """
    # Check if video_plume section exists
    if "video_plume" not in config:
        raise ConfigValidationError("Missing required 'video_plume' section")
    
    video_plume_config = config["video_plume"]
    
    # Check required fields
    required_fields = ["flip", "kernel_size", "kernel_sigma"]
    for field in required_fields:
        if field not in video_plume_config:
            raise ConfigValidationError(f"Missing required field '{field}' in video_plume section")
    
    # Check types
    if not isinstance(video_plume_config["flip"], bool):
        raise ConfigValidationError(f"Invalid type for video_plume.flip: expected bool, got {type(video_plume_config['flip']).__name__}")
    
    if not isinstance(video_plume_config["kernel_size"], int):
        raise ConfigValidationError(f"Invalid type for video_plume.kernel_size: expected int, got {type(video_plume_config['kernel_size']).__name__}")
    
    if not isinstance(video_plume_config["kernel_sigma"], (int, float)):
        raise ConfigValidationError(f"Invalid type for video_plume.kernel_sigma: expected float, got {type(video_plume_config['kernel_sigma']).__name__}")
    
    # Check values
    if video_plume_config["kernel_size"] < 0:
        raise ConfigValidationError("kernel_size must be non-negative")
    
    if video_plume_config["kernel_sigma"] <= 0:
        raise ConfigValidationError("kernel_sigma must be positive")


def validate_navigator_config(config: Dict) -> None:
    """
    Validate the navigator section of a configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ConfigValidationError: If the configuration is invalid
    """
    # Check if navigator section exists
    if "navigator" not in config:
        raise ConfigValidationError("Missing required 'navigator' section")
    
    navigator_config = config["navigator"]
    
    # Check required fields
    required_fields = ["orientation", "speed", "max_speed"]
    for field in required_fields:
        if field not in navigator_config:
            raise ConfigValidationError(f"Missing required field '{field}' in navigator section")
    
    # Check types
    if not isinstance(navigator_config["orientation"], (int, float)):
        raise ConfigValidationError(f"Invalid type for navigator.orientation: expected float, got {type(navigator_config['orientation']).__name__}")
    
    if not isinstance(navigator_config["speed"], (int, float)):
        raise ConfigValidationError(f"Invalid type for navigator.speed: expected float, got {type(navigator_config['speed']).__name__}")
    
    if not isinstance(navigator_config["max_speed"], (int, float)):
        raise ConfigValidationError(f"Invalid type for navigator.max_speed: expected float, got {type(navigator_config['max_speed']).__name__}")
    
    # Check values
    if navigator_config["max_speed"] < 0:
        raise ConfigValidationError("max_speed must be non-negative")
    
    if abs(navigator_config["speed"]) > navigator_config["max_speed"]:
        raise ConfigValidationError(f"speed ({navigator_config['speed']}) exceeds max_speed ({navigator_config['max_speed']})")


def validate_config(config: Dict, required_sections: Optional[List[str]] = None) -> None:
    """
    Validate a complete configuration.
    
    Args:
        config: Configuration dictionary
        required_sections: List of sections to validate (None for all)
        
    Raises:
        ConfigValidationError: If the configuration is invalid
    """
    # Check that config is a dictionary
    if not isinstance(config, dict):
        raise ConfigValidationError(f"Configuration must be a dictionary, got {type(config).__name__}")
    
    # If no sections specified, validate all supported sections
    if required_sections is None:
        required_sections = ["video_plume", "navigator"]
    
    # Validate requested sections
    for section in required_sections:
        if section == "video_plume":
            validate_video_plume_config(config)
        elif section == "navigator":
            validate_navigator_config(config)
        else:
            raise ConfigValidationError(f"Unknown section '{section}'")


__all__ = [
    "validate_config",
    "validate_video_plume_config",
    "validate_navigator_config",
    "ConfigValidationError",
]
