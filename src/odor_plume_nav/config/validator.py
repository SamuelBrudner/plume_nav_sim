"""Configuration validation for odor_plume_nav.

This module provides functions to validate configuration dictionaries
and ensure they have the required structure and values.
"""

from typing import Dict, Any, List, Optional


class ConfigValidationError(Exception):
    """Exception raised when a configuration validation fails."""
    pass


def validate_video_plume_config(config: Dict[str, Any]) -> None:
    """Validate video plume configuration section.
    
    Args:
        config: Dictionary containing video plume configuration
        
    Raises:
        ConfigValidationError: If validation fails
    """
    if 'video_plume' not in config:
        raise ConfigValidationError("Missing required 'video_plume' section in configuration")
    
    video_plume = config['video_plume']
    
    # Check required fields
    required_fields = {'flip': bool, 'kernel_size': int, 'kernel_sigma': float}
    
    for field, expected_type in required_fields.items():
        if field not in video_plume:
            raise ConfigValidationError(f"Missing required field '{field}' in video_plume configuration")
        
        value = video_plume[field]
        if not isinstance(value, expected_type):
            raise ConfigValidationError(
                f"Invalid type for video_plume.{field}: expected {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
    
    # Additional constraints
    if video_plume['kernel_size'] < 0:
        raise ConfigValidationError("video_plume.kernel_size must be non-negative")
    
    if video_plume['kernel_sigma'] <= 0:
        raise ConfigValidationError("video_plume.kernel_sigma must be positive")


def validate_navigator_config(config: Dict[str, Any]) -> None:
    """Validate navigator configuration section.
    
    Args:
        config: Dictionary containing navigator configuration
        
    Raises:
        ConfigValidationError: If validation fails
    """
    if 'navigator' not in config:
        raise ConfigValidationError("Missing required 'navigator' section in configuration")
    
    navigator = config['navigator']
    
    # Check required fields
    required_fields = {'orientation': (float, int), 'speed': (float, int), 'max_speed': (float, int)}
    
    for field, expected_types in required_fields.items():
        if field not in navigator:
            raise ConfigValidationError(f"Missing required field '{field}' in navigator configuration")
        
        value = navigator[field]
        if not any(isinstance(value, t) for t in expected_types):
            type_names = " or ".join(t.__name__ for t in expected_types)
            raise ConfigValidationError(
                f"Invalid type for navigator.{field}: expected {type_names}, "
                f"got {type(value).__name__}"
            )
    
    # Additional constraints
    if navigator['max_speed'] < 0:
        raise ConfigValidationError("navigator.max_speed must be non-negative")
    
    if abs(navigator['speed']) > navigator['max_speed']:
        raise ConfigValidationError(f"navigator.speed ({navigator['speed']}) exceeds "
                                    f"max_speed ({navigator['max_speed']})")


def validate_config(config: Dict[str, Any], required_sections: Optional[List[str]] = None) -> None:
    """Validate a complete configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        required_sections: List of sections that must be validated. If None,
                           validates all known sections.
        
    Raises:
        ConfigValidationError: If validation fails
    """
    if not isinstance(config, dict):
        raise ConfigValidationError(f"Configuration must be a dictionary, got {type(config).__name__}")
    
    # Default to all known validators if not specified
    if required_sections is None:
        required_sections = ['video_plume', 'navigator']
    
    # Validation mapping
    validators = {
        'video_plume': validate_video_plume_config,
        'navigator': validate_navigator_config,
    }
    
    # Run validators for required sections
    for section in required_sections:
        if section not in validators:
            raise ConfigValidationError(f"Unknown section '{section}' specified for validation")
        
        validators[section](config)
