"""
Configuration utilities for odor plume navigation.

This module provides comprehensive configuration management utilities including
environment variable interpolation, validation, and hierarchical configuration
composition for the unified odor_plume_nav package.
"""

import os
import re
from typing import Any, Union, Optional

from odor_plume_nav.services.config_loader import (
    get_config_dir,
    load_yaml_config,
    load_config,
    load_file_config,
    save_config,
    update_config,
)
from odor_plume_nav.services.validator import (
    validate_config,
    validate_video_plume_config,
    validate_navigator_config,
    ConfigValidationError,
)


def validate_env_interpolation(value: str) -> bool:
    """
    Validate OmegaConf environment variable interpolation syntax.
    
    Validates that the string follows the correct ${oc.env:VAR_NAME} or
    ${oc.env:VAR_NAME,default_value} pattern for Hydra configuration.
    
    Args:
        value: String to validate for environment variable interpolation syntax
        
    Returns:
        True if the syntax is valid, False otherwise
        
    Examples:
        >>> validate_env_interpolation("${oc.env:HOME}")
        True
        >>> validate_env_interpolation("${oc.env:MISSING_VAR,/default/path}")
        True
        >>> validate_env_interpolation("invalid_syntax")
        False
    """
    if not isinstance(value, str):
        return False
    
    # Pattern for ${oc.env:VAR_NAME} or ${oc.env:VAR_NAME,default_value}
    pattern = r'^\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(,.*?)?\}$'
    return bool(re.match(pattern, value))


def resolve_env_value(interpolation_str: str) -> Any:
    """
    Resolve environment variable from OmegaConf interpolation syntax.
    
    Parses ${oc.env:VAR_NAME} or ${oc.env:VAR_NAME,default_value} syntax
    and returns the environment variable value or default value if specified.
    
    Args:
        interpolation_str: String containing environment variable interpolation
        
    Returns:
        Resolved environment variable value or default value
        
    Raises:
        ValueError: If the interpolation syntax is invalid
        KeyError: If environment variable doesn't exist and no default is provided
        
    Examples:
        >>> os.environ['TEST_VAR'] = 'test_value'
        >>> resolve_env_value("${oc.env:TEST_VAR}")
        'test_value'
        >>> resolve_env_value("${oc.env:MISSING_VAR,default}")
        'default'
    """
    if not validate_env_interpolation(interpolation_str):
        raise ValueError(f"Invalid environment variable interpolation syntax: {interpolation_str}")
    
    # Extract variable name and optional default value
    pattern = r'^\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(,(.*)?)?\}$'
    match = re.match(pattern, interpolation_str)
    
    if not match:
        raise ValueError(f"Failed to parse environment variable interpolation: {interpolation_str}")
    
    var_name = match.group(1)
    has_default = match.group(2) is not None
    default_value = match.group(3) if has_default else None
    
    # Try to get environment variable
    env_value = os.environ.get(var_name)
    
    if env_value is not None:
        # Try to convert to appropriate type (int, float, bool, or string)
        return _convert_env_value(env_value)
    elif has_default:
        # Use default value if provided
        return _convert_env_value(default_value) if default_value is not None else None
    else:
        # No environment variable and no default - raise error
        raise KeyError(f"Environment variable '{var_name}' not found and no default value provided")


def _convert_env_value(value: str) -> Any:
    """
    Convert environment variable string to appropriate Python type.
    
    Args:
        value: String value from environment variable
        
    Returns:
        Converted value (int, float, bool, or original string)
    """
    if value is None:
        return None
    
    # Handle boolean values
    if value.lower() in ('true', 'yes', '1', 'on'):
        return True
    elif value.lower() in ('false', 'no', '0', 'off'):
        return False
    
    # Try to convert to int
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try to convert to float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string if no conversion possible
    return value


__all__ = [
    "get_config_dir",
    "load_yaml_config",
    "load_config",
    "load_file_config",
    "save_config",
    "update_config",
    "validate_config",
    "validate_video_plume_config",
    "validate_navigator_config",
    "ConfigValidationError",
    "validate_env_interpolation",
    "resolve_env_value",
]
