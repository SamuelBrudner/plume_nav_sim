"""
Configuration utilities for odor plume navigation.

This module provides comprehensive configuration management utilities including
environment variable interpolation, validation, and hierarchical configuration
composition for the unified odor_plume_nav package. Enhanced to support 
dataclass-based schema validation and composition per Hydra structured configs.
"""

import os
import re
import inspect
from dataclasses import dataclass, fields, Field, is_dataclass
from typing import Any, Union, Optional, Type, Dict, List, get_type_hints, get_origin, get_args
from pathlib import Path

from odor_plume_nav.services.config_loader import (
    get_config_dir,
    load_yaml_config as _load_yaml_config,
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


def load_config(
    user_config_path: Optional[Union[str, Path]] = None,
    validate: bool = False,
    validate_sections: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load and optionally validate configuration files.

    This wrapper ensures that the config directory can be patched during tests
    by delegating directory resolution to :func:`get_config_dir` from this
    module rather than the copy inside ``services.config_loader``.
    """
    config_dir = get_config_dir()
    default_path = config_dir / "default.yaml"
    config = load_yaml_config(default_path)

    if user_config_path:
        user_cfg = load_file_config(user_config_path)
        config = update_config(config, user_cfg)

    if validate:
        validate_config(config, required_sections=validate_sections)

    return config


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file with basic path security checks."""
    path = Path(config_path)
    real_path = path.resolve()
    if not real_path.is_file() or not real_path.is_relative_to(Path.cwd()):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    return _load_yaml_config(real_path)


def validate_env_interpolation(value: str, field_metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Validate OmegaConf environment variable interpolation syntax with dataclass field metadata support.
    
    Validates that the string follows the correct ${oc.env:VAR_NAME} or
    ${oc.env:VAR_NAME,default_value} pattern for Hydra configuration. Enhanced
    to consider dataclass field constraints for validation.
    
    Args:
        value: String to validate for environment variable interpolation syntax
        field_metadata: Optional dataclass field metadata for enhanced validation
        
    Returns:
        True if the syntax is valid, False otherwise
        
    Examples:
        >>> validate_env_interpolation("${oc.env:HOME}")
        True
        >>> validate_env_interpolation("${oc.env:MISSING_VAR,/default/path}")
        True
        >>> validate_env_interpolation("invalid_syntax")
        False
        >>> # With field metadata
        >>> metadata = {"ge": 0.0, "le": 360.0}
        >>> validate_env_interpolation("${oc.env:ORIENTATION,90.0}", metadata)
        True
    """
    if not isinstance(value, str):
        return False
    
    # Pattern for ${oc.env:VAR_NAME} or ${oc.env:VAR_NAME,default_value}
    pattern = r'^\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(,([^}]+))?\}$'
    match = re.match(pattern, value)
    
    if not match:
        return False
    
    # If field metadata is provided, validate the default value against constraints
    if field_metadata and match.group(2):
        default_value = match.group(2)[1:]  # Remove leading comma
        try:
            # Convert and validate default value against field constraints
            converted_value = _convert_env_value(default_value)
            _validate_field_constraints(converted_value, field_metadata)
        except (ValueError, TypeError):
            return False
    
    return True


def resolve_env_value(
    interpolation_str: str,
    field_type: Optional[Type] = None,
    field_metadata: Optional[Dict[str, Any]] = None
) -> Any:
    """Resolve environment variable interpolation returning raw strings."""
    if not validate_env_interpolation(interpolation_str, field_metadata):
        raise ValueError(f"Invalid environment variable interpolation syntax: {interpolation_str}")

    pattern = r'^\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(,(.*)?)?\}$'
    match = re.match(pattern, interpolation_str)
    if not match:
        raise ValueError(f"Failed to parse environment variable interpolation: {interpolation_str}")

    var_name = match.group(1)
    has_default = match.group(2) is not None
    default_value = match.group(3) if has_default else None

    env_value = os.environ.get(var_name, default_value)
    if env_value is None:
        raise KeyError(f"Environment variable '{var_name}' not found and no default value provided")

    return str(env_value)


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


def _convert_env_value_with_type(value: str, field_type: Optional[Type] = None) -> Any:
    """
    Convert environment variable string to specified type with enhanced type handling.
    
    Args:
        value: String value from environment variable
        field_type: Optional target type for conversion
        
    Returns:
        Converted value according to field_type or best-guess type
    """
    if value is None:
        return None
    
    # If no specific type provided, use basic conversion
    if field_type is None:
        return _convert_env_value(value)
    
    # Handle Optional types (Union[T, None])
    if get_origin(field_type) is Union:
        args = get_args(field_type)
        if len(args) == 2 and type(None) in args:
            # This is Optional[T], extract the non-None type
            field_type = next(arg for arg in args if arg is not type(None))
    
    # Handle specific type conversions
    try:
        if field_type == bool:
            return value.lower() in ('true', 'yes', '1', 'on')
        elif field_type == int:
            return int(value)
        elif field_type == float:
            return float(value)
        elif field_type == str:
            return value
        elif field_type == Path:
            return Path(value)
        elif get_origin(field_type) is list:
            # Handle List[T] types by parsing comma-separated values
            inner_type = get_args(field_type)[0] if get_args(field_type) else str
            items = [item.strip() for item in value.split(',')]
            return [_convert_env_value_with_type(item, inner_type) for item in items]
        elif get_origin(field_type) is tuple:
            # Handle Tuple[T, ...] types
            inner_types = get_args(field_type)
            items = [item.strip() for item in value.split(',')]
            if len(inner_types) == 2 and inner_types[1] is ...:
                # Variable length tuple Tuple[T, ...]
                inner_type = inner_types[0]
                return tuple(_convert_env_value_with_type(item, inner_type) for item in items)
            else:
                # Fixed length tuple
                if len(items) != len(inner_types):
                    raise ValueError(f"Expected {len(inner_types)} items for tuple, got {len(items)}")
                return tuple(_convert_env_value_with_type(item, t) for item, t in zip(items, inner_types))
        else:
            # Fallback to basic conversion for unknown types
            return _convert_env_value(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert '{value}' to type {field_type}: {e}")


def _validate_field_constraints(value: Any, field_metadata: Dict[str, Any]) -> None:
    """
    Validate a value against dataclass field constraints.
    
    Args:
        value: Value to validate
        field_metadata: Field metadata containing validation constraints
        
    Raises:
        ValueError: If value doesn't meet constraints
    """
    if value is None:
        return
    
    # Check numeric constraints
    if 'ge' in field_metadata and hasattr(value, '__ge__'):
        if not (value >= field_metadata['ge']):
            raise ValueError(f"Value {value} must be >= {field_metadata['ge']}")
    
    if 'gt' in field_metadata and hasattr(value, '__gt__'):
        if not (value > field_metadata['gt']):
            raise ValueError(f"Value {value} must be > {field_metadata['gt']}")
    
    if 'le' in field_metadata and hasattr(value, '__le__'):
        if not (value <= field_metadata['le']):
            raise ValueError(f"Value {value} must be <= {field_metadata['le']}")
    
    if 'lt' in field_metadata and hasattr(value, '__lt__'):
        if not (value < field_metadata['lt']):
            raise ValueError(f"Value {value} must be < {field_metadata['lt']}")
    
    # Check string constraints
    if 'min_length' in field_metadata and hasattr(value, '__len__'):
        if len(value) < field_metadata['min_length']:
            raise ValueError(f"Value length {len(value)} must be >= {field_metadata['min_length']}")
    
    if 'max_length' in field_metadata and hasattr(value, '__len__'):
        if len(value) > field_metadata['max_length']:
            raise ValueError(f"Value length {len(value)} must be <= {field_metadata['max_length']}")
    
    # Check pattern constraints
    if 'pattern' in field_metadata and isinstance(value, str):
        if not re.match(field_metadata['pattern'], value):
            raise ValueError(f"Value '{value}' doesn't match pattern '{field_metadata['pattern']}'")


# Dataclass schema introspection utilities

def get_dataclass_fields_info(dataclass_type: Type) -> Dict[str, Dict[str, Any]]:
    """
    Extract field information from a dataclass for configuration validation.
    
    Args:
        dataclass_type: Dataclass type to introspect
        
    Returns:
        Dictionary mapping field names to field information including type and metadata
        
    Raises:
        ValueError: If provided type is not a dataclass
        
    Examples:
        >>> @dataclass
        ... class Config:
        ...     speed: float = field(metadata={"ge": 0.0})
        >>> info = get_dataclass_fields_info(Config)
        >>> info['speed']['type']
        <class 'float'>
        >>> info['speed']['metadata']['ge']
        0.0
    """
    if not is_dataclass(dataclass_type):
        raise ValueError(f"Type {dataclass_type} is not a dataclass")
    
    field_info = {}
    type_hints = get_type_hints(dataclass_type)
    
    for field_obj in fields(dataclass_type):
        field_info[field_obj.name] = {
            'type': type_hints.get(field_obj.name, Any),
            'default': field_obj.default if field_obj.default is not Field else None,
            'default_factory': field_obj.default_factory if field_obj.default_factory is not Field else None,
            'metadata': field_obj.metadata,
            'required': field_obj.default is Field and field_obj.default_factory is Field
        }
    
    return field_info


def validate_dataclass_env_interpolation(
    dataclass_type: Type, 
    config_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate and resolve environment variable interpolations in a configuration dictionary
    against a dataclass schema.
    
    Args:
        dataclass_type: Dataclass type defining the schema
        config_dict: Configuration dictionary with potential environment interpolations
        
    Returns:
        Dictionary with resolved environment variables and validated values
        
    Raises:
        ValueError: If validation fails or interpolation is invalid
        
    Examples:
        >>> @dataclass
        ... class Config:
        ...     speed: float = field(metadata={"ge": 0.0, "le": 2.0})
        >>> config = {"speed": "${oc.env:AGENT_SPEED,1.0}"}
        >>> validated = validate_dataclass_env_interpolation(Config, config)
    """
    if not is_dataclass(dataclass_type):
        raise ValueError(f"Type {dataclass_type} is not a dataclass")
    
    field_info = get_dataclass_fields_info(dataclass_type)
    resolved_config = {}
    
    for key, value in config_dict.items():
        if key not in field_info:
            # Pass through unknown fields
            resolved_config[key] = value
            continue
        
        field_data = field_info[key]
        
        if isinstance(value, str) and value.startswith('${oc.env:'):
            # Resolve environment variable interpolation
            try:
                resolved_value = resolve_env_value(
                    value,
                    field_type=field_data['type'],
                    field_metadata=field_data['metadata']
                )
                resolved_config[key] = resolved_value
            except (ValueError, KeyError) as e:
                raise ValueError(f"Failed to resolve environment variable for field '{key}': {e}")
        else:
            # Validate non-interpolated values against field constraints
            if field_data['metadata']:
                try:
                    _validate_field_constraints(value, field_data['metadata'])
                except ValueError as e:
                    raise ValueError(f"Validation failed for field '{key}': {e}")
            resolved_config[key] = value
    
    return resolved_config


def create_dataclass_instance_with_env_resolution(
    dataclass_type: Type, 
    config_dict: Dict[str, Any]
) -> Any:
    """
    Create a dataclass instance with environment variable resolution and validation.
    
    Args:
        dataclass_type: Dataclass type to instantiate
        config_dict: Configuration dictionary with potential environment interpolations
        
    Returns:
        Instantiated and validated dataclass instance
        
    Raises:
        ValueError: If validation fails or instantiation fails
        
    Examples:
        >>> @dataclass
        ... class Config:
        ...     speed: float = field(default=1.0, metadata={"ge": 0.0})
        >>> config = {"speed": "${oc.env:AGENT_SPEED,1.5}"}
        >>> instance = create_dataclass_instance_with_env_resolution(Config, config)
        >>> instance.speed
        1.5
    """
    # Validate and resolve environment variables
    resolved_config = validate_dataclass_env_interpolation(dataclass_type, config_dict)
    
    # Create instance with resolved values
    try:
        return dataclass_type(**resolved_config)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to create {dataclass_type.__name__} instance: {e}")


def extract_env_vars_from_dataclass(dataclass_type: Type) -> List[str]:
    """
    Extract all potential environment variable names from dataclass field descriptions.
    
    Args:
        dataclass_type: Dataclass type to analyze
        
    Returns:
        List of environment variable names found in field descriptions
        
    Examples:
        >>> @dataclass
        ... class Config:
        ...     speed: float = field(description="Speed value. Supports ${oc.env:AGENT_SPEED,1.0}")
        >>> env_vars = extract_env_vars_from_dataclass(Config)
        >>> 'AGENT_SPEED' in env_vars
        True
    """
    if not is_dataclass(dataclass_type):
        raise ValueError(f"Type {dataclass_type} is not a dataclass")
    
    env_vars = []
    pattern = r'\$\{oc\.env:([A-Z_][A-Z0-9_]*)'
    
    for field_obj in fields(dataclass_type):
        description = field_obj.metadata.get('description', '')
        if isinstance(description, str):
            matches = re.findall(pattern, description)
            env_vars.extend(matches)
    
    return list(set(env_vars))  # Remove duplicates


def generate_env_file_template(dataclass_type: Type) -> str:
    """
    Generate a .env file template from dataclass field metadata.
    
    Args:
        dataclass_type: Dataclass type to generate template from
        
    Returns:
        String content for .env file template
        
    Examples:
        >>> @dataclass
        ... class Config:
        ...     speed: float = field(default=1.0, metadata={"description": "Agent speed ${oc.env:AGENT_SPEED,1.0}"})
        >>> template = generate_env_file_template(Config)
        >>> "AGENT_SPEED=" in template
        True
    """
    if not is_dataclass(dataclass_type):
        raise ValueError(f"Type {dataclass_type} is not a dataclass")
    
    env_vars = extract_env_vars_from_dataclass(dataclass_type)
    field_info = get_dataclass_fields_info(dataclass_type)
    
    template_lines = [
        f"# Environment variables for {dataclass_type.__name__}",
        f"# Generated automatically from dataclass schema",
        ""
    ]
    
    for env_var in sorted(env_vars):
        # Find field that references this env var
        field_name = None
        for name, info in field_info.items():
            description = info['metadata'].get('description', '')
            if env_var in description:
                field_name = name
                break
        
        if field_name:
            field_data = field_info[field_name]
            template_lines.append(f"# Field: {field_name} (type: {field_data['type'].__name__})")
            if field_data['metadata'].get('description'):
                template_lines.append(f"# {field_data['metadata']['description']}")
            if field_data['default'] is not None:
                template_lines.append(f"{env_var}={field_data['default']}")
            else:
                template_lines.append(f"# {env_var}=")
            template_lines.append("")
    
    return "\n".join(template_lines)


__all__ = [
    # Legacy configuration utilities (maintained for backward compatibility)
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
    # Enhanced environment variable interpolation with dataclass support
    "validate_env_interpolation",
    "resolve_env_value",
    # Dataclass schema introspection and validation utilities
    "get_dataclass_fields_info",
    "validate_dataclass_env_interpolation",
    "create_dataclass_instance_with_env_resolution",
    "extract_env_vars_from_dataclass",
    "generate_env_file_template",
]
