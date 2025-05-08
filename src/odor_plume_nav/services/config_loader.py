"""
Configuration utilities for odor plume navigation.

This module provides utility functions for loading and managing configurations.
"""

import os
from typing import Dict, Union, Optional, Any, List
import pathlib
import yaml
import json
import copy
from pathlib import Path

from odor_plume_nav.config.validator import validate_config


def get_config_dir() -> Path:
    """
    Get the path to the configuration directory.
    
    Returns:
        Path to the configuration directory
    """
    # Look for configs in the standard location
    # Start with the package root (one level up from this file)
    package_root = Path(__file__).parent.parent.parent.parent
    config_dir = package_root / "configs"
    
    # Check if environment variable is set to override the config directory
    if env_config_dir := os.environ.get("ODOR_PLUME_NAV_CONFIG_DIR"):
        config_dir = Path(env_config_dir)
        
    return config_dir


def load_yaml_config(config_path: Union[str, Path]) -> Dict:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_config(
    user_config_path: Optional[Union[str, Path]] = None,
    validate: bool = False,
    validate_sections: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Load and merge configuration files.
    
    Loads the default configuration and merges it with a user-provided configuration
    if specified.
    
    Args:
        user_config_path: Path to a user configuration file (optional)
        validate: Whether to validate the configuration after loading
        validate_sections: List of sections to validate (None for all)
        
    Returns:
        Dictionary containing the merged configuration
        
    Raises:
        FileNotFoundError: If any of the configuration files don't exist
        yaml.YAMLError: If any of the YAML files are invalid
        ConfigValidationError: If validation is enabled and the configuration is invalid
    """
    config_dir = get_config_dir()
    default_config_path = config_dir / "default.yaml"
    
    # Load default configuration
    config = load_yaml_config(default_config_path)
    
    # If user config is provided, merge it with the default config
    if user_config_path:
        user_config = load_file_config(user_config_path)
        config = update_config(config, user_config)
    
    # Validate the configuration if requested
    if validate:
        validate_config(config, required_sections=validate_sections)
        
    return config


def load_file_config(file_path: Union[str, pathlib.Path]) -> Dict:
    """
    Load configuration from a file with support for different formats.
    
    Args:
        file_path: Path to configuration file (yaml or json)
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If file extension is not supported
        FileNotFoundError: If file does not exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix in {'.yaml', '.yml'}:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
    elif suffix == '.json':
        with open(file_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Use .yaml, .yml, or .json")

    return config


def save_config(data: Dict, file_path: Union[str, pathlib.Path]) -> None:
    """
    Save configuration to a file.
    
    Args:
        data: Configuration dictionary
        file_path: Path to save the configuration (yaml or json)
        
    Raises:
        ValueError: If file extension is not supported
    """
    file_path = Path(file_path)
    
    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = file_path.suffix.lower()

    if suffix in {'.yaml', '.yml'}:
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    elif suffix == '.json':
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Use .yaml, .yml, or .json")


def update_config(base_config: Dict, updates: Dict) -> Dict:
    """
    Deep update a configuration dictionary with another dictionary.
    
    Args:
        base_config: Base configuration dictionary
        updates: Dictionary with updates
        
    Returns:
        Updated configuration dictionary
    """
    result = copy.deepcopy(base_config)
    
    def _deep_update(original, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                _deep_update(original[key], value)
            else:
                original[key] = value
    
    _deep_update(result, updates)
    return result


__all__ = [
    "load_config",
    "load_file_config",
    "save_config",
    "validate_config",
    "update_config",
    "get_config_dir",
]
