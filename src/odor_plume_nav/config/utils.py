"""
Configuration utilities for odor plume navigation.

This module provides utility functions for loading and managing configurations.
"""

from typing import Dict, Union, Optional, Any
import pathlib
import yaml
import json
import copy
from pathlib import Path

from odor_plume_nav.config.validator import validate_config


def load_config(file_path: Union[str, pathlib.Path]) -> Dict:
    """
    Load configuration from a file.
    
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
    suffix = file_path.suffix.lower()

    if suffix in {'.yaml', '.yml'}:
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
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
    "save_config",
    "validate_config",
    "update_config",
]
