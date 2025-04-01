"""
Configuration utilities for the odor_plume_nav package.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union, Any, List
import yaml
from copy import deepcopy

from odor_plume_nav.config_validator import validate_config


def deep_update(original: Dict, update: Dict) -> Dict:
    """
    Recursively update a nested dictionary.
    
    Args:
        original: The original dictionary to update
        update: Dictionary with values to update original with
        
    Returns:
        Updated dictionary
    """
    result = deepcopy(original)
    
    for key, value in update.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
            
    return result


def get_config_dir() -> Path:
    """
    Get the path to the configuration directory.
    
    Returns:
        Path to the configuration directory
    """
    # Look for configs in the standard location
    # Start with the package root (one level up from this file)
    package_root = Path(__file__).parent.parent.parent
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
        user_config = load_yaml_config(user_config_path)
        config = deep_update(config, user_config)
    
    # Validate the configuration if requested
    if validate:
        validate_config(config, required_sections=validate_sections)
        
    return config
