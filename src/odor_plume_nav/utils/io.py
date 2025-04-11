"""
IO utilities for odor plume navigation.

This module provides utility functions for input/output operations.
"""

from typing import Dict, Any, Union, Optional
import pathlib
import yaml
import json
import numpy as np

# Import from the existing io_utils module for now
# This will be refactored in a future update
try:
    from odor_plume_nav.io_utils import (
        load_yaml,
        save_yaml,
        load_json,
        save_json,
        load_numpy,
        save_numpy,
    )
    imported_io_utils = True
except ImportError:
    imported_io_utils = False


# Define functions if not imported
if not imported_io_utils:
    def load_yaml(file_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
        """
        Load a YAML file and return its contents as a dictionary.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Dictionary with the YAML contents
        """
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def save_yaml(data: Dict[str, Any], file_path: Union[str, pathlib.Path]) -> None:
        """
        Save a dictionary to a YAML file.
        
        Args:
            data: Dictionary to save
            file_path: Path to the output YAML file
        """
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def load_json(file_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
        """
        Load a JSON file and return its contents as a dictionary.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with the JSON contents
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def save_json(data: Dict[str, Any], file_path: Union[str, pathlib.Path]) -> None:
        """
        Save a dictionary to a JSON file.
        
        Args:
            data: Dictionary to save
            file_path: Path to the output JSON file
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_numpy(file_path: Union[str, pathlib.Path]) -> np.ndarray:
        """
        Load a NumPy array from a .npy file.
        
        Args:
            file_path: Path to the .npy file
            
        Returns:
            NumPy array
        """
        return np.load(file_path)
    
    def save_numpy(data: np.ndarray, file_path: Union[str, pathlib.Path]) -> None:
        """
        Save a NumPy array to a .npy file.
        
        Args:
            data: NumPy array to save
            file_path: Path to the output .npy file
        """
        np.save(file_path, data)
