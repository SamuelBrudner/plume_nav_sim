"""
Atomic I/O utilities for plume navigation simulation.

This module provides secure and efficient utility functions for input/output 
operations with comprehensive error handling and pathlib integration. All file 
operations are atomic to ensure data integrity and provide consistent behavior 
across different platforms.

Key features:
- Atomic file operations with temporary file staging
- Comprehensive error handling and logging
- Secure YAML parsing with safe_load/dump
- High-performance NumPy array persistence
- Full pathlib.Path integration
- Type-safe interfaces with comprehensive annotations
"""

import json
import os
import pathlib
import tempfile
import warnings
from typing import Any, Dict, Optional, Union
import os

import numpy as np
import yaml


__all__ = [
    'load_yaml',
    'save_yaml', 
    'load_json',
    'save_json',
    'load_numpy',
    'save_numpy',
    'IOError',
    'YAMLError',
    'JSONError',
    'NumpyError'
]


# Custom exception classes for better error handling
class IOError(Exception):
    """Base exception for I/O operations."""
    pass


class YAMLError(IOError):
    """Exception raised for YAML-related errors."""
    pass


class JSONError(IOError):
    """Exception raised for JSON-related errors."""
    pass


class NumpyError(IOError):
    """Exception raised for NumPy file operation errors."""
    pass


def _ensure_parent_directory(file_path: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Ensure the parent directory exists for the given file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        pathlib.Path object for the file
        
    Raises:
        IOError: If directory creation fails
    """
    path_obj = pathlib.Path(file_path)
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        return path_obj
    except OSError as e:
        raise IOError(f"Failed to create parent directory for {file_path}: {e}") from e


def _atomic_write(content: bytes, file_path: pathlib.Path) -> None:
    """
    Write content to file atomically using temporary file staging.
    
    Args:
        content: Raw bytes to write
        file_path: Destination file path
        
    Raises:
        IOError: If atomic write operation fails
    """
    try:
        # Create temporary file in the same directory to ensure atomic move
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=file_path.parent,
            prefix=f'.{file_path.name}_',
            suffix='.tmp',
            delete=False
        ) as temp_file:
            temp_path = pathlib.Path(temp_file.name)
            temp_file.write(content)
            temp_file.flush()
            # Ensure data is written to disk
            os.fsync(temp_file.fileno())
        
        # Atomic move from temporary file to final destination
        temp_path.replace(file_path)
        
    except OSError as e:
        # Clean up temporary file if it exists
        try:
            temp_path.unlink(missing_ok=True)
        except (NameError, OSError):
            pass
        raise IOError(f"Failed to write file atomically to {file_path}: {e}") from e


def load_yaml(file_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.
    
    Uses PyYAML's safe_load for security, preventing execution of arbitrary
    Python code from YAML files. Provides comprehensive error handling for
    file system and parsing errors.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary with the YAML contents
        
    Raises:
        YAMLError: If file loading or parsing fails
        IOError: If file access fails
    """
    path_obj = pathlib.Path(file_path)
    
    try:
        if not path_obj.exists():
            raise YAMLError(f"YAML file not found: {file_path}")
        
        if not path_obj.is_file():
            raise YAMLError(f"Path is not a file: {file_path}")
        
        with path_obj.open('r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
                
                # Handle empty files gracefully
                if data is None:
                    return {}
                
                # Ensure we return a dictionary
                if not isinstance(data, dict):
                    raise YAMLError(f"YAML file must contain a dictionary, got {type(data).__name__}")
                
                return data
                
            except yaml.YAMLError as e:
                raise YAMLError(f"Failed to parse YAML file {file_path}: {e}") from e
            
    except OSError as e:
        raise IOError(f"Failed to read YAML file {file_path}: {e}") from e


def save_yaml(
    data: Dict[str, Any], 
    file_path: Union[str, pathlib.Path],
    indent: int = 2,
    sort_keys: bool = False
) -> None:
    """
    Save a dictionary to a YAML file atomically.
    
    Uses PyYAML's safe_dump for security and atomic file operations to prevent
    data corruption. Creates parent directories as needed.
    
    Args:
        data: Dictionary to save
        file_path: Path to the output YAML file
        indent: Number of spaces for indentation (default: 2)
        sort_keys: Whether to sort keys alphabetically (default: False)
        
    Raises:
        YAMLError: If data serialization fails
        IOError: If file writing fails
    """
    if not isinstance(data, dict):
        raise YAMLError(f"Data must be a dictionary, got {type(data).__name__}")
    
    path_obj = _ensure_parent_directory(file_path)
    
    try:
        # Serialize to YAML string first to catch any serialization errors
        yaml_content = yaml.safe_dump(
            data,
            default_flow_style=False,
            indent=indent,
            sort_keys=sort_keys,
            allow_unicode=True
        )
        
        # Write atomically
        _atomic_write(yaml_content.encode('utf-8'), path_obj)
        
    except yaml.YAMLError as e:
        raise YAMLError(f"Failed to serialize data to YAML: {e}") from e


def load_json(file_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents as a dictionary.
    
    Provides comprehensive error handling for file system and parsing errors
    with clear error messages for debugging.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with the JSON contents
        
    Raises:
        JSONError: If file loading or parsing fails
        IOError: If file access fails
    """
    path_obj = pathlib.Path(file_path)
    
    try:
        if not path_obj.exists():
            raise JSONError(f"JSON file not found: {file_path}")
        
        if not path_obj.is_file():
            raise JSONError(f"Path is not a file: {file_path}")
        
        with path_obj.open('r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                # Ensure we return a dictionary
                if not isinstance(data, dict):
                    raise JSONError(f"JSON file must contain an object, got {type(data).__name__}")
                
                return data
                
            except json.JSONDecodeError as e:
                raise JSONError(f"Failed to parse JSON file {file_path}: {e}") from e
            
    except OSError as e:
        raise IOError(f"Failed to read JSON file {file_path}: {e}") from e


def save_json(
    data: Dict[str, Any], 
    file_path: Union[str, pathlib.Path],
    indent: Optional[int] = 2,
    sort_keys: bool = False
) -> None:
    """
    Save a dictionary to a JSON file atomically.
    
    Uses atomic file operations to prevent data corruption and creates parent
    directories as needed. Provides configurable formatting options.
    
    Args:
        data: Dictionary to save
        file_path: Path to the output JSON file
        indent: Number of spaces for indentation (None for compact, default: 2)
        sort_keys: Whether to sort keys alphabetically (default: False)
        
    Raises:
        JSONError: If data serialization fails
        IOError: If file writing fails
    """
    if not isinstance(data, dict):
        raise JSONError(f"Data must be a dictionary, got {type(data).__name__}")
    
    path_obj = _ensure_parent_directory(file_path)
    
    try:
        # Serialize to JSON string first to catch any serialization errors
        json_content = json.dumps(
            data,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=False
        )
        
        # Write atomically
        _atomic_write(json_content.encode('utf-8'), path_obj)
        
    except (TypeError, ValueError) as e:
        raise JSONError(f"Failed to serialize data to JSON: {e}") from e


def load_numpy(file_path: Union[str, pathlib.Path]) -> np.ndarray:
    """
    Load a NumPy array from a .npy file.
    
    Provides comprehensive error handling for file system and format errors
    with validation of the loaded data.
    
    Args:
        file_path: Path to the .npy file
        
    Returns:
        NumPy array loaded from the file
        
    Raises:
        NumpyError: If file loading or format validation fails
        IOError: If file access fails
    """
    path_obj = pathlib.Path(file_path)
    
    try:
        if not path_obj.exists():
            raise NumpyError(f"NumPy file not found: {file_path}")
        
        if not path_obj.is_file():
            raise NumpyError(f"Path is not a file: {file_path}")
        
        # Check file extension
        if path_obj.suffix.lower() not in ['.npy', '.npz']:
            warnings.warn(
                f"Loading NumPy data from file without .npy/.npz extension: {file_path}",
                UserWarning,
                stacklevel=2
            )
        
        try:
            array = np.load(path_obj)
            
            # Validate that we got a proper array
            if not isinstance(array, np.ndarray):
                raise NumpyError(f"Loaded data is not a NumPy array, got {type(array).__name__}")
            
            return array
            
        except (ValueError, OSError) as e:
            raise NumpyError(f"Failed to load NumPy array from {file_path}: {e}") from e
            
    except OSError as e:
        raise IOError(f"Failed to access NumPy file {file_path}: {e}") from e


def save_numpy(
    data: np.ndarray, 
    file_path: Union[str, pathlib.Path],
    compress: bool = False
) -> None:
    """
    Save a NumPy array to a .npy file atomically.
    
    Uses atomic file operations to prevent data corruption and validates input
    data before saving. Creates parent directories as needed.
    
    Args:
        data: NumPy array to save
        file_path: Path to the output .npy file
        compress: Whether to use compressed .npz format (default: False)
        
    Raises:
        NumpyError: If data validation or serialization fails
        IOError: If file writing fails
    """
    if not isinstance(data, np.ndarray):
        raise NumpyError(f"Data must be a NumPy array, got {type(data).__name__}")
    
    path_obj = _ensure_parent_directory(file_path)
    
    # Adjust extension based on compression setting
    if compress and not str(path_obj).endswith('.npz'):
        path_obj = path_obj.with_suffix('.npz')
    elif not compress and not str(path_obj).endswith('.npy'):
        path_obj = path_obj.with_suffix('.npy')
    
    try:
        # Create temporary file in the same directory
        with tempfile.NamedTemporaryFile(
            dir=path_obj.parent,
            prefix=f'.{path_obj.name}_',
            suffix='.tmp',
            delete=False
        ) as temp_file:
            temp_path = pathlib.Path(temp_file.name)
            
            # Save to temporary file
            if compress:
                np.savez_compressed(temp_file, data)
            else:
                np.save(temp_file, data)
            
            temp_file.flush()
            # Ensure data is written to disk
            os.fsync(temp_file.fileno())
        
        # Atomic move from temporary file to final destination
        temp_path.replace(path_obj)
        
    except OSError as e:
        # Clean up temporary file if it exists
        try:
            temp_path.unlink(missing_ok=True)
        except (NameError, OSError):
            pass
        raise IOError(f"Failed to write NumPy array to {path_obj}: {e}") from e
    
    except Exception as e:
        # Clean up temporary file if it exists
        try:
            temp_path.unlink(missing_ok=True)
        except (NameError, OSError):
            pass
        raise NumpyError(f"Failed to serialize NumPy array: {e}") from e

