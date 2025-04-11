"""Tests for the configuration utilities."""

import pytest
import tempfile
from pathlib import Path
import yaml
import json
from unittest.mock import patch, MagicMock

# Import from the new config module
from odor_plume_nav.config import load_config, save_config, validate_config, update_config


@pytest.fixture
def temp_yaml_file():
    """Create a temporary YAML file for testing."""
    test_config = {
        "navigator": {
            "positions": [[10, 20], [30, 40]],
            "orientations": [45, 90],
            "speeds": [0.5, 0.7],
            "orientation": 45,
            "speed": 0.5,
            "max_speed": 1.0
        },
        "video_plume": {
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.0
        }
    }
    
    # Create file with write text mode (not binary)
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode='w') as temp:
        yaml.dump(test_config, temp)
        temp_path = temp.name
    
    yield temp_path
    
    # Clean up the temporary file
    Path(temp_path).unlink()


@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file for testing."""
    test_config = {
        "navigator": {
            "positions": [[10, 20], [30, 40]],
            "orientations": [45, 90],
            "speeds": [0.5, 0.7],
            "orientation": 45,
            "speed": 0.5,
            "max_speed": 1.0
        },
        "video_plume": {
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.0
        }
    }
    
    # Create file with write text mode (not binary)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as temp:
        json.dump(test_config, temp)
        temp_path = temp.name
    
    yield temp_path
    
    # Clean up the temporary file
    Path(temp_path).unlink()


def test_load_config_yaml(temp_yaml_file):
    """Test loading configuration from a YAML file."""
    # Load the configuration
    config = load_config(temp_yaml_file)
    
    # Check that the configuration was loaded correctly
    assert "navigator" in config
    assert "video_plume" in config
    assert config["navigator"]["orientations"] == [45, 90]
    assert config["video_plume"]["flip"] is True


def test_load_config_json(temp_json_file):
    """Test loading configuration from a JSON file."""
    # Load the configuration
    config = load_config(temp_json_file)
    
    # Check that the configuration was loaded correctly
    assert "navigator" in config
    assert "video_plume" in config
    assert config["navigator"]["orientations"] == [45, 90]
    assert config["video_plume"]["flip"] is True


def test_load_config_unsupported_extension():
    """Test that loading configuration from an unsupported file type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_config("config.txt")


def test_load_config_file_not_found():
    """Test that loading configuration from a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


def test_save_config_yaml():
    """Test saving configuration to a YAML file."""
    test_config = {
        "navigator": {
            "positions": [[10, 20], [30, 40]],
            "orientations": [45, 90],
            "speeds": [0.5, 0.7],
            "orientation": 45,
            "speed": 0.5,
            "max_speed": 1.0
        },
        "video_plume": {
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.0
        }
    }
    
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode='w') as temp:
        temp_path = temp.name
    
    try:
        # Save the configuration
        save_config(test_config, temp_path)
        
        # Load the configuration to verify it was saved correctly
        with open(temp_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Check that the configuration was saved correctly
        assert loaded_config == test_config
    finally:
        # Clean up the temporary file
        Path(temp_path).unlink()


def test_save_config_json():
    """Test saving configuration to a JSON file."""
    test_config = {
        "navigator": {
            "positions": [[10, 20], [30, 40]],
            "orientations": [45, 90],
            "speeds": [0.5, 0.7],
            "orientation": 45,
            "speed": 0.5,
            "max_speed": 1.0
        },
        "video_plume": {
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.0
        }
    }
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as temp:
        temp_path = temp.name
    
    try:
        # Save the configuration
        save_config(test_config, temp_path)
        
        # Load the configuration to verify it was saved correctly
        with open(temp_path, 'r') as f:
            loaded_config = json.load(f)
        
        # Check that the configuration was saved correctly
        assert loaded_config == test_config
    finally:
        # Clean up the temporary file
        Path(temp_path).unlink()


def test_save_config_unsupported_extension():
    """Test that saving configuration to an unsupported file type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported file extension"):
        save_config({}, "config.txt")


def test_validate_config():
    """Test validating a configuration."""
    # This is a basic test - more comprehensive validation tests are in test_config_validator.py
    valid_config = {
        "navigator": {
            "positions": [[10, 20], [30, 40]],
            "orientations": [45, 90],
            "speeds": [0.5, 0.7],
            "orientation": 45,
            "speed": 0.5,
            "max_speed": 1.0
        },
        "video_plume": {
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.0
        }
    }
    
    # Validation should pass for valid config
    validate_config(valid_config)
    
    # Test with invalid config (missing required fields)
    invalid_config = {
        "navigator": {
            "positions": [[10, 20], [30, 40]],
            # Missing orientations
            "speeds": [0.5, 0.7]
        }
    }
    
    # Should raise an exception for invalid config
    with pytest.raises(Exception):
        validate_config(invalid_config)


def test_update_config():
    """Test updating a configuration."""
    base_config = {
        "navigator": {
            "positions": [[10, 20], [30, 40]],
            "orientations": [45, 90],
            "speeds": [0.5, 0.7],
            "orientation": 45,
            "speed": 0.5,
            "max_speed": 1.0
        },
        "video_plume": {
            "flip": False,
            "kernel_size": 3,
            "kernel_sigma": 1.0
        }
    }
    
    updates = {
        "navigator": {
            "speeds": [0.8, 1.0]  # Update speeds
        },
        "video_plume": {
            "flip": True  # Update flip
        }
    }
    
    # Update the configuration
    updated_config = update_config(base_config, updates)
    
    # Check that the configuration was updated correctly
    assert updated_config["navigator"]["speeds"] == [0.8, 1.0]  # Updated
    assert updated_config["navigator"]["positions"] == [[10, 20], [30, 40]]  # Unchanged
    assert updated_config["navigator"]["orientations"] == [45, 90]  # Unchanged
    assert updated_config["video_plume"]["flip"] is True  # Updated
    assert updated_config["video_plume"]["kernel_size"] == 3  # Unchanged
    
    # Test that the original configuration was not modified
    assert base_config["navigator"]["speeds"] == [0.5, 0.7]
    assert base_config["video_plume"]["flip"] is False
