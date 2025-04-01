"""Tests for configuration utilities."""

import pytest
from pathlib import Path
import yaml
import os
from unittest.mock import patch
import tempfile

from odor_plume_nav.config_utils import (
    deep_update,
    get_config_dir,
    load_yaml_config,
    load_config
)
from odor_plume_nav.config_validator import ConfigValidationError


def test_deep_update_basic():
    """Test that deep_update correctly updates flat dictionaries."""
    original = {"a": 1, "b": 2}
    update = {"b": 3, "c": 4}
    result = deep_update(original, update)
    
    assert result == {"a": 1, "b": 3, "c": 4}
    # Original should be unchanged
    assert original == {"a": 1, "b": 2}


def test_deep_update_nested():
    """Test that deep_update correctly updates nested dictionaries."""
    original = {"a": 1, "b": {"x": 10, "y": 20}}
    update = {"b": {"y": 30, "z": 40}}
    result = deep_update(original, update)
    
    assert result == {"a": 1, "b": {"x": 10, "y": 30, "z": 40}}
    # Original should be unchanged
    assert original == {"a": 1, "b": {"x": 10, "y": 20}}


def test_get_config_dir_default():
    """Test that get_config_dir returns the standard config directory."""
    with patch.dict('os.environ', {}, clear=True):
        config_dir = get_config_dir()
        expected_path = Path(__file__).parent.parent / "configs"
        assert config_dir == expected_path


def test_get_config_dir_env_override():
    """Test that get_config_dir respects the environment variable."""
    with patch.dict('os.environ', {"ODOR_PLUME_NAV_CONFIG_DIR": "/custom/config/path"}):
        config_dir = get_config_dir()
        assert config_dir == Path("/custom/config/path")


def test_load_yaml_config():
    """Test that load_yaml_config correctly loads a YAML file."""
    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.yaml') as tmp:
        test_config = {
            "test": {"key": "value"},
            "number": 42
        }
        yaml.dump(test_config, tmp)
        tmp.flush()
        
        # Load and verify the config
        loaded_config = load_yaml_config(tmp.name)
        assert loaded_config == test_config


def test_load_yaml_config_file_not_found():
    """Test that load_yaml_config raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config("/nonexistent/config.yaml")


@pytest.fixture
def mock_config_files(tmp_path):
    """Create temporary default and user config files for testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    
    # Create default config
    default_config = {
        "environment": {
            "dimensions": [10.0, 8.0, 2.0],
            "wind": {"speed": 0.5}
        },
        "video_plume": {"flip": False}
    }
    default_path = config_dir / "default.yaml"
    with open(default_path, 'w') as f:
        yaml.dump(default_config, f)
    
    # Create user config
    user_config = {
        "environment": {
            "wind": {"speed": 0.8}
        },
        "video_plume": {"kernel_size": 5}
    }
    user_path = tmp_path / "user_config.yaml"
    with open(user_path, 'w') as f:
        yaml.dump(user_config, f)
    
    return {
        "config_dir": config_dir,
        "default_path": default_path,
        "user_path": user_path,
        "default_config": default_config,
        "user_config": user_config
    }


def test_load_config_default_only(mock_config_files):
    """Test loading only the default configuration."""
    with patch('odor_plume_nav.config_utils.get_config_dir', 
               return_value=mock_config_files["config_dir"]):
        config = load_config()
        assert config == mock_config_files["default_config"]


def test_load_config_with_user_override(mock_config_files):
    """Test loading and merging default and user configurations."""
    with patch('odor_plume_nav.config_utils.get_config_dir',
               return_value=mock_config_files["config_dir"]):
        config = load_config(user_config_path=mock_config_files["user_path"])
        
        # Expected merged config
        expected = {
            "environment": {
                "dimensions": [10.0, 8.0, 2.0],
                "wind": {"speed": 0.8}  # Overridden by user config
            },
            "video_plume": {
                "flip": False,
                "kernel_size": 5  # Added by user config
            }
        }
        
        assert config == expected


# New tests for configuration validation
@pytest.fixture
def valid_config_files(tmp_path):
    """Create temporary config files with valid configurations."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    
    # Create valid default config
    default_config = {
        "video_plume": {
            "flip": False,
            "kernel_size": 0,
            "kernel_sigma": 1.0
        },
        "navigator": {
            "orientation": 0.0,
            "speed": 0.0,
            "max_speed": 1.0
        }
    }
    default_path = config_dir / "default.yaml"
    with open(default_path, 'w') as f:
        yaml.dump(default_config, f)
    
    # Create valid user config
    user_config = {
        "video_plume": {
            "flip": True,
            "kernel_size": 3
        }
    }
    user_path = tmp_path / "valid_user.yaml"
    with open(user_path, 'w') as f:
        yaml.dump(user_config, f)
    
    return {
        "config_dir": config_dir,
        "default_path": default_path,
        "user_path": user_path,
        "default_config": default_config,
        "user_config": user_config
    }


@pytest.fixture
def invalid_config_files(tmp_path):
    """Create temporary config files with invalid configurations."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    
    # Test-specific configs
    test_configs = {
        # Base default config with missing fields for most tests
        "default": {
            "video_plume": {
                # Missing some required fields
                "flip": False
            },
            "navigator": {
                # Missing some required fields
                "orientation": 0.0
            }
        },
        # Complete default config for speed validation test
        "speed_test_default": {
            "video_plume": {
                "flip": False,
                "kernel_size": 0,
                "kernel_sigma": 1.0
            },
            "navigator": {
                "orientation": 0.0,
                "speed": 0.0,
                "max_speed": 1.0
            }
        }
    }
    
    # Write config files based on test needs
    config_paths = {}
    for name, config in test_configs.items():
        path = config_dir / f"{name}.yaml"
        with open(path, 'w') as f:
            yaml.dump(config, f)
        config_paths[name] = path
    
    # Invalid user configs
    invalid_configs = {
        # Missing required field (no complete config even after merging)
        "missing_field": {
            "video_plume": {
                # Missing kernel_sigma
                "flip": True,
                "kernel_size": 0
            }
        },
        # Invalid value
        "invalid_kernel_size": {
            "video_plume": {
                "flip": False,
                "kernel_size": -1,  # Negative, which is invalid
                "kernel_sigma": 1.0
            }
        },
        # Speed exceeds max_speed
        "invalid_speed": {
            "navigator": {
                "orientation": 0.0,
                "speed": 2.0,  # Exceeds max_speed
                "max_speed": 1.0
            }
        }
    }
    
    # Write invalid configs to files
    invalid_paths = {}
    for name, config in invalid_configs.items():
        path = tmp_path / f"{name}.yaml"
        with open(path, 'w') as f:
            yaml.dump(config, f)
        invalid_paths[name] = path
    
    return {
        "config_dir": config_dir,
        "default_path": config_paths["default"],
        "speed_test_default_path": config_paths["speed_test_default"],
        "invalid_paths": invalid_paths
    }


class TestConfigValidationIntegration:
    """Test integration of config validation with config loading."""
    
    def test_load_config_validate_valid(self, valid_config_files):
        """Test that loading a valid config with validation works."""
        with patch('odor_plume_nav.config_utils.get_config_dir', 
                  return_value=valid_config_files["config_dir"]):
            # This should not raise any exceptions
            try:
                config = load_config(
                    user_config_path=valid_config_files["user_path"],
                    validate=True
                )
                # If we get here, validation is either working correctly or not implemented
                assert 'video_plume' in config
                assert 'navigator' in config
                assert config['video_plume']['flip'] is True  # From user config
            except TypeError:
                pytest.fail("load_config does not support validation yet")
    
    def test_load_config_validate_missing_field(self, invalid_config_files):
        """Test that validation catches missing required fields."""
        with patch('odor_plume_nav.config_utils.get_config_dir', 
                  return_value=invalid_config_files["config_dir"]):
            # Try to load config with missing required field
            with pytest.raises(ConfigValidationError) as excinfo:
                load_config(
                    user_config_path=invalid_config_files["invalid_paths"]["missing_field"],
                    validate=True
                )
            # Check that the error message mentions the missing field (kernel_sigma)
            assert "Missing required field" in str(excinfo.value)
            assert "kernel_sigma" in str(excinfo.value)
    
    def test_load_config_validate_invalid_kernel_size(self, invalid_config_files):
        """Test that validation catches invalid kernel_size."""
        with patch('odor_plume_nav.config_utils.get_config_dir', 
                  return_value=invalid_config_files["config_dir"]):
            # Try to load config with invalid kernel_size
            with pytest.raises(ConfigValidationError) as excinfo:
                load_config(
                    user_config_path=invalid_config_files["invalid_paths"]["invalid_kernel_size"],
                    validate=True
                )
            # Check that the error message mentions the invalid kernel_size
            assert "kernel_size" in str(excinfo.value)
    
    def test_load_config_validate_invalid_speed(self, invalid_config_files):
        """Test that validation catches speed exceeding max_speed."""
        # Use the complete default config for this test
        with patch('odor_plume_nav.config_utils.get_config_dir', 
                  return_value=invalid_config_files["config_dir"]), \
             patch('odor_plume_nav.config_utils.load_yaml_config', side_effect=[
                  # Load the speed_test_default config with complete video_plume settings 
                  # when the default config is requested
                  load_yaml_config(invalid_config_files["speed_test_default_path"]),
                  # Load the invalid_speed config when the user config is requested
                  load_yaml_config(invalid_config_files["invalid_paths"]["invalid_speed"])
             ]):
            # Try to load config with speed exceeding max_speed
            with pytest.raises(ConfigValidationError) as excinfo:
                load_config(
                    user_config_path=invalid_config_files["invalid_paths"]["invalid_speed"],
                    validate=True
                )
            # Check that the error message mentions speed and max_speed
            assert "speed" in str(excinfo.value)
            assert "max_speed" in str(excinfo.value)
    
    def test_load_config_validate_disabled(self, invalid_config_files):
        """Test that validation can be disabled."""
        with patch('odor_plume_nav.config_utils.get_config_dir', 
                  return_value=invalid_config_files["config_dir"]):
            try:
                # This should not raise validation errors when validate=False
                config = load_config(
                    user_config_path=invalid_config_files["invalid_paths"]["invalid_kernel_size"],
                    validate=False
                )
                
                # Check that the invalid value passed through
                assert config['video_plume']['kernel_size'] == -1
            except TypeError:
                pytest.fail("load_config does not support validation yet")
