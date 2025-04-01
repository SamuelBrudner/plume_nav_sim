"""Tests for config validation integration with config_utils module."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

from odor_plume_nav.config_utils import load_config, deep_update
from odor_plume_nav.config_validator import ConfigValidationError, validate_config


def validate_loaded_config(config, validate=True, validate_sections=None):
    """
    Function that will validate a loaded config.
    This represents what we want to integrate into load_config.
    
    Args:
        config: Configuration dictionary to validate
        validate: Whether to perform validation
        validate_sections: List of sections to validate (or None for all)
        
    Returns:
        The validated configuration
        
    Raises:
        ConfigValidationError: If validation fails
    """
    if validate:
        validate_config(config, required_sections=validate_sections)
    return config


@pytest.fixture
def valid_config_file():
    """Create a temporary file with valid configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'video_plume': {
                'flip': False,
                'kernel_size': 0,
                'kernel_sigma': 1.0
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0
            }
        }, f)
        return Path(f.name)


@pytest.fixture
def invalid_video_plume_config_file():
    """Create a temporary file with invalid video_plume configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'video_plume': {
                'flip': False,
                'kernel_size': -1,  # Invalid negative value
                'kernel_sigma': 1.0
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0
            }
        }, f)
        return Path(f.name)


@pytest.fixture
def invalid_navigator_config_file():
    """Create a temporary file with invalid navigator configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'video_plume': {
                'flip': False,
                'kernel_size': 0,
                'kernel_sigma': 1.0
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 2.0,  # Exceeds max_speed
                'max_speed': 1.0
            }
        }, f)
        return Path(f.name)


@pytest.fixture
def missing_section_config_file():
    """Create a temporary file with a missing required section."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'video_plume': {
                'flip': False,
                'kernel_size': 0,
                'kernel_sigma': 1.0
            }
            # Missing navigator section
        }, f)
        return Path(f.name)


@pytest.fixture
def missing_field_config_file():
    """Create a temporary file with a missing required field."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'video_plume': {
                'flip': False,
                'kernel_size': 0
                # Missing kernel_sigma
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0
            }
        }, f)
        return Path(f.name)


@pytest.fixture
def default_config():
    """Create a valid default config dict."""
    return {
        'video_plume': {
            'flip': False,
            'kernel_size': 0,
            'kernel_sigma': 1.0
        },
        'navigator': {
            'orientation': 0.0,
            'speed': 0.0,
            'max_speed': 1.0
        }
    }


class TestConfigValidationLogic:
    """Tests for validating configs using the validation function."""
    
    def test_validate_valid_config(self, default_config):
        """Test that validating a valid config works."""
        # This should not raise any exceptions
        result = validate_loaded_config(default_config, validate=True)
        assert result == default_config
    
    def test_validate_invalid_video_plume_config(self):
        """Test that validating a config with invalid video_plume raises error."""
        invalid_config = {
            'video_plume': {
                'flip': False,
                'kernel_size': -1,  # Invalid negative value
                'kernel_sigma': 1.0
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0
            }
        }
        
        # Should raise a validation error for negative kernel_size
        with pytest.raises(ConfigValidationError, match="kernel_size must be non-negative"):
            validate_loaded_config(invalid_config, validate=True)
    
    def test_validate_invalid_navigator_config(self):
        """Test that validating a config with invalid navigator raises error."""
        invalid_config = {
            'video_plume': {
                'flip': False,
                'kernel_size': 0,
                'kernel_sigma': 1.0
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 2.0,  # Exceeds max_speed
                'max_speed': 1.0
            }
        }
        
        # Should raise a validation error for speed exceeding max_speed
        with pytest.raises(ConfigValidationError, match="exceeds max_speed"):
            validate_loaded_config(invalid_config, validate=True)
    
    def test_validate_missing_section(self):
        """Test that validating a config with missing section raises error."""
        invalid_config = {
            'video_plume': {
                'flip': False,
                'kernel_size': 0,
                'kernel_sigma': 1.0
            }
            # Missing navigator section
        }
        
        # Should raise a validation error for missing navigator section
        with pytest.raises(ConfigValidationError, match="Missing required 'navigator' section"):
            validate_loaded_config(invalid_config, validate=True)
    
    def test_validate_missing_field(self):
        """Test that validating a config with missing field raises error."""
        invalid_config = {
            'video_plume': {
                'flip': False,
                'kernel_size': 0
                # Missing kernel_sigma
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0
            }
        }
        
        # Should raise a validation error for missing kernel_sigma
        with pytest.raises(ConfigValidationError, match="Missing required field 'kernel_sigma'"):
            validate_loaded_config(invalid_config, validate=True)
    
    def test_validation_disabled(self):
        """Test that validation can be disabled."""
        invalid_config = {
            'video_plume': {
                'flip': False,
                'kernel_size': -1,  # Invalid but ignored when validation is off
                'kernel_sigma': 1.0
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0
            }
        }
        
        # This should not raise any exceptions when validate=False
        result = validate_loaded_config(invalid_config, validate=False)
        assert result['video_plume']['kernel_size'] == -1
    
    def test_selective_validation(self):
        """Test that validation can be limited to specific sections."""
        invalid_config = {
            'video_plume': {
                'flip': False,
                'kernel_size': 0,
                'kernel_sigma': 1.0
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 2.0,  # Invalid but not validated
                'max_speed': 1.0
            }
        }
        
        # This should not raise when we only validate video_plume
        result = validate_loaded_config(invalid_config, validate=True, validate_sections=['video_plume'])
        assert result['navigator']['speed'] == 2.0
