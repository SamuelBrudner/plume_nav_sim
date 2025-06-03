"""Tests for config validation integration with config_utils module."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from {{cookiecutter.project_slug}}.config.schemas import (
    NavigatorConfig,
    VideoPlumeConfig,
    ConfigValidationError,
    validate_config,
    validate_video_plume_config,
    validate_navigator_config,
    load_config,
    update_config
)

try:
    from hydra import compose, initialize
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False


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
                'video_path': '/path/to/video.mp4',
                'flip': False,
                'kernel_size': 3,
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
                'video_path': '/path/to/video.mp4',
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
                'video_path': '/path/to/video.mp4',
                'flip': False,
                'kernel_size': 3,
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
                'video_path': '/path/to/video.mp4',
                'flip': False,
                'kernel_size': 3,
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
                'video_path': '/path/to/video.mp4',
                'flip': False,
                'kernel_size': 3
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
            'video_path': '/path/to/video.mp4',
            'flip': False,
            'kernel_size': 3,
            'kernel_sigma': 1.0
        },
        'navigator': {
            'orientation': 0.0,
            'speed': 0.0,
            'max_speed': 1.0
        }
    }


@pytest.fixture
def hierarchical_config_base():
    """Create a base configuration for hierarchical testing."""
    return {
        'defaults': [
            'base_config',
            'navigator: single_agent'
        ],
        'video_plume': {
            'flip': False,
            'kernel_size': 3
        },
        'navigator': {
            'max_speed': 1.0
        }
    }


@pytest.fixture
def hierarchical_config_override():
    """Create an override configuration for hierarchical testing."""
    return {
        'video_plume': {
            'kernel_sigma': 2.0,  # Override the base value
            'flip': True  # Override the base value
        },
        'navigator': {
            'orientation': 45.0,  # Add new value
            'speed': 0.5  # Add new value
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
                'video_path': '/path/to/video.mp4',
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
        with pytest.raises(ConfigValidationError, match="kernel_size must be"):
            validate_loaded_config(invalid_config, validate=True)
    
    def test_validate_invalid_navigator_config(self):
        """Test that validating a config with invalid navigator raises error."""
        invalid_config = {
            'video_plume': {
                'video_path': '/path/to/video.mp4',
                'flip': False,
                'kernel_size': 3,
                'kernel_sigma': 1.0
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 2.0,  # Exceeds max_speed
                'max_speed': 1.0
            }
        }
        
        # Should raise a validation error for speed exceeding max_speed
        with pytest.raises(ConfigValidationError, match="cannot exceed max_speed"):
            validate_loaded_config(invalid_config, validate=True)
    
    def test_validate_missing_section(self):
        """Test that validating a config with missing section raises error."""
        invalid_config = {
            'video_plume': {
                'video_path': '/path/to/video.mp4',
                'flip': False,
                'kernel_size': 3,
                'kernel_sigma': 1.0
            }
            # Missing navigator section
        }
        
        # Should raise a validation error for missing navigator section
        with pytest.raises(ConfigValidationError, match="Missing required.*navigator"):
            validate_loaded_config(invalid_config, validate=True)
    
    def test_validate_missing_field(self):
        """Test that validating a config with missing field raises error."""
        invalid_config = {
            'video_plume': {
                'video_path': '/path/to/video.mp4',
                'flip': False,
                'kernel_size': 3
                # Missing kernel_sigma and video_path is required
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0
            }
        }
        
        # Should raise a validation error for missing required field
        with pytest.raises(ConfigValidationError):
            validate_loaded_config(invalid_config, validate=True)
    
    def test_validation_disabled(self):
        """Test that validation can be disabled."""
        invalid_config = {
            'video_plume': {
                'video_path': '/path/to/video.mp4',
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
                'video_path': '/path/to/video.mp4',
                'flip': False,
                'kernel_size': 3,
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


class TestPydanticSchemaValidation:
    """Tests for Pydantic schema-based configuration validation."""
    
    def test_navigator_config_validation_success(self):
        """Test successful validation of NavigatorConfig."""
        config_data = {
            'orientation': 45.0,
            'speed': 0.5,
            'max_speed': 1.0,
            'angular_velocity': 10.0
        }
        
        navigator_config = NavigatorConfig(**config_data)
        assert navigator_config.orientation == 45.0
        assert navigator_config.speed == 0.5
        assert navigator_config.max_speed == 1.0
    
    def test_navigator_config_validation_failure(self):
        """Test NavigatorConfig validation with invalid data."""
        # Speed exceeds max_speed
        invalid_config = {
            'orientation': 45.0,
            'speed': 2.0,
            'max_speed': 1.0
        }
        
        with pytest.raises(ValueError, match="cannot exceed max_speed"):
            NavigatorConfig(**invalid_config)
    
    def test_video_plume_config_validation_success(self):
        """Test successful validation of VideoPlumeConfig."""
        config_data = {
            'video_path': '/path/to/video.mp4',
            'flip': True,
            'kernel_size': 5,
            'kernel_sigma': 2.0
        }
        
        video_config = VideoPlumeConfig(**config_data)
        assert video_config.video_path == '/path/to/video.mp4'
        assert video_config.flip is True
        assert video_config.kernel_size == 5
    
    def test_video_plume_config_validation_failure(self):
        """Test VideoPlumeConfig validation with invalid data."""
        # Invalid kernel_size (even number)
        invalid_config = {
            'video_path': '/path/to/video.mp4',
            'kernel_size': 4  # Must be odd
        }
        
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            VideoPlumeConfig(**invalid_config)
    
    def test_multi_agent_config_validation(self):
        """Test multi-agent configuration validation."""
        multi_agent_config = {
            'positions': [[0, 0], [10, 10], [20, 20]],
            'orientations': [0.0, 45.0, 90.0],
            'speeds': [0.1, 0.2, 0.3],
            'max_speeds': [1.0, 1.0, 1.0],
            'num_agents': 3
        }
        
        navigator_config = NavigatorConfig(**multi_agent_config)
        assert len(navigator_config.positions) == 3
        assert len(navigator_config.orientations) == 3
    
    def test_multi_agent_config_length_mismatch(self):
        """Test multi-agent configuration with mismatched lengths."""
        invalid_config = {
            'positions': [[0, 0], [10, 10]],  # 2 agents
            'orientations': [0.0, 45.0, 90.0],  # 3 orientations
            'speeds': [0.1, 0.2]
        }
        
        with pytest.raises(ValueError, match="length.*does not match"):
            NavigatorConfig(**invalid_config)


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
class TestHydraConfigurationIntegration:
    """Tests for Hydra configuration composition and override scenarios."""
    
    def test_hydra_config_composition(self, hierarchical_config_base, hierarchical_config_override):
        """Test Hydra configuration composition from multiple sources."""
        # Simulate Hydra's config composition
        base_config = OmegaConf.create(hierarchical_config_base)
        override_config = OmegaConf.create(hierarchical_config_override)
        
        # Merge configurations (simulating Hydra behavior)
        composed_config = OmegaConf.merge(base_config, override_config)
        
        # Convert to dict for validation
        config_dict = OmegaConf.to_container(composed_config, resolve=True)
        
        # Should be valid after composition
        result = validate_loaded_config(config_dict, validate=True)
        
        assert result['video_plume']['kernel_sigma'] == 2.0  # Override value
        assert result['video_plume']['flip'] is True  # Override value
        assert result['navigator']['orientation'] == 45.0  # New value
    
    def test_hydra_environment_variable_interpolation(self):
        """Test Hydra environment variable interpolation."""
        config_with_env = {
            'video_plume': {
                'video_path': '${oc.env:VIDEO_PATH,/default/video.mp4}',
                'flip': False,
                'kernel_size': 3,
                'kernel_sigma': 1.0
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0
            }
        }
        
        # Mock environment variable
        with patch.dict('os.environ', {'VIDEO_PATH': '/env/video.mp4'}):
            omega_config = OmegaConf.create(config_with_env)
            # Resolve environment variables
            resolved_config = OmegaConf.to_container(omega_config, resolve=True)
            
            # Should resolve environment variable
            assert resolved_config['video_plume']['video_path'] == '/env/video.mp4'
            
            # Should validate successfully
            validate_loaded_config(resolved_config, validate=True)
    
    def test_hydra_config_override_validation(self):
        """Test that Hydra overrides are properly validated."""
        base_config = {
            'video_plume': {
                'video_path': '/path/to/video.mp4',
                'flip': False,
                'kernel_size': 3,
                'kernel_sigma': 1.0
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0
            }
        }
        
        # Override that creates invalid configuration
        invalid_override = {
            'navigator': {
                'speed': 2.0  # This exceeds max_speed from base
            }
        }
        
        base_omega = OmegaConf.create(base_config)
        override_omega = OmegaConf.create(invalid_override)
        
        # Merge configurations
        merged_config = OmegaConf.merge(base_omega, override_omega)
        config_dict = OmegaConf.to_container(merged_config, resolve=True)
        
        # Should fail validation after merge
        with pytest.raises(ConfigValidationError, match="cannot exceed max_speed"):
            validate_loaded_config(config_dict, validate=True)
    
    def test_hydra_structured_config_validation(self):
        """Test validation of Hydra structured configs with Pydantic models."""
        # Create a structured config that should map to our Pydantic models
        structured_config = {
            'video_plume': {
                'video_path': '/path/to/video.mp4',
                'flip': True,
                'kernel_size': 5,
                'kernel_sigma': 1.5
            },
            'navigator': {
                'orientation': 30.0,
                'speed': 0.8,
                'max_speed': 1.0
            }
        }
        
        omega_config = OmegaConf.create(structured_config)
        config_dict = OmegaConf.to_container(omega_config, resolve=True)
        
        # Validate with Pydantic models
        video_config = VideoPlumeConfig(**config_dict['video_plume'])
        navigator_config = NavigatorConfig(**config_dict['navigator'])
        
        assert video_config.kernel_size == 5
        assert navigator_config.speed == 0.8
        
        # Also validate with traditional validator
        validate_loaded_config(config_dict, validate=True)


class TestHierarchicalConfigurationValidation:
    """Tests for hierarchical configuration validation with fixture support."""
    
    def test_base_config_validation(self, hierarchical_config_base):
        """Test validation of base configuration."""
        # Add required fields to make it valid
        complete_base = {
            **hierarchical_config_base,
            'video_plume': {
                **hierarchical_config_base['video_plume'],
                'video_path': '/path/to/video.mp4',
                'kernel_sigma': 1.0
            },
            'navigator': {
                **hierarchical_config_base['navigator'],
                'orientation': 0.0,
                'speed': 0.0
            }
        }
        
        # Remove defaults key for validation (it's Hydra-specific)
        validation_config = {k: v for k, v in complete_base.items() if k != 'defaults'}
        
        result = validate_loaded_config(validation_config, validate=True)
        assert result['video_plume']['flip'] is False
        assert result['navigator']['max_speed'] == 1.0
    
    def test_override_config_merging(self, hierarchical_config_base, hierarchical_config_override):
        """Test merging base and override configurations."""
        # Create complete base config
        complete_base = {
            'video_plume': {
                'video_path': '/path/to/video.mp4',
                'flip': False,
                'kernel_size': 3,
                'kernel_sigma': 1.0
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0
            }
        }
        
        # Manually merge (simulating what Hydra would do)
        merged_config = {}
        for key in complete_base:
            if key in hierarchical_config_override:
                merged_config[key] = {
                    **complete_base[key],
                    **hierarchical_config_override[key]
                }
            else:
                merged_config[key] = complete_base[key]
        
        result = validate_loaded_config(merged_config, validate=True)
        
        # Check override values
        assert result['video_plume']['kernel_sigma'] == 2.0  # Overridden
        assert result['video_plume']['flip'] is True  # Overridden
        assert result['navigator']['orientation'] == 45.0  # Added
        assert result['navigator']['speed'] == 0.5  # Added
        assert result['navigator']['max_speed'] == 1.0  # From base
    
    def test_partial_section_validation(self):
        """Test validation of partial configuration sections."""
        partial_config = {
            'video_plume': {
                'video_path': '/path/to/video.mp4',
                'flip': True,
                'kernel_size': 3,
                'kernel_sigma': 1.0
            }
            # Missing navigator section entirely
        }
        
        # Should work when validating only video_plume section
        result = validate_loaded_config(
            partial_config, 
            validate=True, 
            validate_sections=['video_plume']
        )
        assert result['video_plume']['flip'] is True
        
        # Should fail when trying to validate all sections
        with pytest.raises(ConfigValidationError, match="Missing required.*navigator"):
            validate_loaded_config(partial_config, validate=True)
    
    def test_deep_config_hierarchy_validation(self):
        """Test validation of deeply nested configuration hierarchies."""
        deep_config = {
            'video_plume': {
                'video_path': '/path/to/video.mp4',
                'processing': {
                    'flip': True,
                    'filters': {
                        'gaussian': {
                            'kernel_size': 5,
                            'kernel_sigma': 2.0
                        }
                    }
                }
            },
            'navigator': {
                'dynamics': {
                    'linear': {
                        'speed': 0.5,
                        'max_speed': 1.0
                    },
                    'angular': {
                        'orientation': 45.0,
                        'angular_velocity': 10.0
                    }
                }
            }
        }
        
        # Flatten for validation (current validators expect flat structure)
        flattened_config = {
            'video_plume': {
                'video_path': deep_config['video_plume']['video_path'],
                'flip': deep_config['video_plume']['processing']['flip'],
                'kernel_size': deep_config['video_plume']['processing']['filters']['gaussian']['kernel_size'],
                'kernel_sigma': deep_config['video_plume']['processing']['filters']['gaussian']['kernel_sigma']
            },
            'navigator': {
                'speed': deep_config['navigator']['dynamics']['linear']['speed'],
                'max_speed': deep_config['navigator']['dynamics']['linear']['max_speed'],
                'orientation': deep_config['navigator']['dynamics']['angular']['orientation'],
                'angular_velocity': deep_config['navigator']['dynamics']['angular']['angular_velocity']
            }
        }
        
        result = validate_loaded_config(flattened_config, validate=True)
        assert result['video_plume']['kernel_size'] == 5
        assert result['navigator']['orientation'] == 45.0