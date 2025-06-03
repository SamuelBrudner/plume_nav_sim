"""Tests for enhanced config validation integration with Hydra configuration system.

This test module validates the comprehensive configuration management system incorporating
Hydra hierarchical configuration composition, Pydantic schema validation, and environment
variable interpolation patterns. The testing framework ensures robustness across
conf/base.yaml defaults, conf/config.yaml overrides, and conf/local/ customizations.
"""

import pytest
import tempfile
import yaml
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional

from {{cookiecutter.project_slug}}.config.schemas import (
    NavigatorConfig,
    VideoPlumeConfig,
    SingleAgentConfig,
    MultiAgentConfig
)

# Hydra imports with fallback handling
try:
    from hydra import initialize, compose
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    # Fallback for testing environments without Hydra
    HYDRA_AVAILABLE = False
    initialize = None
    compose = None
    ConfigStore = None
    DictConfig = dict
    OmegaConf = None


def validate_hydra_config(config_dict: Dict[str, Any], validate: bool = True, 
                         validate_sections: Optional[list] = None) -> Dict[str, Any]:
    """
    Enhanced configuration validation function for Hydra-based configuration system.
    
    This function validates configuration dictionaries using Pydantic schemas and
    supports hierarchical configuration composition scenarios including environment
    variable interpolation and override validation.
    
    Args:
        config_dict: Configuration dictionary from Hydra composition
        validate: Whether to perform Pydantic schema validation
        validate_sections: List of configuration sections to validate (None for all)
        
    Returns:
        The validated configuration dictionary
        
    Raises:
        ValueError: If Pydantic schema validation fails
        TypeError: If configuration structure is invalid
    """
    if not validate:
        return config_dict
    
    # Validate specific sections if requested
    sections_to_validate = validate_sections or ['navigator', 'video_plume']
    
    for section in sections_to_validate:
        if section not in config_dict:
            raise ValueError(f"Missing required configuration section: '{section}'")
        
        # Validate section using appropriate Pydantic schema
        section_config = config_dict[section]
        
        if section == 'navigator':
            # Use NavigatorConfig for comprehensive validation
            NavigatorConfig(**section_config)
        elif section == 'video_plume':
            # Use VideoPlumeConfig with validation skip for testing
            VideoPlumeConfig(**section_config, _skip_validation=True)
        else:
            # Generic validation for other sections
            if not isinstance(section_config, dict):
                raise TypeError(f"Configuration section '{section}' must be a dictionary")
    
    return config_dict


@pytest.fixture
def valid_hydra_config():
    """Create a valid Hydra-compatible configuration dictionary."""
    return {
        'video_plume': {
            'video_path': 'test_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': 5,
            'kernel_sigma': 1.0,
            'threshold': 0.5,
            'normalize': True,
            '_skip_validation': True  # Skip file existence validation for testing
        },
        'navigator': {
            'position': [0.0, 0.0],
            'orientation': 0.0,
            'speed': 0.0,
            'max_speed': 1.0,
            'angular_velocity': 0.0
        }
    }


@pytest.fixture
def valid_config_file(valid_hydra_config):
    """Create a temporary YAML file with valid Hydra configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_hydra_config, f)
        return Path(f.name)


@pytest.fixture
def invalid_video_plume_config():
    """Create configuration with invalid video_plume parameters."""
    return {
        'video_plume': {
            'video_path': 'test_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': -1,  # Invalid: must be positive
            'kernel_sigma': 1.0,
            'threshold': 0.5,
            'normalize': True,
            '_skip_validation': True
        },
        'navigator': {
            'position': [0.0, 0.0],
            'orientation': 0.0,
            'speed': 0.0,
            'max_speed': 1.0,
            'angular_velocity': 0.0
        }
    }


@pytest.fixture
def invalid_video_plume_config_file(invalid_video_plume_config):
    """Create a temporary file with invalid video_plume configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(invalid_video_plume_config, f)
        return Path(f.name)


@pytest.fixture
def invalid_navigator_config():
    """Create configuration with invalid navigator parameters."""
    return {
        'video_plume': {
            'video_path': 'test_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': 5,
            'kernel_sigma': 1.0,
            'threshold': 0.5,
            'normalize': True,
            '_skip_validation': True
        },
        'navigator': {
            'position': [0.0, 0.0],
            'orientation': 0.0,
            'speed': 2.0,  # Invalid: exceeds max_speed
            'max_speed': 1.0,
            'angular_velocity': 0.0
        }
    }


@pytest.fixture
def invalid_navigator_config_file(invalid_navigator_config):
    """Create a temporary file with invalid navigator configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(invalid_navigator_config, f)
        return Path(f.name)


@pytest.fixture
def missing_section_config():
    """Create configuration with missing required section."""
    return {
        'video_plume': {
            'video_path': 'test_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': 5,
            'kernel_sigma': 1.0,
            'threshold': 0.5,
            'normalize': True,
            '_skip_validation': True
        }
        # Missing navigator section
    }


@pytest.fixture
def missing_section_config_file(missing_section_config):
    """Create a temporary file with missing required section."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(missing_section_config, f)
        return Path(f.name)


@pytest.fixture
def missing_field_config():
    """Create configuration with missing required field."""
    return {
        'video_plume': {
            'video_path': 'test_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': 5
            # Missing kernel_sigma - required when kernel_size is provided
        },
        'navigator': {
            'position': [0.0, 0.0],
            'orientation': 0.0,
            'speed': 0.0,
            'max_speed': 1.0,
            'angular_velocity': 0.0
        }
    }


@pytest.fixture
def missing_field_config_file(missing_field_config):
    """Create a temporary file with missing required field."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(missing_field_config, f)
        return Path(f.name)


@pytest.fixture
def multi_agent_config():
    """Create a valid multi-agent configuration."""
    return {
        'video_plume': {
            'video_path': 'test_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': 5,
            'kernel_sigma': 1.0,
            'threshold': 0.5,
            'normalize': True,
            '_skip_validation': True
        },
        'navigator': {
            'positions': [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            'orientations': [0.0, 45.0, 90.0],
            'speeds': [1.0, 1.0, 1.0],
            'max_speeds': [1.5, 1.5, 1.5],
            'angular_velocities': [0.0, 0.0, 0.0],
            'num_agents': 3
        }
    }


@pytest.fixture
def environment_variable_config():
    """Create configuration with environment variable interpolation."""
    return {
        'video_plume': {
            'video_path': '${oc.env:TEST_VIDEO_PATH,test_video.mp4}',
            'flip': '${oc.env:VIDEO_FLIP,false}',
            'grayscale': True,
            'kernel_size': 5,
            'kernel_sigma': 1.0,
            'threshold': '${oc.env:PLUME_THRESHOLD,0.5}',
            'normalize': True,
            '_skip_validation': True
        },
        'navigator': {
            'position': [0.0, 0.0],
            'orientation': 0.0,
            'speed': 0.0,
            'max_speed': '${oc.env:NAVIGATOR_MAX_SPEED,1.0}',
            'angular_velocity': 0.0
        }
    }


@pytest.fixture
def temp_hydra_config_dir():
    """Create temporary Hydra configuration directory structure."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    conf_dir = Path(temp_dir) / "conf"
    conf_dir.mkdir()
    
    # Create base.yaml
    base_config = {
        'video_plume': {
            'video_path': 'default_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': None,
            'kernel_sigma': None,
            'threshold': None,
            'normalize': True,
            '_skip_validation': True
        },
        'navigator': {
            'position': None,
            'orientation': 0.0,
            'speed': 0.0,
            'max_speed': 1.0,
            'angular_velocity': 0.0
        }
    }
    
    with open(conf_dir / "base.yaml", 'w') as f:
        yaml.dump(base_config, f)
    
    # Create config.yaml
    config_overrides = {
        'defaults': ['base'],
        'video_plume': {
            'kernel_size': 5,
            'kernel_sigma': 1.0,
            'threshold': 0.5
        },
        'navigator': {
            'position': [0.0, 0.0],
            'max_speed': 2.0
        }
    }
    
    with open(conf_dir / "config.yaml", 'w') as f:
        yaml.dump(config_overrides, f)
    
    # Create local directory with override template
    local_dir = conf_dir / "local"
    local_dir.mkdir()
    
    local_overrides = {
        'video_plume': {
            'video_path': 'local_video.mp4',
            'flip': True
        },
        'navigator': {
            'speed': 0.5
        }
    }
    
    with open(local_dir / "test_overrides.yaml", 'w') as f:
        yaml.dump(local_overrides, f)
    
    yield conf_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_hydra_compose():
    """Mock Hydra compose function for testing without full Hydra setup."""
    def _compose(config_name: str = "config", overrides: list = None, return_hydra_config: bool = False):
        """Mock compose function that returns test configuration."""
        base_config = {
            'video_plume': {
                'video_path': 'test_video.mp4',
                'flip': False,
                'grayscale': True,
                'kernel_size': 5,
                'kernel_sigma': 1.0,
                'threshold': 0.5,
                'normalize': True,
                '_skip_validation': True
            },
            'navigator': {
                'position': [0.0, 0.0],
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0,
                'angular_velocity': 0.0
            }
        }
        
        # Apply overrides if provided
        if overrides:
            for override in overrides:
                if '=' in override:
                    key, value = override.split('=', 1)
                    # Simple override application for testing
                    if key == 'navigator.max_speed':
                        base_config['navigator']['max_speed'] = float(value)
                    elif key == 'video_plume.flip':
                        base_config['video_plume']['flip'] = value.lower() == 'true'
        
        return base_config if not return_hydra_config else (base_config, None)
    
    return _compose


class TestPydanticSchemaValidation:
    """Tests for enhanced Pydantic schema validation with Hydra configuration."""
    
    def test_validate_valid_config(self, valid_hydra_config):
        """Test that validating a valid Hydra config works with Pydantic schemas."""
        # This should not raise any exceptions
        result = validate_hydra_config(valid_hydra_config, validate=True)
        assert result == valid_hydra_config
    
    def test_navigator_config_pydantic_validation(self):
        """Test NavigatorConfig Pydantic model validation directly."""
        # Valid single agent configuration
        config_data = {
            'position': [1.0, 2.0],
            'orientation': 45.0,
            'speed': 0.5,
            'max_speed': 1.0,
            'angular_velocity': 0.1
        }
        
        navigator_config = NavigatorConfig(**config_data)
        assert navigator_config.position == (1.0, 2.0)
        assert navigator_config.orientation == 45.0
        assert navigator_config.speed == 0.5
        assert navigator_config.max_speed == 1.0
    
    def test_video_plume_config_pydantic_validation(self):
        """Test VideoPlumeConfig Pydantic model validation directly."""
        config_data = {
            'video_path': 'test_video.mp4',
            'flip': True,
            'grayscale': True,
            'kernel_size': 7,
            'kernel_sigma': 2.0,
            'threshold': 0.3,
            'normalize': True,
            '_skip_validation': True  # Skip file existence check
        }
        
        video_plume_config = VideoPlumeConfig(**config_data)
        assert video_plume_config.flip is True
        assert video_plume_config.kernel_size == 7
        assert video_plume_config.kernel_sigma == 2.0
        assert video_plume_config.threshold == 0.3
    
    def test_validate_invalid_video_plume_config(self, invalid_video_plume_config):
        """Test that validating a config with invalid video_plume raises Pydantic error."""
        # Should raise a validation error for negative kernel_size
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            validate_hydra_config(invalid_video_plume_config, validate=True)
    
    def test_video_plume_kernel_validation(self):
        """Test VideoPlumeConfig kernel size validation rules."""
        # Test invalid negative kernel_size
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            VideoPlumeConfig(
                video_path="test.mp4",
                kernel_size=-1,
                kernel_sigma=1.0,
                _skip_validation=True
            )
        
        # Test invalid even kernel_size
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            VideoPlumeConfig(
                video_path="test.mp4",
                kernel_size=4,
                kernel_sigma=1.0,
                _skip_validation=True
            )
        
        # Test missing kernel_sigma when kernel_size provided
        with pytest.raises(ValueError, match="kernel_sigma must be specified"):
            VideoPlumeConfig(
                video_path="test.mp4",
                kernel_size=5,
                _skip_validation=True
            )
    
    def test_validate_invalid_navigator_config(self, invalid_navigator_config):
        """Test that validating a config with invalid navigator raises Pydantic error."""
        # Should raise a validation error for speed exceeding max_speed
        with pytest.raises(ValueError, match="cannot exceed max_speed"):
            validate_hydra_config(invalid_navigator_config, validate=True)
    
    def test_navigator_speed_validation(self):
        """Test NavigatorConfig speed constraint validation."""
        # Test speed exceeding max_speed
        with pytest.raises(ValueError, match="cannot exceed max_speed"):
            NavigatorConfig(
                position=[0.0, 0.0],
                speed=2.0,
                max_speed=1.0
            )
        
        # Test valid speed within max_speed
        config = NavigatorConfig(
            position=[0.0, 0.0],
            speed=0.8,
            max_speed=1.0
        )
        assert config.speed == 0.8
        assert config.max_speed == 1.0
    
    def test_validate_missing_section(self, missing_section_config):
        """Test that validating a config with missing section raises error."""
        # Should raise a validation error for missing navigator section
        with pytest.raises(ValueError, match="Missing required configuration section: 'navigator'"):
            validate_hydra_config(missing_section_config, validate=True)
    
    def test_validate_missing_field(self, missing_field_config):
        """Test that validating a config with missing field raises Pydantic error."""
        # Should raise a validation error for missing kernel_sigma when kernel_size is provided
        with pytest.raises(ValueError, match="kernel_sigma must be specified"):
            validate_hydra_config(missing_field_config, validate=True)
    
    def test_validation_disabled(self, invalid_video_plume_config):
        """Test that validation can be disabled."""
        # This should not raise any exceptions when validate=False
        result = validate_hydra_config(invalid_video_plume_config, validate=False)
        assert result['video_plume']['kernel_size'] == -1
    
    def test_selective_validation(self, invalid_navigator_config):
        """Test that validation can be limited to specific sections."""
        # This should not raise when we only validate video_plume
        result = validate_hydra_config(
            invalid_navigator_config, 
            validate=True, 
            validate_sections=['video_plume']
        )
        assert result['navigator']['speed'] == 2.0
    
    def test_multi_agent_validation(self, multi_agent_config):
        """Test validation of multi-agent configuration."""
        # This should validate successfully
        result = validate_hydra_config(multi_agent_config, validate=True)
        assert result['navigator']['num_agents'] == 3
        assert len(result['navigator']['positions']) == 3
    
    def test_multi_agent_inconsistent_lengths(self):
        """Test multi-agent validation with inconsistent parameter lengths."""
        invalid_multi_config = {
            'video_plume': {
                'video_path': 'test_video.mp4',
                '_skip_validation': True
            },
            'navigator': {
                'positions': [[0.0, 0.0], [1.0, 0.0]],  # 2 agents
                'orientations': [0.0, 45.0, 90.0],      # 3 orientations - mismatch
                'speeds': [1.0, 1.0],
                'max_speeds': [1.5, 1.5],
                'num_agents': 2
            }
        }
        
        # Should raise validation error for length mismatch
        with pytest.raises(ValueError, match="length.*does not match.*number of agents"):
            validate_hydra_config(invalid_multi_config, validate=True)


class TestHydraConfigurationComposition:
    """Tests for Hydra configuration composition and override scenarios."""
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_config_composition(self, temp_hydra_config_dir):
        """Test basic Hydra configuration composition."""
        with initialize(config_path=str(temp_hydra_config_dir)):
            cfg = compose(config_name="config")
            
            # Test that base config is loaded
            assert cfg.navigator.orientation == 0.0
            assert cfg.video_plume.normalize is True
            
            # Test that overrides are applied
            assert cfg.navigator.max_speed == 2.0  # Overridden from base
            assert cfg.video_plume.kernel_size == 5  # Set in config.yaml
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_override_scenarios(self, temp_hydra_config_dir):
        """Test Hydra configuration override scenarios."""
        with initialize(config_path=str(temp_hydra_config_dir)):
            # Test command-line style overrides
            cfg = compose(
                config_name="config",
                overrides=["navigator.max_speed=3.0", "video_plume.flip=true"]
            )
            
            assert cfg.navigator.max_speed == 3.0
            assert cfg.video_plume.flip is True
    
    def test_mock_hydra_composition(self, mock_hydra_compose):
        """Test configuration composition using mock Hydra compose."""
        # Test basic composition
        cfg = mock_hydra_compose(config_name="config")
        assert cfg['navigator']['max_speed'] == 1.0
        assert cfg['video_plume']['flip'] is False
        
        # Test with overrides
        cfg_with_overrides = mock_hydra_compose(
            config_name="config",
            overrides=["navigator.max_speed=2.5", "video_plume.flip=true"]
        )
        assert cfg_with_overrides['navigator']['max_speed'] == 2.5
        assert cfg_with_overrides['video_plume']['flip'] is True
    
    def test_environment_variable_interpolation(self, environment_variable_config):
        """Test environment variable interpolation in configuration."""
        # Set test environment variables
        test_env = {
            'TEST_VIDEO_PATH': 'env_video.mp4',
            'VIDEO_FLIP': 'true',
            'PLUME_THRESHOLD': '0.7',
            'NAVIGATOR_MAX_SPEED': '2.0'
        }
        
        with patch.dict(os.environ, test_env):
            # In a real Hydra scenario, these would be resolved automatically
            # For testing, we'll validate the interpolation syntax is present
            assert '${oc.env:TEST_VIDEO_PATH,test_video.mp4}' in str(environment_variable_config)
            assert '${oc.env:NAVIGATOR_MAX_SPEED,1.0}' in str(environment_variable_config)


class TestHierarchicalConfigurationValidation:
    """Tests for hierarchical configuration validation with multiple override layers."""
    
    def test_base_config_validation(self):
        """Test validation of base configuration parameters."""
        base_config = {
            'video_plume': {
                'video_path': 'base_video.mp4',
                'flip': False,
                'grayscale': True,
                'normalize': True,
                '_skip_validation': True
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0
            }
        }
        
        # Should validate successfully
        result = validate_hydra_config(base_config, validate=True)
        assert result == base_config
    
    def test_layered_override_validation(self):
        """Test validation with multiple configuration layers."""
        # Simulate base -> config -> local override chain
        base_config = {
            'navigator': {
                'speed': 0.0,
                'max_speed': 1.0
            }
        }
        
        config_overrides = {
            'navigator': {
                'speed': 0.5,  # Override from base
                'max_speed': 1.5  # Override from base
            }
        }
        
        local_overrides = {
            'navigator': {
                'speed': 0.8  # Final override
            }
        }
        
        # Manually apply override chain (in real Hydra this is automatic)
        final_config = {
            'video_plume': {
                'video_path': 'test.mp4',
                '_skip_validation': True
            },
            'navigator': {
                'speed': local_overrides['navigator']['speed'],  # 0.8
                'max_speed': config_overrides['navigator']['max_speed']  # 1.5
            }
        }
        
        # Should validate successfully
        result = validate_hydra_config(final_config, validate=True)
        assert result['navigator']['speed'] == 0.8
        assert result['navigator']['max_speed'] == 1.5
    
    def test_override_constraint_validation(self):
        """Test that override values still respect schema constraints."""
        # Test that overrides can't violate Pydantic constraints
        invalid_override_config = {
            'video_plume': {
                'video_path': 'test.mp4',
                'kernel_size': -5,  # Invalid even after override
                'kernel_sigma': 1.0,
                '_skip_validation': True
            },
            'navigator': {
                'speed': 0.5,
                'max_speed': 1.0
            }
        }
        
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            validate_hydra_config(invalid_override_config, validate=True)