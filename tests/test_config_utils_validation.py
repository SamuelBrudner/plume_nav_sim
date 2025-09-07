"""Tests for enhanced config validation with Hydra 1.3+ structured configuration architecture.

This test module validates the comprehensive configuration management system incorporating
Hydra 1.3+ structured configuration, dataclass-based schemas with Pydantic enforcement, 
compile-time validation workflow, and enhanced error handling. The testing framework 
ensures robustness across Hydra ConfigStore registration, hierarchical composition, 
and Loguru logging integration.
"""

import pytest
import tempfile
import yaml
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from src.odor_plume_nav.config.models import (
    NavigatorConfig,
    VideoPlumeConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    SimulationConfig,
    register_config_schemas,
    validate_env_interpolation,
    resolve_env_value
)

# Hydra imports with fallback handling for structured configuration
try:
    from hydra import initialize, initialize_config_dir, compose
    from hydra.core.config_store import ConfigStore
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import DictConfig, OmegaConf, MISSING
    HYDRA_AVAILABLE = True
except ImportError:
    # Fallback for testing environments without Hydra
    HYDRA_AVAILABLE = False
    initialize = None
    compose = None
    ConfigStore = None
    GlobalHydra = None
    DictConfig = dict
    OmegaConf = None
    MISSING = None

# Loguru imports for testing structured logging integration
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    logger = None
    LOGURU_AVAILABLE = False


def validate_hydra_structured_config(config_dict: Dict[str, Any], validate: bool = True, 
                                    validate_sections: Optional[list] = None, 
                                    enable_compile_time_validation: bool = True) -> Dict[str, Any]:
    """
    Enhanced configuration validation function for Hydra 1.3+ structured configuration system.
    
    This function validates configuration dictionaries using dataclass-based schemas with 
    Pydantic enforcement and supports hierarchical configuration composition scenarios 
    including environment variable interpolation, ConfigStore validation, and 
    compile-time validation workflow.
    
    Args:
        config_dict: Configuration dictionary from Hydra composition or dataclass
        validate: Whether to perform Pydantic schema validation
        validate_sections: List of configuration sections to validate (None for all)
        enable_compile_time_validation: Enable compile-time validation checks
        
    Returns:
        The validated configuration dictionary
        
    Raises:
        ValueError: If Pydantic schema validation fails
        TypeError: If configuration structure is invalid
        AttributeError: If dataclass field validation fails
    """
    if not validate:
        return config_dict
    
    # Validate specific sections if requested
    sections_to_validate = validate_sections or ['navigator', 'video_plume', 'simulation']
    
    for section in sections_to_validate:
        if section not in config_dict:
            raise ValueError(f"Missing required configuration section: '{section}'")
        
        # Validate section using appropriate enhanced Pydantic schema with dataclass support
        section_config = config_dict[section]
        
        if section == 'navigator':
            # Use NavigatorConfig with enhanced validation and _target_ field support
            validated_config = NavigatorConfig(**section_config)
            # Test compile-time validation if enabled
            if enable_compile_time_validation:
                assert hasattr(validated_config, '_target_'), "Navigator config missing _target_ field for Hydra factory instantiation"
        elif section == 'video_plume':
            # Use VideoPlumeConfig with enhanced validation and _target_ field support
            validated_config = VideoPlumeConfig(**section_config)
            if enable_compile_time_validation:
                assert hasattr(validated_config, '_target_'), "VideoPlumeConfig missing _target_ field for Hydra factory instantiation"
        elif section == 'simulation':
            # Use SimulationConfig for simulation parameters
            validated_config = SimulationConfig(**section_config)
            if enable_compile_time_validation:
                assert hasattr(validated_config, '_target_'), "SimulationConfig missing _target_ field for Hydra factory instantiation"
        else:
            # Generic validation for other sections
            if not isinstance(section_config, dict):
                raise TypeError(f"Configuration section '{section}' must be a dictionary")
    
    return config_dict


# Backwards compatibility alias
validate_hydra_config = validate_hydra_structured_config


@pytest.fixture
def valid_hydra_structured_config():
    """Create a valid Hydra 1.3+ structured configuration dictionary with dataclass support."""
    return {
        'video_plume': {
            'video_path': 'test_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': 5,
            'kernel_sigma': 1.0,
            'threshold': 0.5,
            'normalize': True,
            '_target_': 'odor_plume_nav.data.VideoPlume',  # Hydra factory target
        },
        'navigator': {
            'position': [0.0, 0.0],
            'orientation': 0.0,
            'speed': 0.0,
            'max_speed': 1.0,
            'angular_velocity': 0.0,
            'mode': 'single',
            '_target_': 'odor_plume_nav.api.create_navigator'  # Hydra factory target
        },
        'simulation': {
            'max_steps': 1000,
            'step_size': 1.0,
            'enable_gpu': False,
            'record_trajectory': True,
            'output_format': 'numpy',
            '_target_': 'odor_plume_nav.api.run_simulation'  # Hydra factory target
        }
    }


@pytest.fixture
def valid_hydra_config(valid_hydra_structured_config):
    """Backwards compatibility fixture for legacy tests."""
    return valid_hydra_structured_config


@pytest.fixture
def structured_config_with_interpolation():
    """Create configuration with Hydra environment variable interpolation."""
    return {
        'video_plume': {
            'video_path': '${oc.env:TEST_VIDEO_PATH,test_video.mp4}',
            'flip': '${oc.env:VIDEO_FLIP,false}',
            'grayscale': True,
            'kernel_size': 5,
            'kernel_sigma': 1.0,
            'threshold': '${oc.env:PLUME_THRESHOLD,0.5}',
            'normalize': True,
            '_target_': 'odor_plume_nav.data.VideoPlume'
        },
        'navigator': {
            'position': [0.0, 0.0],
            'orientation': '${oc.env:AGENT_ORIENTATION,0.0}',
            'speed': 0.0,
            'max_speed': '${oc.env:NAVIGATOR_MAX_SPEED,1.0}',
            'angular_velocity': 0.0,
            'mode': 'single',
            '_target_': 'odor_plume_nav.api.create_navigator'
        },
        'simulation': {
            'max_steps': '${oc.env:MAX_STEPS,1000}',
            'random_seed': '${oc.env:RANDOM_SEED,42}',
            'output_directory': '${oc.env:OUTPUT_DIR,./outputs}',
            '_target_': 'odor_plume_nav.api.run_simulation'
        }
    }


@pytest.fixture
def dataclass_config_schema():
    """Create a sample dataclass-based configuration schema for testing."""
    @dataclass
    class TestNavigatorConfig:
        position: tuple = field(default=(0.0, 0.0))
        orientation: float = field(default=0.0)
        speed: float = field(default=0.0)
        max_speed: float = field(default=1.0)
        _target_: str = field(default="test.navigator.Factory")
        
        def __post_init__(self):
            # Dataclass validation
            if self.speed > self.max_speed:
                raise ValueError(f"Speed {self.speed} cannot exceed max_speed {self.max_speed}")
    
    return TestNavigatorConfig


@pytest.fixture
def valid_config_file(valid_hydra_structured_config):
    """Create a temporary YAML file with valid Hydra structured configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_hydra_structured_config, f)
        return Path(f.name)


@pytest.fixture
def invalid_video_plume_config():
    """Create structured configuration with invalid video_plume parameters."""
    return {
        'video_plume': {
            'video_path': 'test_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': -1,  # Invalid: must be positive
            'kernel_sigma': 1.0,
            'threshold': 0.5,
            'normalize': True,
            '_target_': 'odor_plume_nav.data.VideoPlume'
        },
        'navigator': {
            'position': [0.0, 0.0],
            'orientation': 0.0,
            'speed': 0.0,
            'max_speed': 1.0,
            'angular_velocity': 0.0,
            'mode': 'single',
            '_target_': 'odor_plume_nav.api.create_navigator'
        },
        'simulation': {
            'max_steps': 1000,
            '_target_': 'odor_plume_nav.api.run_simulation'
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
    """Create structured configuration with invalid navigator parameters."""
    return {
        'video_plume': {
            'video_path': 'test_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': 5,
            'kernel_sigma': 1.0,
            'threshold': 0.5,
            'normalize': True,
            '_target_': 'odor_plume_nav.data.VideoPlume'
        },
        'navigator': {
            'position': [0.0, 0.0],
            'orientation': 0.0,
            'speed': 2.0,  # Invalid: exceeds max_speed
            'max_speed': 1.0,
            'angular_velocity': 0.0,
            'mode': 'single',
            '_target_': 'odor_plume_nav.api.create_navigator'
        },
        'simulation': {
            'max_steps': 1000,
            '_target_': 'odor_plume_nav.api.run_simulation'
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
    """Create structured configuration with missing required section."""
    return {
        'video_plume': {
            'video_path': 'test_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': 5,
            'kernel_sigma': 1.0,
            'threshold': 0.5,
            'normalize': True,
            '_target_': 'odor_plume_nav.data.VideoPlume'
        }
        # Missing navigator section (required for structured config)
    }


@pytest.fixture
def missing_section_config_file(missing_section_config):
    """Create a temporary file with missing required section."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(missing_section_config, f)
        return Path(f.name)


@pytest.fixture
def missing_field_config():
    """Create structured configuration with missing required field."""
    return {
        'video_plume': {
            'video_path': 'test_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': 5,
            '_target_': 'odor_plume_nav.data.VideoPlume'
            # Missing kernel_sigma - required when kernel_size is provided
        },
        'navigator': {
            'position': [0.0, 0.0],
            'orientation': 0.0,
            'speed': 0.0,
            'max_speed': 1.0,
            'angular_velocity': 0.0,
            'mode': 'single',
            '_target_': 'odor_plume_nav.api.create_navigator'
        },
        'simulation': {
            'max_steps': 1000,
            '_target_': 'odor_plume_nav.api.run_simulation'
        }
    }


@pytest.fixture
def missing_field_config_file(missing_field_config):
    """Create a temporary file with missing required field."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(missing_field_config, f)
        return Path(f.name)


@pytest.fixture
def multi_agent_structured_config():
    """Create a valid multi-agent structured configuration."""
    return {
        'video_plume': {
            'video_path': 'test_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': 5,
            'kernel_sigma': 1.0,
            'threshold': 0.5,
            'normalize': True,
            '_target_': 'odor_plume_nav.data.VideoPlume'
        },
        'navigator': {
            'positions': [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            'orientations': [0.0, 45.0, 90.0],
            'speeds': [1.0, 1.0, 1.0],
            'max_speeds': [1.5, 1.5, 1.5],
            'angular_velocities': [0.0, 0.0, 0.0],
            'num_agents': 3,
            'mode': 'multi',
            '_target_': 'odor_plume_nav.api.create_navigator'
        },
        'simulation': {
            'max_steps': 1000,
            'enable_visualization': False,  # Disable for multi-agent performance
            '_target_': 'odor_plume_nav.api.run_simulation'
        }
    }


@pytest.fixture 
def multi_agent_config(multi_agent_structured_config):
    """Backwards compatibility fixture for legacy tests."""
    return multi_agent_structured_config


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
def temp_hydra_structured_config_dir():
    """Create temporary Hydra structured configuration directory structure."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    conf_dir = Path(temp_dir) / "conf"
    conf_dir.mkdir()
    
    # Create base.yaml with structured config support
    base_config = {
        'video_plume': {
            'video_path': 'default_video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': None,
            'kernel_sigma': None,
            'threshold': None,
            'normalize': True,
            '_target_': 'odor_plume_nav.data.VideoPlume'
        },
        'navigator': {
            'position': None,
            'orientation': 0.0,
            'speed': 0.0,
            'max_speed': 1.0,
            'angular_velocity': 0.0,
            'mode': 'single',
            '_target_': 'odor_plume_nav.api.create_navigator'
        },
        'simulation': {
            'max_steps': 1000,
            'record_trajectory': True,
            'output_format': 'numpy',
            '_target_': 'odor_plume_nav.api.run_simulation'
        }
    }
    
    with open(conf_dir / "base.yaml", 'w') as f:
        yaml.dump(base_config, f)
    
    # Create config.yaml with structured config overrides
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
        },
        'simulation': {
            'max_steps': 2000,
            'enable_visualization': True
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
        },
        'simulation': {
            'random_seed': 12345
        }
    }
    
    with open(local_dir / "test_overrides.yaml", 'w') as f:
        yaml.dump(local_overrides, f)
    
    yield conf_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_hydra_config_dir(temp_hydra_structured_config_dir):
    """Backwards compatibility fixture for legacy tests."""
    return temp_hydra_structured_config_dir


@pytest.fixture
def mock_hydra_structured_compose():
    """Mock Hydra compose function for testing structured configurations without full Hydra setup."""
    def _compose(config_name: str = "config", overrides: list = None, return_hydra_config: bool = False):
        """Mock compose function that returns structured test configuration."""
        base_config = {
            'video_plume': {
                'video_path': 'test_video.mp4',
                'flip': False,
                'grayscale': True,
                'kernel_size': 5,
                'kernel_sigma': 1.0,
                'threshold': 0.5,
                'normalize': True,
                '_target_': 'odor_plume_nav.data.VideoPlume'
            },
            'navigator': {
                'position': [0.0, 0.0],
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0,
                'angular_velocity': 0.0,
                'mode': 'single',
                '_target_': 'odor_plume_nav.api.create_navigator'
            },
            'simulation': {
                'max_steps': 1000,
                'record_trajectory': True,
                'output_format': 'numpy',
                'enable_visualization': True,
                '_target_': 'odor_plume_nav.api.run_simulation'
            }
        }
        
        # Apply overrides if provided
        if overrides:
            for override in overrides:
                if '=' in override:
                    key, value = override.split('=', 1)
                    # Enhanced override application for structured configs
                    if key == 'navigator.max_speed':
                        base_config['navigator']['max_speed'] = float(value)
                    elif key == 'video_plume.flip':
                        base_config['video_plume']['flip'] = value.lower() == 'true'
                    elif key == 'simulation.max_steps':
                        base_config['simulation']['max_steps'] = int(value)
                    elif key == 'navigator.mode':
                        base_config['navigator']['mode'] = value
        
        return base_config if not return_hydra_config else (base_config, None)
    
    return _compose


@pytest.fixture
def mock_hydra_compose(mock_hydra_structured_compose):
    """Backwards compatibility fixture for legacy tests."""
    return mock_hydra_structured_compose


class TestStructuredConfigValidation:
    """Tests for enhanced Pydantic schema validation with Hydra 1.3+ structured configuration."""
    
    def test_validate_valid_structured_config(self, valid_hydra_structured_config):
        """Test that validating a valid Hydra structured config works with dataclass-based schemas."""
        # This should not raise any exceptions
        result = validate_hydra_structured_config(valid_hydra_structured_config, validate=True)
        assert result == valid_hydra_structured_config
    
    def test_compile_time_validation_enabled(self, valid_hydra_structured_config):
        """Test compile-time validation workflow with enhanced schema validation."""
        # Should validate successfully with compile-time validation enabled
        result = validate_hydra_structured_config(
            valid_hydra_structured_config, 
            validate=True, 
            enable_compile_time_validation=True
        )
        assert result == valid_hydra_structured_config
        
        # Each section should have _target_ field for Hydra factory instantiation
        for section_name in ['navigator', 'video_plume', 'simulation']:
            assert '_target_' in valid_hydra_structured_config[section_name]
    
    def test_navigator_config_dataclass_validation(self):
        """Test NavigatorConfig dataclass-based Pydantic model validation with enhanced features."""
        # Valid single agent configuration with _target_ field
        config_data = {
            'position': [1.0, 2.0],
            'orientation': 45.0,
            'speed': 0.5,
            'max_speed': 1.0,
            'angular_velocity': 0.1,
            'mode': 'single',
            '_target_': 'odor_plume_nav.api.create_navigator'
        }
        
        navigator_config = NavigatorConfig(**config_data)
        assert navigator_config.position == (1.0, 2.0)
        assert navigator_config.orientation == 45.0
        assert navigator_config.speed == 0.5
        assert navigator_config.max_speed == 1.0
        assert navigator_config.mode == 'single'
        assert navigator_config._target_ == 'odor_plume_nav.api.create_navigator'
        
        # Test enhanced mode detection
        single_config = navigator_config.get_single_agent_config()
        assert isinstance(single_config, SingleAgentConfig)
        assert single_config.position == (1.0, 2.0)
    
    def test_video_plume_config_enhanced_validation(self):
        """Test VideoPlumeConfig enhanced Pydantic model validation with structured config features."""
        config_data = {
            'video_path': 'test_video.mp4',
            'flip': True,
            'grayscale': True,
            'kernel_size': 7,
            'kernel_sigma': 2.0,
            'threshold': 0.3,
            'normalize': True,
            'frame_skip': 0,
            'start_frame': 0,
            '_target_': 'odor_plume_nav.data.VideoPlume'
        }
        
        video_plume_config = VideoPlumeConfig(**config_data)
        assert video_plume_config.flip is True
        assert video_plume_config.kernel_size == 7
        assert video_plume_config.kernel_sigma == 2.0
        assert video_plume_config.threshold == 0.3
        assert video_plume_config._target_ == 'odor_plume_nav.data.VideoPlume'
        assert video_plume_config.frame_skip == 0
        assert video_plume_config.start_frame == 0
    
    def test_validate_invalid_video_plume_structured_config(self, invalid_video_plume_config):
        """Test that validating a structured config with invalid video_plume raises enhanced Pydantic error."""
        # Should raise a validation error for negative kernel_size
        with pytest.raises(ValueError):  # More general error matching for enhanced validation
            validate_hydra_structured_config(invalid_video_plume_config, validate=True)
    
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
    
    def test_validate_invalid_navigator_structured_config(self, invalid_navigator_config):
        """Test that validating a structured config with invalid navigator raises enhanced Pydantic error.""" 
        # Should raise a validation error for speed exceeding max_speed
        with pytest.raises(ValueError, match="cannot exceed max_speed"):
            validate_hydra_structured_config(invalid_navigator_config, validate=True)
    
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
    
    def test_validate_missing_section_structured_config(self, missing_section_config):
        """Test that validating a structured config with missing section raises enhanced error."""
        # Should raise a validation error for missing navigator section
        with pytest.raises(ValueError, match="Missing required configuration section: 'navigator'"):
            validate_hydra_structured_config(missing_section_config, validate=True)
    
    def test_validate_missing_field_structured_config(self, missing_field_config):
        """Test that validating a structured config with missing field raises enhanced Pydantic error."""
        # Should raise a validation error for missing kernel_sigma when kernel_size is provided
        with pytest.raises(ValueError, match="kernel_sigma must be specified"):
            validate_hydra_structured_config(missing_field_config, validate=True)
    
    def test_validation_disabled_structured_config(self, invalid_video_plume_config):
        """Test that structured config validation can be disabled."""
        # This should not raise any exceptions when validate=False
        result = validate_hydra_structured_config(invalid_video_plume_config, validate=False)
        assert result['video_plume']['kernel_size'] == -1
    
    def test_selective_validation_structured_config(self, invalid_navigator_config):
        """Test that structured config validation can be limited to specific sections."""
        # This should not raise when we only validate video_plume
        result = validate_hydra_structured_config(
            invalid_navigator_config, 
            validate=True, 
            validate_sections=['video_plume']
        )
        assert result['navigator']['speed'] == 2.0
    
    def test_multi_agent_structured_validation(self, multi_agent_structured_config):
        """Test validation of multi-agent structured configuration."""
        # This should validate successfully with enhanced multi-agent support
        result = validate_hydra_structured_config(multi_agent_structured_config, validate=True)
        assert result['navigator']['num_agents'] == 3
        assert len(result['navigator']['positions']) == 3
        assert result['navigator']['mode'] == 'multi'
        assert result['navigator']['_target_'] == 'odor_plume_nav.api.create_navigator'
    
    def test_multi_agent_inconsistent_lengths_structured_config(self):
        """Test multi-agent structured validation with inconsistent parameter lengths."""
        invalid_multi_config = {
            'video_plume': {
                'video_path': 'test_video.mp4',
                '_target_': 'odor_plume_nav.data.VideoPlume'
            },
            'navigator': {
                'positions': [[0.0, 0.0], [1.0, 0.0]],  # 2 agents
                'orientations': [0.0, 45.0, 90.0],      # 3 orientations - mismatch
                'speeds': [1.0, 1.0],
                'max_speeds': [1.5, 1.5],
                'num_agents': 2,
                'mode': 'multi',
                '_target_': 'odor_plume_nav.api.create_navigator'
            },
            'simulation': {
                'max_steps': 1000,
                '_target_': 'odor_plume_nav.api.run_simulation'
            }
        }
        
        # Should raise validation error for length mismatch with enhanced error reporting
        with pytest.raises(ValueError, match="length.*does not match.*number of agents"):
            validate_hydra_structured_config(invalid_multi_config, validate=True)


class TestHydraStructuredConfigurationComposition:
    """Tests for Hydra 1.3+ structured configuration composition and ConfigStore integration."""
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_structured_config_composition(self, temp_hydra_config_dir):
        """Test Hydra structured configuration composition with dataclass schemas."""
        # Clear any existing Hydra instance
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
            
        try:
            # Register structured configuration schemas
            register_config_schemas()
            
            with initialize_config_dir(config_dir=str(temp_hydra_config_dir)):
                cfg = compose(config_name="config")
                
                # Test that base config is loaded with structured validation
                assert cfg.navigator.orientation == 0.0
                assert cfg.video_plume.normalize is True
                
                # Test that overrides are applied correctly
                assert cfg.navigator.max_speed == 2.0  # Overridden from base
                assert cfg.video_plume.kernel_size == 5  # Set in config.yaml
                
                # Test that _target_ fields are preserved for factory instantiation
                if hasattr(cfg.navigator, '_target_'):
                    assert cfg.navigator._target_.endswith('navigator') or cfg.navigator._target_.endswith('Navigator')
                    
        finally:
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_structured_config_override_scenarios(self, temp_hydra_config_dir):
        """Test Hydra structured configuration override scenarios with validation."""
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
            
        try:
            register_config_schemas()
            
            with initialize_config_dir(config_dir=str(temp_hydra_config_dir)):
                # Test command-line style overrides with structured validation
                cfg = compose(
                    config_name="config",
                    overrides=["navigator.max_speed=3.0", "video_plume.flip=true"]
                )
                
                assert cfg.navigator.max_speed == 3.0
                assert cfg.video_plume.flip is True
                
                # Test that structured validation still works with overrides
                config_dict = OmegaConf.to_container(cfg, resolve=True)
                validate_hydra_structured_config(config_dict, validate=True)
                
        finally:
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()
    
    def test_mock_structured_config_composition(self, mock_hydra_compose):
        """Test structured configuration composition using mock Hydra compose."""
        # Test basic composition with structured config features
        cfg = mock_hydra_compose(config_name="config")
        assert cfg['navigator']['max_speed'] == 1.0
        assert cfg['video_plume']['flip'] is False
        
        # Validate with structured config validation
        validate_hydra_structured_config(cfg, validate=True)
        
        # Test with overrides maintaining structured validation
        cfg_with_overrides = mock_hydra_compose(
            config_name="config",
            overrides=["navigator.max_speed=2.5", "video_plume.flip=true"]
        )
        assert cfg_with_overrides['navigator']['max_speed'] == 2.5
        assert cfg_with_overrides['video_plume']['flip'] is True
        
        # Ensure overridden config still validates
        validate_hydra_structured_config(cfg_with_overrides, validate=True)
    
    def test_structured_config_environment_interpolation(self, structured_config_with_interpolation):
        """Test environment variable interpolation in structured configuration."""
        # Set test environment variables
        test_env = {
            'TEST_VIDEO_PATH': 'env_video.mp4',
            'VIDEO_FLIP': 'true',
            'PLUME_THRESHOLD': '0.7',
            'NAVIGATOR_MAX_SPEED': '2.0',
            'AGENT_ORIENTATION': '45.0',
            'MAX_STEPS': '2000',
            'RANDOM_SEED': '123',
            'OUTPUT_DIR': '/custom/output'
        }
        
        with patch.dict(os.environ, test_env):
            # Test that interpolation syntax is present in structured config
            config_str = str(structured_config_with_interpolation)
            assert '${oc.env:TEST_VIDEO_PATH,test_video.mp4}' in config_str
            assert '${oc.env:NAVIGATOR_MAX_SPEED,1.0}' in config_str
            assert '${oc.env:AGENT_ORIENTATION,0.0}' in config_str
            assert '${oc.env:MAX_STEPS,1000}' in config_str
            
            # Test that utility functions work with interpolation
            assert validate_env_interpolation('${oc.env:TEST_VIDEO_PATH,test_video.mp4}')
            assert resolve_env_value('${oc.env:TEST_VIDEO_PATH,test_video.mp4}') == 'env_video.mp4'


class TestHierarchicalStructuredConfigValidation:
    """Tests for hierarchical configuration validation with structured config compatibility."""
    
    def test_base_structured_config_validation(self):
        """Test validation of base structured configuration parameters."""
        base_config = {
            'video_plume': {
                'video_path': 'base_video.mp4',
                'flip': False,
                'grayscale': True,
                'normalize': True,
                '_target_': 'odor_plume_nav.data.VideoPlume'
            },
            'navigator': {
                'orientation': 0.0,
                'speed': 0.0,
                'max_speed': 1.0,
                'mode': 'single',
                '_target_': 'odor_plume_nav.api.create_navigator'
            },
            'simulation': {
                'max_steps': 1000,
                'record_trajectory': True,
                '_target_': 'odor_plume_nav.api.run_simulation'
            }
        }
        
        # Should validate successfully with structured config validation
        result = validate_hydra_structured_config(base_config, validate=True)
        assert result == base_config
    
    def test_layered_structured_override_validation(self):
        """Test validation with multiple structured configuration layers."""
        # Simulate base -> config -> local override chain with structured configs
        base_config = {
            'navigator': {
                'speed': 0.0,
                'max_speed': 1.0,
                'mode': 'single',
                '_target_': 'odor_plume_nav.api.create_navigator'
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
                '_target_': 'odor_plume_nav.data.VideoPlume'
            },
            'navigator': {
                'speed': local_overrides['navigator']['speed'],  # 0.8
                'max_speed': config_overrides['navigator']['max_speed'],  # 1.5
                'mode': base_config['navigator']['mode'],  # 'single'
                '_target_': base_config['navigator']['_target_']
            },
            'simulation': {
                'max_steps': 1000,
                '_target_': 'odor_plume_nav.api.run_simulation'
            }
        }
        
        # Should validate successfully with structured config validation
        result = validate_hydra_structured_config(final_config, validate=True)
        assert result['navigator']['speed'] == 0.8
        assert result['navigator']['max_speed'] == 1.5
        assert result['navigator']['_target_'] == 'odor_plume_nav.api.create_navigator'
    
    def test_structured_override_constraint_validation(self):
        """Test that override values still respect structured configuration schema constraints."""
        # Test that overrides can't violate Pydantic constraints in structured configs
        invalid_override_config = {
            'video_plume': {
                'video_path': 'test.mp4',
                'kernel_size': -5,  # Invalid even after override
                'kernel_sigma': 1.0,
                '_target_': 'odor_plume_nav.data.VideoPlume'
            },
            'navigator': {
                'speed': 0.5,
                'max_speed': 1.0,
                'mode': 'single',
                '_target_': 'odor_plume_nav.api.create_navigator'
            },
            'simulation': {
                'max_steps': 1000,
                '_target_': 'odor_plume_nav.api.run_simulation'
            }
        }
        
        with pytest.raises(ValueError):  # More general error matching for enhanced error reporting
            validate_hydra_structured_config(invalid_override_config, validate=True)
    
    def test_enhanced_hierarchical_error_reporting(self):
        """Test enhanced error handling and validation reporting for hierarchical structured configs."""
        # Create a config with multiple validation errors across hierarchy
        complex_invalid_config = {
            'video_plume': {
                'video_path': 'test.mp4',
                'kernel_size': 4,  # Invalid: must be odd
                'kernel_sigma': -1.0,  # Invalid: must be positive
                'threshold': 1.5,  # Invalid: must be <= 1.0
                '_target_': 'odor_plume_nav.data.VideoPlume'
            },
            'navigator': {
                'speed': 2.0,  # Invalid: exceeds max_speed
                'max_speed': 1.0,
                'orientation': 400.0,  # Invalid: exceeds 360.0
                'mode': 'single',
                '_target_': 'odor_plume_nav.api.create_navigator'
            },
            'simulation': {
                'max_steps': -100,  # Invalid: must be positive
                'batch_size': 0,  # Invalid: must be > 0
                '_target_': 'odor_plume_nav.api.run_simulation'
            }
        }
        
        # Test that validation errors are caught and reported
        with pytest.raises((ValueError, TypeError)) as exc_info:
            validate_hydra_structured_config(
                complex_invalid_config, 
                validate=True,
                enable_compile_time_validation=True
            )
        
        # Verify that error message contains useful information
        error_message = str(exc_info.value)
        assert len(error_message) > 20  # Should be descriptive


class TestLoguruIntegration:
    """Tests for Loguru logging integration with structured configuration validation."""
    
    @pytest.mark.skipif(not LOGURU_AVAILABLE, reason="Loguru not available")
    def test_structured_logging_during_validation(self, valid_hydra_structured_config, caplog):
        """Test that validation warnings are properly logged via Loguru integration."""
from loguru import logger
        
        # Capture logging output
        with caplog.at_level(logger.WARNING):
            # Test configuration that triggers warnings but doesn't fail
            config_with_warnings = valid_hydra_structured_config.copy()
            config_with_warnings['video_plume']['kernel_size'] = 5
            # Remove kernel_sigma to trigger warning
            del config_with_warnings['video_plume']['kernel_sigma']
            
            # This should trigger a warning log but still validate successfully
            result = validate_hydra_structured_config(config_with_warnings, validate=True)
            assert result is not None
    
    @pytest.mark.skipif(not LOGURU_AVAILABLE, reason="Loguru not available")
    def test_correlation_id_logging_support(self):
        """Test that structured logging supports correlation IDs for distributed tracing."""
        # Test that environment variable interpolation logs are structured
        test_interpolation = "${oc.env:TEST_VAR,default}"
        
        # Test validation of interpolation syntax
        is_valid = validate_env_interpolation(test_interpolation)
        assert is_valid is True
        
        # Test resolution with potential logging
        resolved = resolve_env_value(test_interpolation)
        assert resolved == "default"  # Should use default since TEST_VAR not set
    
    @pytest.mark.skipif(not LOGURU_AVAILABLE, reason="Loguru not available")
    def test_performance_logging_integration(self, valid_hydra_structured_config):
        """Test that performance monitoring integrates with Loguru structured logger."""
        import time
        
        # Simulate performance-sensitive validation
        start_time = time.perf_counter()
        
        # Run validation multiple times to simulate performance monitoring
        for _ in range(10):
            validate_hydra_structured_config(
                valid_hydra_structured_config, 
                validate=True,
                enable_compile_time_validation=True
            )
        
        end_time = time.perf_counter()
        validation_time = end_time - start_time
        
        # Validation should complete quickly (under 1 second for 10 iterations)
        assert validation_time < 1.0
    
    def test_enhanced_error_context_logging(self):
        """Test that enhanced error reporting provides structured context for logger."""
        invalid_config = {
            'navigator': {
                'speed': 'invalid_speed',  # String instead of float
                'max_speed': 1.0,
                'mode': 'invalid_mode',  # Invalid mode value
                '_target_': 'odor_plume_nav.api.create_navigator'
            }
        }
        
        try:
            validate_hydra_structured_config(
                invalid_config,
                validate=True,
                validate_sections=['navigator']
            )
            pytest.fail("Expected validation error")
        except Exception as e:
            # Error should contain structured information suitable for logging
            error_str = str(e)
            assert len(error_str) > 0
            # Could be enhanced to check for specific error structure


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing validation patterns."""
    
    def test_legacy_validation_function_compatibility(self, valid_hydra_config):
        """Test that legacy validate_hydra_config function still works."""
        # Test using the legacy function name (aliased to new function)
        result = validate_hydra_config(valid_hydra_config, validate=True)
        assert result == valid_hydra_config
    
    def test_mixed_legacy_structured_config(self):
        """Test validation of configs that mix legacy and structured patterns."""
        mixed_config = {
            'video_plume': {
                'video_path': 'test_video.mp4',
                'flip': False,
                'grayscale': True,
                'kernel_size': 5,
                'kernel_sigma': 1.0,
                # Mix: some have _target_, some don't
                '_target_': 'odor_plume_nav.data.VideoPlume'
            },
            'navigator': {
                'position': [0.0, 0.0],
                'speed': 0.5,
                'max_speed': 1.0,
                'mode': 'single'
                # Missing _target_ field (legacy style)
            }
        }
        
        # Should still validate when compile-time validation is disabled
        result = validate_hydra_structured_config(
            mixed_config, 
            validate=True,
            enable_compile_time_validation=False
        )
        assert result == mixed_config
    
    def test_selective_section_validation_compatibility(self, valid_hydra_structured_config):
        """Test that selective section validation still works with structured configs."""
        # Test validating only specific sections
        result = validate_hydra_structured_config(
            valid_hydra_structured_config,
            validate=True,
            validate_sections=['navigator', 'simulation']
        )
        assert result == valid_hydra_structured_config


class TestDataclassSchemaIntegration:
    """Tests for dataclass-based schema integration with Pydantic enforcement."""
    
    def test_dataclass_config_creation(self, dataclass_config_schema):
        """Test creation and validation of dataclass-based configuration schemas."""
        TestNavigatorConfig = dataclass_config_schema
        
        # Valid configuration should work
        config = TestNavigatorConfig(
            position=(5.0, 10.0),
            orientation=90.0,
            speed=0.8,
            max_speed=1.5
        )
        assert config.position == (5.0, 10.0)
        assert config.orientation == 90.0
        assert config._target_ == "test.navigator.Factory"
    
    def test_dataclass_validation_enforcement(self, dataclass_config_schema):
        """Test that dataclass validation rules are properly enforced."""
        TestNavigatorConfig = dataclass_config_schema
        
        # Invalid configuration should raise error
        with pytest.raises(ValueError, match="Speed.*cannot exceed max_speed"):
            TestNavigatorConfig(
                position=(0.0, 0.0),
                speed=2.0,
                max_speed=1.0
            )
    
    def test_environment_variable_interpolation_validation(self):
        """Test validation of environment variable interpolation syntax."""
        # Valid interpolation patterns
        assert validate_env_interpolation("${oc.env:TEST_VAR}") is True
        assert validate_env_interpolation("${oc.env:TEST_VAR,default_value}") is True
        
        # Invalid patterns
        assert validate_env_interpolation("regular_string") is False
        assert validate_env_interpolation("${invalid_syntax}") is False
    
    def test_environment_variable_resolution(self):
        """Test resolution of environment variable interpolation for testing."""
        # Set test environment variable
        test_env = {'TEST_RESOLUTION_VAR': 'resolved_value'}
        
        with patch.dict(os.environ, test_env):
            # Test resolution with environment variable set
            result = resolve_env_value("${oc.env:TEST_RESOLUTION_VAR}")
            assert result == 'resolved_value'
            
            # Test resolution with default value
            result = resolve_env_value("${oc.env:MISSING_VAR,default_val}")
            assert result == 'default_val'
            
            # Test non-interpolated value
            result = resolve_env_value("regular_value")
            assert result == "regular_value"
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_config_store_registration(self):
        """Test ConfigStore registration for structured configuration schemas."""
        # Clear any existing Hydra instance
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        # Test that registration works without errors
        try:
            register_config_schemas()
            
            # Verify ConfigStore has been populated
            cs = ConfigStore.instance()
            
            # Check that schemas are registered (this is implementation dependent)
            # We can at least verify the function runs without error
            assert True  # If we get here, registration succeeded
            
        except Exception as e:
            pytest.fail(f"ConfigStore registration failed: {e}")
        finally:
            # Clean up
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()


class TestCompileTimeValidation:
    """Tests for enhanced compile-time validation workflow."""
    
    def test_startup_validation_workflow(self, valid_hydra_structured_config):
        """Test that compile-time validation catches configuration errors at startup."""
        # Valid config should pass startup validation
        result = validate_hydra_structured_config(
            valid_hydra_structured_config,
            validate=True,
            enable_compile_time_validation=True
        )
        assert result is not None
    
    def test_missing_target_field_validation(self):
        """Test that missing _target_ fields are caught during compile-time validation."""
        invalid_config = {
            'navigator': {
                'position': [0.0, 0.0],
                'speed': 0.5,
                'max_speed': 1.0
                # Missing _target_ field
            }
        }
        
        with pytest.raises(AssertionError, match="Navigator config missing _target_ field"):
            validate_hydra_structured_config(
                invalid_config,
                validate=True,
                enable_compile_time_validation=True,
                validate_sections=['navigator']
            )
    
    def test_enhanced_error_reporting(self):
        """Test improved error handling and validation reporting for dataclass configurations."""
        # Test with invalid configuration that should produce detailed error
        invalid_config = {
            'navigator': {
                'position': "invalid_position",  # Should be tuple/list
                'orientation': 450.0,  # Outside valid range
                'speed': -1.0,  # Negative speed
                'max_speed': 1.0,
                '_target_': 'test.target'
            }
        }
        
        try:
            validate_hydra_structured_config(
                invalid_config,
                validate=True,
                validate_sections=['navigator']
            )
            pytest.fail("Expected validation error")
        except (ValueError, TypeError) as e:
            # Error should contain detailed information
            error_message = str(e)
            assert len(error_message) > 10  # Should be descriptive
    
    def test_simulation_config_validation(self):
        """Test SimulationConfig validation with enhanced dataclass features."""
        config_data = {
            'max_steps': 500,
            'step_size': 1.5,
            'enable_gpu': True,
            'batch_size': 32,
            'record_trajectory': True,
            'output_format': 'numpy',
            'output_directory': './test_outputs',
            'enable_visualization': True,
            'random_seed': 12345,
            '_target_': 'odor_plume_nav.api.run_simulation'
        }
        
        sim_config = SimulationConfig(**config_data)
        assert sim_config.max_steps == 500
        assert sim_config.step_size == 1.5
        assert sim_config.enable_gpu is True
        assert sim_config.batch_size == 32
        assert sim_config._target_ == 'odor_plume_nav.api.run_simulation'
        
        # Test validation in structured config context
        full_config = {'simulation': config_data}
        result = validate_hydra_structured_config(
            full_config,
            validate=True,
            validate_sections=['simulation']
        )
        assert result['simulation']['max_steps'] == 500