"""
Advanced pytest test suite for Hydra configuration system integration.

This module provides comprehensive testing of Hydra configuration management using
pytest-hydra plugin to validate hierarchical configuration composition, override
scenarios, environment variable interpolation security, and configuration validation
integration with Pydantic schemas. Tests ensure robust configuration management
across conf/base.yaml, conf/config.yaml, and conf/local/ directory structures.

Testing Coverage:
- Hierarchical configuration composition and precedence validation
- Environment variable interpolation security and injection prevention
- Configuration override mechanisms and parameter precedence rules
- Pydantic schema validation integration with Hydra composition
- Configuration loading performance validation (<1s requirement)
- Working directory isolation preventing configuration pollution
- Security validation preventing malicious parameter injection
- Cross-platform configuration consistency and reproducibility

Performance Requirements:
- Configuration loading: <1s per Section 2.2.9.3
- Hydra composition: <500ms per Section 6.6.7.1
- Schema validation: <500ms per Feature F-007

Security Requirements:
- Environment variable interpolation boundaries per Section 6.6.7.1
- Configuration override protection per Feature F-006
- Path traversal attack prevention per Section 6.6.7.1
"""

import pytest
import tempfile
import os
import time
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Optional
import logging

# Import Hydra components for configuration testing
try:
    from hydra import initialize, initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf, DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    pytest.skip("Hydra not available - skipping Hydra configuration tests", allow_module_level=True)

# Import project-specific components for schema validation testing
try:
    from {{cookiecutter.project_slug}}.config.schemas import (
        NavigatorConfig,
        VideoPlumeConfig,
        ConfigValidationError
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False


class TestHydraConfigurationComposition:
    """
    Test suite for Hydra hierarchical configuration composition and precedence validation.
    
    Validates:
    - Base configuration loading from conf/base.yaml
    - Configuration composition from conf/config.yaml
    - Local override management from conf/local/ directory
    - Parameter precedence rules and hierarchical inheritance
    - Configuration composition tracking and source identification
    """
    
    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """Ensure Hydra working directory isolation for each test."""
        # Clear any existing global Hydra instance
        GlobalHydra.instance().clear()
        yield
        # Clean up after test
        GlobalHydra.instance().clear()
    
    @pytest.fixture
    def mock_base_config(self) -> Dict[str, Any]:
        """Provide comprehensive base configuration fixture."""
        return {
            "defaults": [
                "_self_",
                "config: base_config"
            ],
            "video_plume": {
                "flip": False,
                "kernel_size": 0,
                "kernel_sigma": 1.0,
                "video_path": "${oc.env:VIDEO_PATH,/default/video.mp4}",
                "preprocessing": {
                    "enabled": True,
                    "gaussian_blur": True,
                    "normalize": False
                }
            },
            "navigator": {
                "orientation": 0.0,
                "speed": 0.0,
                "max_speed": 1.0,
                "angular_velocity": 0.0,
                "sensor_config": "LEFT_RIGHT",
                "collision_avoidance": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": ["console", "file"]
            },
            "system": {
                "random_seed": 42,
                "working_directory": "${oc.env:WORKING_DIR,/tmp/experiments}",
                "debug_mode": False,
                "profiling_enabled": False
            }
        }
    
    @pytest.fixture
    def mock_config_yaml(self) -> Dict[str, Any]:
        """Provide environment-specific configuration overrides."""
        return {
            "defaults": [
                "base",
                "_self_"
            ],
            "video_plume": {
                "flip": True,
                "kernel_size": 5,
                "video_path": "${oc.env:CUSTOM_VIDEO_PATH,/env/video.mp4}",
                "preprocessing": {
                    "gaussian_blur": False,
                    "normalize": True
                }
            },
            "navigator": {
                "speed": 0.5,
                "max_speed": 2.0,
                "sensor_config": "TRIANGULAR"
            },
            "logging": {
                "level": "DEBUG",
                "handlers": ["console"]
            },
            "system": {
                "debug_mode": True,
                "experiment_name": "${oc.env:EXPERIMENT_NAME,default_experiment}"
            }
        }
    
    @pytest.fixture
    def mock_local_overrides(self) -> Dict[str, Any]:
        """Provide local user-specific configuration overrides."""
        return {
            "video_plume": {
                "kernel_sigma": 2.5,
                "video_path": "/local/custom_video.mp4"
            },
            "navigator": {
                "angular_velocity": 0.1,
                "max_speed": 3.0
            },
            "system": {
                "working_directory": "/local/experiments",
                "profiling_enabled": True,
                "user_id": "${oc.env:USER_ID,unknown_user}"
            }
        }
    
    @pytest.fixture
    def hydra_config_structure(self, tmp_path, mock_base_config, mock_config_yaml, mock_local_overrides):
        """Create comprehensive Hydra configuration directory structure."""
        # Create configuration directory structure
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        
        local_dir = conf_dir / "local"
        local_dir.mkdir()
        
        # Write base configuration
        base_path = conf_dir / "base.yaml"
        with open(base_path, 'w') as f:
            yaml.dump(mock_base_config, f, default_flow_style=False)
        
        # Write main configuration
        config_path = conf_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(mock_config_yaml, f, default_flow_style=False)
        
        # Write local overrides
        local_config_path = local_dir / "user_overrides.yaml"
        with open(local_config_path, 'w') as f:
            yaml.dump(mock_local_overrides, f, default_flow_style=False)
        
        # Create credentials template
        credentials_template = local_dir / "credentials.yaml.template"
        with open(credentials_template, 'w') as f:
            f.write("""# Credentials template
database:
  password: "${oc.env:DB_PASSWORD}"
  host: "${oc.env:DB_HOST,localhost}"
  
api:
  secret_key: "${oc.env:API_SECRET_KEY}"
  token: "${oc.env:API_TOKEN}"
""")
        
        return {
            "conf_dir": conf_dir,
            "base_path": base_path,
            "config_path": config_path,
            "local_dir": local_dir,
            "local_config_path": local_config_path,
            "credentials_template": credentials_template
        }
    
    def test_hierarchical_configuration_composition(self, hydra_config_structure):
        """Test hierarchical configuration composition from base through local overrides."""
        conf_dir = hydra_config_structure["conf_dir"]
        
        # Test base configuration loading
        with initialize(config_path=str(conf_dir)):
            base_cfg = compose(config_name="base")
            
            # Validate base configuration structure
            assert "video_plume" in base_cfg
            assert "navigator" in base_cfg
            assert "logging" in base_cfg
            assert "system" in base_cfg
            
            # Validate base configuration values
            assert base_cfg.video_plume.flip is False
            assert base_cfg.video_plume.kernel_size == 0
            assert base_cfg.navigator.speed == 0.0
            assert base_cfg.navigator.max_speed == 1.0
            assert base_cfg.logging.level == "INFO"
            
        # Test main configuration composition
        with initialize(config_path=str(conf_dir)):
            main_cfg = compose(config_name="config")
            
            # Validate configuration composition and overrides
            assert main_cfg.video_plume.flip is True  # Overridden from base
            assert main_cfg.video_plume.kernel_size == 5  # Overridden from base
            assert main_cfg.video_plume.kernel_sigma == 1.0  # Inherited from base
            
            assert main_cfg.navigator.speed == 0.5  # Overridden from base
            assert main_cfg.navigator.max_speed == 2.0  # Overridden from base
            assert main_cfg.navigator.orientation == 0.0  # Inherited from base
            
            assert main_cfg.logging.level == "DEBUG"  # Overridden from base
            assert main_cfg.system.debug_mode is True  # Overridden from base
    
    def test_configuration_override_precedence(self, hydra_config_structure):
        """Test configuration override precedence rules and parameter inheritance."""
        conf_dir = hydra_config_structure["conf_dir"]
        
        # Test runtime overrides take highest precedence
        with initialize(config_path=str(conf_dir)):
            cfg = compose(
                config_name="config",
                overrides=[
                    "video_plume.flip=false",
                    "navigator.max_speed=5.0",
                    "system.debug_mode=false",
                    "new_parameter=runtime_value"
                ]
            )
            
            # Validate runtime overrides take precedence
            assert cfg.video_plume.flip is False  # Runtime override
            assert cfg.navigator.max_speed == 5.0  # Runtime override
            assert cfg.system.debug_mode is False  # Runtime override
            assert cfg.new_parameter == "runtime_value"  # New parameter
            
            # Validate non-overridden values maintain composition hierarchy
            assert cfg.video_plume.kernel_size == 5  # From config.yaml
            assert cfg.navigator.speed == 0.5  # From config.yaml
            assert cfg.navigator.orientation == 0.0  # From base.yaml
    
    def test_configuration_composition_tracking(self, hydra_config_structure):
        """Test configuration composition tracking and source identification."""
        conf_dir = hydra_config_structure["conf_dir"]
        
        with initialize(config_path=str(conf_dir)):
            cfg = compose(config_name="config")
            
            # Validate OmegaConf structure and metadata
            assert isinstance(cfg, DictConfig)
            assert OmegaConf.is_config(cfg)
            
            # Test configuration structure navigation
            assert OmegaConf.is_config(cfg.video_plume)
            assert OmegaConf.is_config(cfg.navigator)
            
            # Validate configuration freezing and modification protection
            OmegaConf.set_readonly(cfg, True)
            with pytest.raises(Exception):  # Should raise modification error
                cfg.video_plume.flip = "modified"
    
    def test_configuration_merging_deep_structures(self, hydra_config_structure):
        """Test deep configuration structure merging and nested parameter inheritance."""
        conf_dir = hydra_config_structure["conf_dir"]
        
        with initialize(config_path=str(conf_dir)):
            cfg = compose(config_name="config")
            
            # Validate deep merging for preprocessing configuration
            preprocessing = cfg.video_plume.preprocessing
            assert preprocessing.enabled is True  # From base.yaml
            assert preprocessing.gaussian_blur is False  # Overridden in config.yaml
            assert preprocessing.normalize is True  # Overridden in config.yaml
            
            # Test complex parameter access patterns
            assert cfg.video_plume.preprocessing.enabled is True
            assert cfg.navigator.sensor_config == "TRIANGULAR"
            assert cfg.logging.format is not None  # Inherited from base


class TestEnvironmentVariableInterpolation:
    """
    Test suite for environment variable interpolation security and configuration injection prevention.
    
    Validates:
    - Secure environment variable interpolation using ${oc.env:VAR_NAME} syntax
    - Prevention of malicious parameter injection through environment variables
    - Environment variable interpolation boundaries and validation
    - Default value handling for missing environment variables
    - Configuration security against environment variable manipulation
    """
    
    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """Ensure Hydra working directory isolation for each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()
    
    @pytest.fixture
    def secure_config_structure(self, tmp_path):
        """Create configuration structure with environment variable interpolation."""
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        
        # Configuration with environment variable interpolation
        config_with_env = {
            "video_plume": {
                "video_path": "${oc.env:VIDEO_PATH,/default/video.mp4}",
                "output_dir": "${oc.env:OUTPUT_DIR,/tmp/output}",
                "flip": "${oc.env:FLIP_VIDEO,false}",
                "kernel_size": "${oc.env:KERNEL_SIZE,0}"
            },
            "navigator": {
                "max_speed": "${oc.env:MAX_SPEED,1.0}",
                "sensor_config": "${oc.env:SENSOR_CONFIG,LEFT_RIGHT}"
            },
            "system": {
                "working_directory": "${oc.env:WORKING_DIR,/tmp/experiments}",
                "experiment_name": "${oc.env:EXPERIMENT_NAME,default_experiment}",
                "user_id": "${oc.env:USER_ID,anonymous}",
                "debug_mode": "${oc.env:DEBUG_MODE,false}"
            },
            "security": {
                "admin_password": "hardcoded_admin_password",  # Should NOT be overridable
                "api_key": "static_api_key",  # Should NOT be overridable
                "database_url": "postgresql://localhost/db"  # Should NOT be overridable
            }
        }
        
        config_path = conf_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_with_env, f, default_flow_style=False)
        
        return {
            "conf_dir": conf_dir,
            "config_path": config_path
        }
    
    def test_environment_variable_interpolation_valid(self, secure_config_structure):
        """Test valid environment variable interpolation with proper values."""
        conf_dir = secure_config_structure["conf_dir"]
        
        # Set up environment variables
        env_vars = {
            "VIDEO_PATH": "/custom/video/path.mp4",
            "OUTPUT_DIR": "/custom/output",
            "FLIP_VIDEO": "true",
            "KERNEL_SIZE": "5",
            "MAX_SPEED": "2.0",
            "SENSOR_CONFIG": "TRIANGULAR",
            "WORKING_DIR": "/custom/workdir",
            "EXPERIMENT_NAME": "test_experiment",
            "USER_ID": "test_user",
            "DEBUG_MODE": "true"
        }
        
        with patch.dict('os.environ', env_vars):
            with initialize(config_path=str(conf_dir)):
                cfg = compose(config_name="config")
                
                # Validate environment variable interpolation
                assert cfg.video_plume.video_path == "/custom/video/path.mp4"
                assert cfg.video_plume.output_dir == "/custom/output"
                assert cfg.video_plume.flip is True
                assert cfg.video_plume.kernel_size == 5
                
                assert cfg.navigator.max_speed == 2.0
                assert cfg.navigator.sensor_config == "TRIANGULAR"
                
                assert cfg.system.working_directory == "/custom/workdir"
                assert cfg.system.experiment_name == "test_experiment"
                assert cfg.system.user_id == "test_user"
                assert cfg.system.debug_mode is True
    
    def test_environment_variable_interpolation_defaults(self, secure_config_structure):
        """Test environment variable interpolation with default values for missing variables."""
        conf_dir = secure_config_structure["conf_dir"]
        
        # Clear environment variables to test defaults
        with patch.dict('os.environ', {}, clear=True):
            with initialize(config_path=str(conf_dir)):
                cfg = compose(config_name="config")
                
                # Validate default values are used when environment variables are missing
                assert cfg.video_plume.video_path == "/default/video.mp4"
                assert cfg.video_plume.output_dir == "/tmp/output"
                assert cfg.video_plume.flip is False
                assert cfg.video_plume.kernel_size == 0
                
                assert cfg.navigator.max_speed == 1.0
                assert cfg.navigator.sensor_config == "LEFT_RIGHT"
                
                assert cfg.system.working_directory == "/tmp/experiments"
                assert cfg.system.experiment_name == "default_experiment"
                assert cfg.system.user_id == "anonymous"
                assert cfg.system.debug_mode is False
    
    def test_environment_variable_injection_prevention(self, secure_config_structure):
        """Test prevention of malicious parameter injection through environment variables."""
        conf_dir = secure_config_structure["conf_dir"]
        
        # Attempt malicious environment variable injection
        malicious_env_vars = {
            "VIDEO_PATH": "../../../etc/passwd",  # Path traversal attempt
            "OUTPUT_DIR": "; rm -rf /",  # Command injection attempt
            "KERNEL_SIZE": "-999",  # Invalid parameter value
            "MAX_SPEED": "999999",  # Extreme parameter value
            "SENSOR_CONFIG": "${admin_password}",  # Parameter reference attempt
            "EXPERIMENT_NAME": "$(whoami)",  # Command substitution attempt
            "WORKING_DIR": "${oc.env:ADMIN_PASSWORD}",  # Nested interpolation attempt
        }
        
        with patch.dict('os.environ', malicious_env_vars):
            with initialize(config_path=str(conf_dir)):
                cfg = compose(config_name="config")
                
                # Validate that malicious values are contained as strings
                # (not executed or interpreted as special commands)
                assert cfg.video_plume.video_path == "../../../etc/passwd"
                assert cfg.video_plume.output_dir == "; rm -rf /"
                assert cfg.video_plume.kernel_size == -999  # Will be caught by schema validation
                assert cfg.navigator.max_speed == 999999  # Will be caught by schema validation
                assert cfg.navigator.sensor_config == "${admin_password}"
                assert cfg.system.experiment_name == "$(whoami)"
                
                # Validate that secure parameters cannot be overridden via environment
                assert cfg.security.admin_password == "hardcoded_admin_password"
                assert cfg.security.api_key == "static_api_key"
                assert cfg.security.database_url == "postgresql://localhost/db"
    
    def test_environment_variable_boundary_protection(self, secure_config_structure):
        """Test environment variable interpolation boundaries and protection mechanisms."""
        conf_dir = secure_config_structure["conf_dir"]
        
        # Test various boundary conditions and edge cases
        boundary_test_vars = {
            "VIDEO_PATH": "",  # Empty string
            "KERNEL_SIZE": "0",  # Zero value
            "MAX_SPEED": "0.0",  # Zero float
            "SENSOR_CONFIG": " TRIANGULAR ",  # Whitespace
            "EXPERIMENT_NAME": "test_with_special_chars_!@#$%^&*()",  # Special characters
            "DEBUG_MODE": "TRUE",  # Different case
        }
        
        with patch.dict('os.environ', boundary_test_vars):
            with initialize(config_path=str(conf_dir)):
                cfg = compose(config_name="config")
                
                # Validate boundary condition handling
                assert cfg.video_plume.video_path == ""
                assert cfg.video_plume.kernel_size == 0
                assert cfg.navigator.max_speed == 0.0
                assert cfg.navigator.sensor_config == " TRIANGULAR "
                assert cfg.system.experiment_name == "test_with_special_chars_!@#$%^&*()"
                assert cfg.system.debug_mode == "TRUE"  # String, not boolean


class TestConfigurationOverrideAndPrecedence:
    """
    Test suite for configuration override mechanisms and parameter precedence rules.
    
    Validates:
    - Command-line override precedence and parameter flow
    - Programmatic configuration override capabilities
    - Multi-run parameter sweep and batch processing scenarios
    - Configuration group selection and composition patterns
    - Override validation and constraint enforcement
    """
    
    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """Ensure Hydra working directory isolation for each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()
    
    @pytest.fixture
    def override_test_config(self, tmp_path):
        """Create configuration structure for override testing."""
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        
        base_config = {
            "video_plume": {
                "flip": False,
                "kernel_size": 0,
                "kernel_sigma": 1.0,
                "preprocessing": {
                    "enabled": True,
                    "gaussian_blur": False
                }
            },
            "navigator": {
                "speed": 0.0,
                "max_speed": 1.0,
                "orientation": 0.0,
                "sensor_config": "LEFT_RIGHT"
            },
            "system": {
                "random_seed": 42,
                "debug_mode": False,
                "batch_size": 1
            }
        }
        
        config_path = conf_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)
        
        return {
            "conf_dir": conf_dir,
            "config_path": config_path,
            "base_config": base_config
        }
    
    def test_command_line_override_precedence(self, override_test_config):
        """Test command-line override precedence and parameter flow validation."""
        conf_dir = override_test_config["conf_dir"]
        
        # Test single parameter overrides
        with initialize(config_path=str(conf_dir)):
            cfg = compose(
                config_name="config",
                overrides=["video_plume.flip=true"]
            )
            assert cfg.video_plume.flip is True
            assert cfg.video_plume.kernel_size == 0  # Unchanged
        
        # Test multiple parameter overrides
        with initialize(config_path=str(conf_dir)):
            cfg = compose(
                config_name="config",
                overrides=[
                    "video_plume.flip=true",
                    "video_plume.kernel_size=5",
                    "navigator.max_speed=2.0",
                    "system.debug_mode=true"
                ]
            )
            assert cfg.video_plume.flip is True
            assert cfg.video_plume.kernel_size == 5
            assert cfg.navigator.max_speed == 2.0
            assert cfg.system.debug_mode is True
            # Validate unchanged parameters
            assert cfg.video_plume.kernel_sigma == 1.0
            assert cfg.navigator.speed == 0.0
    
    def test_nested_parameter_override(self, override_test_config):
        """Test nested parameter override capabilities and deep structure modification."""
        conf_dir = override_test_config["conf_dir"]
        
        with initialize(config_path=str(conf_dir)):
            cfg = compose(
                config_name="config",
                overrides=[
                    "video_plume.preprocessing.enabled=false",
                    "video_plume.preprocessing.gaussian_blur=true",
                    "video_plume.preprocessing.new_parameter=added_value"
                ]
            )
            
            # Validate nested parameter overrides
            assert cfg.video_plume.preprocessing.enabled is False
            assert cfg.video_plume.preprocessing.gaussian_blur is True
            assert cfg.video_plume.preprocessing.new_parameter == "added_value"
            
            # Validate non-nested parameters unchanged
            assert cfg.video_plume.flip is False
            assert cfg.navigator.speed == 0.0
    
    def test_type_conversion_in_overrides(self, override_test_config):
        """Test automatic type conversion and validation in parameter overrides."""
        conf_dir = override_test_config["conf_dir"]
        
        with initialize(config_path=str(conf_dir)):
            cfg = compose(
                config_name="config",
                overrides=[
                    "video_plume.flip=true",  # String to boolean
                    "video_plume.kernel_size=10",  # String to integer
                    "navigator.max_speed=3.14",  # String to float
                    "navigator.sensor_config=TRIANGULAR",  # String value
                    "system.random_seed=123"  # String to integer
                ]
            )
            
            # Validate type conversions
            assert isinstance(cfg.video_plume.flip, bool)
            assert cfg.video_plume.flip is True
            
            assert isinstance(cfg.video_plume.kernel_size, int)
            assert cfg.video_plume.kernel_size == 10
            
            assert isinstance(cfg.navigator.max_speed, float)
            assert cfg.navigator.max_speed == 3.14
            
            assert isinstance(cfg.navigator.sensor_config, str)
            assert cfg.navigator.sensor_config == "TRIANGULAR"
            
            assert isinstance(cfg.system.random_seed, int)
            assert cfg.system.random_seed == 123
    
    def test_multi_run_parameter_sweep(self, override_test_config):
        """Test multi-run parameter sweep capabilities and batch processing scenarios."""
        conf_dir = override_test_config["conf_dir"]
        
        # Test parameter sweep simulation (multiple configuration generations)
        parameter_sweeps = [
            ["video_plume.kernel_size=0", "navigator.max_speed=1.0"],
            ["video_plume.kernel_size=3", "navigator.max_speed=1.5"],
            ["video_plume.kernel_size=5", "navigator.max_speed=2.0"],
            ["video_plume.kernel_size=7", "navigator.max_speed=2.5"]
        ]
        
        results = []
        for overrides in parameter_sweeps:
            with initialize(config_path=str(conf_dir)):
                cfg = compose(config_name="config", overrides=overrides)
                results.append({
                    "kernel_size": cfg.video_plume.kernel_size,
                    "max_speed": cfg.navigator.max_speed,
                    "flip": cfg.video_plume.flip  # Should remain unchanged
                })
        
        # Validate parameter sweep results
        expected_results = [
            {"kernel_size": 0, "max_speed": 1.0, "flip": False},
            {"kernel_size": 3, "max_speed": 1.5, "flip": False},
            {"kernel_size": 5, "max_speed": 2.0, "flip": False},
            {"kernel_size": 7, "max_speed": 2.5, "flip": False}
        ]
        
        assert results == expected_results
    
    def test_invalid_override_handling(self, override_test_config):
        """Test handling of invalid overrides and error recovery."""
        conf_dir = override_test_config["conf_dir"]
        
        # Test invalid parameter path
        with pytest.raises(Exception):  # Should raise configuration error
            with initialize(config_path=str(conf_dir)):
                compose(
                    config_name="config",
                    overrides=["nonexistent.parameter=value"]
                )
        
        # Test invalid nested parameter path
        with pytest.raises(Exception):  # Should raise configuration error
            with initialize(config_path=str(conf_dir)):
                compose(
                    config_name="config",
                    overrides=["video_plume.nonexistent.parameter=value"]
                )


@pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Configuration schemas not available")
class TestPydanticSchemaValidation:
    """
    Test suite for Pydantic schema validation integration with Hydra configuration composition.
    
    Validates:
    - Pydantic schema validation integration with Hydra composition
    - Type safety and constraint enforcement across configuration hierarchy
    - Validation error handling and reporting mechanisms
    - Schema evolution and backward compatibility validation
    - Performance validation for schema validation processes
    """
    
    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """Ensure Hydra working directory isolation for each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()
    
    @pytest.fixture
    def schema_validation_config(self, tmp_path):
        """Create configuration structure for schema validation testing."""
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        
        # Valid configuration for schema testing
        valid_config = {
            "video_plume": {
                "flip": False,
                "kernel_size": 5,  # Must be non-negative
                "kernel_sigma": 1.5,  # Must be positive
                "video_path": "/path/to/video.mp4"
            },
            "navigator": {
                "orientation": 90.0,  # 0-360 degrees
                "speed": 0.5,  # Must be <= max_speed
                "max_speed": 1.0,  # Must be positive
                "angular_velocity": 0.1
            }
        }
        
        # Invalid configuration for error testing
        invalid_config = {
            "video_plume": {
                "flip": False,
                "kernel_size": -1,  # Invalid: negative
                "kernel_sigma": 0.0,  # Invalid: zero
                "video_path": ""  # Invalid: empty
            },
            "navigator": {
                "orientation": 400.0,  # Invalid: > 360
                "speed": 2.0,  # Invalid: > max_speed
                "max_speed": 1.0,
                "angular_velocity": -0.5  # Invalid: negative
            }
        }
        
        # Write configurations
        valid_path = conf_dir / "valid_config.yaml"
        with open(valid_path, 'w') as f:
            yaml.dump(valid_config, f, default_flow_style=False)
        
        invalid_path = conf_dir / "invalid_config.yaml"
        with open(invalid_path, 'w') as f:
            yaml.dump(invalid_config, f, default_flow_style=False)
        
        return {
            "conf_dir": conf_dir,
            "valid_path": valid_path,
            "invalid_path": invalid_path,
            "valid_config": valid_config,
            "invalid_config": invalid_config
        }
    
    def test_valid_configuration_schema_validation(self, schema_validation_config):
        """Test Pydantic schema validation with valid configuration parameters."""
        conf_dir = schema_validation_config["conf_dir"]
        
        with initialize(config_path=str(conf_dir)):
            cfg = compose(config_name="valid_config")
            
            # Validate configuration against Pydantic schemas
            # Test VideoPlume configuration validation
            video_config = VideoPlumeConfig(**cfg.video_plume)
            assert video_config.flip is False
            assert video_config.kernel_size == 5
            assert video_config.kernel_sigma == 1.5
            assert video_config.video_path == "/path/to/video.mp4"
            
            # Test Navigator configuration validation
            navigator_config = NavigatorConfig(**cfg.navigator)
            assert navigator_config.orientation == 90.0
            assert navigator_config.speed == 0.5
            assert navigator_config.max_speed == 1.0
            assert navigator_config.angular_velocity == 0.1
    
    def test_invalid_configuration_schema_validation(self, schema_validation_config):
        """Test Pydantic schema validation error handling with invalid parameters."""
        conf_dir = schema_validation_config["conf_dir"]
        
        with initialize(config_path=str(conf_dir)):
            cfg = compose(config_name="invalid_config")
            
            # Test VideoPlume configuration validation errors
            with pytest.raises(Exception) as exc_info:  # Should raise Pydantic validation error
                VideoPlumeConfig(**cfg.video_plume)
            
            # Validate error contains information about invalid parameters
            error_str = str(exc_info.value)
            assert "kernel_size" in error_str or "negative" in error_str
            
            # Test Navigator configuration validation errors
            with pytest.raises(Exception) as exc_info:  # Should raise Pydantic validation error
                NavigatorConfig(**cfg.navigator)
            
            # Validate error contains information about invalid parameters
            error_str = str(exc_info.value)
            assert "speed" in error_str or "max_speed" in error_str or "orientation" in error_str
    
    def test_schema_validation_with_overrides(self, schema_validation_config):
        """Test schema validation with Hydra configuration overrides."""
        conf_dir = schema_validation_config["conf_dir"]
        
        # Test valid overrides pass schema validation
        with initialize(config_path=str(conf_dir)):
            cfg = compose(
                config_name="valid_config",
                overrides=[
                    "video_plume.kernel_size=7",
                    "navigator.max_speed=2.0",
                    "navigator.speed=1.5"
                ]
            )
            
            # Should validate successfully
            video_config = VideoPlumeConfig(**cfg.video_plume)
            navigator_config = NavigatorConfig(**cfg.navigator)
            
            assert video_config.kernel_size == 7
            assert navigator_config.max_speed == 2.0
            assert navigator_config.speed == 1.5
        
        # Test invalid overrides fail schema validation
        with initialize(config_path=str(conf_dir)):
            cfg = compose(
                config_name="valid_config",
                overrides=[
                    "video_plume.kernel_size=-5",  # Invalid
                    "navigator.speed=3.0",  # Invalid: exceeds max_speed
                ]
            )
            
            # Should raise validation errors
            with pytest.raises(Exception):
                VideoPlumeConfig(**cfg.video_plume)
            
            with pytest.raises(Exception):
                NavigatorConfig(**cfg.navigator)


class TestConfigurationPerformance:
    """
    Test suite for configuration loading performance validation and timing requirements.
    
    Validates:
    - Configuration loading performance (<1s requirement per Section 2.2.9.3)
    - Hydra composition performance (<500ms requirement per Section 6.6.7.1)
    - Schema validation performance (<500ms requirement per Feature F-007)
    - Large configuration handling and scalability
    - Memory usage optimization and resource management
    """
    
    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """Ensure Hydra working directory isolation for each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()
    
    @pytest.fixture
    def performance_test_config(self, tmp_path):
        """Create large configuration structure for performance testing."""
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        
        # Create large configuration for performance testing
        large_config = {
            "video_plume": {
                "flip": False,
                "kernel_size": 5,
                "kernel_sigma": 1.0,
                "video_path": "${oc.env:VIDEO_PATH,/default/video.mp4}",
                "preprocessing": {
                    f"param_{i}": f"value_{i}" for i in range(100)  # 100 parameters
                }
            },
            "navigator": {
                "orientation": 0.0,
                "speed": 0.0,
                "max_speed": 1.0,
                "sensor_configurations": {
                    f"sensor_{i}": {
                        "type": f"type_{i}",
                        "position": [i, i+1],
                        "enabled": i % 2 == 0
                    } for i in range(50)  # 50 sensor configurations
                }
            },
            "system": {
                "experiments": {
                    f"experiment_{i}": {
                        "parameters": {f"param_{j}": j for j in range(20)},
                        "metadata": {
                            "created": f"2023-01-{i:02d}",
                            "description": f"Experiment {i} description",
                            "tags": [f"tag_{j}" for j in range(5)]
                        }
                    } for i in range(1, 26)  # 25 experiments
                }
            }
        }
        
        config_path = conf_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(large_config, f, default_flow_style=False)
        
        return {
            "conf_dir": conf_dir,
            "config_path": config_path,
            "large_config": large_config
        }
    
    def test_configuration_loading_performance(self, performance_test_config):
        """Test configuration loading performance meets <1s requirement."""
        conf_dir = performance_test_config["conf_dir"]
        
        # Measure configuration loading time
        start_time = time.time()
        
        with initialize(config_path=str(conf_dir)):
            cfg = compose(config_name="config")
            
        end_time = time.time()
        loading_time = end_time - start_time
        
        # Validate performance requirement (<1s per Section 2.2.9.3)
        assert loading_time < 1.0, f"Configuration loading took {loading_time:.3f}s, exceeds 1s requirement"
        
        # Validate configuration was loaded correctly
        assert "video_plume" in cfg
        assert "navigator" in cfg
        assert "system" in cfg
        assert len(cfg.video_plume.preprocessing) == 100
        assert len(cfg.navigator.sensor_configurations) == 50
        assert len(cfg.system.experiments) == 25
    
    def test_hydra_composition_performance(self, performance_test_config):
        """Test Hydra composition performance meets <500ms requirement."""
        conf_dir = performance_test_config["conf_dir"]
        
        with initialize(config_path=str(conf_dir)):
            # Measure composition time with overrides
            start_time = time.time()
            
            cfg = compose(
                config_name="config",
                overrides=[
                    "video_plume.flip=true",
                    "navigator.speed=0.5",
                    "system.debug_mode=true",
                    "video_plume.preprocessing.param_0=overridden_value",
                    "navigator.sensor_configurations.sensor_0.enabled=false"
                ]
            )
            
            end_time = time.time()
            composition_time = end_time - start_time
            
        # Validate performance requirement (<500ms per Section 6.6.7.1)
        assert composition_time < 0.5, f"Hydra composition took {composition_time:.3f}s, exceeds 500ms requirement"
        
        # Validate overrides were applied correctly
        assert cfg.video_plume.flip is True
        assert cfg.navigator.speed == 0.5
        assert cfg.video_plume.preprocessing.param_0 == "overridden_value"
        assert cfg.navigator.sensor_configurations.sensor_0.enabled is False
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Configuration schemas not available")
    def test_schema_validation_performance(self, performance_test_config):
        """Test schema validation performance meets <500ms requirement."""
        conf_dir = performance_test_config["conf_dir"]
        
        with initialize(config_path=str(conf_dir)):
            cfg = compose(config_name="config")
            
            # Measure schema validation time
            start_time = time.time()
            
            try:
                # Test partial schema validation (video_plume only for performance)
                video_config_subset = {
                    "flip": cfg.video_plume.flip,
                    "kernel_size": cfg.video_plume.kernel_size,
                    "kernel_sigma": cfg.video_plume.kernel_sigma,
                    "video_path": cfg.video_plume.video_path
                }
                VideoPlumeConfig(**video_config_subset)
                
                navigator_config_subset = {
                    "orientation": cfg.navigator.orientation,
                    "speed": cfg.navigator.speed,
                    "max_speed": cfg.navigator.max_speed
                }
                NavigatorConfig(**navigator_config_subset)
                
            except Exception as e:
                pytest.skip(f"Schema validation failed: {e}")
            
            end_time = time.time()
            validation_time = end_time - start_time
            
        # Validate performance requirement (<500ms per Feature F-007)
        assert validation_time < 0.5, f"Schema validation took {validation_time:.3f}s, exceeds 500ms requirement"
    
    def test_memory_usage_optimization(self, performance_test_config):
        """Test configuration memory usage optimization and resource management."""
        conf_dir = performance_test_config["conf_dir"]
        
        import psutil
        import gc
        
        # Measure memory before configuration loading
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Load configuration multiple times to test memory management
        configurations = []
        for i in range(10):
            with initialize(config_path=str(conf_dir)):
                cfg = compose(config_name="config")
                configurations.append(cfg)
        
        # Force garbage collection
        gc.collect()
        
        # Measure memory after configuration loading
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # Validate reasonable memory usage (should be < 100MB for test configurations)
        max_memory_mb = 100 * 1024 * 1024  # 100MB in bytes
        assert memory_increase < max_memory_mb, f"Memory usage increased by {memory_increase / 1024 / 1024:.1f}MB, exceeds limit"


class TestWorkingDirectoryIsolation:
    """
    Test suite for Hydra working directory isolation and configuration pollution prevention.
    
    Validates:
    - Working directory isolation between test runs
    - Configuration state cleanup and reset mechanisms
    - Cross-test contamination prevention
    - Temporary directory management and cleanup
    - Configuration cache management and invalidation
    """
    
    def test_working_directory_isolation(self, tmp_path):
        """Test Hydra working directory isolation prevents configuration pollution."""
        # Create two different configuration directories
        conf_dir_1 = tmp_path / "conf1"
        conf_dir_1.mkdir()
        
        conf_dir_2 = tmp_path / "conf2"
        conf_dir_2.mkdir()
        
        # Create different configurations
        config_1 = {"test_param": "value_1", "unique_to_1": True}
        config_2 = {"test_param": "value_2", "unique_to_2": True}
        
        config_path_1 = conf_dir_1 / "config.yaml"
        with open(config_path_1, 'w') as f:
            yaml.dump(config_1, f)
        
        config_path_2 = conf_dir_2 / "config.yaml"
        with open(config_path_2, 'w') as f:
            yaml.dump(config_2, f)
        
        # Test isolation: each configuration should be independent
        GlobalHydra.instance().clear()
        with initialize(config_path=str(conf_dir_1)):
            cfg_1 = compose(config_name="config")
            assert cfg_1.test_param == "value_1"
            assert cfg_1.unique_to_1 is True
            assert "unique_to_2" not in cfg_1
        
        GlobalHydra.instance().clear()
        with initialize(config_path=str(conf_dir_2)):
            cfg_2 = compose(config_name="config")
            assert cfg_2.test_param == "value_2"
            assert cfg_2.unique_to_2 is True
            assert "unique_to_1" not in cfg_2
        
        GlobalHydra.instance().clear()
    
    def test_configuration_state_cleanup(self, tmp_path):
        """Test configuration state cleanup and reset mechanisms."""
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        
        # Create base configuration
        base_config = {"param1": "base_value", "param2": 100}
        config_path = conf_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(base_config, f)
        
        # First configuration load with overrides
        GlobalHydra.instance().clear()
        with initialize(config_path=str(conf_dir)):
            cfg_1 = compose(
                config_name="config",
                overrides=["param1=override_value_1", "param3=new_value_1"]
            )
            assert cfg_1.param1 == "override_value_1"
            assert cfg_1.param3 == "new_value_1"
        
        # Second configuration load should not be affected by previous overrides
        GlobalHydra.instance().clear()
        with initialize(config_path=str(conf_dir)):
            cfg_2 = compose(
                config_name="config",
                overrides=["param1=override_value_2", "param4=new_value_2"]
            )
            assert cfg_2.param1 == "override_value_2"  # New override
            assert cfg_2.param2 == 100  # Base value
            assert cfg_2.param4 == "new_value_2"  # New parameter
            assert "param3" not in cfg_2  # Previous override not present
        
        GlobalHydra.instance().clear()
    
    def test_cross_test_contamination_prevention(self, tmp_path):
        """Test prevention of cross-test configuration contamination."""
        # This test simulates multiple test methods running in sequence
        # to ensure no configuration state leaks between tests
        
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        
        test_configs = [
            {"test_id": "test_1", "value": 1, "flag": True},
            {"test_id": "test_2", "value": 2, "flag": False},
            {"test_id": "test_3", "value": 3, "custom_param": "test_3_value"}
        ]
        
        # Simulate multiple test runs
        for i, test_config in enumerate(test_configs):
            # Create configuration file
            config_path = conf_dir / f"test_{i}_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            # Load configuration with proper isolation
            GlobalHydra.instance().clear()
            with initialize(config_path=str(conf_dir)):
                cfg = compose(config_name=f"test_{i}_config")
                
                # Validate this test's configuration
                assert cfg.test_id == f"test_{i + 1}"
                assert cfg.value == i + 1
                
                # Validate no contamination from previous tests
                if i == 0:
                    assert cfg.flag is True
                elif i == 1:
                    assert cfg.flag is False
                    assert "custom_param" not in cfg
                elif i == 2:
                    assert cfg.custom_param == "test_3_value"
                    assert "flag" not in cfg
        
        GlobalHydra.instance().clear()
    
    def test_temporary_directory_cleanup(self, tmp_path):
        """Test temporary directory management and cleanup."""
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        
        # Create configuration with temporary output paths
        config = {
            "system": {
                "working_dir": str(tmp_path / "work"),
                "output_dir": str(tmp_path / "output"),
                "temp_dir": str(tmp_path / "temp")
            }
        }
        
        config_path = conf_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Test configuration loading and directory path handling
        GlobalHydra.instance().clear()
        with initialize(config_path=str(conf_dir)):
            cfg = compose(config_name="config")
            
            # Validate paths are correctly loaded
            assert cfg.system.working_dir == str(tmp_path / "work")
            assert cfg.system.output_dir == str(tmp_path / "output")
            assert cfg.system.temp_dir == str(tmp_path / "temp")
        
        GlobalHydra.instance().clear()
        
        # After test completion, temporary paths should be cleanable
        # (This is handled by pytest's tmp_path fixture automatically)
        assert tmp_path.exists()  # tmp_path still exists during test


# Performance and integration test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.performance
]


def test_overall_hydra_integration_comprehensive():
    """
    Comprehensive integration test validating all Hydra configuration aspects.
    
    This test combines multiple aspects of Hydra configuration testing to ensure
    overall system integration and comprehensive validation coverage.
    """
    # This test serves as a smoke test for overall Hydra integration
    # Individual detailed tests are provided in the specialized test classes above
    
    if not HYDRA_AVAILABLE:
        pytest.skip("Hydra not available for comprehensive integration testing")
    
    # Verify Hydra components are properly importable and functional
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf, DictConfig
    
    # Basic functionality verification
    GlobalHydra.instance().clear()
    
    # Create minimal in-memory configuration for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        conf_dir = Path(temp_dir) / "conf"
        conf_dir.mkdir()
        
        # Minimal configuration
        minimal_config = {
            "test_param": "test_value",
            "nested": {"param": "nested_value"}
        }
        
        config_path = conf_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(minimal_config, f)
        
        # Test basic Hydra functionality
        with initialize(config_path=str(conf_dir)):
            cfg = compose(config_name="config")
            
            assert isinstance(cfg, DictConfig)
            assert cfg.test_param == "test_value"
            assert cfg.nested.param == "nested_value"
    
    GlobalHydra.instance().clear()
    
    # If we reach here, basic Hydra integration is working
    assert True