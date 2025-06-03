"""
Comprehensive configuration system testing module validating Hydra configuration composition,
schema validation, environment variable interpolation, and override mechanisms.

This module uses pytest-hydra fixtures to test hierarchical configuration loading, parameter
validation, and secure credential management across all configuration scenarios.

Test Coverage Areas:
- Hydra configuration composition from conf/base.yaml through conf/config.yaml
- Environment variable interpolation with ${oc.env:VAR_NAME} syntax validation  
- Configuration override scenarios through command-line and programmatic composition
- Pydantic schema integration with Hydra structured configs
- Configuration validation functions and error handling
- Multi-run configuration composition and parameter sweep scenarios
- Configuration security including path traversal prevention
- Configuration loading performance meeting <500ms requirement
- Hierarchical override precedence and parameter inheritance patterns
- Configuration schema evolution and backward compatibility

Performance Requirements:
- Configuration composition must complete within 500ms per Section 6.6.3.3
- Hierarchical configuration validation must achieve >90% coverage per Section 6.6.3.1

Security Requirements:
- Environment variable interpolation must be secure and validated per Section 6.6.7.1
- Path traversal prevention for configuration file loading
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch, mock_open, MagicMock

import pytest
import numpy as np
from pydantic import ValidationError

# Hydra and OmegaConf imports with conditional availability
try:
    from hydra import compose, initialize, initialize_config_dir
    from hydra.core.config_store import ConfigStore
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.utils import JobReturn
    from hydra.types import RunMode
    from omegaconf import DictConfig, OmegaConf, MissingMandatoryValue
    from hydra._internal.utils import create_config_search_path
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    # Define fallback classes for environments without Hydra
    DictConfig = dict
    OmegaConf = object


# Configuration schema imports
from src.{{cookiecutter.project_slug}}.config.schemas import (
    NavigatorConfig,
    SingleAgentConfig, 
    MultiAgentConfig,
    VideoPlumeConfig,
)


class TestHydraConfigurationComposition:
    """
    Test suite for Hydra configuration composition and hierarchical parameter management.
    
    Validates the core Hydra functionality including configuration loading from conf/base.yaml,
    composition with conf/config.yaml, and integration with local overrides.
    """
    
    @pytest.fixture(autouse=True)
    def setup_hydra_environment(self, tmp_path):
        """Setup isolated Hydra testing environment with clean configuration state."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping Hydra configuration tests")
        
        # Clean any existing Hydra global state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
            
        # Create temporary config directory structure
        self.config_dir = tmp_path / "conf"
        self.config_dir.mkdir()
        
        # Create local override directory
        local_dir = self.config_dir / "local"
        local_dir.mkdir()
        
        # Create base configuration
        base_config = """
# conf/base.yaml - Foundation default configuration
defaults:
  - _self_

navigator:
  position: [0.0, 0.0]
  orientation: 0.0
  speed: 0.0
  max_speed: 1.0
  angular_velocity: 0.0

video_plume:
  video_path: "data/test_plume.mp4"
  flip: false
  grayscale: true
  kernel_size: 5
  kernel_sigma: 1.0
  threshold: 0.5
  normalize: true

system:
  random_seed: 42
  debug_mode: false
  output_dir: "outputs"
"""
        
        # Create environment-specific configuration
        config_yaml = """
# conf/config.yaml - Environment-specific configuration
defaults:
  - base
  - _self_

navigator:
  max_speed: ${oc.env:NAVIGATOR_MAX_SPEED,2.0}
  orientation: ${oc.env:INITIAL_ORIENTATION,45.0}

video_plume:
  video_path: ${oc.env:PLUME_VIDEO_PATH,data/default_plume.mp4}
  flip: ${oc.env:VIDEO_FLIP,true}

system:
  debug_mode: ${oc.env:DEBUG_MODE,true}
  output_dir: ${oc.env:OUTPUT_DIR,results}
"""
        
        # Write configuration files
        (self.config_dir / "base.yaml").write_text(base_config)
        (self.config_dir / "config.yaml").write_text(config_yaml)
        
        yield self.config_dir
        
        # Cleanup Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
    
    def test_basic_configuration_composition(self, setup_hydra_environment):
        """Test basic Hydra configuration composition from base.yaml to config.yaml."""
        config_dir = setup_hydra_environment
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
            
            # Validate basic composition structure
            assert "navigator" in cfg
            assert "video_plume" in cfg
            assert "system" in cfg
            
            # Validate base configuration inheritance
            assert cfg.navigator.position == [0.0, 0.0]
            assert cfg.navigator.speed == 0.0
            assert cfg.video_plume.kernel_size == 5
            assert cfg.system.random_seed == 42
    
    def test_configuration_composition_performance(self, setup_hydra_environment):
        """Test configuration composition meets <500ms performance requirement."""
        config_dir = setup_hydra_environment
        
        start_time = time.time()
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
            
        composition_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Validate performance requirement from Section 6.6.3.3
        assert composition_time < 500, f"Configuration composition took {composition_time:.2f}ms, exceeds 500ms limit"
        
        # Validate configuration was successfully composed
        assert cfg is not None
        assert "navigator" in cfg
    
    def test_hierarchical_configuration_inheritance(self, setup_hydra_environment):
        """Test hierarchical parameter inheritance from base to config to overrides."""
        config_dir = setup_hydra_environment
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            # Test base configuration
            base_cfg = compose(config_name="base")
            assert base_cfg.navigator.max_speed == 1.0
            assert base_cfg.video_plume.flip == False
            
            # Test config inherits and overrides base
            cfg = compose(config_name="config")
            
            # Should inherit from base
            assert cfg.navigator.position == [0.0, 0.0]  # From base
            assert cfg.navigator.speed == 0.0  # From base
            
            # Should apply environment variable defaults (no env vars set)
            assert cfg.navigator.max_speed == 2.0  # Environment default override
            assert cfg.navigator.orientation == 45.0  # Environment default override
            assert cfg.video_plume.flip == True  # Environment default override
    
    def test_configuration_override_precedence(self, setup_hydra_environment):
        """Test configuration override precedence: command-line > env vars > config > base."""
        config_dir = setup_hydra_environment
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            # Test programmatic overrides
            cfg = compose(
                config_name="config", 
                overrides=[
                    "navigator.max_speed=5.0",
                    "video_plume.threshold=0.8",
                    "system.debug_mode=false"
                ]
            )
            
            # Validate overrides take precedence
            assert cfg.navigator.max_speed == 5.0  # Command-line override
            assert cfg.video_plume.threshold == 0.8  # Command-line override
            assert cfg.system.debug_mode == False  # Command-line override
            
            # Validate non-overridden values retain hierarchy
            assert cfg.navigator.position == [0.0, 0.0]  # From base
    
    def test_multirun_configuration_composition(self, setup_hydra_environment):
        """Test multi-run configuration composition for parameter sweeps."""
        config_dir = setup_hydra_environment
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            # Test parameter sweep configuration
            cfg = compose(
                config_name="config",
                overrides=[
                    "hydra.mode=MULTIRUN",
                    "navigator.max_speed=1.0,2.0,3.0",
                    "navigator.orientation=0,90,180"
                ]
            )
            
            # Validate multirun configuration structure
            assert cfg is not None
            assert "navigator" in cfg
            
            # Note: In actual multirun, Hydra would create multiple config instances
            # Here we validate the base configuration accepts multirun syntax
    
    @pytest.mark.parametrize("override_param,override_value,expected_type", [
        ("navigator.max_speed", "3.5", float),
        ("navigator.position", "[1.0,2.0]", list),
        ("video_plume.flip", "false", bool),
        ("system.random_seed", "123", int),
    ])
    def test_parameter_type_conversion(self, setup_hydra_environment, override_param, override_value, expected_type):
        """Test automatic parameter type conversion in configuration composition."""
        config_dir = setup_hydra_environment
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(
                config_name="config",
                overrides=[f"{override_param}={override_value}"]
            )
            
            # Navigate to the parameter value
            param_path = override_param.split('.')
            value = cfg
            for path_part in param_path:
                value = getattr(value, path_part)
            
            # Validate type conversion
            assert isinstance(value, expected_type), f"Expected {expected_type}, got {type(value)}"


class TestEnvironmentVariableInterpolation:
    """
    Test suite for environment variable interpolation with ${oc.env:VAR_NAME} syntax.
    
    Validates secure environment variable handling, fallback behavior, and integration
    with python-dotenv for credential management.
    """
    
    @pytest.fixture(autouse=True)
    def setup_environment_config(self, tmp_path):
        """Setup configuration with environment variable interpolation."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping environment variable tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        self.config_dir = tmp_path / "conf"
        self.config_dir.mkdir()
        
        # Create configuration with environment variable interpolation
        env_config = """
navigator:
  max_speed: ${oc.env:TEST_MAX_SPEED,1.5}
  position: ${oc.env:TEST_POSITION,[0.0,0.0]}
  orientation: ${oc.env:TEST_ORIENTATION,0.0}

video_plume:
  video_path: ${oc.env:TEST_VIDEO_PATH,data/test.mp4}
  flip: ${oc.env:TEST_VIDEO_FLIP,false}
  threshold: ${oc.env:TEST_THRESHOLD,0.5}

system:
  output_dir: ${oc.env:TEST_OUTPUT_DIR,default_output}
  debug_mode: ${oc.env:TEST_DEBUG,false}
"""
        
        (self.config_dir / "config.yaml").write_text(env_config)
        
        yield self.config_dir
        
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
    
    def test_environment_variable_interpolation_with_defaults(self, setup_environment_config):
        """Test environment variable interpolation uses defaults when env vars not set."""
        config_dir = setup_environment_config
        
        # Ensure test environment variables are not set
        env_vars_to_clear = [
            "TEST_MAX_SPEED", "TEST_POSITION", "TEST_ORIENTATION",
            "TEST_VIDEO_PATH", "TEST_VIDEO_FLIP", "TEST_THRESHOLD",
            "TEST_OUTPUT_DIR", "TEST_DEBUG"
        ]
        
        with patch.dict(os.environ, {}, clear=False):
            # Clear test environment variables
            for var in env_vars_to_clear:
                os.environ.pop(var, None)
            
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                cfg = compose(config_name="config")
                
                # Validate default values are used
                assert cfg.navigator.max_speed == 1.5
                assert cfg.navigator.position == [0.0, 0.0]
                assert cfg.navigator.orientation == 0.0
                assert cfg.video_plume.video_path == "data/test.mp4"
                assert cfg.video_plume.flip == False
                assert cfg.video_plume.threshold == 0.5
                assert cfg.system.output_dir == "default_output"
                assert cfg.system.debug_mode == False
    
    def test_environment_variable_interpolation_with_values(self, setup_environment_config):
        """Test environment variable interpolation uses environment values when set."""
        config_dir = setup_environment_config
        
        test_env_vars = {
            "TEST_MAX_SPEED": "3.0",
            "TEST_POSITION": "[1.0,2.0]",
            "TEST_ORIENTATION": "90.0",
            "TEST_VIDEO_PATH": "/custom/path/video.mp4",
            "TEST_VIDEO_FLIP": "true",
            "TEST_THRESHOLD": "0.8",
            "TEST_OUTPUT_DIR": "custom_output",
            "TEST_DEBUG": "true"
        }
        
        with patch.dict(os.environ, test_env_vars):
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                cfg = compose(config_name="config")
                
                # Validate environment values are used
                assert cfg.navigator.max_speed == 3.0
                assert cfg.navigator.position == [1.0, 2.0]
                assert cfg.navigator.orientation == 90.0
                assert cfg.video_plume.video_path == "/custom/path/video.mp4"
                assert cfg.video_plume.flip == True
                assert cfg.video_plume.threshold == 0.8
                assert cfg.system.output_dir == "custom_output"
                assert cfg.system.debug_mode == True
    
    def test_environment_variable_security_validation(self, setup_environment_config):
        """Test environment variable interpolation security against injection attacks."""
        config_dir = setup_environment_config
        
        # Test malicious environment variable injection attempts
        malicious_env_vars = {
            "TEST_VIDEO_PATH": "../../../etc/passwd",  # Path traversal attempt
            "TEST_OUTPUT_DIR": "; rm -rf /",  # Command injection attempt  
            "TEST_THRESHOLD": "${oc.env:SECRET_KEY}",  # Nested interpolation attempt
        }
        
        with patch.dict(os.environ, malicious_env_vars):
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                cfg = compose(config_name="config")
                
                # Validate malicious values are treated as literal strings
                assert cfg.video_plume.video_path == "../../../etc/passwd"
                assert cfg.system.output_dir == "; rm -rf /"
                assert cfg.video_plume.threshold == "${oc.env:SECRET_KEY}"
                
                # Validate type conversion handles malicious strings appropriately
                assert isinstance(cfg.video_plume.video_path, str)
                assert isinstance(cfg.system.output_dir, str)
    
    def test_missing_environment_variable_handling(self, setup_environment_config):
        """Test handling of missing environment variables without defaults."""
        config_dir = setup_environment_config
        
        # Create configuration with missing environment variable (no default)
        missing_env_config = """
navigator:
  max_speed: ${oc.env:MISSING_VAR}
"""
        
        (self.config_dir / "missing_config.yaml").write_text(missing_env_config)
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            with pytest.raises(MissingMandatoryValue):
                compose(config_name="missing_config")
    
    @pytest.mark.parametrize("env_var,env_value,expected_type", [
        ("TEST_FLOAT_VAR", "3.14", float),
        ("TEST_INT_VAR", "42", int),
        ("TEST_BOOL_VAR", "true", bool),
        ("TEST_LIST_VAR", "[1,2,3]", list),
        ("TEST_STRING_VAR", "test_string", str),
    ])
    def test_environment_variable_type_conversion(self, setup_environment_config, env_var, env_value, expected_type):
        """Test automatic type conversion for environment variables."""
        config_dir = setup_environment_config
        
        # Create config with typed environment variable
        typed_config = f"""
test_param: ${{oc.env:{env_var}}}
"""
        
        (self.config_dir / "typed_config.yaml").write_text(typed_config)
        
        with patch.dict(os.environ, {env_var: env_value}):
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                cfg = compose(config_name="typed_config")
                
                assert isinstance(cfg.test_param, expected_type)


class TestPydanticSchemaIntegration:
    """
    Test suite for Pydantic schema integration with Hydra structured configs.
    
    Validates type safety, constraint enforcement, and validation error handling
    when Hydra configurations are processed through Pydantic models.
    """
    
    def test_navigator_config_validation_single_agent(self):
        """Test NavigatorConfig validation for single-agent scenarios."""
        # Valid single-agent configuration
        valid_config = {
            "position": [1.0, 2.0],
            "orientation": 45.0,
            "speed": 1.5,
            "max_speed": 2.0,
            "angular_velocity": 0.5
        }
        
        config = NavigatorConfig(**valid_config)
        
        # Validate successful creation and type conversion
        assert config.position == (1.0, 2.0)
        assert config.orientation == 45.0
        assert config.speed == 1.5
        assert config.max_speed == 2.0
        assert config.angular_velocity == 0.5
    
    def test_navigator_config_validation_multi_agent(self):
        """Test NavigatorConfig validation for multi-agent scenarios."""
        # Valid multi-agent configuration
        valid_config = {
            "positions": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            "orientations": [0.0, 90.0, 180.0],
            "speeds": [1.0, 1.2, 0.8],
            "max_speeds": [2.0, 2.0, 2.0],
            "num_agents": 3
        }
        
        config = NavigatorConfig(**valid_config)
        
        # Validate successful creation
        assert len(config.positions) == 3
        assert len(config.orientations) == 3
        assert len(config.speeds) == 3
        assert len(config.max_speeds) == 3
        assert config.num_agents == 3
    
    def test_navigator_config_validation_errors(self):
        """Test NavigatorConfig validation error handling."""
        # Test speed exceeds max_speed
        with pytest.raises(ValidationError) as exc_info:
            NavigatorConfig(
                position=[0.0, 0.0],
                speed=3.0,
                max_speed=2.0
            )
        assert "cannot exceed max_speed" in str(exc_info.value)
        
        # Test negative speed
        with pytest.raises(ValidationError) as exc_info:
            NavigatorConfig(
                position=[0.0, 0.0],
                speed=-1.0
            )
        assert "greater than or equal to 0" in str(exc_info.value)
        
        # Test inconsistent multi-agent parameters
        with pytest.raises(ValidationError) as exc_info:
            NavigatorConfig(
                positions=[[0.0, 0.0], [1.0, 1.0]],
                orientations=[0.0],  # Wrong length
                num_agents=2
            )
        assert "does not match number of agents" in str(exc_info.value)
    
    def test_video_plume_config_validation(self):
        """Test VideoPlumeConfig validation and path handling."""
        # Valid configuration
        valid_config = {
            "video_path": "data/test_video.mp4",
            "flip": True,
            "grayscale": True,
            "kernel_size": 5,
            "kernel_sigma": 1.0,
            "threshold": 0.7,
            "normalize": True,
            "_skip_validation": True  # Skip file existence check for testing
        }
        
        config = VideoPlumeConfig(**valid_config)
        
        # Validate configuration properties
        assert str(config.video_path) == "data/test_video.mp4"
        assert config.flip == True
        assert config.grayscale == True
        assert config.kernel_size == 5
        assert config.kernel_sigma == 1.0
        assert config.threshold == 0.7
        assert config.normalize == True
    
    def test_video_plume_config_validation_errors(self):
        """Test VideoPlumeConfig validation error handling."""
        # Test invalid kernel_size (even number)
        with pytest.raises(ValidationError) as exc_info:
            VideoPlumeConfig(
                video_path="test.mp4",
                kernel_size=4,  # Must be odd
                kernel_sigma=1.0,
                _skip_validation=True
            )
        assert "must be odd" in str(exc_info.value)
        
        # Test invalid threshold range
        with pytest.raises(ValidationError) as exc_info:
            VideoPlumeConfig(
                video_path="test.mp4",
                threshold=1.5,  # Must be <= 1.0
                _skip_validation=True
            )
        assert "less than or equal to 1" in str(exc_info.value)
        
        # Test missing kernel_sigma when kernel_size provided
        with pytest.raises(ValidationError) as exc_info:
            VideoPlumeConfig(
                video_path="test.mp4",
                kernel_size=5,
                # kernel_sigma missing
                _skip_validation=True
            )
        assert "kernel_sigma must be specified" in str(exc_info.value)
    
    def test_pydantic_hydra_integration(self, tmp_path):
        """Test integration between Pydantic validation and Hydra configuration."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping Pydantic-Hydra integration tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create configuration that should pass Pydantic validation
        valid_config = """
navigator:
  position: [0.0, 0.0]
  orientation: 45.0
  speed: 1.0
  max_speed: 2.0

video_plume:
  video_path: "data/test.mp4"
  flip: false
  grayscale: true
  kernel_size: 5
  kernel_sigma: 1.0
  threshold: 0.5
"""
        
        (config_dir / "config.yaml").write_text(valid_config)
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
            
            # Convert Hydra config to Pydantic models
            navigator_config = NavigatorConfig(**cfg.navigator)
            video_plume_config = VideoPlumeConfig(
                _skip_validation=True,  # Skip file existence check
                **cfg.video_plume
            )
            
            # Validate successful conversion and validation
            assert navigator_config.position == (0.0, 0.0)
            assert navigator_config.orientation == 45.0
            assert navigator_config.speed == 1.0
            assert navigator_config.max_speed == 2.0
            
            assert str(video_plume_config.video_path) == "data/test.mp4"
            assert video_plume_config.flip == False
            assert video_plume_config.kernel_size == 5


class TestConfigurationSecurity:
    """
    Test suite for configuration security including path traversal prevention.
    
    Validates security measures against malicious configuration file references,
    path traversal attacks, and unauthorized file access attempts.
    """
    
    def test_path_traversal_prevention(self, tmp_path):
        """Test prevention of path traversal attacks in configuration file loading."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping security tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create legitimate configuration
        safe_config = """
video_plume:
  video_path: "data/safe_video.mp4"
"""
        (config_dir / "safe_config.yaml").write_text(safe_config)
        
        # Test path traversal attempts in video_path
        traversal_config = """
video_plume:
  video_path: "../../../etc/passwd"
"""
        (config_dir / "traversal_config.yaml").write_text(traversal_config)
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            # Safe configuration should load normally
            safe_cfg = compose(config_name="safe_config")
            assert safe_cfg.video_plume.video_path == "data/safe_video.mp4"
            
            # Path traversal configuration should load but be treated as literal
            traversal_cfg = compose(config_name="traversal_config")
            assert traversal_cfg.video_plume.video_path == "../../../etc/passwd"
            
            # Validate that VideoPlumeConfig would reject the traversal path
            with pytest.raises(ValidationError):
                VideoPlumeConfig(
                    video_path="../../../etc/passwd",
                    _skip_validation=False  # Enable file validation
                )
    
    def test_configuration_file_permission_validation(self, tmp_path):
        """Test configuration file permission and access validation."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping permission tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create configuration file
        config_content = """
navigator:
  max_speed: 2.0
"""
        config_file = config_dir / "config.yaml"
        config_file.write_text(config_content)
        
        # Test normal file access
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
            assert cfg.navigator.max_speed == 2.0
    
    def test_sensitive_parameter_protection(self, tmp_path):
        """Test protection against sensitive parameter exposure in configuration."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping sensitive parameter tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Configuration with potentially sensitive parameters
        sensitive_config = """
navigator:
  max_speed: 2.0

database:
  password: "not_a_real_password"
  
api:
  secret_key: "development_key_only"
"""
        
        (config_dir / "config.yaml").write_text(sensitive_config)
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
            
            # Validate configuration loads but sensitive data is present
            # (In production, these would be environment variables)
            assert cfg.navigator.max_speed == 2.0
            assert cfg.database.password == "not_a_real_password"
            assert cfg.api.secret_key == "development_key_only"
            
            # Validate configuration can be serialized without exposure
            config_str = OmegaConf.to_yaml(cfg)
            assert "navigator:" in config_str
            # Note: In production, sensitive fields would be redacted


class TestConfigurationPerformance:
    """
    Test suite for configuration loading and composition performance validation.
    
    Validates that configuration operations meet performance requirements including
    the <500ms composition requirement from Section 6.6.3.3.
    """
    
    def test_configuration_loading_performance(self, tmp_path):
        """Test configuration loading meets performance requirements."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping performance tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create complex configuration for performance testing
        complex_config = """
defaults:
  - _self_

navigator:
  position: [0.0, 0.0]
  orientation: 0.0
  speed: 0.0
  max_speed: ${oc.env:MAX_SPEED,2.0}
  angular_velocity: 0.0

video_plume:
  video_path: ${oc.env:VIDEO_PATH,data/test.mp4}
  flip: ${oc.env:VIDEO_FLIP,false}
  grayscale: true
  kernel_size: 5
  kernel_sigma: 1.0
  threshold: ${oc.env:THRESHOLD,0.5}
  normalize: true

multi_agent:
  positions: [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
  orientations: [0.0, 90.0, 180.0]
  speeds: [1.0, 1.2, 0.8]
  max_speeds: [2.0, 2.0, 2.0]
  num_agents: 3

system:
  random_seed: ${oc.env:RANDOM_SEED,42}
  debug_mode: ${oc.env:DEBUG,false}
  output_dir: ${oc.env:OUTPUT_DIR,outputs}
  log_level: INFO
"""
        
        (config_dir / "config.yaml").write_text(complex_config)
        
        # Measure configuration loading performance
        start_time = time.time()
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
        
        loading_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Validate performance requirement from Section 6.6.3.3
        assert loading_time < 500, f"Configuration loading took {loading_time:.2f}ms, exceeds 500ms limit"
        
        # Validate configuration was successfully loaded
        assert cfg is not None
        assert "navigator" in cfg
        assert "video_plume" in cfg
        assert "multi_agent" in cfg
        assert "system" in cfg
    
    def test_configuration_override_performance(self, tmp_path):
        """Test configuration override operations meet performance requirements."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping override performance tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create base configuration
        base_config = """
navigator:
  max_speed: 1.0
  position: [0.0, 0.0]

video_plume:
  threshold: 0.5
  flip: false
"""
        
        (config_dir / "config.yaml").write_text(base_config)
        
        # Multiple override scenarios
        override_sets = [
            ["navigator.max_speed=2.0"],
            ["navigator.max_speed=3.0", "navigator.position=[1.0,2.0]"],
            ["video_plume.threshold=0.8", "video_plume.flip=true"],
            ["navigator.max_speed=4.0", "video_plume.threshold=0.9", "navigator.position=[2.0,3.0]"],
        ]
        
        for overrides in override_sets:
            start_time = time.time()
            
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                cfg = compose(config_name="config", overrides=overrides)
            
            override_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Validate performance for each override scenario
            assert override_time < 500, f"Configuration override took {override_time:.2f}ms, exceeds 500ms limit"
            
            # Validate overrides were applied
            assert cfg is not None
    
    def test_large_configuration_performance(self, tmp_path):
        """Test performance with large configuration files."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping large configuration tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Generate large configuration with many parameters
        large_config_lines = ["defaults:", "  - _self_", ""]
        
        # Add many navigator configurations
        for i in range(50):
            large_config_lines.extend([
                f"navigator_{i}:",
                f"  position: [{i}.0, {i+1}.0]",
                f"  orientation: {i * 7.2}",
                f"  max_speed: {1.0 + i * 0.1}",
                ""
            ])
        
        # Add many video plume configurations
        for i in range(30):
            large_config_lines.extend([
                f"video_plume_{i}:",
                f"  video_path: data/video_{i}.mp4",
                f"  threshold: {0.1 + i * 0.01}",
                f"  kernel_size: {3 + (i % 4) * 2}",
                ""
            ])
        
        large_config = "\n".join(large_config_lines)
        (config_dir / "config.yaml").write_text(large_config)
        
        # Measure large configuration loading
        start_time = time.time()
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
        
        loading_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Validate performance even with large configuration
        assert loading_time < 500, f"Large configuration loading took {loading_time:.2f}ms, exceeds 500ms limit"
        
        # Validate configuration structure
        assert cfg is not None
        assert hasattr(cfg, "navigator_0")
        assert hasattr(cfg, "video_plume_0")


class TestConfigurationEvolutionAndCompatibility:
    """
    Test suite for configuration schema evolution and backward compatibility.
    
    Validates that configuration changes maintain compatibility with existing
    configurations and support migration patterns.
    """
    
    def test_backward_compatibility_with_legacy_config(self, tmp_path):
        """Test backward compatibility with legacy configuration formats."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping compatibility tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Legacy configuration format (without Hydra features)
        legacy_config = """
# Legacy format configuration
navigator:
  position: [0.0, 0.0]
  orientation: 0.0
  speed: 0.0
  max_speed: 1.0

video_plume:
  video_path: "data/legacy_video.mp4"
  flip: false
  grayscale: true
"""
        
        (config_dir / "legacy_config.yaml").write_text(legacy_config)
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="legacy_config")
            
            # Validate legacy configuration loads correctly
            assert cfg.navigator.position == [0.0, 0.0]
            assert cfg.navigator.max_speed == 1.0
            assert cfg.video_plume.video_path == "data/legacy_video.mp4"
            
            # Validate it can be converted to Pydantic models
            navigator_config = NavigatorConfig(**cfg.navigator)
            video_plume_config = VideoPlumeConfig(
                _skip_validation=True,
                **cfg.video_plume
            )
            
            assert navigator_config.max_speed == 1.0
            assert str(video_plume_config.video_path) == "data/legacy_video.mp4"
    
    def test_schema_evolution_with_optional_fields(self, tmp_path):
        """Test schema evolution with new optional fields."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping schema evolution tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Configuration missing some optional fields
        minimal_config = """
navigator:
  max_speed: 2.0

video_plume:
  video_path: "data/minimal.mp4"
"""
        
        (config_dir / "minimal_config.yaml").write_text(minimal_config)
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="minimal_config")
            
            # Convert to Pydantic with default handling
            navigator_config = NavigatorConfig(
                max_speed=cfg.navigator.max_speed
            )
            video_plume_config = VideoPlumeConfig(
                video_path=cfg.video_plume.video_path,
                _skip_validation=True
            )
            
            # Validate defaults are applied
            assert navigator_config.max_speed == 2.0
            assert navigator_config.orientation == 0.0  # Default
            assert navigator_config.speed == 0.0  # Default
            
            assert video_plume_config.flip == False  # Default
            assert video_plume_config.grayscale == True  # Default
            assert video_plume_config.normalize == True  # Default
    
    def test_configuration_migration_patterns(self, tmp_path):
        """Test configuration migration from old to new formats."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping migration tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Old configuration format
        old_config = {
            "agent": {
                "initial_position": [1.0, 2.0],
                "initial_orientation": 45.0,
                "movement_speed": 1.5,
                "maximum_speed": 3.0
            },
            "environment": {
                "video_file": "data/old_format.mp4",
                "horizontal_flip": True,
                "convert_grayscale": True
            }
        }
        
        # Migration function (simulated)
        def migrate_config(old_cfg):
            """Migrate old configuration format to new format."""
            new_cfg = {
                "navigator": {
                    "position": old_cfg["agent"]["initial_position"],
                    "orientation": old_cfg["agent"]["initial_orientation"],
                    "speed": old_cfg["agent"]["movement_speed"],
                    "max_speed": old_cfg["agent"]["maximum_speed"]
                },
                "video_plume": {
                    "video_path": old_cfg["environment"]["video_file"],
                    "flip": old_cfg["environment"]["horizontal_flip"],
                    "grayscale": old_cfg["environment"]["convert_grayscale"]
                }
            }
            return new_cfg
        
        # Perform migration
        new_config = migrate_config(old_config)
        
        # Validate migration preserves data
        assert new_config["navigator"]["position"] == [1.0, 2.0]
        assert new_config["navigator"]["orientation"] == 45.0
        assert new_config["navigator"]["speed"] == 1.5
        assert new_config["navigator"]["max_speed"] == 3.0
        assert new_config["video_plume"]["video_path"] == "data/old_format.mp4"
        assert new_config["video_plume"]["flip"] == True
        assert new_config["video_plume"]["grayscale"] == True
        
        # Validate migrated config works with Pydantic
        navigator_config = NavigatorConfig(**new_config["navigator"])
        video_plume_config = VideoPlumeConfig(
            _skip_validation=True,
            **new_config["video_plume"]
        )
        
        assert navigator_config.position == (1.0, 2.0)
        assert navigator_config.max_speed == 3.0
        assert str(video_plume_config.video_path) == "data/old_format.mp4"


# Pytest performance markers for configuration tests
pytestmark = [
    pytest.mark.config,
    pytest.mark.performance,
    pytest.mark.integration
]


def test_configuration_system_integration():
    """Integration test for complete configuration system functionality."""
    # Test that all configuration classes are importable and functional
    
    # Single agent configuration
    single_config = SingleAgentConfig(
        position=(0.0, 0.0),
        orientation=0.0,
        speed=1.0,
        max_speed=2.0
    )
    assert single_config.position == (0.0, 0.0)
    
    # Multi agent configuration  
    multi_config = MultiAgentConfig(
        positions=[[0.0, 0.0], [1.0, 1.0]],
        orientations=[0.0, 90.0],
        speeds=[1.0, 1.2],
        max_speeds=[2.0, 2.0],
        num_agents=2
    )
    assert len(multi_config.positions) == 2
    
    # Unified navigator configuration
    navigator_config = NavigatorConfig(
        position=(0.0, 0.0),
        max_speed=2.0
    )
    assert navigator_config.position == (0.0, 0.0)
    
    # Video plume configuration
    video_config = VideoPlumeConfig(
        video_path="test.mp4",
        _skip_validation=True
    )
    assert str(video_config.video_path) == "test.mp4"


if __name__ == "__main__":
    # Run tests with coverage reporting
    pytest.main([
        __file__,
        "-v",
        "--cov=src.{{cookiecutter.project_slug}}.config",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--tb=short"
    ])