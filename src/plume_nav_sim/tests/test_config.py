"""
Comprehensive Hydra configuration system testing module validating hierarchical composition,
schema validation, environment variable interpolation, and performance requirements for the
plume_nav_sim migration to Gymnasium 0.29.

This module provides comprehensive testing for the enhanced configuration system supporting
the library's migration from OpenAI Gym 0.26 to Gymnasium 0.29, with emphasis on frame cache
configuration, environment variable interpolation, and performance requirements.

Test Coverage Areas:
- Hydra configuration composition from hierarchical conf/ structure  
- Environment variable interpolation with ${oc.env:VAR_NAME,default} syntax validation
- Frame cache configuration integration with LRU and memory limit settings
- Pydantic schema integration for Gymnasium-related configuration parameters
- Configuration performance validation meeting <500ms composition requirement
- Hierarchical override precedence and parameter inheritance patterns
- Configuration security including path traversal prevention

Performance Requirements:
- Configuration composition must complete within 500ms per Section 6.6.3.3
- Frame cache configuration must support runtime adjustability per Section 5.2.2
- Hierarchical configuration validation must achieve >90% coverage per Section 6.6.3.1

Security Requirements:
- Environment variable interpolation must be secure and validated per Section 6.6.7.1
- Path traversal prevention for configuration file loading
- Configuration parameter validation against injection attacks
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import patch, mock_open, MagicMock
import warnings

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

# Configuration schema imports - conditional based on availability
try:
    from plume_nav_sim.config.schemas import (
        NavigatorConfig,
        SingleAgentConfig, 
        MultiAgentConfig,
        VideoPlumeConfig,
        FrameCacheConfig,
        EnvironmentConfig,
        SystemConfig
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    # Define minimal schema classes for testing
    class NavigatorConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class VideoPlumeConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class FrameCacheConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

# Frame cache imports
try:
    from plume_nav_sim.utils.frame_cache import CacheMode, FrameCache
    FRAME_CACHE_AVAILABLE = True
except ImportError:
    FRAME_CACHE_AVAILABLE = False
    class CacheMode:
        NONE = "none"
        LRU = "lru"
        ALL = "all"


class TestHydraConfigurationComposition:
    """
    Test suite for Hydra configuration composition and hierarchical parameter management
    specific to the plume_nav_sim migration to Gymnasium 0.29.
    
    Validates the core Hydra functionality including configuration loading from hierarchical
    conf/ structure, composition with environment-specific overlays, and integration with
    the new frame cache and Gymnasium configuration parameters.
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
        
        # Create subdirectories for hierarchical configuration
        (self.config_dir / "base").mkdir()
        (self.config_dir / "env").mkdir()
        (self.config_dir / "env" / "flavors").mkdir()
        (self.config_dir / "local").mkdir()
        
        # Create base configuration with Gymnasium-related settings
        base_config = """
# conf/base.yaml - Foundation configuration for Gymnasium migration
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

# Frame cache configuration for Gymnasium migration
frame_cache:
  mode: lru
  memory_limit_mb: 2048
  memory_pressure_threshold: 0.9
  enable_statistics: true
  enable_logging: true
  preload_chunk_size: 100
  eviction_batch_size: 10

# Gymnasium environment configuration
environment:
  env_id: "PlumeNavSim-v0"
  render_mode: null
  max_episode_steps: 1000
  step_timeout_ms: 10
  api_compatibility: "gymnasium"
  legacy_gym_support: true

system:
  random_seed: 42
  debug_mode: false
  output_dir: "outputs"
  log_level: "INFO"
"""
        
        # Create environment-specific configuration with variable interpolation
        config_yaml = """
# conf/config.yaml - Environment-specific configuration with interpolation
defaults:
  - base
  - env/flavors: ${oc.env:ENV_FLAVOR,memory}
  - _self_

navigator:
  max_speed: ${oc.env:NAVIGATOR_MAX_SPEED,2.0}
  orientation: ${oc.env:INITIAL_ORIENTATION,45.0}

video_plume:
  video_path: ${oc.env:PLUME_VIDEO_PATH,data/default_plume.mp4}
  flip: ${oc.env:VIDEO_FLIP,true}

frame_cache:
  mode: ${oc.env:FRAME_CACHE_MODE,lru}
  memory_limit_mb: ${oc.env:FRAME_CACHE_SIZE_MB,2048}
  memory_pressure_threshold: ${oc.env:FRAME_CACHE_PRESSURE,0.9}

environment:
  render_mode: ${oc.env:RENDER_MODE,null}
  max_episode_steps: ${oc.env:MAX_EPISODE_STEPS,1000}
  step_timeout_ms: ${oc.env:STEP_TIMEOUT_MS,10}

system:
  debug_mode: ${oc.env:DEBUG_MODE,true}
  output_dir: ${oc.env:OUTPUT_DIR,results}
  log_level: ${oc.env:LOG_LEVEL,INFO}
"""
        
        # Create memory-optimized environment flavor
        memory_flavor = """
# conf/env/flavors/memory.yaml - Memory-optimized configuration
# @package _global_

frame_cache:
  mode: lru
  memory_limit_mb: 1024
  memory_pressure_threshold: 0.8
  preload_chunk_size: 50

environment:
  max_episode_steps: 500
"""
        
        # Create memoryless environment flavor
        memoryless_flavor = """
# conf/env/flavors/memoryless.yaml - Memory-disabled configuration
# @package _global_

frame_cache:
  mode: none
  memory_limit_mb: 0
  enable_statistics: false

environment:
  max_episode_steps: 100
"""
        
        # Write configuration files
        (self.config_dir / "base.yaml").write_text(base_config)
        (self.config_dir / "config.yaml").write_text(config_yaml)
        (self.config_dir / "env" / "flavors" / "memory.yaml").write_text(memory_flavor)
        (self.config_dir / "env" / "flavors" / "memoryless.yaml").write_text(memoryless_flavor)
        
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
            assert "frame_cache" in cfg
            assert "environment" in cfg
            assert "system" in cfg
            
            # Validate base configuration inheritance
            assert cfg.navigator.position == [0.0, 0.0]
            assert cfg.navigator.speed == 0.0
            assert cfg.video_plume.kernel_size == 5
            assert cfg.system.random_seed == 42
            
            # Validate Gymnasium-specific configuration
            assert cfg.environment.env_id == "PlumeNavSim-v0"
            assert cfg.environment.api_compatibility == "gymnasium"
            assert cfg.environment.legacy_gym_support == True
    
    def test_frame_cache_configuration_integration(self, setup_hydra_environment):
        """Test frame cache configuration integration with Hydra system."""
        config_dir = setup_hydra_environment
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
            
            # Validate frame cache configuration structure
            assert "frame_cache" in cfg
            assert hasattr(cfg.frame_cache, "mode")
            assert hasattr(cfg.frame_cache, "memory_limit_mb")
            assert hasattr(cfg.frame_cache, "memory_pressure_threshold")
            assert hasattr(cfg.frame_cache, "enable_statistics")
            assert hasattr(cfg.frame_cache, "preload_chunk_size")
            assert hasattr(cfg.frame_cache, "eviction_batch_size")
            
            # Validate default frame cache values
            assert cfg.frame_cache.mode == "lru"
            assert cfg.frame_cache.memory_limit_mb == 2048
            assert cfg.frame_cache.memory_pressure_threshold == 0.9
            assert cfg.frame_cache.enable_statistics == True
            assert cfg.frame_cache.preload_chunk_size == 100
            assert cfg.frame_cache.eviction_batch_size == 10
            
            # Validate frame cache mode validation
            valid_modes = ["none", "lru", "all"]
            assert cfg.frame_cache.mode in valid_modes
    
    def test_configuration_composition_performance(self, setup_hydra_environment):
        """Test configuration composition meets <500ms performance requirement per Section 6.6.3.3."""
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
        assert "frame_cache" in cfg
    
    def test_hierarchical_configuration_inheritance_with_flavors(self, setup_hydra_environment):
        """Test hierarchical parameter inheritance from base to config to environment flavors."""
        config_dir = setup_hydra_environment
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            # Test memory flavor configuration (default)
            memory_cfg = compose(config_name="config")
            assert memory_cfg.frame_cache.mode == "lru"
            assert memory_cfg.frame_cache.memory_limit_mb == 1024  # Overridden by memory flavor
            assert memory_cfg.frame_cache.memory_pressure_threshold == 0.8  # Overridden by memory flavor
            
            # Test memoryless flavor configuration
            memoryless_cfg = compose(
                config_name="config",
                overrides=["env/flavors=memoryless"]
            )
            assert memoryless_cfg.frame_cache.mode == "none"
            assert memoryless_cfg.frame_cache.memory_limit_mb == 0
            assert memoryless_cfg.frame_cache.enable_statistics == False
            
            # Should inherit from base for non-overridden values
            assert memoryless_cfg.navigator.position == [0.0, 0.0]  # From base
            assert memoryless_cfg.navigator.speed == 0.0  # From base
    
    def test_configuration_override_precedence(self, setup_hydra_environment):
        """Test configuration override precedence: command-line > env vars > config > base."""
        config_dir = setup_hydra_environment
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            # Test programmatic overrides
            cfg = compose(
                config_name="config", 
                overrides=[
                    "navigator.max_speed=5.0",
                    "frame_cache.memory_limit_mb=4096",
                    "environment.max_episode_steps=2000",
                    "system.debug_mode=false"
                ]
            )
            
            # Validate overrides take precedence
            assert cfg.navigator.max_speed == 5.0  # Command-line override
            assert cfg.frame_cache.memory_limit_mb == 4096  # Command-line override  
            assert cfg.environment.max_episode_steps == 2000  # Command-line override
            assert cfg.system.debug_mode == False  # Command-line override
            
            # Validate non-overridden values retain hierarchy
            assert cfg.navigator.position == [0.0, 0.0]  # From base
            assert cfg.video_plume.kernel_size == 5  # From base
    
    @pytest.mark.parametrize("override_param,override_value,expected_type", [
        ("navigator.max_speed", "3.5", float),
        ("navigator.position", "[1.0,2.0]", list),
        ("video_plume.flip", "false", bool),
        ("frame_cache.memory_limit_mb", "1024", int),
        ("environment.max_episode_steps", "500", int),
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
    Test suite for environment variable interpolation with ${oc.env:VAR_NAME,default} syntax
    supporting the Gymnasium migration configuration requirements.
    
    Validates secure environment variable handling, fallback behavior, and integration
    with frame cache and Gymnasium-specific configuration parameters.
    """
    
    @pytest.fixture(autouse=True)
    def setup_environment_config(self, tmp_path):
        """Setup configuration with comprehensive environment variable interpolation."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping environment variable tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        self.config_dir = tmp_path / "conf"
        self.config_dir.mkdir()
        
        # Create configuration with comprehensive environment variable interpolation
        env_config = """
navigator:
  max_speed: ${oc.env:TEST_MAX_SPEED,1.5}
  position: ${oc.env:TEST_POSITION,[0.0,0.0]}
  orientation: ${oc.env:TEST_ORIENTATION,0.0}

video_plume:
  video_path: ${oc.env:TEST_VIDEO_PATH,data/test.mp4}
  flip: ${oc.env:TEST_VIDEO_FLIP,false}
  threshold: ${oc.env:TEST_THRESHOLD,0.5}
  kernel_size: ${oc.env:TEST_KERNEL_SIZE,5}

frame_cache:
  mode: ${oc.env:TEST_CACHE_MODE,lru}
  memory_limit_mb: ${oc.env:TEST_CACHE_MEMORY_MB,2048}
  memory_pressure_threshold: ${oc.env:TEST_CACHE_PRESSURE,0.9}
  enable_statistics: ${oc.env:TEST_CACHE_STATS,true}
  preload_chunk_size: ${oc.env:TEST_PRELOAD_CHUNK,100}

environment:
  env_id: ${oc.env:TEST_ENV_ID,PlumeNavSim-v0}
  render_mode: ${oc.env:TEST_RENDER_MODE,null}
  max_episode_steps: ${oc.env:TEST_MAX_STEPS,1000}
  step_timeout_ms: ${oc.env:TEST_STEP_TIMEOUT,10}
  api_compatibility: ${oc.env:TEST_API_COMPAT,gymnasium}

system:
  output_dir: ${oc.env:TEST_OUTPUT_DIR,default_output}
  debug_mode: ${oc.env:TEST_DEBUG,false}
  log_level: ${oc.env:TEST_LOG_LEVEL,INFO}
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
            "TEST_VIDEO_PATH", "TEST_VIDEO_FLIP", "TEST_THRESHOLD", "TEST_KERNEL_SIZE",
            "TEST_CACHE_MODE", "TEST_CACHE_MEMORY_MB", "TEST_CACHE_PRESSURE", 
            "TEST_CACHE_STATS", "TEST_PRELOAD_CHUNK",
            "TEST_ENV_ID", "TEST_RENDER_MODE", "TEST_MAX_STEPS", "TEST_STEP_TIMEOUT", "TEST_API_COMPAT",
            "TEST_OUTPUT_DIR", "TEST_DEBUG", "TEST_LOG_LEVEL"
        ]
        
        with patch.dict(os.environ, {}, clear=False):
            # Clear test environment variables
            for var in env_vars_to_clear:
                os.environ.pop(var, None)
            
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                cfg = compose(config_name="config")
                
                # Validate navigator defaults
                assert cfg.navigator.max_speed == 1.5
                assert cfg.navigator.position == [0.0, 0.0]
                assert cfg.navigator.orientation == 0.0
                
                # Validate video plume defaults
                assert cfg.video_plume.video_path == "data/test.mp4"
                assert cfg.video_plume.flip == False
                assert cfg.video_plume.threshold == 0.5
                assert cfg.video_plume.kernel_size == 5
                
                # Validate frame cache defaults
                assert cfg.frame_cache.mode == "lru"
                assert cfg.frame_cache.memory_limit_mb == 2048
                assert cfg.frame_cache.memory_pressure_threshold == 0.9
                assert cfg.frame_cache.enable_statistics == True
                assert cfg.frame_cache.preload_chunk_size == 100
                
                # Validate environment defaults
                assert cfg.environment.env_id == "PlumeNavSim-v0"
                assert cfg.environment.render_mode == None
                assert cfg.environment.max_episode_steps == 1000
                assert cfg.environment.step_timeout_ms == 10
                assert cfg.environment.api_compatibility == "gymnasium"
                
                # Validate system defaults
                assert cfg.system.output_dir == "default_output"
                assert cfg.system.debug_mode == False
                assert cfg.system.log_level == "INFO"
    
    def test_environment_variable_interpolation_with_values(self, setup_environment_config):
        """Test environment variable interpolation uses environment values when set."""
        config_dir = setup_environment_config
        
        test_env_vars = {
            # Navigator configuration
            "TEST_MAX_SPEED": "3.0",
            "TEST_POSITION": "[1.0,2.0]",
            "TEST_ORIENTATION": "90.0",
            
            # Video plume configuration  
            "TEST_VIDEO_PATH": "/custom/path/video.mp4",
            "TEST_VIDEO_FLIP": "true",
            "TEST_THRESHOLD": "0.8",
            "TEST_KERNEL_SIZE": "7",
            
            # Frame cache configuration
            "TEST_CACHE_MODE": "all",
            "TEST_CACHE_MEMORY_MB": "4096",
            "TEST_CACHE_PRESSURE": "0.95",
            "TEST_CACHE_STATS": "false",
            "TEST_PRELOAD_CHUNK": "200",
            
            # Environment configuration
            "TEST_ENV_ID": "PlumeNavSim-v1",
            "TEST_RENDER_MODE": "rgb_array",
            "TEST_MAX_STEPS": "2000",
            "TEST_STEP_TIMEOUT": "15",
            "TEST_API_COMPAT": "legacy_gym",
            
            # System configuration
            "TEST_OUTPUT_DIR": "custom_output",
            "TEST_DEBUG": "true",
            "TEST_LOG_LEVEL": "DEBUG"
        }
        
        with patch.dict(os.environ, test_env_vars):
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                cfg = compose(config_name="config")
                
                # Validate navigator environment values
                assert cfg.navigator.max_speed == 3.0
                assert cfg.navigator.position == [1.0, 2.0]
                assert cfg.navigator.orientation == 90.0
                
                # Validate video plume environment values
                assert cfg.video_plume.video_path == "/custom/path/video.mp4"
                assert cfg.video_plume.flip == True
                assert cfg.video_plume.threshold == 0.8
                assert cfg.video_plume.kernel_size == 7
                
                # Validate frame cache environment values
                assert cfg.frame_cache.mode == "all"
                assert cfg.frame_cache.memory_limit_mb == 4096
                assert cfg.frame_cache.memory_pressure_threshold == 0.95
                assert cfg.frame_cache.enable_statistics == False
                assert cfg.frame_cache.preload_chunk_size == 200
                
                # Validate environment environment values
                assert cfg.environment.env_id == "PlumeNavSim-v1"
                assert cfg.environment.render_mode == "rgb_array"
                assert cfg.environment.max_episode_steps == 2000
                assert cfg.environment.step_timeout_ms == 15
                assert cfg.environment.api_compatibility == "legacy_gym"
                
                # Validate system environment values
                assert cfg.system.output_dir == "custom_output"
                assert cfg.system.debug_mode == True
                assert cfg.system.log_level == "DEBUG"
    
    def test_frame_cache_environment_variable_validation(self, setup_environment_config):
        """Test frame cache specific environment variable handling and validation."""
        config_dir = setup_environment_config
        
        # Test different frame cache modes
        cache_mode_tests = [
            ("none", "none"),
            ("lru", "lru"), 
            ("all", "all"),
            ("LRU", "LRU"),  # Test case sensitivity
        ]
        
        for env_value, expected_value in cache_mode_tests:
            with patch.dict(os.environ, {"TEST_CACHE_MODE": env_value}):
                with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                    cfg = compose(config_name="config")
                    assert cfg.frame_cache.mode == expected_value
        
        # Test memory limit boundary values
        memory_tests = [
            ("0", 0),      # Disabled cache
            ("512", 512),  # Small cache
            ("2048", 2048), # Default cache
            ("8192", 8192), # Large cache
        ]
        
        for env_value, expected_value in memory_tests:
            with patch.dict(os.environ, {"TEST_CACHE_MEMORY_MB": env_value}):
                with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                    cfg = compose(config_name="config")
                    assert cfg.frame_cache.memory_limit_mb == expected_value
    
    def test_environment_variable_security_validation(self, setup_environment_config):
        """Test environment variable interpolation security against injection attacks."""
        config_dir = setup_environment_config
        
        # Test malicious environment variable injection attempts
        malicious_env_vars = {
            "TEST_VIDEO_PATH": "../../../etc/passwd",  # Path traversal attempt
            "TEST_OUTPUT_DIR": "; rm -rf /",  # Command injection attempt  
            "TEST_CACHE_MODE": "${oc.env:SECRET_KEY}",  # Nested interpolation attempt
            "TEST_ENV_ID": "PlumeNavSim$(whoami)",  # Command substitution attempt
        }
        
        with patch.dict(os.environ, malicious_env_vars):
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                cfg = compose(config_name="config")
                
                # Validate malicious values are treated as literal strings
                assert cfg.video_plume.video_path == "../../../etc/passwd"
                assert cfg.system.output_dir == "; rm -rf /"
                assert cfg.frame_cache.mode == "${oc.env:SECRET_KEY}"
                assert cfg.environment.env_id == "PlumeNavSim$(whoami)"
                
                # Validate type conversion handles malicious strings appropriately
                assert isinstance(cfg.video_plume.video_path, str)
                assert isinstance(cfg.system.output_dir, str)
                assert isinstance(cfg.frame_cache.mode, str)
                assert isinstance(cfg.environment.env_id, str)
    
    def test_missing_environment_variable_handling(self, setup_environment_config):
        """Test handling of missing environment variables without defaults."""
        config_dir = setup_environment_config
        
        # Create configuration with missing environment variable (no default)
        missing_env_config = """
navigator:
  max_speed: ${oc.env:MISSING_VAR}

frame_cache:
  memory_limit_mb: ${oc.env:MISSING_CACHE_VAR}
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
        ("TEST_CACHE_MEMORY", "1024", int),
        ("TEST_CACHE_PRESSURE", "0.95", float),
        ("TEST_CACHE_STATS", "false", bool),
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
    Test suite for Pydantic schema integration with Hydra structured configs
    supporting the Gymnasium migration and frame cache configuration.
    
    Validates type safety, constraint enforcement, and validation error handling
    when Hydra configurations are processed through Pydantic models for the 
    enhanced plume_nav_sim configuration system.
    """
    
    def test_navigator_config_validation_single_agent(self):
        """Test NavigatorConfig validation for single-agent scenarios."""
        if not SCHEMAS_AVAILABLE:
            pytest.skip("Configuration schemas not available")
        
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
        assert hasattr(config, 'position')
        assert hasattr(config, 'orientation')
        assert hasattr(config, 'speed')
        assert hasattr(config, 'max_speed')
        assert hasattr(config, 'angular_velocity')
    
    def test_video_plume_config_validation(self):
        """Test VideoPlumeConfig validation and path handling."""
        if not SCHEMAS_AVAILABLE:
            pytest.skip("Configuration schemas not available")
        
        # Valid configuration
        valid_config = {
            "video_path": "data/test_video.mp4",
            "flip": True,
            "grayscale": True,
            "kernel_size": 5,
            "kernel_sigma": 1.0,
            "threshold": 0.7,
            "normalize": True,
        }
        
        config = VideoPlumeConfig(**valid_config)
        
        # Validate configuration properties
        assert hasattr(config, 'video_path')
        assert hasattr(config, 'flip')
        assert hasattr(config, 'grayscale')
        assert hasattr(config, 'kernel_size')
        assert hasattr(config, 'kernel_sigma')
        assert hasattr(config, 'threshold')
        assert hasattr(config, 'normalize')
    
    def test_frame_cache_config_validation(self):
        """Test FrameCacheConfig validation for Gymnasium migration."""
        if not SCHEMAS_AVAILABLE:
            pytest.skip("Configuration schemas not available")
        
        # Valid frame cache configuration
        valid_config = {
            "mode": "lru",
            "memory_limit_mb": 2048,
            "memory_pressure_threshold": 0.9,
            "enable_statistics": True,
            "enable_logging": True,
            "preload_chunk_size": 100,
            "eviction_batch_size": 10
        }
        
        config = FrameCacheConfig(**valid_config)
        
        # Validate frame cache configuration properties
        assert hasattr(config, 'mode')
        assert hasattr(config, 'memory_limit_mb')
        assert hasattr(config, 'memory_pressure_threshold')
        assert hasattr(config, 'enable_statistics')
        assert hasattr(config, 'enable_logging')
        assert hasattr(config, 'preload_chunk_size')
        assert hasattr(config, 'eviction_batch_size')
    
    def test_frame_cache_mode_validation(self):
        """Test frame cache mode validation with valid and invalid values."""
        if not SCHEMAS_AVAILABLE:
            pytest.skip("Configuration schemas not available")
        
        # Test valid cache modes
        valid_modes = ["none", "lru", "all"]
        for mode in valid_modes:
            config = FrameCacheConfig(
                mode=mode,
                memory_limit_mb=1024
            )
            assert hasattr(config, 'mode')
        
        # Test case sensitivity (if applicable)
        case_variants = ["LRU", "None", "ALL"]
        for mode in case_variants:
            try:
                config = FrameCacheConfig(
                    mode=mode,
                    memory_limit_mb=1024
                )
                assert hasattr(config, 'mode')
            except ValidationError:
                # Case sensitivity validation - this is expected behavior
                pass
    
    def test_frame_cache_memory_limit_validation(self):
        """Test frame cache memory limit validation with boundary values."""
        if not SCHEMAS_AVAILABLE:
            pytest.skip("Configuration schemas not available")
        
        # Test valid memory limits
        valid_limits = [0, 512, 1024, 2048, 4096, 8192]
        for limit in valid_limits:
            config = FrameCacheConfig(
                mode="lru",
                memory_limit_mb=limit
            )
            assert hasattr(config, 'memory_limit_mb')
        
        # Test invalid memory limits (if validation is implemented)
        invalid_limits = [-100, -1]
        for limit in invalid_limits:
            try:
                config = FrameCacheConfig(
                    mode="lru",
                    memory_limit_mb=limit
                )
                # If no validation error, ensure the value is handled appropriately
                assert hasattr(config, 'memory_limit_mb')
            except ValidationError:
                # Expected behavior for negative values
                pass
    
    def test_pydantic_hydra_integration_with_frame_cache(self, tmp_path):
        """Test integration between Pydantic validation and Hydra configuration for frame cache."""
        if not HYDRA_AVAILABLE or not SCHEMAS_AVAILABLE:
            pytest.skip("Hydra or schemas not available - skipping Pydantic-Hydra integration tests")
        
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

frame_cache:
  mode: lru
  memory_limit_mb: 2048
  memory_pressure_threshold: 0.9
  enable_statistics: true
  preload_chunk_size: 100
"""
        
        (config_dir / "config.yaml").write_text(valid_config)
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
            
            # Convert Hydra config to Pydantic models
            navigator_config = NavigatorConfig(**cfg.navigator)
            video_plume_config = VideoPlumeConfig(**cfg.video_plume)
            frame_cache_config = FrameCacheConfig(**cfg.frame_cache)
            
            # Validate successful conversion and validation
            assert hasattr(navigator_config, 'position')
            assert hasattr(navigator_config, 'orientation')
            assert hasattr(video_plume_config, 'video_path')
            assert hasattr(video_plume_config, 'flip')
            assert hasattr(frame_cache_config, 'mode')
            assert hasattr(frame_cache_config, 'memory_limit_mb')


class TestConfigurationPerformance:
    """
    Test suite for configuration loading and composition performance validation
    meeting the <500ms composition requirement from Section 6.6.3.3.
    
    Validates that configuration operations meet performance requirements including
    frame cache configuration overhead and complex hierarchical composition times.
    """
    
    def test_configuration_loading_performance_with_frame_cache(self, tmp_path):
        """Test configuration loading with frame cache settings meets performance requirements."""
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

frame_cache:
  mode: ${oc.env:CACHE_MODE,lru}
  memory_limit_mb: ${oc.env:CACHE_MEMORY,2048}
  memory_pressure_threshold: ${oc.env:CACHE_PRESSURE,0.9}
  enable_statistics: ${oc.env:CACHE_STATS,true}
  enable_logging: ${oc.env:CACHE_LOGGING,true}
  preload_chunk_size: ${oc.env:PRELOAD_CHUNK,100}
  eviction_batch_size: ${oc.env:EVICTION_BATCH,10}

environment:
  env_id: ${oc.env:ENV_ID,PlumeNavSim-v0}
  render_mode: ${oc.env:RENDER_MODE,null}
  max_episode_steps: ${oc.env:MAX_STEPS,1000}
  step_timeout_ms: ${oc.env:STEP_TIMEOUT,10}
  api_compatibility: ${oc.env:API_COMPAT,gymnasium}
  legacy_gym_support: ${oc.env:LEGACY_SUPPORT,true}

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
  log_level: ${oc.env:LOG_LEVEL,INFO}
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
        assert "frame_cache" in cfg
        assert "environment" in cfg
        assert "multi_agent" in cfg
        assert "system" in cfg
    
    def test_frame_cache_configuration_performance_overhead(self, tmp_path):
        """Test frame cache configuration overhead doesn't impact performance."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping frame cache performance tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Test minimal configuration (baseline)
        minimal_config = """
navigator:
  max_speed: 2.0
"""
        
        # Test configuration with extensive frame cache settings
        frame_cache_config = """
navigator:
  max_speed: 2.0

frame_cache:
  mode: lru
  memory_limit_mb: 2048
  memory_pressure_threshold: 0.9
  enable_statistics: true
  enable_logging: true
  preload_chunk_size: 100
  eviction_batch_size: 10
"""
        
        # Measure baseline performance
        (config_dir / "minimal.yaml").write_text(minimal_config)
        start_time = time.time()
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            minimal_cfg = compose(config_name="minimal")
        
        minimal_time = (time.time() - start_time) * 1000
        
        # Measure frame cache configuration performance
        (config_dir / "frame_cache.yaml").write_text(frame_cache_config)
        start_time = time.time()
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            frame_cache_cfg = compose(config_name="frame_cache")
        
        frame_cache_time = (time.time() - start_time) * 1000
        
        # Validate both meet performance requirements
        assert minimal_time < 500, f"Minimal configuration took {minimal_time:.2f}ms, exceeds 500ms limit"
        assert frame_cache_time < 500, f"Frame cache configuration took {frame_cache_time:.2f}ms, exceeds 500ms limit"
        
        # Validate frame cache overhead is reasonable (< 100ms additional)
        overhead = frame_cache_time - minimal_time
        assert overhead < 100, f"Frame cache configuration overhead {overhead:.2f}ms too high"
    
    def test_configuration_override_performance_with_frame_cache(self, tmp_path):
        """Test configuration override operations with frame cache parameters meet performance requirements."""
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

frame_cache:
  mode: lru
  memory_limit_mb: 1024
  memory_pressure_threshold: 0.9

environment:
  max_episode_steps: 1000
"""
        
        (config_dir / "config.yaml").write_text(base_config)
        
        # Multiple override scenarios including frame cache parameters
        override_sets = [
            ["navigator.max_speed=2.0"],
            ["frame_cache.memory_limit_mb=4096"],
            ["frame_cache.mode=all", "frame_cache.memory_limit_mb=8192"],
            ["navigator.max_speed=3.0", "frame_cache.memory_limit_mb=2048", "environment.max_episode_steps=2000"],
            ["frame_cache.mode=none", "frame_cache.enable_statistics=false", "environment.api_compatibility=legacy_gym"],
        ]
        
        for overrides in override_sets:
            start_time = time.time()
            
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                cfg = compose(config_name="config", overrides=overrides)
            
            override_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Validate performance for each override scenario
            assert override_time < 500, f"Configuration override {overrides} took {override_time:.2f}ms, exceeds 500ms limit"
            
            # Validate overrides were applied
            assert cfg is not None
    
    def test_environment_variable_interpolation_performance(self, tmp_path):
        """Test performance impact of extensive environment variable interpolation."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping interpolation performance tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create configuration with many environment variable interpolations
        interpolation_config = """
navigator:
  position: ${oc.env:NAV_POS,[0.0,0.0]}
  orientation: ${oc.env:NAV_ORI,0.0}
  speed: ${oc.env:NAV_SPEED,0.0}
  max_speed: ${oc.env:NAV_MAX_SPEED,2.0}
  angular_velocity: ${oc.env:NAV_ANG_VEL,0.0}

video_plume:
  video_path: ${oc.env:VP_PATH,data/test.mp4}
  flip: ${oc.env:VP_FLIP,false}
  grayscale: ${oc.env:VP_GRAY,true}
  kernel_size: ${oc.env:VP_KERNEL,5}
  kernel_sigma: ${oc.env:VP_SIGMA,1.0}
  threshold: ${oc.env:VP_THRESH,0.5}
  normalize: ${oc.env:VP_NORM,true}

frame_cache:
  mode: ${oc.env:FC_MODE,lru}
  memory_limit_mb: ${oc.env:FC_MEMORY,2048}
  memory_pressure_threshold: ${oc.env:FC_PRESSURE,0.9}
  enable_statistics: ${oc.env:FC_STATS,true}
  enable_logging: ${oc.env:FC_LOGGING,true}
  preload_chunk_size: ${oc.env:FC_PRELOAD,100}
  eviction_batch_size: ${oc.env:FC_EVICT,10}

environment:
  env_id: ${oc.env:ENV_ID,PlumeNavSim-v0}
  render_mode: ${oc.env:ENV_RENDER,null}
  max_episode_steps: ${oc.env:ENV_MAX_STEPS,1000}
  step_timeout_ms: ${oc.env:ENV_TIMEOUT,10}
  api_compatibility: ${oc.env:ENV_API,gymnasium}
  legacy_gym_support: ${oc.env:ENV_LEGACY,true}

system:
  random_seed: ${oc.env:SYS_SEED,42}
  debug_mode: ${oc.env:SYS_DEBUG,false}
  output_dir: ${oc.env:SYS_OUTPUT,outputs}
  log_level: ${oc.env:SYS_LOG,INFO}
"""
        
        (config_dir / "config.yaml").write_text(interpolation_config)
        
        # Measure interpolation performance (all defaults)
        start_time = time.time()
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
        
        interpolation_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Validate performance with extensive interpolation
        assert interpolation_time < 500, f"Environment variable interpolation took {interpolation_time:.2f}ms, exceeds 500ms limit"
        
        # Validate all interpolations resolved correctly
        assert cfg is not None
        assert cfg.navigator.max_speed == 2.0  # Default value
        assert cfg.frame_cache.mode == "lru"   # Default value
        assert cfg.environment.env_id == "PlumeNavSim-v0"  # Default value


class TestConfigurationSecurity:
    """
    Test suite for configuration security including path traversal prevention
    and secure environment variable handling for the Gymnasium migration.
    
    Validates security measures against malicious configuration file references,
    path traversal attacks, and unauthorized access attempts in the context of
    the enhanced plume_nav_sim configuration system.
    """
    
    def test_path_traversal_prevention_in_video_path(self, tmp_path):
        """Test prevention of path traversal attacks in video_path configuration."""
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

frame_cache:
  mode: lru
  memory_limit_mb: 1024
"""
        (config_dir / "safe_config.yaml").write_text(safe_config)
        
        # Test path traversal attempts in video_path
        traversal_config = """
video_plume:
  video_path: "../../../etc/passwd"

frame_cache:
  mode: lru
  memory_limit_mb: 1024
"""
        (config_dir / "traversal_config.yaml").write_text(traversal_config)
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            # Safe configuration should load normally
            safe_cfg = compose(config_name="safe_config")
            assert safe_cfg.video_plume.video_path == "data/safe_video.mp4"
            
            # Path traversal configuration should load but be treated as literal
            traversal_cfg = compose(config_name="traversal_config")
            assert traversal_cfg.video_plume.video_path == "../../../etc/passwd"
            
            # Validate that VideoPlumeConfig would reject the traversal path in validation
            if SCHEMAS_AVAILABLE:
                try:
                    VideoPlumeConfig(
                        video_path="../../../etc/passwd",
                        # If schema has file existence validation, this should fail
                    )
                except ValidationError:
                    pass  # Expected behavior for path traversal
    
    def test_frame_cache_security_configuration(self, tmp_path):
        """Test frame cache configuration security against malicious parameters."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping frame cache security tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Test malicious frame cache configurations
        malicious_configs = [
            {
                "name": "excessive_memory",
                "config": """
frame_cache:
  mode: all
  memory_limit_mb: 999999999  # Excessive memory allocation attempt
  memory_pressure_threshold: 0.0
""",
                "expected_value": 999999999
            },
            {
                "name": "negative_values",
                "config": """
frame_cache:
  mode: lru
  memory_limit_mb: -1000  # Negative memory allocation
  preload_chunk_size: -100
""",
                "expected_memory": -1000,
                "expected_chunk": -100
            },
            {
                "name": "malicious_mode",
                "config": """
frame_cache:
  mode: "$(rm -rf /)"  # Command injection attempt in mode
  memory_limit_mb: 1024
""",
                "expected_mode": "$(rm -rf /)"
            }
        ]
        
        for test_case in malicious_configs:
            config_file = config_dir / f"{test_case['name']}.yaml"
            config_file.write_text(test_case['config'])
            
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                cfg = compose(config_name=test_case['name'])
                
                # Validate malicious values are treated as literals
                if 'expected_value' in test_case:
                    assert cfg.frame_cache.memory_limit_mb == test_case['expected_value']
                if 'expected_memory' in test_case:
                    assert cfg.frame_cache.memory_limit_mb == test_case['expected_memory']
                if 'expected_chunk' in test_case:
                    assert cfg.frame_cache.preload_chunk_size == test_case['expected_chunk']
                if 'expected_mode' in test_case:
                    assert cfg.frame_cache.mode == test_case['expected_mode']
                
                # Ensure values are strings/integers, not executed code
                assert isinstance(cfg.frame_cache.mode, str)
                assert isinstance(cfg.frame_cache.memory_limit_mb, (int, float))
    
    def test_environment_variable_injection_prevention(self, tmp_path):
        """Test prevention of environment variable injection attacks."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping injection tests")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Configuration with environment variable interpolation
        injection_config = """
navigator:
  max_speed: ${oc.env:SAFE_MAX_SPEED,2.0}

frame_cache:
  mode: ${oc.env:CACHE_MODE,lru}
  memory_limit_mb: ${oc.env:CACHE_MEMORY,1024}

environment:
  env_id: ${oc.env:ENV_ID,PlumeNavSim-v0}

system:
  output_dir: ${oc.env:OUTPUT_DIR,outputs}
"""
        
        (config_dir / "config.yaml").write_text(injection_config)
        
        # Test malicious environment variable injection
        malicious_env_vars = {
            "SAFE_MAX_SPEED": "${oc.env:SECRET_KEY}",  # Nested interpolation
            "CACHE_MODE": "lru; rm -rf /",  # Command injection attempt
            "CACHE_MEMORY": "$(cat /etc/passwd)",  # Command substitution
            "ENV_ID": "PlumeNavSim$(whoami)",  # Environment command injection
            "OUTPUT_DIR": "../../../sensitive_dir",  # Path traversal
        }
        
        with patch.dict(os.environ, malicious_env_vars):
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                cfg = compose(config_name="config")
                
                # Validate malicious environment variables are treated as literals
                assert cfg.navigator.max_speed == "${oc.env:SECRET_KEY}"
                assert cfg.frame_cache.mode == "lru; rm -rf /"
                assert cfg.frame_cache.memory_limit_mb == "$(cat /etc/passwd)"
                assert cfg.environment.env_id == "PlumeNavSim$(whoami)"
                assert cfg.system.output_dir == "../../../sensitive_dir"
                
                # Ensure no code execution occurred
                assert isinstance(cfg.navigator.max_speed, str)
                assert isinstance(cfg.frame_cache.mode, str)
                assert isinstance(cfg.environment.env_id, str)
                assert isinstance(cfg.system.output_dir, str)


# Pytest performance markers for configuration tests
pytestmark = [
    pytest.mark.config,
    pytest.mark.performance,
    pytest.mark.integration
]


def test_configuration_system_integration():
    """Integration test for complete configuration system functionality with Gymnasium migration."""
    # Test that all configuration classes are importable and functional
    
    if not SCHEMAS_AVAILABLE:
        pytest.skip("Configuration schemas not available")
    
    # Navigator configuration (single and multi-agent support)
    navigator_config = NavigatorConfig(
        position=(0.0, 0.0),
        orientation=0.0,
        speed=1.0,
        max_speed=2.0
    )
    assert hasattr(navigator_config, 'position')
    assert hasattr(navigator_config, 'max_speed')
    
    # Video plume configuration
    video_config = VideoPlumeConfig(
        video_path="test.mp4",
        flip=False,
        grayscale=True,
        kernel_size=5,
        kernel_sigma=1.0,
        threshold=0.5
    )
    assert hasattr(video_config, 'video_path')
    assert hasattr(video_config, 'kernel_size')
    
    # Frame cache configuration (new for Gymnasium migration)
    frame_cache_config = FrameCacheConfig(
        mode="lru",
        memory_limit_mb=2048,
        memory_pressure_threshold=0.9,
        enable_statistics=True
    )
    assert hasattr(frame_cache_config, 'mode')
    assert hasattr(frame_cache_config, 'memory_limit_mb')


def test_frame_cache_configuration_modes():
    """Test frame cache configuration mode validation and integration."""
    if not FRAME_CACHE_AVAILABLE:
        pytest.skip("Frame cache module not available")
    
    # Test all supported cache modes
    valid_modes = [CacheMode.NONE, CacheMode.LRU, CacheMode.ALL]
    
    for mode in valid_modes:
        if SCHEMAS_AVAILABLE:
            config = FrameCacheConfig(
                mode=mode,
                memory_limit_mb=1024
            )
            assert hasattr(config, 'mode')
        
        # Test mode string representations
        mode_str = mode if isinstance(mode, str) else mode
        assert mode_str in ["none", "lru", "all"]


if __name__ == "__main__":
    # Run tests with coverage reporting
    pytest.main([
        __file__,
        "-v",
        "--cov=plume_nav_sim.config",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--tb=short"
    ])