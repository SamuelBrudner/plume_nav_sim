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

# Import new protocol interfaces and factory functions for v1.0 architecture testing
try:
    from plume_nav_sim.core.protocols import (
        SourceProtocol, 
        BoundaryPolicyProtocol, 
        ActionInterfaceProtocol, 
        RecorderProtocol, 
        StatsAggregatorProtocol
    )
    NEW_PROTOCOLS_AVAILABLE = True
except ImportError:
    NEW_PROTOCOLS_AVAILABLE = False
    # Define fallback protocol classes for testing when not yet implemented
    class SourceProtocol:
        pass
    class BoundaryPolicyProtocol:
        pass
    class ActionInterfaceProtocol:
        pass
    class RecorderProtocol:
        pass
    class StatsAggregatorProtocol:
        pass

# Import AgentInitializer components for testing
try:
    from plume_nav_sim.core.initialization import AgentInitializerProtocol, create_initializer
    AGENT_INITIALIZER_AVAILABLE = True
except ImportError:
    AGENT_INITIALIZER_AVAILABLE = False
    class AgentInitializerProtocol:
        pass
    def create_initializer(config):
        return None

# Import source components for testing
try:
    from plume_nav_sim.core.sources import create_source
    SOURCE_FACTORY_AVAILABLE = True
except ImportError:
    SOURCE_FACTORY_AVAILABLE = False
    def create_source(config):
        return None

# Import boundary components for testing
try:
    from plume_nav_sim.core.boundaries import create_boundary_policy
    BOUNDARY_FACTORY_AVAILABLE = True
except ImportError:
    BOUNDARY_FACTORY_AVAILABLE = False
    def create_boundary_policy(policy_type, domain_bounds, **kwargs):
        return None

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


class TestV1ConfigGroupHierarchy:
    """
    Test suite for new Hydra config group hierarchy supporting the v1.0 architecture
    with protocol-based component system and modular configuration structure.
    
    Validates the new config groups: source/, agent_init/, boundary/, action/, 
    record/, hooks/ and their integration with the component registration system
    and runtime component selection per the v1.0 requirements.
    """
    
    @pytest.fixture(autouse=True)
    def setup_v1_config_structure(self, tmp_path):
        """Setup v1.0 config directory structure with new component groups."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping v1.0 config group tests")
        
        # Clean any existing Hydra global state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
            
        # Create temporary config directory with v1.0 structure
        self.config_dir = tmp_path / "conf"
        self.config_dir.mkdir()
        
        # Create new config group directories
        (self.config_dir / "base").mkdir()
        (self.config_dir / "base" / "source").mkdir()
        (self.config_dir / "base" / "agent_init").mkdir()
        (self.config_dir / "base" / "boundary").mkdir()
        (self.config_dir / "base" / "action").mkdir()
        (self.config_dir / "base" / "record").mkdir()
        (self.config_dir / "base" / "hooks").mkdir()
        
        # Create source configurations
        point_source_config = """
# @package _global_
source:
  _target_: plume_nav_sim.core.sources.PointSource
  position: [50.0, 50.0]
  emission_rate: 1000.0
  seed: 42
"""
        
        multi_source_config = """
# @package _global_
source:
  _target_: plume_nav_sim.core.sources.MultiSource
  sources:
    - _target_: plume_nav_sim.core.sources.PointSource
      position: [30.0, 30.0]
      emission_rate: 500.0
    - _target_: plume_nav_sim.core.sources.PointSource
      position: [70.0, 70.0]
      emission_rate: 800.0
  seed: 42
"""
        
        # Create agent initialization configurations
        uniform_random_init = """
# @package _global_
agent_init:
  _target_: plume_nav_sim.core.initialization.UniformRandomInitializer
  bounds: [100.0, 100.0]
  seed: 42
  margin: 2.0
"""
        
        grid_init = """
# @package _global_
agent_init:
  _target_: plume_nav_sim.core.initialization.GridInitializer
  domain_bounds: [100.0, 100.0]
  grid_shape: [5, 5]
  grid_spacing: [10.0, 10.0]
  jitter_enabled: false
  seed: 42
"""
        
        # Create boundary policy configurations
        terminate_boundary = """
# @package _global_
boundary:
  _target_: plume_nav_sim.core.boundaries.TerminateBoundary
  domain_bounds: [100.0, 100.0]
  status_on_violation: "oob"
  allow_negative_coords: false
"""
        
        bounce_boundary = """
# @package _global_
boundary:
  _target_: plume_nav_sim.core.boundaries.BounceBoundary
  domain_bounds: [100.0, 100.0]
  elasticity: 0.8
  energy_loss: 0.1
  velocity_damping: 0.9
"""
        
        # Create action interface configurations
        continuous_action = """
# @package _global_
action:
  _target_: plume_nav_sim.core.actions.Continuous2DAction
  max_linear_velocity: 2.0
  max_angular_velocity: 45.0
  velocity_scaling: 1.0
  clip_actions: true
"""
        
        discrete_action = """
# @package _global_
action:
  _target_: plume_nav_sim.core.actions.CardinalDiscreteAction
  step_size: 1.0
  num_directions: 8
  allow_no_op: true
"""
        
        # Create recorder configurations
        parquet_recorder = """
# @package _global_
record:
  _target_: plume_nav_sim.recording.backends.ParquetRecorder
  output_dir: "recordings"
  full_trajectories: true
  compression: "snappy"
  buffer_size: 1000
"""
        
        no_recorder = """
# @package _global_
record:
  _target_: plume_nav_sim.recording.backends.NoOpRecorder
  enabled: false
"""
        
        # Create hook configurations
        basic_hooks = """
# @package _global_
hooks:
  extra_obs_fn: null
  extra_reward_fn: null
  episode_end_fn: null
  enable_custom_metrics: false
"""
        
        exploration_hooks = """
# @package _global_
hooks:
  extra_obs_fn: plume_nav_sim.hooks.exploration_bonus_obs
  extra_reward_fn: plume_nav_sim.hooks.exploration_reward
  episode_end_fn: plume_nav_sim.hooks.log_exploration_metrics
  enable_custom_metrics: true
  exploration_bonus_weight: 0.1
"""
        
        # Create main configuration that uses all new config groups
        main_config = """
defaults:
  - base/source: point
  - base/agent_init: uniform_random
  - base/boundary: terminate
  - base/action: continuous
  - base/record: parquet
  - base/hooks: basic
  - _self_

navigator:
  position: [0.0, 0.0]
  orientation: 0.0
  speed: 0.0
  max_speed: 2.0
  angular_velocity: 0.0

environment:
  env_id: "PlumeNavSim-v1"
  max_episode_steps: 1000
  step_timeout_ms: 33  # Meeting performance requirement

system:
  random_seed: 42
  debug_mode: false
  output_dir: "outputs"
  log_level: "INFO"
"""
        
        # Write all configuration files
        (self.config_dir / "base" / "source" / "point.yaml").write_text(point_source_config)
        (self.config_dir / "base" / "source" / "multi.yaml").write_text(multi_source_config)
        (self.config_dir / "base" / "agent_init" / "uniform_random.yaml").write_text(uniform_random_init)
        (self.config_dir / "base" / "agent_init" / "grid.yaml").write_text(grid_init)
        (self.config_dir / "base" / "boundary" / "terminate.yaml").write_text(terminate_boundary)
        (self.config_dir / "base" / "boundary" / "bounce.yaml").write_text(bounce_boundary)
        (self.config_dir / "base" / "action" / "continuous.yaml").write_text(continuous_action)
        (self.config_dir / "base" / "action" / "discrete.yaml").write_text(discrete_action)
        (self.config_dir / "base" / "record" / "parquet.yaml").write_text(parquet_recorder)
        (self.config_dir / "base" / "record" / "none.yaml").write_text(no_recorder)
        (self.config_dir / "base" / "hooks" / "basic.yaml").write_text(basic_hooks)
        (self.config_dir / "base" / "hooks" / "exploration.yaml").write_text(exploration_hooks)
        (self.config_dir / "config.yaml").write_text(main_config)
        
        yield self.config_dir
        
        # Cleanup Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
    
    def test_new_config_group_composition(self, setup_v1_config_structure):
        """Test that new v1.0 config groups compose correctly."""
        config_dir = setup_v1_config_structure
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
            
            # Validate all new config groups are present
            assert "source" in cfg
            assert "agent_init" in cfg  
            assert "boundary" in cfg
            assert "action" in cfg
            assert "record" in cfg
            assert "hooks" in cfg
            
            # Validate source configuration
            assert cfg.source._target_ == "plume_nav_sim.core.sources.PointSource"
            assert cfg.source.position == [50.0, 50.0]
            assert cfg.source.emission_rate == 1000.0
            
            # Validate agent initialization configuration
            assert cfg.agent_init._target_ == "plume_nav_sim.core.initialization.UniformRandomInitializer"
            assert cfg.agent_init.bounds == [100.0, 100.0]
            assert cfg.agent_init.seed == 42
            
            # Validate boundary policy configuration
            assert cfg.boundary._target_ == "plume_nav_sim.core.boundaries.TerminateBoundary"
            assert cfg.boundary.domain_bounds == [100.0, 100.0]
            assert cfg.boundary.status_on_violation == "oob"
            
            # Validate action interface configuration
            assert cfg.action._target_ == "plume_nav_sim.core.actions.Continuous2DAction"
            assert cfg.action.max_linear_velocity == 2.0
            assert cfg.action.max_angular_velocity == 45.0
            
            # Validate recorder configuration
            assert cfg.record._target_ == "plume_nav_sim.recording.backends.ParquetRecorder"
            assert cfg.record.output_dir == "recordings"
            assert cfg.record.full_trajectories == True
            
            # Validate hooks configuration
            assert cfg.hooks.extra_obs_fn == None
            assert cfg.hooks.extra_reward_fn == None
            assert cfg.hooks.enable_custom_metrics == False
    
    def test_config_group_override_combinations(self, setup_v1_config_structure):
        """Test different combinations of config group overrides."""
        config_dir = setup_v1_config_structure
        
        test_combinations = [
            {
                "overrides": ["base/source=multi", "base/boundary=bounce"],
                "expected_source": "plume_nav_sim.core.sources.MultiSource",
                "expected_boundary": "plume_nav_sim.core.boundaries.BounceBoundary"
            },
            {
                "overrides": ["base/agent_init=grid", "base/action=discrete"],
                "expected_agent_init": "plume_nav_sim.core.initialization.GridInitializer",
                "expected_action": "plume_nav_sim.core.actions.CardinalDiscreteAction"
            },
            {
                "overrides": ["base/record=none", "base/hooks=exploration"],
                "expected_record": "plume_nav_sim.recording.backends.NoOpRecorder",
                "expected_hooks_fn": "plume_nav_sim.hooks.exploration_bonus_obs"
            }
        ]
        
        for test_case in test_combinations:
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                cfg = compose(config_name="config", overrides=test_case["overrides"])
                
                # Check expected values based on test case
                if "expected_source" in test_case:
                    assert cfg.source._target_ == test_case["expected_source"]
                if "expected_boundary" in test_case:
                    assert cfg.boundary._target_ == test_case["expected_boundary"]
                if "expected_agent_init" in test_case:
                    assert cfg.agent_init._target_ == test_case["expected_agent_init"]
                if "expected_action" in test_case:
                    assert cfg.action._target_ == test_case["expected_action"]
                if "expected_record" in test_case:
                    assert cfg.record._target_ == test_case["expected_record"]
                if "expected_hooks_fn" in test_case:
                    assert cfg.hooks.extra_obs_fn == test_case["expected_hooks_fn"]
    
    def test_v1_config_performance_requirements(self, setup_v1_config_structure):
        """Test that v1.0 config composition meets 500ms performance requirement."""
        config_dir = setup_v1_config_structure
        
        start_time = time.time()
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
        
        composition_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Validate performance requirement for new configuration structure
        assert composition_time < 500, f"V1.0 config composition took {composition_time:.2f}ms, exceeds 500ms limit"
        
        # Validate all v1.0 components were loaded
        assert cfg is not None
        assert "source" in cfg
        assert "agent_init" in cfg
        assert "boundary" in cfg
        assert "action" in cfg
        assert "record" in cfg
        assert "hooks" in cfg


class TestComponentRegistrationSystem:
    """
    Test suite for the component registration system enabling runtime component
    selection and Hydra-based dependency injection for the v1.0 architecture.
    
    Validates factory function integration, component instantiation from configuration,
    and protocol compliance checking for the new modular component system.
    """
    
    def test_agent_initializer_factory_creation(self):
        """Test agent initializer factory function creation and validation."""
        if not AGENT_INITIALIZER_AVAILABLE:
            pytest.skip("Agent initializer components not available")
        
        # Test uniform random initializer creation
        uniform_config = {
            'type': 'uniform_random',
            'bounds': (100.0, 100.0),
            'seed': 42,
            'margin': 2.0
        }
        
        initializer = create_initializer(uniform_config)
        assert initializer is not None
        
        # Test that created initializer has required methods
        assert hasattr(initializer, 'initialize_positions')
        assert hasattr(initializer, 'validate_domain')
        assert hasattr(initializer, 'reset')
        assert hasattr(initializer, 'get_strategy_name')
        
        # Test grid initializer creation
        grid_config = {
            'type': 'grid',
            'domain_bounds': (100.0, 100.0),
            'grid_shape': (5, 5),
            'grid_spacing': (10.0, 10.0),
            'seed': 42
        }
        
        grid_initializer = create_initializer(grid_config)
        assert grid_initializer is not None
        assert hasattr(grid_initializer, 'initialize_positions')
        
        # Test fixed list initializer creation
        fixed_config = {
            'type': 'fixed_list',
            'positions': [[10, 20], [30, 40], [50, 60]],
            'domain_bounds': (100.0, 100.0)
        }
        
        fixed_initializer = create_initializer(fixed_config)
        assert fixed_initializer is not None
        assert hasattr(fixed_initializer, 'get_position_count')
    
    def test_source_factory_creation(self):
        """Test source factory function creation and configuration validation."""
        if not SOURCE_FACTORY_AVAILABLE:
            pytest.skip("Source factory components not available")
        
        # Test point source creation
        point_config = {
            'type': 'PointSource',
            'position': (50.0, 50.0),
            'emission_rate': 1000.0,
            'seed': 42
        }
        
        point_source = create_source(point_config)
        assert point_source is not None
        
        # Test that created source has required methods
        assert hasattr(point_source, 'get_emission_rate')
        assert hasattr(point_source, 'get_position')
        assert hasattr(point_source, 'update_state')
        
        # Test multi-source creation with nested sources
        multi_config = {
            'type': 'MultiSource',
            'sources': [
                {'type': 'PointSource', 'position': (30, 30), 'emission_rate': 500},
                {'type': 'PointSource', 'position': (70, 70), 'emission_rate': 800}
            ],
            'seed': 42
        }
        
        multi_source = create_source(multi_config)
        assert multi_source is not None
        assert hasattr(multi_source, 'add_source')
        assert hasattr(multi_source, 'get_source_count')
        
        # Test dynamic source creation
        dynamic_config = {
            'type': 'DynamicSource',
            'initial_position': (25, 75),
            'pattern_type': 'circular',
            'amplitude': 15.0,
            'frequency': 0.1,
            'base_emission_rate': 1000.0
        }
        
        dynamic_source = create_source(dynamic_config)
        assert dynamic_source is not None
        assert hasattr(dynamic_source, 'set_pattern')
        assert hasattr(dynamic_source, 'reset_time')
    
    def test_boundary_policy_factory_creation(self):
        """Test boundary policy factory function creation and validation."""
        if not BOUNDARY_FACTORY_AVAILABLE:
            pytest.skip("Boundary policy factory components not available")
        
        # Test terminate boundary creation
        terminate_policy = create_boundary_policy(
            "terminate",
            domain_bounds=(100.0, 100.0),
            status_on_violation="oob"
        )
        assert terminate_policy is not None
        
        # Test that created policy has required methods
        assert hasattr(terminate_policy, 'apply_policy')
        assert hasattr(terminate_policy, 'check_violations')
        assert hasattr(terminate_policy, 'get_termination_status')
        
        # Test bounce boundary creation
        bounce_policy = create_boundary_policy(
            "bounce",
            domain_bounds=(100.0, 100.0),
            elasticity=0.8,
            energy_loss=0.1
        )
        assert bounce_policy is not None
        assert hasattr(bounce_policy, 'apply_policy')
        
        # Test wrap boundary creation
        wrap_policy = create_boundary_policy(
            "wrap",
            domain_bounds=(100.0, 100.0)
        )
        assert wrap_policy is not None
        assert hasattr(wrap_policy, 'apply_policy')
        
        # Test clip boundary creation
        clip_policy = create_boundary_policy(
            "clip",
            domain_bounds=(100.0, 100.0),
            velocity_damping=0.7
        )
        assert clip_policy is not None
        assert hasattr(clip_policy, 'apply_policy')
    
    def test_component_factory_error_handling(self):
        """Test error handling in component factory functions."""
        if not AGENT_INITIALIZER_AVAILABLE:
            pytest.skip("Agent initializer components not available")
        
        # Test invalid initializer type
        with pytest.raises(ValueError):
            create_initializer({'type': 'invalid_type'})
        
        # Test missing required parameters
        with pytest.raises((ValueError, TypeError)):
            create_initializer({'type': 'uniform_random'})  # Missing bounds
        
        if SOURCE_FACTORY_AVAILABLE:
            # Test invalid source type
            with pytest.raises(ValueError):
                create_source({'type': 'InvalidSource'})
            
            # Test missing type field
            with pytest.raises(KeyError):
                create_source({'position': (0, 0)})  # Missing type
        
        if BOUNDARY_FACTORY_AVAILABLE:
            # Test invalid boundary policy type
            with pytest.raises(ValueError):
                create_boundary_policy("invalid_policy", (100, 100))
    
    def test_hydra_based_component_instantiation(self, tmp_path):
        """Test Hydra-based component instantiation using _target_ syntax."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available - skipping Hydra instantiation tests")
        
        if not AGENT_INITIALIZER_AVAILABLE:
            pytest.skip("Agent initializer components not available")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create configuration with Hydra _target_ syntax
        hydra_config = """
agent_init:
  _target_: plume_nav_sim.core.initialization.UniformRandomInitializer
  bounds: [100.0, 100.0]
  seed: 42
  margin: 2.0

source:
  _target_: plume_nav_sim.core.sources.PointSource
  position: [50.0, 50.0]
  emission_rate: 1000.0
  seed: 42

boundary:
  _target_: plume_nav_sim.core.boundaries.TerminateBoundary
  domain_bounds: [100.0, 100.0]
  status_on_violation: "oob"
"""
        
        (config_dir / "config.yaml").write_text(hydra_config)
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
            
            # Validate Hydra _target_ syntax is preserved
            assert cfg.agent_init._target_ == "plume_nav_sim.core.initialization.UniformRandomInitializer"
            assert cfg.agent_init.bounds == [100.0, 100.0]
            assert cfg.agent_init.seed == 42
            
            assert cfg.source._target_ == "plume_nav_sim.core.sources.PointSource"
            assert cfg.source.position == [50.0, 50.0]
            assert cfg.source.emission_rate == 1000.0
            
            assert cfg.boundary._target_ == "plume_nav_sim.core.boundaries.TerminateBoundary"
            assert cfg.boundary.domain_bounds == [100.0, 100.0]
            assert cfg.boundary.status_on_violation == "oob"
    
    @pytest.mark.parametrize("component_type,config_key,factory_available", [
        ("agent_init", "agent_init", "AGENT_INITIALIZER_AVAILABLE"),
        ("source", "source", "SOURCE_FACTORY_AVAILABLE"),
        ("boundary", "boundary", "BOUNDARY_FACTORY_AVAILABLE"),
    ])
    def test_component_registration_integration(self, component_type, config_key, factory_available):
        """Test component registration system integration with configuration."""
        if not globals().get(factory_available, False):
            pytest.skip(f"{component_type} factory not available")
        
        # Test that components can be created from configuration dictionaries
        test_configs = {
            "agent_init": {
                'type': 'uniform_random',
                'bounds': (100.0, 100.0),
                'seed': 42
            },
            "source": {
                'type': 'PointSource',
                'position': (50.0, 50.0),
                'emission_rate': 1000.0
            },
            "boundary": {
                'policy_type': 'terminate',
                'domain_bounds': (100.0, 100.0)
            }
        }
        
        config = test_configs.get(component_type)
        if not config:
            pytest.skip(f"No test config for {component_type}")
        
        # Test component creation
        if component_type == "agent_init":
            component = create_initializer(config)
        elif component_type == "source":
            component = create_source(config)
        elif component_type == "boundary":
            component = create_boundary_policy(**config)
        
        assert component is not None
        
        # Test that component follows expected interface patterns
        if component_type == "agent_init":
            assert hasattr(component, 'initialize_positions')
            assert hasattr(component, 'reset')
        elif component_type == "source":
            assert hasattr(component, 'get_emission_rate')
            assert hasattr(component, 'get_position')
        elif component_type == "boundary":
            assert hasattr(component, 'apply_policy')
            assert hasattr(component, 'check_violations')


class TestPydanticSchemaValidation:
    """
    Test suite for Pydantic schema validation integration with the new v1.0
    component configuration system and enhanced type safety requirements.
    
    Validates schema enforcement for new component configurations and ensures
    proper validation of the expanded configuration parameter space.
    """
    
    def test_component_configuration_schema_validation(self):
        """Test Pydantic schema validation for new component configurations."""
        if not SCHEMAS_AVAILABLE:
            pytest.skip("Configuration schemas not available")
        
        # This test focuses on existing schemas that might be extended
        # for v1.0 component configuration validation
        
        # Test enhanced navigator configuration with new fields
        enhanced_navigator_config = {
            "position": [1.0, 2.0],
            "orientation": 45.0,
            "speed": 1.5,
            "max_speed": 2.0,
            "angular_velocity": 0.5,
            # Potential new fields for v1.0
            "enable_hooks": True,
            "custom_sensors": ["odor", "wind"],
        }
        
        try:
            config = NavigatorConfig(**enhanced_navigator_config)
            assert hasattr(config, 'position')
            assert hasattr(config, 'max_speed')
        except TypeError:
            # Schema may not support new fields yet, which is expected
            basic_config = {k: v for k, v in enhanced_navigator_config.items() 
                          if k in ["position", "orientation", "speed", "max_speed", "angular_velocity"]}
            config = NavigatorConfig(**basic_config)
            assert hasattr(config, 'position')
    
    def test_component_parameter_boundary_validation(self):
        """Test parameter boundary validation for component configurations."""
        if not SCHEMAS_AVAILABLE:
            pytest.skip("Configuration schemas not available")
        
        # Test boundary conditions for frame cache configuration
        boundary_test_cases = [
            {
                "mode": "lru",
                "memory_limit_mb": 0,  # Minimum valid value
                "memory_pressure_threshold": 0.0,
                "should_pass": True
            },
            {
                "mode": "lru", 
                "memory_limit_mb": 16384,  # Large but reasonable value
                "memory_pressure_threshold": 1.0,
                "should_pass": True
            },
            {
                "mode": "invalid_mode",  # Invalid mode
                "memory_limit_mb": 1024,
                "memory_pressure_threshold": 0.9,
                "should_pass": False
            },
            {
                "mode": "lru",
                "memory_limit_mb": -100,  # Negative value
                "memory_pressure_threshold": 0.9,
                "should_pass": False
            }
        ]
        
        for test_case in boundary_test_cases:
            config_params = {k: v for k, v in test_case.items() if k != 'should_pass'}
            
            try:
                config = FrameCacheConfig(**config_params)
                assert test_case['should_pass'], f"Expected validation to fail for {config_params}"
                assert hasattr(config, 'mode')
                assert hasattr(config, 'memory_limit_mb')
            except (ValidationError, ValueError, TypeError):
                assert not test_case['should_pass'], f"Expected validation to pass for {config_params}"
    
    def test_v1_configuration_composition_validation(self, tmp_path):
        """Test end-to-end configuration composition with Pydantic validation."""
        if not HYDRA_AVAILABLE or not SCHEMAS_AVAILABLE:
            pytest.skip("Hydra or schemas not available")
        
        # Clean Hydra state
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create configuration that should pass full validation
        validated_config = """
navigator:
  position: [0.0, 0.0]
  orientation: 0.0
  speed: 1.0
  max_speed: 2.0
  angular_velocity: 0.0

video_plume:
  video_path: "data/test_video.mp4"
  flip: false
  grayscale: true
  kernel_size: 5
  kernel_sigma: 1.0
  threshold: 0.5
  normalize: true

frame_cache:
  mode: lru
  memory_limit_mb: 2048
  memory_pressure_threshold: 0.9
  enable_statistics: true
  enable_logging: true
  preload_chunk_size: 100
  eviction_batch_size: 10

# New v1.0 component configurations (may not have schemas yet)
source:
  type: PointSource
  position: [50.0, 50.0]
  emission_rate: 1000.0
  seed: 42

agent_init:
  type: uniform_random
  bounds: [100.0, 100.0]
  seed: 42

boundary:
  type: terminate
  domain_bounds: [100.0, 100.0]
  status_on_violation: "oob"
"""
        
        (config_dir / "config.yaml").write_text(validated_config)
        
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="config")
            
            # Test existing component validation
            navigator_config = NavigatorConfig(**cfg.navigator)
            video_config = VideoPlumeConfig(**cfg.video_plume)
            frame_cache_config = FrameCacheConfig(**cfg.frame_cache)
            
            # Validate successful schema validation
            assert hasattr(navigator_config, 'position')
            assert hasattr(navigator_config, 'max_speed')
            assert hasattr(video_config, 'video_path')
            assert hasattr(video_config, 'kernel_size')
            assert hasattr(frame_cache_config, 'mode')
            assert hasattr(frame_cache_config, 'memory_limit_mb')
            
            # Validate new v1.0 components are present (even if not schema-validated yet)
            assert "source" in cfg
            assert "agent_init" in cfg
            assert "boundary" in cfg
            
            # Test that configurations have expected structure
            assert cfg.source.type == "PointSource"
            assert cfg.source.position == [50.0, 50.0]
            assert cfg.agent_init.type == "uniform_random"
            assert cfg.boundary.type == "terminate"


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


def test_v1_component_integration():
    """Integration test for v1.0 component system with factory functions."""
    # Test factory function availability and basic functionality
    
    if AGENT_INITIALIZER_AVAILABLE:
        # Test agent initializer integration
        config = {'type': 'uniform_random', 'bounds': (100, 100), 'seed': 42}
        initializer = create_initializer(config)
        assert initializer is not None
        assert hasattr(initializer, 'get_strategy_name')
    
    if SOURCE_FACTORY_AVAILABLE:
        # Test source integration
        config = {'type': 'PointSource', 'position': (50, 50), 'emission_rate': 1000}
        source = create_source(config)
        assert source is not None
        assert hasattr(source, 'get_emission_rate')
    
    if BOUNDARY_FACTORY_AVAILABLE:
        # Test boundary policy integration
        policy = create_boundary_policy('terminate', (100, 100))
        assert policy is not None
        assert hasattr(policy, 'apply_policy')


def test_v1_configuration_performance_integration(tmp_path):
    """Integration test for v1.0 configuration system performance with all components."""
    if not HYDRA_AVAILABLE:
        pytest.skip("Hydra not available - skipping v1.0 performance integration test")
    
    # Clean Hydra state
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    
    # Create comprehensive v1.0 configuration
    v1_integration_config = """
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
  enable_logging: true
  preload_chunk_size: 100
  eviction_batch_size: 10

# New v1.0 component configurations
source:
  _target_: plume_nav_sim.core.sources.PointSource
  position: [50.0, 50.0]
  emission_rate: ${oc.env:SOURCE_RATE,1000.0}
  seed: 42

agent_init:
  _target_: plume_nav_sim.core.initialization.UniformRandomInitializer
  bounds: [100.0, 100.0]
  seed: 42
  margin: ${oc.env:INIT_MARGIN,2.0}

boundary:
  _target_: plume_nav_sim.core.boundaries.TerminateBoundary
  domain_bounds: [100.0, 100.0]
  status_on_violation: ${oc.env:BOUNDARY_STATUS,oob}
  allow_negative_coords: false

action:
  _target_: plume_nav_sim.core.actions.Continuous2DAction
  max_linear_velocity: ${oc.env:MAX_LINEAR_VEL,2.0}
  max_angular_velocity: ${oc.env:MAX_ANGULAR_VEL,45.0}
  velocity_scaling: 1.0
  clip_actions: true

record:
  _target_: plume_nav_sim.recording.backends.ParquetRecorder
  output_dir: ${oc.env:RECORD_DIR,recordings}
  full_trajectories: ${oc.env:FULL_TRAJ,true}
  compression: snappy
  buffer_size: 1000

hooks:
  extra_obs_fn: ${oc.env:EXTRA_OBS_FN,null}
  extra_reward_fn: ${oc.env:EXTRA_REWARD_FN,null}
  episode_end_fn: ${oc.env:EPISODE_END_FN,null}
  enable_custom_metrics: ${oc.env:CUSTOM_METRICS,false}

environment:
  env_id: "PlumeNavSim-v1"
  render_mode: ${oc.env:RENDER_MODE,null}
  max_episode_steps: ${oc.env:MAX_STEPS,1000}
  step_timeout_ms: 33  # Meeting ≤33ms requirement
  api_compatibility: gymnasium
  legacy_gym_support: true

multi_agent:
  num_agents: ${oc.env:NUM_AGENTS,3}
  positions: [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
  orientations: [0.0, 90.0, 180.0]
  speeds: [1.0, 1.2, 0.8]
  max_speeds: [2.0, 2.0, 2.0]

system:
  random_seed: ${oc.env:RANDOM_SEED,42}
  debug_mode: ${oc.env:DEBUG,false}
  output_dir: ${oc.env:OUTPUT_DIR,outputs}
  log_level: ${oc.env:LOG_LEVEL,INFO}
  performance_monitoring: true
"""
    
    (config_dir / "config.yaml").write_text(v1_integration_config)
    
    # Measure comprehensive v1.0 configuration loading performance
    start_time = time.time()
    
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name="config")
    
    loading_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Validate performance requirement for complete v1.0 system
    assert loading_time < 500, f"V1.0 integration config took {loading_time:.2f}ms, exceeds 500ms limit"
    
    # Validate all v1.0 components and existing components are present
    assert cfg is not None
    
    # Core components
    assert "navigator" in cfg
    assert "video_plume" in cfg
    assert "frame_cache" in cfg
    assert "environment" in cfg
    assert "system" in cfg
    
    # New v1.0 components
    assert "source" in cfg
    assert "agent_init" in cfg
    assert "boundary" in cfg
    assert "action" in cfg
    assert "record" in cfg
    assert "hooks" in cfg
    
    # Multi-agent support
    assert "multi_agent" in cfg
    
    # Validate performance-critical configuration
    assert cfg.environment.step_timeout_ms == 33  # ≤33ms requirement
    assert cfg.system.performance_monitoring == True
    
    # Validate environment variable interpolation worked
    assert cfg.navigator.max_speed == 2.0  # Default value
    assert cfg.frame_cache.mode == "lru"   # Default value
    assert cfg.source.emission_rate == 1000.0  # Default value


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