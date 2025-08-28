"""
Tests for the enhanced configuration models with Enum support.

This test suite validates the comprehensive configuration system including:
- Enum-based configuration models replacing Literal types
- Dataclass field validation with Enum support
- Environment variable interpolation with Enum types
- Serialization/deserialization with Enum preservation
- Backward compatibility with string values
- Hydra integration for structured configs
"""

import os
import pytest
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import asdict, is_dataclass
from enum import Enum

# Import configuration models with Enum support
from odor_plume_nav.config.models import (
    # Enum classes
    NavigatorMode,
    OutputFormat,
    LogLevel,
    CacheMode,
    EvictionPolicy,
    PreloadStrategy,
    
    # Configuration dataclasses
    NavigatorConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    VideoPlumeConfig,
    SimulationConfig,
    FrameCacheConfig,
    
    # Helper functions
    dataclass_to_dict,
    dict_to_dataclass,
    validate_env_interpolation,
    resolve_env_value,
    _coerce_enum,
    
    # Factory functions
    create_default_navigator_config,
    create_default_single_agent_config,
    create_default_multi_agent_config,
    create_default_video_plume_config,
    create_default_simulation_config,
    create_default_frame_cache_config,
)


class TestEnumClasses:
    """Tests for the Enum classes that replaced Literal types."""
    
    def test_navigator_mode_enum(self):
        """Test NavigatorMode enum values and string conversion."""
        assert NavigatorMode.SINGLE.value == 'single'
        assert NavigatorMode.MULTI.value == 'multi'
        assert NavigatorMode.AUTO.value == 'auto'
        
        # Test string conversion
        assert str(NavigatorMode.SINGLE) == 'NavigatorMode.SINGLE'
        assert NavigatorMode.SINGLE.value == 'single'
    
    def test_output_format_enum(self):
        """Test OutputFormat enum values and string conversion."""
        assert OutputFormat.NUMPY.value == 'numpy'
        assert OutputFormat.CSV.value == 'csv'
        assert OutputFormat.HDF5.value == 'hdf5'
        assert OutputFormat.JSON.value == 'json'
    
    def test_log_level_enum(self):
        """Test LogLevel enum values and string conversion."""
        assert LogLevel.DEBUG.value == 'DEBUG'
        assert LogLevel.INFO.value == 'INFO'
        assert LogLevel.WARNING.value == 'WARNING'
        assert LogLevel.ERROR.value == 'ERROR'
    
    def test_cache_mode_enum(self):
        """Test CacheMode enum values and string conversion."""
        assert CacheMode.NONE.value == 'none'
        assert CacheMode.LRU.value == 'lru'
        assert CacheMode.ALL.value == 'all'
    
    def test_eviction_policy_enum(self):
        """Test EvictionPolicy enum values and string conversion."""
        assert EvictionPolicy.LRU.value == 'lru'
        assert EvictionPolicy.FIFO.value == 'fifo'
        assert EvictionPolicy.RANDOM.value == 'random'
    
    def test_preload_strategy_enum(self):
        """Test PreloadStrategy enum values and string conversion."""
        assert PreloadStrategy.SEQUENTIAL.value == 'sequential'
        assert PreloadStrategy.RANDOM.value == 'random'
        assert PreloadStrategy.ADAPTIVE.value == 'adaptive'


class TestEnumCoercion:
    """Tests for enum coercion functionality."""
    
    def test_coerce_enum_with_enum_value(self):
        """Test coercing an existing enum value."""
        value = NavigatorMode.SINGLE
        result = _coerce_enum(value, NavigatorMode)
        assert result is value
    
    def test_coerce_enum_with_string_value(self):
        """Test coercing a string to enum."""
        value = 'single'
        result = _coerce_enum(value, NavigatorMode)
        assert result == NavigatorMode.SINGLE
        assert isinstance(result, NavigatorMode)
    
    def test_coerce_enum_with_case_insensitive_string(self):
        """Test coercing a case-insensitive string to enum."""
        value = 'SiNgLe'
        result = _coerce_enum(value, NavigatorMode)
        assert result == NavigatorMode.SINGLE
    
    def test_coerce_enum_with_environment_variable(self):
        """Test coercing an environment variable pattern."""
        value = '${oc.env:NAVIGATOR_MODE,single}'
        result = _coerce_enum(value, NavigatorMode)
        assert result == value  # Should preserve env var pattern
    
    def test_coerce_enum_with_none(self):
        """Test coercing None value."""
        result = _coerce_enum(None, NavigatorMode)
        assert result is None
    
    def test_coerce_enum_with_invalid_string(self):
        """Test coercing an invalid string raises ValueError."""
        with pytest.raises(ValueError):
            _coerce_enum('invalid_mode', NavigatorMode)
    
    def test_coerce_enum_with_invalid_type(self):
        """Test coercing an invalid type raises ValueError."""
        with pytest.raises(ValueError):
            _coerce_enum(123, NavigatorMode)


class TestNavigatorConfig:
    """Tests for NavigatorConfig with Enum support."""
    
    def test_create_with_enum_mode(self):
        """Test creating NavigatorConfig with enum mode."""
        config = NavigatorConfig(mode=NavigatorMode.SINGLE)
        assert config.mode == NavigatorMode.SINGLE
    
    def test_create_with_string_mode(self):
        """Test creating NavigatorConfig with string mode."""
        config = NavigatorConfig(mode='single')
        assert config.mode == NavigatorMode.SINGLE
        assert isinstance(config.mode, NavigatorMode)
    
    def test_auto_mode_detection_single(self):
        """Test auto mode detection for single agent."""
        config = NavigatorConfig(
            mode=NavigatorMode.AUTO,
            position=(50.0, 50.0)
        )
        # Post-init should detect single agent mode
        assert config.mode == NavigatorMode.SINGLE
    
    def test_auto_mode_detection_multi(self):
        """Test auto mode detection for multi agent."""
        config = NavigatorConfig(
            mode=NavigatorMode.AUTO,
            positions=[[10.0, 10.0], [20.0, 20.0]]
        )
        # Post-init should detect multi agent mode
        assert config.mode == NavigatorMode.MULTI
    
    def test_get_single_agent_config(self):
        """Test extracting SingleAgentConfig from NavigatorConfig."""
        config = NavigatorConfig(
            mode=NavigatorMode.SINGLE,
            position=(50.0, 50.0),
            orientation=45.0,
            speed=1.0,
            max_speed=2.0,
            angular_velocity=0.1
        )
        
        single_config = config.get_single_agent_config()
        assert isinstance(single_config, SingleAgentConfig)
        assert single_config.position == (50.0, 50.0)
        assert single_config.orientation == 45.0
    
    def test_get_multi_agent_config(self):
        """Test extracting MultiAgentConfig from NavigatorConfig."""
        config = NavigatorConfig(
            mode=NavigatorMode.MULTI,
            positions=[[10.0, 10.0], [20.0, 20.0]],
            orientations=[0.0, 90.0],
            speeds=[1.0, 2.0],
            max_speeds=[3.0, 4.0],
            angular_velocities=[0.1, 0.2],
            num_agents=2
        )
        
        multi_config = config.get_multi_agent_config()
        assert isinstance(multi_config, MultiAgentConfig)
        assert multi_config.positions == [[10.0, 10.0], [20.0, 20.0]]
        assert multi_config.num_agents == 2
    
    def test_serialization_to_dict(self):
        """Test serialization of NavigatorConfig to dict with enum values."""
        config = NavigatorConfig(mode=NavigatorMode.SINGLE)
        config_dict = dataclass_to_dict(config)
        
        # Enum should be converted to string value
        assert config_dict['mode'] == 'single'
        assert isinstance(config_dict['mode'], str)
    
    def test_deserialization_from_dict(self):
        """Test deserialization of dict to NavigatorConfig with enum values."""
        config_dict = {'mode': 'single', 'position': (50.0, 50.0)}
        config = dict_to_dataclass(config_dict, NavigatorConfig)
        
        # String should be converted to enum
        assert config.mode == NavigatorMode.SINGLE
        assert isinstance(config.mode, NavigatorMode)


class TestSimulationConfig:
    """Tests for SimulationConfig with Enum support."""
    
    def test_create_with_enum_values(self):
        """Test creating SimulationConfig with enum values."""
        config = SimulationConfig(
            output_format=OutputFormat.JSON,
            log_level=LogLevel.DEBUG
        )
        assert config.output_format == OutputFormat.JSON
        assert config.log_level == LogLevel.DEBUG
    
    def test_create_with_string_values(self):
        """Test creating SimulationConfig with string values."""
        config = SimulationConfig(
            output_format='json',
            log_level='DEBUG'
        )
        assert config.output_format == OutputFormat.JSON
        assert isinstance(config.output_format, OutputFormat)
        assert config.log_level == LogLevel.DEBUG
        assert isinstance(config.log_level, LogLevel)
    
    def test_serialization_to_dict(self):
        """Test serialization of SimulationConfig to dict with enum values."""
        config = SimulationConfig(
            output_format=OutputFormat.JSON,
            log_level=LogLevel.DEBUG
        )
        config_dict = dataclass_to_dict(config)
        
        # Enums should be converted to string values
        assert config_dict['output_format'] == 'json'
        assert isinstance(config_dict['output_format'], str)
        assert config_dict['log_level'] == 'DEBUG'
        assert isinstance(config_dict['log_level'], str)


class TestFrameCacheConfig:
    """Tests for FrameCacheConfig with Enum support."""
    
    def test_create_with_enum_values(self):
        """Test creating FrameCacheConfig with enum values."""
        config = FrameCacheConfig(
            mode=CacheMode.LRU,
            eviction_policy=EvictionPolicy.FIFO,
            preload_strategy=PreloadStrategy.SEQUENTIAL
        )
        assert config.mode == CacheMode.LRU
        assert config.eviction_policy == EvictionPolicy.FIFO
        assert config.preload_strategy == PreloadStrategy.SEQUENTIAL
    
    def test_create_with_string_values(self):
        """Test creating FrameCacheConfig with string values."""
        config = FrameCacheConfig(
            mode='lru',
            eviction_policy='fifo',
            preload_strategy='sequential'
        )
        assert config.mode == CacheMode.LRU
        assert isinstance(config.mode, CacheMode)
        assert config.eviction_policy == EvictionPolicy.FIFO
        assert isinstance(config.eviction_policy, EvictionPolicy)
        assert config.preload_strategy == PreloadStrategy.SEQUENTIAL
        assert isinstance(config.preload_strategy, PreloadStrategy)
    
    def test_is_cache_enabled(self):
        """Test is_cache_enabled method with different modes."""
        config_none = FrameCacheConfig(mode=CacheMode.NONE)
        config_lru = FrameCacheConfig(mode=CacheMode.LRU)
        config_all = FrameCacheConfig(mode=CacheMode.ALL)
        
        assert not config_none.is_cache_enabled()
        assert config_lru.is_cache_enabled()
        assert config_all.is_cache_enabled()
    
    def test_get_memory_limit_bytes(self):
        """Test get_memory_limit_bytes method with different memory specifications."""
        # Test with MB value
        config_mb = FrameCacheConfig(memory_limit_mb="1024MB")
        assert config_mb.get_memory_limit_bytes() == 1024 * 1000 * 1000
        
        # Test with GiB value
        config_gib = FrameCacheConfig(memory_limit_mb="2GiB")
        assert config_gib.get_memory_limit_bytes() == 2 * 1024 * 1024 * 1024
        
        # Test with numeric value
        config_num = FrameCacheConfig(memory_limit_mb=512)
        assert config_num.get_memory_limit_bytes() == 512
        
        # Test with None value
        config_none = FrameCacheConfig(memory_limit_mb=None)
        assert config_none.get_memory_limit_bytes() is None
        
        # Test with environment variable
        config_env = FrameCacheConfig(memory_limit_mb="${oc.env:CACHE_SIZE,1GiB}")
        assert config_env.get_memory_limit_bytes() is None  # Can't resolve env vars in dataclass


class TestFactoryFunctions:
    """Tests for factory functions with Enum support."""
    
    def test_create_default_navigator_config(self):
        """Test create_default_navigator_config with different modes."""
        # Single mode
        single_config = create_default_navigator_config("single")
        assert single_config.mode == NavigatorMode.SINGLE
        assert isinstance(single_config.mode, NavigatorMode)
        
        # Multi mode
        multi_config = create_default_navigator_config("multi")
        assert multi_config.mode == NavigatorMode.MULTI
        assert isinstance(multi_config.mode, NavigatorMode)
        
        # Default mode
        default_config = create_default_navigator_config()
        assert isinstance(default_config.mode, NavigatorMode)
    
    def test_create_default_simulation_config(self):
        """Test create_default_simulation_config with Enum values."""
        config = create_default_simulation_config()
        assert isinstance(config.output_format, OutputFormat)
        assert isinstance(config.log_level, LogLevel)
    
    def test_create_default_frame_cache_config(self):
        """Test create_default_frame_cache_config with different modes."""
        # None mode
        none_config = create_default_frame_cache_config("none")
        assert none_config.mode == CacheMode.NONE
        assert isinstance(none_config.mode, CacheMode)
        
        # LRU mode
        lru_config = create_default_frame_cache_config("lru")
        assert lru_config.mode == CacheMode.LRU
        assert isinstance(lru_config.mode, CacheMode)
        assert lru_config.eviction_policy == EvictionPolicy.LRU
        
        # All mode
        all_config = create_default_frame_cache_config("all")
        assert all_config.mode == CacheMode.ALL
        assert isinstance(all_config.mode, CacheMode)
        assert all_config.preload_strategy == PreloadStrategy.SEQUENTIAL


class TestEnvironmentVariableSupport:
    """Tests for environment variable support with Enum types."""
    
    def test_validate_env_interpolation(self):
        """Test validate_env_interpolation function."""
        assert validate_env_interpolation("${oc.env:NAVIGATOR_MODE,single}")
        assert validate_env_interpolation("${oc.env:OUTPUT_FORMAT}")
        assert not validate_env_interpolation("regular_string")
    
    def test_resolve_env_value(self):
        """Test resolve_env_value function."""
        # Set test environment variable
        os.environ["TEST_MODE"] = "multi"
        
        # Test with existing env var
        assert resolve_env_value("${oc.env:TEST_MODE}") == "multi"
        
        # Test with default value
        assert resolve_env_value("${oc.env:MISSING_VAR,single}") == "single"
        
        # Test with regular string
        assert resolve_env_value("regular_string") == "regular_string"
        
        # Clean up
        del os.environ["TEST_MODE"]
    
    def test_navigator_config_with_env_vars(self):
        """Test NavigatorConfig with environment variable patterns."""
        config = NavigatorConfig(mode="${oc.env:NAVIGATOR_MODE,single}")
        
        # Environment variable pattern should be preserved
        assert config.mode == "${oc.env:NAVIGATOR_MODE,single}"
        
        # When serialized, should preserve the pattern
        config_dict = dataclass_to_dict(config)
        assert config_dict["mode"] == "${oc.env:NAVIGATOR_MODE,single}"


class TestBackwardCompatibility:
    """Tests for backward compatibility with string values."""
    
    def test_navigator_config_string_compatibility(self):
        """Test NavigatorConfig backward compatibility with string mode."""
        # Create with string
        config1 = NavigatorConfig(mode="single")
        assert config1.mode == NavigatorMode.SINGLE
        assert isinstance(config1.mode, NavigatorMode)
        
        # Serialize to dict (should convert to string)
        config_dict = dataclass_to_dict(config1)
        assert config_dict["mode"] == "single"
        assert isinstance(config_dict["mode"], str)
        
        # Deserialize back (should convert to enum)
        config2 = dict_to_dataclass(config_dict, NavigatorConfig)
        assert config2.mode == NavigatorMode.SINGLE
        assert isinstance(config2.mode, NavigatorMode)
    
    def test_simulation_config_string_compatibility(self):
        """Test SimulationConfig backward compatibility with string values."""
        # Create with strings
        config1 = SimulationConfig(
            output_format="json",
            log_level="DEBUG"
        )
        assert config1.output_format == OutputFormat.JSON
        assert isinstance(config1.output_format, OutputFormat)
        assert config1.log_level == LogLevel.DEBUG
        assert isinstance(config1.log_level, LogLevel)
        
        # Serialize to dict (should convert to strings)
        config_dict = dataclass_to_dict(config1)
        assert config_dict["output_format"] == "json"
        assert isinstance(config_dict["output_format"], str)
        assert config_dict["log_level"] == "DEBUG"
        assert isinstance(config_dict["log_level"], str)
        
        # Deserialize back (should convert to enums)
        config2 = dict_to_dataclass(config_dict, SimulationConfig)
        assert config2.output_format == OutputFormat.JSON
        assert isinstance(config2.output_format, OutputFormat)
        assert config2.log_level == LogLevel.DEBUG
        assert isinstance(config2.log_level, LogLevel)
    
    def test_frame_cache_config_string_compatibility(self):
        """Test FrameCacheConfig backward compatibility with string values."""
        # Create with strings
        config1 = FrameCacheConfig(
            mode="lru",
            eviction_policy="fifo",
            preload_strategy="sequential"
        )
        assert config1.mode == CacheMode.LRU
        assert isinstance(config1.mode, CacheMode)
        assert config1.eviction_policy == EvictionPolicy.FIFO
        assert isinstance(config1.eviction_policy, EvictionPolicy)
        assert config1.preload_strategy == PreloadStrategy.SEQUENTIAL
        assert isinstance(config1.preload_strategy, PreloadStrategy)
        
        # Serialize to dict (should convert to strings)
        config_dict = dataclass_to_dict(config1)
        assert config_dict["mode"] == "lru"
        assert isinstance(config_dict["mode"], str)
        assert config_dict["eviction_policy"] == "fifo"
        assert isinstance(config_dict["eviction_policy"], str)
        assert config_dict["preload_strategy"] == "sequential"
        assert isinstance(config_dict["preload_strategy"], str)
