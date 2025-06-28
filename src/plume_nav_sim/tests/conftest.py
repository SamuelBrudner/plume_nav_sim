"""
Comprehensive pytest configuration and fixture management for plume navigation simulation testing.

This module serves as the central testing infrastructure providing shared fixtures for Gymnasium API testing,
shim compatibility validation, CLI testing, Hydra configuration composition, database session management,
and comprehensive mocking strategies. The implementation supports pytest-hydra integration, Click CliRunner
setup, in-memory SQLite sessions, performance monitoring, and domain-specific mocks for scientific computing
workflows.

Enhanced Testing Features for Gymnasium Migration:
- Gymnasium 0.29.x API compliance fixtures for environment testing per Section 6.6.1.1
- gym_make() shim compatibility testing with automatic 4/5-tuple detection per Section 0.2.2
- Performance monitoring fixtures for sub-10ms step validation per Section 6.6.2.1
- Frame cache testing with LRU and memory pressure monitoring per Section 0.2.2
- pytest-hydra fixtures for hierarchical configuration testing per Section 6.6.1.1
- Click CliRunner integration for comprehensive CLI interface validation
- In-memory SQLAlchemy database session fixtures for database testing per Section 6.6.5.2
- Deterministic test behavior through controlled randomization per Section 6.6.8
- Comprehensive mock framework integration supporting unittest.mock patterns
- Configuration override and validation test fixtures per Section 6.6.1.1

Test Coverage Targets:
- Overall: >70% line coverage per Section 6.6.3.1
- Core environment modules: ≥90% coverage per enhanced testing standards
- Shim compatibility layer: ≥85% coverage per Section 6.6.3.1
- Configuration schemas: ≥80% coverage per Section 6.6.3.1
- Frame cache implementation: ≥85% coverage per Section 6.6.3.1
- Utility modules: ≥80% coverage per Section 6.6.3.1

Architecture Integration:
- Supports plume_nav_sim package structure conventions
- Integrates with Hydra configuration system for parameter management
- Provides ready-to-activate database testing without external dependencies
- Maintains backward compatibility with existing test patterns
- Enables comprehensive integration testing across CLI, configuration, and database domains
- Supports Gymnasium API compliance validation and legacy gym.make() shim testing

Usage Examples:

    # Basic fixture usage for Gymnasium environment testing
    def test_gymnasium_env_compliance(gymnasium_env_fixture, mock_seed_manager):
        env = gymnasium_env_fixture()
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        
        # Test 5-tuple step returns
        action = env.action_space.sample()
        step_result = env.step(action)
        assert len(step_result) == 5
        obs, reward, terminated, truncated, info = step_result

    # Shim compatibility testing
    def test_gym_make_shim_compatibility(gym_make_shim_fixture, mock_seed_manager):
        gym_make_mock, legacy_wrapper = gym_make_shim_fixture
        env = gym_make_mock("PlumeNavSim-v0")
        # Test automatic 4-tuple conversion for legacy callers
        assert hasattr(env, '_legacy_mode')

    # Performance monitoring for frame cache
    def test_frame_cache_performance(frame_cache_fixture, performance_monitor):
        cache = frame_cache_fixture("lru", memory_limit_mb=512)
        with performance_monitor["time_operation"]("cache_access") as timer:
            frame = cache.get_frame(0)
        assert timer.duration < 0.01  # < 10ms requirement
    
    # CLI testing with parameter validation
    def test_cli_command_execution(cli_runner_fixture):
        runner, env = cli_runner_fixture
        result = runner.invoke(main_command, ["--help"], env=env)
        assert result.exit_code == 0
    
    # Database session testing with isolation
    def test_database_operations(db_session_fixture):
        session = db_session_fixture
        if session:  # Graceful handling when database not enabled
            # Perform database operations
            pass
    
    # Comprehensive integration testing
    def test_full_system_integration(
        hydra_config_fixture, 
        cli_runner_fixture, 
        mock_video_plume, 
        db_session_fixture,
        gymnasium_env_fixture
    ):
        # Multi-domain integration test
        pass

Dependencies Integration:
- pytest ≥7.0.0: Enhanced fixture support and parameterized testing
- pytest-hydra ≥0.1.0: Specialized Hydra configuration testing plugin
- pytest-benchmark ≥4.0.0: Performance regression testing for sub-10ms validation
- click-testing ≥8.0.0: Click CLI testing utilities and CliRunner support
- SQLAlchemy ≥2.0: Database session management with in-memory testing
- unittest.mock: Standard library mocking framework for dependency isolation
- numpy.testing: Specialized numerical array assertions with tolerance support
- gymnasium ≥0.29.0: Modern RL environment framework for API testing
- psutil ≥5.9.0: Memory monitoring for frame cache testing

Configuration Management:
- Hierarchical configuration composition testing through pytest-hydra
- Environment variable interpolation validation with secure credential handling
- Configuration override scenario testing across development and production settings
- Schema validation testing with Pydantic integration and type safety verification
- Frame cache configuration testing with multiple modes (none, lru, all)

Authors: Generated by Blitzy Template Engine for plume_nav_sim
Version: 0.3.0
License: MIT
"""

import os
import sys
import time
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union, List, Tuple, Callable
from unittest.mock import Mock, MagicMock, patch
from contextlib import contextmanager

import pytest
import numpy as np

# Core testing framework imports
try:
    from click.testing import CliRunner
    CLICK_TESTING_AVAILABLE = True
except ImportError:
    CLICK_TESTING_AVAILABLE = False
    warnings.warn("Click testing not available. CLI tests will be skipped.", ImportWarning)

# Hydra testing integration
try:
    from hydra import compose, initialize_config_store
    from hydra.core.config_store import ConfigStore
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    warnings.warn("Hydra not available. Configuration testing will be limited.", ImportWarning)

# pytest-hydra plugin integration
try:
    import pytest_hydra
    PYTEST_HYDRA_AVAILABLE = True
except ImportError:
    PYTEST_HYDRA_AVAILABLE = False
    warnings.warn("pytest-hydra not available. Advanced configuration testing disabled.", ImportWarning)

# Database testing infrastructure
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    warnings.warn("SQLAlchemy not available. Database testing will be skipped.", ImportWarning)

# Gymnasium testing integration
try:
    import gymnasium
    from gymnasium.utils.env_checker import check_env
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    warnings.warn("Gymnasium not available. Environment testing will be limited.", ImportWarning)

# Performance monitoring for frame cache testing
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Memory monitoring tests will be skipped.", ImportWarning)

# Import project modules with graceful fallback
try:
    from plume_nav_sim.config.schemas import (
        NavigatorConfig, VideoPlumeConfig, SingleAgentConfig, MultiAgentConfig
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    warnings.warn("Project schemas not available. Some fixtures will be limited.", ImportWarning)

try:
    from plume_nav_sim.utils.seed_manager import SeedManager, SeedConfig
    SEED_MANAGER_AVAILABLE = True
except ImportError:
    SEED_MANAGER_AVAILABLE = False
    warnings.warn("Seed manager not available. Deterministic testing will be limited.", ImportWarning)

try:
    from plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv
    from plume_nav_sim.envs.spaces import create_action_space, create_observation_space
    ENV_AVAILABLE = True
except ImportError:
    ENV_AVAILABLE = False
    warnings.warn("Environment modules not available. Environment testing will be limited.", ImportWarning)

try:
    from plume_nav_sim.shims.gym_make import gym_make
    SHIMS_AVAILABLE = True
except ImportError:
    SHIMS_AVAILABLE = False
    warnings.warn("Shims not available. Compatibility testing will be limited.", ImportWarning)

try:
    from plume_nav_sim.utils.frame_cache import FrameCache, CacheMode, CacheStatistics
    FRAME_CACHE_AVAILABLE = True
except ImportError:
    FRAME_CACHE_AVAILABLE = False
    warnings.warn("Frame cache not available. Cache testing will be limited.", ImportWarning)

try:
    from plume_nav_sim.cli.main import cli as main_cli
    from plume_nav_sim.db.session import SessionManager, get_session
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False
    warnings.warn("CLI modules not available. CLI testing will be limited.", ImportWarning)


# ================================================================================================
# PYTEST CONFIGURATION AND SETUP
# ================================================================================================

def pytest_configure(config):
    """
    Configure pytest environment for enhanced scientific computing testing.
    
    Sets up markers, plugins, and environment variables required for comprehensive
    testing across CLI, configuration, database, and scientific computation domains,
    with specific support for Gymnasium API compliance and shim compatibility testing.
    """
    # Register custom markers for test categorization
    config.addinivalue_line(
        "markers", 
        "hydra: mark test as requiring Hydra configuration system"
    )
    config.addinivalue_line(
        "markers", 
        "cli: mark test as requiring CLI infrastructure"
    )
    config.addinivalue_line(
        "markers", 
        "database: mark test as requiring database session support"
    )
    config.addinivalue_line(
        "markers", 
        "integration: mark test as integration test spanning multiple components"
    )
    config.addinivalue_line(
        "markers", 
        "slow: mark test as slow running (>1s execution time)"
    )
    config.addinivalue_line(
        "markers", 
        "scientific: mark test as requiring scientific computing validation"
    )
    config.addinivalue_line(
        "markers", 
        "gymnasium: mark test as requiring Gymnasium API testing"
    )
    config.addinivalue_line(
        "markers", 
        "shim: mark test as requiring gym_make shim compatibility testing"
    )
    config.addinivalue_line(
        "markers", 
        "performance: mark test as requiring performance validation (<10ms steps)"
    )
    config.addinivalue_line(
        "markers", 
        "frame_cache: mark test as requiring frame cache functionality"
    )
    
    # Configure environment for headless testing
    os.environ.setdefault('MATPLOTLIB_BACKEND', 'Agg')
    os.environ.setdefault('PYTEST_RUNNING', 'true')
    os.environ.setdefault('HYDRA_DISABLE_LOGGING', 'true')
    os.environ.setdefault('FRAME_CACHE_MODE', 'none')  # Disable cache during testing
    
    # Disable user warnings during testing for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning, module="hydra")
    warnings.filterwarnings("ignore", category=UserWarning, module="omegaconf")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium")


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add automatic markers and skip conditions.
    
    Automatically applies markers based on test names and dependencies,
    and skips tests when required dependencies are not available.
    """
    for item in items:
        # Auto-mark tests based on naming patterns
        if "hydra" in item.name.lower() or "config" in item.name.lower():
            item.add_marker(pytest.mark.hydra)
        
        if "cli" in item.name.lower() or "command" in item.name.lower():
            item.add_marker(pytest.mark.cli)
        
        if "database" in item.name.lower() or "session" in item.name.lower():
            item.add_marker(pytest.mark.database)
        
        if "integration" in item.nodeid or "test_integration" in item.name:
            item.add_marker(pytest.mark.integration)
        
        if "gymnasium" in item.name.lower() or "env" in item.name.lower():
            item.add_marker(pytest.mark.gymnasium)
        
        if "shim" in item.name.lower() or "gym_make" in item.name.lower():
            item.add_marker(pytest.mark.shim)
        
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        if "cache" in item.name.lower() or "frame" in item.name.lower():
            item.add_marker(pytest.mark.frame_cache)
        
        # Skip tests based on dependency availability
        if item.get_closest_marker("hydra") and not HYDRA_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Hydra not available"))
        
        if item.get_closest_marker("cli") and not CLICK_TESTING_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Click testing not available"))
        
        if item.get_closest_marker("database") and not SQLALCHEMY_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="SQLAlchemy not available"))
        
        if item.get_closest_marker("gymnasium") and not GYMNASIUM_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Gymnasium not available"))
        
        if item.get_closest_marker("shim") and not SHIMS_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Shims not available"))
        
        if item.get_closest_marker("frame_cache") and not FRAME_CACHE_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Frame cache not available"))


# ================================================================================================
# HYDRA CONFIGURATION TESTING FIXTURES
# ================================================================================================

@pytest.fixture(scope="session")
def hydra_config_store():
    """
    Initialize and provide access to Hydra ConfigStore for configuration testing.
    
    This session-scoped fixture ensures proper ConfigStore initialization and cleanup
    for hierarchical configuration composition testing across all test modules.
    
    Returns:
        ConfigStore instance for configuration schema registration
        
    Example:
        def test_config_registration(hydra_config_store):
            # Access registered configuration schemas
            assert "navigator_config" in hydra_config_store.list()
    """
    if not HYDRA_AVAILABLE:
        pytest.skip("Hydra not available for configuration testing")
    
    # Clear any existing Hydra global state
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    # Initialize ConfigStore
    cs = ConfigStore.instance()
    
    # Register test configuration schemas if available
    if SCHEMAS_AVAILABLE:
        cs.store(name="test_navigator_config", node=NavigatorConfig)
        cs.store(name="test_video_plume_config", node=VideoPlumeConfig)
        cs.store(name="test_single_agent_config", node=SingleAgentConfig)
        cs.store(name="test_multi_agent_config", node=MultiAgentConfig)
    
    if SEED_MANAGER_AVAILABLE:
        cs.store(name="test_seed_config", node=SeedConfig)
    
    yield cs
    
    # Cleanup after session
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()


@pytest.fixture
def hydra_config_fixture(hydra_config_store, tmp_path):
    """
    Comprehensive Hydra configuration fixture for hierarchical composition testing.
    
    Provides a factory function for creating test configurations with override support,
    environment variable interpolation, and validation. Supports both simple parameter
    overrides and complex hierarchical configuration scenarios.
    
    Args:
        hydra_config_store: Session-scoped ConfigStore fixture
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Factory function for creating DictConfig objects with overrides
        
    Example:
        def test_navigator_configuration(hydra_config_fixture):
            # Basic configuration with overrides
            config = hydra_config_fixture({
                "navigator": {"max_speed": 2.0, "num_agents": 3},
                "simulation": {"fps": 60},
                "frame_cache": {"mode": "lru", "memory_limit_mb": 1024}
            })
            assert config.navigator.max_speed == 2.0
            assert config.frame_cache.mode == "lru"
    """
    if not HYDRA_AVAILABLE:
        pytest.skip("Hydra not available for configuration fixture")
    
    def _create_config(
        overrides: Optional[Dict[str, Any]] = None,
        config_name: str = "config",
        config_path: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None
    ) -> DictConfig:
        """
        Create test configuration with overrides and environment variable support.
        
        Args:
            overrides: Configuration parameter overrides
            config_name: Base configuration name
            config_path: Configuration path (uses tmp_path if None)
            env_vars: Environment variables to set during configuration
            
        Returns:
            Configured DictConfig object
        """
        # Set up temporary environment variables
        original_env = {}
        if env_vars:
            for key, value in env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
        
        try:
            # Clear any existing Hydra state
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()
            
            # Create base configuration with plume_nav_sim specific settings
            base_config = {
                "navigator": {
                    "position": [0.0, 0.0],
                    "orientation": 0.0,
                    "max_speed": 1.0,
                    "speed": 0.0
                },
                "video_plume": {
                    "video_path": str(tmp_path / "test_video.mp4"),
                    "flip": False,
                    "grayscale": True,
                    "kernel_size": 5,
                    "kernel_sigma": 1.0
                },
                "simulation": {
                    "num_steps": 100,
                    "fps": 30,
                    "max_duration": 10.0
                },
                "environment": {
                    "debug_mode": True,
                    "headless": True
                },
                "frame_cache": {
                    "mode": "none",  # Disable for testing
                    "memory_limit_mb": 512,
                    "enable_statistics": True
                },
                "seed": {
                    "enabled": True,
                    "global_seed": 42,
                    "numpy_seed": 42
                }
            }
            
            # Apply overrides
            if overrides:
                base_config = OmegaConf.merge(base_config, overrides)
            
            # Create DictConfig
            config = OmegaConf.create(base_config)
            
            # Validate configuration structure
            OmegaConf.set_struct(config, True)
            
            return config
            
        finally:
            # Restore original environment variables
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    return _create_config


# ================================================================================================
# GYMNASIUM ENVIRONMENT TESTING FIXTURES
# ================================================================================================

@pytest.fixture
def gymnasium_env_fixture(tmp_path):
    """
    Gymnasium environment fixture for API compliance testing.
    
    Provides factory function for creating properly configured Gymnasium environments
    with test-appropriate settings and comprehensive validation support.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Factory function for creating PlumeNavigationEnv instances
        
    Example:
        def test_gymnasium_compliance(gymnasium_env_fixture):
            env = gymnasium_env_fixture()
            
            # Test modern reset interface
            obs, info = env.reset(seed=42)
            assert isinstance(obs, np.ndarray)
            assert isinstance(info, dict)
            
            # Test 5-tuple step interface
            action = env.action_space.sample()
            step_result = env.step(action)
            assert len(step_result) == 5
            obs, reward, terminated, truncated, info = step_result
    """
    if not GYMNASIUM_AVAILABLE or not ENV_AVAILABLE:
        pytest.skip("Gymnasium or environment modules not available")
    
    def _create_env(config_overrides: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Create Gymnasium environment with test configuration.
        
        Args:
            config_overrides: Configuration parameter overrides
            **kwargs: Additional environment arguments
            
        Returns:
            Configured PlumeNavigationEnv instance
        """
        # Create test video file
        test_video = tmp_path / "test_video.mp4"
        test_video.write_bytes(b"MOCK_VIDEO_DATA")
        
        # Base test configuration
        base_config = {
            "video_plume": {
                "video_path": str(test_video),
                "flip": False,
                "grayscale": True,
                "kernel_size": 5,
                "kernel_sigma": 1.0
            },
            "navigator": {
                "position": [0.0, 0.0],
                "orientation": 0.0,
                "max_speed": 1.0
            },
            "simulation": {
                "num_steps": 10,  # Short for testing
                "fps": 30
            },
            "frame_cache": {
                "mode": "none",  # Disable for testing
                "memory_limit_mb": 256
            },
            "environment": {
                "debug_mode": True,
                "headless": True
            }
        }
        
        # Apply configuration overrides
        if config_overrides:
            from omegaconf import OmegaConf
            base_config = OmegaConf.merge(base_config, config_overrides)
        
        # Create environment
        env = PlumeNavigationEnv(config=base_config, **kwargs)
        
        return env
    
    return _create_env


@pytest.fixture
def gymnasium_spaces_fixture():
    """
    Gymnasium action and observation space testing fixture.
    
    Provides properly configured action and observation spaces for testing
    space compliance and validation logic.
    
    Returns:
        Dictionary containing action and observation space factories
        
    Example:
        def test_space_compliance(gymnasium_spaces_fixture):
            spaces = gymnasium_spaces_fixture
            action_space = spaces["action_space"]()
            obs_space = spaces["observation_space"]()
            
            # Test space properties
            assert action_space.contains(action_space.sample())
            assert obs_space.contains(obs_space.sample())
    """
    if not GYMNASIUM_AVAILABLE or not ENV_AVAILABLE:
        pytest.skip("Gymnasium or environment modules not available")
    
    def create_test_action_space():
        """Create test action space."""
        return create_action_space()
    
    def create_test_observation_space(obs_shape=(64, 64, 1)):
        """Create test observation space with configurable shape."""
        return create_observation_space(obs_shape=obs_shape)
    
    def validate_space_compliance(space, sample_count=100):
        """Validate space compliance with multiple samples."""
        for _ in range(sample_count):
            sample = space.sample()
            assert space.contains(sample), f"Space does not contain its own sample: {sample}"
        return True
    
    return {
        "action_space": create_test_action_space,
        "observation_space": create_test_observation_space,
        "validate_compliance": validate_space_compliance
    }


# ================================================================================================
# GYM MAKE SHIM COMPATIBILITY TESTING FIXTURES
# ================================================================================================

@pytest.fixture
def gym_make_shim_fixture():
    """
    gym_make() shim compatibility testing fixture.
    
    Provides mocked gym_make function and legacy wrapper for testing
    automatic API version detection and tuple format conversion.
    
    Returns:
        Tuple of (gym_make_mock, legacy_wrapper_class)
        
    Example:
        def test_shim_compatibility(gym_make_shim_fixture):
            gym_make_mock, legacy_wrapper = gym_make_shim_fixture
            
            # Test legacy caller detection
            env = gym_make_mock("PlumeNavSim-v0")
            
            # Test 4-tuple conversion for legacy callers
            if hasattr(env, '_legacy_mode'):
                obs = env.reset()  # Should return observation only
                step_result = env.step(env.action_space.sample())
                assert len(step_result) == 4  # Legacy 4-tuple
    """
    if not SHIMS_AVAILABLE:
        pytest.skip("Shims not available for compatibility testing")
    
    # Mock gymnasium environment for testing
    mock_gymnasium_env = Mock()
    mock_gymnasium_env.reset.return_value = (np.zeros((64, 64, 1)), {})
    mock_gymnasium_env.step.return_value = (
        np.zeros((64, 64, 1)),  # observation
        0.0,                    # reward
        False,                  # terminated
        False,                  # truncated
        {}                      # info
    )
    mock_gymnasium_env.action_space = Mock()
    mock_gymnasium_env.action_space.sample.return_value = np.array([0.0, 0.0])
    mock_gymnasium_env.observation_space = Mock()
    
    class LegacyWrapper:
        """Mock legacy wrapper that converts 5-tuple to 4-tuple."""
        
        def __init__(self, env):
            self.env = env
            self._legacy_mode = True
        
        def reset(self, **kwargs):
            """Legacy reset returning observation only."""
            obs, info = self.env.reset(**kwargs)
            return obs
        
        def step(self, action):
            """Legacy step returning 4-tuple."""
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Convert to legacy 4-tuple (done = terminated OR truncated)
            done = terminated or truncated
            return obs, reward, done, info
        
        def __getattr__(self, name):
            """Delegate to wrapped environment."""
            return getattr(self.env, name)
    
    # Mock gym_make function
    def mock_gym_make(env_id: str, **kwargs):
        """Mock gym_make with deprecation warning and legacy wrapper."""
        warnings.warn(
            "Using gym_make is deprecated and will be removed in v1.0. "
            "Please update to: gymnasium.make('PlumeNavSim-v0')",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Detect if caller expects legacy API (simplified detection)
        import inspect
        frame = inspect.currentframe()
        try:
            # Check if calling code looks like legacy pattern
            caller_locals = frame.f_back.f_locals
            legacy_caller = any(
                "gym" in str(caller_locals.get(var, "")).lower() 
                for var in caller_locals
            )
        except:
            legacy_caller = False
        finally:
            del frame
        
        if legacy_caller or kwargs.get('legacy_mode', False):
            return LegacyWrapper(mock_gymnasium_env)
        
        return mock_gymnasium_env
    
    return mock_gym_make, LegacyWrapper


@pytest.fixture
def deprecation_warning_fixture():
    """
    Deprecation warning capture fixture for shim testing.
    
    Provides utilities for capturing and validating deprecation warnings
    emitted by the gym_make shim layer.
    
    Returns:
        Dictionary containing warning capture utilities
        
    Example:
        def test_deprecation_warnings(deprecation_warning_fixture):
            warning_catcher = deprecation_warning_fixture["capture_warnings"]
            
            with warning_catcher() as warnings_list:
                # Code that should emit deprecation warning
                pass
            
            assert len(warnings_list) > 0
            assert "deprecated" in str(warnings_list[0].message).lower()
    """
    @contextmanager
    def capture_warnings():
        """Capture deprecation warnings."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always", DeprecationWarning)
            yield warning_list
    
    def validate_deprecation_warning(warning_list, expected_message_parts=None):
        """Validate deprecation warning content."""
        if expected_message_parts is None:
            expected_message_parts = ["deprecated", "gymnasium"]
        
        deprecation_warnings = [
            w for w in warning_list 
            if issubclass(w.category, DeprecationWarning)
        ]
        
        assert len(deprecation_warnings) > 0, "No deprecation warnings found"
        
        warning_message = str(deprecation_warnings[0].message).lower()
        for expected_part in expected_message_parts:
            assert expected_part.lower() in warning_message, \
                f"Expected '{expected_part}' in warning message: {warning_message}"
        
        return True
    
    return {
        "capture_warnings": capture_warnings,
        "validate_warning": validate_deprecation_warning
    }


# ================================================================================================
# FRAME CACHE TESTING FIXTURES
# ================================================================================================

@pytest.fixture
def frame_cache_fixture(tmp_path):
    """
    Frame cache testing fixture with configurable modes and memory monitoring.
    
    Provides factory function for creating frame cache instances with different
    configurations for testing LRU behavior, memory pressure, and performance.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Factory function for creating FrameCache instances
        
    Example:
        def test_frame_cache_lru_behavior(frame_cache_fixture):
            cache = frame_cache_fixture("lru", memory_limit_mb=256)
            
            # Test cache behavior
            frame = cache.get_frame(0)
            assert cache.statistics.hit_count >= 0
            assert cache.statistics.miss_count >= 0
    """
    if not FRAME_CACHE_AVAILABLE:
        pytest.skip("Frame cache not available for testing")
    
    def _create_cache(
        mode: str = "lru",
        memory_limit_mb: int = 512,
        frame_count: int = 100,
        frame_shape: Tuple[int, int, int] = (64, 64, 1)
    ):
        """
        Create frame cache with test configuration.
        
        Args:
            mode: Cache mode ("none", "lru", "all")
            memory_limit_mb: Memory limit in megabytes
            frame_count: Number of test frames to generate
            frame_shape: Shape of test frames (height, width, channels)
            
        Returns:
            Configured FrameCache instance
        """
        # Create test video file
        test_video = tmp_path / f"test_video_{mode}.mp4"
        test_video.write_bytes(b"MOCK_VIDEO_DATA")
        
        # Create cache configuration
        cache_config = {
            "mode": mode,
            "memory_limit_mb": memory_limit_mb,
            "enable_statistics": True,
            "pressure_threshold": 0.9
        }
        
        # Create mock video source
        class MockVideoSource:
            def __init__(self, frame_count, frame_shape):
                self.frame_count = frame_count
                self.frame_shape = frame_shape
            
            def get_frame(self, index):
                if 0 <= index < self.frame_count:
                    # Generate deterministic test frame
                    frame = np.zeros(self.frame_shape, dtype=np.uint8)
                    frame.fill(index % 256)  # Unique pattern per frame
                    return frame
                return None
            
            def get_frame_count(self):
                return self.frame_count
        
        video_source = MockVideoSource(frame_count, frame_shape)
        
        # Create cache instance
        cache_mode = CacheMode.from_string(mode)
        cache = FrameCache.create_cache(cache_mode, video_source, cache_config)
        
        return cache
    
    return _create_cache


@pytest.fixture
def memory_monitor_fixture():
    """
    Memory monitoring fixture for frame cache testing.
    
    Provides utilities for monitoring memory usage during cache operations
    and validating memory pressure thresholds.
    
    Returns:
        Dictionary containing memory monitoring utilities
        
    Example:
        def test_memory_pressure(memory_monitor_fixture):
            monitor = memory_monitor_fixture
            
            with monitor["track_memory"]() as tracker:
                # Operations that consume memory
                pass
            
            assert tracker.peak_memory_mb < 2048  # Memory limit
    """
    if not PSUTIL_AVAILABLE:
        pytest.skip("psutil not available for memory monitoring")
    
    @contextmanager
    def track_memory():
        """Track memory usage during test execution."""
        class MemoryTracker:
            def __init__(self):
                self.process = psutil.Process()
                self.start_memory_mb = self.process.memory_info().rss / (1024 * 1024)
                self.peak_memory_mb = self.start_memory_mb
            
            def update_peak(self):
                current_memory_mb = self.process.memory_info().rss / (1024 * 1024)
                self.peak_memory_mb = max(self.peak_memory_mb, current_memory_mb)
                return current_memory_mb
        
        tracker = MemoryTracker()
        
        try:
            yield tracker
        finally:
            tracker.update_peak()
    
    def validate_memory_limit(limit_mb: float):
        """Validate current memory usage against limit."""
        current_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        return current_memory_mb <= limit_mb
    
    def get_memory_pressure():
        """Get current memory pressure as percentage of system memory."""
        memory_info = psutil.virtual_memory()
        return memory_info.percent / 100.0
    
    return {
        "track_memory": track_memory,
        "validate_limit": validate_memory_limit,
        "get_pressure": get_memory_pressure
    }


# ================================================================================================
# PERFORMANCE MONITORING FIXTURES
# ================================================================================================

@pytest.fixture
def performance_monitor():
    """
    Performance monitoring fixture for validating timing requirements.
    
    Provides utilities for measuring and validating performance against specified
    SLA requirements including environment step latency, frame cache performance,
    and memory efficiency.
    
    Returns:
        Dictionary containing performance monitoring utilities
        
    Example:
        def test_step_performance(performance_monitor):
            monitor = performance_monitor
            
            with monitor["time_operation"]("env_step") as timer:
                # Environment step operation
                pass
            
            assert timer.duration < 0.01  # < 10ms requirement
    """
    
    @contextmanager
    def time_operation(operation_name: str):
        """Context manager for timing operations."""
        class Timer:
            def __init__(self, name):
                self.name = name
                self.start_time = None
                self.end_time = None
                self.duration = None
        
        timer = Timer(operation_name)
        timer.start_time = time.perf_counter()
        
        try:
            yield timer
        finally:
            timer.end_time = time.perf_counter()
            timer.duration = timer.end_time - timer.start_time
    
    def validate_performance_requirements(timings: Dict[str, float]) -> Dict[str, bool]:
        """Validate timing against performance requirements."""
        requirements = {
            'env_step': 0.01,              # Environment step < 10ms
            'env_reset': 0.05,             # Environment reset < 50ms
            'frame_access': 0.01,          # Frame access < 10ms
            'cache_lookup': 0.001,         # Cache lookup < 1ms
            'config_loading': 0.5,         # Configuration loading < 500ms
            'cli_init': 2.0,               # CLI initialization < 2s
        }
        
        results = {}
        for operation, duration in timings.items():
            if operation in requirements:
                results[operation] = duration <= requirements[operation]
            else:
                results[operation] = True  # Unknown operations pass
        
        return results
    
    def create_performance_report(timings: Dict[str, float]) -> str:
        """Generate performance report string."""
        validation_results = validate_performance_requirements(timings)
        
        report_lines = ["Performance Report:"]
        report_lines.append("=" * 50)
        
        for operation, duration in timings.items():
            status = "✓ PASS" if validation_results.get(operation, True) else "✗ FAIL"
            report_lines.append(f"{operation:<30} {duration*1000:>8.2f}ms {status}")
        
        overall_pass = all(validation_results.values())
        status_line = "Overall: PASS" if overall_pass else "Overall: FAIL"
        report_lines.append("=" * 50)
        report_lines.append(status_line)
        
        return "\n".join(report_lines)
    
    @contextmanager
    def benchmark_environment_performance(env, num_steps=100):
        """Benchmark environment performance over multiple steps."""
        step_times = []
        reset_time = None
        
        # Measure reset time
        start_time = time.perf_counter()
        env.reset(seed=42)
        reset_time = time.perf_counter() - start_time
        
        # Measure step times
        for _ in range(num_steps):
            action = env.action_space.sample()
            start_time = time.perf_counter()
            env.step(action)
            step_times.append(time.perf_counter() - start_time)
        
        class BenchmarkResults:
            def __init__(self):
                self.reset_time = reset_time
                self.step_times = step_times
                self.mean_step_time = np.mean(step_times)
                self.p95_step_time = np.percentile(step_times, 95)
                self.p99_step_time = np.percentile(step_times, 99)
                self.max_step_time = np.max(step_times)
                
            def meets_requirements(self):
                """Check if performance meets requirements."""
                return (
                    self.p95_step_time < 0.01 and  # 95th percentile < 10ms
                    self.reset_time < 0.05         # Reset < 50ms
                )
        
        yield BenchmarkResults()
    
    return {
        'time_operation': time_operation,
        'validate_requirements': validate_performance_requirements,
        'create_report': create_performance_report,
        'benchmark_environment': benchmark_environment_performance
    }


# ================================================================================================
# CLI TESTING FIXTURES
# ================================================================================================

@pytest.fixture
def cli_runner_fixture():
    """
    Comprehensive CLI testing fixture using Click CliRunner.
    
    Provides isolated CliRunner instance with proper environment setup for testing
    command-line interface interactions, parameter validation, and error handling.
    Includes temporary directory management and environment variable isolation.
    
    Returns:
        Tuple of (CliRunner, environment_dict) for CLI testing
        
    Example:
        def test_cli_help_command(cli_runner_fixture):
            runner, env = cli_runner_fixture
            result = runner.invoke(main_cli, ["--help"], env=env)
            assert result.exit_code == 0
            assert "Usage:" in result.output
    """
    if not CLICK_TESTING_AVAILABLE:
        pytest.skip("Click testing not available for CLI fixture")
    
    # Create isolated CliRunner
    runner = CliRunner(mix_stderr=False)
    
    # Set up isolated environment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test environment with isolated paths
        test_env = {
            'HYDRA_WORKING_DIR': str(temp_path / 'hydra'),
            'PYTEST_RUNNING': 'true',
            'MATPLOTLIB_BACKEND': 'Agg',
            'LOG_LEVEL': 'ERROR',  # Reduce logging noise in tests
            'HYDRA_DISABLE_LOGGING': 'true',
            'FRAME_CACHE_MODE': 'none',  # Disable cache for CLI tests
            'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
            'PATH': os.environ.get('PATH', ''),
        }
        
        # Create necessary directories
        (temp_path / 'hydra').mkdir(parents=True, exist_ok=True)
        (temp_path / 'outputs').mkdir(parents=True, exist_ok=True)
        (temp_path / 'config').mkdir(parents=True, exist_ok=True)
        
        # Create minimal configuration files for CLI testing
        config_dir = temp_path / 'config'
        base_config = {
            'navigator': {'max_speed': 1.0, 'position': [0.0, 0.0]},
            'simulation': {'num_steps': 10, 'fps': 30},
            'environment': {'debug_mode': True, 'headless': True},
            'frame_cache': {'mode': 'none', 'memory_limit_mb': 256}
        }
        
        with open(config_dir / 'config.yaml', 'w') as f:
            import yaml
            yaml.dump(base_config, f)
        
        yield runner, test_env


# ================================================================================================
# DATABASE SESSION TESTING FIXTURES
# ================================================================================================

@pytest.fixture
def db_session_fixture():
    """
    In-memory SQLAlchemy database session fixture for isolated testing.
    
    Provides completely isolated database sessions using in-memory SQLite for testing
    database operations without external dependencies. Sessions are automatically
    cleaned up after test completion with proper transaction rollback.
    
    Returns:
        SQLAlchemy session object or None if database features disabled
        
    Example:
        def test_database_operations(db_session_fixture):
            session = db_session_fixture
            if session:  # Graceful handling when database not enabled
                # Perform database operations
                session.execute(text("SELECT 1"))
                # Automatic cleanup handled by fixture
            else:
                # Test file-based operations instead
                pass
    """
    if not SQLALCHEMY_AVAILABLE:
        yield None
        return
    
    # Create in-memory SQLite database
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,  # Disable SQL logging in tests
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
            "isolation_level": None  # Autocommit mode for testing
        }
    )
    
    # Create session factory
    SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=True,
        expire_on_commit=False  # Keep objects accessible after commit
    )
    
    # Create session
    session = SessionLocal()
    
    try:
        # Validate connection
        session.execute(text("SELECT 1"))
        
        yield session
        
    except Exception as e:
        # Rollback on any error
        session.rollback()
        raise e
        
    finally:
        # Cleanup session and engine
        session.close()
        engine.dispose()


# ================================================================================================
# SCIENTIFIC COMPUTING AND MOCKING FIXTURES
# ================================================================================================

@pytest.fixture
def mock_seed_manager():
    """
    Mock seed manager fixture for deterministic testing behavior.
    
    Provides controlled randomization for reproducible test execution across
    utilities and scientific computing components. Ensures test determinism
    while validating seed management functionality.
    
    Returns:
        Mock SeedManager instance with controlled behavior
        
    Example:
        def test_reproducible_computation(mock_seed_manager):
            # Deterministic test execution
            seed_manager = mock_seed_manager
            assert seed_manager.current_seed == 42
            
            # Test seed management functionality
            state = seed_manager.get_state()
            assert state is not None
    """
    if SEED_MANAGER_AVAILABLE:
        # Use real SeedManager for better testing
        from plume_nav_sim.utils.seed_manager import SeedManager
        manager = SeedManager(enabled=True, global_seed=42)
        manager.initialize()
        return manager
    else:
        # Fallback to mock
        mock_manager = Mock(spec=SeedManager)
        
        # Configure mock behavior
        mock_manager.current_seed = 42
        mock_manager.run_id = "test_run_001"
        mock_manager.environment_hash = "test_env_hash"
        mock_manager.enabled = True
        
        # Mock numpy generator
        mock_generator = Mock()
        mock_generator.random.return_value = 0.5
        mock_generator.integers.return_value = np.array([1, 2, 3, 4, 5])
        mock_manager.numpy_generator = mock_generator
        
        # Mock state management
        mock_state = {
            'python_state': ('dummy_state',),
            'numpy_legacy_state': ('dummy_numpy_state',),
            'numpy_generator_state': {'dummy': 'state'},
            'seed': 42,
            'timestamp': time.time()
        }
        mock_manager.get_state.return_value = mock_state
        mock_manager.restore_state.return_value = None
        
        # Mock initialization
        mock_manager.initialize.return_value = 42
        
        # Mock validation
        mock_manager.validate_reproducibility.return_value = True
        
        # Mock temporary seed context
        @contextmanager
        def mock_temporary_seed(seed):
            original_seed = mock_manager.current_seed
            mock_manager.current_seed = seed
            try:
                yield seed
            finally:
                mock_manager.current_seed = original_seed
        
        mock_manager.temporary_seed = mock_temporary_seed
        
        # Mock experiment seed generation
        mock_manager.generate_experiment_seeds.return_value = [42, 43, 44, 45, 46]
        
        return mock_manager


@pytest.fixture
def mock_video_plume(tmp_path):
    """
    Comprehensive VideoPlume mock fixture for environment testing.
    
    Provides complete VideoPlume mock with realistic metadata, frame processing,
    and configuration integration for isolated environment testing.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        Mock VideoPlume instance with comprehensive behavior
        
    Example:
        def test_plume_environment(mock_video_plume):
            plume = mock_video_plume
            assert plume.frame_count == 300
            assert plume.width == 640
            assert plume.height == 480
            
            frame = plume.get_frame(0)
            assert frame is not None
            assert isinstance(frame, np.ndarray)
    """
    mock_plume = Mock()
    
    # Configure basic properties
    mock_plume.frame_count = 300
    mock_plume.width = 640
    mock_plume.height = 480
    mock_plume.fps = 30.0
    mock_plume.video_path = tmp_path / "test_video.mp4"
    
    # Configuration properties
    mock_plume.flip = False
    mock_plume.grayscale = True
    mock_plume.kernel_size = 5
    mock_plume.kernel_sigma = 1.0
    mock_plume.threshold = None
    mock_plume.normalize = True
    
    def mock_get_frame(frame_index):
        """Mock get_frame method with preprocessing simulation."""
        if 0 <= frame_index < mock_plume.frame_count:
            # Generate deterministic test frame
            shape = (mock_plume.height, mock_plume.width, 1 if mock_plume.grayscale else 3)
            frame = np.zeros(shape, dtype=np.uint8)
            
            # Add unique pattern per frame
            frame.fill((frame_index * 17) % 256)
            
            if mock_plume.normalize:
                frame = frame.astype(np.float32) / 255.0
            
            return frame
        
        return None
    
    mock_plume.get_frame = mock_get_frame
    
    # Mock metadata methods
    def mock_get_metadata():
        """Mock metadata extraction."""
        return {
            'frame_count': mock_plume.frame_count,
            'width': mock_plume.width,
            'height': mock_plume.height,
            'fps': mock_plume.fps,
            'duration': mock_plume.frame_count / mock_plume.fps,
            'video_path': str(mock_plume.video_path),
            'preprocessing': {
                'flip': mock_plume.flip,
                'grayscale': mock_plume.grayscale,
                'kernel_size': mock_plume.kernel_size,
                'kernel_sigma': mock_plume.kernel_sigma,
                'normalize': mock_plume.normalize
            }
        }
    
    mock_plume.get_metadata = mock_get_metadata
    
    # Mock context manager methods
    mock_plume.__enter__ = lambda self: self
    mock_plume.__exit__ = lambda self, *args: None
    
    # Mock close method
    mock_plume.close = Mock()
    
    return mock_plume


# ================================================================================================
# COMPREHENSIVE FIXTURE COMBINATIONS
# ================================================================================================

@pytest.fixture
def comprehensive_test_setup(
    hydra_config_fixture,
    cli_runner_fixture, 
    db_session_fixture,
    mock_seed_manager,
    mock_video_plume,
    performance_monitor,
    gymnasium_env_fixture
):
    """
    Comprehensive test setup fixture combining all testing domains.
    
    Provides unified access to all testing fixtures for comprehensive integration
    testing across CLI, configuration, database, Gymnasium environments,
    and scientific computing domains.
    
    Args:
        hydra_config_fixture: Hydra configuration testing fixture
        cli_runner_fixture: CLI testing infrastructure
        db_session_fixture: Database session testing
        mock_seed_manager: Seed management mocking
        mock_video_plume: Video environment mocking
        performance_monitor: Performance monitoring utilities
        gymnasium_env_fixture: Gymnasium environment testing
        
    Returns:
        Dictionary containing all test infrastructure components
        
    Example:
        def test_comprehensive_system(comprehensive_test_setup):
            setup = comprehensive_test_setup
            
            # Configuration testing
            config = setup["config_factory"]({"navigator": {"max_speed": 3.0}})
            
            # Environment testing
            env = setup["env_factory"]()
            obs, info = env.reset(seed=42)
            
            # Performance validation
            with setup["performance"]["time_operation"]("full_test") as timer:
                # Comprehensive test operations
                pass
    """
    cli_runner, cli_env = cli_runner_fixture
    
    def create_integrated_config(overrides: Optional[Dict[str, Any]] = None):
        """Create configuration integrated with test environment."""
        base_overrides = {
            'environment': {
                'debug_mode': True,
                'headless': True
            },
            'frame_cache': {
                'mode': 'none',  # Disable for testing
                'memory_limit_mb': 256
            },
            'seed': {
                'enabled': True,
                'global_seed': 42
            }
        }
        
        if overrides:
            base_overrides = {**base_overrides, **overrides}
        
        return hydra_config_fixture(base_overrides)
    
    def execute_cli_with_config(command: List[str], config_overrides: Optional[Dict[str, Any]] = None):
        """Execute CLI command with configuration integration."""
        # Create configuration if overrides provided
        if config_overrides:
            config = create_integrated_config(config_overrides)
            # Add config overrides to CLI command
            for key, value in config_overrides.items():
                command.append(f"{key}={value}")
        
        return cli_runner.invoke(main_cli if CLI_AVAILABLE else Mock(), command, env=cli_env)
    
    @contextmanager
    def managed_db_session():
        """Context manager for database session with graceful fallback."""
        if db_session_fixture:
            yield db_session_fixture
        else:
            yield None
    
    return {
        # Configuration testing
        'config_factory': create_integrated_config,
        'hydra_config': hydra_config_fixture,
        
        # Environment testing
        'env_factory': gymnasium_env_fixture,
        
        # CLI testing
        'cli_runner': cli_runner,
        'cli_env': cli_env,
        'cli_execute': execute_cli_with_config,
        
        # Database testing
        'db_session': managed_db_session,
        'db_session_raw': db_session_fixture,
        
        # Mocking infrastructure
        'seed_manager': mock_seed_manager,
        'video_plume': mock_video_plume,
        
        # Performance monitoring
        'performance': performance_monitor,
        
        # Integration utilities
        'create_test_environment': lambda: {
            'config': create_integrated_config(),
            'env': gymnasium_env_fixture(),
            'cli_runner': cli_runner,
            'db_session': db_session_fixture
        }
    }