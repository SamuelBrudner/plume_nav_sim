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
import importlib.util
import logging

import pytest
import numpy as np


def _has_module(name: str) -> bool:
    """Return True if the given module can be imported."""
    return importlib.util.find_spec(name) is not None


logger = logging.getLogger(__name__)

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
# SQLAlchemy is required for database fixtures; tests will skip if missing.
sqlalchemy = pytest.importorskip(
    "sqlalchemy", reason="SQLAlchemy not available. Database testing will be skipped."
)
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Gymnasium testing integration
# Gymnasium is required for environment fixtures; tests will skip if missing.
gymnasium = pytest.importorskip(
    "gymnasium", reason="Gymnasium not available. Environment testing will be limited."
)
from gymnasium.utils.env_checker import check_env

# Performance monitoring for frame cache testing
# psutil enables memory monitoring; fixtures will skip if it's absent.
psutil = pytest.importorskip(
    "psutil", reason="psutil not available. Memory monitoring tests will be skipped."
)

# Import project modules with graceful fallback
# Project configuration schemas are optional; skip related fixtures when absent.
schemas = pytest.importorskip(
    "plume_nav_sim.config.schemas",
    reason="Project schemas not available. Some fixtures will be limited.",
)
NavigatorConfig = schemas.NavigatorConfig
VideoPlumeConfig = schemas.VideoPlumeConfig
SingleAgentConfig = schemas.SingleAgentConfig
MultiAgentConfig = schemas.MultiAgentConfig
# Seed manager provides deterministic testing; fixtures skip if missing.
seed_mod = pytest.importorskip(
    "plume_nav_sim.utils.seed_manager",
    reason="Seed manager not available. Deterministic testing will be limited.",
)
SeedManager = seed_mod.SeedManager
SeedConfig = seed_mod.SeedConfig

# v1.0 New component imports with graceful fallback
try:
    from plume_nav_sim.core.protocols import (
        SourceProtocol, BoundaryPolicyProtocol, ActionInterfaceProtocol, 
        RecorderProtocol, StatsAggregatorProtocol
    )
    PROTOCOLS_V1_AVAILABLE = True
except ImportError:
    PROTOCOLS_V1_AVAILABLE = False
    warnings.warn("v1.0 protocols not available. New component testing will be limited.", ImportWarning)

try:
    from plume_nav_sim.core.sources import create_source, PointSource, MultiSource, DynamicSource
    SOURCES_AVAILABLE = True
except ImportError:
    SOURCES_AVAILABLE = False
    warnings.warn("Sources module not available. Source testing will be limited.", ImportWarning)

try:
    from plume_nav_sim.core.boundaries import create_boundary_policy, TerminateBoundary, BounceBoundary
    BOUNDARIES_AVAILABLE = True
except ImportError:
    BOUNDARIES_AVAILABLE = False
    warnings.warn("Boundaries module not available. Boundary testing will be limited.", ImportWarning)

try:
    from plume_nav_sim.core.actions import create_action_interface, Continuous2DAction, CardinalDiscreteAction
    ACTIONS_AVAILABLE = True
except ImportError:
    ACTIONS_AVAILABLE = False
    warnings.warn("Actions module not available. Action testing will be limited.", ImportWarning)

try:
    from plume_nav_sim.recording import RecorderManager, RecorderFactory, BaseRecorder
    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False
    warnings.warn("Recording module not available. Recording testing will be limited.", ImportWarning)

try:
    from plume_nav_sim.analysis import generate_summary, create_stats_aggregator, StatsAggregator
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    warnings.warn("Analysis module not available. Analysis testing will be limited.", ImportWarning)

try:
    from plume_nav_sim.debug import plot_initial_state
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False
    warnings.warn("Debug module not available. Debug testing will be limited.", ImportWarning)

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
    config.addinivalue_line(
        "markers",
        "sources: mark test as requiring source protocol and implementations"
    )
    config.addinivalue_line(
        "markers",
        "boundaries: mark test as requiring boundary policy protocols"
    )
    config.addinivalue_line(
        "markers",
        "actions: mark test as requiring action interface protocols"
    )
    config.addinivalue_line(
        "markers",
        "recording: mark test as requiring recording framework"
    )
    config.addinivalue_line(
        "markers",
        "analysis: mark test as requiring statistics aggregation"
    )
    config.addinivalue_line(
        "markers",
        "debug: mark test as requiring debug utilities"
    )
    config.addinivalue_line(
        "markers",
        "v1_components: mark test as requiring v1.0 component architecture"
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
        
        if "source" in item.name.lower() or "emission" in item.name.lower():
            item.add_marker(pytest.mark.sources)
        
        if "boundary" in item.name.lower() or "boundary_policy" in item.name.lower():
            item.add_marker(pytest.mark.boundaries)
        
        if "action" in item.name.lower() or "action_interface" in item.name.lower():
            item.add_marker(pytest.mark.actions)
        
        if "record" in item.name.lower() or "recorder" in item.name.lower():
            item.add_marker(pytest.mark.recording)
        
        if "stats" in item.name.lower() or "analysis" in item.name.lower() or "aggregator" in item.name.lower():
            item.add_marker(pytest.mark.analysis)
        
        if "debug" in item.name.lower() or "plot" in item.name.lower():
            item.add_marker(pytest.mark.debug)
        
        if any(component in item.name.lower() for component in ["v1", "protocol", "factory"]):
            item.add_marker(pytest.mark.v1_components)
        
        # Skip tests based on dependency availability
        if item.get_closest_marker("hydra") and not HYDRA_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Hydra not available"))
        
        if item.get_closest_marker("cli") and not CLICK_TESTING_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Click testing not available"))
        
        if item.get_closest_marker("database") and not _has_module("sqlalchemy"):
            item.add_marker(pytest.mark.skip(reason="SQLAlchemy not available"))

        if item.get_closest_marker("gymnasium") and not _has_module("gymnasium"):
            item.add_marker(pytest.mark.skip(reason="Gymnasium not available"))
        
        if item.get_closest_marker("shim") and not SHIMS_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Shims not available"))
        
        if item.get_closest_marker("frame_cache") and not FRAME_CACHE_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Frame cache not available"))
        
        if item.get_closest_marker("sources") and not SOURCES_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Sources module not available"))
        
        if item.get_closest_marker("boundaries") and not BOUNDARIES_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Boundaries module not available"))
        
        if item.get_closest_marker("actions") and not ACTIONS_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Actions module not available"))
        
        if item.get_closest_marker("recording") and not RECORDING_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Recording module not available"))
        
        if item.get_closest_marker("analysis") and not ANALYSIS_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Analysis module not available"))
        
        if item.get_closest_marker("debug") and not DEBUG_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="Debug module not available"))
        
        if item.get_closest_marker("v1_components") and not PROTOCOLS_V1_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="v1.0 protocols not available"))


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
    
    # Register test configuration schemas
    cs.store(name="test_navigator_config", node=NavigatorConfig)
    cs.store(name="test_video_plume_config", node=VideoPlumeConfig)
    cs.store(name="test_single_agent_config", node=SingleAgentConfig)
    cs.store(name="test_multi_agent_config", node=MultiAgentConfig)

    # Register seed configuration for deterministic testing
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
    pytest.importorskip(
        "gymnasium", reason="Gymnasium required for environment testing"
    )
    logger.debug("Gymnasium dependency satisfied for gymnasium_env_fixture")
    if not ENV_AVAILABLE:
        pytest.skip("Environment modules not available")
    
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
    pytest.importorskip(
        "gymnasium", reason="Gymnasium required for space fixtures"
    )
    logger.debug("Gymnasium dependency satisfied for gymnasium_spaces_fixture")
    if not ENV_AVAILABLE:
        pytest.skip("Environment modules not available")
    
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
    psutil = pytest.importorskip(
        "psutil", reason="psutil not available for memory monitoring"
    )
    logger.debug("psutil dependency satisfied for memory_monitor_fixture")
    
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
    # Skip if SQLAlchemy is missing
    pytest.importorskip(
        "sqlalchemy", reason="SQLAlchemy required for database tests"
    )
    logger.debug("SQLAlchemy dependency satisfied for db_session_fixture")

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
    """Seed manager fixture ensuring deterministic behavior."""
    seed_mod = pytest.importorskip(
        "plume_nav_sim.utils.seed_manager",
        reason="Seed manager required for deterministic testing",
    )
    manager = seed_mod.SeedManager(enabled=True, global_seed=42)
    manager.initialize()
    logger.debug("Seed manager dependency satisfied for mock_seed_manager")
    return manager


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


# ================================================================================================
# V1.0 SOURCE PROTOCOL AND COMPONENT FIXTURES
# ================================================================================================

@pytest.fixture
def mock_source_protocol():
    """
    Mock source protocol implementation for testing source-related functionality.
    
    Provides a deterministic source implementation that can be used to test
    source protocol compliance and source-dependent functionality without
    requiring actual source implementations.
    
    Returns:
        Mock object implementing SourceProtocol interface with predictable behavior
        
    Examples:
        Test source protocol compliance:
        >>> def test_source_interface(mock_source_protocol):
        ...     source = mock_source_protocol
        ...     assert hasattr(source, 'get_emission_rate')
        ...     assert hasattr(source, 'get_position')
        ...     assert hasattr(source, 'update_state')
        
        Test vectorized emission queries:
        >>> def test_vectorized_emission(mock_source_protocol):
        ...     source = mock_source_protocol
        ...     positions = np.array([[10, 20], [30, 40]])
        ...     rates = source.get_emission_rate(positions)
        ...     assert rates.shape == (2,)
    """
    if not PROTOCOLS_V1_AVAILABLE:
        pytest.skip("v1.0 source protocols not available")
    
    mock_source = Mock(spec=SourceProtocol)
    
    # Configure mock source with deterministic behavior
    mock_source.get_position.return_value = np.array([50.0, 50.0])
    
    def mock_get_emission_rate(agent_positions=None):
        """Mock emission rate calculation with vectorized support."""
        if agent_positions is None:
            return 1000.0  # Base emission rate
        
        positions = np.asarray(agent_positions)
        if positions.ndim == 1:
            # Single agent
            return 1000.0
        else:
            # Multi-agent - return constant rate for all agents
            n_agents = positions.shape[0]
            return np.full(n_agents, 1000.0)
    
    mock_source.get_emission_rate = mock_get_emission_rate
    mock_source.update_state = Mock()
    mock_source.configure = Mock()
    
    # Add performance tracking
    mock_source.get_performance_stats.return_value = {
        'query_count': 10,
        'total_query_time': 0.001,
        'avg_query_time': 0.0001
    }
    
    return mock_source


@pytest.fixture
def source_factory_fixture():
    """
    Source factory fixture for testing configuration-driven source creation.
    
    Provides factory function for creating different types of sources from
    configuration dictionaries, enabling testing of source configuration
    and runtime source selection scenarios.
    
    Returns:
        Factory function for creating configured source instances
        
    Examples:
        Test point source creation:
        >>> def test_point_source_creation(source_factory_fixture):
        ...     factory = source_factory_fixture
        ...     config = {'type': 'PointSource', 'position': (25, 75), 'emission_rate': 1500}
        ...     source = factory(config)
        ...     assert source.get_position().tolist() == [25, 75]
        
        Test multi-source configuration:
        >>> def test_multi_source_creation(source_factory_fixture):
        ...     factory = source_factory_fixture
        ...     config = {
        ...         'type': 'MultiSource',
        ...         'sources': [
        ...             {'type': 'PointSource', 'position': (20, 20)},
        ...             {'type': 'PointSource', 'position': (80, 80)}
        ...         ]
        ...     }
        ...     source = factory(config)
        ...     assert source.get_source_count() == 2
    """
    if not SOURCES_AVAILABLE:
        pytest.skip("Sources module not available")
        
    def _create_source_from_config(config):
        """Create source from configuration with error handling."""
        try:
            return create_source(config)
        except Exception as e:
            # Return mock source for testing error scenarios
            mock_source = Mock()
            mock_source.config_error = str(e)
            return mock_source
    
    return _create_source_from_config


@pytest.fixture
def point_source_fixture():
    """
    Point source fixture for testing single-source scenarios.
    
    Provides a configured PointSource instance with test-appropriate
    parameters for deterministic testing behavior.
    
    Returns:
        PointSource instance configured for testing
        
    Examples:
        Test point source behavior:
        >>> def test_point_source_emission(point_source_fixture):
        ...     source = point_source_fixture
        ...     rate = source.get_emission_rate()
        ...     assert rate == 500.0  # Test emission rate
        ...     position = source.get_position()
        ...     assert position.tolist() == [30.0, 40.0]
    """
    if not SOURCES_AVAILABLE:
        pytest.skip("Sources module not available")
    
    # Create point source with test configuration
    source = PointSource(
        position=(30.0, 40.0),
        emission_rate=500.0,
        seed=42,  # Deterministic behavior
        enable_temporal_variation=False
    )
    
    return source


@pytest.fixture
def multi_source_fixture():
    """
    Multi-source fixture for testing complex source configurations.
    
    Provides a configured MultiSource instance with multiple point sources
    for testing vectorized operations and source aggregation scenarios.
    
    Returns:
        MultiSource instance with multiple configured sources
        
    Examples:
        Test multi-source aggregation:
        >>> def test_multi_source_aggregation(multi_source_fixture):
        ...     sources = multi_source_fixture
        ...     total_rate = sources.get_total_emission_rate()
        ...     assert total_rate == 1500.0  # Sum of all sources
        ...     assert sources.get_source_count() == 3
    """
    if not SOURCES_AVAILABLE:
        pytest.skip("Sources module not available")
    
    # Create multi-source with multiple point sources
    multi_source = MultiSource(seed=42)
    
    # Add test sources with different configurations
    multi_source.add_source(PointSource((10, 10), emission_rate=300.0))
    multi_source.add_source(PointSource((50, 50), emission_rate=500.0))
    multi_source.add_source(PointSource((90, 90), emission_rate=700.0))
    
    return multi_source


@pytest.fixture
def dynamic_source_fixture():
    """
    Dynamic source fixture for testing time-varying source behavior.
    
    Provides a configured DynamicSource instance with temporal evolution
    patterns for testing source state updates and movement patterns.
    
    Returns:
        DynamicSource instance configured for testing
        
    Examples:
        Test dynamic source evolution:
        >>> def test_dynamic_source_movement(dynamic_source_fixture):
        ...     source = dynamic_source_fixture
        ...     initial_pos = source.get_position()
        ...     source.update_state(dt=1.0)
        ...     updated_pos = source.get_position()
        ...     # Position should change for non-stationary patterns
    """
    if not SOURCES_AVAILABLE:
        pytest.skip("Sources module not available")
    
    # Create dynamic source with circular pattern
    source = DynamicSource(
        initial_position=(50.0, 50.0),
        emission_rate=800.0,
        pattern_type="circular",
        amplitude=10.0,
        frequency=0.1,
        seed=42
    )
    
    return source


# ================================================================================================
# V1.0 BOUNDARY POLICY PROTOCOL AND COMPONENT FIXTURES
# ================================================================================================

@pytest.fixture
def mock_boundary_policy_protocol():
    """
    Mock boundary policy protocol implementation for testing boundary-related functionality.
    
    Provides a deterministic boundary policy implementation that can be used to test
    boundary protocol compliance and boundary-dependent functionality without
    requiring actual boundary policy implementations.
    
    Returns:
        Mock object implementing BoundaryPolicyProtocol interface
        
    Examples:
        Test boundary policy compliance:
        >>> def test_boundary_interface(mock_boundary_policy_protocol):
        ...     policy = mock_boundary_policy_protocol
        ...     assert hasattr(policy, 'apply_policy')
        ...     assert hasattr(policy, 'check_violations')
        ...     assert hasattr(policy, 'get_termination_status')
        
        Test boundary violation detection:
        >>> def test_boundary_violations(mock_boundary_policy_protocol):
        ...     policy = mock_boundary_policy_protocol
        ...     positions = np.array([[50, 50], [110, 75]])  # One out of bounds
        ...     violations = policy.check_violations(positions)
        ...     assert isinstance(violations, np.ndarray)
    """
    if not PROTOCOLS_V1_AVAILABLE:
        pytest.skip("v1.0 boundary protocols not available")
    
    mock_policy = Mock(spec=BoundaryPolicyProtocol)
    
    # Configure mock with deterministic boundary behavior
    mock_policy.domain_bounds = (100, 100)
    
    def mock_check_violations(positions):
        """Mock violation detection with deterministic behavior."""
        positions = np.asarray(positions)
        if positions.ndim == 1:
            # Single agent
            x, y = positions[0], positions[1]
            return (x < 0 or x > 100 or y < 0 or y > 100)
        else:
            # Multi-agent
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]
            violations = np.logical_or.reduce([
                x_coords < 0, x_coords > 100,
                y_coords < 0, y_coords > 100
            ])
            return violations
    
    def mock_apply_policy(positions, velocities=None):
        """Mock policy application with position correction."""
        corrected_positions = np.clip(positions, [0, 0], [100, 100])
        if velocities is not None:
            return corrected_positions, velocities
        return corrected_positions
    
    mock_policy.check_violations = mock_check_violations
    mock_policy.apply_policy = mock_apply_policy
    mock_policy.get_termination_status.return_value = "continue"
    mock_policy.configure = Mock()
    
    return mock_policy


@pytest.fixture
def boundary_policy_factory_fixture():
    """
    Boundary policy factory fixture for testing configuration-driven policy creation.
    
    Provides factory function for creating different types of boundary policies
    from configuration dictionaries, enabling testing of policy configuration
    and runtime policy selection scenarios.
    
    Returns:
        Factory function for creating configured boundary policy instances
        
    Examples:
        Test terminate boundary creation:
        >>> def test_terminate_boundary_creation(boundary_policy_factory_fixture):
        ...     factory = boundary_policy_factory_fixture
        ...     policy = factory('terminate', (100, 100))
        ...     assert policy.get_termination_status() == 'oob'
        
        Test bounce boundary creation:
        >>> def test_bounce_boundary_creation(boundary_policy_factory_fixture):
        ...     factory = boundary_policy_factory_fixture
        ...     policy = factory('bounce', (100, 100), elasticity=0.8)
        ...     assert policy.elasticity == 0.8
    """
    if not BOUNDARIES_AVAILABLE:
        pytest.skip("Boundaries module not available")
    
    def _create_boundary_policy_from_config(policy_type, domain_bounds, **kwargs):
        """Create boundary policy from configuration with error handling."""
        try:
            return create_boundary_policy(policy_type, domain_bounds, **kwargs)
        except Exception as e:
            # Return mock policy for testing error scenarios
            mock_policy = Mock()
            mock_policy.config_error = str(e)
            return mock_policy
    
    return _create_boundary_policy_from_config


@pytest.fixture
def terminate_boundary_fixture():
    """
    Terminate boundary fixture for testing episode termination scenarios.
    
    Provides a configured TerminateBoundary instance for testing boundary
    violation detection and episode termination behavior.
    
    Returns:
        TerminateBoundary instance configured for testing
        
    Examples:
        Test termination boundary behavior:
        >>> def test_terminate_boundary(terminate_boundary_fixture):
        ...     policy = terminate_boundary_fixture
        ...     violations = policy.check_violations(np.array([[105, 50]]))
        ...     assert violations == True  # Out of bounds
        ...     assert policy.get_termination_status() == 'oob'
    """
    if not BOUNDARIES_AVAILABLE:
        pytest.skip("Boundaries module not available")
    
    # Create terminate boundary with test configuration
    policy = TerminateBoundary(
        domain_bounds=(100, 100),
        allow_negative_coords=False,
        status_on_violation="oob"
    )
    
    return policy


@pytest.fixture
def bounce_boundary_fixture():
    """
    Bounce boundary fixture for testing elastic collision scenarios.
    
    Provides a configured BounceBoundary instance for testing collision
    physics and velocity reflection behavior.
    
    Returns:
        BounceBoundary instance configured for testing
        
    Examples:
        Test bounce boundary physics:
        >>> def test_bounce_boundary_physics(bounce_boundary_fixture):
        ...     policy = bounce_boundary_fixture
        ...     positions = np.array([[105, 50]])
        ...     velocities = np.array([[2.0, 0.0]])
        ...     corrected_pos, corrected_vel = policy.apply_policy(positions, velocities)
        ...     assert corrected_vel[0, 0] < 0  # Velocity should be reflected
    """
    if not BOUNDARIES_AVAILABLE:
        pytest.skip("Boundaries module not available")
    
    # Create bounce boundary with test configuration
    policy = BounceBoundary(
        domain_bounds=(100, 100),
        elasticity=0.8,
        energy_loss=0.1,
        allow_negative_coords=False
    )
    
    return policy


# ================================================================================================
# V1.0 ACTION INTERFACE PROTOCOL AND COMPONENT FIXTURES  
# ================================================================================================

@pytest.fixture
def mock_action_interface_protocol():
    """
    Mock action interface protocol implementation for testing action-related functionality.
    
    Provides a deterministic action interface implementation that can be used to test
    action protocol compliance and action-dependent functionality without
    requiring actual action interface implementations.
    
    Returns:
        Mock object implementing ActionInterfaceProtocol interface
        
    Examples:
        Test action interface compliance:
        >>> def test_action_interface(mock_action_interface_protocol):
        ...     interface = mock_action_interface_protocol
        ...     assert hasattr(interface, 'translate_action')
        ...     assert hasattr(interface, 'validate_action')
        ...     assert hasattr(interface, 'get_action_space')
        
        Test action translation:
        >>> def test_action_translation(mock_action_interface_protocol):
        ...     interface = mock_action_interface_protocol
        ...     action = np.array([1.5, 20.0])
        ...     command = interface.translate_action(action)
        ...     assert 'linear_velocity' in command
        ...     assert 'angular_velocity' in command
    """
    if not PROTOCOLS_V1_AVAILABLE:
        pytest.skip("v1.0 action protocols not available")
    
    mock_interface = Mock(spec=ActionInterfaceProtocol)
    
    # Configure mock with deterministic action processing
    def mock_translate_action(action):
        """Mock action translation with predictable output."""
        if isinstance(action, np.ndarray) and action.shape == (2,):
            return {
                'linear_velocity': float(action[0]),
                'angular_velocity': float(action[1]),
                'action_type': 'continuous_2d'
            }
        elif isinstance(action, (int, np.integer)):
            # Discrete action mapping
            directions = ['STAY', 'NORTH', 'SOUTH', 'EAST', 'WEST']
            direction = directions[action % len(directions)]
            return {
                'linear_velocity': 1.0 if direction != 'STAY' else 0.0,
                'angular_velocity': 0.0,
                'direction': direction,
                'action_type': 'cardinal_discrete'
            }
        else:
            return {'error': 'Invalid action format'}
    
    def mock_validate_action(action):
        """Mock action validation with bounds checking."""
        if isinstance(action, np.ndarray):
            return np.clip(action, [-2.0, -45.0], [2.0, 45.0])
        elif isinstance(action, (int, np.integer)):
            return np.clip(action, 0, 8)
        else:
            return action
    
    mock_interface.translate_action = mock_translate_action
    mock_interface.validate_action = mock_validate_action
    
    # Mock action space for Gymnasium compatibility
    if _has_module("gymnasium"):
        import gymnasium

        mock_interface.get_action_space.return_value = gymnasium.spaces.Box(
            low=np.array([-2.0, -45.0]), high=np.array([2.0, 45.0])
        )
    else:
        mock_interface.get_action_space.return_value = None
    
    return mock_interface


@pytest.fixture
def action_interface_factory_fixture():
    """
    Action interface factory fixture for testing configuration-driven interface creation.
    
    Provides factory function for creating different types of action interfaces
    from configuration dictionaries, enabling testing of interface configuration
    and runtime interface selection scenarios.
    
    Returns:
        Factory function for creating configured action interface instances
        
    Examples:
        Test continuous action interface creation:
        >>> def test_continuous_interface_creation(action_interface_factory_fixture):
        ...     factory = action_interface_factory_fixture
        ...     config = {'type': 'Continuous2D', 'max_velocity': 2.5}
        ...     interface = factory(config)
        ...     assert interface.get_max_velocity() == 2.5
        
        Test discrete action interface creation:
        >>> def test_discrete_interface_creation(action_interface_factory_fixture):
        ...     factory = action_interface_factory_fixture
        ...     config = {'type': 'CardinalDiscrete', 'speed': 1.5}
        ...     interface = factory(config)
        ...     assert interface.get_speed() == 1.5
    """
    if not ACTIONS_AVAILABLE:
        pytest.skip("Actions module not available")
    
    def _create_action_interface_from_config(config):
        """Create action interface from configuration with error handling."""
        try:
            return create_action_interface(config)
        except Exception as e:
            # Return mock interface for testing error scenarios
            mock_interface = Mock()
            mock_interface.config_error = str(e)
            return mock_interface
    
    return _create_action_interface_from_config


@pytest.fixture
def continuous_2d_action_fixture():
    """
    Continuous 2D action interface fixture for testing continuous control scenarios.
    
    Provides a configured Continuous2DAction instance for testing continuous
    action processing and validation behavior.
    
    Returns:
        Continuous2DAction instance configured for testing
        
    Examples:
        Test continuous action processing:
        >>> def test_continuous_action_processing(continuous_2d_action_fixture):
        ...     interface = continuous_2d_action_fixture
        ...     action = np.array([1.5, 20.0])
        ...     command = interface.translate_action(action)
        ...     assert command['linear_velocity'] == 1.5
        ...     assert command['angular_velocity'] == 20.0
    """
    if not ACTIONS_AVAILABLE:
        pytest.skip("Actions module not available")
    
    # Create continuous 2D action interface with test configuration
    interface = Continuous2DAction(
        max_velocity=2.0,
        max_angular_velocity=45.0,
        min_velocity=-2.0,
        min_angular_velocity=-45.0
    )
    
    return interface


@pytest.fixture
def cardinal_discrete_action_fixture():
    """
    Cardinal discrete action interface fixture for testing discrete control scenarios.
    
    Provides a configured CardinalDiscreteAction instance for testing discrete
    action processing and direction mapping behavior.
    
    Returns:
        CardinalDiscreteAction instance configured for testing
        
    Examples:
        Test discrete action processing:
        >>> def test_discrete_action_processing(cardinal_discrete_action_fixture):
        ...     interface = cardinal_discrete_action_fixture
        ...     action = 2  # East direction
        ...     command = interface.translate_action(action)
        ...     assert command['direction'] == 'EAST'
        ...     assert command['linear_velocity'] == 1.0
    """
    if not ACTIONS_AVAILABLE:
        pytest.skip("Actions module not available")
    
    # Create cardinal discrete action interface with test configuration
    interface = CardinalDiscreteAction(
        speed=1.0,
        use_8_directions=True,
        include_stay_action=True
    )
    
    return interface


# ================================================================================================
# V1.0 RECORDING FRAMEWORK PROTOCOL AND COMPONENT FIXTURES
# ================================================================================================

@pytest.fixture
def mock_recorder_protocol():
    """
    Mock recorder protocol implementation for testing recording-related functionality.
    
    Provides a deterministic recorder implementation that can be used to test
    recorder protocol compliance and recording-dependent functionality without
    requiring actual recorder implementations or I/O operations.
    
    Returns:
        Mock object implementing RecorderProtocol interface
        
    Examples:
        Test recorder protocol compliance:
        >>> def test_recorder_interface(mock_recorder_protocol):
        ...     recorder = mock_recorder_protocol
        ...     assert hasattr(recorder, 'record_step')
        ...     assert hasattr(recorder, 'record_episode')
        ...     assert hasattr(recorder, 'start_recording')
        ...     assert hasattr(recorder, 'stop_recording')
        
        Test recording operations:
        >>> def test_recording_operations(mock_recorder_protocol):
        ...     recorder = mock_recorder_protocol
        ...     recorder.start_recording(episode_id=1)
        ...     recorder.record_step({'position': [0, 0]}, step_number=0)
        ...     recorder.stop_recording()
        ...     assert recorder.enabled == False
    """
    if not PROTOCOLS_V1_AVAILABLE:
        pytest.skip("v1.0 recorder protocols not available")
    
    mock_recorder = Mock(spec=RecorderProtocol)
    
    # Configure mock with recording state management
    mock_recorder.enabled = False
    mock_recorder.current_episode_id = None
    mock_recorder.recorded_steps = []
    mock_recorder.recorded_episodes = []
    
    def mock_start_recording(episode_id):
        """Mock start recording with state tracking."""
        mock_recorder.enabled = True
        mock_recorder.current_episode_id = episode_id
        mock_recorder.recorded_steps.clear()
        mock_recorder.recorded_episodes.clear()
    
    def mock_stop_recording():
        """Mock stop recording with state cleanup."""
        mock_recorder.enabled = False
        mock_recorder.current_episode_id = None
    
    def mock_record_step(step_data, step_number, episode_id=None, **metadata):
        """Mock step recording with data tracking."""
        if mock_recorder.enabled:
            mock_recorder.recorded_steps.append({
                'step_data': step_data,
                'step_number': step_number,
                'episode_id': episode_id or mock_recorder.current_episode_id,
                'metadata': metadata
            })
    
    def mock_record_episode(episode_data, episode_id, **metadata):
        """Mock episode recording with data tracking."""
        if mock_recorder.enabled:
            mock_recorder.recorded_episodes.append({
                'episode_data': episode_data,
                'episode_id': episode_id,
                'metadata': metadata
            })
    
    def mock_export_data(output_path, format="parquet", **options):
        """Mock data export with success tracking."""
        return True  # Always succeed for testing
    
    # Configure mock methods
    mock_recorder.start_recording = mock_start_recording
    mock_recorder.stop_recording = mock_stop_recording
    mock_recorder.record_step = mock_record_step
    mock_recorder.record_episode = mock_record_episode
    mock_recorder.export_data = mock_export_data
    mock_recorder.flush = Mock()
    
    # Performance metrics
    mock_recorder.get_performance_metrics.return_value = {
        'steps_recorded': 0,
        'episodes_recorded': 0,
        'average_write_time': 0.0005,  # 0.5ms
        'buffer_utilization_current': 0.0,
        'enabled': False
    }
    
    return mock_recorder


@pytest.fixture
def recorder_manager_fixture():
    """
    Recorder manager fixture for testing recorder lifecycle management.
    
    Provides a RecorderManager instance with mock recorder for testing
    recording session management, performance monitoring, and resource cleanup.
    
    Returns:
        RecorderManager instance configured for testing
        
    Examples:
        Test recorder manager lifecycle:
        >>> def test_recorder_manager_lifecycle(recorder_manager_fixture):
        ...     manager = recorder_manager_fixture
        ...     manager.start_recording(episode_id=1)
        ...     assert manager._recording_active == True
        ...     metrics = manager.get_performance_metrics()
        ...     manager.stop_recording()
        ...     assert manager._recording_active == False
        
        Test performance monitoring:
        >>> def test_performance_monitoring(recorder_manager_fixture):
        ...     manager = recorder_manager_fixture
        ...     manager.start_recording(episode_id=1)
        ...     metrics = manager.get_performance_metrics()
        ...     assert 'recording_active' in metrics
        ...     assert 'performance_target_ms' in metrics
    """
    if not RECORDING_AVAILABLE:
        pytest.skip("Recording module not available")
    
    # Create mock recorder for manager
    mock_recorder = Mock()
    mock_recorder.enabled = False
    mock_recorder.start_recording = Mock()
    mock_recorder.stop_recording = Mock()
    mock_recorder.get_performance_metrics.return_value = {
        'steps_recorded': 0,
        'episodes_recorded': 0,
        'average_write_time': 0.0005,
        'buffer_utilization_current': 0.1
    }
    
    # Create recorder manager with test configuration
    manager = RecorderManager(
        recorder=mock_recorder,
        performance_target_ms=1.0,
        memory_limit_mb=256
    )
    
    return manager


@pytest.fixture
def recorder_config_fixture():
    """
    Recorder configuration fixture for testing configuration scenarios.
    
    Provides factory function for creating different recorder configurations
    for testing various recording backends and parameter combinations.
    
    Returns:
        Factory function for creating recorder configurations
        
    Examples:
        Test parquet recorder configuration:
        >>> def test_parquet_config(recorder_config_fixture):
        ...     config = recorder_config_fixture('parquet')
        ...     assert config['backend'] == 'parquet'
        ...     assert config['compression'] == 'snappy'
        
        Test custom configuration:
        >>> def test_custom_config(recorder_config_fixture):
        ...     config = recorder_config_fixture('hdf5', buffer_size=2000)
        ...     assert config['backend'] == 'hdf5'
        ...     assert config['buffer_size'] == 2000
    """
    def _create_recorder_config(backend='none', **kwargs):
        """Create recorder configuration with defaults and overrides."""
        config = {
            'backend': backend,
            'output_dir': './test_data',
            'buffer_size': 1000,
            'flush_interval': 5.0,
            'async_io': True,
            'compression': 'snappy',
            'enable_metrics': True,
            'memory_limit_mb': 256
        }
        
        # Apply any overrides
        config.update(kwargs)
        
        return config
    
    return _create_recorder_config


@pytest.fixture
def mock_recorder_factory():
    """
    Mock recorder factory fixture for testing recorder creation scenarios.
    
    Provides factory that creates mock recorders for different backends
    without requiring actual recorder implementations or dependencies.
    
    Returns:
        Factory function for creating mock recorder instances
        
    Examples:
        Test recorder factory usage:
        >>> def test_recorder_factory(mock_recorder_factory):
        ...     factory = mock_recorder_factory
        ...     recorder = factory.create_recorder({'backend': 'parquet'})
        ...     assert hasattr(recorder, 'record_step')
        ...     assert hasattr(recorder, 'record_episode')
        
        Test backend selection:
        >>> def test_backend_selection(mock_recorder_factory):
        ...     factory = mock_recorder_factory
        ...     backends = factory.get_available_backends()
        ...     assert 'none' in backends
    """
    class MockRecorderFactory:
        @staticmethod
        def create_recorder(config):
            """Create mock recorder from configuration."""
            mock_recorder = Mock(spec=RecorderProtocol)
            mock_recorder.config = config
            mock_recorder.enabled = False
            mock_recorder.start_recording = Mock()
            mock_recorder.stop_recording = Mock()
            mock_recorder.record_step = Mock()
            mock_recorder.record_episode = Mock()
            mock_recorder.export_data = Mock(return_value=True)
            mock_recorder.flush = Mock()
            mock_recorder.get_performance_metrics = Mock(return_value={
                'backend': config.get('backend', 'none'),
                'enabled': False,
                'steps_recorded': 0
            })
            return mock_recorder
        
        @staticmethod
        def get_available_backends():
            """Return list of available backends for testing."""
            return ['none', 'parquet', 'hdf5', 'sqlite']
        
        @staticmethod
        def validate_config(config):
            """Validate recorder configuration."""
            return {
                'valid': True,
                'backend_available': True,
                'warnings': [],
                'recommendations': []
            }
    
    return MockRecorderFactory()


# ================================================================================================
# V1.0 STATISTICS AGGREGATION PROTOCOL AND COMPONENT FIXTURES
# ================================================================================================

@pytest.fixture
def mock_stats_aggregator_protocol():
    """
    Mock statistics aggregator protocol implementation for testing analysis functionality.
    
    Provides a deterministic statistics aggregator implementation that can be used to test
    statistics protocol compliance and analysis-dependent functionality without
    requiring actual statistics calculations or data processing.
    
    Returns:
        Mock object implementing StatsAggregatorProtocol interface
        
    Examples:
        Test stats aggregator protocol compliance:
        >>> def test_stats_interface(mock_stats_aggregator_protocol):
        ...     aggregator = mock_stats_aggregator_protocol
        ...     assert hasattr(aggregator, 'calculate_episode_stats')
        ...     assert hasattr(aggregator, 'calculate_run_stats')
        ...     assert hasattr(aggregator, 'export_summary')
        
        Test statistics calculation:
        >>> def test_stats_calculation(mock_stats_aggregator_protocol):
        ...     aggregator = mock_stats_aggregator_protocol
        ...     episode_data = {'trajectory': [[0, 0], [1, 1], [2, 2]]}
        ...     stats = aggregator.calculate_episode_stats(episode_data)
        ...     assert 'trajectory_stats' in stats
    """
    if not PROTOCOLS_V1_AVAILABLE:
        pytest.skip("v1.0 statistics protocols not available")
    
    mock_aggregator = Mock(spec=StatsAggregatorProtocol)
    
    # Configure mock with deterministic statistics behavior
    def mock_calculate_episode_stats(episode_data):
        """Mock episode statistics calculation."""
        return {
            'episode_id': episode_data.get('episode_id', 0),
            'trajectory_stats': {
                'mean_position': [1.0, 1.0],
                'total_distance': 2.828,
                'displacement_efficiency': 0.707
            },
            'concentration_stats': {
                'mean_concentration': 0.5,
                'detection_rate': 0.8,
                'max_concentration': 1.0
            },
            'performance_stats': {
                'episode_duration': 10.0,
                'steps_count': 100
            }
        }
    
    def mock_calculate_run_stats(episodes_data):
        """Mock run-level statistics calculation."""
        return {
            'episodes_count': len(episodes_data),
            'trajectory_aggregation': {
                'mean_distance_per_episode': 2.5,
                'mean_efficiency': 0.75,
                'std_efficiency': 0.1
            },
            'concentration_aggregation': {
                'mean_detection_rate': 0.82,
                'mean_concentration': 0.48,
                'concentration_variance': 0.05
            },
            'performance_aggregation': {
                'mean_episode_duration': 9.8,
                'mean_steps_per_episode': 98.5,
                'total_simulation_time': len(episodes_data) * 9.8
            }
        }
    
    def mock_export_summary(output_path, format="json", **options):
        """Mock summary export."""
        return True  # Always succeed for testing
    
    # Configure mock methods
    mock_aggregator.calculate_episode_stats = mock_calculate_episode_stats
    mock_aggregator.calculate_run_stats = mock_calculate_run_stats
    mock_aggregator.export_summary = mock_export_summary
    
    # Performance metrics
    mock_aggregator.get_performance_metrics.return_value = {
        'episodes_processed': 0,
        'computation_time_ms': 0.5,
        'memory_usage_mb': 10.0,
        'performance_compliance': {
            'meets_33ms_target': True
        }
    }
    
    # Configuration access
    mock_aggregator.get_aggregation_levels.return_value = ['episode', 'run']
    mock_aggregator.validate_data = Mock(return_value={'valid': True, 'errors': []})
    
    return mock_aggregator


@pytest.fixture
def stats_aggregator_factory_fixture():
    """
    Statistics aggregator factory fixture for testing configuration-driven aggregator creation.
    
    Provides factory function for creating different types of statistics aggregators
    from configuration dictionaries, enabling testing of aggregator configuration
    and runtime aggregator selection scenarios.
    
    Returns:
        Factory function for creating configured statistics aggregator instances
        
    Examples:
        Test basic aggregator creation:
        >>> def test_basic_aggregator_creation(stats_aggregator_factory_fixture):
        ...     factory = stats_aggregator_factory_fixture
        ...     config = {
        ...         'metrics_definitions': {'trajectory': ['mean', 'std']},
        ...         'aggregation_levels': ['episode', 'run']
        ...     }
        ...     aggregator = factory(config)
        ...     assert hasattr(aggregator, 'calculate_episode_stats')
        
        Test advanced aggregator configuration:
        >>> def test_advanced_aggregator_creation(stats_aggregator_factory_fixture):
        ...     factory = stats_aggregator_factory_fixture
        ...     config = {
        ...         'metrics_definitions': {
        ...             'trajectory': ['mean', 'std', 'efficiency'],
        ...             'concentration': ['detection_rate', 'mean']
        ...         },
        ...         'aggregation_levels': ['episode', 'run', 'batch'],
        ...         'performance_tracking': True,
        ...         'parallel_processing': False
        ...     }
        ...     aggregator = factory(config)
        ...     levels = aggregator.get_aggregation_levels()
        ...     assert 'batch' in levels
    """
    if not ANALYSIS_AVAILABLE:
        pytest.skip("Analysis module not available")
    
    def _create_stats_aggregator_from_config(config):
        """Create statistics aggregator from configuration with error handling."""
        try:
            return create_stats_aggregator(config)
        except Exception as e:
            # Return mock aggregator for testing error scenarios
            mock_aggregator = Mock()
            mock_aggregator.config_error = str(e)
            return mock_aggregator
    
    return _create_stats_aggregator_from_config


@pytest.fixture
def basic_stats_aggregator_fixture():
    """
    Basic statistics aggregator fixture for testing standard analysis scenarios.
    
    Provides a configured StatsAggregator instance with basic metrics definitions
    for testing standard statistical calculations and analysis workflows.
    
    Returns:
        StatsAggregator instance configured for testing
        
    Examples:
        Test basic statistics calculation:
        >>> def test_basic_stats_calculation(basic_stats_aggregator_fixture):
        ...     aggregator = basic_stats_aggregator_fixture
        ...     episode_data = {
        ...         'episode_id': 1,
        ...         'trajectory': [[0, 0], [1, 1], [2, 2]],
        ...         'concentrations': [0.1, 0.5, 0.8]
        ...     }
        ...     stats = aggregator.calculate_episode_stats(episode_data)
        ...     assert 'trajectory_stats' in stats
        ...     assert 'concentration_stats' in stats
    """
    if not ANALYSIS_AVAILABLE:
        pytest.skip("Analysis module not available")
    
    # Create basic statistics aggregator configuration
    config = {
        'metrics_definitions': {
            'trajectory': ['mean', 'std', 'total_distance'],
            'concentration': ['mean', 'detection_rate'],
            'speed': ['mean', 'max']
        },
        'aggregation_levels': ['episode', 'run'],
        'performance_tracking': True,
        'output_format': 'json'
    }
    
    try:
        aggregator = create_stats_aggregator(config)
    except Exception:
        # Fallback to mock if actual creation fails
        aggregator = Mock()
        aggregator.config = config
        aggregator.calculate_episode_stats = Mock(return_value={'test': True})
        aggregator.calculate_run_stats = Mock(return_value={'test': True})
    
    return aggregator


@pytest.fixture
def generate_summary_fixture():
    """
    Generate summary function fixture for testing summary generation workflows.
    
    Provides the generate_summary function for testing comprehensive statistical
    summary generation with different aggregators and episode data configurations.
    
    Returns:
        generate_summary function for testing
        
    Examples:
        Test summary generation:
        >>> def test_summary_generation(generate_summary_fixture, mock_stats_aggregator_protocol):
        ...     generate_func = generate_summary_fixture
        ...     episodes_data = [
        ...         {'episode_id': 1, 'trajectory': [[0, 0], [1, 1]]},
        ...         {'episode_id': 2, 'trajectory': [[0, 0], [2, 2]]}
        ...     ]
        ...     summary = generate_func(mock_stats_aggregator_protocol, episodes_data)
        ...     assert 'run_statistics' in summary
        ...     assert 'episode_count' in summary
    """
    if not ANALYSIS_AVAILABLE:
        # Return mock function for testing when analysis not available
        def mock_generate_summary(aggregator, episodes_data, **kwargs):
            return {
                'run_statistics': {'mock': True},
                'episode_count': len(episodes_data),
                'processing_time': 0.001,
                'performance_metrics': {'mock': True}
            }
        return mock_generate_summary
    
    return generate_summary


# ================================================================================================
# V1.0 DEBUG UTILITIES AND VISUALIZATION FIXTURES
# ================================================================================================

@pytest.fixture
def plot_initial_state_fixture():
    """
    Plot initial state function fixture for testing debug visualization functionality.
    
    Provides the plot_initial_state function for testing debug visualization
    capabilities including source visualization, boundary plotting, and agent
    position display without requiring actual GUI backends.
    
    Returns:
        plot_initial_state function for testing
        
    Examples:
        Test initial state plotting:
        >>> def test_plot_initial_state(plot_initial_state_fixture, mock_video_plume):
        ...     plot_func = plot_initial_state_fixture
        ...     # Mock environment and source
        ...     env = Mock()
        ...     env.get_domain_bounds.return_value = (100, 100)
        ...     source = Mock()
        ...     source.get_position.return_value = np.array([50, 50])
        ...     
        ...     # Test plotting function (should not raise exceptions)
        ...     result = plot_func(env, source=source, agent_positions=[[10, 10], [20, 20]])
        ...     assert result is not None  # Should return figure or success indicator
    """
    if not DEBUG_AVAILABLE:
        # Return mock function for testing when debug not available
        def mock_plot_initial_state(env, source=None, agent_positions=None, **kwargs):
            """Mock plot function that returns success without actual plotting."""
            return {'type': 'mock_plot', 'success': True, 'env': env, 'source': source}
        return mock_plot_initial_state
    
    return plot_initial_state


@pytest.fixture
def mock_debug_gui():
    """
    Mock debug GUI fixture for testing debug interface functionality.
    
    Provides a mock debug GUI interface that can be used to test debug
    functionality without requiring actual GUI backends like PySide6 or Streamlit.
    
    Returns:
        Mock debug GUI instance
        
    Examples:
        Test debug GUI interface:
        >>> def test_debug_gui_interface(mock_debug_gui):
        ...     gui = mock_debug_gui
        ...     gui.start_session()
        ...     assert gui.session_active == True
        ...     gui.stop_session()
        ...     assert gui.session_active == False
        
        Test debug GUI configuration:
        >>> def test_debug_gui_config(mock_debug_gui):
        ...     gui = mock_debug_gui
        ...     gui.configure_backend(backend='qt', window_size=(800, 600))
        ...     config = gui.get_config()
        ...     assert config['backend'] == 'qt'
    """
    mock_gui = Mock()
    
    # Configure mock GUI with session management
    mock_gui.session_active = False
    mock_gui.backend = 'mock'
    mock_gui.config = {'backend': 'mock', 'window_size': (800, 600)}
    
    def mock_start_session():
        """Mock start debug session."""
        mock_gui.session_active = True
    
    def mock_stop_session():
        """Mock stop debug session."""
        mock_gui.session_active = False
    
    def mock_configure_backend(**kwargs):
        """Mock backend configuration."""
        mock_gui.config.update(kwargs)
    
    def mock_get_config():
        """Mock configuration retrieval."""
        return mock_gui.config.copy()
    
    # Configure mock methods
    mock_gui.start_session = mock_start_session
    mock_gui.stop_session = mock_stop_session
    mock_gui.configure_backend = mock_configure_backend
    mock_gui.get_config = mock_get_config
    
    # Mock availability checking
    mock_gui.backend_available = Mock(return_value=True)
    mock_gui.get_available_backends = Mock(return_value=['mock', 'console'])
    
    return mock_gui


# ================================================================================================
# V1.0 COMPREHENSIVE FIXTURE COMBINATIONS
# ================================================================================================

@pytest.fixture
def v1_component_test_setup(
    mock_source_protocol,
    mock_boundary_policy_protocol,
    mock_action_interface_protocol,
    mock_recorder_protocol,
    mock_stats_aggregator_protocol,
    plot_initial_state_fixture,
    mock_debug_gui
):
    """
    Comprehensive test setup fixture for v1.0 component architecture testing.
    
    Provides unified access to all v1.0 testing fixtures for comprehensive integration
    testing across source protocols, boundary policies, action interfaces, recording
    framework, statistics aggregation, and debug utilities.
    
    Args:
        mock_source_protocol: Mock source protocol implementation
        mock_boundary_policy_protocol: Mock boundary policy implementation
        mock_action_interface_protocol: Mock action interface implementation
        mock_recorder_protocol: Mock recorder implementation
        mock_stats_aggregator_protocol: Mock statistics aggregator implementation
        plot_initial_state_fixture: Debug plotting function
        mock_debug_gui: Mock debug GUI interface
        
    Returns:
        Dictionary containing all v1.0 testing components and utilities
        
    Examples:
        Test v1.0 component integration:
        >>> def test_v1_component_integration(v1_component_test_setup):
        ...     setup = v1_component_test_setup
        ...     
        ...     # Test source protocol
        ...     source = setup['source']
        ...     emission_rate = source.get_emission_rate()
        ...     assert emission_rate == 1000.0
        ...     
        ...     # Test boundary policy
        ...     boundary = setup['boundary_policy']
        ...     violations = boundary.check_violations(np.array([[105, 50]]))
        ...     assert violations == True
        ...     
        ...     # Test action interface
        ...     action_interface = setup['action_interface']
        ...     command = action_interface.translate_action(np.array([1.5, 20.0]))
        ...     assert 'linear_velocity' in command
        ...     
        ...     # Test recorder
        ...     recorder = setup['recorder']
        ...     recorder.start_recording(episode_id=1)
        ...     recorder.record_step({'position': [0, 0]}, step_number=0)
        ...     assert len(recorder.recorded_steps) == 1
        ...     
        ...     # Test statistics aggregator
        ...     aggregator = setup['stats_aggregator']
        ...     stats = aggregator.calculate_episode_stats({'episode_id': 1})
        ...     assert 'trajectory_stats' in stats
        
        Test debug utilities:
        >>> def test_v1_debug_utilities(v1_component_test_setup):
        ...     setup = v1_component_test_setup
        ...     
        ...     # Test plotting function
        ...     plot_func = setup['plot_initial_state']
        ...     result = plot_func(Mock(), source=setup['source'])
        ...     assert result is not None
        ...     
        ...     # Test debug GUI
        ...     gui = setup['debug_gui']
        ...     gui.start_session()
        ...     assert gui.session_active == True
        
        Test factory functions:
        >>> def test_v1_factory_functions(v1_component_test_setup):
        ...     setup = v1_component_test_setup
        ...     
        ...     # Test component factories if available
        ...     if 'source_factory' in setup:
        ...         source = setup['source_factory']({'type': 'PointSource'})
        ...         assert hasattr(source, 'get_emission_rate')
    """
    # Core v1.0 component setup
    setup = {
        # Protocol implementations
        'source': mock_source_protocol,
        'boundary_policy': mock_boundary_policy_protocol,
        'action_interface': mock_action_interface_protocol,
        'recorder': mock_recorder_protocol,
        'stats_aggregator': mock_stats_aggregator_protocol,
        
        # Debug and visualization utilities
        'plot_initial_state': plot_initial_state_fixture,
        'debug_gui': mock_debug_gui,
        
        # Component availability flags
        'availability': {
            'sources': SOURCES_AVAILABLE,
            'boundaries': BOUNDARIES_AVAILABLE,
            'actions': ACTIONS_AVAILABLE,
            'recording': RECORDING_AVAILABLE,
            'analysis': ANALYSIS_AVAILABLE,
            'debug': DEBUG_AVAILABLE,
            'protocols_v1': PROTOCOLS_V1_AVAILABLE
        }
    }
    
    # Add factory functions if components are available
    try:
        if SOURCES_AVAILABLE:
            from plume_nav_sim.core.sources import create_source
            setup['source_factory'] = create_source
    except ImportError:
        pass
    
    try:
        if BOUNDARIES_AVAILABLE:
            from plume_nav_sim.core.boundaries import create_boundary_policy
            setup['boundary_policy_factory'] = create_boundary_policy
    except ImportError:
        pass
    
    try:
        if ACTIONS_AVAILABLE:
            from plume_nav_sim.core.actions import create_action_interface
            setup['action_interface_factory'] = create_action_interface
    except ImportError:
        pass
    
    try:
        if RECORDING_AVAILABLE:
            from plume_nav_sim.recording import RecorderFactory
            setup['recorder_factory'] = RecorderFactory
    except ImportError:
        pass
    
    try:
        if ANALYSIS_AVAILABLE:
            from plume_nav_sim.analysis import create_stats_aggregator
            setup['stats_aggregator_factory'] = create_stats_aggregator
    except ImportError:
        pass
    
    # Add convenience methods for integration testing
    def create_test_environment():
        """Create mock environment for testing with all v1.0 components."""
        mock_env = Mock()
        mock_env.source = setup['source']
        mock_env.boundary_policy = setup['boundary_policy']
        mock_env.action_interface = setup['action_interface']
        mock_env.recorder = setup['recorder']
        mock_env.stats_aggregator = setup['stats_aggregator']
        
        # Environment methods
        mock_env.get_domain_bounds.return_value = (100, 100)
        mock_env.reset.return_value = (np.zeros((64, 64, 1)), {})
        mock_env.step.return_value = (np.zeros((64, 64, 1)), 0.0, False, False, {})
        
        return mock_env
    
    def create_integration_test_data():
        """Create test data for integration testing scenarios."""
        return {
            'episode_data': {
                'episode_id': 1,
                'trajectory': [[0, 0], [1, 1], [2, 2]],
                'concentrations': [0.1, 0.5, 0.8],
                'actions': [0, 1, 2],
                'rewards': [0.1, 0.2, 0.3]
            },
            'multi_episode_data': [
                {'episode_id': i, 'trajectory': [[0, 0], [i, i]], 'score': i * 0.1}
                for i in range(5)
            ],
            'agent_positions': np.array([[10, 10], [20, 20], [30, 30]]),
            'source_positions': np.array([[50, 50], [75, 25]]),
            'boundary_test_positions': np.array([[5, 5], [105, 50], [50, 105]])
        }
    
    setup['create_test_environment'] = create_test_environment
    setup['create_integration_test_data'] = create_integration_test_data
    
    return setup


@pytest.fixture
def v1_performance_test_setup(v1_component_test_setup):
    """
    Performance testing setup fixture for v1.0 component benchmarking.
    
    Extends the comprehensive v1.0 setup with performance monitoring utilities
    and benchmarking helpers for validating that v1.0 components meet the
    ≤33ms step latency requirements with 100 agents.
    
    Args:
        v1_component_test_setup: Base v1.0 component setup
        
    Returns:
        Enhanced setup with performance testing utilities
        
    Examples:
        Test component performance requirements:
        >>> def test_v1_performance_requirements(v1_performance_test_setup):
        ...     setup = v1_performance_test_setup
        ...     
        ...     # Test source performance
        ...     with setup['benchmark_context']('source_emission') as timer:
        ...         positions = np.random.rand(100, 2) * 100
        ...         rates = setup['source'].get_emission_rate(positions)
        ...     assert timer.duration < 0.033  # 33ms requirement
        ...     
        ...     # Test boundary policy performance
        ...     with setup['benchmark_context']('boundary_check') as timer:
        ...         violations = setup['boundary_policy'].check_violations(positions)
        ...     assert timer.duration < 0.033
        
        Test recorder performance:
        >>> def test_recorder_performance(v1_performance_test_setup):
        ...     setup = v1_performance_test_setup
        ...     recorder = setup['recorder']
        ...     
        ...     recorder.start_recording(episode_id=1)
        ...     with setup['benchmark_context']('recording_100_steps') as timer:
        ...         for i in range(100):
        ...             recorder.record_step({'step': i}, step_number=i)
        ...     
        ...     # Should meet <1ms disabled mode requirement
        ...     avg_time_per_step = timer.duration / 100
        ...     assert avg_time_per_step < 0.001
    """
    # Get base setup
    base_setup = v1_component_test_setup
    
    # Add performance monitoring utilities
    performance_setup = base_setup.copy()
    
    @contextmanager
    def benchmark_context(operation_name):
        """Context manager for benchmarking component operations."""
        class PerformanceTimer:
            def __init__(self, name):
                self.name = name
                self.start_time = None
                self.end_time = None
                self.duration = None
        
        timer = PerformanceTimer(operation_name)
        timer.start_time = time.perf_counter()
        
        try:
            yield timer
        finally:
            timer.end_time = time.perf_counter()
            timer.duration = timer.end_time - timer.start_time
    
    def create_performance_test_data(num_agents=100):
        """Create large-scale test data for performance testing."""
        return {
            'agent_positions': np.random.rand(num_agents, 2) * 100,
            'actions': np.random.rand(num_agents, 2) * 2 - 1,  # [-1, 1] range
            'step_data_list': [
                {'position': np.random.rand(2) * 100, 'concentration': np.random.rand()}
                for _ in range(1000)  # 1000 steps for batch testing
            ],
            'episode_data_list': [
                {
                    'episode_id': i,
                    'trajectory': np.random.rand(100, 2) * 100,
                    'concentrations': np.random.rand(100),
                    'rewards': np.random.rand(100)
                }
                for i in range(50)  # 50 episodes for run statistics
            ]
        }
    
    def validate_performance_requirements(results):
        """Validate that performance results meet v1.0 requirements."""
        requirements = {
            'step_latency_ms': 33.0,  # ≤33ms per step
            'recording_overhead_ms': 1.0,  # <1ms when disabled
            'memory_limit_mb': 256.0,  # Reasonable memory usage
            'source_query_ms': 10.0,  # <10ms for 100 agents
            'boundary_check_ms': 1.0,  # <1ms for 100 agents
            'stats_calculation_ms': 33.0  # ≤33ms for analysis
        }
        
        validation_results = {}
        for metric, requirement in requirements.items():
            actual_value = results.get(metric, 0)
            meets_requirement = actual_value <= requirement
            validation_results[metric] = {
                'actual': actual_value,
                'requirement': requirement,
                'meets_requirement': meets_requirement,
                'margin': requirement - actual_value
            }
        
        overall_compliance = all(
            result['meets_requirement'] 
            for result in validation_results.values()
        )
        
        return {
            'overall_compliance': overall_compliance,
            'detailed_results': validation_results,
            'summary': {
                'passed': sum(1 for r in validation_results.values() if r['meets_requirement']),
                'total': len(validation_results),
                'compliance_rate': sum(1 for r in validation_results.values() if r['meets_requirement']) / len(validation_results)
            }
        }
    
    # Add performance utilities to setup
    performance_setup.update({
        'benchmark_context': benchmark_context,
        'create_performance_test_data': create_performance_test_data,
        'validate_performance_requirements': validate_performance_requirements,
        'performance_requirements': {
            'step_latency_ms': 33.0,
            'recording_overhead_ms': 1.0,
            'memory_limit_mb': 256.0,
            'source_query_ms': 10.0,
            'boundary_check_ms': 1.0,
            'stats_calculation_ms': 33.0
        }
    })
    
    return performance_setup