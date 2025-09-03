"""
Shared fixtures for tests in the plume_nav_sim package.

This file contains pytest fixtures that are shared across multiple test files
to reduce duplication and ensure consistency across the enhanced cookiecutter-based
architecture including Hydra configuration management, CLI interface testing,
database session management, and v1.0 protocol-based architecture validation.

Key Testing Domains:
- Configuration Management: pytest-hydra integration for hierarchical config testing
- CLI Interface Testing: click.testing.CliRunner for command-line validation
- Database Session Management: In-memory SQLAlchemy sessions for persistence testing
- Seed Management: Controlled random state for reproducible test execution
- v1.0 Protocol Testing: RecorderProtocol, SourceProtocol, BoundaryPolicyProtocol testing
- Performance Monitoring: Recorder overhead validation and timing measurement utilities

Fixture Categories:
1. Core Scientific Fixtures: Video capture, navigator, and simulation mocks
2. Configuration Fixtures: Hydra config composition and validation fixtures
3. CLI Testing Fixtures: CliRunner instances and parameter validation fixtures
4. Database Fixtures: In-memory SQLAlchemy sessions and transaction management
5. Utility Fixtures: Seed management, environment isolation, and file system mocks
6. v1.0 Protocol Fixtures: Protocol-based component mocks for architecture testing
7. Recorder Testing Fixtures: Multi-backend recorder testing with performance monitoring
8. Performance Testing Fixtures: Timing utilities and overhead validation

Usage Examples:
    # Hydra configuration testing
    def test_config_validation(mock_hydra_config):
        assert mock_hydra_config.navigator.max_speed > 0
    
    # CLI interface testing  
    def test_cli_command(mock_cli_runner):
        runner, env = mock_cli_runner
        result = runner.invoke(main, ['run', '--dry-run'])
        assert result.exit_code == 0
    
    # Database session testing
    def test_persistence(mock_db_session):
        if mock_db_session:
            # Database operations with automatic transaction management
            pass
    
    # Seed management testing
    def test_reproducibility(mock_seed_manager):
        seed_manager = mock_seed_manager
        seed = seed_manager.initialize()
        assert seed is not None
    
    # v1.0 Protocol testing
    def test_recorder_protocol(mock_parquet_recorder, performance_monitor):
        if mock_parquet_recorder:
            timing_ctx = performance_monitor.start_timing('record_step')
            mock_parquet_recorder.record_step({'test': 'data'})
            perf_data = performance_monitor.end_timing(timing_ctx)
            assert perf_data['duration_ms'] < 1.0
    
    # Action configuration testing (new export)
    def test_action_config(mock_action_config):
        config = mock_action_config()
        assert config['continuous2d']['max_velocity'] > 0
        assert 'cardinal_discrete' in config

Author: Blitzy Platform v1.0
Version: 1.0.0
"""

import pytest
import numpy as np
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union, Generator, Tuple
from unittest.mock import patch, MagicMock, Mock
import sys
import pathlib
import logging

# Update sys.path for new project structure
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

# Enhanced imports for cookiecutter-based architecture
try:
    # CLI testing infrastructure
    from click.testing import CliRunner
    CLI_TESTING_AVAILABLE = True
except ImportError:
    CLI_TESTING_AVAILABLE = False
    CliRunner = None

try:
    # Hydra configuration testing support
    from hydra import initialize, compose, GlobalHydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict

try:
    # Database session testing infrastructure
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker, Session as SQLASession
    from sqlalchemy.pool import StaticPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    SQLASession = None

try:
    # Environment variable management for testing
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    # v1.0 Protocol-based architecture imports
    from plume_nav_sim.core.protocols import (
        SourceProtocol, BoundaryPolicyProtocol, ActionInterfaceProtocol,
        RecorderProtocol, StatsAggregatorProtocol
    )
    PROTOCOLS_AVAILABLE = True
except Exception as exc:
    PROTOCOLS_AVAILABLE = False
    # Fallback protocol types for testing
    SourceProtocol = object
    BoundaryPolicyProtocol = object
    ActionInterfaceProtocol = object
    RecorderProtocol = object
    StatsAggregatorProtocol = object
    logging.getLogger(__name__).warning(
        "Protocol imports unavailable: %s", exc
    )

try:
    # Recorder backend testing dependencies
    import pandas as pd
    import h5py
    import sqlite3
    import pyarrow.parquet as pq
    RECORDER_BACKENDS_AVAILABLE = True
except ImportError:
    RECORDER_BACKENDS_AVAILABLE = False
    pd = None
    h5py = None
    sqlite3 = None
    pq = None

try:
    # Performance monitoring for recorder testing
    import psutil
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False
    psutil = None

# Core package imports with graceful fallbacks
try:
    from plume_nav_sim.utils.seed_manager import SeedManager, SeedConfig
    SEED_MANAGER_AVAILABLE = True
except Exception as exc:
    SEED_MANAGER_AVAILABLE = False
    SeedManager = None
    SeedConfig = None
    logging.getLogger(__name__).warning(
        "SeedManager unavailable: %s", exc
    )


@pytest.fixture
def mock_video_capture():
    """
    Create a mock for cv2.VideoCapture.
    
    This provides a consistent way to mock video files across tests.
    """
    mock_cap = MagicMock()
    # Set up basic properties that VideoPlume will access
    mock_cap.get.side_effect = lambda prop: {
        0: 640,  # CAP_PROP_FRAME_WIDTH
        1: 480,  # CAP_PROP_FRAME_HEIGHT
        5: 30.0,  # CAP_PROP_FPS
        7: 300,  # CAP_PROP_FRAME_COUNT
    }.get(prop, 0)
    # Configure read to return a frame
    mock_cap.read.return_value = (True, np.zeros((480, 640), dtype=np.uint8))
    # Ensure isOpened returns True
    mock_cap.isOpened.return_value = True
    return mock_cap


@pytest.fixture
def mock_video_plume():
    """
    Create a mock VideoPlume instance.
    
    This provides a consistent mock for tests that use VideoPlume.
    """
    mock = MagicMock()
    # Configure mock properties
    mock.width = 640
    mock.height = 480
    mock.fps = 30.0
    mock.frame_count = 300
    mock.duration = 10.0
    mock.shape = (480, 640)
    # Configure return values for methods
    mock.get_frame.return_value = np.zeros((480, 640), dtype=np.uint8)
    mock.get_metadata.return_value = {
        "width": 640,
        "height": 480,
        "fps": 30.0,
        "frame_count": 300,
        "duration": 10.0,
        "shape": (480, 640)
    }
    return mock


@pytest.fixture
def mock_navigator():
    """
    Create a mock Navigator instance compatible with NavigatorProtocol.
    
    This provides a consistent mock for tests that use the NavigatorProtocol
    interface from the new project structure.
    """
    mock = MagicMock()
    
    # Configure NavigatorProtocol properties as numpy arrays
    mock.positions = np.array([[0.0, 0.0]])  # Single agent at origin
    mock.orientations = np.array([0.0])      # Facing east (0 degrees)
    mock.speeds = np.array([0.0])            # Initially at rest
    mock.max_speeds = np.array([1.0])        # Maximum speed limit
    mock.angular_velocities = np.array([0.0]) # No initial rotation
    mock.num_agents = 1                      # Single agent navigator
    
    # Configure NavigatorProtocol methods
    mock.reset.return_value = None
    mock.step.return_value = None
    mock.sample_odor.return_value = 0.0
    mock.read_single_antenna_odor.return_value = 0.0
    mock.sample_multiple_sensors.return_value = np.array([0.0, 0.0])
    
    return mock


@pytest.fixture
def mock_exists():
    """Mock Path.exists to always return True."""
    with patch('pathlib.Path.exists', return_value=True):
        yield


@pytest.fixture
def mock_cli_runner() -> Generator[Tuple[Optional[CliRunner], Dict[str, str]], None, None]:
    """
    Provide Click CliRunner for command-line interface testing per Section 6.6.1.1.
    
    Creates an isolated CLI testing environment with CliRunner instance and
    clean environment variables for comprehensive command-line interface validation.
    Includes Hydra working directory isolation and environment variable management
    for deterministic CLI testing across all command scenarios.
    
    Usage Examples:
        def test_cli_command_execution(mock_cli_runner):
            runner, env = mock_cli_runner
            result = runner.invoke(main, ['run', '--dry-run'])
            assert result.exit_code == 0
            
        def test_cli_parameter_validation(mock_cli_runner):
            runner, env = mock_cli_runner
            result = runner.invoke(main, ['config', 'validate'])
            assert 'validation' in result.output.lower()
    
    Returns:
        Tuple containing:
        - CliRunner instance (or None if click.testing not available)
        - Environment variables dictionary with isolated testing context
        
    Features:
        - Isolated temporary working directory for each test
        - Clean environment variable state preventing test pollution
        - Hydra working directory management for configuration isolation
        - Automatic cleanup of CLI artifacts and temporary files
    """
    if not CLI_TESTING_AVAILABLE:
        yield None, {}
        return
    
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up isolated environment for CLI testing
        env = os.environ.copy()
        env['HYDRA_WORKING_DIR'] = temp_dir
        env['PYTHONPATH'] = str(Path(__file__).parent.parent / "src")
        
        # Isolate environment variables for clean testing
        test_env_vars = {
            'TEST_MODE': 'true',
            'LOG_LEVEL': 'ERROR',  # Suppress verbose logging in tests
            'MATPLOTLIB_BACKEND': 'Agg',  # Headless matplotlib for CI
        }
        
        # Apply test environment
        for key, value in test_env_vars.items():
            env[key] = value
        
        yield runner, env


@pytest.fixture
def mock_hydra_config() -> Generator[Optional[DictConfig], None, None]:
    """
    Provide Hydra configuration composition testing with pytest-hydra integration.
    
    Creates a complete Hydra configuration composition for testing hierarchical
    parameter management, override scenarios, and schema validation. Integrates
    with pytest-hydra plugin patterns for comprehensive configuration testing
    across conf/base.yaml, conf/config.yaml, and conf/local/ structures.
    
    Usage Examples:
        def test_config_composition(mock_hydra_config):
            assert mock_hydra_config.navigator.max_speed > 0
            assert hasattr(mock_hydra_config, 'video_plume')
            
        def test_config_override_scenarios(mock_hydra_config):
            # Test parameter override validation
            config = mock_hydra_config
            assert config.simulation.fps >= 30
    
    Returns:
        DictConfig with complete hierarchical configuration or None if Hydra unavailable
        
    Features:
        - Hierarchical configuration composition from base, config, and local layers
        - Environment variable interpolation testing with ${oc.env:VAR_NAME} patterns
        - Schema validation integration with Pydantic models
        - Working directory isolation preventing configuration pollution
        - Override scenario testing with parameter validation
    """
    if not HYDRA_AVAILABLE:
        yield None
        return
    
    # Create mock configuration structure with v1.0 protocol support
    config_dict = {
        'navigator': {
            'type': 'single',
            'max_speed': 2.0,
            'num_agents': 1,
            'orientation': 0.0,
            'speed': 0.0
        },
        'video_plume': {
            'flip': False,
            'kernel_size': 3,
            'kernel_sigma': 1.0,
            'video_path': 'test_video.mp4'
        },
        'simulation': {
            'fps': 30,
            'max_duration': 10.0,
            'step_size': 0.1
        },
        # v1.0 Protocol-based configurations
        'source': {
            '_target_': 'plume_nav_sim.core.sources.PointSource',
            'position': [50.0, 50.0],
            'emission_rate': 1000.0,
            'active': True
        },
        'boundary': {
            '_target_': 'plume_nav_sim.core.boundaries.TerminateBoundary',
            'domain_bounds': [[0.0, 100.0], [0.0, 100.0]],
            'status_on_violation': 'oob'
        },
        'action': {
            '_target_': 'plume_nav_sim.core.actions.Continuous2DAction',
            'max_velocity': 2.0,
            'max_angular_velocity': 3.14159,
            'normalize_actions': True,
            'clip_actions': True
        },
        'record': {
            '_target_': 'plume_nav_sim.recording.backends.NullRecorder',
            'log_calls': False,
            'validate_interface': True,
            'full': False,
            'overhead_threshold_ms': 1.0
        },
        'hooks': {
            'extra_obs_fn': None,
            'extra_reward_fn': None,
            'episode_end_fn': None,
            'enabled': False  # Disabled for testing performance
        },
        'visualization': {
            'animation': {
                'enabled': False,  # Disabled for testing
                'save_path': '${hydra:runtime.output_dir}/animation.mp4'
            },
            'plotting': {
                'show_trails': True,
                'trail_length': 50
            }
        },
        'database': {
            'enabled': False,  # Disabled by default for testing
            'url': 'sqlite:///:memory:',
            'pool_size': 1
        },
        'reproducibility': {
            'global_seed': 42,
            'auto_seed': False,
            'validate_initialization': True
        },
        'logging': {
            'level': 'ERROR',  # Suppress logs in testing
            'format': 'simple'
        }
    }
    
    # Create DictConfig from mock data
    mock_config = OmegaConf.create(config_dict)
    
    # Ensure config is read-only for test safety
    OmegaConf.set_readonly(mock_config, True)
    
    yield mock_config


@pytest.fixture
def mock_db_session() -> Generator[Optional[SQLASession], None, None]:
    """
    Provide in-memory SQLAlchemy session for database testing per Section 6.6.5.4.
    
    Creates an isolated in-memory SQLite database session for comprehensive
    database testing without external dependencies. Supports transaction testing,
    rollback scenarios, and session lifecycle validation with automatic cleanup
    and state isolation between tests.
    
    Usage Examples:
        def test_database_operations(mock_db_session):
            if mock_db_session:
                # Database operations with transaction support
                result = mock_db_session.execute(text("SELECT 1"))
                assert result.scalar() == 1
                
        def test_session_lifecycle(mock_db_session):
            if mock_db_session:
                # Test session management patterns
                assert mock_db_session.is_active
    
    Returns:
        SQLAlchemy Session instance or None if SQLAlchemy not available
        
    Features:
        - In-memory SQLite database for isolation and speed
        - Automatic transaction rollback for test independence
        - Session lifecycle management with proper cleanup
        - Connection pooling configuration for testing scenarios
        - Support for database schema creation and teardown
    """
    if not SQLALCHEMY_AVAILABLE:
        yield None
        return
    
    # Create in-memory SQLite database for testing
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,  # Suppress SQL logging in tests
        poolclass=StaticPool,
        connect_args={
            'check_same_thread': False,
            'timeout': 20
        }
    )
    
    # Create session factory
    SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=True,
        expire_on_commit=False  # Keep objects accessible after commit
    )
    
    # Create session instance
    session = SessionLocal()
    
    try:
        # Test basic database connectivity
        session.execute(text("SELECT 1"))
        yield session
    except Exception:
        # If session creation fails, yield None for graceful degradation
        yield None
    finally:
        # Ensure proper cleanup
        if session:
            session.rollback()  # Rollback any uncommitted changes
            session.close()     # Close session
        engine.dispose()        # Dispose of engine and connections


@pytest.fixture
def mock_seed_manager() -> Generator[Optional[object], None, None]:
    """
    Provide controlled random seed management testing per Feature F-014.
    
    Creates a SeedManager instance for testing reproducible random state management
    across scientific computing workflows. Supports seed initialization validation,
    state preservation testing, and deterministic behavior verification essential
    for research reproducibility and experiment validation.
    
    Usage Examples:
        def test_seed_initialization(mock_seed_manager):
            seed_manager = mock_seed_manager
            seed = seed_manager.initialize({'seed': 42})
            assert seed == 42
            assert seed_manager.current_seed == 42
            
        def test_reproducible_behavior(mock_seed_manager):
            seed_manager = mock_seed_manager
            seed_manager.initialize({'seed': 123})
            
            # Test deterministic random generation
            import numpy as np
            first_random = np.random.random()
            seed_manager.initialize({'seed': 123})
            second_random = np.random.random()
            assert first_random == second_random
    
    Returns:
        SeedManager instance or mock object if SeedManager not available
        
    Features:
        - Deterministic seed initialization for reproducible testing
        - State preservation and restoration capabilities
        - Cross-platform random state consistency validation
        - Integration with NumPy and Python random modules
        - Automatic cleanup and state isolation between tests
    """
    if not SEED_MANAGER_AVAILABLE:
        # Create a mock seed manager if the real one isn't available
        mock = MagicMock()
        mock.current_seed = 42
        mock.initialize.return_value = 42
        mock.get_state.return_value = {'seed': 42, 'timestamp': time.time()}
        mock.restore_state.return_value = None
        yield mock
        return
    
    # Create a fresh SeedManager instance for testing
    # Reset any existing singleton state
    SeedManager.reset()
    seed_manager = SeedManager()
    
    try:
        # Initialize with test configuration
        test_config = SeedConfig(
            seed=42,
            auto_seed=False,
            validate_initialization=True,
            preserve_state=True,
            log_seed_context=False  # Disable logging in tests
        )
        
        # Initialize the seed manager
        seed_manager.initialize(config=test_config)
        
        yield seed_manager
        
    finally:
        # Clean up seed manager state
        SeedManager.reset()


@pytest.fixture
def config_files():
    """
    Fixture to provide test configuration data compatible with v1.0 Hydra structure.
    
    Returns a dictionary with hierarchical test configurations following the
    enhanced v1.0 protocol-based architecture with conf/base/ group structure
    and comprehensive parameter coverage for all system components including
    new source, boundary, action, record, and hooks configurations.
    """
    # Base configuration (conf/base.yaml equivalent) - v1.0 structure
    base_config = {
        "defaults": [
            "_self_",
            "navigator: single",
            "video_plume: default",
            "simulation: default",
            "source: point",
            "boundary: terminate", 
            "action: continuous2d",
            "record: parquet",
            "hooks: default"
        ],
        "navigator": {
            "type": "single",
            "orientation": 0.0,
            "speed": 0.0,
            "max_speed": 1.0,
            "num_agents": 1
        },
        "video_plume": {
            "flip": False,
            "kernel_size": 0,
            "kernel_sigma": 1.0,
            "video_path": "test_video.mp4"
        },
        "simulation": {
            "fps": 30,
            "max_duration": 10.0,
            "step_size": 0.1
        },
        # v1.0 Protocol-based configurations
        "source": {
            "_target_": "plume_nav_sim.core.sources.PointSource",
            "position": [50.0, 50.0],
            "emission_rate": 1000.0,
            "active": True
        },
        "boundary": {
            "_target_": "plume_nav_sim.core.boundaries.TerminateBoundary",
            "domain_bounds": [[0.0, 100.0], [0.0, 100.0]],
            "status_on_violation": "oob"
        },
        "action": {
            "_target_": "plume_nav_sim.core.actions.Continuous2DAction",
            "max_velocity": 2.0,
            "max_angular_velocity": 3.14159,
            "normalize_actions": True
        },
        "record": {
            "_target_": "plume_nav_sim.recording.backends.ParquetRecorder",
            "output_dir": "${hydra:runtime.output_dir}/recordings",
            "compression": "snappy",
            "buffer_size": 1000,
            "full": True
        },
        "hooks": {
            "extra_obs_fn": None,
            "extra_reward_fn": None,
            "episode_end_fn": None,
            "enabled": True
        },
        "database": {
            "enabled": False,
            "url": "sqlite:///:memory:",
            "pool_size": 1
        },
        "reproducibility": {
            "global_seed": None,
            "auto_seed": True,
            "validate_initialization": False
        },
        "logging": {
            "level": "INFO",
            "format": "detailed"
        }
    }
    
    # User configuration with overrides (conf/config.yaml equivalent) - v1.0 structure
    user_config = {
        "navigator": {
            "orientation": 45.0,
            "speed": 0.5,
            "max_speed": 2.0
        },
        "video_plume": {
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 2.0
        },
        "simulation": {
            "fps": 60,
            "max_duration": 20.0
        },
        # v1.0 Protocol-based overrides
        "source": {
            "_target_": "plume_nav_sim.core.sources.MultiSource",
            "positions": [[25.0, 25.0], [75.0, 75.0]],
            "emission_rates": [800.0, 1200.0],
            "active": True
        },
        "boundary": {
            "_target_": "plume_nav_sim.core.boundaries.BounceBoundary",
            "domain_bounds": [[0.0, 150.0], [0.0, 150.0]],
            "restitution": 0.8
        },
        "action": {
            "_target_": "plume_nav_sim.core.actions.CardinalDiscreteAction",
            "step_size": 0.2,
            "allowed_directions": ["north", "south", "east", "west", "stay"]
        },
        "record": {
            "_target_": "plume_nav_sim.recording.backends.HDF5Recorder",
            "output_dir": "${hydra:runtime.output_dir}/recordings",
            "compression": "gzip",
            "compression_level": 6,
            "full": True
        },
        "hooks": {
            "extra_obs_fn": "custom_observation_function",
            "extra_reward_fn": "custom_reward_shaping",
            "episode_end_fn": "custom_episode_summary",
            "enabled": True
        },
        "reproducibility": {
            "global_seed": 42,
            "auto_seed": False,
            "validate_initialization": True
        },
        "logging": {
            "level": "DEBUG"
        }
    }
    
    # Local overrides (conf/local/ equivalent) - v1.0 testing overrides
    local_config = {
        "database": {
            "enabled": True,
            "url": "sqlite:///test.db"
        },
        # v1.0 Testing-specific overrides
        "record": {
            "_target_": "plume_nav_sim.recording.backends.NullRecorder",
            "log_calls": False,
            "validate_interface": True
        },
        "hooks": {
            "enabled": False  # Disable hooks for performance testing
        },
        "source": {
            "emission_rate": 500.0  # Reduced for testing
        },
        "boundary": {
            "domain_bounds": [[0.0, 50.0], [0.0, 50.0]]  # Smaller domain for testing
        },
        "logging": {
            "level": "ERROR"  # Suppress logs in testing
        }
    }
    
    # v1.0 Protocol-specific configurations for testing
    v1_protocol_configs = {
        "source_configs": {
            "point": {
                "_target_": "plume_nav_sim.core.sources.PointSource",
                "position": [50.0, 50.0],
                "emission_rate": 1000.0
            },
            "multi": {
                "_target_": "plume_nav_sim.core.sources.MultiSource",
                "positions": [[25.0, 25.0], [75.0, 75.0]],
                "emission_rates": [800.0, 1200.0]
            },
            "dynamic": {
                "_target_": "plume_nav_sim.core.sources.DynamicSource",
                "initial_position": [50.0, 50.0],
                "velocity": [1.0, 0.5],
                "emission_pattern": "variable"
            }
        },
        "boundary_configs": {
            "terminate": {
                "_target_": "plume_nav_sim.core.boundaries.TerminateBoundary",
                "status_on_violation": "oob"
            },
            "bounce": {
                "_target_": "plume_nav_sim.core.boundaries.BounceBoundary",
                "restitution": 0.8
            },
            "wrap": {
                "_target_": "plume_nav_sim.core.boundaries.WrapBoundary"
            },
            "clip": {
                "_target_": "plume_nav_sim.core.boundaries.ClipBoundary"
            }
        },
        "action_configs": {
            "continuous2d": {
                "_target_": "plume_nav_sim.core.actions.Continuous2DAction",
                "max_velocity": 2.0,
                "normalize_actions": True
            },
            "cardinal_discrete": {
                "_target_": "plume_nav_sim.core.actions.CardinalDiscreteAction",
                "step_size": 0.1,
                "allowed_directions": ["north", "south", "east", "west", "stay"]
            }
        },
        "recorder_configs": {
            "parquet": {
                "_target_": "plume_nav_sim.recording.backends.ParquetRecorder",
                "compression": "snappy",
                "buffer_size": 1000
            },
            "hdf5": {
                "_target_": "plume_nav_sim.recording.backends.HDF5Recorder",
                "compression": "gzip",
                "compression_level": 6
            },
            "sqlite": {
                "_target_": "plume_nav_sim.recording.backends.SQLiteRecorder",
                "transaction_size": 50
            },
            "none": {
                "_target_": "plume_nav_sim.recording.backends.NullRecorder",
                "log_calls": False
            }
        }
    }
    
    return {
        "base_config": base_config,
        "user_config": user_config,
        "local_config": local_config,
        "v1_protocol_configs": v1_protocol_configs,
        # Legacy support
        "default_config": base_config,
    }


@pytest.fixture
def temp_config_files(tmp_path, config_files):
    """
    Create temporary Hydra-compatible configuration files for testing.
    
    Creates a complete Hydra configuration directory structure with conf/base.yaml,
    conf/config.yaml, and conf/local/ files for comprehensive configuration testing.
    Supports hierarchical configuration composition and override scenario validation.
    
    Args:
        tmp_path: Pytest built-in fixture that provides a temporary directory
        config_files: Configuration data fixture with hierarchical structure
    
    Returns:
        Dictionary containing paths to created files and configuration objects
        
    Features:
        - Complete Hydra conf/ directory structure creation
        - Hierarchical configuration file generation with proper YAML formatting
        - Local override files with credential template patterns  
        - Environment variable interpolation examples
        - Automatic cleanup through pytest tmp_path management
    """
    import yaml
    
    # Get configurations from fixture
    configs = config_files
    
    # Create Hydra-compatible conf directory structure
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()
    
    # Create local subdirectory for user-specific overrides
    local_dir = conf_dir / "local"
    local_dir.mkdir()
    
    # Create base.yaml (default parameters)
    base_path = conf_dir / "base.yaml"
    with open(base_path, 'w') as f:
        yaml.dump(configs["base_config"], f, default_flow_style=False)
    
    # Create config.yaml (main configuration)
    config_path = conf_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(configs["user_config"], f, default_flow_style=False)
    
    # Create local configuration files
    local_config_path = local_dir / "config.yaml"
    with open(local_config_path, 'w') as f:
        yaml.dump(configs["local_config"], f, default_flow_style=False)
    
    # Create credential template for testing
    credentials_template_path = local_dir / "credentials.yaml.template"
    credentials_template = {
        "database": {
            "password": "${oc.env:DB_PASSWORD,null}",
            "username": "${oc.env:DB_USERNAME,testuser}"
        },
        "api": {
            "secret_key": "${oc.env:API_SECRET_KEY,test_secret}",
            "token": "${oc.env:API_TOKEN,test_token}"
        }
    }
    with open(credentials_template_path, 'w') as f:
        yaml.dump(credentials_template, f, default_flow_style=False)
    
    # Create paths template for testing
    paths_template_path = local_dir / "paths.yaml.template"
    paths_template = {
        "environment": {
            "paths": {
                "data_dir": "${oc.env:DATA_DIR,/tmp/data}",
                "output_dir": "${oc.env:OUTPUT_DIR,/tmp/output}",
                "cache_dir": "${oc.env:CACHE_DIR,/tmp/cache}"
            }
        }
    }
    with open(paths_template_path, 'w') as f:
        yaml.dump(paths_template, f, default_flow_style=False)
    
    return {
        "conf_dir": conf_dir,
        "local_dir": local_dir,
        "base_path": base_path,
        "config_path": config_path,
        "local_config_path": local_config_path,
        "credentials_template_path": credentials_template_path,
        "paths_template_path": paths_template_path,
        "base_config": configs["base_config"],
        "user_config": configs["user_config"],
        "local_config": configs["local_config"],
        # Legacy support
        "config_dir": conf_dir,
        "default_path": base_path,
        "user_path": config_path,
        "default_config": configs["base_config"]
    }


@pytest.fixture
def isolated_environment():
    """
    Provide environment variable isolation and Hydra working directory management.
    
    Creates a completely isolated testing environment with clean environment variables,
    isolated working directory, and proper cleanup for deterministic test execution.
    Integrates with Hydra working directory management to prevent configuration
    pollution between test runs.
    
    Usage Examples:
        def test_with_isolation(isolated_environment):
            env_vars, working_dir = isolated_environment
            # Test runs in completely isolated environment
            assert env_vars['TEST_MODE'] == 'true'
            assert Path(working_dir).exists()
    
    Returns:
        Tuple containing:
        - Dictionary of isolated environment variables
        - Path to isolated working directory
        
    Features:
        - Complete environment variable isolation
        - Hydra working directory management
        - Automatic cleanup of temporary files and directories
        - Cross-platform compatibility
        - Integration with pytest fixtures
    """
    # Store original environment
    original_env = os.environ.copy()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up isolated environment
        isolated_env = {
            'TEST_MODE': 'true',
            'PYTHONPATH': str(Path(__file__).parent.parent / "src"),
            'HYDRA_WORKING_DIR': temp_dir,
            'MATPLOTLIB_BACKEND': 'Agg',
            'LOG_LEVEL': 'ERROR',
            'NUMBA_DISABLE_JIT': '1',  # Disable JIT for faster testing
        }
        
        # Apply isolated environment
        os.environ.clear()
        os.environ.update(isolated_env)
        
        try:
            yield isolated_env, temp_dir
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)


@pytest.fixture
def mock_action_config():
    """
    Create mock action configuration for v1.0 ActionInterface testing.
    
    Provides comprehensive action configuration fixtures supporting both
    Continuous2D and CardinalDiscrete action interfaces for protocol-based
    architecture testing. Includes validation settings and action space
    bounds for comprehensive ActionInterfaceProtocol testing.
    
    Returns:
        Dictionary containing action configuration parameters for testing
        
    Features:
        - Support for both continuous and discrete action spaces
        - Configurable action bounds and validation parameters
        - Protocol-compliant configuration structure
        - Integration with Hydra action config group testing
    """
    return {
        'continuous2d': {
            '_target_': 'plume_nav_sim.core.actions.Continuous2DAction',
            'max_velocity': 2.0,
            'max_angular_velocity': np.pi,
            'velocity_bounds': [-2.0, 2.0],
            'angular_bounds': [-np.pi, np.pi],
            'action_scale': 1.0,
            'normalize_actions': True,
            'clip_actions': True
        },
        'cardinal_discrete': {
            '_target_': 'plume_nav_sim.core.actions.CardinalDiscreteAction',
            'step_size': 0.1,
            'allowed_directions': ['north', 'south', 'east', 'west', 'stay'],
            'diagonal_actions': False,
            'action_mapping': {
                0: 'stay',
                1: 'north', 
                2: 'south',
                3: 'east',
                4: 'west'
            }
        },
        'validation': {
            'validate_bounds': True,
            'strict_validation': False,
            'error_on_invalid': True,
            'clip_invalid': True
        }
    }


@pytest.fixture 
def mock_recorder_config():
    """
    Create comprehensive recorder configuration for RecorderProtocol testing.
    
    Provides complete recorder configuration fixtures supporting all backend
    types (parquet, hdf5, sqlite, none) with performance monitoring,
    compression settings, and structured output organization per F-017
    requirements for comprehensive recorder testing.
    
    Returns:
        Dictionary containing recorder configuration parameters for all backends
        
    Features:
        - Configuration for all recorder backends with optimal settings
        - Performance monitoring and buffering configuration
        - Compression testing with multiple algorithms and levels
        - Structured output directory organization with run_id/episode_id hierarchy
        - Backend availability detection for graceful degradation testing
    """
    return {
        'parquet': {
            '_target_': 'plume_nav_sim.recording.backends.ParquetRecorder',
            'output_dir': '${hydra:runtime.output_dir}/recordings',
            'compression': 'snappy',
            'compression_level': 3,
            'buffer_size': 1000,
            'flush_interval': 10.0,
            'schema_validation': True,
            'partition_cols': ['run_id', 'episode_id'],
            'metadata_tracking': True
        },
        'hdf5': {
            '_target_': 'plume_nav_sim.recording.backends.HDF5Recorder',
            'output_dir': '${hydra:runtime.output_dir}/recordings', 
            'compression': 'gzip',
            'compression_level': 6,
            'buffer_size': 500,
            'flush_interval': 5.0,
            'chunking': True,
            'chunk_size': [100, 10],
            'track_order': True,
            'metadata_tracking': True
        },
        'sqlite': {
            '_target_': 'plume_nav_sim.recording.backends.SQLiteRecorder',
            'output_dir': '${hydra:runtime.output_dir}/recordings',
            'database_file': 'experiment_data.db',
            'buffer_size': 100,
            'flush_interval': 2.0,
            'transaction_size': 50,
            'pragma_settings': {
                'journal_mode': 'WAL',
                'synchronous': 'NORMAL',
                'cache_size': 10000
            },
            'metadata_tracking': True
        },
        'none': {
            '_target_': 'plume_nav_sim.recording.backends.NullRecorder',
            'log_calls': False,
            'validate_interface': True
        },
        'performance': {
            'overhead_threshold_ms': 1.0,
            'memory_limit_mb': 100,
            'buffer_monitoring': True,
            'timing_precision': 'microsecond'
        },
        'output_structure': {
            'use_run_hierarchy': True,
            'episode_subdirs': True,
            'metadata_files': True,
            'summary_generation': True
        }
    }


@pytest.fixture
def mock_parquet_recorder(tmp_path, mock_recorder_config):
    """
    Create mock Parquet recorder for comprehensive backend testing.
    
    Provides a fully configured mock ParquetRecorder instance with buffering,
    compression, and performance monitoring capabilities for testing the
    RecorderProtocol implementation with Parquet backend specifics.
    
    Args:
        tmp_path: Pytest temporary directory for isolated file operations
        mock_recorder_config: Recorder configuration fixture
        
    Returns:
        Mock ParquetRecorder instance or None if dependencies unavailable
        
    Features:
        - Parquet-specific compression and serialization testing
        - Buffered I/O with configurable flush intervals
        - Schema validation and metadata tracking
        - Performance overhead monitoring with timing precision
    """
    if not RECORDER_BACKENDS_AVAILABLE or pd is None or pq is None:
        return None
        
    mock = MagicMock()
    config = mock_recorder_config['parquet']
    
    # Configure ParquetRecorder-specific properties
    mock.output_dir = tmp_path / "recordings"
    mock.compression = config['compression']
    mock.compression_level = config['compression_level']
    mock.buffer_size = config['buffer_size']
    mock.flush_interval = config['flush_interval']
    mock.schema_validation = config['schema_validation']
    
    # Create output directory structure
    mock.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure RecorderProtocol methods
    mock.record_step.return_value = None
    mock.record_episode.return_value = None
    mock.flush_buffer.return_value = None
    mock.close.return_value = None
    mock.get_statistics.return_value = {
        'total_steps_recorded': 0,
        'total_episodes_recorded': 0,
        'buffer_usage_percent': 0.0,
        'compression_ratio': 1.0,
        'average_write_time_ms': 0.5
    }
    
    # Configure compression testing
    mock.test_compression.return_value = {
        'algorithm': config['compression'],
        'level': config['compression_level'],
        'original_size_mb': 10.0,
        'compressed_size_mb': 3.2,
        'compression_ratio': 3.125,
        'compression_time_ms': 15.2
    }

    # Ensure export_data behaves as successful operation for all formats
    mock.export_data.return_value = True
    
    return mock


@pytest.fixture
def mock_hdf5_recorder(tmp_path, mock_recorder_config):
    """
    Create mock HDF5 recorder for hierarchical data testing.
    
    Provides a fully configured mock HDF5Recorder instance with chunking,
    compression, and hierarchical data organization capabilities for testing
    the RecorderProtocol implementation with HDF5 backend specifics.
    
    Args:
        tmp_path: Pytest temporary directory for isolated file operations
        mock_recorder_config: Recorder configuration fixture
        
    Returns:
        Mock HDF5Recorder instance or None if dependencies unavailable
        
    Features:
        - HDF5-specific chunking and hierarchical data organization
        - Advanced compression algorithms with performance monitoring
        - Metadata tracking and schema evolution support
        - Memory-efficient streaming for large datasets
    """
    if not RECORDER_BACKENDS_AVAILABLE or h5py is None:
        return None
        
    mock = MagicMock()
    config = mock_recorder_config['hdf5']
    
    # Configure HDF5Recorder-specific properties
    mock.output_dir = tmp_path / "recordings"
    mock.compression = config['compression']
    mock.compression_level = config['compression_level']
    mock.chunking = config['chunking']
    mock.chunk_size = config['chunk_size']
    mock.track_order = config['track_order']
    
    # Create output directory structure
    mock.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure RecorderProtocol methods
    mock.record_step.return_value = None
    mock.record_episode.return_value = None
    mock.flush_buffer.return_value = None
    mock.close.return_value = None
    mock.get_statistics.return_value = {
        'total_steps_recorded': 0,
        'total_episodes_recorded': 0,
        'buffer_usage_percent': 0.0,
        'compression_ratio': 1.5,
        'average_write_time_ms': 0.8,
        'dataset_size_mb': 25.6,
        'chunk_cache_hits': 0.95
    }
    
    # Configure HDF5-specific testing methods
    mock.create_dataset.return_value = None
    mock.create_group.return_value = None
    mock.list_datasets.return_value = [
        '/episodes/episode_001/positions',
        '/episodes/episode_001/actions', 
        '/episodes/episode_001/rewards',
        '/metadata/configuration',
        '/metadata/statistics'
    ]
    
    return mock


@pytest.fixture
def mock_sqlite_recorder(tmp_path, mock_recorder_config):
    """
    Create mock SQLite recorder for relational data testing.
    
    Provides a fully configured mock SQLiteRecorder instance with transaction
    management, indexing, and query capabilities for testing the RecorderProtocol
    implementation with SQLite backend specifics.
    
    Args:
        tmp_path: Pytest temporary directory for isolated database operations
        mock_recorder_config: Recorder configuration fixture
        
    Returns:
        Mock SQLiteRecorder instance or None if dependencies unavailable
        
    Features:
        - SQLite-specific transaction management and optimization
        - Index creation and query performance testing
        - WAL mode and pragma configuration testing
        - Concurrent access and locking behavior validation
    """
    if not RECORDER_BACKENDS_AVAILABLE or sqlite3 is None:
        return None
        
    mock = MagicMock()
    config = mock_recorder_config['sqlite']
    
    # Configure SQLiteRecorder-specific properties
    mock.output_dir = tmp_path / "recordings"
    mock.database_file = config['database_file']
    mock.transaction_size = config['transaction_size']
    mock.pragma_settings = config['pragma_settings']
    
    # Create output directory structure
    mock.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure RecorderProtocol methods
    mock.record_step.return_value = None
    mock.record_episode.return_value = None
    mock.flush_buffer.return_value = None
    mock.close.return_value = None
    mock.get_statistics.return_value = {
        'total_steps_recorded': 0,
        'total_episodes_recorded': 0,
        'buffer_usage_percent': 0.0,
        'database_size_mb': 5.2,
        'average_write_time_ms': 0.3,
        'transaction_count': 0,
        'index_usage_percent': 0.85
    }
    
    # Configure SQLite-specific testing methods
    mock.execute_query.return_value = []
    mock.create_table.return_value = None
    mock.create_index.return_value = None
    mock.get_schema.return_value = {
        'steps': ['id', 'episode_id', 'step_num', 'position_x', 'position_y', 'action', 'reward', 'timestamp'],
        'episodes': ['id', 'run_id', 'episode_num', 'total_steps', 'total_reward', 'duration_sec'],
        'metadata': ['key', 'value', 'episode_id', 'timestamp']
    }
    
    return mock


@pytest.fixture
def mock_null_recorder(mock_recorder_config):
    """
    Create mock null recorder for performance baseline testing.
    
    Provides a mock NullRecorder instance that implements RecorderProtocol
    without performing actual I/O operations, serving as a performance
    baseline and interface compliance testing tool.
    
    Args:
        mock_recorder_config: Recorder configuration fixture
        
    Returns:
        Mock NullRecorder instance for performance baseline testing
        
    Features:
        - Zero-overhead recording for performance baseline establishment
        - Interface compliance validation without I/O operations
        - Call logging and validation for testing hook integration
        - Memory usage monitoring for overhead detection
    """
    mock = MagicMock()
    config = mock_recorder_config['none']
    
    # Configure NullRecorder-specific properties
    mock.log_calls = config['log_calls']
    mock.validate_interface = config['validate_interface']
    mock.call_count = 0
    mock.interface_violations = []
    
    # Configure RecorderProtocol methods with call tracking
    def track_call(method_name):
        def wrapper(*args, **kwargs):
            mock.call_count += 1
            if mock.log_calls:
                print(f"NullRecorder.{method_name} called with args={args}, kwargs={kwargs}")
            return None
        return wrapper
    
    mock.record_step = track_call('record_step')
    mock.record_episode = track_call('record_episode')
    mock.flush_buffer = track_call('flush_buffer')
    mock.close = track_call('close')
    mock.get_statistics.return_value = {
        'total_steps_recorded': mock.call_count,
        'total_episodes_recorded': 0,
        'buffer_usage_percent': 0.0,
        'average_write_time_ms': 0.0,
        'interface_violations': len(mock.interface_violations)
    }
    
    return mock


@pytest.fixture
def performance_monitor():
    """
    Create performance monitoring utilities for recorder overhead validation.
    
    Provides comprehensive performance monitoring tools for testing recorder
    backend overhead, timing precision, memory usage, and I/O performance
    characteristics essential for maintaining ≤33ms step latency requirements.
    
    Returns:
        Dictionary containing performance monitoring utilities and thresholds
        
    Features:
        - High-precision timing measurement utilities (microsecond precision)
        - Memory usage monitoring with configurable thresholds
        - I/O performance measurement and buffering analysis
        - Overhead detection and automatic performance regression testing
    """
    class PerformanceMonitor:
        def __init__(self):
            self.timing_data = []
            self.memory_snapshots = []
            self.overhead_threshold_ms = 1.0
            self.memory_limit_mb = 100
            
        def start_timing(self, operation_name: str):
            """Start timing for a specific operation."""
            start_time = time.perf_counter()
            return {
                'operation': operation_name,
                'start_time': start_time,
                'start_memory': self._get_memory_usage()
            }
            
        def end_timing(self, timing_context: dict):
            """End timing and record performance data."""
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            duration_ms = (end_time - timing_context['start_time']) * 1000
            memory_delta_mb = end_memory - timing_context['start_memory']
            
            performance_data = {
                'operation': timing_context['operation'],
                'duration_ms': duration_ms,
                'memory_delta_mb': memory_delta_mb,
                'overhead_violation': duration_ms > self.overhead_threshold_ms,
                'memory_violation': memory_delta_mb > self.memory_limit_mb,
                'timestamp': time.time()
            }
            
            self.timing_data.append(performance_data)
            return performance_data
            
        def _get_memory_usage(self):
            """Get current memory usage in MB."""
            if PERFORMANCE_MONITORING_AVAILABLE and psutil:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            return 0.0
            
        def get_summary_stats(self):
            """Calculate summary performance statistics."""
            if not self.timing_data:
                return {'no_data': True}
                
            durations = [d['duration_ms'] for d in self.timing_data]
            memory_deltas = [d['memory_delta_mb'] for d in self.timing_data]
            
            return {
                'total_operations': len(self.timing_data),
                'avg_duration_ms': np.mean(durations),
                'max_duration_ms': np.max(durations),
                'min_duration_ms': np.min(durations),
                'std_duration_ms': np.std(durations),
                'avg_memory_delta_mb': np.mean(memory_deltas),
                'max_memory_delta_mb': np.max(memory_deltas),
                'overhead_violations': sum(1 for d in self.timing_data if d['overhead_violation']),
                'memory_violations': sum(1 for d in self.timing_data if d['memory_violation']),
                'performance_grade': 'PASS' if all(not d['overhead_violation'] for d in self.timing_data) else 'FAIL'
            }
            
        def reset(self):
            """Reset all collected performance data."""
            self.timing_data.clear()
            self.memory_snapshots.clear()
    
    return PerformanceMonitor()


@pytest.fixture 
def structured_output_fixture(tmp_path):
    """
    Create structured output directory fixtures for testing run_id/episode_id hierarchy.
    
    Provides comprehensive directory structure testing utilities that validate
    the F-017 requirement for organized data output with run_id/episode_id
    hierarchy, metadata tracking, and proper file organization.
    
    Args:
        tmp_path: Pytest temporary directory for isolated file system testing
        
    Returns:
        Dictionary containing structured output testing utilities and paths
        
    Features:
        - Run/episode hierarchy creation and validation
        - Metadata file organization and schema testing
        - Directory permission and access testing
        - Path generation utilities for consistent naming
    """
    def create_run_structure(run_id: str, num_episodes: int = 3):
        """Create a complete run directory structure."""
        run_dir = tmp_path / f"run_{run_id}"
        run_dir.mkdir(exist_ok=True)
        
        # Create metadata files
        metadata_dir = run_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # Create run-level metadata
        run_metadata = {
            'run_id': run_id,
            'start_time': time.time(),
            'configuration': {'test': True},
            'total_episodes': num_episodes
        }
        
        import json
        with open(metadata_dir / "run_config.json", 'w') as f:
            json.dump(run_metadata, f, indent=2)
            
        # Create episode directories
        episode_dirs = []
        for episode_num in range(num_episodes):
            episode_id = f"episode_{episode_num:03d}"
            episode_dir = run_dir / "episodes" / episode_id
            episode_dir.mkdir(parents=True, exist_ok=True)
            
            # Create episode-level files
            (episode_dir / "trajectory.parquet").touch()
            (episode_dir / "summary.json").touch()
            (episode_dir / "metadata.json").touch()
            
            episode_dirs.append(episode_dir)
            
        return {
            'run_dir': run_dir,
            'metadata_dir': metadata_dir,
            'episode_dirs': episode_dirs,
            'run_metadata': run_metadata
        }
    
    def validate_structure(structure_info: dict):
        """Validate that directory structure meets F-017 requirements."""
        run_dir = structure_info['run_dir']
        
        # Check required directories exist
        assert (run_dir / "metadata").exists(), "Missing metadata directory"
        assert (run_dir / "episodes").exists(), "Missing episodes directory"
        
        # Check run-level metadata
        assert (run_dir / "metadata" / "run_config.json").exists(), "Missing run configuration"
        
        # Check episode structure
        for episode_dir in structure_info['episode_dirs']:
            assert episode_dir.exists(), f"Missing episode directory: {episode_dir}"
            assert (episode_dir / "trajectory.parquet").exists(), "Missing trajectory file"
            assert (episode_dir / "summary.json").exists(), "Missing summary file"
            assert (episode_dir / "metadata.json").exists(), "Missing episode metadata"
            
        return True
    
    def generate_path(run_id: str, episode_id: str = None, file_type: str = None):
        """Generate standardized paths following F-017 hierarchy."""
        base_path = tmp_path / f"run_{run_id}"
        
        if episode_id is None:
            return base_path
            
        episode_path = base_path / "episodes" / episode_id
        
        if file_type is None:
            return episode_path
            
        file_extensions = {
            'trajectory': 'trajectory.parquet',
            'summary': 'summary.json',
            'metadata': 'metadata.json',
            'hdf5': 'data.h5',
            'sqlite': 'data.db'
        }
        
        return episode_path / file_extensions.get(file_type, f"{file_type}.dat")
    
    return {
        'base_dir': tmp_path,
        'create_run_structure': create_run_structure,
        'validate_structure': validate_structure,
        'generate_path': generate_path
    }


@pytest.fixture
def compression_testing_utilities():
    """
    Create compression testing utilities for recorder backend validation.
    
    Provides comprehensive compression algorithm testing tools for validating
    recorder backend compression capabilities, performance impact analysis,
    and compression ratio optimization across different data types and sizes.
    
    Returns:
        Dictionary containing compression testing utilities and benchmarks
        
    Features:
        - Multi-algorithm compression testing (snappy, gzip, lz4, zstd)
        - Performance impact analysis with timing measurements
        - Compression ratio optimization for different data patterns
        - Memory usage monitoring during compression operations
    """
    def generate_test_data(data_type: str, size_mb: float = 1.0):
        """Generate test data of specified type and size."""
        num_rows = int((size_mb * 1024 * 1024) / 100)  # Rough estimate for 100 bytes per row
        
        if data_type == 'trajectory':
            return {
                'positions': np.random.rand(num_rows, 2) * 100,
                'velocities': np.random.rand(num_rows, 2) * 10,
                'timestamps': np.linspace(0, num_rows * 0.1, num_rows),
                'actions': np.random.randint(0, 5, num_rows),
                'rewards': np.random.rand(num_rows) * 10 - 5
            }
        elif data_type == 'sensor':
            return {
                'concentrations': np.random.rand(num_rows, 8) * 1000,
                'gradients': np.random.rand(num_rows, 2) * 100,
                'binary_detections': np.random.randint(0, 2, (num_rows, 4)),
                'timestamps': np.linspace(0, num_rows * 0.1, num_rows)
            }
        elif data_type == 'metadata':
            return {
                'episode_ids': [f"episode_{i:03d}" for i in range(num_rows)],
                'configurations': [{'param': np.random.rand()} for _ in range(num_rows)],
                'statistics': np.random.rand(num_rows, 10)
            }
        else:
            # Random data
            return np.random.rand(num_rows, 10)
    
    def test_compression_algorithm(data: dict, algorithm: str, level: int = None):
        """Test compression performance for a specific algorithm."""
        import pickle
        
        # Serialize data for compression testing
        serialized_data = pickle.dumps(data)
        original_size = len(serialized_data)
        
        # Mock compression results (real implementations would use actual algorithms)
        compression_ratios = {
            'snappy': 2.5,
            'gzip': 3.2,
            'lz4': 2.1,
            'zstd': 3.8,
            'none': 1.0
        }
        
        compression_times_ms = {
            'snappy': 15.2,
            'gzip': 45.6,
            'lz4': 8.9,
            'zstd': 28.3,
            'none': 0.1
        }
        
        ratio = compression_ratios.get(algorithm, 1.0)
        if level and algorithm in ['gzip', 'zstd']:
            # Higher compression levels give better ratios but slower speed
            ratio *= (1 + level * 0.1)
            compression_times_ms[algorithm] *= (1 + level * 0.3)
        
        compressed_size = original_size / ratio
        
        return {
            'algorithm': algorithm,
            'level': level,
            'original_size_bytes': original_size,
            'compressed_size_bytes': int(compressed_size),
            'compression_ratio': ratio,
            'compression_time_ms': compression_times_ms.get(algorithm, 10.0),
            'compression_speed_mb_per_sec': (original_size / 1024 / 1024) / (compression_times_ms.get(algorithm, 10.0) / 1000)
        }
    
    def benchmark_algorithms(data: dict, algorithms: list = None):
        """Benchmark multiple compression algorithms against test data."""
        import pickle  # ensure pickle available for data size calculation
        if algorithms is None:
            algorithms = ['snappy', 'gzip', 'lz4', 'zstd', 'none']
        
        results = []
        for algorithm in algorithms:
            if algorithm in ['gzip', 'zstd']:
                # Test multiple compression levels
                for level in [1, 3, 6, 9]:
                    results.append(test_compression_algorithm(data, algorithm, level))
            else:
                results.append(test_compression_algorithm(data, algorithm))
        
        # Sort by compression ratio (best first)
        results.sort(key=lambda x: x['compression_ratio'], reverse=True)
        
        return {
            'test_data_size_mb': len(pickle.dumps(data)) / 1024 / 1024,
            'algorithm_results': results,
            'best_ratio': results[0] if results else None,
            'fastest': min(results, key=lambda x: x['compression_time_ms']) if results else None
        }
    
    return {
        'generate_test_data': generate_test_data,
        'test_compression_algorithm': test_compression_algorithm,
        'benchmark_algorithms': benchmark_algorithms
    }


@pytest.fixture
def backend_availability_detector():
    """
    Create backend availability detection for graceful degradation testing.
    
    Provides utilities for testing graceful degradation when optional recorder
    backend dependencies are unavailable, ensuring robust fallback behavior
    and proper error handling across different deployment environments.
    
    Returns:
        Dictionary containing availability detection utilities and fallback testing
        
    Features:
        - Dynamic dependency availability detection
        - Fallback behavior testing and validation
        - Error message testing for missing dependencies
        - Runtime backend selection based on availability
    """
    def check_backend_availability():
        """Check availability of all recorder backend dependencies."""
        availability = {
            'pandas': pd is not None,
            'pyarrow': pq is not None,
            'h5py': h5py is not None,
            'sqlite3': sqlite3 is not None,
            'psutil': psutil is not None
        }
        
        backend_support = {
            'parquet': availability['pandas'] and availability['pyarrow'],
            'hdf5': availability['h5py'],
            'sqlite': availability['sqlite3'],
            'performance_monitoring': availability['psutil']
        }
        
        return {
            'dependencies': availability,
            'backends': backend_support,
            'available_backends': [k for k, v in backend_support.items() if v],
            'unavailable_backends': [k for k, v in backend_support.items() if not v]
        }
    
    def test_fallback_behavior(backend_name: str, mock_unavailable: bool = True):
        """Test fallback behavior when backend is unavailable."""
        availability = check_backend_availability()
        
        if mock_unavailable or not availability['backends'].get(backend_name, False):
            # Simulate unavailable backend
            fallback_config = {
                'backend': 'none',
                'warning_message': f"Backend '{backend_name}' unavailable, falling back to NullRecorder",
                'fallback_reason': 'missing_dependencies',
                'original_backend': backend_name
            }
            
            return {
                'fallback_triggered': True,
                'fallback_config': fallback_config,
                'expected_warnings': [f"Warning: {backend_name} backend dependencies not available"]
            }
        else:
            return {
                'fallback_triggered': False,
                'backend_available': True,
                'backend': backend_name
            }
    
    def get_minimal_config():
        """Get minimal configuration that works regardless of dependencies."""
        availability = check_backend_availability()
        
        # Always available - NullRecorder requires no dependencies
        minimal_config = {
            'recorder': {
                '_target_': 'plume_nav_sim.recording.backends.NullRecorder',
                'log_calls': False,
                'validate_interface': True
            },
            'performance': {
                'overhead_threshold_ms': 1.0,
                'memory_limit_mb': 100
            }
        }
        
        # Add SQLite if available (built into Python)
        if availability['backends']['sqlite']:
            minimal_config['alternative_recorder'] = {
                '_target_': 'plume_nav_sim.recording.backends.SQLiteRecorder',
                'output_dir': '/tmp/test_recordings',
                'database_file': 'test_data.db'
            }
        
        return minimal_config
    
    return {
        'check_backend_availability': check_backend_availability,
        'test_fallback_behavior': test_fallback_behavior,
        'get_minimal_config': get_minimal_config
    }


@pytest.fixture
def mock_multi_navigator():
    """
    Create a mock multi-agent Navigator instance for integration testing.
    
    Provides a Navigator mock configured for multi-agent scenarios with multiple
    agents, positions, and orientations for comprehensive integration testing
    of multi-agent navigation algorithms and collision avoidance systems.
    
    Returns:
        MagicMock configured for multi-agent navigation testing
    """
    mock = MagicMock()
    
    # Configure for 3 agents
    num_agents = 3
    mock.num_agents = num_agents
    
    # Configure multi-agent properties as numpy arrays
    mock.positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]])  # 3 agents
    mock.orientations = np.array([0.0, 90.0, 180.0])                     # Different orientations
    mock.speeds = np.array([0.5, 1.0, 0.8])                             # Different speeds
    mock.max_speeds = np.array([2.0, 2.0, 2.0])                         # Same max speeds
    mock.angular_velocities = np.array([0.0, 5.0, -3.0])                # Different rotations
    
    # Configure NavigatorProtocol methods for multi-agent scenarios
    mock.reset.return_value = None
    mock.step.return_value = None
    mock.sample_odor.return_value = np.array([0.1, 0.3, 0.2])           # Different odor readings
    mock.read_single_antenna_odor.return_value = np.array([0.15, 0.25, 0.18])
    mock.sample_multiple_sensors.return_value = np.array([
        [0.1, 0.2],    # Agent 1 sensors
        [0.3, 0.4],    # Agent 2 sensors  
        [0.2, 0.1]     # Agent 3 sensors
    ])
    
    return mock


@pytest.fixture
def mock_source_protocol():
    """
    Create mock source for v1.0 SourceProtocol testing.
    
    Provides a comprehensive mock implementation of SourceProtocol interface
    supporting PointSource, MultiSource, and DynamicSource behaviors for
    testing the new pluggable source abstraction architecture.
    
    Returns:
        Mock SourceProtocol implementation with configurable source types
        
    Features:
        - Support for multiple source types and emission patterns
        - Dynamic source positioning and emission rate configuration
        - Vectorized operations for multi-agent scenarios
        - Integration with plume model testing
    """
    mock = MagicMock()
    
    # Configure SourceProtocol properties
    mock.source_type = 'point'
    mock.position = np.array([50.0, 50.0])  # Center position
    mock.emission_rate = 1000.0  # Strong emission
    mock.is_active = True
    mock.num_sources = 1
    
    # Configure SourceProtocol methods
    mock.get_position.return_value = np.array([50.0, 50.0])
    mock.get_emission_rate.return_value = 1000.0
    mock.get_positions.return_value = np.array([[50.0, 50.0]])  # Multi-source support
    mock.get_emission_rates.return_value = np.array([1000.0])
    mock.update_state.return_value = None
    mock.reset.return_value = None
    mock.is_within_range.return_value = True
    
    # Configure multi-source scenarios
    def configure_multi_source(num_sources=3):
        mock.num_sources = num_sources
        mock.source_type = 'multi'
        positions = np.random.rand(num_sources, 2) * 100
        emission_rates = np.random.rand(num_sources) * 1000 + 500
        mock.get_positions.return_value = positions
        mock.get_emission_rates.return_value = emission_rates
        return positions, emission_rates
    
    # Configure dynamic source behavior
    def configure_dynamic_source():
        mock.source_type = 'dynamic'
        mock.velocity = np.array([1.0, 0.5])  # Moving source
        mock.emission_pattern = 'variable'
        mock.get_velocity.return_value = np.array([1.0, 0.5])
        
        # Dynamic position based on time
        def dynamic_position(time_step=0):
            base_pos = np.array([50.0, 50.0])
            movement = mock.velocity * time_step * 0.1
            return base_pos + movement
        
        mock.get_position.side_effect = dynamic_position
        return mock.velocity
    
    mock.configure_multi_source = configure_multi_source
    mock.configure_dynamic_source = configure_dynamic_source
    
    return mock


@pytest.fixture
def mock_boundary_policy():
    """
    Create mock boundary policy for v1.0 BoundaryPolicyProtocol testing.
    
    Provides a comprehensive mock implementation of BoundaryPolicyProtocol
    interface supporting terminate, bounce, wrap, and clip behaviors for
    testing the new pluggable boundary handling architecture.
    
    Returns:
        Mock BoundaryPolicyProtocol implementation with configurable policies
        
    Features:
        - Support for all boundary policy types (terminate, bounce, wrap, clip)
        - Vectorized boundary checking for multi-agent scenarios
        - Configurable domain boundaries and violation handling
        - Integration with navigation controller testing
    """
    mock = MagicMock()
    
    # Configure BoundaryPolicyProtocol properties
    mock.policy_type = 'terminate'
    mock.domain_bounds = np.array([[0.0, 100.0], [0.0, 100.0]])  # 100x100 domain
    mock.violation_threshold = 0.1
    mock.status_on_violation = 'oob'
    
    # Configure BoundaryPolicyProtocol methods
    mock.check_violations.return_value = np.array([False])  # No violations
    mock.apply_policy.return_value = np.array([[50.0, 50.0]])  # Return corrected positions
    mock.get_termination_status.return_value = False
    mock.reset.return_value = None
    
    # Configure different policy behaviors
    def configure_terminate_policy():
        mock.policy_type = 'terminate'
        mock.status_on_violation = 'oob'
        
        def check_terminate_violations(positions):
            violations = []
            for pos in positions:
                violation = (pos[0] < 0 or pos[0] > 100 or 
                           pos[1] < 0 or pos[1] > 100)
                violations.append(violation)
            return np.array(violations)
        
        mock.check_violations.side_effect = check_terminate_violations
        return mock
    
    def configure_bounce_policy():
        mock.policy_type = 'bounce'
        
        def apply_bounce_policy(positions, velocities=None):
            corrected_positions = []
            for pos in positions:
                corrected_pos = np.copy(pos)
                # Bounce off boundaries
                if pos[0] < 0:
                    corrected_pos[0] = -pos[0]
                elif pos[0] > 100:
                    corrected_pos[0] = 200 - pos[0]
                if pos[1] < 0:
                    corrected_pos[1] = -pos[1]
                elif pos[1] > 100:
                    corrected_pos[1] = 200 - pos[1]
                corrected_positions.append(corrected_pos)
            return np.array(corrected_positions)
        
        mock.apply_policy.side_effect = apply_bounce_policy
        return mock
    
    def configure_wrap_policy():
        mock.policy_type = 'wrap'
        
        def apply_wrap_policy(positions):
            corrected_positions = []
            for pos in positions:
                corrected_pos = np.copy(pos)
                # Wrap around boundaries
                corrected_pos[0] = pos[0] % 100
                corrected_pos[1] = pos[1] % 100
                corrected_positions.append(corrected_pos)
            return np.array(corrected_positions)
        
        mock.apply_policy.side_effect = apply_wrap_policy
        return mock
    
    def configure_clip_policy():
        mock.policy_type = 'clip'
        
        def apply_clip_policy(positions):
            corrected_positions = []
            for pos in positions:
                corrected_pos = np.copy(pos)
                # Clip to boundaries
                corrected_pos[0] = np.clip(pos[0], 0, 100)
                corrected_pos[1] = np.clip(pos[1], 0, 100)
                corrected_positions.append(corrected_pos)
            return np.array(corrected_positions)
        
        mock.apply_policy.side_effect = apply_clip_policy
        return mock
    
    mock.configure_terminate_policy = configure_terminate_policy
    mock.configure_bounce_policy = configure_bounce_policy
    mock.configure_wrap_policy = configure_wrap_policy
    mock.configure_clip_policy = configure_clip_policy
    
    return mock


@pytest.fixture
def mock_action_interface():
    """
    Create mock action interface for v1.0 ActionInterfaceProtocol testing.
    
    Provides a comprehensive mock implementation of ActionInterfaceProtocol
    interface supporting Continuous2D and CardinalDiscrete action space
    translations for testing the new standardized action processing architecture.
    
    Returns:
        Mock ActionInterfaceProtocol implementation with configurable action spaces
        
    Features:
        - Support for continuous and discrete action spaces
        - Action validation and bounds checking
        - Integration with Gymnasium environment testing
        - Configurable action space parameters and translation logic
    """
    mock = MagicMock()
    
    # Configure ActionInterfaceProtocol properties
    mock.action_type = 'continuous2d'
    mock.action_bounds = np.array([[-2.0, 2.0], [-np.pi, np.pi]])  # velocity, angular_velocity
    mock.normalize_actions = True
    mock.clip_actions = True
    
    # Configure ActionInterfaceProtocol methods
    mock.translate_action.return_value = {'velocity': np.array([1.0, 0.5]), 'angular_velocity': 0.1}
    mock.validate_action.return_value = True
    mock.get_action_space.return_value = None  # Will be configured per type
    mock.reset.return_value = None
    
    # Configure continuous action space
    def configure_continuous2d():
        mock.action_type = 'continuous2d'
        mock.action_dim = 2
        
        def translate_continuous_action(action):
            # action is [velocity_magnitude, angular_velocity]
            velocity_magnitude = np.clip(action[0], 0, 2.0)
            angular_velocity = np.clip(action[1], -np.pi, np.pi)
            
            return {
                'velocity_magnitude': velocity_magnitude,
                'angular_velocity': angular_velocity,
                'action_type': 'continuous'
            }
        
        def validate_continuous_action(action):
            return (len(action) == 2 and 
                   -2.0 <= action[0] <= 2.0 and 
                   -np.pi <= action[1] <= np.pi)
        
        mock.translate_action.side_effect = translate_continuous_action
        mock.validate_action.side_effect = validate_continuous_action
        
        # Mock Gymnasium Box space
        action_space_mock = MagicMock()
        action_space_mock.shape = (2,)
        action_space_mock.low = np.array([-2.0, -np.pi])
        action_space_mock.high = np.array([2.0, np.pi])
        mock.get_action_space.return_value = action_space_mock
        
        return mock
    
    # Configure discrete action space
    def configure_cardinal_discrete():
        mock.action_type = 'cardinal_discrete'
        mock.action_dim = 5  # stay, north, south, east, west
        
        action_mapping = {
            0: {'velocity': np.array([0.0, 0.0]), 'name': 'stay'},
            1: {'velocity': np.array([0.0, 1.0]), 'name': 'north'},
            2: {'velocity': np.array([0.0, -1.0]), 'name': 'south'},
            3: {'velocity': np.array([1.0, 0.0]), 'name': 'east'},
            4: {'velocity': np.array([-1.0, 0.0]), 'name': 'west'}
        }
        
        def translate_discrete_action(action):
            action_idx = int(action)
            if action_idx in action_mapping:
                return {
                    'velocity': action_mapping[action_idx]['velocity'],
                    'action_name': action_mapping[action_idx]['name'],
                    'action_type': 'discrete'
                }
            else:
                return {
                    'velocity': np.array([0.0, 0.0]),
                    'action_name': 'stay',
                    'action_type': 'discrete'
                }
        
        def validate_discrete_action(action):
            try:
                action_idx = int(action)
                return 0 <= action_idx <= 4
            except (ValueError, TypeError):
                return False
        
        mock.translate_action.side_effect = translate_discrete_action
        mock.validate_action.side_effect = validate_discrete_action
        
        # Mock Gymnasium Discrete space
        action_space_mock = MagicMock()
        action_space_mock.n = 5
        mock.get_action_space.return_value = action_space_mock
        
        return mock
    
    mock.configure_continuous2d = configure_continuous2d
    mock.configure_cardinal_discrete = configure_cardinal_discrete
    
    return mock


@pytest.fixture
def mock_stats_aggregator():
    """
    Create mock statistics aggregator for v1.0 StatsAggregatorProtocol testing.
    
    Provides a comprehensive mock implementation of StatsAggregatorProtocol
    interface for testing automated statistics collection, metric calculation,
    and summary generation capabilities.
    
    Returns:
        Mock StatsAggregatorProtocol implementation with research metrics
        
    Features:
        - Automated episode and run-level statistics calculation
        - Research-focused metrics with publication-quality output
        - Integration with recorder system for data source connectivity
        - Configurable metrics definitions and custom calculations
    """
    mock = MagicMock()
    
    # Configure StatsAggregatorProtocol properties
    mock.metrics_enabled = True
    mock.auto_calculation = True
    mock.output_format = 'json'
    
    # Configure StatsAggregatorProtocol methods
    mock.calculate_episode_stats.return_value = {
        'episode_length': 100,
        'total_reward': 45.6,
        'average_reward': 0.456,
        'max_reward': 2.1,
        'min_reward': -1.8,
        'success_rate': 0.85,
        'path_efficiency': 0.72,
        'exploration_coverage': 0.64,
        'computation_time_ms': 23.4
    }
    
    mock.calculate_run_stats.return_value = {
        'total_episodes': 50,
        'successful_episodes': 42,
        'average_episode_length': 95.6,
        'average_total_reward': 43.2,
        'reward_std': 12.8,
        'success_rate': 0.84,
        'path_efficiency_mean': 0.71,
        'path_efficiency_std': 0.15,
        'learning_progress': 0.23,
        'total_computation_time_sec': 1.17
    }
    
    mock.export_summary.return_value = {
        'timestamp': time.time(),
        'summary_format': 'json',
        'output_path': '/tmp/summary.json',
        'metrics_count': 15,
        'export_success': True
    }
    
    mock.reset.return_value = None
    
    # Configure custom metrics
    def add_custom_metric(metric_name: str, calculation_fn):
        """Add a custom metric calculation."""
        if not hasattr(mock, 'custom_metrics'):
            mock.custom_metrics = {}
        mock.custom_metrics[metric_name] = calculation_fn
        return True
    
    def calculate_custom_metrics(episode_data: dict):
        """Calculate all custom metrics for episode data."""
        if not hasattr(mock, 'custom_metrics'):
            return {}
        
        custom_results = {}
        for metric_name, calc_fn in mock.custom_metrics.items():
            try:
                custom_results[metric_name] = calc_fn(episode_data)
            except Exception as e:
                custom_results[metric_name] = f"calculation_error: {str(e)}"
        
        return custom_results
    
    mock.add_custom_metric = add_custom_metric
    mock.calculate_custom_metrics = calculate_custom_metrics
    
    return mock


@pytest.fixture
def mock_hook_system():
    """
    Create mock hook system for v1.0 extensibility testing.
    
    Provides a comprehensive mock implementation of the hook system that
    enables non-invasive extensions through extra_obs_fn, extra_reward_fn,
    and episode_end_fn callback integration for testing extensibility.
    
    Returns:
        Mock hook system with configurable callback functions
        
    Features:
        - Lifecycle hook integration (pre/post step, episode start/end)
        - Custom observation and reward function testing
        - Non-invasive system extension validation
        - Hook performance monitoring and call tracking
    """
    mock = MagicMock()
    
    # Configure hook system properties
    mock.hooks_enabled = True
    mock.hook_call_count = 0
    mock.registered_hooks = {}
    
    # Configure default hook functions
    def default_extra_obs_fn(state):
        """Default extra observation function."""
        return {
            'custom_observation': np.random.rand(5),
            'hook_timestamp': time.time(),
            'hook_call_id': mock.hook_call_count
        }
    
    def default_extra_reward_fn(state, action, reward):
        """Default extra reward function."""
        return {
            'shaped_reward': reward * 1.1,
            'exploration_bonus': 0.01,
            'custom_penalty': -0.005 if state.get('boundary_violation', False) else 0.0
        }
    
    def default_episode_end_fn(episode_data):
        """Default episode end function."""
        return {
            'custom_summary': {'episode_processed': True},
            'additional_metrics': {'custom_metric': np.random.rand()},
            'hook_execution_time_ms': 2.3
        }
    
    # Configure hook methods
    mock.extra_obs_fn = default_extra_obs_fn
    mock.extra_reward_fn = default_extra_reward_fn
    mock.episode_end_fn = default_episode_end_fn
    
    def register_hook(hook_name: str, hook_fn):
        """Register a custom hook function."""
        mock.registered_hooks[hook_name] = hook_fn
        return True
    
    def call_hook(hook_name: str, *args, **kwargs):
        """Call a registered hook function."""
        mock.hook_call_count += 1
        if hook_name in mock.registered_hooks:
            return mock.registered_hooks[hook_name](*args, **kwargs)
        else:
            return None
    
    def get_hook_statistics():
        """Get hook system performance statistics."""
        return {
            'total_hook_calls': mock.hook_call_count,
            'registered_hooks': list(mock.registered_hooks.keys()),
            'hooks_enabled': mock.hooks_enabled,
            'average_execution_time_ms': 1.2
        }
    
    mock.register_hook = register_hook
    mock.call_hook = call_hook
    mock.get_hook_statistics = get_hook_statistics
    
    return mock


@pytest.fixture
def mock_v1_environment(mock_source_protocol, mock_boundary_policy, 
                       mock_action_interface, mock_parquet_recorder, 
                       mock_stats_aggregator, mock_hook_system):
    """
    Create comprehensive mock v1.0 environment with all protocol integrations.
    
    Provides a fully integrated mock PlumeNavigationEnv instance that incorporates
    all v1.0 protocol-based components for comprehensive integration testing
    of the new architecture with dependency injection and hook system.
    
    Args:
        mock_source_protocol: Mock source implementation
        mock_boundary_policy: Mock boundary policy implementation
        mock_action_interface: Mock action interface implementation
        mock_parquet_recorder: Mock recorder implementation
        mock_stats_aggregator: Mock statistics aggregator implementation
        mock_hook_system: Mock hook system implementation
        
    Returns:
        Mock v1.0 PlumeNavigationEnv with integrated protocol components
        
    Features:
        - Complete v1.0 protocol integration testing
        - Dependency injection validation
        - Hook system integration verification
        - Performance monitoring with all components
    """
    mock = MagicMock()
    
    # Inject protocol-based components
    mock.source = mock_source_protocol
    mock.boundary_policy = mock_boundary_policy
    mock.action_interface = mock_action_interface
    mock.recorder = mock_parquet_recorder
    mock.stats_aggregator = mock_stats_aggregator
    mock.hooks = mock_hook_system
    
    # Configure environment properties
    mock.observation_space = None  # Will be set by action interface
    mock.action_space = None      # Will be set by action interface
    mock.deterministic_seeding = True
    mock.current_seed = 42
    mock.step_count = 0
    mock.episode_count = 0
    
    # Configure environment methods
    def reset_environment(seed=None):
        """Reset environment with v1.0 protocol integration."""
        if seed is not None:
            mock.current_seed = seed
            
        # Reset all components
        mock.source.reset()
        mock.boundary_policy.reset()
        mock.action_interface.reset()
        if mock.recorder:
            mock.recorder.record_episode({'type': 'episode_start', 'seed': seed})
        
        mock.step_count = 0
        mock.episode_count += 1
        
        # Generate initial observation with hooks
        core_obs = {
            'position': np.array([10.0, 10.0]),
            'concentration': 0.5,
            'velocity': np.array([0.0, 0.0])
        }
        
        if mock.hooks.hooks_enabled and mock.hooks.extra_obs_fn:
            extra_obs = mock.hooks.extra_obs_fn({'position': core_obs['position']})
            core_obs.update(extra_obs)
        
        return core_obs, {'seed': seed, 'episode_id': mock.episode_count}
    
    def step_environment(action):
        """Step environment with v1.0 protocol integration."""
        mock.step_count += 1
        
        # Translate action through action interface
        translated_action = mock.action_interface.translate_action(action)
        
        # Update source state
        mock.source.update_state()
        
        # Apply boundary policy
        current_position = np.array([10.0 + mock.step_count * 0.1, 10.0])
        violations = mock.boundary_policy.check_violations([current_position])
        if violations.any():
            current_position = mock.boundary_policy.apply_policy([current_position])[0]
        
        # Generate observation
        core_obs = {
            'position': current_position,
            'concentration': mock.source.get_emission_rate() * 0.001,
            'velocity': translated_action.get('velocity', np.array([0.0, 0.0]))
        }
        
        # Calculate core reward
        core_reward = 1.0 if core_obs['concentration'] > 0.5 else 0.1
        
        # Apply hook system
        if mock.hooks.hooks_enabled:
            if mock.hooks.extra_obs_fn:
                extra_obs = mock.hooks.extra_obs_fn(core_obs)
                core_obs.update(extra_obs)
            
            if mock.hooks.extra_reward_fn:
                extra_reward_data = mock.hooks.extra_reward_fn(core_obs, action, core_reward)
                core_reward = extra_reward_data.get('shaped_reward', core_reward)
        
        # Check termination
        done = mock.step_count >= 100 or mock.boundary_policy.get_termination_status()
        truncated = False
        
        # Record step data
        if mock.recorder:
            step_data = {
                'step': mock.step_count,
                'action': action,
                'observation': core_obs,
                'reward': core_reward,
                'done': done
            }
            mock.recorder.record_step(step_data)
        
        # Handle episode end
        info = {'step_count': mock.step_count}
        if done and mock.hooks.hooks_enabled and mock.hooks.episode_end_fn:
            episode_data = {
                'total_steps': mock.step_count,
                'total_reward': core_reward * mock.step_count,  # Simplified
                'final_position': current_position
            }
            end_info = mock.hooks.episode_end_fn(episode_data)
            info.update(end_info)
        
        return core_obs, core_reward, done, truncated, info
    
    mock.reset = reset_environment
    mock.step = step_environment
    
    return mock


@pytest.fixture(autouse=True)
def reset_global_state():
    """
    Automatically reset global state between tests for isolation.
    
    This fixture runs automatically for every test to ensure clean global state
    including Hydra GlobalHydra instance, seed manager singletons, and any
    other global state that might affect test determinism.
    
    Features:
        - Automatic execution for every test (autouse=True)
        - Hydra GlobalHydra instance cleanup
        - Seed manager singleton reset
        - Global configuration state isolation
        - Prevents test interference through shared state
    """
    # Reset Hydra global state if available
    if HYDRA_AVAILABLE:
        try:
            GlobalHydra.instance().clear()
        except Exception:
            pass  # GlobalHydra might not be initialized
    
    # Reset SeedManager singleton if available
    if SEED_MANAGER_AVAILABLE and hasattr(SeedManager, 'reset'):
        try:
            SeedManager.reset()
        except Exception:
            pass  # SeedManager might not be initialized
    
    yield  # Run the test
    
    # Post-test cleanup
    if HYDRA_AVAILABLE:
        try:
            GlobalHydra.instance().clear()
        except Exception:
            pass
    
    if SEED_MANAGER_AVAILABLE and hasattr(SeedManager, 'reset'):
        try:
            SeedManager.reset()
        except Exception:
            pass


# Additional utility fixtures for v1.0 testing

@pytest.fixture
def mock_agent_initializer():
    """
    Create mock agent initializer for v1.0 AgentInitializer testing.
    
    Provides mock implementations of different agent initialization strategies
    (uniform_random, grid, fixed_list, from_dataset) for testing the new
    configurable agent initialization patterns.
    
    Returns:
        Mock AgentInitializer with configurable strategies
        
    Features:
        - Support for all initialization strategies
        - Deterministic seeding and validation
        - Multi-agent scenario testing
        - Domain constraint validation
    """
    mock = MagicMock()
    
    # Configure AgentInitializer properties
    mock.strategy = 'uniform_random'
    mock.domain_bounds = np.array([[0.0, 100.0], [0.0, 100.0]])
    mock.num_agents = 1
    mock.deterministic = True
    mock.seed = 42
    
    # Configure initialization methods
    def uniform_random_positions(num_agents=1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        positions = np.random.rand(num_agents, 2) * 100
        return positions
    
    def grid_positions(num_agents=1, grid_size=None):
        if grid_size is None:
            grid_size = int(np.ceil(np.sqrt(num_agents)))
        
        positions = []
        for i in range(num_agents):
            row = i // grid_size
            col = i % grid_size
            x = (col + 0.5) * (100.0 / grid_size)
            y = (row + 0.5) * (100.0 / grid_size)
            positions.append([x, y])
        
        return np.array(positions)
    
    def fixed_list_positions(position_list=None):
        if position_list is None:
            position_list = [[10.0, 10.0], [90.0, 90.0], [50.0, 50.0]]
        return np.array(position_list)
    
    mock.initialize_positions = uniform_random_positions
    mock.uniform_random_positions = uniform_random_positions
    mock.grid_positions = grid_positions
    mock.fixed_list_positions = fixed_list_positions
    
    # Strategy configuration methods
    def configure_uniform_random(domain_bounds=None, seed=42):
        mock.strategy = 'uniform_random'
        if domain_bounds is not None:
            mock.domain_bounds = domain_bounds
        mock.seed = seed
        mock.initialize_positions = lambda num_agents=1: uniform_random_positions(num_agents, seed)
        return mock
    
    def configure_grid(grid_size=None):
        mock.strategy = 'grid'
        mock.grid_size = grid_size
        mock.initialize_positions = lambda num_agents=1: grid_positions(num_agents, grid_size)
        return mock
    
    def configure_fixed_list(positions):
        mock.strategy = 'fixed_list'
        mock.fixed_positions = positions
        mock.initialize_positions = lambda num_agents=None: fixed_list_positions(positions)
        return mock
    
    mock.configure_uniform_random = configure_uniform_random
    mock.configure_grid = configure_grid
    mock.configure_fixed_list = configure_fixed_list
    
    return mock


@pytest.fixture
def test_data_generator():
    """
    Create test data generator for comprehensive v1.0 testing scenarios.
    
    Provides utilities for generating realistic test data across different
    simulation scenarios, agent configurations, and environmental conditions
    for comprehensive protocol testing and validation.
    
    Returns:
        Dictionary containing test data generation utilities
        
    Features:
        - Realistic simulation data generation
        - Multi-agent scenario creation
        - Environmental condition variation
        - Performance testing data sets
    """
    def generate_trajectory_data(num_steps=100, num_agents=1, seed=42):
        """Generate realistic trajectory data for testing."""
        np.random.seed(seed)
        
        trajectory_data = {
            'positions': np.random.rand(num_steps, num_agents, 2) * 100,
            'velocities': np.random.rand(num_steps, num_agents, 2) * 10 - 5,
            'orientations': np.random.rand(num_steps, num_agents) * 2 * np.pi,
            'concentrations': np.random.exponential(scale=0.5, size=(num_steps, num_agents)),
            'actions': np.random.randint(0, 5, size=(num_steps, num_agents)),
            'rewards': np.random.rand(num_steps, num_agents) * 10 - 2,
            'timestamps': np.linspace(0, num_steps * 0.1, num_steps)
        }
        
        return trajectory_data
    
    def generate_episode_data(episode_length=100, num_agents=1, seed=42):
        """Generate complete episode data for testing."""
        trajectory = generate_trajectory_data(episode_length, num_agents, seed)
        
        episode_data = {
            'episode_id': f'test_episode_{seed}',
            'total_steps': episode_length,
            'num_agents': num_agents,
            'total_reward': np.sum(trajectory['rewards']),
            'average_reward': np.mean(trajectory['rewards']),
            'success_rate': 0.75,  # Mock success rate
            'path_efficiency': 0.68,  # Mock efficiency
            'trajectory': trajectory,
            'metadata': {
                'seed': seed,
                'environment': 'test',
                'configuration': {'test': True}
            }
        }
        
        return episode_data
    
    def generate_performance_test_data(data_size_mb=1.0):
        """Generate data for performance testing."""
        # Calculate approximate number of records for target size
        bytes_per_record = 200  # Rough estimate
        num_records = int((data_size_mb * 1024 * 1024) / bytes_per_record)
        
        performance_data = {
            'large_positions': np.random.rand(num_records, 2) * 100,
            'large_actions': np.random.randint(0, 5, num_records),
            'large_rewards': np.random.rand(num_records) * 10,
            'large_metadata': [{'id': i, 'value': np.random.rand()} for i in range(num_records)],
            'size_mb': data_size_mb,
            'num_records': num_records
        }
        
        return performance_data
    
    def generate_multi_agent_scenario(num_agents=5, scenario_type='cooperative'):
        """Generate multi-agent scenario data for testing."""
        scenarios = {
            'cooperative': {
                'agent_types': ['explorer'] * num_agents,
                'communication': True,
                'shared_reward': True
            },
            'competitive': {
                'agent_types': ['competitor'] * num_agents,
                'communication': False,
                'shared_reward': False
            },
            'mixed': {
                'agent_types': ['explorer', 'competitor', 'neutral'] * (num_agents // 3) + ['explorer'] * (num_agents % 3),
                'communication': 'partial',
                'shared_reward': 'partial'
            }
        }
        
        scenario_config = scenarios.get(scenario_type, scenarios['cooperative'])
        scenario_config['num_agents'] = num_agents
        scenario_config['initial_positions'] = np.random.rand(num_agents, 2) * 100
        
        return scenario_config
    
    return {
        'generate_trajectory_data': generate_trajectory_data,
        'generate_episode_data': generate_episode_data,
        'generate_performance_test_data': generate_performance_test_data,
        'generate_multi_agent_scenario': generate_multi_agent_scenario
    }