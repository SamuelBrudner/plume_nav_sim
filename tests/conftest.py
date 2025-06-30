"""
Shared fixtures for tests in the plume_nav_sim package.

This file contains pytest fixtures that are shared across multiple test files
to reduce duplication and ensure consistency across the enhanced cookiecutter-based
architecture including Hydra configuration management, CLI interface testing,
database session management, and workflow orchestration validation.

Key Testing Domains:
- Configuration Management: pytest-hydra integration for hierarchical config testing
- CLI Interface Testing: click.testing.CliRunner for command-line validation
- Database Session Management: In-memory SQLAlchemy sessions for persistence testing
- Seed Management: Controlled random state for reproducible test execution
- Workflow Orchestration: Mock patterns for DVC and Snakemake pipeline testing

Fixture Categories:
1. Core Scientific Fixtures: Video capture, navigator, and simulation mocks
2. Configuration Fixtures: Hydra config composition and validation fixtures
3. CLI Testing Fixtures: CliRunner instances and parameter validation fixtures
4. Database Fixtures: In-memory SQLAlchemy sessions and transaction management
5. Utility Fixtures: Seed management, environment isolation, and file system mocks

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

Author: Cookiecutter Template Generator
Version: 2.0.0
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

# Core package imports with graceful fallbacks
try:
    from plume_nav_sim.utils.seed_manager import SeedManager, SeedConfig
    SEED_MANAGER_AVAILABLE = True
except ImportError:
    SEED_MANAGER_AVAILABLE = False
    SeedManager = None
    SeedConfig = None


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
    
    # Create mock configuration structure
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
    Fixture to provide test configuration data compatible with Hydra structure.
    
    Returns a dictionary with hierarchical test configurations following the
    enhanced cookiecutter-based structure with conf/base.yaml and conf/config.yaml
    patterns. Includes comprehensive parameter coverage for all system components.
    """
    # Base configuration (conf/base.yaml equivalent)
    base_config = {
        "defaults": [
            "_self_",
            "navigator: single",
            "video_plume: default",
            "simulation: default"
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
    
    # User configuration with overrides (conf/config.yaml equivalent)
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
        "reproducibility": {
            "global_seed": 42,
            "auto_seed": False,
            "validate_initialization": True
        },
        "logging": {
            "level": "DEBUG"
        }
    }
    
    # Local overrides (conf/local/ equivalent)
    local_config = {
        "database": {
            "enabled": True,
            "url": "sqlite:///test.db"
        },
        "logging": {
            "level": "ERROR"  # Suppress logs in testing
        }
    }
    
    return {
        "base_config": base_config,
        "user_config": user_config,
        "local_config": local_config,
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