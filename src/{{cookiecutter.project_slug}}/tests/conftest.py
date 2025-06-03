"""
Central pytest configuration and fixture management for comprehensive test infrastructure.

This module provides shared test infrastructure supporting CLI testing, Hydra configuration composition,
database session management, and comprehensive mocking strategies for scientific computing workflows.
The implementation follows the enhanced testing strategy defined in Section 6.6 of the technical
specification, providing enterprise-grade testing capabilities with pytest-hydra integration,
Click CliRunner setup, in-memory SQLite sessions, and domain-specific mocks.

Key Features:
- pytest-hydra fixtures for hierarchical configuration testing
- Click CliRunner integration for CLI interface validation
- In-memory SQLAlchemy database session fixtures for isolation
- Comprehensive mock framework supporting unittest.mock patterns
- Seed management fixtures for reproducible randomized testing
- Configuration override and validation test fixtures
- Scientific computing mock implementations for video processing

Coverage Requirements:
- Configuration modules: >85% coverage per Section 6.6.3.1
- CLI modules: >85% coverage per enhanced testing standards
- Database modules: >80% coverage per Section 6.6.5.2
- Overall system: >80% coverage with deterministic test behavior

Usage Examples:
    # Configuration testing with pytest-hydra
    def test_hydra_config_composition(hydra_config_fixture):
        assert hydra_config_fixture.navigator.num_agents == 2
    
    # CLI testing with Click runner
    def test_cli_command_execution(cli_runner_fixture):
        runner, env = cli_runner_fixture
        result = runner.invoke(main, ['--help'], env=env)
        assert result.exit_code == 0
    
    # Database testing with in-memory sessions
    def test_database_operations(db_session_fixture):
        # Test database operations with isolated session
        pass
    
    # Scientific computing with deterministic seeds
    def test_navigation_reproducibility(seed_fixture):
        # Test deterministic behavior
        pass
"""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, Generator, Tuple, Optional
from unittest.mock import Mock, MagicMock, patch
from contextlib import contextmanager

# Core testing framework imports
import pytest
from click.testing import CliRunner

# Hydra and configuration testing imports
try:
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

# SQLAlchemy imports for database session testing
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import StaticPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Scientific computing imports
import cv2


# =============================================================================
# Core Test Configuration and Setup
# =============================================================================

def pytest_configure(config):
    """
    Configure pytest session with enhanced testing capabilities.
    
    Sets up global test environment configuration including:
    - Enhanced test markers for different testing domains
    - pytest-hydra plugin configuration
    - Scientific computing test setup
    - Coverage tracking configuration
    
    Args:
        config: pytest configuration object
    """
    # Register custom test markers for enhanced categorization
    config.addinivalue_line(
        "markers", 
        "hydra_config: mark test as requiring Hydra configuration fixtures"
    )
    config.addinivalue_line(
        "markers", 
        "cli_integration: mark test as CLI interface integration test"
    )
    config.addinivalue_line(
        "markers", 
        "database_session: mark test as requiring database session fixtures"
    )
    config.addinivalue_line(
        "markers", 
        "scientific_computing: mark test as scientific computation validation"
    )
    config.addinivalue_line(
        "markers", 
        "video_processing: mark test as video/image processing test"
    )
    config.addinivalue_line(
        "markers", 
        "reproducibility: mark test as requiring deterministic random behavior"
    )
    config.addinivalue_line(
        "markers", 
        "integration: mark test as cross-component integration test"
    )


def pytest_sessionstart(session):
    """
    Initialize test session with global setup.
    
    Performs session-level initialization including:
    - Global Hydra cleanup
    - Test environment variable setup
    - Temporary directory preparation
    - Scientific computing environment setup
    
    Args:
        session: pytest session object
    """
    # Clean up any existing Hydra global state
    if HYDRA_AVAILABLE:
        GlobalHydra.instance().clear()
    
    # Set up test environment variables
    os.environ.setdefault('PYTEST_RUNNING', 'true')
    os.environ.setdefault('HYDRA_FULL_ERROR', '1')
    
    # Configure scientific computing for deterministic behavior
    np.random.seed(42)


def pytest_sessionfinish(session, exitstatus):
    """
    Clean up test session resources.
    
    Performs session-level cleanup including:
    - Hydra global state cleanup
    - Temporary file cleanup
    - Environment variable restoration
    
    Args:
        session: pytest session object
        exitstatus: test session exit status
    """
    # Clean up Hydra global state
    if HYDRA_AVAILABLE:
        GlobalHydra.instance().clear()
    
    # Clean up test environment variables
    os.environ.pop('PYTEST_RUNNING', None)
    os.environ.pop('HYDRA_FULL_ERROR', None)


# =============================================================================
# Configuration Testing Fixtures (pytest-hydra integration)
# =============================================================================

@pytest.fixture(scope="session")
def test_config_dir(tmp_path_factory) -> Path:
    """
    Create temporary configuration directory structure for testing.
    
    Provides a session-scoped temporary directory containing test configuration
    files that mirror the production conf/ directory structure. This enables
    isolated configuration testing without affecting the actual configuration.
    
    Returns:
        Path: Temporary directory containing test configuration structure
        
    Directory Structure:
        test_config_dir/
        ├── base.yaml
        ├── config.yaml
        └── local/
            ├── credentials.yaml.template
            └── paths.yaml.template
    """
    config_dir = tmp_path_factory.mktemp("test_config")
    
    # Create base configuration
    base_config = {
        'navigator': {
            'type': 'SingleAgentController',
            'num_agents': 1,
            'max_speed': 1.0,
            'angular_velocity': 0.5,
            'sensor_strategy': 'bilateral',
            'collision_avoidance': True
        },
        'video_plume': {
            'video_path': '${oc.env:TEST_VIDEO_PATH,test_video.mp4}',
            'frame_width': 640,
            'frame_height': 480,
            'preprocessing': {
                'gaussian_blur': True,
                'horizontal_flip': False
            }
        },
        'simulation': {
            'num_steps': 100,
            'step_size': 0.1,
            'initial_positions': [[0.0, 0.0]],
            'initial_orientations': [0.0]
        },
        'database': {
            'enabled': False,
            'url': 'sqlite:///:memory:',
            'echo': False
        },
        'logging': {
            'level': 'INFO',
            'format': 'structured',
            'correlation_id': True
        },
        'seed': {
            'value': 42,
            'strict_reproducibility': True
        }
    }
    
    # Create environment-specific configuration
    env_config = {
        'defaults': ['base'],
        'navigator': {
            'num_agents': 2,
            'max_speed': 1.5
        },
        'hydra': {
            'job': {
                'name': 'test_simulation'
            },
            'run': {
                'dir': '${oc.env:HYDRA_RUN_DIR,./outputs/${now:%Y-%m-%d_%H-%M-%S}}'
            }
        }
    }
    
    # Write configuration files
    import yaml
    
    with open(config_dir / "base.yaml", "w") as f:
        yaml.dump(base_config, f, default_flow_style=False)
    
    with open(config_dir / "config.yaml", "w") as f:
        yaml.dump(env_config, f, default_flow_style=False)
    
    # Create local override directory and templates
    local_dir = config_dir / "local"
    local_dir.mkdir()
    
    credentials_template = {
        'database': {
            'username': '${oc.env:DB_USERNAME}',
            'password': '${oc.env:DB_PASSWORD}',
            'host': '${oc.env:DB_HOST,localhost}',
            'port': '${oc.env:DB_PORT,5432}'
        }
    }
    
    with open(local_dir / "credentials.yaml.template", "w") as f:
        yaml.dump(credentials_template, f, default_flow_style=False)
    
    return config_dir


@pytest.fixture
def hydra_config_fixture(test_config_dir) -> Generator[DictConfig, None, None]:
    """
    Provide Hydra configuration fixture for hierarchical composition testing.
    
    Creates an isolated Hydra configuration context with test configuration
    directory structure. Supports override testing, environment variable
    interpolation, and schema validation scenarios. Automatically cleans up
    Hydra global state after test execution.
    
    Args:
        test_config_dir: Test configuration directory fixture
        
    Yields:
        DictConfig: Composed Hydra configuration object
        
    Example:
        def test_config_composition(hydra_config_fixture):
            cfg = hydra_config_fixture
            assert cfg.navigator.num_agents == 2
            assert cfg.navigator.max_speed == 1.5
    """
    if not HYDRA_AVAILABLE:
        pytest.skip("Hydra not available for configuration testing")
    
    # Ensure clean Hydra state
    GlobalHydra.instance().clear()
    
    try:
        with initialize(config_path=str(test_config_dir), version_base=None):
            # Compose configuration with test overrides
            cfg = compose(config_name="config")
            yield cfg
    finally:
        # Clean up Hydra global state
        GlobalHydra.instance().clear()


@pytest.fixture
def hydra_config_with_overrides(test_config_dir):
    """
    Provide Hydra configuration fixture with runtime override testing.
    
    Enables testing of Hydra's override mechanism with various parameter
    combinations, environment variable interpolation, and hierarchical
    composition scenarios. Supports both command-line style overrides
    and programmatic configuration modification.
    
    Args:
        test_config_dir: Test configuration directory fixture
        
    Returns:
        callable: Function to create configuration with specific overrides
        
    Example:
        def test_parameter_overrides(hydra_config_with_overrides):
            cfg = hydra_config_with_overrides([
                "navigator.num_agents=5",
                "simulation.num_steps=200"
            ])
            assert cfg.navigator.num_agents == 5
            assert cfg.simulation.num_steps == 200
    """
    if not HYDRA_AVAILABLE:
        pytest.skip("Hydra not available for configuration testing")
    
    def _create_config_with_overrides(overrides: list = None):
        GlobalHydra.instance().clear()
        
        with initialize(config_path=str(test_config_dir), version_base=None):
            cfg = compose(config_name="config", overrides=overrides or [])
            return cfg
    
    return _create_config_with_overrides


@pytest.fixture
def mock_hydra_config():
    """
    Provide mock Hydra configuration for lightweight testing scenarios.
    
    Creates a mock DictConfig object with realistic configuration structure
    for tests that need configuration objects but don't require full Hydra
    composition testing. Useful for unit tests that focus on business logic
    rather than configuration management.
    
    Returns:
        Mock: Mock DictConfig object with realistic structure
        
    Example:
        def test_navigator_creation(mock_hydra_config):
            navigator = create_navigator_from_config(mock_hydra_config)
            assert navigator is not None
    """
    mock_config = Mock()
    mock_config.navigator = Mock()
    mock_config.navigator.type = "SingleAgentController"
    mock_config.navigator.num_agents = 1
    mock_config.navigator.max_speed = 1.0
    mock_config.navigator.angular_velocity = 0.5
    mock_config.navigator.sensor_strategy = "bilateral"
    mock_config.navigator.collision_avoidance = True
    
    mock_config.video_plume = Mock()
    mock_config.video_plume.video_path = "test_video.mp4"
    mock_config.video_plume.frame_width = 640
    mock_config.video_plume.frame_height = 480
    
    mock_config.simulation = Mock()
    mock_config.simulation.num_steps = 100
    mock_config.simulation.step_size = 0.1
    mock_config.simulation.initial_positions = [[0.0, 0.0]]
    mock_config.simulation.initial_orientations = [0.0]
    
    mock_config.seed = Mock()
    mock_config.seed.value = 42
    mock_config.seed.strict_reproducibility = True
    
    return mock_config


# =============================================================================
# CLI Testing Fixtures (Click CliRunner integration)
# =============================================================================

@pytest.fixture
def cli_runner_fixture() -> Generator[Tuple[CliRunner, Dict[str, str]], None, None]:
    """
    Provide Click CliRunner fixture for command-line interface testing.
    
    Creates an isolated CliRunner instance with temporary working directory
    and controlled environment variables. Supports parameter validation,
    command execution testing, and error handling verification for all
    CLI commands and subcommands.
    
    Yields:
        Tuple[CliRunner, Dict[str, str]]: CLI runner instance and environment variables
        
    Example:
        def test_cli_help_command(cli_runner_fixture):
            runner, env = cli_runner_fixture
            result = runner.invoke(main, ['--help'], env=env)
            assert result.exit_code == 0
            assert 'Usage:' in result.output
    """
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up isolated environment for CLI testing
        env = os.environ.copy()
        env.update({
            'HYDRA_WORKING_DIR': temp_dir,
            'PYTEST_RUNNING': 'true',
            'HYDRA_FULL_ERROR': '1',
            'TEST_VIDEO_PATH': os.path.join(temp_dir, 'test_video.mp4'),
            'LOG_LEVEL': 'DEBUG'
        })
        
        # Create minimal test video file for CLI testing
        test_video_path = os.path.join(temp_dir, 'test_video.mp4')
        with open(test_video_path, 'wb') as f:
            f.write(b'\x00' * 1024)  # Minimal placeholder file
        
        yield runner, env


@pytest.fixture
def mock_cli_runner():
    """
    Provide mock CLI runner for lightweight command testing.
    
    Creates a mock CliRunner that simulates command execution without
    actual subprocess invocation. Useful for unit tests focusing on
    command logic rather than CLI integration.
    
    Returns:
        Mock: Mock CliRunner with configurable result behavior
        
    Example:
        def test_command_logic(mock_cli_runner):
            mock_cli_runner.invoke.return_value.exit_code = 0
            result = mock_cli_runner.invoke(some_command, [])
            assert result.exit_code == 0
    """
    mock_runner = Mock(spec=CliRunner)
    mock_result = Mock()
    mock_result.exit_code = 0
    mock_result.output = "Mock command output\n"
    mock_result.exception = None
    mock_runner.invoke.return_value = mock_result
    
    return mock_runner


@pytest.fixture
def cli_isolated_fs(cli_runner_fixture):
    """
    Provide isolated filesystem for CLI command testing.
    
    Creates an isolated filesystem context for CLI commands that interact
    with files and directories. Ensures clean state for each test and
    prevents interference with actual filesystem.
    
    Args:
        cli_runner_fixture: CLI runner fixture with environment
        
    Yields:
        CliRunner: CLI runner in isolated filesystem context
        
    Example:
        def test_file_output_command(cli_isolated_fs):
            runner = cli_isolated_fs
            result = runner.invoke(export_command, ['--output', 'test.json'])
            assert os.path.exists('test.json')
    """
    runner, env = cli_runner_fixture
    
    with runner.isolated_filesystem():
        yield runner


# =============================================================================
# Database Session Testing Fixtures (SQLAlchemy integration)
# =============================================================================

@pytest.fixture(scope="session")
def db_engine():
    """
    Provide in-memory SQLite database engine for testing.
    
    Creates a session-scoped in-memory SQLite database engine with
    appropriate configuration for testing. Includes connection pooling
    configuration and echo settings for debugging.
    
    Returns:
        Engine: SQLAlchemy engine instance for testing
    """
    if not SQLALCHEMY_AVAILABLE:
        pytest.skip("SQLAlchemy not available for database testing")
    
    # Create in-memory SQLite database with appropriate configuration
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={
            'check_same_thread': False,
        },
        echo=False  # Set to True for SQL debugging
    )
    
    return engine


@pytest.fixture
def db_session_fixture(db_engine) -> Generator[Session, None, None]:
    """
    Provide isolated database session fixture for testing.
    
    Creates a transaction-isolated database session for each test.
    All database operations within the test are automatically rolled back
    after test completion, ensuring complete isolation between tests.
    
    Args:
        db_engine: Database engine fixture
        
    Yields:
        Session: SQLAlchemy session with transaction isolation
        
    Example:
        def test_database_operations(db_session_fixture):
            session = db_session_fixture
            # Perform database operations
            # All changes automatically rolled back after test
    """
    if not SQLALCHEMY_AVAILABLE:
        pytest.skip("SQLAlchemy not available for database testing")
    
    # Create all tables (would normally be done via schema creation)
    try:
        # Import and create database schema if available
        from src.{{cookiecutter.project_slug}}.db.session import Base
        Base.metadata.create_all(db_engine)
    except ImportError:
        # Schema not available, continue without table creation
        pass
    
    # Create session factory
    SessionLocal = sessionmaker(bind=db_engine)
    session = SessionLocal()
    
    try:
        yield session
    finally:
        # Roll back all changes and close session
        session.rollback()
        session.close()


@pytest.fixture
def mock_db_session():
    """
    Provide mock database session for lightweight testing.
    
    Creates a mock database session that simulates SQLAlchemy session
    behavior without actual database operations. Useful for unit tests
    that need session objects but don't require database interaction.
    
    Returns:
        Mock: Mock session with standard SQLAlchemy methods
        
    Example:
        def test_session_usage(mock_db_session):
            mock_db_session.query.return_value.filter.return_value.first.return_value = None
            result = some_function(mock_db_session)
            assert result is None
    """
    mock_session = Mock(spec=Session)
    mock_session.query.return_value = Mock()
    mock_session.add.return_value = None
    mock_session.commit.return_value = None
    mock_session.rollback.return_value = None
    mock_session.close.return_value = None
    
    return mock_session


# =============================================================================
# Scientific Computing and Video Processing Fixtures
# =============================================================================

@pytest.fixture
def mock_video_capture():
    """
    Provide mock OpenCV VideoCapture for video processing tests.
    
    Creates a comprehensive mock of cv2.VideoCapture with realistic
    behavior including frame generation, metadata access, and resource
    management. Supports configurable frame sequences and properties.
    
    Returns:
        Mock: Mock VideoCapture with realistic frame behavior
        
    Example:
        def test_video_processing(mock_video_capture):
            mock_video_capture.read.return_value = (True, np.zeros((480, 640, 3)))
            success, frame = mock_video_capture.read()
            assert success
            assert frame.shape == (480, 640, 3)
    """
    mock_capture = Mock(spec=cv2.VideoCapture)
    
    # Configure realistic video properties
    mock_capture.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 1000,
        cv2.CAP_PROP_POS_FRAMES: 0
    }.get(prop, 0)
    
    # Configure frame reading behavior
    frame_count = 0
    max_frames = 1000
    
    def mock_read():
        nonlocal frame_count
        if frame_count < max_frames:
            frame_count += 1
            # Generate synthetic frame data
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            return True, frame
        else:
            return False, None
    
    mock_capture.read.side_effect = mock_read
    mock_capture.isOpened.return_value = True
    mock_capture.release.return_value = None
    
    return mock_capture


@pytest.fixture
def mock_video_plume(mock_video_capture):
    """
    Provide mock VideoPlume environment for testing.
    
    Creates a mock VideoPlume instance with realistic behavior including
    frame access, metadata extraction, and configuration validation.
    Integrates with mock_video_capture for consistent video behavior.
    
    Args:
        mock_video_capture: Mock video capture fixture
        
    Returns:
        Mock: Mock VideoPlume with realistic environment behavior
        
    Example:
        def test_video_plume_usage(mock_video_plume):
            frame = mock_video_plume.get_frame(0)
            assert frame is not None
            assert mock_video_plume.width == 640
            assert mock_video_plume.height == 480
    """
    mock_plume = Mock()
    
    # Configure video properties
    mock_plume.width = 640
    mock_plume.height = 480
    mock_plume.fps = 30.0
    mock_plume.frame_count = 1000
    mock_plume.duration = 1000 / 30.0
    
    # Configure frame access
    def mock_get_frame(frame_idx):
        if 0 <= frame_idx < 1000:
            # Generate synthetic odor plume frame
            frame = np.random.exponential(scale=0.1, size=(480, 640))
            frame = np.clip(frame, 0, 1)
            return frame.astype(np.float32)
        else:
            raise IndexError(f"Frame index {frame_idx} out of range")
    
    mock_plume.get_frame.side_effect = mock_get_frame
    mock_plume.close.return_value = None
    
    # Configure metadata
    mock_plume.get_metadata_string.return_value = (
        "Test Video: 640x480 @ 30.0 FPS, 1000 frames (33.33s)"
    )
    
    return mock_plume


@pytest.fixture
def mock_navigator():
    """
    Provide mock Navigator for simulation testing.
    
    Creates a mock Navigator instance with realistic agent behavior
    including position tracking, orientation management, and sensor
    sampling. Supports both single-agent and multi-agent scenarios.
    
    Returns:
        Mock: Mock Navigator with realistic agent behavior
        
    Example:
        def test_simulation_step(mock_navigator):
            mock_navigator.step(np.zeros((480, 640)))
            assert mock_navigator.step.called
            positions = mock_navigator.positions
            assert positions.shape == (1, 2)  # Single agent, 2D position
    """
    mock_nav = Mock()
    
    # Configure agent properties
    num_agents = 1
    mock_nav.num_agents = num_agents
    mock_nav.positions = np.array([[0.0, 0.0]])
    mock_nav.orientations = np.array([0.0])
    mock_nav.speeds = np.array([0.0])
    mock_nav.max_speeds = np.array([1.0])
    mock_nav.angular_velocities = np.array([0.0])
    
    # Configure step behavior
    def mock_step(odor_frame):
        # Simulate agent movement
        mock_nav.positions += np.random.normal(0, 0.01, mock_nav.positions.shape)
        mock_nav.orientations += np.random.normal(0, 0.1, mock_nav.orientations.shape)
        mock_nav.speeds = np.random.uniform(0, 1, mock_nav.speeds.shape)
    
    mock_nav.step.side_effect = mock_step
    
    # Configure sensor sampling
    def mock_sample_odor(positions, odor_frame):
        # Simulate odor sampling
        return np.random.exponential(0.1, len(positions))
    
    mock_nav.sample_odor.side_effect = mock_sample_odor
    
    # Configure reset behavior
    def mock_reset():
        mock_nav.positions = np.array([[0.0, 0.0]])
        mock_nav.orientations = np.array([0.0])
        mock_nav.speeds = np.array([0.0])
        mock_nav.angular_velocities = np.array([0.0])
    
    mock_nav.reset.side_effect = mock_reset
    
    return mock_nav


# =============================================================================
# Seed Management and Reproducibility Fixtures
# =============================================================================

@pytest.fixture
def seed_fixture():
    """
    Provide deterministic seed management for reproducible testing.
    
    Sets up controlled random state for tests requiring deterministic
    behavior. Manages NumPy random state and Python random module
    to ensure reproducible test execution across different environments.
    
    Returns:
        int: Test seed value for reproducible randomization
        
    Example:
        def test_deterministic_behavior(seed_fixture):
            np.random.seed(seed_fixture)
            result1 = np.random.random(10)
            np.random.seed(seed_fixture)
            result2 = np.random.random(10)
            np.testing.assert_array_equal(result1, result2)
    """
    test_seed = 42
    
    # Store original random states
    original_np_state = np.random.get_state()
    
    # Set deterministic seeds
    np.random.seed(test_seed)
    
    try:
        yield test_seed
    finally:
        # Restore original random states
        np.random.set_state(original_np_state)


@pytest.fixture
def mock_seed_manager():
    """
    Provide mock seed manager for testing reproducibility components.
    
    Creates a mock seed manager that simulates seed management behavior
    without actually affecting global random state. Useful for testing
    seed management logic without side effects.
    
    Returns:
        Mock: Mock seed manager with realistic behavior
        
    Example:
        def test_seed_management(mock_seed_manager):
            mock_seed_manager.set_global_seed(123)
            assert mock_seed_manager.set_global_seed.called
            assert mock_seed_manager.get_current_seed() == 123
    """
    mock_manager = Mock()
    mock_manager.current_seed = 42
    
    def mock_set_global_seed(seed):
        mock_manager.current_seed = seed
    
    def mock_get_current_seed():
        return mock_manager.current_seed
    
    mock_manager.set_global_seed.side_effect = mock_set_global_seed
    mock_manager.get_current_seed.side_effect = mock_get_current_seed
    mock_manager.initialize_reproducibility.return_value = None
    
    return mock_manager


# =============================================================================
# Temporary File and Directory Management Fixtures
# =============================================================================

@pytest.fixture
def temp_config_files(tmp_path):
    """
    Provide temporary configuration files for testing.
    
    Creates realistic temporary configuration files with various
    parameter combinations for configuration loading and validation
    testing. Files are automatically cleaned up after test execution.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        Dict[str, Path]: Dictionary mapping config names to file paths
        
    Example:
        def test_config_loading(temp_config_files):
            config_path = temp_config_files['navigator']
            config = load_config(config_path)
            assert config is not None
    """
    import yaml
    
    config_files = {}
    
    # Navigator configuration
    navigator_config = {
        'type': 'SingleAgentController',
        'num_agents': 1,
        'max_speed': 1.0,
        'angular_velocity': 0.5,
        'sensor_strategy': 'bilateral',
        'collision_avoidance': True
    }
    
    navigator_path = tmp_path / "navigator.yaml"
    with open(navigator_path, "w") as f:
        yaml.dump(navigator_config, f)
    config_files['navigator'] = navigator_path
    
    # Video plume configuration
    video_config = {
        'video_path': 'test_video.mp4',
        'frame_width': 640,
        'frame_height': 480,
        'preprocessing': {
            'gaussian_blur': True,
            'horizontal_flip': False
        }
    }
    
    video_path = tmp_path / "video_plume.yaml"
    with open(video_path, "w") as f:
        yaml.dump(video_config, f)
    config_files['video_plume'] = video_path
    
    # Simulation configuration
    simulation_config = {
        'num_steps': 100,
        'step_size': 0.1,
        'initial_positions': [[0.0, 0.0]],
        'initial_orientations': [0.0]
    }
    
    simulation_path = tmp_path / "simulation.yaml"
    with open(simulation_path, "w") as f:
        yaml.dump(simulation_config, f)
    config_files['simulation'] = simulation_path
    
    return config_files


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Provide temporary output directory for testing file operations.
    
    Creates a temporary directory for output file testing including
    trajectory data, visualization exports, and log files. Directory
    is automatically cleaned up after test execution.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        Path: Temporary output directory path
        
    Example:
        def test_file_export(temp_output_dir):
            output_path = temp_output_dir / "trajectory.npy"
            save_trajectory(trajectory_data, output_path)
            assert output_path.exists()
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    
    # Create subdirectories for different output types
    (output_dir / "trajectories").mkdir()
    (output_dir / "visualizations").mkdir()
    (output_dir / "logs").mkdir()
    (output_dir / "configs").mkdir()
    
    return output_dir


# =============================================================================
# Integration and Cross-Component Testing Fixtures
# =============================================================================

@pytest.fixture
def full_system_mock(mock_navigator, mock_video_plume, mock_hydra_config):
    """
    Provide comprehensive system mock for integration testing.
    
    Combines all major system components (navigator, video plume, configuration)
    into a unified mock system for end-to-end integration testing. Supports
    realistic cross-component interaction patterns and data flow validation.
    
    Args:
        mock_navigator: Mock navigator fixture
        mock_video_plume: Mock video plume fixture
        mock_hydra_config: Mock configuration fixture
        
    Returns:
        Dict[str, Mock]: Dictionary containing all system component mocks
        
    Example:
        def test_full_simulation(full_system_mock):
            nav = full_system_mock['navigator']
            plume = full_system_mock['video_plume']
            config = full_system_mock['config']
            
            # Test integrated simulation workflow
            frame = plume.get_frame(0)
            nav.step(frame)
            assert nav.step.called
    """
    return {
        'navigator': mock_navigator,
        'video_plume': mock_video_plume,
        'config': mock_hydra_config,
        'seed_manager': mock_seed_manager(),
        'db_session': mock_db_session()
    }


@pytest.fixture
def performance_benchmark():
    """
    Provide performance benchmarking utilities for testing.
    
    Creates utilities for measuring and validating performance requirements
    including timing assertions, memory usage tracking, and throughput
    validation. Supports SLA requirement validation from technical specification.
    
    Returns:
        Dict[str, callable]: Performance measurement utilities
        
    Example:
        def test_performance_requirements(performance_benchmark):
            timer = performance_benchmark['timer']
            with timer('frame_processing') as t:
                process_frame(frame)
            assert t.elapsed < 0.033  # 33ms SLA requirement
    """
    import time
    import psutil
    from contextlib import contextmanager
    
    @contextmanager
    def timer(operation_name):
        """Context manager for timing operations."""
        class TimingResult:
            def __init__(self):
                self.elapsed = 0
                self.operation = operation_name
        
        result = TimingResult()
        start_time = time.perf_counter()
        try:
            yield result
        finally:
            result.elapsed = time.perf_counter() - start_time
    
    def memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def assert_performance_sla(elapsed_time, max_time, operation):
        """Assert performance SLA compliance."""
        assert elapsed_time <= max_time, (
            f"{operation} took {elapsed_time:.3f}s, "
            f"exceeds SLA of {max_time:.3f}s"
        )
    
    return {
        'timer': timer,
        'memory_usage': memory_usage,
        'assert_sla': assert_performance_sla
    }


# =============================================================================
# Test Data Generation Utilities
# =============================================================================

@pytest.fixture
def synthetic_test_data():
    """
    Provide synthetic test data for scientific computing validation.
    
    Generates realistic test data including odor fields, agent trajectories,
    and video sequences for comprehensive testing. All data generation is
    deterministic and reproducible across test runs.
    
    Returns:
        Dict[str, callable]: Test data generation functions
        
    Example:
        def test_odor_sampling(synthetic_test_data):
            odor_field = synthetic_test_data['odor_field'](width=640, height=480)
            assert odor_field.shape == (480, 640)
            assert 0 <= odor_field.max() <= 1
    """
    def generate_odor_field(width=640, height=480, seed=42):
        """Generate synthetic 2D odor field."""
        np.random.seed(seed)
        
        # Create Gaussian plume with realistic turbulence
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width // 4, height // 2
        
        # Base Gaussian plume
        base_field = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 50**2))
        
        # Add turbulence
        turbulence = np.random.normal(0, 0.1, (height, width))
        field = base_field + turbulence
        
        # Normalize and ensure non-negative
        field = np.clip(field, 0, None)
        field = field / field.max() if field.max() > 0 else field
        
        return field.astype(np.float32)
    
    def generate_trajectory(num_steps=100, num_agents=1, seed=42):
        """Generate synthetic agent trajectory."""
        np.random.seed(seed)
        
        positions = np.zeros((num_agents, num_steps, 2))
        orientations = np.zeros((num_agents, num_steps))
        
        for agent in range(num_agents):
            # Random walk with bias
            positions[agent, 0] = np.random.uniform(-1, 1, 2)
            orientations[agent, 0] = np.random.uniform(0, 2*np.pi)
            
            for step in range(1, num_steps):
                # Add movement with momentum
                delta_pos = 0.1 * np.array([
                    np.cos(orientations[agent, step-1]),
                    np.sin(orientations[agent, step-1])
                ]) + np.random.normal(0, 0.01, 2)
                
                positions[agent, step] = positions[agent, step-1] + delta_pos
                orientations[agent, step] = (
                    orientations[agent, step-1] + 
                    np.random.normal(0, 0.1)
                ) % (2 * np.pi)
        
        return positions, orientations
    
    def generate_video_sequence(num_frames=100, width=640, height=480, seed=42):
        """Generate synthetic video frame sequence."""
        np.random.seed(seed)
        
        frames = []
        for frame_idx in range(num_frames):
            # Evolving odor field
            field = generate_odor_field(width, height, seed + frame_idx)
            frames.append(field)
        
        return np.array(frames)
    
    return {
        'odor_field': generate_odor_field,
        'trajectory': generate_trajectory,
        'video_sequence': generate_video_sequence
    }


# =============================================================================
# Error Simulation and Edge Case Testing Fixtures
# =============================================================================

@pytest.fixture
def error_scenarios():
    """
    Provide error scenario simulation for robust testing.
    
    Creates utilities for simulating various error conditions including
    file I/O errors, network failures, invalid configurations, and
    resource exhaustion scenarios. Supports comprehensive error handling
    validation and recovery testing.
    
    Returns:
        Dict[str, callable]: Error simulation utilities
        
    Example:
        def test_error_handling(error_scenarios):
            with error_scenarios['file_not_found']():
                result = load_video_file('nonexistent.mp4')
                assert result is None
    """
    @contextmanager
    def file_not_found_error():
        """Simulate file not found errors."""
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            yield
    
    @contextmanager
    def invalid_video_error():
        """Simulate invalid video file errors."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap.return_value.isOpened.return_value = False
            yield
    
    @contextmanager
    def configuration_error():
        """Simulate configuration validation errors."""
        def raise_validation_error(*args, **kwargs):
            from pydantic import ValidationError
            raise ValidationError([], type('MockModel', (), {}))
        
        with patch('pydantic.BaseModel.validate', side_effect=raise_validation_error):
            yield
    
    @contextmanager
    def memory_error():
        """Simulate memory allocation errors."""
        with patch('numpy.zeros', side_effect=MemoryError("Out of memory")):
            yield
    
    return {
        'file_not_found': file_not_found_error,
        'invalid_video': invalid_video_error,
        'configuration_error': configuration_error,
        'memory_error': memory_error
    }


# =============================================================================
# Documentation and Coverage Testing Support
# =============================================================================

@pytest.fixture(autouse=True)
def test_documentation():
    """
    Automatic fixture providing test documentation and coverage support.
    
    This fixture automatically runs for all tests, providing enhanced
    documentation capabilities and coverage tracking integration. Supports
    test result documentation, failure analysis, and coverage reporting
    required by the technical specification.
    
    Note:
        This fixture runs automatically for all tests (autouse=True)
        and provides enhanced test infrastructure without explicit usage.
    """
    # Pre-test setup
    test_start_time = time.time()
    
    yield
    
    # Post-test cleanup and documentation
    test_duration = time.time() - test_start_time
    
    # Log test execution metadata (in production environment)
    if os.getenv('PYTEST_DETAILED_LOGGING'):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Test execution time: {test_duration:.3f}s")


# =============================================================================
# Module Cleanup and Resource Management
# =============================================================================

import time

def pytest_runtest_teardown(item, nextitem):
    """
    Clean up resources after each test execution.
    
    Performs test-level cleanup including:
    - Hydra global state cleanup
    - Mock object reset
    - Temporary file cleanup
    - Random state restoration
    
    Args:
        item: Current test item
        nextitem: Next test item (if any)
    """
    # Clean up Hydra global state if needed
    if HYDRA_AVAILABLE:
        try:
            GlobalHydra.instance().clear()
        except:
            pass  # Ignore cleanup errors
    
    # Reset scientific computing random state
    np.random.seed(None)


# =============================================================================
# Pytest Plugin Integration
# =============================================================================

# Additional pytest configuration can be added here for plugin integration
# such as pytest-xdist for parallel execution, pytest-benchmark for
# performance testing, or custom plugins for scientific computing workflows.

# Note: The comprehensive fixture infrastructure provided above supports
# the enhanced testing strategy requirements specified in Section 6.6 of
# the technical specification, providing >85% coverage capabilities for
# configuration and CLI modules while maintaining deterministic test
# behavior and scientific computing accuracy.