"""
Shared fixtures for tests in the odor_plume_nav package.

This file contains pytest fixtures that are shared across multiple test files
to reduce duplication and ensure consistency.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))


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
    Create a mock Navigator instance.
    
    This provides a consistent mock for tests that use SimpleNavigator.
    """
    mock = MagicMock()
    # Configure mock properties
    mock.orientation = 0.0
    mock.speed = 0.0
    mock.max_speed = 1.0
    # Configure return values for methods
    mock.get_position.return_value = (0.0, 0.0)
    mock.get_movement_vector.return_value = (0.0, 0.0)
    return mock


@pytest.fixture
def mock_exists():
    """Mock Path.exists to always return True."""
    with patch('pathlib.Path.exists', return_value=True):
        yield


@pytest.fixture
def config_files():
    """
    Fixture to provide test configuration data.
    
    Returns a dictionary with standard test configurations.
    """
    # Default configuration
    default_config = {
        "video_plume": {
            "flip": False,
            "kernel_size": 0,
            "kernel_sigma": 1.0
        },
        "navigator": {
            "orientation": 0.0,
            "speed": 0.0,
            "max_speed": 1.0
        }
    }
    
    # User configuration with some overrides
    user_config = {
        "video_plume": {
            "flip": True,
            "kernel_size": 5
        },
        "navigator": {
            "orientation": 45.0,
            "speed": 0.5,
            "max_speed": 2.0
        }
    }
    
    return {
        "default_config": default_config,
        "user_config": user_config
    }


@pytest.fixture
def temp_config_files(tmp_path, config_files):
    """
    Create temporary Hydra-compatible configuration files for testing.
    
    This fixture creates actual YAML files that match the Hydra hierarchical
    configuration structure, supporting both unit testing and integration
    testing scenarios that require real file system configuration loading.
    
    Args:
        tmp_path: Pytest built-in fixture providing temporary directory
        config_files: Configuration data fixture with hierarchical structure
    
    Returns:
        dict: Paths to created files and configuration objects
    
    Usage:
        def test_hydra_file_loading(temp_config_files):
            configs = temp_config_files
            base_path = configs["base_path"]
            # Test actual file loading with Hydra
            
        def test_config_file_composition(temp_config_files):
            configs = temp_config_files
            config_dir = configs["config_dir"]
            # Test hierarchical configuration file loading
    
    File Structure:
        Creates a realistic conf/ directory structure:
        - conf/base.yaml (foundation configuration)
        - conf/config.yaml (environment settings)
        - conf/local/development.yaml (dev overrides)
        - conf/local/production.yaml (prod settings)
        - conf/local/credentials.yaml.template (credential template)
    """
    import yaml
    
    # Get configuration data
    configs = config_files
    
    # Create Hydra-compatible directory structure
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    
    local_dir = config_dir / "local"
    local_dir.mkdir()
    
    # Create base configuration file (conf/base.yaml)
    base_path = config_dir / "base.yaml"
    with open(base_path, 'w') as f:
        yaml.dump(configs["base_config"], f, default_flow_style=False)
    
    # Create main configuration file (conf/config.yaml)  
    config_path = config_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(configs["user_config"], f, default_flow_style=False)
    
    # Create local development configuration (conf/local/development.yaml)
    local_dev_path = local_dir / "development.yaml"
    with open(local_dev_path, 'w') as f:
        yaml.dump(configs["local_config"], f, default_flow_style=False)
    
    # Create production configuration (conf/local/production.yaml)
    local_prod_path = local_dir / "production.yaml"
    with open(local_prod_path, 'w') as f:
        yaml.dump(configs["production_config"], f, default_flow_style=False)
    
    # Create credentials template (conf/local/credentials.yaml.template)
    credentials_template = {
        "database": {
            "username": "${oc.env:DB_USER}",
            "password": "${oc.env:DB_PASSWORD}"
        },
        "api_keys": {
            "visualization_service": "${oc.env:VIZ_API_KEY}",
            "data_storage": "${oc.env:STORAGE_KEY}"
        }
    }
    
    credentials_path = local_dir / "credentials.yaml.template"
    with open(credentials_path, 'w') as f:
        yaml.dump(credentials_template, f, default_flow_style=False)
    
    # Create paths configuration template (conf/local/paths.yaml.template)
    paths_template = {
        "data_dir": "${oc.env:DATA_DIR,/tmp/data}",
        "output_dir": "${oc.env:OUTPUT_DIR,/tmp/outputs}",
        "video_dir": "${oc.env:VIDEO_DIR,/tmp/videos}",
        "results_dir": "${oc.env:RESULTS_DIR,/tmp/results}"
    }
    
    paths_path = local_dir / "paths.yaml.template"
    with open(paths_path, 'w') as f:
        yaml.dump(paths_template, f, default_flow_style=False)
    
    return {
        "config_dir": config_dir,
        "base_path": base_path,
        "config_path": config_path,
        "local_dev_path": local_dev_path,
        "local_prod_path": local_prod_path,
        "credentials_path": credentials_path,
        "paths_path": paths_path,
        "base_config": configs["base_config"],
        "user_config": configs["user_config"],
        "local_config": configs["local_config"],
        "production_config": configs["production_config"]
    }


# =============================================================================
# TESTING INTEGRATION AND WORKFLOW FIXTURES
# =============================================================================

@pytest.fixture
def mock_workflow_runner():
    """
    Create a mock workflow runner for testing DVC and Snakemake integration.
    
    This fixture provides mock interfaces for workflow orchestration testing
    without requiring actual DVC or Snakemake installations, enabling
    comprehensive testing of workflow integration patterns.
    
    Returns:
        MagicMock: Workflow runner mock with DVC and Snakemake interfaces
    
    Usage:
        def test_dvc_pipeline_execution(mock_workflow_runner):
            runner = mock_workflow_runner
            result = runner.dvc.repro(dry_run=True)
            assert result.returncode == 0
            
        def test_snakemake_rule_validation(mock_workflow_runner):
            runner = mock_workflow_runner
            runner.snakemake.validate_workflow()
            assert runner.snakemake.validate_workflow.called
    
    Workflow Integration:
        Supports testing patterns for batch processing, experiment automation,
        and reproducible research pipeline validation without external
        workflow orchestration dependencies.
    """
    mock_runner = MagicMock()
    
    # Configure DVC mock interface
    mock_runner.dvc.repro.return_value = MagicMock(returncode=0, stdout="", stderr="")
    mock_runner.dvc.status.return_value = MagicMock(returncode=0, stdout="Data and pipelines are up to date.")
    mock_runner.dvc.pipeline.show.return_value = {"stages": [], "params": {}}
    
    # Configure Snakemake mock interface
    mock_runner.snakemake.validate_workflow.return_value = True
    mock_runner.snakemake.dry_run.return_value = MagicMock(success=True, jobs=[])
    mock_runner.snakemake.list_rules.return_value = ["simulate", "analyze", "visualize"]
    
    # Configure workflow execution context
    mock_runner.working_dir = "/tmp/workflow_test"
    mock_runner.config_overrides = {}
    
    return mock_runner


# =============================================================================
# COMPREHENSIVE TESTING DOCUMENTATION AND BEST PRACTICES
# =============================================================================

"""
TESTING BEST PRACTICES FOR {{cookiecutter.project_slug}}

This conftest.py provides comprehensive testing infrastructure supporting the enhanced
cookiecutter-based architecture with sophisticated configuration management, CLI interface
testing, database session validation, and workflow orchestration testing.

=== CONFIGURATION TESTING WITH PYTEST-HYDRA ===

pytest-hydra Integration:
    Install: pip install pytest-hydra
    Usage: @pytest.mark.hydra_config for automatic config injection
    
    Example:
        @pytest.mark.hydra_config(config_name="config", config_path="../conf")
        def test_hydra_integration(hydra_cfg):
            assert hydra_cfg.navigator.max_speed > 0

Hydra Configuration Composition Testing:
    def test_config_override_scenarios(mock_hydra_config):
        cfg = mock_hydra_config
        cfg.navigator.max_speed = 15.0
        navigator = create_navigator(cfg.navigator)
        assert navigator.max_speed == 15.0

Hierarchical Configuration Validation:
    def test_config_hierarchy_loading(temp_config_files):
        configs = temp_config_files
        # Test conf/base.yaml + conf/config.yaml + conf/local/*.yaml composition

Environment Variable Interpolation Testing:
    def test_env_var_interpolation(isolated_environment):
        env = isolated_environment
        env['DATABASE_URL'] = 'sqlite:///test.db'
        # Test ${oc.env:DATABASE_URL} interpolation


=== CLI INTERFACE TESTING WITH CLICK ===

CliRunner Pattern:
    def test_cli_command_execution(mock_cli_runner):
        runner, env = mock_cli_runner
        from {{cookiecutter.project_slug}}.cli.main import main
        result = runner.invoke(main, ['--help'], env=env)
        assert result.exit_code == 0

Parameter Validation Testing:
    def test_cli_parameter_types(mock_cli_runner):
        runner, env = mock_cli_runner
        result = runner.invoke(main, ['--max-speed', 'invalid'], env=env)
        assert result.exit_code != 0
        assert 'Invalid value' in result.output

Interactive CLI Testing:
    def test_cli_interactive_prompts(mock_cli_runner):
        runner, env = mock_cli_runner
        result = runner.invoke(main, ['interactive'], input='y\n', env=env)
        assert 'Continue?' in result.output

Error Handling Validation:
    def test_cli_error_scenarios(mock_cli_runner):
        runner, env = mock_cli_runner
        result = runner.invoke(main, ['--config-file', 'nonexistent.yaml'], env=env)
        assert result.exit_code != 0


=== DATABASE SESSION TESTING WITH SQLALCHEMY ===

In-Memory Session Testing:
    def test_database_operations(mock_db_session):
        session = mock_db_session
        # Perform database operations with automatic rollback
        # Session automatically cleaned up after test

Transaction Testing:
    def test_db_transaction_handling(mock_db_session):
        session = mock_db_session
        try:
            # Test transaction operations
            session.commit()
        except Exception:
            session.rollback()
            raise

Connection Lifecycle Testing:
    def test_db_connection_management(mock_db_session):
        session = mock_db_session
        assert session.is_active
        # Session lifecycle managed automatically


=== SEED MANAGEMENT FOR REPRODUCIBLE TESTING ===

Deterministic Testing:
    def test_reproducible_computation(mock_seed_manager):
        seed_manager = mock_seed_manager
        seed_manager.set_seed(42)
        
        result1 = np.random.random(10)
        seed_manager.set_seed(42)
        result2 = np.random.random(10)
        
        np.testing.assert_array_equal(result1, result2)

Seed Isolation:
    def test_seed_isolation_between_tests(mock_seed_manager):
        seed_manager = mock_seed_manager
        seed_manager.set_seed(12345)
        # Test operations isolated from global random state

Cross-Platform Reproducibility:
    def test_cross_platform_determinism(mock_seed_manager):
        seed_manager = mock_seed_manager
        seed_manager.set_seed(42)
        # Ensure consistent results across Windows, macOS, Linux


=== INTEGRATION TESTING PATTERNS ===

Full System Integration:
    def test_end_to_end_workflow(mock_hydra_config, mock_cli_runner, 
                                mock_db_session, mock_seed_manager):
        # Test complete workflow from CLI through database persistence

Configuration + CLI Integration:
    def test_cli_config_integration(mock_cli_runner, temp_config_files):
        runner, env = mock_cli_runner
        configs = temp_config_files
        result = runner.invoke(main, ['--config-path', str(configs['config_dir'])], env=env)

Workflow Orchestration Testing:
    def test_workflow_pipeline_validation(mock_workflow_runner):
        runner = mock_workflow_runner
        result = runner.dvc.repro(dry_run=True)
        assert result.returncode == 0


=== ENVIRONMENT ISOLATION AND CLEANUP ===

Environment Variable Management:
    def test_with_clean_environment(isolated_environment):
        env = isolated_environment
        env['CUSTOM_VAR'] = 'test_value'
        # Test with controlled environment

Temporary Directory Management:
    def test_file_operations(tmp_path):
        test_file = tmp_path / "test.yaml"
        # Use pytest's tmp_path for file operations

Hydra Working Directory Isolation:
    def test_hydra_isolation(mock_cli_runner):
        runner, env = mock_cli_runner
        # HYDRA_WORKING_DIR automatically isolated per test


=== PERFORMANCE AND TESTING EFFICIENCY ===

Mock Performance:
    - Use shared fixtures to reduce setup overhead
    - Mock expensive operations (video I/O, file system access)
    - Leverage pytest fixture scoping (function, class, module, session)

Memory Management:
    - Automatic cleanup through fixture teardown
    - In-memory databases prevent persistent state
    - Temporary directories automatically removed

Test Execution Speed:
    - Mocked dependencies enable fast test execution
    - Deterministic behavior reduces flaky test issues
    - Parallel execution supported through isolated fixtures


=== TESTING COMMAND REFERENCE ===

Basic Test Execution:
    pytest                              # Run all tests
    pytest tests/config/                # Test configuration management
    pytest tests/cli/                   # Test CLI interfaces
    pytest tests/db/                    # Test database sessions
    pytest -k "hydra"                   # Test Hydra-related functionality

Coverage Analysis:
    pytest --cov={{cookiecutter.project_slug}} --cov-report=html
    pytest --cov={{cookiecutter.project_slug}} --cov-report=xml --cov-fail-under=80

Performance Testing:
    pytest --benchmark-only             # Run performance benchmarks
    pytest -x --tb=short                # Stop on first failure

Configuration Testing:
    pytest tests/config/ -v             # Verbose configuration testing
    pytest -m "hydra"                   # Run only Hydra-marked tests

CLI Testing:
    pytest tests/cli/ --tb=line         # Test CLI with line-level tracebacks
    pytest -k "cli and parameter"       # Test CLI parameter validation

Database Testing:
    pytest tests/db/ -v                 # Verbose database testing
    pytest -k "session"                 # Test session management

Integration Testing:
    pytest tests/integration/ -v        # Full integration test suite
    pytest -k "workflow"                # Test workflow orchestration


This comprehensive testing framework ensures robust validation across all system
components while maintaining the scalability and efficiency required for modern
scientific computing development workflows.
"""
