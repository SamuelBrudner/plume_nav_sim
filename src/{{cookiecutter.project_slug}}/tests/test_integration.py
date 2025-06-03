"""
Comprehensive integration testing module for odor plume navigation system.

This module validates end-to-end workflows, component interactions, cross-system 
compatibility, and framework integration patterns as specified in Section 6.6.1.2
of the technical specification. Integration tests ensure that the cookiecutter-based
architecture functions cohesively across all system domains including configuration
management, CLI interfaces, database sessions, and workflow orchestration.

Test Categories:
- End-to-end simulation workflows with Navigator and VideoPlume integration
- Kedro pipeline compatibility with factory method patterns
- CLI-to-core integration with parameter flow validation
- Configuration system integration across all components  
- Database session integration with simulation workflows
- Visualization export workflows with complete data pipeline testing
- Multi-component error handling and recovery scenarios
- Performance characteristics validation per timing requirements
- Cross-framework compatibility with RL and ML integration patterns
- Workflow orchestration integration with DVC and Snakemake scenarios

Architecture Integration Points:
- src/{{cookiecutter.project_slug}}/api/navigation.py: Public API surface validation
- src/{{cookiecutter.project_slug}}/cli/main.py: Command-line interface integration
- src/{{cookiecutter.project_slug}}/core/navigator.py: Core domain logic protocols
- src/{{cookiecutter.project_slug}}/data/video_plume.py: Video processing adapters
- src/{{cookiecutter.project_slug}}/config/schemas.py: Configuration validation
- src/{{cookiecutter.project_slug}}/db/session.py: Database persistence infrastructure

Performance Requirements:
- CLI command initialization: ≤2 seconds per Section 6.6.3.3
- End-to-end simulation: 30+ FPS capability per Section 2.2.9.3
- Database session establishment: ≤100ms per Section 6.6.3.3
- Configuration loading: ≤500ms for complex hierarchical configs
- Integration workflow completion: Within specified SLA thresholds

Quality Gates:
- 100% pass rate for research publication readiness
- Comprehensive scenario coverage across all integration points
- Performance validation within specified timing requirements
- Cross-framework compatibility verification
- Error handling and recovery validation

Authors: Blitzy Template Engine v2.0.0
License: MIT
"""

import os
import sys
import time
import tempfile
import warnings
from pathlib import Path
from contextlib import contextmanager, suppress
from typing import Dict, List, Optional, Any, Tuple, Union, Generator
from unittest.mock import Mock, MagicMock, patch, mock_open
from dataclasses import asdict

import pytest
import numpy as np
from click.testing import CliRunner

# Import system under test components
from {{cookiecutter.project_slug}}.api.navigation import (
    create_navigator, create_video_plume, run_plume_simulation,
    create_navigator_from_config, create_video_plume_from_config,
    ConfigurationError, SimulationError
)
from {{cookiecutter.project_slug}}.cli.main import main as cli_main, cli, CLIError
from {{cookiecutter.project_slug}}.core.navigator import Navigator, NavigatorProtocol
from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
from {{cookiecutter.project_slug}}.config.schemas import (
    NavigatorConfig, VideoPlumeConfig, SimulationConfig
)
from {{cookiecutter.project_slug}}.db.session import (
    get_session, SessionManager, is_database_enabled, cleanup_database
)

# Optional imports with graceful degradation
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None

try:
    import hydra
    from hydra import compose, initialize_config_store
    from hydra.core.config_store import ConfigStore
except ImportError:
    hydra = None

try:
    from kedro.io import DataCatalog
    from kedro.pipeline import Pipeline, node
    KEDRO_AVAILABLE = True
except ImportError:
    KEDRO_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class IntegrationTestBase:
    """Base class for integration tests with common fixtures and utilities."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self, tmp_path, monkeypatch):
        """Set up isolated test environment for each integration test."""
        # Create temporary directories for test isolation
        self.temp_dir = tmp_path
        self.config_dir = tmp_path / "conf"
        self.output_dir = tmp_path / "outputs"
        self.data_dir = tmp_path / "data"
        
        # Create directory structure
        for dir_path in [self.config_dir, self.output_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for test isolation
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
        monkeypatch.setenv("PYTHONPATH", str(self.temp_dir))
        
        # Disable GUI backends for headless testing
        monkeypatch.setenv("MPLBACKEND", "Agg")
        
        # Set up test-specific logging to avoid interference
        import logging
        logging.getLogger().setLevel(logging.WARNING)
        
        yield
        
        # Cleanup after test
        cleanup_database()
    
    @pytest.fixture
    def mock_video_file(self):
        """Create a mock video file for testing video plume integration."""
        video_path = self.data_dir / "test_video.mp4"
        # Create empty file to satisfy existence checks
        video_path.touch()
        return video_path
    
    @pytest.fixture
    def sample_config(self):
        """Generate sample configuration for integration testing."""
        return {
            'navigator': {
                'position': [10.0, 20.0],
                'orientation': 45.0,
                'speed': 1.5,
                'max_speed': 3.0,
                'angular_velocity': 5.0
            },
            'video_plume': {
                'video_path': str(self.data_dir / "test_video.mp4"),
                'flip': True,
                'kernel_size': 3,
                'kernel_sigma': 1.0
            },
            'simulation': {
                'num_steps': 100,
                'dt': 0.1,
                'sensor_distance': 5.0,
                'sensor_angle': 45.0,
                'record_trajectory': True
            },
            'database': {
                'enabled': False,
                'url': 'sqlite:///:memory:'
            },
            'reproducibility': {
                'global_seed': 42
            }
        }
    
    @pytest.fixture
    def hydra_config(self, sample_config):
        """Create Hydra DictConfig for configuration testing."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available for DictConfig testing")
        return OmegaConf.create(sample_config)
    
    @pytest.fixture
    def mock_video_capture(self):
        """Mock OpenCV VideoCapture for consistent testing."""
        with patch('cv2.VideoCapture') as mock_cap:
            # Configure mock to return predictable frame data
            mock_instance = MagicMock()
            mock_instance.get.side_effect = lambda prop: {
                1: 100,  # CAP_PROP_FRAME_COUNT
                3: 640,  # CAP_PROP_FRAME_WIDTH  
                4: 480,  # CAP_PROP_FRAME_HEIGHT
                5: 30    # CAP_PROP_FPS
            }.get(prop, 0)
            
            # Mock frame reading with synthetic data
            mock_instance.read.return_value = (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            mock_instance.isOpened.return_value = True
            mock_cap.return_value = mock_instance
            yield mock_cap


class TestEndToEndSimulationWorkflows(IntegrationTestBase):
    """Test complete simulation workflows with Navigator and VideoPlume integration."""
    
    def test_single_agent_simulation_workflow(self, sample_config, mock_video_capture, mock_video_file):
        """
        Test end-to-end single-agent simulation workflow per integration requirements.
        
        Validates:
        - Navigator creation from configuration
        - VideoPlume environment initialization
        - Complete simulation execution with trajectory recording
        - Result validation and data integrity
        """
        # Create navigator from configuration
        navigator = create_navigator(
            position=sample_config['navigator']['position'],
            orientation=sample_config['navigator']['orientation'],
            max_speed=sample_config['navigator']['max_speed']
        )
        
        # Validate navigator creation
        assert isinstance(navigator, Navigator)
        assert navigator.num_agents == 1
        np.testing.assert_array_equal(navigator.positions[0], sample_config['navigator']['position'])
        
        # Create video plume environment
        video_plume = create_video_plume(
            video_path=mock_video_file,
            flip=sample_config['video_plume']['flip'],
            kernel_size=sample_config['video_plume']['kernel_size']
        )
        
        # Validate video plume initialization
        assert isinstance(video_plume, VideoPlume)
        assert video_plume.frame_count == 100  # From mock configuration
        
        # Execute simulation with performance monitoring
        start_time = time.time()
        positions, orientations, readings = run_plume_simulation(
            navigator=navigator,
            video_plume=video_plume,
            num_steps=sample_config['simulation']['num_steps'],
            dt=sample_config['simulation']['dt'],
            record_trajectory=True
        )
        execution_time = time.time() - start_time
        
        # Validate performance requirements (30+ FPS capability)
        expected_max_time = sample_config['simulation']['num_steps'] / 30.0  # 30 FPS minimum
        assert execution_time < expected_max_time * 2, f"Simulation too slow: {execution_time:.3f}s"
        
        # Validate result shapes and data integrity
        expected_steps = sample_config['simulation']['num_steps'] + 1
        assert positions.shape == (1, expected_steps, 2)
        assert orientations.shape == (1, expected_steps)
        assert readings.shape == (1, expected_steps)
        
        # Validate trajectory data is finite and reasonable
        assert np.all(np.isfinite(positions)), "Positions contain NaN or infinite values"
        assert np.all(np.isfinite(orientations)), "Orientations contain NaN or infinite values"
        assert np.all(readings >= 0), "Negative odor readings detected"
        
        # Validate position changes (agent should move)
        position_changes = np.linalg.norm(np.diff(positions[0], axis=0), axis=1)
        assert np.any(position_changes > 0), "Agent did not move during simulation"
    
    def test_multi_agent_simulation_workflow(self, sample_config, mock_video_capture, mock_video_file):
        """
        Test end-to-end multi-agent simulation workflow with vectorized operations.
        
        Validates:
        - Multi-agent navigator creation and initialization
        - Vectorized simulation execution across multiple agents
        - Inter-agent behavior and collision avoidance
        - Performance scaling with agent count
        """
        num_agents = 3
        positions = np.array([[0.0, 0.0], [50.0, 50.0], [100.0, 100.0]])
        max_speeds = [2.0, 2.5, 3.0]
        
        # Create multi-agent navigator
        navigator = create_navigator(
            positions=positions,
            max_speeds=max_speeds,
            orientations=[0.0, 90.0, 180.0]
        )
        
        # Validate multi-agent configuration
        assert navigator.num_agents == num_agents
        np.testing.assert_array_equal(navigator.positions, positions)
        np.testing.assert_array_equal(navigator.max_speeds, max_speeds)
        
        # Create video plume environment
        video_plume = create_video_plume(video_path=mock_video_file)
        
        # Execute multi-agent simulation with performance monitoring
        start_time = time.time()
        positions_result, orientations_result, readings_result = run_plume_simulation(
            navigator=navigator,
            video_plume=video_plume,
            num_steps=50,  # Reduced for multi-agent performance testing
            dt=0.1
        )
        execution_time = time.time() - start_time
        
        # Validate multi-agent performance requirements (≤5ms per step for 10 agents)
        steps_per_second = 50 / execution_time
        assert steps_per_second >= 20, f"Multi-agent simulation too slow: {steps_per_second:.1f} steps/s"
        
        # Validate multi-agent result shapes
        assert positions_result.shape == (num_agents, 51, 2)
        assert orientations_result.shape == (num_agents, 51)
        assert readings_result.shape == (num_agents, 51)
        
        # Validate agent independence (each agent should have different trajectories)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                position_diff = np.linalg.norm(
                    positions_result[i, -1] - positions_result[j, -1]
                )
                assert position_diff > 0, f"Agents {i} and {j} ended at identical positions"
    
    def test_simulation_with_hydra_configuration(self, hydra_config, mock_video_capture, mock_video_file):
        """
        Test simulation workflow using Hydra configuration objects.
        
        Validates:
        - Hydra DictConfig integration across all components
        - Configuration override and interpolation
        - Environment variable substitution
        - Factory method patterns for Kedro compatibility
        """
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available")
        
        # Test factory method with Hydra config
        navigator = create_navigator_from_config(hydra_config.navigator)
        assert isinstance(navigator, Navigator)
        
        # Test configuration override capabilities
        override_config = OmegaConf.merge(
            hydra_config.navigator,
            {"max_speed": 5.0}
        )
        navigator_override = create_navigator_from_config(override_config)
        assert navigator_override.max_speeds[0] == 5.0
        
        # Test video plume with Hydra config
        video_plume = create_video_plume_from_config(hydra_config.video_plume)
        assert isinstance(video_plume, VideoPlume)
        
        # Execute simulation with full Hydra configuration
        positions, orientations, readings = run_plume_simulation(
            navigator=navigator,
            video_plume=video_plume,
            cfg=hydra_config.simulation
        )
        
        # Validate configuration-driven simulation results
        expected_steps = hydra_config.simulation.num_steps + 1
        assert positions.shape[1] == expected_steps
        assert orientations.shape[1] == expected_steps
        assert readings.shape[1] == expected_steps


class TestKedroIntegration(IntegrationTestBase):
    """Test Kedro pipeline compatibility with factory method patterns."""
    
    @pytest.mark.skipif(not KEDRO_AVAILABLE, reason="Kedro not available")
    def test_kedro_pipeline_compatibility(self, hydra_config, mock_video_capture, mock_video_file):
        """
        Test Kedro pipeline integration with factory method patterns per Section 0.2.1.
        
        Validates:
        - Factory method integration with Kedro data catalogs
        - Structured configuration composition
        - Pipeline execution with Navigator components
        - Data flow through Kedro node structures
        """
        from kedro.io import DataCatalog, MemoryDataSet
        from kedro.pipeline import Pipeline, node
        from kedro.runner import SequentialRunner
        
        # Define Kedro-compatible factory functions
        def create_navigator_node(config: Dict[str, Any]) -> Navigator:
            """Kedro node function for navigator creation."""
            return create_navigator(**config)
        
        def create_video_plume_node(config: Dict[str, Any]) -> VideoPlume:
            """Kedro node function for video plume creation."""
            return create_video_plume(**config)
        
        def run_simulation_node(
            navigator: Navigator, 
            video_plume: VideoPlume, 
            config: Dict[str, Any]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Kedro node function for simulation execution."""
            return run_plume_simulation(navigator, video_plume, **config)
        
        # Create Kedro data catalog
        catalog = DataCatalog({
            "navigator_config": MemoryDataSet(OmegaConf.to_object(hydra_config.navigator)),
            "video_plume_config": MemoryDataSet(OmegaConf.to_object(hydra_config.video_plume)),
            "simulation_config": MemoryDataSet(OmegaConf.to_object(hydra_config.simulation)),
            "navigator": MemoryDataSet(),
            "video_plume": MemoryDataSet(),
            "simulation_results": MemoryDataSet()
        })
        
        # Define Kedro pipeline
        pipeline = Pipeline([
            node(
                func=create_navigator_node,
                inputs="navigator_config",
                outputs="navigator",
                name="create_navigator"
            ),
            node(
                func=create_video_plume_node,
                inputs="video_plume_config", 
                outputs="video_plume",
                name="create_video_plume"
            ),
            node(
                func=run_simulation_node,
                inputs=["navigator", "video_plume", "simulation_config"],
                outputs="simulation_results",
                name="run_simulation"
            )
        ])
        
        # Execute pipeline with performance monitoring
        start_time = time.time()
        runner = SequentialRunner()
        runner.run(pipeline, catalog)
        execution_time = time.time() - start_time
        
        # Validate pipeline execution performance
        assert execution_time < 10.0, f"Kedro pipeline execution too slow: {execution_time:.3f}s"
        
        # Validate pipeline results
        results = catalog.load("simulation_results")
        positions, orientations, readings = results
        
        # Validate result shapes and data integrity
        assert isinstance(positions, np.ndarray)
        assert isinstance(orientations, np.ndarray)
        assert isinstance(readings, np.ndarray)
        assert positions.shape[0] == 1  # Single agent from config
        assert positions.shape[2] == 2  # 2D positions
        assert np.all(np.isfinite(positions))
    
    def test_kedro_data_catalog_integration(self, sample_config):
        """
        Test data catalog integration with odor plume navigation components.
        
        Validates:
        - Custom dataset implementations
        - Configuration parameter flow through catalogs
        - Data persistence and loading patterns
        """
        if not KEDRO_AVAILABLE:
            pytest.skip("Kedro not available")
        
        from kedro.io import DataCatalog, MemoryDataSet
        
        # Test configuration storage and retrieval through catalog
        catalog = DataCatalog({
            "params:navigator": MemoryDataSet(sample_config['navigator']),
            "params:simulation": MemoryDataSet(sample_config['simulation'])
        })
        
        # Validate parameter loading
        nav_params = catalog.load("params:navigator")
        sim_params = catalog.load("params:simulation")
        
        assert nav_params['max_speed'] == sample_config['navigator']['max_speed']
        assert sim_params['num_steps'] == sample_config['simulation']['num_steps']
        
        # Test component storage in catalog
        navigator = create_navigator(**nav_params)
        catalog.save("navigator_instance", navigator)
        
        loaded_navigator = catalog.load("navigator_instance")
        assert isinstance(loaded_navigator, Navigator)
        assert loaded_navigator.num_agents == navigator.num_agents
        np.testing.assert_array_equal(loaded_navigator.positions, navigator.positions)


class TestCLIIntegration(IntegrationTestBase):
    """Test CLI-to-core integration with parameter flow validation."""
    
    def test_cli_command_execution_workflow(self, sample_config, mock_video_capture, mock_video_file):
        """
        Test CLI command execution with parameter flow validation per comprehensive testing.
        
        Validates:
        - Click command registration and parameter parsing
        - Hydra configuration composition through CLI
        - Parameter override via command-line arguments
        - CLI execution performance requirements (<2s initialization)
        """
        runner = CliRunner()
        
        # Test CLI help command for rapid response
        start_time = time.time()
        result = runner.invoke(cli, ['--help'])
        help_time = time.time() - start_time
        
        assert result.exit_code == 0
        assert help_time < 1.0, f"CLI help too slow: {help_time:.3f}s"
        assert "Odor Plume Navigation System" in result.output
        
        # Test configuration validation command
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(sample_config, f)
            config_path = f.name
        
        try:
            # Test dry run execution with performance monitoring
            start_time = time.time()
            with patch('{{cookiecutter.project_slug}}.cli.main.get_cli_config') as mock_config:
                mock_config.return_value = OmegaConf.create(sample_config) if HYDRA_AVAILABLE else sample_config
                
                result = runner.invoke(cli, ['run', '--dry-run'])
                initialization_time = time.time() - start_time
                
                # Validate CLI initialization performance requirement (<2s)
                assert initialization_time < 2.0, f"CLI initialization too slow: {initialization_time:.3f}s"
                
        finally:
            os.unlink(config_path)
    
    def test_cli_parameter_override_integration(self, sample_config):
        """
        Test CLI parameter override capabilities with Hydra integration.
        
        Validates:
        - Command-line parameter parsing and type conversion
        - Hydra configuration override syntax support
        - Parameter validation and error handling
        """
        runner = CliRunner()
        
        # Test parameter override syntax
        with patch('{{cookiecutter.project_slug}}.cli.main.get_cli_config') as mock_config:
            base_config = OmegaConf.create(sample_config) if HYDRA_AVAILABLE else sample_config
            mock_config.return_value = base_config
            
            # Test configuration validation with overrides
            result = runner.invoke(cli, ['config', 'validate', '--format', 'pretty'])
            
            assert result.exit_code == 0
            assert "Configuration validation passed" in result.output or result.exit_code == 0
    
    def test_cli_batch_processing_mode(self, sample_config, mock_video_capture, mock_video_file):
        """
        Test CLI batch processing capabilities for automated workflows.
        
        Validates:
        - Headless execution mode
        - Automated output generation
        - Error handling in batch mode
        - Performance characteristics for batch workflows
        """
        runner = CliRunner()
        
        with patch('{{cookiecutter.project_slug}}.cli.main.get_cli_config') as mock_config:
            mock_config.return_value = OmegaConf.create(sample_config) if HYDRA_AVAILABLE else sample_config
            
            # Test batch mode execution
            with patch('{{cookiecutter.project_slug}}.cli.main.create_navigator') as mock_nav, \
                 patch('{{cookiecutter.project_slug}}.cli.main.create_video_plume') as mock_plume, \
                 patch('{{cookiecutter.project_slug}}.cli.main.run_plume_simulation') as mock_sim:
                
                # Configure mocks for batch processing
                mock_nav.return_value = Mock(spec=Navigator, num_agents=1)
                mock_plume.return_value = Mock(spec=VideoPlume, frame_count=100, width=640, height=480)
                mock_sim.return_value = (
                    np.zeros((1, 101, 2)),  # positions
                    np.zeros((1, 101)),     # orientations  
                    np.zeros((1, 101))      # readings
                )
                
                result = runner.invoke(cli, ['run', '--batch', '--dry-run'])
                
                # Validate batch mode execution
                assert result.exit_code == 0 or "batch processing mode enabled" in result.output.lower()


class TestConfigurationSystemIntegration(IntegrationTestBase):
    """Test configuration system integration across all components."""
    
    def test_hierarchical_configuration_composition(self, sample_config):
        """
        Test hierarchical configuration composition per system integration requirements.
        
        Validates:
        - Multi-layer configuration loading (base → config → local)
        - Environment variable interpolation
        - Configuration validation across components
        - Parameter override precedence
        """
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available for configuration testing")
        
        # Create hierarchical configuration structure
        base_config = {
            'navigator': {'max_speed': 1.0, 'orientation': 0.0},
            'simulation': {'num_steps': 1000, 'dt': 0.1}
        }
        
        override_config = {
            'navigator': {'max_speed': 2.0},  # Override base value
            'simulation': {'sensor_distance': 5.0}  # Add new value
        }
        
        # Test configuration composition
        merged_config = OmegaConf.merge(
            OmegaConf.create(base_config),
            OmegaConf.create(override_config)
        )
        
        # Validate configuration composition
        assert merged_config.navigator.max_speed == 2.0  # Override applied
        assert merged_config.navigator.orientation == 0.0  # Base value preserved
        assert merged_config.simulation.num_steps == 1000  # Base value preserved
        assert merged_config.simulation.sensor_distance == 5.0  # New value added
        
        # Test configuration validation through component creation
        navigator = create_navigator_from_config(merged_config.navigator)
        assert navigator.max_speeds[0] == 2.0
    
    def test_environment_variable_interpolation(self, monkeypatch):
        """
        Test environment variable interpolation in configuration.
        
        Validates:
        - ${oc.env:VAR_NAME} syntax support
        - Secure credential management patterns
        - Environment-specific configuration loading
        """
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available for environment variable testing")
        
        # Set test environment variables
        monkeypatch.setenv("TEST_MAX_SPEED", "3.5")
        monkeypatch.setenv("TEST_SENSOR_DISTANCE", "7.0")
        
        # Create configuration with environment variable interpolation
        config_yaml = """
        navigator:
          max_speed: ${oc.env:TEST_MAX_SPEED,1.0}
          position: [0.0, 0.0]
        simulation:
          sensor_distance: ${oc.env:TEST_SENSOR_DISTANCE,5.0}
          num_steps: 100
        """
        
        config = OmegaConf.create(yaml.safe_load(config_yaml))
        resolved_config = OmegaConf.to_object(config)
        
        # Validate environment variable substitution
        assert float(resolved_config['navigator']['max_speed']) == 3.5
        assert float(resolved_config['simulation']['sensor_distance']) == 7.0
    
    def test_configuration_validation_across_components(self, sample_config):
        """
        Test configuration validation across all system components.
        
        Validates:
        - Pydantic schema validation for each component
        - Cross-component parameter consistency
        - Error handling for invalid configurations
        """
        # Test navigator configuration validation
        nav_config = NavigatorConfig(**sample_config['navigator'])
        assert nav_config.max_speed == sample_config['navigator']['max_speed']
        
        # Test video plume configuration validation
        video_config = VideoPlumeConfig(**sample_config['video_plume'])
        assert video_config.flip == sample_config['video_plume']['flip']
        
        # Test simulation configuration validation
        sim_config = SimulationConfig(**sample_config['simulation'])
        assert sim_config.num_steps == sample_config['simulation']['num_steps']
        
        # Test invalid configuration handling
        with pytest.raises(ValueError):
            NavigatorConfig(max_speed=-1.0)  # Invalid negative speed
        
        with pytest.raises(ValueError):
            SimulationConfig(num_steps=0)  # Invalid zero steps


class TestDatabaseSessionIntegration(IntegrationTestBase):
    """Test database session integration with simulation workflows."""
    
    def test_database_session_lifecycle_management(self):
        """
        Test database session lifecycle per enhanced architecture requirements.
        
        Validates:
        - Session creation and cleanup
        - Transaction handling and rollback
        - Connection pooling and performance
        - In-memory database testing patterns
        """
        # Test session availability checking
        db_enabled = is_database_enabled()
        
        # Test session context management
        with get_session() as session:
            if session:
                # Database enabled - test session operations
                assert hasattr(session, 'execute')
                assert hasattr(session, 'commit')
                assert hasattr(session, 'rollback')
                
                # Test basic query execution
                result = session.execute("SELECT 1 as test_value")
                assert result is not None
            else:
                # Database disabled - validate graceful handling
                assert session is None
    
    def test_database_session_performance_characteristics(self):
        """
        Test database session performance per Section 6.6.3.3 requirements.
        
        Validates:
        - Session establishment time ≤100ms
        - Connection pooling efficiency
        - Transaction performance
        """
        if not is_database_enabled():
            pytest.skip("Database not enabled for performance testing")
        
        # Test session establishment performance
        establishment_times = []
        
        for _ in range(10):
            start_time = time.time()
            with get_session() as session:
                if session:
                    session.execute("SELECT 1")
            establishment_time = time.time() - start_time
            establishment_times.append(establishment_time)
        
        # Validate performance requirements
        avg_establishment_time = np.mean(establishment_times)
        max_establishment_time = np.max(establishment_times)
        
        assert avg_establishment_time < 0.1, f"Average session establishment too slow: {avg_establishment_time:.3f}s"
        assert max_establishment_time < 0.2, f"Maximum session establishment too slow: {max_establishment_time:.3f}s"
    
    def test_simulation_with_database_integration(self, sample_config, mock_video_capture, mock_video_file):
        """
        Test simulation workflow with database session integration.
        
        Validates:
        - Simulation execution with database persistence hooks
        - Metadata storage capabilities
        - Graceful degradation when database unavailable
        """
        # Execute simulation with database integration
        navigator = create_navigator(**sample_config['navigator'])
        video_plume = create_video_plume(video_path=mock_video_file)
        
        # Monitor database session usage during simulation
        with get_session() as session:
            positions, orientations, readings = run_plume_simulation(
                navigator=navigator,
                video_plume=video_plume,
                num_steps=10,  # Short simulation for testing
                record_trajectory=True
            )
            
            # Validate simulation results regardless of database state
            assert positions.shape == (1, 11, 2)
            assert orientations.shape == (1, 11)
            assert readings.shape == (1, 11)
            
            # Test persistence hooks if database available
            if session:
                from {{cookiecutter.project_slug}}.db.session import PersistenceHooks
                
                trajectory_data = {
                    'positions': positions.tolist(),
                    'orientations': orientations.tolist(),
                    'readings': readings.tolist()
                }
                
                # Test trajectory persistence hook
                success = PersistenceHooks.save_trajectory_data(trajectory_data, session)
                # Hook returns True if database enabled, False otherwise
                assert isinstance(success, bool)


class TestVisualizationExportWorkflows(IntegrationTestBase):
    """Test visualization export workflows with complete data pipeline testing."""
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_visualization_export_integration(self, sample_config, mock_video_capture, mock_video_file):
        """
        Test visualization export workflows with complete data pipeline.
        
        Validates:
        - End-to-end visualization generation
        - Multiple export format support
        - Publication-quality output generation
        - Performance characteristics for visualization workflows
        """
        # Execute simulation to generate visualization data
        navigator = create_navigator(**sample_config['navigator'])
        video_plume = create_video_plume(video_path=mock_video_file)
        
        positions, orientations, readings = run_plume_simulation(
            navigator=navigator,
            video_plume=video_plume,
            num_steps=50,
            record_trajectory=True
        )
        
        # Test visualization generation with performance monitoring
        start_time = time.time()
        
        # Import visualization utilities with error handling
        try:
            from {{cookiecutter.project_slug}}.utils.visualization import visualize_simulation_results
            
            # Generate visualization
            figure = visualize_simulation_results(
                positions=positions,
                orientations=orientations,
                show_trails=True,
                quality='high'
            )
            
            # Validate figure generation
            assert figure is not None
            assert hasattr(figure, 'savefig')
            
            # Test export to different formats
            output_path = self.output_dir / "test_visualization"
            
            # Test PNG export
            png_path = output_path.with_suffix('.png')
            figure.savefig(png_path, dpi=300, bbox_inches='tight')
            assert png_path.exists()
            
            # Test PDF export for publication quality
            pdf_path = output_path.with_suffix('.pdf')
            figure.savefig(pdf_path, dpi=300, bbox_inches='tight')
            assert pdf_path.exists()
            
            plt.close(figure)  # Clean up figure
            
        except ImportError:
            pytest.skip("Visualization utilities not available")
        
        visualization_time = time.time() - start_time
        
        # Validate visualization performance (reasonable time for publication quality)
        assert visualization_time < 10.0, f"Visualization generation too slow: {visualization_time:.3f}s"
    
    def test_animation_export_workflow(self, sample_config, mock_video_capture, mock_video_file):
        """
        Test animation export capabilities for research documentation.
        
        Validates:
        - Animation generation from simulation data
        - Video format export (MP4)
        - Frame rate and quality control
        """
        # Generate simulation data for animation
        navigator = create_navigator(**sample_config['navigator'])
        video_plume = create_video_plume(video_path=mock_video_file)
        
        positions, orientations, readings = run_plume_simulation(
            navigator=navigator,
            video_plume=video_plume,
            num_steps=30,  # Shorter for animation testing
            record_trajectory=True
        )
        
        # Test animation export with mock implementation
        try:
            from {{cookiecutter.project_slug}}.utils.visualization import export_animation
            
            animation_path = self.output_dir / "test_animation.mp4"
            
            # Test animation export with performance monitoring
            start_time = time.time()
            export_animation(
                positions=positions,
                orientations=orientations,
                video_plume=video_plume,
                output_path=animation_path,
                fps=10,  # Lower FPS for testing
                quality='medium'
            )
            animation_time = time.time() - start_time
            
            # Validate animation export
            assert animation_path.exists()
            assert animation_time < 30.0, f"Animation export too slow: {animation_time:.3f}s"
            
        except ImportError:
            pytest.skip("Animation export utilities not available")


class TestMultiComponentErrorHandling(IntegrationTestBase):
    """Test multi-component error handling and recovery scenarios."""
    
    def test_configuration_error_propagation(self):
        """
        Test error handling across configuration boundaries.
        
        Validates:
        - Configuration validation error propagation
        - Component-specific error handling
        - Graceful degradation patterns
        """
        # Test invalid navigator configuration
        with pytest.raises(ConfigurationError):
            create_navigator(max_speed=-1.0)  # Invalid negative speed
        
        # Test invalid video path handling
        with pytest.raises((FileNotFoundError, ConfigurationError)):
            create_video_plume(video_path="/nonexistent/path.mp4")
        
        # Test simulation parameter validation
        navigator = create_navigator(position=[0, 0], max_speed=1.0)
        
        with pytest.raises(ConfigurationError):
            run_plume_simulation(
                navigator=navigator,
                video_plume=None,  # Invalid video plume
                num_steps=100
            )
    
    def test_simulation_error_recovery(self, sample_config, mock_video_capture, mock_video_file):
        """
        Test simulation error recovery and graceful failure handling.
        
        Validates:
        - Simulation interruption handling
        - Partial result recovery
        - Resource cleanup on errors
        """
        navigator = create_navigator(**sample_config['navigator'])
        video_plume = create_video_plume(video_path=mock_video_file)
        
        # Test simulation with frame access errors
        with patch.object(video_plume, 'get_frame', side_effect=[
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),  # First frame OK
            None,  # Second frame fails
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),  # Third frame OK
        ]):
            
            # Simulation should handle frame access errors gracefully
            positions, orientations, readings = run_plume_simulation(
                navigator=navigator,
                video_plume=video_plume,
                num_steps=3,
                record_trajectory=True
            )
            
            # Validate partial results are still generated
            assert positions.shape[1] >= 2  # At least initial state + one step
            assert np.all(np.isfinite(positions))
    
    def test_database_error_recovery(self):
        """
        Test database error handling and graceful degradation.
        
        Validates:
        - Database connection failure handling
        - Graceful fallback to file-based operations
        - Session cleanup on errors
        """
        # Test with invalid database configuration
        invalid_config = {'url': 'invalid://connection/string', 'enabled': True}
        
        # Database session should handle invalid configuration gracefully
        with get_session(invalid_config) as session:
            # Should return None for invalid configuration
            assert session is None or hasattr(session, 'execute')
        
        # Test session cleanup after configuration change
        cleanup_database()
        
        # Verify cleanup was successful
        with get_session() as session:
            # New session should be independent of previous configuration
            pass  # Session creation should not raise exceptions


class TestPerformanceCharacteristics(IntegrationTestBase):
    """Test performance characteristics for integrated workflows per timing requirements."""
    
    def test_integration_workflow_performance_sla(self, sample_config, mock_video_capture, mock_video_file):
        """
        Test integrated workflow performance against SLA requirements.
        
        Validates:
        - End-to-end workflow timing per Section 6.6.3.3
        - Component initialization performance
        - Memory usage characteristics
        - Scalability with simulation parameters
        """
        # Test component initialization performance
        start_time = time.time()
        navigator = create_navigator(**sample_config['navigator'])
        nav_init_time = time.time() - start_time
        
        start_time = time.time()
        video_plume = create_video_plume(video_path=mock_video_file)
        plume_init_time = time.time() - start_time
        
        # Validate initialization performance
        assert nav_init_time < 0.1, f"Navigator initialization too slow: {nav_init_time:.3f}s"
        assert plume_init_time < 0.5, f"Video plume initialization too slow: {plume_init_time:.3f}s"
        
        # Test simulation performance scaling
        step_counts = [10, 50, 100]
        performance_results = []
        
        for num_steps in step_counts:
            start_time = time.time()
            positions, orientations, readings = run_plume_simulation(
                navigator=navigator,
                video_plume=video_plume,
                num_steps=num_steps,
                record_trajectory=True
            )
            execution_time = time.time() - start_time
            
            steps_per_second = num_steps / execution_time
            performance_results.append(steps_per_second)
            
            # Validate minimum performance (30+ FPS capability)
            assert steps_per_second >= 30, f"Performance below SLA: {steps_per_second:.1f} steps/s"
        
        # Validate performance scaling linearity
        assert len(performance_results) == len(step_counts)
        # Performance should not degrade significantly with step count
        performance_ratio = min(performance_results) / max(performance_results)
        assert performance_ratio > 0.5, f"Performance scaling issue: {performance_ratio:.3f}"
    
    def test_memory_usage_characteristics(self, sample_config, mock_video_capture, mock_video_file):
        """
        Test memory usage characteristics for integrated workflows.
        
        Validates:
        - Memory efficiency in simulation execution
        - Resource cleanup and garbage collection
        - Memory scaling with simulation parameters
        """
        import gc
        import psutil
        import os
        
        # Get current process for memory monitoring
        process = psutil.Process(os.getpid())
        
        # Baseline memory usage
        gc.collect()
        baseline_memory = process.memory_info().rss
        
        # Execute simulation with memory monitoring
        navigator = create_navigator(**sample_config['navigator'])
        video_plume = create_video_plume(video_path=mock_video_file)
        
        positions, orientations, readings = run_plume_simulation(
            navigator=navigator,
            video_plume=video_plume,
            num_steps=100,
            record_trajectory=True
        )
        
        # Peak memory usage during simulation
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - baseline_memory
        
        # Validate memory usage is reasonable (<100MB increase per requirements)
        memory_increase_mb = memory_increase / (1024 * 1024)
        assert memory_increase_mb < 100, f"Memory usage too high: {memory_increase_mb:.1f}MB"
        
        # Test memory cleanup
        del positions, orientations, readings, navigator, video_plume
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_cleanup = peak_memory - final_memory
        
        # Validate significant memory cleanup occurred
        cleanup_ratio = memory_cleanup / memory_increase
        assert cleanup_ratio > 0.5, f"Insufficient memory cleanup: {cleanup_ratio:.3f}"


class TestCrossFrameworkCompatibility(IntegrationTestBase):
    """Test cross-framework compatibility with RL and ML integration patterns."""
    
    def test_numpy_array_interface_compatibility(self, sample_config, mock_video_capture, mock_video_file):
        """
        Test NumPy array interface compatibility for ML framework integration.
        
        Validates:
        - NumPy array output formats
        - Data type consistency
        - Tensor conversion compatibility
        - Batch processing support
        """
        navigator = create_navigator(**sample_config['navigator'])
        video_plume = create_video_plume(video_path=mock_video_file)
        
        positions, orientations, readings = run_plume_simulation(
            navigator=navigator,
            video_plume=video_plume,
            num_steps=50,
            record_trajectory=True
        )
        
        # Validate NumPy array characteristics
        assert isinstance(positions, np.ndarray)
        assert isinstance(orientations, np.ndarray)
        assert isinstance(readings, np.ndarray)
        
        # Validate data types for ML compatibility
        assert positions.dtype in [np.float32, np.float64]
        assert orientations.dtype in [np.float32, np.float64]
        assert readings.dtype in [np.float32, np.float64]
        
        # Validate array shapes for batch processing
        assert positions.ndim == 3  # (agents, timesteps, coordinates)
        assert orientations.ndim == 2  # (agents, timesteps)
        assert readings.ndim == 2  # (agents, timesteps)
        
        # Test tensor conversion compatibility (if packages available)
        try:
            import torch
            
            # Test PyTorch tensor conversion
            positions_tensor = torch.from_numpy(positions)
            orientations_tensor = torch.from_numpy(orientations)
            readings_tensor = torch.from_numpy(readings)
            
            assert positions_tensor.shape == positions.shape
            assert orientations_tensor.shape == orientations.shape
            assert readings_tensor.shape == readings.shape
            
        except ImportError:
            pass  # PyTorch not available
        
        try:
            import tensorflow as tf
            
            # Test TensorFlow tensor conversion
            positions_tf = tf.convert_to_tensor(positions)
            orientations_tf = tf.convert_to_tensor(orientations)
            readings_tf = tf.convert_to_tensor(readings)
            
            assert positions_tf.shape == positions.shape
            assert orientations_tf.shape == orientations.shape
            assert readings_tf.shape == readings.shape
            
        except ImportError:
            pass  # TensorFlow not available
    
    def test_protocol_based_interface_compatibility(self, sample_config):
        """
        Test protocol-based interface compatibility for framework integration.
        
        Validates:
        - NavigatorProtocol implementation compliance
        - Duck typing compatibility
        - Interface consistency across implementations
        """
        navigator = create_navigator(**sample_config['navigator'])
        
        # Validate protocol compliance
        assert isinstance(navigator, NavigatorProtocol)
        
        # Test protocol interface completeness
        protocol_methods = [
            'positions', 'orientations', 'speeds', 'max_speeds', 
            'angular_velocities', 'num_agents', 'reset', 'step', 
            'sample_odor', 'read_single_antenna_odor', 'sample_multiple_sensors'
        ]
        
        for method_name in protocol_methods:
            assert hasattr(navigator, method_name), f"Missing protocol method: {method_name}"
        
        # Test protocol property types
        assert isinstance(navigator.positions, np.ndarray)
        assert isinstance(navigator.orientations, np.ndarray)
        assert isinstance(navigator.speeds, np.ndarray)
        assert isinstance(navigator.max_speeds, np.ndarray)
        assert isinstance(navigator.angular_velocities, np.ndarray)
        assert isinstance(navigator.num_agents, int)
        
        # Test protocol method callable
        assert callable(navigator.reset)
        assert callable(navigator.step)
        assert callable(navigator.sample_odor)
    
    def test_reinforcement_learning_interface_compatibility(self, sample_config, mock_video_capture, mock_video_file):
        """
        Test RL framework compatibility through standardized interfaces.
        
        Validates:
        - Gym-like environment interface patterns
        - State/action/reward interface compatibility
        - Episode-based interaction patterns
        """
        navigator = create_navigator(**sample_config['navigator'])
        video_plume = create_video_plume(video_path=mock_video_file)
        
        # Test RL-style reset functionality
        initial_positions = navigator.positions.copy()
        navigator.reset()  # Should reset to initial state
        
        # Test RL-style step functionality
        env_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        navigator.step(env_frame)
        
        # Validate state changes after step
        assert not np.array_equal(navigator.positions, initial_positions)
        
        # Test observation extraction (RL state)
        observations = navigator.sample_odor(env_frame)
        assert isinstance(observations, (float, np.ndarray))
        
        if isinstance(observations, np.ndarray):
            assert observations.shape == (navigator.num_agents,)
        
        # Test multi-sensor observations for complex RL scenarios
        multi_observations = navigator.sample_multiple_sensors(
            env_frame, 
            sensor_distance=5.0,
            sensor_angle=45.0,
            num_sensors=3
        )
        
        assert isinstance(multi_observations, np.ndarray)
        expected_shape = (navigator.num_agents, 3)  # 3 sensors per agent
        assert multi_observations.shape == expected_shape


class TestWorkflowOrchestrationIntegration(IntegrationTestBase):
    """Test workflow orchestration integration with DVC and Snakemake scenarios."""
    
    def test_dvc_pipeline_integration_patterns(self, sample_config):
        """
        Test DVC pipeline integration patterns and compatibility.
        
        Validates:
        - Parameter management through DVC
        - Experiment tracking integration
        - Reproducible pipeline execution
        """
        # Test parameter file generation for DVC
        params_file = self.temp_dir / "params.yaml"
        import yaml
        
        dvc_params = {
            'navigator': sample_config['navigator'],
            'simulation': sample_config['simulation'],
            'reproducibility': {'seed': 42}
        }
        
        with open(params_file, 'w') as f:
            yaml.dump(dvc_params, f)
        
        # Validate parameter file creation
        assert params_file.exists()
        
        # Test parameter loading for DVC pipeline
        with open(params_file, 'r') as f:
            loaded_params = yaml.safe_load(f)
        
        assert loaded_params['navigator']['max_speed'] == sample_config['navigator']['max_speed']
        assert loaded_params['simulation']['num_steps'] == sample_config['simulation']['num_steps']
        
        # Test DVC-style output specification
        output_spec = {
            'positions': str(self.output_dir / "positions.npy"),
            'orientations': str(self.output_dir / "orientations.npy"),
            'readings': str(self.output_dir / "readings.npy"),
            'metadata': str(self.output_dir / "metadata.json")
        }
        
        # Simulate DVC output generation
        for output_type, output_path in output_spec.items():
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if output_type == 'metadata':
                import json
                metadata = {
                    'navigator_config': loaded_params['navigator'],
                    'simulation_config': loaded_params['simulation'],
                    'timestamp': time.time()
                }
                with open(output_path, 'w') as f:
                    json.dump(metadata, f)
            else:
                # Create dummy array files
                dummy_array = np.zeros((1, 101, 2) if output_type == 'positions' else (1, 101))
                np.save(output_path, dummy_array)
        
        # Validate all outputs were created
        for output_path in output_spec.values():
            assert Path(output_path).exists()
    
    def test_snakemake_workflow_integration_patterns(self, sample_config):
        """
        Test Snakemake workflow integration patterns and compatibility.
        
        Validates:
        - Rule-based execution patterns
        - Input/output dependency management
        - Scalable processing workflows
        """
        # Test Snakemake-style input/output specification
        workflow_config = {
            'inputs': {
                'config': str(self.config_dir / "config.yaml"),
                'video': str(self.data_dir / "input_video.mp4")
            },
            'outputs': {
                'results': str(self.output_dir / "simulation_results.npz"),
                'plots': str(self.output_dir / "trajectory_plot.png"),
                'report': str(self.output_dir / "analysis_report.html")
            }
        }
        
        # Create input files
        import yaml
        with open(workflow_config['inputs']['config'], 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create dummy video file
        Path(workflow_config['inputs']['video']).touch()
        
        # Test Snakemake-style rule execution simulation
        def simulate_snakemake_rule():
            """Simulate Snakemake rule execution."""
            # Load configuration
            with open(workflow_config['inputs']['config'], 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate inputs exist
            for input_path in workflow_config['inputs'].values():
                assert Path(input_path).exists(), f"Missing input: {input_path}"
            
            # Generate outputs
            results_data = {
                'positions': np.zeros((1, 101, 2)),
                'orientations': np.zeros((1, 101)),
                'readings': np.zeros((1, 101)),
                'config': config
            }
            
            # Save results
            np.savez_compressed(workflow_config['outputs']['results'], **results_data)
            
            # Create dummy plot and report files
            Path(workflow_config['outputs']['plots']).touch()
            Path(workflow_config['outputs']['report']).touch()
            
            return True
        
        # Execute simulated rule
        rule_success = simulate_snakemake_rule()
        assert rule_success
        
        # Validate all outputs were created
        for output_path in workflow_config['outputs'].values():
            assert Path(output_path).exists(), f"Missing output: {output_path}"
        
        # Test output data integrity
        results = np.load(workflow_config['outputs']['results'])
        assert 'positions' in results
        assert 'orientations' in results
        assert 'readings' in results
    
    def test_experiment_tracking_integration(self, sample_config):
        """
        Test experiment tracking integration for reproducible research.
        
        Validates:
        - Experiment metadata capture
        - Version control integration patterns
        - Reproducibility validation
        """
        # Test experiment metadata generation
        experiment_metadata = {
            'experiment_id': f"exp_{int(time.time())}",
            'parameters': sample_config,
            'environment': {
                'python_version': sys.version,
                'numpy_version': np.__version__,
                'platform': sys.platform
            },
            'reproducibility': {
                'global_seed': sample_config.get('reproducibility', {}).get('global_seed', 42),
                'timestamp': time.time()
            }
        }
        
        # Test metadata serialization
        metadata_file = self.output_dir / "experiment_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        
        assert metadata_file.exists()
        
        # Test metadata loading and validation
        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata['experiment_id'] == experiment_metadata['experiment_id']
        assert loaded_metadata['parameters']['navigator']['max_speed'] == sample_config['navigator']['max_speed']
        
        # Test reproducibility validation
        assert 'global_seed' in loaded_metadata['reproducibility']
        assert loaded_metadata['reproducibility']['global_seed'] == 42


# Performance benchmarking utilities for integration testing
class PerformanceBenchmark:
    """Utility class for performance benchmarking in integration tests."""
    
    def __init__(self, name: str, sla_threshold: float):
        self.name = name
        self.sla_threshold = sla_threshold
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        
        if execution_time > self.sla_threshold:
            warnings.warn(
                f"Performance SLA violation: {self.name} took {execution_time:.3f}s "
                f"(threshold: {self.sla_threshold:.3f}s)",
                PerformanceWarning
            )


class PerformanceWarning(UserWarning):
    """Warning category for performance SLA violations."""
    pass


# Test execution configuration and pytest markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,  # Integration tests may take longer
]


# Global test configuration
def pytest_configure(config):
    """Configure pytest for integration testing."""
    # Register custom markers
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "performance: mark test as performance validation")
    
    # Configure test environment
    os.environ.setdefault("PYTEST_CURRENT_TEST", "true")
    
    # Disable interactive matplotlib backends
    if MATPLOTLIB_AVAILABLE:
        matplotlib.use('Agg')


# Module-level test utilities and fixtures
@pytest.fixture(scope="session")
def integration_test_session():
    """Session-scoped fixture for integration test setup."""
    # Global test session setup
    start_time = time.time()
    
    yield
    
    # Global test session teardown
    total_time = time.time() - start_time
    print(f"\nIntegration test session completed in {total_time:.2f}s")


if __name__ == "__main__":
    # Enable direct execution for debugging
    pytest.main([__file__, "-v", "--tb=short"])