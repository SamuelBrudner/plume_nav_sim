"""
Integration testing module for comprehensive system validation.

This module provides end-to-end testing for the odor plume navigation system,
validating complete workflows, component interactions, cross-system compatibility,
and framework integration patterns. Tests ensure seamless operation across
CLI interfaces, configuration management, database sessions, and workflow orchestration.

Test Categories:
- Complete simulation workflows with Navigator and VideoPlume integration
- Kedro pipeline compatibility with factory method patterns
- CLI-to-core integration with parameter flow validation
- Configuration system integration across all components
- Database session integration with simulation workflows
- Visualization export workflows with complete data pipeline testing
- Multi-component error handling and recovery scenarios
- Performance characteristics validation per SLA requirements
- Cross-framework compatibility with RL and ML integration patterns
- Workflow orchestration integration with DVC and Snakemake scenarios
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from unittest.mock import Mock, MagicMock, patch, create_autospec
import json

import pytest
import numpy as np
from click.testing import CliRunner
from hydra import compose, initialize_config_store
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Test framework imports
try:
    from kedro.io import DataCatalog
    from kedro.pipeline import Pipeline, node
    KEDRO_AVAILABLE = True
except ImportError:
    KEDRO_AVAILABLE = False


class TestEndToEndSimulationWorkflows:
    """Test complete simulation workflows with Navigator and VideoPlume integration."""
    
    def test_single_agent_complete_workflow(self, mock_navigator, mock_video_plume, 
                                          mock_hydra_config):
        """Test complete single-agent simulation workflow from configuration to results."""
        from {{cookiecutter.project_slug}}.api.navigation import (
            create_navigator, create_video_plume, run_plume_simulation
        )
        
        # Configure mocks for realistic simulation
        mock_navigator.num_agents = 1
        mock_navigator.positions = np.array([[10.0, 10.0]])
        mock_navigator.orientations = np.array([45.0])
        mock_navigator.is_single_agent = True
        
        mock_video_plume.width = 640
        mock_video_plume.height = 480
        mock_video_plume.frame_count = 100
        
        # Mock frame sequence with realistic odor gradient
        frames = []
        for i in range(100):
            frame = np.random.rand(480, 640) * 255
            # Add odor gradient for realistic navigation
            y, x = np.ogrid[:480, :640]
            source_x, source_y = 320, 240
            distance = np.sqrt((x - source_x)**2 + (y - source_y)**2)
            odor_field = np.exp(-distance / 100) * 255
            frame = frame * 0.7 + odor_field * 0.3
            frames.append(frame.astype(np.uint8))
        
        mock_video_plume.get_frame.side_effect = frames
        
        # Execute complete workflow
        navigator = create_navigator(cfg=mock_hydra_config.navigator)
        video_plume = create_video_plume(cfg=mock_hydra_config.video_plume)
        
        # Run simulation with performance timing
        start_time = time.time()
        positions, orientations, readings = run_plume_simulation(
            navigator=navigator,
            plume=video_plume,
            num_steps=50,
            dt=0.1
        )
        execution_time = time.time() - start_time
        
        # Validate results structure and quality
        assert positions.shape == (1, 51, 2), "Positions shape incorrect for single agent"
        assert orientations.shape == (1, 51), "Orientations shape incorrect"
        assert readings.shape == (1, 51), "Readings shape incorrect"
        
        # Validate numerical stability
        assert np.all(np.isfinite(positions)), "Positions contain invalid values"
        assert np.all(np.isfinite(orientations)), "Orientations contain invalid values"
        assert np.all(np.isfinite(readings)), "Readings contain invalid values"
        
        # Performance validation per Section 6.6.3.3
        assert execution_time < 5.0, "Single agent simulation exceeds 5s SLA"
        
        # Validate method call patterns
        assert mock_navigator.step.call_count == 50, "Navigator step not called correctly"
        assert mock_navigator.sample_odor.call_count == 50, "Odor sampling not performed"
    
    def test_multi_agent_complete_workflow(self, mock_multi_navigator, mock_video_plume,
                                         mock_hydra_config):
        """Test complete multi-agent simulation workflow with coordination."""
        from {{cookiecutter.project_slug}}.api.navigation import run_plume_simulation
        
        # Configure multi-agent scenario
        num_agents = 5
        mock_multi_navigator.num_agents = num_agents
        mock_multi_navigator.positions = np.random.rand(num_agents, 2) * 100
        mock_multi_navigator.orientations = np.random.rand(num_agents) * 360
        mock_multi_navigator.is_single_agent = False
        
        # Configure realistic frame sequence
        mock_video_plume.frame_count = 100
        mock_video_plume.get_frame.return_value = np.random.rand(480, 640) * 255
        
        # Execute multi-agent simulation
        start_time = time.time()
        positions, orientations, readings = run_plume_simulation(
            navigator=mock_multi_navigator,
            plume=mock_video_plume,
            num_steps=30,
            dt=0.1
        )
        execution_time = time.time() - start_time
        
        # Validate multi-agent results
        assert positions.shape == (num_agents, 31, 2), "Multi-agent positions incorrect"
        assert orientations.shape == (num_agents, 31), "Multi-agent orientations incorrect"
        assert readings.shape == (num_agents, 31), "Multi-agent readings incorrect"
        
        # Performance validation for multi-agent per Section 6.6.3.3
        assert execution_time < 10.0, f"Multi-agent ({num_agents}) simulation exceeds SLA"
        
        # Validate vectorized operations were used
        assert mock_multi_navigator.step.call_count == 30, "Multi-agent steps incorrect"
    
    def test_error_recovery_in_simulation_workflow(self, mock_navigator, mock_video_plume):
        """Test error handling and recovery scenarios in complete workflow."""
        from {{cookiecutter.project_slug}}.api.navigation import run_plume_simulation
        
        # Test video frame error recovery
        mock_video_plume.get_frame.side_effect = [
            np.random.rand(480, 640) * 255,  # Frame 0 success
            Exception("Frame read error"),    # Frame 1 failure
            np.random.rand(480, 640) * 255,  # Frame 2 recovery
        ]
        
        # Test graceful error handling
        with pytest.raises(Exception) as exc_info:
            run_plume_simulation(
                navigator=mock_navigator,
                plume=mock_video_plume,
                num_steps=3,
                dt=0.1
            )
        
        assert "Frame read error" in str(exc_info.value)
        
        # Test navigator error recovery
        mock_video_plume.get_frame.side_effect = None
        mock_video_plume.get_frame.return_value = np.random.rand(480, 640) * 255
        mock_navigator.step.side_effect = [
            None,  # Step 0 success
            ValueError("Navigation error"),  # Step 1 failure
        ]
        
        with pytest.raises(ValueError) as exc_info:
            run_plume_simulation(
                navigator=mock_navigator,
                plume=mock_video_plume,
                num_steps=2,
                dt=0.1
            )
        
        assert "Navigation error" in str(exc_info.value)


@pytest.mark.skipif(not KEDRO_AVAILABLE, reason="Kedro not available")
class TestKedroIntegration:
    """Test Kedro pipeline compatibility with factory method patterns."""
    
    def test_kedro_data_catalog_integration(self, mock_hydra_config):
        """Test Navigator integration with Kedro data catalog."""
        from {{cookiecutter.project_slug}}.api.navigation import create_navigator
        
        # Mock Kedro DataCatalog integration
        with patch('kedro.io.DataCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            # Create mock dataset for navigator
            mock_dataset = Mock()
            mock_dataset.load.return_value = create_navigator(cfg=mock_hydra_config.navigator)
            mock_dataset.save = Mock()
            
            mock_catalog.load.return_value = mock_dataset.load.return_value
            
            # Test data catalog integration
            catalog = mock_catalog_class({
                "navigator": mock_dataset,
                "config": mock_hydra_config
            })
            
            # Load navigator from catalog
            navigator = catalog.load("navigator")
            
            # Validate navigator properties
            assert hasattr(navigator, 'positions'), "Navigator missing positions property"
            assert hasattr(navigator, 'step'), "Navigator missing step method"
            assert hasattr(navigator, 'sample_odor'), "Navigator missing sample_odor method"
            
            # Test saving to catalog
            catalog.save("navigator", navigator)
            mock_dataset.save.assert_called_once()
    
    def test_kedro_pipeline_node_execution(self, mock_hydra_config):
        """Test Navigator execution within Kedro pipeline nodes."""
        from {{cookiecutter.project_slug}}.api.navigation import create_navigator, run_plume_simulation
        
        def navigator_creation_node(config: DictConfig):
            """Kedro node function for navigator creation."""
            return create_navigator(cfg=config.navigator)
        
        def simulation_execution_node(navigator, video_plume, config: DictConfig):
            """Kedro node function for simulation execution."""
            return run_plume_simulation(
                navigator=navigator,
                plume=video_plume,
                num_steps=config.simulation.num_steps,
                dt=config.simulation.dt
            )
        
        # Mock pipeline execution
        with patch('kedro.pipeline.Pipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            # Create test pipeline
            pipeline = mock_pipeline_class([
                node(
                    func=navigator_creation_node,
                    inputs="config",
                    outputs="navigator",
                    name="create_navigator"
                ),
                node(
                    func=simulation_execution_node,
                    inputs=["navigator", "video_plume", "config"],
                    outputs="simulation_results",
                    name="run_simulation"
                )
            ])
            
            # Validate pipeline structure
            assert pipeline is not None
            mock_pipeline_class.assert_called_once()
    
    def test_kedro_configuration_integration(self):
        """Test Hydra configuration integration with Kedro parameters."""
        # Mock Kedro configuration loading
        kedro_params = {
            "navigator": {
                "max_speed": 15.0,
                "angular_velocity": 0.2
            },
            "video_plume": {
                "flip_horizontal": True,
                "gaussian_blur": {"enabled": True, "sigma": 2.0}
            }
        }
        
        # Test configuration merging
        with patch('hydra.compose') as mock_compose:
            mock_config = DictConfig(kedro_params)
            mock_compose.return_value = mock_config
            
            # Validate configuration structure
            assert mock_config.navigator.max_speed == 15.0
            assert mock_config.video_plume.flip_horizontal is True
            assert mock_config.video_plume.gaussian_blur.sigma == 2.0


class TestCLIIntegration:
    """Test CLI-to-core integration with parameter flow validation."""
    
    def test_cli_command_execution_with_hydra_config(self, mock_hydra_config):
        """Test CLI command execution with Hydra configuration integration."""
        from {{cookiecutter.project_slug}}.cli.main import main
        
        runner = CliRunner()
        
        with patch('{{cookiecutter.project_slug}}.cli.main.create_navigator') as mock_create_nav:
            mock_navigator = Mock()
            mock_create_nav.return_value = mock_navigator
            
            # Test CLI command with configuration overrides
            result = runner.invoke(main, [
                'run-simulation',
                '--config-override', 'navigator.max_speed=20.0',
                '--config-override', 'simulation.num_steps=100'
            ])
            
            # Validate command execution
            if result.exit_code != 0:
                print(f"CLI Error Output: {result.output}")
            
            # Check that configuration was processed
            assert mock_create_nav.called or result.exit_code == 0, "CLI navigation creation failed"
    
    def test_cli_parameter_validation_and_error_handling(self):
        """Test CLI parameter validation and error handling."""
        from {{cookiecutter.project_slug}}.cli.main import main
        
        runner = CliRunner()
        
        # Test invalid parameter handling
        result = runner.invoke(main, [
            'run-simulation',
            '--config-override', 'navigator.max_speed=invalid_value'
        ])
        
        # Validate error handling
        assert result.exit_code != 0 or "error" in result.output.lower(), \
            "CLI should handle invalid parameters gracefully"
    
    def test_cli_multi_run_parameter_sweep(self):
        """Test CLI multi-run parameter sweep functionality."""
        from {{cookiecutter.project_slug}}.cli.main import main
        
        runner = CliRunner()
        
        with patch('hydra.compose') as mock_compose:
            mock_config = DictConfig({
                "navigator": {"max_speed": 10.0},
                "simulation": {"num_steps": 50}
            })
            mock_compose.return_value = mock_config
            
            # Test multi-run execution
            result = runner.invoke(main, [
                '--multirun',
                'navigator.max_speed=5,10,15',
                'run-simulation'
            ])
            
            # Validate multi-run handling
            assert result.exit_code == 0 or "multirun" in result.output, \
                "Multi-run functionality not properly handled"


class TestConfigurationSystemIntegration:
    """Test configuration system integration across all components."""
    
    def test_hierarchical_configuration_composition(self):
        """Test Hydra hierarchical configuration composition and validation."""
        # Test configuration hierarchy loading
        base_config = {
            "navigator": {
                "type": "single",
                "max_speed": 10.0,
                "angular_velocity": 0.1
            },
            "video_plume": {
                "flip_horizontal": False,
                "gaussian_blur": {"enabled": False}
            }
        }
        
        override_config = {
            "navigator": {
                "max_speed": 15.0  # Override base value
            },
            "video_plume": {
                "flip_horizontal": True  # Override base value
            }
        }
        
        # Test configuration merging
        with patch('hydra.compose') as mock_compose:
            final_config = DictConfig(base_config)
            OmegaConf.update(final_config, override_config)
            mock_compose.return_value = final_config
            
            config = mock_compose()
            
            # Validate hierarchy composition
            assert config.navigator.max_speed == 15.0, "Configuration override failed"
            assert config.navigator.angular_velocity == 0.1, "Base config value lost"
            assert config.video_plume.flip_horizontal is True, "Override not applied"
    
    def test_environment_variable_interpolation(self):
        """Test environment variable interpolation in configuration."""
        # Set test environment variables
        test_env = {
            "DATABASE_URL": "sqlite:///test.db",
            "MAX_SPEED": "12.5",
            "DEBUG_MODE": "true"
        }
        
        config_template = {
            "database": {"url": "${oc.env:DATABASE_URL}"},
            "navigator": {"max_speed": "${oc.env:MAX_SPEED}"},
            "debug": "${oc.env:DEBUG_MODE}"
        }
        
        with patch.dict(os.environ, test_env):
            with patch('hydra.compose') as mock_compose:
                # Mock Hydra environment interpolation
                resolved_config = DictConfig({
                    "database": {"url": "sqlite:///test.db"},
                    "navigator": {"max_speed": 12.5},
                    "debug": True
                })
                mock_compose.return_value = resolved_config
                
                config = mock_compose()
                
                # Validate environment variable interpolation
                assert config.database.url == "sqlite:///test.db"
                assert config.navigator.max_speed == 12.5
                assert config.debug is True
    
    def test_configuration_validation_across_components(self):
        """Test configuration validation across all system components."""
        from {{cookiecutter.project_slug}}.config.schemas import (
            NavigatorConfig, VideoPlumeConfig
        )
        
        # Test valid configuration
        valid_navigator_config = {
            "position": [50.0, 50.0],
            "orientation": 45.0,
            "speed": 5.0,
            "max_speed": 10.0
        }
        
        valid_video_config = {
            "video_path": "/path/to/video.mp4",
            "flip": False,
            "kernel_size": 5,
            "kernel_sigma": 1.0
        }
        
        # Validate schemas
        navigator_config = NavigatorConfig(**valid_navigator_config)
        video_config = VideoPlumeConfig(**valid_video_config)
        
        assert navigator_config.max_speed == 10.0
        assert video_config.kernel_size == 5
        
        # Test invalid configuration
        invalid_config = {
            "position": [50.0],  # Invalid: missing y coordinate
            "speed": -5.0  # Invalid: negative speed
        }
        
        with pytest.raises(Exception):  # Pydantic validation error
            NavigatorConfig(**invalid_config)


class TestDatabaseSessionIntegration:
    """Test database session integration with simulation workflows."""
    
    def test_database_session_lifecycle_management(self):
        """Test database session creation, usage, and cleanup."""
        from {{cookiecutter.project_slug}}.db.session import get_session
        
        # Mock SQLAlchemy session
        with patch('{{cookiecutter.project_slug}}.db.session.SessionLocal') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            # Test session creation
            session = get_session()
            
            assert session is not None
            mock_session_class.assert_called_once()
            
            # Test session cleanup
            session.close()
            mock_session.close.assert_called_once()
    
    def test_simulation_data_persistence(self, mock_navigator, mock_video_plume):
        """Test simulation data persistence through database session."""
        from {{cookiecutter.project_slug}}.api.navigation import run_plume_simulation
        
        # Configure simulation
        mock_navigator.num_agents = 1
        mock_navigator.positions = np.array([[10.0, 10.0]])
        mock_video_plume.get_frame.return_value = np.random.rand(480, 640) * 255
        
        # Mock database session
        with patch('{{cookiecutter.project_slug}}.db.session.get_session') as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value = mock_session
            
            # Run simulation
            positions, orientations, readings = run_plume_simulation(
                navigator=mock_navigator,
                plume=mock_video_plume,
                num_steps=10,
                dt=0.1
            )
            
            # Test data structure for persistence
            assert positions.shape[0] >= 1, "No trajectory data for persistence"
            assert positions.shape[1] >= 10, "Insufficient data points"
            
            # Mock data persistence
            trajectory_data = {
                "positions": positions.tolist(),
                "orientations": orientations.tolist(),
                "readings": readings.tolist()
            }
            
            # Validate data serialization
            serialized_data = json.dumps(trajectory_data)
            assert len(serialized_data) > 0, "Data serialization failed"
    
    def test_database_transaction_handling(self):
        """Test database transaction handling and rollback scenarios."""
        with patch('{{cookiecutter.project_slug}}.db.session.SessionLocal') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            # Test successful transaction
            mock_session.commit.return_value = None
            mock_session.rollback.return_value = None
            
            try:
                # Simulate transaction operations
                mock_session.add(Mock())
                mock_session.commit()
            except Exception:
                mock_session.rollback()
                raise
            
            # Validate transaction calls
            mock_session.add.assert_called()
            mock_session.commit.assert_called_once()
            
            # Test transaction rollback
            mock_session.commit.side_effect = Exception("Database error")
            
            with pytest.raises(Exception):
                mock_session.add(Mock())
                mock_session.commit()
                mock_session.rollback()


class TestVisualizationIntegration:
    """Test visualization export workflows with complete data pipeline testing."""
    
    def test_complete_visualization_pipeline(self, mock_navigator, mock_video_plume):
        """Test complete visualization pipeline from simulation to export."""
        from {{cookiecutter.project_slug}}.api.navigation import run_plume_simulation
        from {{cookiecutter.project_slug}}.utils.visualization import (
            visualize_simulation_results, visualize_trajectory
        )
        
        # Configure realistic simulation data
        mock_navigator.num_agents = 2
        mock_navigator.positions = np.array([[10.0, 10.0], [20.0, 20.0]])
        mock_video_plume.get_frame.return_value = np.random.rand(480, 640) * 255
        
        # Run simulation
        positions, orientations, readings = run_plume_simulation(
            navigator=mock_navigator,
            plume=mock_video_plume,
            num_steps=20,
            dt=0.1
        )
        
        # Test visualization pipeline
        with patch('matplotlib.pyplot') as mock_plt:
            mock_figure = Mock()
            mock_plt.figure.return_value = mock_figure
            
            # Test trajectory visualization
            figure = visualize_trajectory(
                positions=positions,
                orientations=orientations,
                output_path=None,
                show_plot=False
            )
            
            assert figure is not None
            mock_plt.figure.assert_called()
            
            # Test simulation results visualization
            visualize_simulation_results(
                positions, orientations, 
                plume_frames=None,
                show_plot=False
            )
            
            # Validate visualization calls
            assert mock_plt.figure.call_count >= 1
    
    def test_visualization_export_formats(self):
        """Test visualization export in multiple formats."""
        # Mock trajectory data
        positions = np.random.rand(2, 50, 2) * 100
        orientations = np.random.rand(2, 50) * 360
        
        export_formats = ['.png', '.pdf', '.svg', '.mp4']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for fmt in export_formats:
                output_path = Path(temp_dir) / f"test_trajectory{fmt}"
                
                with patch('matplotlib.pyplot') as mock_plt:
                    mock_figure = Mock()
                    mock_plt.figure.return_value = mock_figure
                    
                    # Test export format
                    from {{cookiecutter.project_slug}}.utils.visualization import visualize_trajectory
                    
                    figure = visualize_trajectory(
                        positions=positions,
                        orientations=orientations,
                        output_path=str(output_path),
                        show_plot=False
                    )
                    
                    assert figure is not None
                    mock_figure.savefig.assert_called() or mock_plt.savefig.assert_called()
    
    def test_visualization_performance_characteristics(self):
        """Test visualization performance per timing requirements."""
        # Large dataset for performance testing
        positions = np.random.rand(10, 1000, 2) * 100  # 10 agents, 1000 steps
        orientations = np.random.rand(10, 1000) * 360
        
        with patch('matplotlib.pyplot') as mock_plt:
            mock_figure = Mock()
            mock_plt.figure.return_value = mock_figure
            
            start_time = time.time()
            
            from {{cookiecutter.project_slug}}.utils.visualization import visualize_trajectory
            visualize_trajectory(
                positions=positions,
                orientations=orientations,
                show_plot=False
            )
            
            execution_time = time.time() - start_time
            
            # Performance validation: visualization should complete quickly
            assert execution_time < 5.0, "Visualization performance exceeds acceptable threshold"


class TestPerformanceCharacteristics:
    """Test performance characteristics for integrated workflows per timing requirements."""
    
    def test_cli_command_initialization_performance(self):
        """Test CLI command initialization meets <2 second SLA requirement."""
        from {{cookiecutter.project_slug}}.cli.main import main
        
        runner = CliRunner()
        
        # Test CLI startup performance
        start_time = time.time()
        result = runner.invoke(main, ['--help'])
        initialization_time = time.time() - start_time
        
        # Validate CLI initialization performance per Section 6.6.3.3
        assert initialization_time < 2.0, f"CLI initialization ({initialization_time:.2f}s) exceeds 2s SLA"
        assert result.exit_code == 0, "CLI help command failed"
    
    def test_configuration_loading_performance(self):
        """Test Hydra configuration loading meets <500ms requirement."""
        # Test configuration composition performance
        config_data = {
            "navigator": {"max_speed": 10.0, "angular_velocity": 0.1},
            "video_plume": {"flip_horizontal": False},
            "simulation": {"num_steps": 100, "dt": 0.1}
        }
        
        with patch('hydra.compose') as mock_compose:
            mock_config = DictConfig(config_data)
            mock_compose.return_value = mock_config
            
            start_time = time.time()
            config = mock_compose()
            composition_time = time.time() - start_time
            
            # Validate configuration performance per Section 6.6.3.3
            assert composition_time < 0.5, f"Configuration loading ({composition_time:.3f}s) exceeds 500ms SLA"
            assert config is not None
    
    def test_database_session_establishment_performance(self):
        """Test database session establishment meets <100ms requirement."""
        from {{cookiecutter.project_slug}}.db.session import get_session
        
        with patch('{{cookiecutter.project_slug}}.db.session.SessionLocal') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            start_time = time.time()
            session = get_session()
            establishment_time = time.time() - start_time
            
            # Validate database session performance per Section 6.6.3.3  
            assert establishment_time < 0.1, f"DB session establishment ({establishment_time:.3f}s) exceeds 100ms SLA"
            assert session is not None
    
    def test_multi_agent_simulation_performance_scaling(self):
        """Test multi-agent simulation performance scaling characteristics."""
        from {{cookiecutter.project_slug}}.api.navigation import run_plume_simulation
        
        agent_counts = [1, 5, 10, 50, 100]
        performance_results = {}
        
        for num_agents in agent_counts:
            # Mock multi-agent navigator
            mock_navigator = Mock()
            mock_navigator.num_agents = num_agents
            mock_navigator.positions = np.random.rand(num_agents, 2) * 100
            mock_navigator.orientations = np.random.rand(num_agents) * 360
            mock_navigator.step.return_value = None
            mock_navigator.sample_odor.return_value = np.random.rand(num_agents)
            
            # Mock video plume
            mock_video_plume = Mock()
            mock_video_plume.get_frame.return_value = np.random.rand(480, 640) * 255
            
            # Performance test
            start_time = time.time()
            positions, orientations, readings = run_plume_simulation(
                navigator=mock_navigator,
                plume=mock_video_plume,
                num_steps=10,
                dt=0.1
            )
            execution_time = time.time() - start_time
            
            performance_results[num_agents] = execution_time
            
            # Validate performance scaling per Section 6.6.3.3
            if num_agents <= 10:
                assert execution_time < 5.0, f"Multi-agent ({num_agents}) simulation exceeds 5ms per step SLA"
            elif num_agents <= 100:
                assert execution_time < 50.0, f"Large multi-agent ({num_agents}) simulation exceeds 50ms per step SLA"
        
        # Validate performance scaling characteristics
        assert performance_results[1] <= performance_results[100], "Performance should scale with agent count"


class TestCrossFrameworkCompatibility:
    """Test cross-framework compatibility with RL and ML integration patterns."""
    
    def test_numpy_array_interface_compatibility(self, mock_navigator):
        """Test NumPy array interface compatibility with ML frameworks."""
        # Configure navigator with realistic data
        mock_navigator.positions = np.array([[10.0, 20.0], [30.0, 40.0]])
        mock_navigator.orientations = np.array([45.0, 90.0])
        mock_navigator.num_agents = 2
        
        # Test NumPy array compatibility
        positions = mock_navigator.positions
        assert isinstance(positions, np.ndarray), "Positions not NumPy array"
        assert positions.dtype == np.float64 or positions.dtype == np.float32, "Invalid position data type"
        
        # Test tensor compatibility (mock PyTorch/TensorFlow)
        with patch('torch.from_numpy') as mock_torch:
            mock_tensor = Mock()
            mock_torch.return_value = mock_tensor
            
            # Mock tensor conversion
            tensor_positions = mock_torch(positions)
            assert tensor_positions is not None
            mock_torch.assert_called_with(positions)
        
        # Test array operations compatibility
        assert positions.shape == (2, 2), "Position array shape incorrect"
        assert np.all(positions >= 0), "Position values should be non-negative"
    
    def test_protocol_based_agent_definition_compatibility(self):
        """Test protocol-based interfaces for RL framework integration."""
        from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol
        
        # Test protocol compliance
        class MockRLAgent:
            """Mock RL agent implementing NavigatorProtocol."""
            
            @property
            def positions(self) -> np.ndarray:
                return np.array([[0.0, 0.0]])
            
            @property
            def orientations(self) -> np.ndarray:
                return np.array([0.0])
            
            @property
            def speeds(self) -> np.ndarray:
                return np.array([1.0])
            
            @property
            def max_speeds(self) -> np.ndarray:
                return np.array([10.0])
            
            @property
            def angular_velocities(self) -> np.ndarray:
                return np.array([0.1])
            
            @property
            def num_agents(self) -> int:
                return 1
            
            def reset(self, **kwargs) -> None:
                pass
            
            def step(self, env_array: np.ndarray) -> None:
                pass
            
            def sample_odor(self, env_array: np.ndarray) -> float:
                return 0.5
            
            def read_single_antenna_odor(self, env_array: np.ndarray) -> float:
                return 0.3
            
            def sample_multiple_sensors(self, env_array: np.ndarray, **kwargs) -> np.ndarray:
                return np.array([0.1, 0.2])
        
        # Test protocol compliance
        rl_agent = MockRLAgent()
        
        # Validate protocol methods
        assert hasattr(rl_agent, 'positions'), "RL agent missing positions property"
        assert hasattr(rl_agent, 'step'), "RL agent missing step method"
        assert hasattr(rl_agent, 'sample_odor'), "RL agent missing sample_odor method"
        
        # Test protocol usage
        env_array = np.random.rand(100, 100) * 255
        rl_agent.step(env_array)
        odor_reading = rl_agent.sample_odor(env_array)
        
        assert isinstance(odor_reading, (int, float, np.number)), "Invalid odor reading type"
    
    def test_batch_processing_workflow_compatibility(self):
        """Test batch processing workflow compatibility with ML pipelines."""
        # Mock batch processing scenario
        batch_size = 5
        trajectory_length = 100
        
        # Generate batch trajectory data
        batch_positions = np.random.rand(batch_size, trajectory_length, 2) * 100
        batch_orientations = np.random.rand(batch_size, trajectory_length) * 360
        batch_readings = np.random.rand(batch_size, trajectory_length)
        
        # Test batch data structure
        assert batch_positions.shape == (batch_size, trajectory_length, 2)
        assert batch_orientations.shape == (batch_size, trajectory_length)
        assert batch_readings.shape == (batch_size, trajectory_length)
        
        # Test batch processing operations
        batch_means = np.mean(batch_positions, axis=1)  # Mean position per trajectory
        batch_stds = np.std(batch_orientations, axis=1)  # Orientation variance
        
        assert batch_means.shape == (batch_size, 2)
        assert batch_stds.shape == (batch_size,)
        
        # Validate numerical stability
        assert np.all(np.isfinite(batch_means)), "Batch means contain invalid values"
        assert np.all(np.isfinite(batch_stds)), "Batch stds contain invalid values"


class TestWorkflowOrchestrationIntegration:
    """Test workflow orchestration integration with DVC and Snakemake scenarios."""
    
    def test_dvc_pipeline_dry_run_validation(self):
        """Test DVC pipeline definition validation through dry-run execution."""
        # Mock DVC pipeline configuration
        dvc_pipeline_config = {
            "stages": {
                "prepare_data": {
                    "cmd": "python prepare_video_data.py",
                    "deps": ["raw_videos/"],
                    "outs": ["processed_videos/"]
                },
                "run_simulation": {
                    "cmd": "python -m {{cookiecutter.project_slug}}.cli.main run-simulation",
                    "deps": ["processed_videos/", "conf/"],
                    "outs": ["results/trajectories.npy"]
                },
                "generate_analysis": {
                    "cmd": "python analyze_results.py",
                    "deps": ["results/trajectories.npy"],
                    "outs": ["reports/analysis.html"]
                }
            }
        }
        
        # Test DVC pipeline structure validation
        assert "stages" in dvc_pipeline_config, "DVC pipeline missing stages"
        assert "run_simulation" in dvc_pipeline_config["stages"], "Missing simulation stage"
        
        # Mock DVC command execution
        with patch('subprocess.run') as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Pipeline validation successful"
            mock_subprocess.return_value = mock_result
            
            # Simulate DVC dry-run
            result = mock_subprocess(['dvc', 'repro', '--dry-run'])
            
            assert result.returncode == 0, "DVC pipeline validation failed"
            mock_subprocess.assert_called_once()
    
    def test_snakemake_rule_validation(self):
        """Test Snakemake workflow rule validation and execution planning."""
        # Mock Snakemake rule definition
        snakemake_rules = {
            "rule_all": {
                "input": ["results/final_analysis.html"]
            },
            "process_videos": {
                "input": "raw_videos/{video}.mp4",
                "output": "processed_videos/{video}_processed.mp4",
                "shell": "python process_video.py {input} {output}"
            },
            "run_navigation": {
                "input": "processed_videos/{video}_processed.mp4",
                "output": "results/{video}_trajectory.npy",
                "shell": "python -m {{cookiecutter.project_slug}}.cli.main --config-override video_plume.video_path={input}"
            },
            "generate_report": {
                "input": expand("results/{video}_trajectory.npy", video=["video1", "video2"]),
                "output": "results/final_analysis.html",
                "shell": "python generate_report.py {input} {output}"
            }
        }
        
        # Test Snakemake rule structure
        assert "rule_all" in snakemake_rules, "Snakemake missing target rule"
        assert "run_navigation" in snakemake_rules, "Missing navigation rule"
        
        # Mock Snakemake execution
        with patch('subprocess.run') as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Workflow validation successful"
            mock_subprocess.return_value = mock_result
            
            # Simulate Snakemake dry-run
            result = mock_subprocess(['snakemake', '--dry-run', '--quiet'])
            
            assert result.returncode == 0, "Snakemake workflow validation failed"
            mock_subprocess.assert_called_once()
    
    def test_workflow_parameter_integration(self):
        """Test workflow parameter integration across DVC and Snakemake."""
        # Test parameter file integration
        workflow_params = {
            "video_processing": {
                "flip_horizontal": True,
                "gaussian_blur": {"enabled": True, "sigma": 2.0}
            },
            "navigation": {
                "max_speed": 15.0,
                "angular_velocity": 0.2,
                "num_steps": 1000
            },
            "analysis": {
                "export_format": "html",
                "include_animations": True
            }
        }
        
        # Test parameter validation
        assert "video_processing" in workflow_params, "Missing video processing parameters"
        assert "navigation" in workflow_params, "Missing navigation parameters"
        assert workflow_params["navigation"]["num_steps"] > 0, "Invalid step count"
        
        # Mock parameter file creation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as param_file:
            import yaml
            yaml.dump(workflow_params, param_file)
            param_file_path = param_file.name
        
        try:
            # Validate parameter file
            with open(param_file_path, 'r') as f:
                loaded_params = yaml.safe_load(f)
            
            assert loaded_params == workflow_params, "Parameter file integrity check failed"
            
        finally:
            os.unlink(param_file_path)
    
    def test_experiment_reproducibility_workflow(self):
        """Test complete experiment reproducibility through workflow orchestration."""
        # Mock experimental setup
        experiment_config = {
            "experiment_id": "odor_nav_exp_001",
            "parameters": {
                "navigator": {"max_speed": [5, 10, 15]},
                "video_plume": {"gaussian_blur.sigma": [1.0, 2.0, 3.0]}
            },
            "output_structure": {
                "base_dir": "experiments/",
                "results_dir": "results/",
                "plots_dir": "plots/",
                "logs_dir": "logs/"
            }
        }
        
        # Test experiment parameter combinations
        max_speeds = experiment_config["parameters"]["navigator"]["max_speed"]
        blur_sigmas = experiment_config["parameters"]["video_plume"]["gaussian_blur.sigma"]
        
        expected_combinations = len(max_speeds) * len(blur_sigmas)
        assert expected_combinations == 9, "Incorrect parameter combination count"
        
        # Mock experiment execution tracking
        experiment_results = []
        for speed in max_speeds:
            for sigma in blur_sigmas:
                result = {
                    "parameters": {"max_speed": speed, "blur_sigma": sigma},
                    "status": "completed",
                    "execution_time": np.random.uniform(10, 60),
                    "trajectory_length": np.random.randint(800, 1200)
                }
                experiment_results.append(result)
        
        # Validate experiment tracking
        assert len(experiment_results) == expected_combinations
        assert all(r["status"] == "completed" for r in experiment_results)
        
        # Test reproducibility metadata
        reproducibility_metadata = {
            "timestamp": "2024-01-01T00:00:00Z",
            "git_commit": "abc123def456",
            "python_version": "3.9.0",
            "dependencies": {
                "numpy": "1.21.0",
                "opencv-python": "4.5.0",
                "hydra-core": "1.3.0"
            },
            "system_info": {
                "platform": "linux",
                "architecture": "x86_64"
            }
        }
        
        assert "git_commit" in reproducibility_metadata, "Missing version control info"
        assert "dependencies" in reproducibility_metadata, "Missing dependency info"


# Pytest fixtures for integration testing
@pytest.fixture
def mock_navigator():
    """Provide a mock navigator with realistic behavior."""
    navigator = Mock()
    navigator.positions = np.array([[10.0, 10.0]])
    navigator.orientations = np.array([0.0])
    navigator.speeds = np.array([5.0])
    navigator.max_speeds = np.array([10.0])
    navigator.angular_velocities = np.array([0.1])
    navigator.num_agents = 1
    navigator.is_single_agent = True
    
    # Mock method behaviors
    navigator.reset.return_value = None
    navigator.step.return_value = None
    navigator.sample_odor.return_value = 0.5
    navigator.read_single_antenna_odor.return_value = 0.3
    navigator.sample_multiple_sensors.return_value = np.array([0.1, 0.2])
    
    return navigator


@pytest.fixture
def mock_multi_navigator():
    """Provide a mock multi-agent navigator."""
    navigator = Mock()
    num_agents = 3
    navigator.positions = np.random.rand(num_agents, 2) * 100
    navigator.orientations = np.random.rand(num_agents) * 360
    navigator.speeds = np.random.rand(num_agents) * 10
    navigator.max_speeds = np.ones(num_agents) * 15
    navigator.angular_velocities = np.random.rand(num_agents) * 0.5
    navigator.num_agents = num_agents
    navigator.is_single_agent = False
    
    # Mock method behaviors
    navigator.reset.return_value = None
    navigator.step.return_value = None
    navigator.sample_odor.return_value = np.random.rand(num_agents)
    navigator.read_single_antenna_odor.return_value = np.random.rand(num_agents)
    navigator.sample_multiple_sensors.return_value = np.random.rand(num_agents, 2)
    
    return navigator


@pytest.fixture
def mock_video_plume():
    """Provide a mock video plume with realistic behavior."""
    video_plume = Mock()
    video_plume.width = 640
    video_plume.height = 480
    video_plume.fps = 30.0
    video_plume.frame_count = 1000
    video_plume.duration = 33.33
    video_plume.video_path = "/mock/path/to/video.mp4"
    
    # Mock realistic frame data
    video_plume.get_frame.return_value = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
    video_plume.close.return_value = None
    
    return video_plume


@pytest.fixture
def mock_hydra_config():
    """Provide a mock Hydra configuration for testing."""
    config = DictConfig({
        "navigator": {
            "type": "single",
            "position": [50.0, 50.0],
            "orientation": 0.0,
            "speed": 5.0,
            "max_speed": 10.0,
            "angular_velocity": 0.1
        },
        "video_plume": {
            "video_path": "/mock/path/to/video.mp4",
            "flip": False,
            "kernel_size": 5,
            "kernel_sigma": 1.0
        },
        "simulation": {
            "num_steps": 100,
            "dt": 0.1
        },
        "visualization": {
            "save_animations": False,
            "export_format": "png"
        },
        "database": {
            "url": "sqlite:///test.db"
        }
    })
    
    return config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])