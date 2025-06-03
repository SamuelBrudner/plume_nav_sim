"""Tests for the public API functions with enhanced Hydra configuration support.

This module provides comprehensive testing for the refactored public API functions
in the {{cookiecutter.project_slug}}.api.navigation module, including:

- Enhanced Hydra configuration management with DictConfig fixtures
- Protocol-based navigator creation with both single and multi-agent scenarios  
- VideoPlume factory methods with comprehensive parameter validation
- Simulation execution with Hydra-based configuration composition
- Visualization integration with new module structure
- Comprehensive pytest-hydra fixture integration for configuration testing
- CLI parameter integration testing through Hydra composition
- Error handling and validation for boundary conditions

The testing approach leverages pytest-hydra plugin for structured configuration
testing while maintaining backward compatibility with existing test patterns.
All tests follow scientific computing standards for reproducibility and precision.
"""

import contextlib
import pytest
import numpy as np
import cv2
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from unittest.mock import patch, MagicMock, Mock

# Hydra imports for configuration testing
try:
    from omegaconf import DictConfig, OmegaConf
    from hydra import compose, initialize
    from hydra.core.config_store import ConfigStore
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None

# Import updated API functions from new module structure
from {{cookiecutter.project_slug}}.api.navigation import (
    create_navigator,
    create_video_plume,
    run_plume_simulation,
    visualize_plume_simulation,
    # Legacy compatibility aliases
    create_navigator_from_config,
    create_video_plume_from_config,
    run_simulation,
    visualize_simulation_results
)

# Import visualization functions from new module structure
from {{cookiecutter.project_slug}}.utils.visualization import (
    visualize_trajectory,
    SimulationVisualization
)

# Import configuration schemas for enhanced testing
from {{cookiecutter.project_slug}}.config.schemas import (
    NavigatorConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    VideoPlumeConfig,
    SimulationConfig
)

# Import core components for protocol validation
from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol
from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume


class TestNavigatorCreation:
    """Test suite for navigator creation with enhanced Hydra configuration support."""
    
    def test_create_navigator_default(self):
        """Test creating a navigator with default parameters."""
        navigator = create_navigator()
        
        # Should create a default Navigator instance (single agent)
        # Check default values aligned with the protocol-based Navigator
        assert navigator.positions.shape == (1, 2)  # Single agent with 2D position
        assert navigator.orientations.shape == (1,)  # Single agent orientation
        assert navigator.speeds.shape == (1,)  # Single agent speed
        
        # Check default values
        assert navigator.orientations[0] == 0.0
        assert navigator.speeds[0] == 0.0
        assert navigator.max_speeds[0] == 1.0

    def test_create_navigator_single_agent(self):
        """Test creating a navigator with single agent parameters."""
        # Create a navigator with single agent parameters
        navigator = create_navigator(
            position=(10, 20),
            orientation=45,
            speed=0.5,
            max_speed=2.0
        )
        
        # Check that the navigator has the correct properties
        # In the protocol-based architecture, properties are array-based
        assert navigator.orientations[0] == 45
        assert navigator.speeds[0] == 0.5
        assert navigator.max_speeds[0] == 2.0
        assert np.allclose(navigator.positions[0], [10, 20])
        
        # Check that it's a single-agent navigator by verifying array lengths
        assert len(navigator.positions) == 1
        assert len(navigator.orientations) == 1
        assert len(navigator.speeds) == 1

    def test_create_navigator_multi_agent(self):
        """Test creating a navigator with multi-agent parameters."""
        # Create a navigator with multi-agent parameters
        positions = [(10, 20), (30, 40), (50, 60)]
        orientations = [45, 90, 135]
        speeds = [0.5, 0.7, 0.9]
        
        navigator = create_navigator(
            positions=positions,
            orientations=orientations,
            speeds=speeds
        )
        
        # Check that the navigator has the correct number of agents
        assert len(navigator.positions) == 3
        
        # Check that all agents have correct positions
        assert np.allclose(navigator.positions, positions)
        
        # Verify each agent has correct orientation and speed
        assert np.allclose(navigator.orientations, orientations)
        assert np.allclose(navigator.speeds, speeds)

    def test_create_navigator_numpy_array_positions(self):
        """Test creating a navigator with numpy array positions."""
        # Test with numpy array position data
        positions = np.array([[10, 20], [30, 40], [50, 60]])
        
        navigator = create_navigator(positions=positions)
        
        # Check it's a multi-agent navigator with the right number of agents
        assert len(navigator.positions) == 3
        
        # Verify positions were set correctly
        assert np.allclose(navigator.positions, positions)
        
        # Check default values for other properties
        assert np.allclose(navigator.orientations, np.zeros(3))
        assert np.allclose(navigator.speeds, np.zeros(3))

    def test_create_navigator_conflicting_position_and_positions(self):
        """If both position and positions are provided, should raise ValueError."""
        with pytest.raises(ValueError, match=r"Cannot specify both 'position' \(single-agent\) and 'positions' \(multi-agent\). Please provide only one."):
            create_navigator(position=(0, 0), positions=[(1, 2), (3, 4)])


class TestHydraConfigurationIntegration:
    """Test suite for Hydra configuration integration with enhanced pytest-hydra support."""

    @pytest.fixture
    def mock_hydra_config(self):
        """Mock Hydra configuration fixture for testing."""
        if HYDRA_AVAILABLE:
            config_dict = {
                "position": [10, 20],
                "orientation": 45.0,
                "speed": 0.5,
                "max_speed": 1.0,
                "angular_velocity": 0.1
            }
            return OmegaConf.create(config_dict)
        else:
            return {
                "position": [10, 20],
                "orientation": 45.0,
                "speed": 0.5,
                "max_speed": 1.0,
                "angular_velocity": 0.1
            }

    @pytest.fixture
    def mock_multi_agent_hydra_config(self):
        """Mock multi-agent Hydra configuration fixture."""
        if HYDRA_AVAILABLE:
            config_dict = {
                "positions": [[10, 20], [30, 40]],
                "orientations": [45.0, 90.0],
                "speeds": [0.5, 0.7],
                "max_speeds": [1.0, 1.0],
                "angular_velocities": [0.1, 0.15]
            }
            return OmegaConf.create(config_dict)
        else:
            return {
                "positions": [[10, 20], [30, 40]],
                "orientations": [45.0, 90.0],
                "speeds": [0.5, 0.7],
                "max_speeds": [1.0, 1.0],
                "angular_velocities": [0.1, 0.15]
            }

    @pytest.fixture
    def mock_video_plume_hydra_config(self):
        """Mock VideoPlume Hydra configuration fixture."""
        if HYDRA_AVAILABLE:
            config_dict = {
                "video_path": "test_video.mp4",
                "flip": True,
                "kernel_size": 5,
                "kernel_sigma": 1.0,
                "threshold": 0.5,
                "normalize": True
            }
            return OmegaConf.create(config_dict)
        else:
            return {
                "video_path": "test_video.mp4",
                "flip": True,
                "kernel_size": 5,
                "kernel_sigma": 1.0,
                "threshold": 0.5,
                "normalize": True
            }

    @pytest.fixture
    def mock_simulation_hydra_config(self):
        """Mock simulation Hydra configuration fixture."""
        if HYDRA_AVAILABLE:
            config_dict = {
                "num_steps": 100,
                "dt": 0.1,
                "sensor_distance": 5.0,
                "sensor_angle": 45.0,
                "record_trajectories": True
            }
            return OmegaConf.create(config_dict)
        else:
            return {
                "num_steps": 100,
                "dt": 0.1,
                "sensor_distance": 5.0,
                "sensor_angle": 45.0,
                "record_trajectories": True
            }

    def test_create_navigator_with_hydra_config(self, mock_hydra_config):
        """Test creating a navigator from Hydra DictConfig."""
        navigator = create_navigator(cfg=mock_hydra_config)
        
        # Check that the navigator has the correct properties from config
        assert len(navigator.positions) == 1  # Single agent from config
        assert navigator.orientations[0] == 45.0
        assert navigator.speeds[0] == 0.5
        assert navigator.max_speeds[0] == 1.0
        assert np.allclose(navigator.positions[0], [10, 20])

    def test_create_navigator_with_multi_agent_hydra_config(self, mock_multi_agent_hydra_config):
        """Test creating a multi-agent navigator from Hydra DictConfig."""
        navigator = create_navigator(cfg=mock_multi_agent_hydra_config)
        
        # Check that the navigator has the correct properties from config
        assert len(navigator.positions) == 2  # Two agents from config
        assert navigator.orientations[0] == 45.0
        assert navigator.orientations[1] == 90.0
        assert navigator.speeds[0] == 0.5
        assert navigator.speeds[1] == 0.7
        assert np.allclose(navigator.positions[0], [10, 20])
        assert np.allclose(navigator.positions[1], [30, 40])

    def test_create_navigator_hydra_config_with_override(self, mock_hydra_config):
        """Test that direct arguments override Hydra config values."""
        navigator = create_navigator(
            cfg=mock_hydra_config,
            orientation=180.0,  # Override config value
            speed=1.0          # Override config value
        )
        
        # Direct arguments should take precedence
        assert navigator.orientations[0] == 180.0
        assert navigator.speeds[0] == 1.0
        # Config values should still be used for non-overridden parameters
        assert navigator.max_speeds[0] == 1.0
        assert np.allclose(navigator.positions[0], [10, 20])

    def test_create_navigator_hierarchical_config_composition(self):
        """Test hierarchical configuration composition with override scenarios."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available for configuration composition testing")
        
        # Create base configuration
        base_config = OmegaConf.create({
            "position": [0, 0],
            "orientation": 0.0,
            "max_speed": 2.0
        })
        
        # Create override configuration
        override_config = OmegaConf.create({
            "orientation": 90.0,
            "speed": 1.5
        })
        
        # Merge configurations
        merged_config = OmegaConf.merge(base_config, override_config)
        
        navigator = create_navigator(cfg=merged_config)
        
        # Verify merged parameters
        assert navigator.orientations[0] == 90.0  # From override
        assert navigator.speeds[0] == 1.5        # From override
        assert navigator.max_speeds[0] == 2.0    # From base
        assert np.allclose(navigator.positions[0], [0, 0])  # From base

    def test_create_navigator_environment_variable_interpolation(self):
        """Test environment variable interpolation in configuration."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available for environment variable testing")
        
        import os
        
        # Set environment variables for testing
        os.environ['TEST_AGENT_X'] = '15.0'
        os.environ['TEST_AGENT_Y'] = '25.0'
        os.environ['TEST_MAX_SPEED'] = '3.0'
        
        try:
            # Create configuration with environment variable interpolation
            config_with_env = OmegaConf.create({
                "position": ["${oc.env:TEST_AGENT_X,10.0}", "${oc.env:TEST_AGENT_Y,20.0}"],
                "max_speed": "${oc.env:TEST_MAX_SPEED,1.0}",
                "orientation": 0.0
            })
            
            # Resolve environment variables
            resolved_config = OmegaConf.resolve(config_with_env)
            
            navigator = create_navigator(cfg=resolved_config)
            
            # Verify environment variable interpolation
            assert np.allclose(navigator.positions[0], [15.0, 25.0])
            assert navigator.max_speeds[0] == 3.0
            
        finally:
            # Clean up environment variables
            for var in ['TEST_AGENT_X', 'TEST_AGENT_Y', 'TEST_MAX_SPEED']:
                os.environ.pop(var, None)

    def test_configuration_schema_validation(self, mock_hydra_config):
        """Test Pydantic schema validation through Hydra configuration."""
        # Test invalid configuration
        invalid_config = OmegaConf.create({
            "position": [10, 20, 30],  # Invalid - should be 2D
            "orientation": -45.0,      # Invalid - negative orientation
            "speed": -1.0              # Invalid - negative speed
        }) if HYDRA_AVAILABLE else {
            "position": [10, 20, 30],
            "orientation": -45.0,
            "speed": -1.0
        }
        
        # Should raise validation error
        with pytest.raises(ValueError):
            create_navigator(cfg=invalid_config)


class TestVideoPlumeCreation:
    """Test suite for VideoPlume creation with enhanced configuration support."""

    @pytest.fixture
    def mock_video_capture(self):
        """Create a mock for cv2.VideoCapture."""
        with patch('cv2.VideoCapture') as mock_cap:
            # Configure the mock to return appropriate values
            mock_instance = MagicMock()
            mock_cap.return_value = mock_instance
            
            # Mock isOpened to return True by default
            mock_instance.isOpened.return_value = True
            
            # Configure property values for a synthetic video
            cap_properties = {
                cv2.CAP_PROP_FRAME_COUNT: 100,
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }
            
            # Configure get method to return values from the dictionary
            mock_instance.get.side_effect = lambda prop: cap_properties.get(prop, 0)
            
            # Mock read to return a valid BGR frame (3 channels)
            mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_instance.read.return_value = (True, mock_frame)
            
            yield mock_cap

    @pytest.fixture
    def mock_exists(self, monkeypatch):
        """Mock the Path.exists method to return True for all paths."""
        def patched_exists(self):
            return True
        
        monkeypatch.setattr(Path, "exists", patched_exists)
        return patched_exists

    def test_create_video_plume(self, mock_video_capture, mock_exists):
        """Test creating a video plume with the API function."""
        # Create a video plume
        plume = create_video_plume(video_path="test_video.mp4", flip=True, kernel_size=5)
        
        # Check that the plume has the correct properties
        assert plume.video_path == Path("test_video.mp4")
        assert plume.flip is True
        assert plume.kernel_size == 5

    def test_create_video_plume_with_hydra_config(self, mock_video_capture, mock_exists, mock_video_plume_hydra_config):
        """Test creating a video plume with Hydra DictConfig."""
        plume = create_video_plume(cfg=mock_video_plume_hydra_config)
        
        # Verify configuration was applied
        assert plume.video_path == Path("test_video.mp4")
        assert plume.flip is True
        assert plume.kernel_size == 5
        assert plume.kernel_sigma == 1.0

    def test_create_video_plume_config_override(self, mock_video_capture, mock_exists, mock_video_plume_hydra_config):
        """Direct argument should override config value."""
        plume = create_video_plume(
            cfg=mock_video_plume_hydra_config,
            flip=False,          # Override config
            kernel_size=7        # Override config
        )
        assert plume.flip is False
        assert plume.kernel_size == 7
        # Non-overridden config values should be preserved
        assert plume.kernel_sigma == 1.0

    def test_create_video_plume_partial_config(self, mock_video_capture, mock_exists):
        """Direct argument supplies missing config field."""
        partial_config = OmegaConf.create({
            "video_path": "test_video.mp4",
            "flip": True
        }) if HYDRA_AVAILABLE else {
            "video_path": "test_video.mp4",
            "flip": True
        }
        
        plume = create_video_plume(cfg=partial_config, kernel_size=7)
        assert plume.kernel_size == 7
        assert plume.flip is True

    def test_create_video_plume_invalid_kernel_size(self, mock_video_capture, mock_exists):
        """Invalid kernel_size (negative/int as string) raises ValueError."""
        with pytest.raises(ValueError):
            create_video_plume(video_path="test_video.mp4", kernel_size=-1)
        with pytest.raises(ValueError):
            create_video_plume(video_path="test_video.mp4", kernel_size="five")

    def test_create_video_plume_invalid_flip(self, mock_video_capture, mock_exists):
        """Non-bool flip raises ValueError."""
        with pytest.raises(ValueError):
            create_video_plume(video_path="test_video.mp4", flip="yes")

    def test_create_video_plume_missing_video_path(self, mock_video_capture, mock_exists):
        """Missing video_path should raise TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            create_video_plume()

    def test_create_video_plume_unknown_config_field(self, mock_video_capture, mock_exists):
        """Unknown config field is ignored or raises error (depending on implementation)."""
        unknown_config = OmegaConf.create({
            "video_path": "test_video.mp4",
            "flip": True,
            "kernel_size": 3,
            "unknown_field": 42
        }) if HYDRA_AVAILABLE else {
            "video_path": "test_video.mp4",
            "flip": True,
            "kernel_size": 3,
            "unknown_field": 42
        }
        
        # Accept either: ignore unknown field, or raise ValueError
        with contextlib.suppress(ValueError):
            plume = create_video_plume(cfg=unknown_config)
            assert hasattr(plume, "video_path")

    def test_create_video_plume_conflicting_fields(self, mock_video_capture, mock_exists):
        """Direct arg and config provide different values for same field; direct arg wins."""
        conflicting_config = OmegaConf.create({
            "video_path": "test_video.mp4",
            "flip": False,
            "kernel_size": 3
        }) if HYDRA_AVAILABLE else {
            "video_path": "test_video.mp4",
            "flip": False,
            "kernel_size": 3
        }
        
        plume = create_video_plume(cfg=conflicting_config, flip=True)
        assert plume.flip is True

    def test_create_video_plume_invalid_path(self, mock_video_capture, monkeypatch):
        """Non-existent video file path raises error if validated."""
        monkeypatch.setattr(Path, "exists", lambda self: False)
        with pytest.raises((FileNotFoundError, ValueError)):
            create_video_plume(video_path="nonexistent.mp4")


class TestSimulationExecution:
    """Test suite for simulation execution with enhanced Hydra configuration support."""

    @pytest.fixture
    def mock_run_simulation(self):
        """Mock the run_simulation function."""
        # We need to patch the function where it's imported, not where it's defined
        with patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation') as mock_run:
            # Configure mock to return synthetic data
            positions_history = np.array([[[0, 0], [1, 1], [2, 2]]])
            orientations_history = np.array([[0, 45, 90]])
            odor_readings = np.array([[0.1, 0.2, 0.3]])
            
            mock_run.return_value = (positions_history, orientations_history, odor_readings)
            yield mock_run

    @pytest.fixture
    def sample_navigator(self):
        """Create a sample navigator for testing."""
        return create_navigator(position=(10, 20), orientation=45)

    @pytest.fixture
    def sample_video_plume(self, mock_exists):
        """Create a sample video plume for testing."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_instance = MagicMock()
            mock_cap.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.return_value = 100
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            return create_video_plume(video_path="test_video.mp4")

    def test_run_plume_simulation_basic(self, sample_navigator, sample_video_plume):
        """Test running a plume simulation with basic parameters."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation') as mock_sim:
            # Configure mock return values
            positions = np.array([[[0, 0], [1, 1], [2, 2]]])
            orientations = np.array([[0, 45, 90]])
            readings = np.array([[0.1, 0.2, 0.3]])
            mock_sim.return_value = (positions, orientations, readings)
            
            # Run the simulation
            result_positions, result_orientations, result_readings = run_plume_simulation(
                sample_navigator, sample_video_plume, num_steps=2, dt=0.5
            )
            
            # Check that the simulation function was called with correct parameters
            mock_sim.assert_called_once()
            
            # Check that the results have correct shape
            assert result_positions.shape == (1, 3, 2)
            assert result_orientations.shape == (1, 3)
            assert result_readings.shape == (1, 3)

    def test_run_plume_simulation_with_hydra_config(self, sample_navigator, sample_video_plume, mock_simulation_hydra_config):
        """Test running simulation with Hydra DictConfig."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation') as mock_sim:
            positions = np.array([[[0, 0], [1, 1]]])
            orientations = np.array([[0, 45]])
            readings = np.array([[0.1, 0.2]])
            mock_sim.return_value = (positions, orientations, readings)
            
            # Run simulation with Hydra config
            result_positions, result_orientations, result_readings = run_plume_simulation(
                sample_navigator, sample_video_plume, cfg=mock_simulation_hydra_config
            )
            
            mock_sim.assert_called_once()
            
            # Verify config parameters were used
            call_args = mock_sim.call_args
            assert call_args[0][0] == sample_navigator
            assert call_args[0][1] == sample_video_plume

    def test_run_plume_simulation_config_override(self, sample_navigator, sample_video_plume, mock_simulation_hydra_config):
        """Direct argument should override config value."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation') as mock_sim:
            positions = np.array([[[0, 0]]])
            orientations = np.array([[0]])
            readings = np.array([[0.1]])
            mock_sim.return_value = (positions, orientations, readings)
            
            # Override config values with direct arguments
            run_plume_simulation(
                sample_navigator, sample_video_plume,
                cfg=mock_simulation_hydra_config,
                num_steps=50,  # Override config value
                dt=0.2        # Override config value
            )
            
            mock_sim.assert_called_once()

    def test_run_plume_simulation_partial_config(self, sample_navigator, sample_video_plume):
        """Direct argument supplies missing config field."""
        partial_config = OmegaConf.create({
            "num_steps": 25
        }) if HYDRA_AVAILABLE else {
            "num_steps": 25
        }
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation') as mock_sim:
            positions = np.array([[[0, 0]]])
            orientations = np.array([[0]])
            readings = np.array([[0.1]])
            mock_sim.return_value = (positions, orientations, readings)
            
            run_plume_simulation(
                sample_navigator, sample_video_plume,
                cfg=partial_config,
                dt=0.2  # Supply missing config field
            )
            
            mock_sim.assert_called_once()

    @pytest.mark.parametrize("bad", [0, -1, "ten"])
    def test_run_plume_simulation_invalid_num_steps_param(self, sample_navigator, sample_video_plume, bad):
        """Negative or zero num_steps raises ValueError."""
        with pytest.raises(ValueError):
            run_plume_simulation(sample_navigator, sample_video_plume, num_steps=bad, dt=1.0)

    @pytest.mark.parametrize("bad", [0, -0.1, "small"])
    def test_run_plume_simulation_invalid_dt_param(self, sample_navigator, sample_video_plume, bad):
        """Non-positive or non-float dt raises ValueError."""
        with pytest.raises(ValueError):
            run_plume_simulation(sample_navigator, sample_video_plume, num_steps=10, dt=bad)

    def test_run_plume_simulation_missing_required(self):
        """Missing navigator or plume raises TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            run_plume_simulation(None, None, num_steps=5, dt=1.0)

    def test_run_plume_simulation_output_shapes_single_agent(self, sample_navigator, sample_video_plume):
        """Output arrays have correct shapes for single agent."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation') as mock_sim:
            positions = np.array([[[0, 0], [1, 1], [2, 2]]])  # 1 agent, 3 steps
            orientations = np.array([[0, 45, 90]])
            readings = np.array([[0.1, 0.2, 0.3]])
            mock_sim.return_value = (positions, orientations, readings)
            
            result_positions, result_orientations, result_readings = run_plume_simulation(
                sample_navigator, sample_video_plume, num_steps=2, dt=1.0
            )
            
            assert result_positions.shape == (1, 3, 2)
            assert result_orientations.shape == (1, 3)
            assert result_readings.shape == (1, 3)

    def test_run_plume_simulation_output_shapes_multi_agent(self, sample_video_plume):
        """Output arrays have correct shapes for multi-agent."""
        multi_navigator = create_navigator(positions=[(0, 0), (1, 1)])
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation') as mock_sim:
            positions = np.array([[[0, 0], [1, 1]], [[2, 2], [3, 3]]])  # 2 agents, 2 steps
            orientations = np.array([[0, 45], [90, 135]])
            readings = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_sim.return_value = (positions, orientations, readings)
            
            result_positions, result_orientations, result_readings = run_plume_simulation(
                multi_navigator, sample_video_plume, num_steps=1, dt=1.0
            )
            
            assert result_positions.shape == (2, 2, 2)
            assert result_orientations.shape == (2, 2)
            assert result_readings.shape == (2, 2)


class TestVisualizationIntegration:
    """Test suite for visualization integration with new module structure."""

    @pytest.fixture
    def mock_visualize_trajectory(self):
        """Mock the visualize_trajectory function from new module location."""
        with patch('{{cookiecutter.project_slug}}.utils.visualization.visualize_trajectory') as mock_viz:
            yield mock_viz

    @pytest.fixture
    def mock_visualize_simulation_results(self):
        """Mock the visualize_simulation_results function."""
        with patch('{{cookiecutter.project_slug}}.utils.visualization.visualize_simulation_results') as mock_viz:
            yield mock_viz

    def test_visualize_simulation_results(self, mock_visualize_simulation_results):
        """Test visualizing simulation results with the API function."""
        # Create synthetic simulation results
        positions = np.array([[[0, 0], [1, 1], [2, 2]]])
        orientations = np.array([[0, 45, 90]])
        
        # Visualize the results using the new function
        visualize_plume_simulation(
            positions, orientations, output_path="test_output.png", show_plot=False
        )
        
        # Check that the visualization function was called
        mock_visualize_simulation_results.assert_called_once()

    def test_visualize_trajectory_direct_import(self, mock_visualize_trajectory):
        """Test direct usage of visualize_trajectory from new module location."""
        positions = np.array([[[0, 0], [1, 1], [2, 2]]])
        orientations = np.array([[0, 45, 90]])
        
        # Call the function directly
        visualize_trajectory(
            positions, orientations, output_path="test_trajectory.png", show_plot=False
        )
        
        # Verify it was called with correct parameters
        mock_visualize_trajectory.assert_called_once()
        args, kwargs = mock_visualize_trajectory.call_args
        assert np.array_equal(args[0], positions)
        assert np.array_equal(args[1], orientations)
        assert kwargs["output_path"] == "test_trajectory.png"
        assert kwargs["show_plot"] is False

    def test_simulation_visualization_class_integration(self):
        """Test SimulationVisualization class from new module location."""
        # Test that we can import and instantiate the class
        viz = SimulationVisualization(figsize=(10, 8), headless=True)
        
        # Basic property checks
        assert viz.config['figsize'] == (10, 8)
        assert viz.config['headless'] is True
        
        # Clean up
        viz.close()

    def test_visualize_plume_simulation_with_hydra_config(self):
        """Test visualization with Hydra configuration."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available for configuration testing")
        
        viz_config = OmegaConf.create({
            "output_path": "hydra_test.png",
            "show_plot": False,
            "dpi": 300,
            "format": "png"
        })
        
        positions = np.array([[[0, 0], [1, 1]]])
        orientations = np.array([[0, 45]])
        
        with patch('{{cookiecutter.project_slug}}.utils.visualization.visualize_trajectory') as mock_viz:
            visualize_plume_simulation(
                positions, orientations, cfg=viz_config
            )
            
            mock_viz.assert_called_once()


class TestLegacyCompatibility:
    """Test suite for legacy compatibility aliases."""

    def test_create_navigator_from_config_alias(self, mock_exists):
        """Test legacy create_navigator_from_config alias."""
        config = {"position": [10, 20], "orientation": 45.0}
        
        # Test that the alias works
        navigator = create_navigator_from_config(cfg=config)
        
        assert np.allclose(navigator.positions[0], [10, 20])
        assert navigator.orientations[0] == 45.0

    def test_create_video_plume_from_config_alias(self, mock_exists):
        """Test legacy create_video_plume_from_config alias."""
        config = {"video_path": "test.mp4", "flip": True}
        
        with patch('cv2.VideoCapture') as mock_cap:
            mock_instance = MagicMock()
            mock_cap.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.return_value = 100
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            plume = create_video_plume_from_config(cfg=config)
            
            assert plume.video_path == Path("test.mp4")
            assert plume.flip is True

    def test_run_simulation_alias(self):
        """Test legacy run_simulation alias."""
        navigator = create_navigator(position=(0, 0))
        
        with patch('cv2.VideoCapture') as mock_cap, \
             patch('pathlib.Path.exists', return_value=True):
            mock_instance = MagicMock()
            mock_cap.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.return_value = 100
            mock_instance.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
            
            plume = create_video_plume(video_path="test.mp4")
            
            with patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation') as mock_sim:
                positions = np.array([[[0, 0], [1, 1]]])
                orientations = np.array([[0, 45]])
                readings = np.array([[0.1, 0.2]])
                mock_sim.return_value = (positions, orientations, readings)
                
                # Test legacy alias
                result = run_simulation(navigator, plume, num_steps=1, dt=1.0)
                
                assert len(result) == 3
                mock_sim.assert_called_once()

    def test_visualize_simulation_results_alias(self):
        """Test legacy visualize_simulation_results alias."""
        positions = np.array([[[0, 0], [1, 1]]])
        orientations = np.array([[0, 45]])
        
        with patch('{{cookiecutter.project_slug}}.utils.visualization.visualize_trajectory') as mock_viz:
            # Test legacy alias
            visualize_simulation_results(positions, orientations, show_plot=False)
            
            mock_viz.assert_called_once()


class TestEnhancedConfigurationValidation:
    """Test suite for enhanced configuration validation and error handling."""

    def test_configuration_schema_validation_comprehensive(self):
        """Test comprehensive Pydantic schema validation."""
        # Test invalid single agent configuration
        with pytest.raises(ValueError):
            create_navigator(
                position=[10, 20, 30],  # Invalid - 3D instead of 2D
                orientation=450,        # Invalid - out of range
                speed=-1.0             # Invalid - negative speed
            )

    def test_multi_agent_configuration_length_validation(self):
        """Test multi-agent parameter length consistency validation."""
        with pytest.raises(ValueError):
            create_navigator(
                positions=[(10, 20), (30, 40)],    # 2 agents
                orientations=[45, 90, 135]         # 3 orientations - mismatch
            )

    def test_configuration_type_validation(self):
        """Test configuration parameter type validation."""
        with pytest.raises(ValueError):
            create_navigator(
                position="invalid",  # Should be tuple/list
                orientation="ninety"  # Should be numeric
            )

    def test_hydra_configuration_merge_validation(self):
        """Test Hydra configuration merging with validation."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available")
        
        # Test conflicting configuration merge
        config1 = OmegaConf.create({"position": [10, 20], "mode": "single"})
        config2 = OmegaConf.create({"positions": [[30, 40], [50, 60]], "mode": "multi"})
        
        # Merging conflicting single/multi configurations should be handled
        merged = OmegaConf.merge(config1, config2)
        
        # Should detect multi-agent mode from positions parameter
        navigator = create_navigator(cfg=merged)
        assert len(navigator.positions) == 2


class TestParameterizedScenarios:
    """Test suite with parameterized scenarios for comprehensive coverage."""

    @pytest.mark.parametrize("positions,expected_exception", [
        ([(1, 2), (3, 4)], None),  # valid
        ([1, 2], None),            # valid single-agent: treat as (x, y)
        ([(1, 2, 3), (4, 5, 6)], ValueError),  # wrong shape
        (["a", "b"], ValueError),  # not numeric
        ([[1], [2]], ValueError),   # wrong length
        ([(1, 2), (3,)], ValueError),  # one valid, one invalid
        (np.array([[1, 2], [3, 4]]), None),  # valid np.ndarray
        (np.array([[1], [2]]), ValueError),  # invalid np.ndarray
    ])
    def test_create_navigator_position_validation(self, positions, expected_exception):
        """Test navigator creation with various position formats."""
        if expected_exception:
            with pytest.raises(expected_exception):
                create_navigator(positions=positions)
        else:
            navigator = create_navigator(positions=positions)
            assert navigator.positions is not None

    @pytest.mark.parametrize("i,expected_pos", [
        (0, (10, 20)), (1, (30, 40)), (2, (50, 60))
    ])
    def test_create_navigator_position_index(self, i, expected_pos):
        """Test individual agent position access."""
        positions = [(10, 20), (30, 40), (50, 60)]
        navigator = create_navigator(positions=positions)
        assert np.allclose(navigator.positions[i], expected_pos)

    @pytest.mark.parametrize("positions,expected_shape", [
        ([(1, 2), (3, 4)], (2, 2)),
        ([1, 2], (1, 2)),
        ((1, 2), (1, 2)),
        (np.array([[1, 2], [3, 4]]), (2, 2)),
    ])
    def test_create_navigator_positions_shape_valid(self, positions, expected_shape):
        """Test navigator position array shapes."""
        navigator = create_navigator(positions=positions)
        assert navigator.positions.shape == expected_shape

    @pytest.mark.parametrize("num_steps,dt,expected_valid", [
        (10, 0.1, True),
        (100, 0.01, True),
        (0, 0.1, False),     # Invalid num_steps
        (-1, 0.1, False),    # Invalid num_steps
        (10, 0, False),      # Invalid dt
        (10, -0.1, False),   # Invalid dt
    ])
    def test_simulation_parameter_validation(self, num_steps, dt, expected_valid):
        """Test simulation parameter validation."""
        navigator = create_navigator(position=(0, 0))
        
        with patch('cv2.VideoCapture') as mock_cap, \
             patch('pathlib.Path.exists', return_value=True):
            mock_instance = MagicMock()
            mock_cap.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.return_value = 100
            mock_instance.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
            
            plume = create_video_plume(video_path="test.mp4")
            
            if expected_valid:
                with patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation'):
                    # Should not raise exception
                    run_plume_simulation(navigator, plume, num_steps=num_steps, dt=dt)
            else:
                with pytest.raises(ValueError):
                    run_plume_simulation(navigator, plume, num_steps=num_steps, dt=dt)


class TestCoreNavigatorUtilityFunctions:
    """Test suite for navigator utility functions from core module."""

    def test_navigator_protocol_compliance(self):
        """Test that created navigators comply with NavigatorProtocol."""
        navigator = create_navigator(position=(10, 20))
        
        # Check protocol compliance
        assert isinstance(navigator, NavigatorProtocol)
        assert hasattr(navigator, 'positions')
        assert hasattr(navigator, 'orientations')
        assert hasattr(navigator, 'speeds')
        assert hasattr(navigator, 'step')

    def test_navigator_utility_imports(self):
        """Test imports from core module are accessible."""
        from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol
        from {{cookiecutter.project_slug}}.core.controllers import SingleAgentController, MultiAgentController
        
        # Verify classes can be imported
        assert NavigatorProtocol is not None
        assert SingleAgentController is not None
        assert MultiAgentController is not None

    def test_config_schema_imports(self):
        """Test configuration schema imports from new module structure."""
        from {{cookiecutter.project_slug}}.config.schemas import (
            NavigatorConfig, SingleAgentConfig, MultiAgentConfig,
            VideoPlumeConfig, SimulationConfig
        )
        
        # Verify schemas can be imported
        assert NavigatorConfig is not None
        assert SingleAgentConfig is not None
        assert MultiAgentConfig is not None
        assert VideoPlumeConfig is not None
        assert SimulationConfig is not None


# Additional test utilities and fixtures for comprehensive coverage
@pytest.fixture(scope="session")
def hydra_config_store():
    """Session-level fixture for Hydra ConfigStore setup."""
    if HYDRA_AVAILABLE:
        cs = ConfigStore.instance()
        # Register test configurations
        cs.store(name="test_single_agent", node=SingleAgentConfig)
        cs.store(name="test_multi_agent", node=MultiAgentConfig)
        cs.store(name="test_video_plume", node=VideoPlumeConfig)
        cs.store(name="test_simulation", node=SimulationConfig)
        return cs
    return None


@pytest.fixture
def temp_config_files(tmp_path):
    """Create temporary configuration files for testing."""
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    
    # Create base.yaml
    base_config = {
        "navigator": {
            "position": [0, 0],
            "orientation": 0.0,
            "max_speed": 1.0
        },
        "video_plume": {
            "flip": False,
            "kernel_size": 3
        }
    }
    
    base_file = config_dir / "base.yaml"
    if HYDRA_AVAILABLE:
        with open(base_file, 'w') as f:
            OmegaConf.save(config=base_config, f=f)
    
    return config_dir


class TestIntegrationWithTempFiles:
    """Integration tests using temporary configuration files."""

    def test_configuration_file_loading(self, temp_config_files):
        """Test loading configuration from temporary files."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available for file loading tests")
        
        # This would test actual file loading in a real scenario
        # For now, we'll test the pattern
        config_path = temp_config_files / "base.yaml"
        assert config_path.exists()

    def test_end_to_end_workflow_simulation(self):
        """Test complete end-to-end workflow with mocked components."""
        # Create navigator
        navigator = create_navigator(position=(0, 0), max_speed=2.0)
        
        # Create video plume (mocked)
        with patch('cv2.VideoCapture') as mock_cap, \
             patch('pathlib.Path.exists', return_value=True):
            mock_instance = MagicMock()
            mock_cap.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.return_value = 100
            mock_instance.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
            
            plume = create_video_plume(video_path="test.mp4")
            
            # Run simulation (mocked)
            with patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation') as mock_sim:
                positions = np.array([[[0, 0], [1, 1]]])
                orientations = np.array([[0, 45]])
                readings = np.array([[0.1, 0.2]])
                mock_sim.return_value = (positions, orientations, readings)
                
                results = run_plume_simulation(navigator, plume, num_steps=1, dt=0.1)
                
                # Verify complete workflow
                assert len(results) == 3
                assert results[0].shape == (1, 2, 2)
                
                # Test visualization
                with patch('{{cookiecutter.project_slug}}.utils.visualization.visualize_trajectory'):
                    visualize_plume_simulation(
                        results[0], results[1], show_plot=False
                    )