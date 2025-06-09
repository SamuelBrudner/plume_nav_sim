"""
Comprehensive test suite for public API surface validation.

This module ensures API contract stability, parameter validation, error handling,
and integration patterns meet specification requirements through systematic testing
of all public interfaces including create_navigator, create_video_plume, 
run_plume_simulation, and visualization functions.

Tests validate:
- API contract stability for external consumers per Section 0.2.1
- Factory method patterns for Kedro integration per enhanced integration testing
- Performance thresholds per Section 6.6.3.3
- Backward compatibility for migration scenarios per technical specifications
- Comprehensive error handling for invalid inputs per API reliability requirements
"""

import pytest
import numpy as np
import pathlib
import tempfile
import time
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Test framework imports
from pytest import approx, raises, warns

# Core API imports under test
from odor_plume_nav.api.navigation import (
    create_navigator,
    create_video_plume,
    run_plume_simulation,
    create_navigator_from_config,
    create_video_plume_from_config,
    ConfigurationError,
    SimulationError,
    _validate_and_merge_config,
    _normalize_positions
)

# Supporting imports for testing
from odor_plume_nav.core.navigator import NavigatorProtocol, Navigator
from odor_plume_nav.data.video_plume import VideoPlume
from odor_plume_nav.config.schemas import (
    NavigatorConfig,
    VideoPlumeConfig,
    SimulationConfig
)

# Mock Hydra imports for configuration testing
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None


class TestCreateNavigator:
    """Test suite for create_navigator API function with comprehensive validation."""

    def test_create_navigator_single_agent_direct_parameters(self):
        """Test create_navigator with direct parameters for single agent mode."""
        navigator = create_navigator(
            position=(10.0, 20.0),
            orientation=45.0,
            speed=2.0,
            max_speed=5.0,
            angular_velocity=0.1
        )
        
        # Validate navigator instance
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.num_agents == 1
        
        # Validate position and orientation
        np.testing.assert_array_equal(navigator.positions, np.array([[10.0, 20.0]]))
        np.testing.assert_array_equal(navigator.orientations, np.array([45.0]))
        np.testing.assert_array_equal(navigator.speeds, np.array([2.0]))
        np.testing.assert_array_equal(navigator.max_speeds, np.array([5.0]))
        np.testing.assert_array_equal(navigator.angular_velocities, np.array([0.1]))

    def test_create_navigator_multi_agent_positions_array(self):
        """Test create_navigator with positions array for multi-agent mode."""
        positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        orientations = [0.0, 90.0, 180.0]
        max_speeds = [5.0, 6.0, 7.0]
        
        navigator = create_navigator(
            positions=positions,
            orientations=orientations,
            max_speeds=max_speeds
        )
        
        # Validate multi-agent navigator
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.num_agents == 3
        np.testing.assert_array_equal(navigator.positions, positions)
        np.testing.assert_array_equal(navigator.orientations, np.array(orientations))
        np.testing.assert_array_equal(navigator.max_speeds, np.array(max_speeds))

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_create_navigator_with_hydra_config(self):
        """Test create_navigator with Hydra DictConfig object."""
        config = DictConfig({
            'position': [15.0, 25.0],
            'orientation': 90.0,
            'speed': 3.0,
            'max_speed': 8.0,
            'angular_velocity': 0.2
        })
        
        navigator = create_navigator(cfg=config)
        
        # Validate configuration was applied
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.num_agents == 1
        np.testing.assert_array_equal(navigator.positions, np.array([[15.0, 25.0]]))
        np.testing.assert_array_equal(navigator.orientations, np.array([90.0]))

    def test_create_navigator_config_override_with_direct_params(self):
        """Test create_navigator configuration override with direct parameters."""
        config = {'max_speed': 5.0, 'speed': 2.0}
        
        navigator = create_navigator(
            cfg=config,
            max_speed=10.0,  # Override config value
            position=(5.0, 5.0),
            orientation=45.0
        )
        
        # Verify override behavior
        assert isinstance(navigator, NavigatorProtocol)
        np.testing.assert_array_equal(navigator.max_speeds, np.array([10.0]))  # Override value
        np.testing.assert_array_equal(navigator.speeds, np.array([2.0]))  # Config value
        np.testing.assert_array_equal(navigator.positions, np.array([[5.0, 5.0]]))

    def test_create_navigator_position_normalization(self):
        """Test position parameter normalization in create_navigator."""
        # Test single position as tuple
        navigator1 = create_navigator(position=(10.0, 20.0))
        np.testing.assert_array_equal(navigator1.positions, np.array([[10.0, 20.0]]))
        
        # Test single position as list
        navigator2 = create_navigator(position=[15.0, 25.0])
        np.testing.assert_array_equal(navigator2.positions, np.array([[15.0, 25.0]]))
        
        # Test single position as numpy array
        pos_array = np.array([5.0, 10.0])
        navigator3 = create_navigator(position=pos_array)
        np.testing.assert_array_equal(navigator3.positions, np.array([[5.0, 10.0]]))

    def test_create_navigator_seed_management(self):
        """Test create_navigator seed management for reproducibility."""
        seed_value = 42
        
        # Create two navigators with same seed
        navigator1 = create_navigator(
            position=(0.0, 0.0),
            seed=seed_value
        )
        navigator2 = create_navigator(
            position=(0.0, 0.0),
            seed=seed_value
        )
        
        # Verify deterministic behavior (positions should be identical)
        np.testing.assert_array_equal(navigator1.positions, navigator2.positions)

    def test_create_navigator_invalid_position_parameters(self):
        """Test create_navigator error handling for invalid position parameters."""
        # Test conflicting position and positions parameters
        with pytest.raises(ConfigurationError, match="Cannot specify both 'position'.*and 'positions'"):
            create_navigator(
                position=(10.0, 20.0),
                positions=[[0.0, 0.0], [10.0, 10.0]]
            )

    def test_create_navigator_invalid_position_format(self):
        """Test create_navigator error handling for invalid position formats."""
        # Test invalid position dimension
        with pytest.raises(ConfigurationError, match="Position must be a 2D coordinate"):
            create_navigator(position=[10.0])  # Only 1D
        
        with pytest.raises(ConfigurationError, match="Position must be a 2D coordinate"):
            create_navigator(position=[10.0, 20.0, 30.0])  # 3D position

    def test_create_navigator_invalid_positions_format(self):
        """Test create_navigator error handling for invalid positions formats."""
        # Test invalid positions structure
        with pytest.raises(ConfigurationError, match="Invalid positions format"):
            create_navigator(positions=[[10.0], [20.0]])  # Invalid 1D positions

    def test_create_navigator_performance_timing(self):
        """Test create_navigator meets performance requirements per Section 6.6.3.3."""
        start_time = time.time()
        
        # Create navigator with typical parameters
        navigator = create_navigator(
            position=(10.0, 20.0),
            max_speed=5.0,
            orientation=45.0
        )
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Verify performance threshold: API initialization ≤10ms per specification
        assert creation_time < 0.01, f"Navigator creation took {creation_time:.3f}s, exceeds 10ms limit"
        assert isinstance(navigator, NavigatorProtocol)

    def test_create_navigator_large_multi_agent_performance(self):
        """Test create_navigator performance with large multi-agent scenarios."""
        # Create 100-agent scenario per performance specification
        num_agents = 100
        positions = [[i * 2.0, i * 3.0] for i in range(num_agents)]
        
        start_time = time.time()
        navigator = create_navigator(positions=positions)
        end_time = time.time()
        
        creation_time = end_time - start_time
        
        # Verify performance threshold for large-scale scenarios
        assert creation_time < 0.05, f"Large navigator creation took {creation_time:.3f}s, exceeds 50ms limit"
        assert navigator.num_agents == num_agents

    def test_create_navigator_configuration_validation_integration(self):
        """Test create_navigator Pydantic configuration validation integration with odor_plume_nav package."""
        # Test with invalid speed configuration (speed > max_speed)
        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            create_navigator(
                position=(0.0, 0.0),
                speed=10.0,
                max_speed=5.0  # Invalid: speed exceeds max_speed
            )


class TestCreateVideoPlume:
    """Test suite for create_video_plume API function with validation."""

    def test_create_video_plume_basic_creation(self, tmp_path):
        """Test create_video_plume basic functionality with mock video file."""
        # Create temporary video file for testing
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"mock_video_content")
        
        with patch('odor_plume_nav.data.video_plume.VideoPlume') as MockVideoPlume:
            mock_instance = Mock()
            MockVideoPlume.return_value = mock_instance
            
            plume = create_video_plume(
                video_path=str(video_file),
                flip=True,
                kernel_size=5,
                kernel_sigma=2.0
            )
            
            # Verify VideoPlume was created with correct parameters
            MockVideoPlume.assert_called_once_with(
                video_path=pathlib.Path(str(video_file)),
                flip=True,
                kernel_size=5,
                kernel_sigma=2.0
            )
            assert plume == mock_instance

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_create_video_plume_with_hydra_config(self, tmp_path):
        """Test create_video_plume with Hydra configuration object."""
        video_file = tmp_path / "config_video.mp4"
        video_file.write_bytes(b"mock_video_content")
        
        config = DictConfig({
            'video_path': str(video_file),
            'flip': False,
            'kernel_size': 3,
            'kernel_sigma': 1.5
        })
        
        with patch('odor_plume_nav.data.video_plume.VideoPlume') as MockVideoPlume:
            mock_instance = Mock()
            MockVideoPlume.return_value = mock_instance
            
            plume = create_video_plume(cfg=config)
            
            # Verify configuration was applied correctly
            MockVideoPlume.assert_called_once_with(
                video_path=pathlib.Path(str(video_file)),
                flip=False,
                kernel_size=3,
                kernel_sigma=1.5
            )

    def test_create_video_plume_config_override(self, tmp_path):
        """Test create_video_plume parameter override functionality."""
        video_file = tmp_path / "override_video.mp4"
        video_file.write_bytes(b"mock_video_content")
        
        config = {
            'video_path': str(video_file),
            'flip': False,
            'kernel_size': 3
        }
        
        with patch('odor_plume_nav.data.video_plume.VideoPlume') as MockVideoPlume:
            mock_instance = Mock()
            MockVideoPlume.return_value = mock_instance
            
            plume = create_video_plume(
                cfg=config,
                flip=True,  # Override config value
                kernel_sigma=2.5  # Additional parameter
            )
            
            # Verify override behavior
            MockVideoPlume.assert_called_once_with(
                video_path=pathlib.Path(str(video_file)),
                flip=True,  # Override value
                kernel_size=3,  # Config value
                kernel_sigma=2.5  # Additional parameter
            )

    def test_create_video_plume_missing_video_path(self):
        """Test create_video_plume error handling for missing video_path."""
        with pytest.raises(ConfigurationError, match="video_path is required"):
            create_video_plume(flip=True, kernel_size=5)

    def test_create_video_plume_nonexistent_file(self):
        """Test create_video_plume error handling for nonexistent video files."""
        nonexistent_path = "/nonexistent/path/video.mp4"
        
        with pytest.raises(FileNotFoundError, match="Video file does not exist"):
            create_video_plume(video_path=nonexistent_path)

    def test_create_video_plume_invalid_flip_parameter(self, tmp_path):
        """Test create_video_plume validation for invalid flip parameter."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"mock_video_content")
        
        with pytest.raises(ConfigurationError, match="flip must be a boolean"):
            create_video_plume(
                video_path=str(video_file),
                flip="invalid_string"  # Invalid type
            )

    def test_create_video_plume_invalid_kernel_parameters(self, tmp_path):
        """Test create_video_plume validation for invalid kernel parameters."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"mock_video_content")
        
        # Test invalid kernel_size
        with pytest.raises(ConfigurationError, match="kernel_size must be a positive integer"):
            create_video_plume(
                video_path=str(video_file),
                kernel_size=-5  # Negative value
            )
        
        # Test invalid kernel_sigma
        with pytest.raises(ConfigurationError, match="kernel_sigma must be a positive number"):
            create_video_plume(
                video_path=str(video_file),
                kernel_sigma=0.0  # Zero value
            )

    def test_create_video_plume_path_types(self, tmp_path):
        """Test create_video_plume accepts different path types."""
        video_file = tmp_path / "path_test.mp4"
        video_file.write_bytes(b"mock_video_content")
        
        with patch('odor_plume_nav.data.video_plume.VideoPlume') as MockVideoPlume:
            mock_instance = Mock()
            MockVideoPlume.return_value = mock_instance
            
            # Test string path
            plume1 = create_video_plume(video_path=str(video_file))
            
            # Test pathlib.Path
            plume2 = create_video_plume(video_path=video_file)
            
            # Verify both calls succeed and convert to pathlib.Path
            assert MockVideoPlume.call_count == 2
            for call_args in MockVideoPlume.call_args_list:
                assert isinstance(call_args[1]['video_path'], pathlib.Path)

    def test_create_video_plume_performance_timing(self, tmp_path):
        """Test create_video_plume meets performance requirements."""
        video_file = tmp_path / "performance_test.mp4"
        video_file.write_bytes(b"mock_video_content")
        
        with patch('odor_plume_nav.data.video_plume.VideoPlume') as MockVideoPlume:
            mock_instance = Mock()
            MockVideoPlume.return_value = mock_instance
            
            start_time = time.time()
            plume = create_video_plume(video_path=str(video_file))
            end_time = time.time()
            
            creation_time = end_time - start_time
            
            # Verify performance threshold: VideoPlume creation ≤100ms per specification
            assert creation_time < 0.1, f"VideoPlume creation took {creation_time:.3f}s, exceeds 100ms limit"


class TestRunPlumeSimulation:
    """Test suite for run_plume_simulation API function with comprehensive validation."""

    def setup_method(self):
        """Set up mock objects for simulation testing."""
        # Create mock navigator
        self.mock_navigator = Mock(spec=NavigatorProtocol)
        self.mock_navigator.num_agents = 2
        self.mock_navigator.positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        self.mock_navigator.orientations = np.array([0.0, 90.0])
        
        # Create mock video plume
        self.mock_video_plume = Mock(spec=VideoPlume)
        self.mock_video_plume.frame_count = 100
        
        # Mock frame data
        self.mock_frame = np.random.rand(100, 100).astype(np.float32)
        self.mock_video_plume.get_frame.return_value = self.mock_frame
        
        # Mock navigator methods
        self.mock_navigator.sample_odor.return_value = np.array([0.5, 0.3])

    def test_run_plume_simulation_basic_execution(self):
        """Test run_plume_simulation basic functionality."""
        num_steps = 10
        dt = 0.1
        
        positions, orientations, odor_readings = run_plume_simulation(
            self.mock_navigator,
            self.mock_video_plume,
            num_steps=num_steps,
            dt=dt
        )
        
        # Validate output shapes
        expected_shape = (2, num_steps + 1)  # 2 agents, 11 time steps (including initial)
        assert positions.shape == (2, num_steps + 1, 2)
        assert orientations.shape == expected_shape
        assert odor_readings.shape == expected_shape
        
        # Verify navigator step was called correctly
        assert self.mock_navigator.step.call_count == num_steps
        
        # Verify video frame access
        assert self.mock_video_plume.get_frame.call_count == num_steps + 1  # Initial + steps

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_run_plume_simulation_with_hydra_config(self):
        """Test run_plume_simulation with Hydra configuration object."""
        config = DictConfig({
            'num_steps': 20,
            'dt': 0.05,
            'sensor_distance': 3.0,
            'sensor_angle': 30.0,
            'record_trajectory': True
        })
        
        positions, orientations, odor_readings = run_plume_simulation(
            self.mock_navigator,
            self.mock_video_plume,
            cfg=config
        )
        
        # Validate configuration was applied
        expected_shape = (2, 21)  # 2 agents, 21 time steps
        assert positions.shape == (2, 21, 2)
        assert orientations.shape == expected_shape
        assert odor_readings.shape == expected_shape
        
        # Verify correct number of steps
        assert self.mock_navigator.step.call_count == 20

    def test_run_plume_simulation_config_override(self):
        """Test run_plume_simulation parameter override functionality."""
        config = {'num_steps': 15, 'dt': 0.2}
        
        positions, orientations, odor_readings = run_plume_simulation(
            self.mock_navigator,
            self.mock_video_plume,
            cfg=config,
            num_steps=25,  # Override config value
            sensor_distance=4.0  # Additional parameter
        )
        
        # Verify override behavior - should use 25 steps, not 15
        expected_shape = (2, 26)  # 2 agents, 26 time steps
        assert positions.shape == (2, 26, 2)
        assert self.mock_navigator.step.call_count == 25

    def test_run_plume_simulation_seed_management(self):
        """Test run_plume_simulation seed management for reproducibility."""
        seed_value = 123
        
        # Run simulation twice with same seed
        with patch('odor_plume_nav.api.navigation.set_global_seed') as mock_set_seed:
            positions1, _, _ = run_plume_simulation(
                self.mock_navigator,
                self.mock_video_plume,
                num_steps=5,
                seed=seed_value
            )
            
            # Verify seed was set
            mock_set_seed.assert_called_with(seed_value)

    def test_run_plume_simulation_backward_compatibility_step_size(self):
        """Test run_plume_simulation backward compatibility for step_size parameter."""
        # Test deprecated step_size parameter
        with pytest.warns(DeprecationWarning, match="Parameter 'step_size' is deprecated"):
            positions, orientations, odor_readings = run_plume_simulation(
                self.mock_navigator,
                self.mock_video_plume,
                num_steps=5,
                step_size=0.15  # Deprecated parameter
            )
        
        # Verify simulation still works
        assert positions.shape == (2, 6, 2)
        assert self.mock_navigator.step.call_count == 5

    def test_run_plume_simulation_frame_limit_handling(self):
        """Test run_plume_simulation handling of video frame limits."""
        # Set video to have fewer frames than requested steps
        self.mock_video_plume.frame_count = 5
        
        with pytest.warns(UserWarning, match="Requested 10 steps, but only 4 frames available"):
            positions, orientations, odor_readings = run_plume_simulation(
                self.mock_navigator,
                self.mock_video_plume,
                num_steps=10  # More than available frames
            )
        
        # Should only run 4 steps (frame_count - 1)
        assert positions.shape == (2, 5, 2)  # 4 steps + initial
        assert self.mock_navigator.step.call_count == 4

    def test_run_plume_simulation_minimal_trajectory_recording(self):
        """Test run_plume_simulation with minimal trajectory recording."""
        positions, orientations, odor_readings = run_plume_simulation(
            self.mock_navigator,
            self.mock_video_plume,
            num_steps=10,
            record_trajectory=False
        )
        
        # Should only record initial and final states
        assert positions.shape == (2, 2, 2)
        assert orientations.shape == (2, 2)
        assert odor_readings.shape == (2, 2)

    def test_run_plume_simulation_error_handling_invalid_navigator(self):
        """Test run_plume_simulation error handling for invalid navigator."""
        invalid_navigator = Mock()  # Missing required attributes
        
        with pytest.raises(SimulationError, match="Navigator must have 'positions' attribute"):
            run_plume_simulation(
                invalid_navigator,
                self.mock_video_plume,
                num_steps=5
            )

    def test_run_plume_simulation_error_handling_invalid_video_plume(self):
        """Test run_plume_simulation error handling for invalid video plume."""
        invalid_plume = Mock()  # Missing required methods
        
        with pytest.raises(SimulationError, match="VideoPlume must have 'get_frame' method"):
            run_plume_simulation(
                self.mock_navigator,
                invalid_plume,
                num_steps=5
            )

    def test_run_plume_simulation_parameter_validation(self):
        """Test run_plume_simulation parameter validation."""
        # Test invalid num_steps
        with pytest.raises(ConfigurationError, match="num_steps must be a positive integer"):
            run_plume_simulation(
                self.mock_navigator,
                self.mock_video_plume,
                num_steps=-5
            )
        
        # Test invalid dt
        with pytest.raises(ConfigurationError, match="dt must be a positive number"):
            run_plume_simulation(
                self.mock_navigator,
                self.mock_video_plume,
                num_steps=5,
                dt=0.0
            )
        
        # Test invalid sensor_distance
        with pytest.raises(ConfigurationError, match="sensor_distance must be non-negative"):
            run_plume_simulation(
                self.mock_navigator,
                self.mock_video_plume,
                num_steps=5,
                sensor_distance=-1.0
            )

    def test_run_plume_simulation_frame_error_handling(self):
        """Test run_plume_simulation handling of frame retrieval errors."""
        # Mock frame retrieval failure
        self.mock_video_plume.get_frame.side_effect = [
            self.mock_frame,  # Initial frame succeeds
            self.mock_frame,  # First step succeeds
            None,             # Second step fails
        ]
        
        with pytest.warns(UserWarning, match="Failed to get frame 2, ending simulation early"):
            positions, orientations, odor_readings = run_plume_simulation(
                self.mock_navigator,
                self.mock_video_plume,
                num_steps=5
            )
        
        # Should only complete 1 step before failure
        assert self.mock_navigator.step.call_count == 1

    def test_run_plume_simulation_odor_sampling_error_handling(self):
        """Test run_plume_simulation handling of odor sampling errors."""
        # Mock odor sampling failure
        self.mock_navigator.sample_odor.side_effect = [
            np.array([0.5, 0.3]),  # Initial sampling succeeds
            RuntimeError("Sampling failed"),  # Subsequent sampling fails
            np.array([0.4, 0.2])   # Recovery
        ]
        
        with pytest.warns(UserWarning, match="Failed to sample odor at step 0"):
            positions, orientations, odor_readings = run_plume_simulation(
                self.mock_navigator,
                self.mock_video_plume,
                num_steps=3
            )
        
        # Simulation should continue despite odor sampling errors
        assert self.mock_navigator.step.call_count == 3
        
        # Failed sampling should result in zero readings
        assert odor_readings[0, 1] == 0.0  # Error step should be zero

    def test_run_plume_simulation_performance_timing(self):
        """Test run_plume_simulation meets performance requirements."""
        # Test single agent performance threshold: ≤1ms per step
        single_navigator = Mock(spec=NavigatorProtocol)
        single_navigator.num_agents = 1
        single_navigator.positions = np.array([[0.0, 0.0]])
        single_navigator.orientations = np.array([0.0])
        single_navigator.sample_odor.return_value = np.array([0.5])
        
        start_time = time.time()
        run_plume_simulation(
            single_navigator,
            self.mock_video_plume,
            num_steps=1,
            record_trajectory=False
        )
        end_time = time.time()
        
        step_time = end_time - start_time
        
        # Verify performance threshold for single agent
        assert step_time < 0.001, f"Single agent step took {step_time:.4f}s, exceeds 1ms limit"

    def test_run_plume_simulation_multi_agent_performance(self):
        """Test run_plume_simulation multi-agent performance requirements."""
        # Test 10-agent performance threshold: ≤5ms per step
        multi_navigator = Mock(spec=NavigatorProtocol)
        multi_navigator.num_agents = 10
        multi_navigator.positions = np.array([[i, i] for i in range(10)])
        multi_navigator.orientations = np.array([i * 36.0 for i in range(10)])
        multi_navigator.sample_odor.return_value = np.array([0.5] * 10)
        
        start_time = time.time()
        run_plume_simulation(
            multi_navigator,
            self.mock_video_plume,
            num_steps=1,
            record_trajectory=False
        )
        end_time = time.time()
        
        step_time = end_time - start_time
        
        # Verify performance threshold for 10 agents
        assert step_time < 0.005, f"10-agent step took {step_time:.4f}s, exceeds 5ms limit"

    def test_run_plume_simulation_output_shape_validation(self):
        """Test run_plume_simulation output shape validation."""
        num_steps = 15
        num_agents = 3
        
        # Set up 3-agent navigator
        self.mock_navigator.num_agents = num_agents
        self.mock_navigator.positions = np.array([[i, i] for i in range(num_agents)])
        self.mock_navigator.orientations = np.array([i * 30.0 for i in range(num_agents)])
        self.mock_navigator.sample_odor.return_value = np.array([0.5] * num_agents)
        
        positions, orientations, odor_readings = run_plume_simulation(
            self.mock_navigator,
            self.mock_video_plume,
            num_steps=num_steps
        )
        
        # Validate exact output shapes
        expected_time_steps = num_steps + 1
        assert positions.shape == (num_agents, expected_time_steps, 2)
        assert orientations.shape == (num_agents, expected_time_steps)
        assert odor_readings.shape == (num_agents, expected_time_steps)
        
        # Validate data types
        assert positions.dtype == np.float64
        assert orientations.dtype == np.float64
        assert odor_readings.dtype == np.float64


class TestBackwardCompatibilityFactoryMethods:
    """Test suite for backward compatibility factory methods with odor_plume_nav package."""

    def test_create_navigator_from_config_basic_functionality(self):
        """Test create_navigator_from_config backward compatibility."""
        config = {
            'position': [10.0, 20.0],
            'max_speed': 5.0,
            'orientation': 45.0
        }
        
        navigator = create_navigator_from_config(config)
        
        # Should delegate to create_navigator
        assert isinstance(navigator, NavigatorProtocol)

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_create_navigator_from_config_with_dictconfig(self):
        """Test create_navigator_from_config with DictConfig object."""
        config = DictConfig({
            'position': [15.0, 25.0],
            'max_speed': 8.0
        })
        
        navigator = create_navigator_from_config(config, speed=2.0)
        
        # Verify configuration and overrides work
        assert isinstance(navigator, NavigatorProtocol)

    def test_create_navigator_from_config_path_rejection(self):
        """Test create_navigator_from_config rejects file paths."""
        with pytest.raises(ConfigurationError, match="File path configuration loading is deprecated"):
            create_navigator_from_config("/path/to/config.yaml")

    def test_create_video_plume_from_config_basic_functionality(self, tmp_path):
        """Test create_video_plume_from_config backward compatibility."""
        video_file = tmp_path / "compat_test.mp4"
        video_file.write_bytes(b"mock_video_content")
        
        config = {
            'video_path': str(video_file),
            'flip': True
        }
        
        with patch('odor_plume_nav.data.video_plume.VideoPlume') as MockVideoPlume:
            mock_instance = Mock()
            MockVideoPlume.return_value = mock_instance
            
            plume = create_video_plume_from_config(config)
            
            # Should delegate to create_video_plume
            MockVideoPlume.assert_called_once()

    def test_create_video_plume_from_config_path_rejection(self):
        """Test create_video_plume_from_config rejects file paths."""
        with pytest.raises(ConfigurationError, match="File path configuration loading is deprecated"):
            create_video_plume_from_config("/path/to/config.yaml")


class TestConfigurationValidationHelpers:
    """Test suite for internal configuration validation helper functions."""

    def test_validate_and_merge_config_basic_functionality(self):
        """Test _validate_and_merge_config basic functionality."""
        config = {'param1': 'value1', 'param2': 2}
        kwargs = {'param2': 3, 'param3': 'value3'}
        
        result = _validate_and_merge_config(cfg=config, **kwargs)
        
        # Verify merging behavior
        assert result['param1'] == 'value1'  # From config
        assert result['param2'] == 3         # Overridden by kwargs
        assert result['param3'] == 'value3'  # From kwargs

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_validate_and_merge_config_with_dictconfig(self):
        """Test _validate_and_merge_config with DictConfig object."""
        config = DictConfig({'param1': 'hydra_value', 'param2': 42})
        
        result = _validate_and_merge_config(cfg=config, param3='extra_value')
        
        # Verify DictConfig processing
        assert result['param1'] == 'hydra_value'
        assert result['param2'] == 42
        assert result['param3'] == 'extra_value'

    def test_validate_and_merge_config_none_values_ignored(self):
        """Test _validate_and_merge_config ignores None values in kwargs."""
        config = {'param1': 'value1'}
        
        result = _validate_and_merge_config(
            cfg=config,
            param1=None,      # Should not override
            param2='value2'   # Should be added
        )
        
        assert result['param1'] == 'value1'  # Not overridden by None
        assert result['param2'] == 'value2'

    def test_validate_and_merge_config_validation_schema(self):
        """Test _validate_and_merge_config with Pydantic validation schema."""
        from odor_plume_nav.config.schemas import NavigatorConfig
        
        config = {
            'position': [10.0, 20.0],
            'max_speed': 5.0,
            'speed': 2.0
        }
        
        result = _validate_and_merge_config(
            cfg=config,
            config_schema=NavigatorConfig
        )
        
        # Should validate and normalize through Pydantic
        assert 'position' in result
        assert result['max_speed'] == 5.0

    def test_validate_and_merge_config_validation_error(self):
        """Test _validate_and_merge_config validation error handling."""
        from odor_plume_nav.config.schemas import NavigatorConfig
        
        invalid_config = {
            'position': [10.0],  # Invalid: only 1D
            'max_speed': -5.0    # Invalid: negative
        }
        
        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            _validate_and_merge_config(
                cfg=invalid_config,
                config_schema=NavigatorConfig
            )

    def test_normalize_positions_single_position_formats(self):
        """Test _normalize_positions with various single position formats."""
        # Test tuple format
        result, is_multi = _normalize_positions(position=(10.0, 20.0))
        np.testing.assert_array_equal(result, np.array([[10.0, 20.0]]))
        assert is_multi is False
        
        # Test list format
        result, is_multi = _normalize_positions(position=[15.0, 25.0])
        np.testing.assert_array_equal(result, np.array([[15.0, 25.0]]))
        assert is_multi is False
        
        # Test numpy array format
        pos_array = np.array([5.0, 10.0])
        result, is_multi = _normalize_positions(position=pos_array)
        np.testing.assert_array_equal(result, np.array([[5.0, 10.0]]))
        assert is_multi is False

    def test_normalize_positions_multi_positions_formats(self):
        """Test _normalize_positions with multi-agent positions."""
        # Test list of lists
        positions = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
        result, is_multi = _normalize_positions(positions=positions)
        np.testing.assert_array_equal(result, np.array(positions))
        assert is_multi is True
        
        # Test numpy array
        pos_array = np.array([[5.0, 15.0], [25.0, 35.0]])
        result, is_multi = _normalize_positions(positions=pos_array)
        np.testing.assert_array_equal(result, pos_array)
        assert is_multi is True

    def test_normalize_positions_single_as_positions(self):
        """Test _normalize_positions with single position as positions parameter."""
        # Single position provided as positions should be treated as single agent
        result, is_multi = _normalize_positions(positions=[10.0, 20.0])
        np.testing.assert_array_equal(result, np.array([[10.0, 20.0]]))
        assert is_multi is False

    def test_normalize_positions_conflicting_parameters(self):
        """Test _normalize_positions error handling for conflicting parameters."""
        with pytest.raises(ConfigurationError, match="Cannot specify both 'position'.*and 'positions'"):
            _normalize_positions(
                position=(10.0, 20.0),
                positions=[[0.0, 0.0], [10.0, 10.0]]
            )

    def test_normalize_positions_invalid_formats(self):
        """Test _normalize_positions error handling for invalid formats."""
        # Invalid position dimension
        with pytest.raises(ConfigurationError, match="Position must be a 2D coordinate"):
            _normalize_positions(position=[10.0])
        
        # Invalid positions structure
        with pytest.raises(ConfigurationError, match="Invalid positions format"):
            _normalize_positions(positions=[[10.0], [20.0]])

    def test_normalize_positions_none_inputs(self):
        """Test _normalize_positions with None inputs."""
        result, is_multi = _normalize_positions(position=None, positions=None)
        assert result is None
        assert is_multi is False


class TestAPIContractStability:
    """Test suite for API contract stability and interface compliance."""

    def test_api_function_signatures_stability(self):
        """Test API function signatures remain stable for external consumers."""
        import inspect
        
        # Test create_navigator signature
        sig = inspect.signature(create_navigator)
        expected_params = [
            'cfg', 'positions', 'orientations', 'speeds', 'max_speeds',
            'angular_velocities', 'position', 'orientation', 'speed',
            'max_speed', 'angular_velocity', 'seed', 'kwargs'
        ]
        
        actual_params = list(sig.parameters.keys())
        for param in expected_params:
            assert param in actual_params, f"Missing parameter: {param}"

    def test_api_return_types_stability(self):
        """Test API return types conform to NavigatorProtocol."""
        navigator = create_navigator(position=(0.0, 0.0))
        
        # Verify return type implements NavigatorProtocol
        assert isinstance(navigator, NavigatorProtocol)
        
        # Verify required protocol methods exist
        assert hasattr(navigator, 'positions')
        assert hasattr(navigator, 'orientations')
        assert hasattr(navigator, 'speeds')
        assert hasattr(navigator, 'max_speeds')
        assert hasattr(navigator, 'angular_velocities')
        assert hasattr(navigator, 'num_agents')
        assert hasattr(navigator, 'reset')
        assert hasattr(navigator, 'step')
        assert hasattr(navigator, 'sample_odor')

    def test_api_exception_types_stability(self):
        """Test API exception types remain consistent."""
        # Test ConfigurationError is raised for configuration issues
        with pytest.raises(ConfigurationError):
            create_navigator(position=(10.0, 20.0), positions=[[0.0, 0.0]])
        
        # Test FileNotFoundError for missing files
        with pytest.raises(FileNotFoundError):
            create_video_plume(video_path="/nonexistent/file.mp4")

    def test_api_import_paths_stability(self):
        """Test API import paths remain stable for external consumers."""
        # Test core API imports work with odor_plume_nav package
        from odor_plume_nav.api.navigation import (
            create_navigator,
            create_video_plume,
            run_plume_simulation
        )
        
        # Test these are callable
        assert callable(create_navigator)
        assert callable(create_video_plume)
        assert callable(run_plume_simulation)

    def test_api_kedro_integration_pattern(self):
        """Test API factory method patterns support Kedro integration with odor_plume_nav package."""
        # Test configuration-first pattern expected by Kedro
        config = {'position': [10.0, 20.0], 'max_speed': 5.0}
        
        navigator = create_navigator_from_config(config)
        assert isinstance(navigator, NavigatorProtocol)
        
        # Test parameter override pattern
        navigator_override = create_navigator_from_config(config, max_speed=10.0)
        assert isinstance(navigator_override, NavigatorProtocol)

    def test_api_numpy_interface_compatibility(self):
        """Test API maintains NumPy array interface compatibility."""
        navigator = create_navigator(
            positions=[[0.0, 0.0], [10.0, 10.0]]
        )
        
        # Verify NumPy array properties
        assert isinstance(navigator.positions, np.ndarray)
        assert isinstance(navigator.orientations, np.ndarray)
        assert isinstance(navigator.speeds, np.ndarray)
        
        # Verify array shapes and dtypes
        assert navigator.positions.shape == (2, 2)
        assert navigator.positions.dtype in [np.float32, np.float64]

    def test_api_performance_contract_compliance(self):
        """Test API meets performance contract requirements."""
        # Test initialization timing contract
        start_time = time.time()
        navigator = create_navigator(position=(0.0, 0.0))
        end_time = time.time()
        
        initialization_time = end_time - start_time
        assert initialization_time < 0.01, "API initialization exceeds 10ms contract"
        
        # Test factory method timing
        config = {'position': [5.0, 5.0]}
        start_time = time.time()
        navigator_config = create_navigator_from_config(config)
        end_time = time.time()
        
        factory_time = end_time - start_time
        assert factory_time < 0.01, "Factory method exceeds 10ms contract"


@pytest.mark.integration
class TestAPIIntegrationScenarios:
    """Integration test scenarios for comprehensive API validation with odor_plume_nav package."""

    def test_complete_simulation_workflow_integration(self, tmp_path):
        """Test complete simulation workflow through API."""
        # Create mock video file
        video_file = tmp_path / "integration_test.mp4"
        video_file.write_bytes(b"mock_video_content")
        
        with patch('odor_plume_nav.data.video_plume.VideoPlume') as MockVideoPlume:
            # Set up mock video plume
            mock_plume = Mock()
            mock_plume.frame_count = 10
            mock_plume.get_frame.return_value = np.random.rand(50, 50)
            MockVideoPlume.return_value = mock_plume
            
            # Create navigator through API
            navigator = create_navigator(
                positions=[[10.0, 10.0], [20.0, 20.0]],
                max_speeds=[5.0, 6.0]
            )
            
            # Create video plume through API
            video_plume = create_video_plume(
                video_path=str(video_file),
                flip=True
            )
            
            # Mock navigator behavior
            with patch.object(navigator, 'step') as mock_step, \
                 patch.object(navigator, 'sample_odor') as mock_sample:
                
                mock_sample.return_value = np.array([0.5, 0.3])
                
                # Execute simulation through API
                positions, orientations, odor_readings = run_plume_simulation(
                    navigator,
                    video_plume,
                    num_steps=5,
                    dt=0.1
                )
                
                # Validate complete workflow
                assert positions.shape == (2, 6, 2)
                assert orientations.shape == (2, 6)
                assert odor_readings.shape == (2, 6)
                assert mock_step.call_count == 5

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_configuration_composition_integration(self, tmp_path):
        """Test Hydra configuration composition across API functions with odor_plume_nav package."""
        video_file = tmp_path / "hydra_test.mp4"
        video_file.write_bytes(b"mock_video_content")
        
        # Create comprehensive Hydra configuration
        config = DictConfig({
            'navigator': {
                'position': [15.0, 25.0],
                'max_speed': 8.0,
                'orientation': 45.0
            },
            'video_plume': {
                'video_path': str(video_file),
                'flip': True,
                'kernel_size': 5,
                'kernel_sigma': 2.0
            },
            'simulation': {
                'num_steps': 10,
                'dt': 0.05,
                'record_trajectory': True
            }
        })
        
        with patch('odor_plume_nav.data.video_plume.VideoPlume') as MockVideoPlume:
            mock_plume = Mock()
            mock_plume.frame_count = 15
            mock_plume.get_frame.return_value = np.random.rand(40, 40)
            MockVideoPlume.return_value = mock_plume
            
            # Create components with Hydra configuration
            navigator = create_navigator(cfg=config.navigator)
            video_plume = create_video_plume(cfg=config.video_plume)
            
            # Mock navigator for simulation
            with patch.object(navigator, 'step') as mock_step, \
                 patch.object(navigator, 'sample_odor') as mock_sample:
                
                mock_sample.return_value = np.array([0.7])
                
                # Execute simulation with Hydra config
                positions, orientations, odor_readings = run_plume_simulation(
                    navigator,
                    video_plume,
                    cfg=config.simulation
                )
                
                # Validate Hydra configuration was applied
                assert positions.shape == (1, 11, 2)  # 10 steps + initial
                assert mock_step.call_count == 10

    def test_error_handling_integration_across_api(self):
        """Test comprehensive error handling across API functions."""
        # Test cascading error handling
        with pytest.raises(ConfigurationError, match="Cannot specify both"):
            navigator = create_navigator(
                position=(10.0, 20.0),
                positions=[[0.0, 0.0]]
            )
        
        # Test file not found error propagation
        with pytest.raises(FileNotFoundError):
            video_plume = create_video_plume(
                video_path="/definitely/nonexistent/path.mp4"
            )
        
        # Test simulation error with invalid navigator
        invalid_navigator = Mock()  # Missing required attributes
        mock_plume = Mock()
        
        with pytest.raises(SimulationError):
            run_plume_simulation(invalid_navigator, mock_plume, num_steps=1)

    def test_multi_agent_scaling_integration(self):
        """Test multi-agent scenario scaling through API."""
        # Test scaling to 50 agents
        num_agents = 50
        positions = [[i * 2.0, i * 3.0] for i in range(num_agents)]
        
        start_time = time.time()
        navigator = create_navigator(positions=positions)
        end_time = time.time()
        
        # Verify performance scaling
        creation_time = end_time - start_time
        assert creation_time < 0.1, f"50-agent creation took {creation_time:.3f}s"
        
        # Verify correct multi-agent setup
        assert navigator.num_agents == num_agents
        assert navigator.positions.shape == (num_agents, 2)

    def test_parameter_validation_integration(self):
        """Test parameter validation consistency across API functions."""
        # Test consistent validation patterns
        with pytest.raises(ConfigurationError):
            create_navigator(speed=10.0, max_speed=5.0)  # Invalid speed constraint
        
        with pytest.raises(ConfigurationError):
            create_video_plume(
                video_path="/dev/null",  # Assuming it exists but is not a video
                kernel_size=-3  # Invalid kernel size
            )
        
        # Test parameter override validation
        config = {'max_speed': 5.0}
        navigator = create_navigator(cfg=config, max_speed=10.0)  # Valid override
        np.testing.assert_array_equal(navigator.max_speeds, np.array([10.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])