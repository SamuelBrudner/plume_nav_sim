"""
Comprehensive test suite for the public API surface validation.

This module ensures API contract stability, parameter validation, error handling,
and integration patterns meet specification requirements through systematic testing
of all public interfaces.

Tests are organized by functional areas:
- Navigator creation API testing with configuration validation
- VideoPlume creation API testing with file and parameter validation  
- Simulation execution API testing with parameter override scenarios
- Visualization API testing with export capabilities
- Error handling validation for invalid configurations
- Factory method patterns for Kedro integration compatibility
- Backward compatibility validation for migration scenarios
- API performance requirements validation

Each test follows scientific computing best practices with deterministic behavior,
comprehensive error scenarios, and research-grade quality validation.
"""

import pytest
import numpy as np
import pathlib
import tempfile
import time
from typing import Dict, Any, Tuple, List
from unittest.mock import Mock, patch, MagicMock

# Import statements for the API under test
# Note: These will be available once the dependency files are created
try:
    from {{cookiecutter.project_slug}}.api.navigation import (
        create_navigator,
        create_video_plume,
        run_plume_simulation,
        visualize_simulation_results
    )
    from {{cookiecutter.project_slug}}.core.navigator import Navigator, NavigatorProtocol
    from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
    from {{cookiecutter.project_slug}}.config.schemas import (
        NavigatorConfig,
        VideoPlumeConfig,
        SimulationConfig
    )
except ImportError:
    # Graceful handling for test discovery before dependency files exist
    pass


class TestNavigatorCreationAPI:
    """
    Test suite for create_navigator API function validation.
    
    Validates API contract stability for external consumers per Section 0.2.1,
    tests factory method patterns for Kedro integration, and ensures robust
    parameter validation and error handling.
    """
    
    def test_create_navigator_single_agent_basic(self):
        """Test basic single agent navigator creation with minimal parameters."""
        navigator = create_navigator(
            position=(10.0, 20.0),
            orientation=45.0,
            speed=2.5,
            max_speed=5.0
        )
        
        assert isinstance(navigator, Navigator)
        assert navigator.is_single_agent == True
        assert navigator.num_agents == 1
        np.testing.assert_array_equal(navigator.positions, [[10.0, 20.0]])
        np.testing.assert_array_equal(navigator.orientations, [45.0])
        np.testing.assert_array_equal(navigator.speeds, [2.5])
        np.testing.assert_array_equal(navigator.max_speeds, [5.0])
    
    def test_create_navigator_multi_agent_basic(self):
        """Test basic multi-agent navigator creation with position arrays."""
        positions = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
        orientations = [0.0, 90.0, 180.0]
        speeds = [1.0, 2.0, 3.0]
        max_speeds = [5.0, 6.0, 7.0]
        
        navigator = create_navigator(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds
        )
        
        assert isinstance(navigator, Navigator)
        assert navigator.is_single_agent == False
        assert navigator.num_agents == 3
        np.testing.assert_array_equal(navigator.positions, positions)
        np.testing.assert_array_equal(navigator.orientations, orientations)
        np.testing.assert_array_equal(navigator.speeds, speeds)
        np.testing.assert_array_equal(navigator.max_speeds, max_speeds)
    
    def test_create_navigator_with_config_file(self):
        """Test navigator creation with configuration file parameter."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
            navigator:
              position: [15.0, 25.0]
              orientation: 90.0
              speed: 3.0
              max_speed: 8.0
              angular_velocity: 0.2
            """)
            config_path = f.name
        
        try:
            navigator = create_navigator(config_path=config_path)
            
            assert isinstance(navigator, Navigator)
            assert navigator.is_single_agent == True
            np.testing.assert_array_almost_equal(navigator.positions, [[15.0, 25.0]])
            np.testing.assert_array_almost_equal(navigator.orientations, [90.0])
            np.testing.assert_array_almost_equal(navigator.speeds, [3.0])
            np.testing.assert_array_almost_equal(navigator.max_speeds, [8.0])
        finally:
            pathlib.Path(config_path).unlink()
    
    def test_create_navigator_config_override_priority(self):
        """Test that direct parameters override configuration file values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
            navigator:
              position: [0.0, 0.0]
              orientation: 0.0
              speed: 1.0
              max_speed: 2.0
            """)
            config_path = f.name
        
        try:
            # Direct parameters should override config file values
            navigator = create_navigator(
                config_path=config_path,
                position=(100.0, 200.0),  # Override position
                speed=10.0  # Override speed
                # orientation and max_speed should come from config
            )
            
            assert isinstance(navigator, Navigator)
            np.testing.assert_array_almost_equal(navigator.positions, [[100.0, 200.0]])
            np.testing.assert_array_almost_equal(navigator.orientations, [0.0])  # From config
            np.testing.assert_array_almost_equal(navigator.speeds, [10.0])  # Overridden
            np.testing.assert_array_almost_equal(navigator.max_speeds, [2.0])  # From config
        finally:
            pathlib.Path(config_path).unlink()
    
    def test_create_navigator_error_both_position_and_positions(self):
        """Test error when both single and multi-agent parameters provided."""
        with pytest.raises(ValueError, match="Cannot specify both 'position'.*and 'positions'"):
            create_navigator(
                position=(10.0, 20.0),
                positions=[[0.0, 0.0], [10.0, 10.0]]
            )
    
    def test_create_navigator_invalid_positions_format(self):
        """Test error handling for invalid position formats."""
        # Test non-2D positions
        with pytest.raises(ValueError):
            create_navigator(positions=[[0.0], [10.0]])  # Missing y coordinate
        
        # Test empty positions
        with pytest.raises(ValueError):
            create_navigator(positions=[])
        
        # Test invalid position type
        with pytest.raises(ValueError):
            create_navigator(position="invalid")
    
    def test_create_navigator_numpy_array_compatibility(self):
        """Test navigator creation with NumPy arrays for positions and parameters."""
        positions = np.array([[5.0, 10.0], [15.0, 20.0]])
        orientations = np.array([30.0, 60.0])
        speeds = np.array([2.5, 3.5])
        max_speeds = np.array([5.0, 7.0])
        
        navigator = create_navigator(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds
        )
        
        assert isinstance(navigator, Navigator)
        assert navigator.num_agents == 2
        np.testing.assert_array_equal(navigator.positions, positions)
        np.testing.assert_array_equal(navigator.orientations, orientations)
        np.testing.assert_array_equal(navigator.speeds, speeds)
        np.testing.assert_array_equal(navigator.max_speeds, max_speeds)
    
    def test_create_navigator_factory_method_patterns(self):
        """Test factory method patterns for Kedro integration compatibility."""
        # Test factory-style creation that would be used in Kedro pipelines
        config_dict = {
            "position": [25.0, 35.0],
            "orientation": 135.0,
            "speed": 4.0,
            "max_speed": 9.0,
            "angular_velocity": 0.15
        }
        
        navigator = create_navigator(**config_dict)
        
        assert isinstance(navigator, Navigator)
        assert navigator.is_single_agent == True
        np.testing.assert_array_almost_equal(navigator.positions, [[25.0, 35.0]])
        np.testing.assert_array_almost_equal(navigator.orientations, [135.0])
        np.testing.assert_array_almost_equal(navigator.speeds, [4.0])
        np.testing.assert_array_almost_equal(navigator.max_speeds, [9.0])
    
    def test_create_navigator_performance_timing(self):
        """Test API performance requirements including initialization timing per Section 6.6.3.3."""
        start_time = time.time()
        
        navigator = create_navigator(
            position=(0.0, 0.0),
            orientation=0.0,
            speed=1.0,
            max_speed=5.0
        )
        
        initialization_time = time.time() - start_time
        
        # API function should initialize within 10ms per performance requirements
        assert initialization_time < 0.01, f"Navigator creation took {initialization_time:.3f}s, expected <0.01s"
        assert isinstance(navigator, Navigator)


class TestVideoPlumeCreationAPI:
    """
    Test suite for create_video_plume API function validation.
    
    Validates video plume factory method with path validation, parameter
    processing, and error handling per technical requirements.
    """
    
    def test_create_video_plume_basic(self):
        """Test basic video plume creation with minimal required parameters."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
            # Create a minimal valid file
            f.write(b'fake_video_content')
        
        try:
            # Mock the VideoPlume class to avoid OpenCV dependency in tests
            with patch('{{cookiecutter.project_slug}}.data.video_plume.VideoPlume') as MockVideoPlume:
                mock_instance = Mock()
                MockVideoPlume.return_value = mock_instance
                
                video_plume = create_video_plume(video_path=video_path)
                
                assert video_plume == mock_instance
                MockVideoPlume.assert_called_once()
                # Verify the path was converted to Path object
                call_args = MockVideoPlume.call_args
                assert isinstance(call_args[1]['video_path'], pathlib.Path)
                assert str(call_args[1]['video_path']) == video_path
        finally:
            pathlib.Path(video_path).unlink()
    
    def test_create_video_plume_with_processing_parameters(self):
        """Test video plume creation with processing parameters."""
        with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as f:
            video_path = f.name
            f.write(b'fake_video_content')
        
        try:
            with patch('{{cookiecutter.project_slug}}.data.video_plume.VideoPlume') as MockVideoPlume:
                mock_instance = Mock()
                MockVideoPlume.return_value = mock_instance
                
                video_plume = create_video_plume(
                    video_path=video_path,
                    flip=True,
                    kernel_size=7,
                    kernel_sigma=2.5
                )
                
                assert video_plume == mock_instance
                MockVideoPlume.assert_called_once_with(
                    video_path=pathlib.Path(video_path),
                    flip=True,
                    kernel_size=7,
                    kernel_sigma=2.5
                )
        finally:
            pathlib.Path(video_path).unlink()
    
    def test_create_video_plume_with_config_file(self):
        """Test video plume creation with configuration file."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as video_file:
            video_path = video_file.name
            video_file.write(b'fake_video_content')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            config_file.write(f"""
            video_plume:
              video_path: {video_path}
              flip: true
              kernel_size: 5
              kernel_sigma: 1.5
            """)
            config_path = config_file.name
        
        try:
            with patch('{{cookiecutter.project_slug}}.data.video_plume.VideoPlume') as MockVideoPlume:
                mock_instance = Mock()
                MockVideoPlume.return_value = mock_instance
                
                video_plume = create_video_plume(config_path=config_path)
                
                assert video_plume == mock_instance
                MockVideoPlume.assert_called_once()
                call_args = MockVideoPlume.call_args
                assert call_args[1]['flip'] == True
                assert call_args[1]['kernel_size'] == 5
                assert call_args[1]['kernel_sigma'] == 1.5
        finally:
            pathlib.Path(video_path).unlink()
            pathlib.Path(config_path).unlink()
    
    def test_create_video_plume_missing_video_path_error(self):
        """Test error when video_path is not provided."""
        with pytest.raises(ValueError, match="video_path is required"):
            create_video_plume()
    
    def test_create_video_plume_nonexistent_file_error(self):
        """Test error when video file does not exist."""
        nonexistent_path = "/path/that/does/not/exist.mp4"
        
        with pytest.raises(FileNotFoundError, match="Video file does not exist"):
            create_video_plume(video_path=nonexistent_path)
    
    def test_create_video_plume_invalid_parameters_error(self):
        """Test error handling for invalid processing parameters."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
            f.write(b'fake_video_content')
        
        try:
            # Test invalid flip parameter
            with pytest.raises(ValueError, match="flip must be a boolean"):
                create_video_plume(video_path=video_path, flip="invalid")
            
            # Test invalid kernel_size parameter
            with pytest.raises(ValueError, match="kernel_size must be a positive integer"):
                create_video_plume(video_path=video_path, kernel_size=0)
            
            with pytest.raises(ValueError, match="kernel_size must be a positive integer"):
                create_video_plume(video_path=video_path, kernel_size=-5)
            
            with pytest.raises(ValueError, match="kernel_size must be a positive integer"):
                create_video_plume(video_path=video_path, kernel_size="invalid")
        finally:
            pathlib.Path(video_path).unlink()
    
    def test_create_video_plume_path_object_support(self):
        """Test video plume creation with pathlib.Path objects."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = pathlib.Path(f.name)
            f.write(b'fake_video_content')
        
        try:
            with patch('{{cookiecutter.project_slug}}.data.video_plume.VideoPlume') as MockVideoPlume:
                mock_instance = Mock()
                MockVideoPlume.return_value = mock_instance
                
                video_plume = create_video_plume(video_path=video_path)
                
                assert video_plume == mock_instance
                MockVideoPlume.assert_called_once()
                call_args = MockVideoPlume.call_args
                assert isinstance(call_args[1]['video_path'], pathlib.Path)
                assert call_args[1]['video_path'] == video_path
        finally:
            video_path.unlink()


class TestSimulationExecutionAPI:
    """
    Test suite for run_plume_simulation API function validation.
    
    Tests simulation orchestration with parameter override scenarios,
    validates API contract and performance requirements.
    """
    
    def test_run_plume_simulation_basic(self):
        """Test basic simulation execution with required parameters."""
        # Create mock navigator and video plume
        mock_navigator = Mock()
        mock_navigator.positions = np.array([[0.0, 0.0]])
        mock_navigator.step = Mock()
        
        mock_video_plume = Mock()
        mock_video_plume.video_path = pathlib.Path("test.mp4")
        
        # Mock the actual simulation runner
        with patch('{{cookiecutter.project_slug}}.api.navigation.run_simulation') as mock_run_sim:
            expected_positions = np.array([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
            expected_orientations = np.array([[0.0, 45.0, 90.0]])
            expected_readings = np.array([[0.1, 0.2, 0.3]])
            
            mock_run_sim.return_value = (expected_positions, expected_orientations, expected_readings)
            
            positions, orientations, readings = run_plume_simulation(
                navigator=mock_navigator,
                plume=mock_video_plume,
                num_steps=3,
                dt=0.1
            )
            
            # Verify simulation was called with correct parameters
            mock_run_sim.assert_called_once_with(
                mock_navigator,
                mock_video_plume,
                num_steps=3,
                dt=0.1
            )
            
            # Verify returned data structure
            np.testing.assert_array_equal(positions, expected_positions)
            np.testing.assert_array_equal(orientations, expected_orientations)
            np.testing.assert_array_equal(readings, expected_readings)
    
    def test_run_plume_simulation_with_config_file(self):
        """Test simulation execution with configuration file parameters."""
        mock_navigator = Mock()
        mock_navigator.positions = np.array([[0.0, 0.0]])
        mock_video_plume = Mock()
        mock_video_plume.video_path = pathlib.Path("test.mp4")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
            simulation:
              num_steps: 100
              dt: 0.05
              recording: true
            """)
            config_path = f.name
        
        try:
            with patch('{{cookiecutter.project_slug}}.api.navigation.run_simulation') as mock_run_sim:
                expected_result = (
                    np.zeros((1, 101, 2)),  # positions
                    np.zeros((1, 101)),     # orientations  
                    np.zeros((1, 101))      # readings
                )
                mock_run_sim.return_value = expected_result
                
                positions, orientations, readings = run_plume_simulation(
                    navigator=mock_navigator,
                    plume=mock_video_plume,
                    config_path=config_path
                )
                
                mock_run_sim.assert_called_once_with(
                    mock_navigator,
                    mock_video_plume,
                    num_steps=100,
                    dt=0.05
                )
        finally:
            pathlib.Path(config_path).unlink()
    
    def test_run_plume_simulation_parameter_override(self):
        """Test parameter override scenarios where direct args override config."""
        mock_navigator = Mock()
        mock_navigator.positions = np.array([[0.0, 0.0]])
        mock_video_plume = Mock()
        mock_video_plume.video_path = pathlib.Path("test.mp4")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
            simulation:
              num_steps: 50
              dt: 0.2
            """)
            config_path = f.name
        
        try:
            with patch('{{cookiecutter.project_slug}}.api.navigation.run_simulation') as mock_run_sim:
                mock_run_sim.return_value = (np.zeros((1, 76, 2)), np.zeros((1, 76)), np.zeros((1, 76)))
                
                # Direct parameters should override config
                run_plume_simulation(
                    navigator=mock_navigator,
                    plume=mock_video_plume,
                    config_path=config_path,
                    num_steps=75,  # Override config value of 50
                    # dt should come from config (0.2)
                )
                
                mock_run_sim.assert_called_once_with(
                    mock_navigator,
                    mock_video_plume,
                    num_steps=75,  # Overridden value
                    dt=0.2         # From config
                )
        finally:
            pathlib.Path(config_path).unlink()
    
    def test_run_plume_simulation_backward_compatibility_step_size(self):
        """Test backward compatibility for legacy 'step_size' parameter."""
        mock_navigator = Mock()
        mock_navigator.positions = np.array([[0.0, 0.0]])
        mock_video_plume = Mock()
        mock_video_plume.video_path = pathlib.Path("test.mp4")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
            simulation:
              num_steps: 10
              step_size: 0.15  # Legacy parameter name
            """)
            config_path = f.name
        
        try:
            with patch('{{cookiecutter.project_slug}}.api.navigation.run_simulation') as mock_run_sim:
                mock_run_sim.return_value = (np.zeros((1, 11, 2)), np.zeros((1, 11)), np.zeros((1, 11)))
                
                run_plume_simulation(
                    navigator=mock_navigator,
                    plume=mock_video_plume,
                    config_path=config_path
                )
                
                # step_size should be converted to dt
                mock_run_sim.assert_called_once_with(
                    mock_navigator,
                    mock_video_plume,
                    num_steps=10,
                    dt=0.15  # Converted from step_size
                )
        finally:
            pathlib.Path(config_path).unlink()
    
    def test_run_plume_simulation_validation_errors(self):
        """Test comprehensive error handling for invalid inputs."""
        # Test None navigator
        with pytest.raises(ValueError, match="navigator and plume are required"):
            run_plume_simulation(navigator=None, plume=Mock(), num_steps=10, dt=0.1)
        
        # Test None plume
        with pytest.raises(ValueError, match="navigator and plume are required"):
            run_plume_simulation(navigator=Mock(), plume=None, num_steps=10, dt=0.1)
        
        # Test invalid navigator (missing positions attribute)
        invalid_navigator = Mock()
        del invalid_navigator.positions  # Remove positions attribute
        with pytest.raises(TypeError, match="navigator must have 'positions' attribute"):
            run_plume_simulation(navigator=invalid_navigator, plume=Mock(), num_steps=10, dt=0.1)
        
        # Test invalid plume (missing video_path attribute)
        invalid_plume = Mock()
        del invalid_plume.video_path
        with pytest.raises(TypeError, match="plume must have 'video_path' attribute"):
            run_plume_simulation(navigator=Mock(), plume=invalid_plume, num_steps=10, dt=0.1)
        
        # Test invalid num_steps
        mock_navigator = Mock()
        mock_navigator.positions = np.array([[0.0, 0.0]])
        mock_plume = Mock()
        mock_plume.video_path = pathlib.Path("test.mp4")
        
        with pytest.raises(ValueError, match="num_steps must be a positive integer"):
            run_plume_simulation(navigator=mock_navigator, plume=mock_plume, num_steps=0, dt=0.1)
        
        with pytest.raises(ValueError, match="num_steps must be a positive integer"):
            run_plume_simulation(navigator=mock_navigator, plume=mock_plume, num_steps=-5, dt=0.1)
        
        with pytest.raises(ValueError, match="num_steps must be a positive integer"):
            run_plume_simulation(navigator=mock_navigator, plume=mock_plume, num_steps="invalid", dt=0.1)
        
        # Test invalid dt
        with pytest.raises(ValueError, match="dt must be a positive float"):
            run_plume_simulation(navigator=mock_navigator, plume=mock_plume, num_steps=10, dt=0)
        
        with pytest.raises(ValueError, match="dt must be a positive float"):
            run_plume_simulation(navigator=mock_navigator, plume=mock_plume, num_steps=10, dt=-0.1)
        
        with pytest.raises(ValueError, match="dt must be a positive float"):
            run_plume_simulation(navigator=mock_navigator, plume=mock_plume, num_steps=10, dt="invalid")
    
    def test_run_plume_simulation_multi_agent_output_shaping(self):
        """Test proper output dimensionality for multi-agent simulations."""
        # Create multi-agent navigator mock
        mock_navigator = Mock()
        mock_navigator.positions = np.array([[0.0, 0.0], [10.0, 10.0]])  # 2 agents
        
        mock_video_plume = Mock()
        mock_video_plume.video_path = pathlib.Path("test.mp4")
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.run_simulation') as mock_run_sim:
            # Simulate single-agent output that needs to be tiled for multi-agent
            single_agent_positions = np.array([[[0.0, 0.0], [1.0, 1.0]]])  # (1, 2, 2)
            single_agent_orientations = np.array([[0.0, 45.0]])             # (1, 2)
            single_agent_readings = np.array([[0.1, 0.2]])                  # (1, 2)
            
            mock_run_sim.return_value = (single_agent_positions, single_agent_orientations, single_agent_readings)
            
            positions, orientations, readings = run_plume_simulation(
                navigator=mock_navigator,
                plume=mock_video_plume,
                num_steps=2,
                dt=0.1
            )
            
            # Verify output is properly shaped for 2 agents
            assert positions.shape == (2, 2, 2)  # (n_agents, n_steps, 2)
            assert orientations.shape == (2, 2)   # (n_agents, n_steps)
            assert readings.shape == (2, 2)       # (n_agents, n_steps)


class TestVisualizationAPI:
    """
    Test suite for visualization API function validation.
    
    Tests visualization integration with export capabilities per F-008-RQ-004
    requirements and validates comprehensive parameter handling.
    """
    
    def test_visualize_simulation_results_basic(self):
        """Test basic visualization function with minimal parameters."""
        positions = np.array([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
        orientations = np.array([[0.0, 45.0, 90.0]])
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.visualize_trajectory') as mock_visualize:
            mock_figure = Mock()
            mock_visualize.return_value = mock_figure
            
            result = visualize_simulation_results(
                positions=positions,
                orientations=orientations
            )
            
            mock_visualize.assert_called_once_with(
                positions=positions,
                orientations=orientations,
                plume_frames=None,
                output_path=None,
                show_plot=True,
                close_plot=None
            )
            assert result == mock_figure
    
    def test_visualize_simulation_results_with_plume_frames(self):
        """Test visualization with plume frame overlay."""
        positions = np.array([[[0.0, 0.0], [1.0, 1.0]]])
        orientations = np.array([[0.0, 45.0]])
        plume_frames = np.random.randint(0, 255, size=(2, 100, 100, 3), dtype=np.uint8)
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.visualize_trajectory') as mock_visualize:
            mock_figure = Mock()
            mock_visualize.return_value = mock_figure
            
            result = visualize_simulation_results(
                positions=positions,
                orientations=orientations,
                plume_frames=plume_frames
            )
            
            mock_visualize.assert_called_once_with(
                positions=positions,
                orientations=orientations,
                plume_frames=plume_frames,
                output_path=None,
                show_plot=True,
                close_plot=None
            )
            assert result == mock_figure
    
    def test_visualize_simulation_results_with_export(self):
        """Test visualization with file export capabilities."""
        positions = np.array([[[0.0, 0.0], [1.0, 1.0]]])
        orientations = np.array([[0.0, 45.0]])
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            with patch('{{cookiecutter.project_slug}}.api.navigation.visualize_trajectory') as mock_visualize:
                mock_figure = Mock()
                mock_visualize.return_value = mock_figure
                
                result = visualize_simulation_results(
                    positions=positions,
                    orientations=orientations,
                    output_path=output_path,
                    show_plot=False,
                    close_plot=True
                )
                
                mock_visualize.assert_called_once_with(
                    positions=positions,
                    orientations=orientations,
                    plume_frames=None,
                    output_path=pathlib.Path(output_path),
                    show_plot=False,
                    close_plot=True
                )
                assert result == mock_figure
        finally:
            pathlib.Path(output_path).unlink(missing_ok=True)
    
    def test_visualize_simulation_results_path_object_support(self):
        """Test visualization API accepts pathlib.Path objects."""
        positions = np.array([[[0.0, 0.0], [1.0, 1.0]]])
        orientations = np.array([[0.0, 45.0]])
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            output_path = pathlib.Path(f.name)
        
        try:
            with patch('{{cookiecutter.project_slug}}.api.navigation.visualize_trajectory') as mock_visualize:
                mock_figure = Mock()
                mock_visualize.return_value = mock_figure
                
                result = visualize_simulation_results(
                    positions=positions,
                    orientations=orientations,
                    output_path=output_path
                )
                
                mock_visualize.assert_called_once()
                call_args = mock_visualize.call_args
                assert call_args[1]['output_path'] == output_path
                assert isinstance(call_args[1]['output_path'], pathlib.Path)
        finally:
            output_path.unlink(missing_ok=True)


class TestAPIErrorHandling:
    """
    Test suite for comprehensive API error handling validation.
    
    Ensures robust error handling for invalid configurations and validates
    error messages provide actionable feedback per API reliability requirements.
    """
    
    def test_api_error_messages_are_descriptive(self):
        """Test that API error messages provide clear, actionable information."""
        # Test navigator creation error
        try:
            create_navigator(position=(10.0, 20.0), positions=[[0.0, 0.0]])
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            error_msg = str(e)
            assert "position" in error_msg.lower()
            assert "positions" in error_msg.lower()
            assert "cannot specify both" in error_msg.lower()
    
    def test_api_type_validation_errors(self):
        """Test type validation errors across API functions."""
        # Test navigator creation with invalid types
        with pytest.raises((TypeError, ValueError)):
            create_navigator(position="invalid_type")
        
        with pytest.raises((TypeError, ValueError)):
            create_navigator(orientation="not_a_number")
        
        with pytest.raises((TypeError, ValueError)):
            create_navigator(speed=[1, 2, 3])  # Should be scalar for single agent
    
    def test_api_boundary_value_validation(self):
        """Test boundary value validation across API functions."""
        # Test negative speed values
        with pytest.raises(ValueError):
            create_navigator(position=(0.0, 0.0), speed=-1.0)
        
        # Test max_speed less than speed
        with pytest.raises(ValueError):
            create_navigator(position=(0.0, 0.0), speed=10.0, max_speed=5.0)
    
    def test_api_graceful_degradation(self):
        """Test API graceful degradation with partial configuration."""
        # Navigator should work with minimal parameters
        navigator = create_navigator()
        assert isinstance(navigator, Navigator)
        assert navigator.is_single_agent == True
        
        # Default values should be reasonable
        assert navigator.num_agents == 1
        assert navigator.positions.shape == (1, 2)
        assert navigator.orientations.shape == (1,)
        assert navigator.speeds.shape == (1,)
        assert navigator.max_speeds.shape == (1,)


class TestAPIBackwardCompatibility:
    """
    Test suite for backward compatibility validation.
    
    Ensures API maintains compatibility for migration scenarios per
    technical specifications and validates legacy import patterns.
    """
    
    def test_legacy_import_patterns(self):
        """Test that legacy import patterns still work for backward compatibility."""
        # Test that the new API supports old-style usage patterns
        # This would typically involve testing alternative import paths
        # or parameter names that existed in the previous version
        
        # Example: Legacy position parameter should work
        navigator = create_navigator(position=(5.0, 10.0))
        assert isinstance(navigator, Navigator)
        np.testing.assert_array_equal(navigator.positions, [[5.0, 10.0]])
    
    def test_legacy_parameter_names(self):
        """Test support for legacy parameter names."""
        # Test that old parameter names are still supported
        # Example: If 'step_size' was renamed to 'dt' in run_plume_simulation
        mock_navigator = Mock()
        mock_navigator.positions = np.array([[0.0, 0.0]])
        mock_video_plume = Mock()
        mock_video_plume.video_path = pathlib.Path("test.mp4")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
            simulation:
              num_steps: 5
              step_size: 0.2  # Legacy parameter name
            """)
            config_path = f.name
        
        try:
            with patch('{{cookiecutter.project_slug}}.api.navigation.run_simulation') as mock_run_sim:
                mock_run_sim.return_value = (np.zeros((1, 6, 2)), np.zeros((1, 6)), np.zeros((1, 6)))
                
                # Should accept legacy parameter name in config
                run_plume_simulation(
                    navigator=mock_navigator,
                    plume=mock_video_plume,
                    config_path=config_path
                )
                
                # Verify legacy parameter was properly converted
                mock_run_sim.assert_called_once()
                call_args = mock_run_sim.call_args
                assert call_args[1]['dt'] == 0.2  # step_size converted to dt
        finally:
            pathlib.Path(config_path).unlink()
    
    def test_migration_scenario_compatibility(self):
        """Test compatibility patterns for common migration scenarios."""
        # Test that users migrating from old API can use both patterns
        
        # Old-style single position
        navigator1 = create_navigator(position=(10.0, 20.0))
        
        # New-style single position (should work the same way)
        navigator2 = create_navigator(positions=[[10.0, 20.0]])
        
        # Both should create equivalent single-agent navigators
        assert navigator1.is_single_agent == navigator2.is_single_agent == True
        assert navigator1.num_agents == navigator2.num_agents == 1
        np.testing.assert_array_equal(navigator1.positions, navigator2.positions)


class TestAPIPerformanceRequirements:
    """
    Test suite for API performance requirements validation.
    
    Validates API functions meet performance thresholds per Section 6.6.3.3
    and ensures initialization timing requirements are met.
    """
    
    def test_navigator_creation_performance(self):
        """Test navigator creation meets performance thresholds."""
        start_time = time.time()
        
        # Create multiple navigators to test consistent performance
        for _ in range(10):
            navigator = create_navigator(
                position=(np.random.random() * 100, np.random.random() * 100),
                orientation=np.random.random() * 360,
                speed=np.random.random() * 5,
                max_speed=5.0 + np.random.random() * 5
            )
            assert isinstance(navigator, Navigator)
        
        total_time = time.time() - start_time
        avg_time_per_creation = total_time / 10
        
        # Each creation should take less than 1ms on average
        assert avg_time_per_creation < 0.001, f"Average creation time {avg_time_per_creation:.4f}s exceeds 1ms threshold"
    
    def test_video_plume_creation_performance(self):
        """Test video plume creation performance with file validation."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
            f.write(b'fake_video_content')
        
        try:
            start_time = time.time()
            
            with patch('{{cookiecutter.project_slug}}.data.video_plume.VideoPlume') as MockVideoPlume:
                MockVideoPlume.return_value = Mock()
                
                # Test file validation and creation performance
                for _ in range(5):
                    video_plume = create_video_plume(video_path=video_path)
                    assert video_plume is not None
            
            total_time = time.time() - start_time
            avg_time = total_time / 5
            
            # Each creation with file validation should be fast
            assert avg_time < 0.01, f"Average video plume creation time {avg_time:.4f}s exceeds 10ms threshold"
        finally:
            pathlib.Path(video_path).unlink()
    
    def test_api_memory_efficiency(self):
        """Test API functions don't create excessive memory overhead."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple objects to test memory efficiency
        navigators = []
        for i in range(100):
            navigator = create_navigator(
                position=(i, i),
                orientation=i,
                speed=1.0,
                max_speed=5.0
            )
            navigators.append(navigator)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 10MB for 100 navigators)
        assert memory_increase < 10 * 1024 * 1024, f"Memory increase {memory_increase / 1024 / 1024:.1f}MB too large"


class TestKedroIntegrationPatterns:
    """
    Test suite for Kedro integration compatibility validation.
    
    Tests factory method patterns for Kedro integration per enhanced
    integration testing requirements from Section 0.2.1.
    """
    
    def test_kedro_factory_method_pattern(self):
        """Test factory method patterns work with Kedro-style configurations."""
        # Simulate Kedro parameter structure
        kedro_config = {
            "navigator": {
                "position": [30.0, 40.0],
                "orientation": 180.0,
                "speed": 3.5,
                "max_speed": 8.0,
                "angular_velocity": 0.25
            },
            "video_plume": {
                "flip": False,
                "kernel_size": 7,
                "kernel_sigma": 1.8
            },
            "simulation": {
                "num_steps": 200,
                "dt": 0.05
            }
        }
        
        # Test navigator creation with Kedro-style config
        navigator = create_navigator(**kedro_config["navigator"])
        assert isinstance(navigator, Navigator)
        assert navigator.is_single_agent == True
        np.testing.assert_array_almost_equal(navigator.positions, [[30.0, 40.0]])
        np.testing.assert_array_almost_equal(navigator.orientations, [180.0])
        
        # Test video plume creation pattern (with mocked file)
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
            f.write(b'fake_video_content')
        
        try:
            with patch('{{cookiecutter.project_slug}}.data.video_plume.VideoPlume') as MockVideoPlume:
                MockVideoPlume.return_value = Mock()
                
                video_plume_config = kedro_config["video_plume"].copy()
                video_plume_config["video_path"] = video_path
                
                video_plume = create_video_plume(**video_plume_config)
                assert video_plume is not None
        finally:
            pathlib.Path(video_path).unlink()
    
    def test_kedro_data_catalog_compatibility(self):
        """Test API objects work with Kedro DataCatalog patterns."""
        # Test that API objects can be serialized/deserialized
        # This is important for Kedro pipeline integration
        
        navigator = create_navigator(
            position=(15.0, 25.0),
            orientation=90.0,
            speed=2.0,
            max_speed=6.0
        )
        
        # Test that navigator has expected attributes for persistence
        assert hasattr(navigator, 'positions')
        assert hasattr(navigator, 'orientations')
        assert hasattr(navigator, 'speeds')
        assert hasattr(navigator, 'max_speeds')
        assert hasattr(navigator, 'is_single_agent')
        assert hasattr(navigator, 'num_agents')
        
        # Test that navigator is pickleable (required for Kedro)
        import pickle
        try:
            serialized = pickle.dumps(navigator)
            deserialized = pickle.loads(serialized)
            
            # Verify key attributes are preserved
            assert deserialized.is_single_agent == navigator.is_single_agent
            assert deserialized.num_agents == navigator.num_agents
            np.testing.assert_array_equal(deserialized.positions, navigator.positions)
        except Exception as e:
            pytest.fail(f"Navigator not pickleable for Kedro compatibility: {e}")


# Test fixtures and utilities for comprehensive API testing
@pytest.fixture
def sample_navigator_config():
    """Provide sample navigator configuration for testing."""
    return {
        "position": [20.0, 30.0],
        "orientation": 45.0,
        "speed": 2.5,
        "max_speed": 7.0,
        "angular_velocity": 0.15
    }


@pytest.fixture
def sample_multi_agent_config():
    """Provide sample multi-agent configuration for testing."""
    return {
        "positions": [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]],
        "orientations": [0.0, 120.0, 240.0],
        "speeds": [1.0, 2.0, 3.0],
        "max_speeds": [5.0, 6.0, 7.0],
        "angular_velocities": [0.1, 0.2, 0.3]
    }


@pytest.fixture
def temporary_video_file():
    """Provide temporary video file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(b'fake_video_content_for_testing')
        video_path = f.name
    
    yield video_path
    
    # Cleanup
    pathlib.Path(video_path).unlink(missing_ok=True)


@pytest.fixture
def mock_simulation_results():
    """Provide mock simulation results for visualization testing."""
    return {
        "positions": np.array([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]),
        "orientations": np.array([[0.0, 45.0, 90.0, 135.0]]),
        "readings": np.array([[0.1, 0.2, 0.3, 0.4]])
    }


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v"])