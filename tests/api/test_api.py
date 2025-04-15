"""Tests for the public API functions."""


import contextlib
import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from pathlib import Path

from odor_plume_nav.api import (
    create_navigator,
    create_video_plume,
    run_plume_simulation,
    visualize_simulation_results,
    visualize_trajectory
)


def test_create_navigator_default():
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


def test_create_navigator_single_agent():
    """Test creating a navigator with single agent parameters."""
    # Create a navigator with single agent parameters
    navigator = create_navigator(
        positions=(10, 20),
        orientations=45,
        speeds=0.5,
        max_speeds=2.0
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


def test_create_navigator_multi_agent():
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


def test_create_navigator_numpy_array_positions():
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


@pytest.fixture
def mock_config_load():
    """Mock the config loading function everywhere it's used/imported."""
    import numpy as np
    from unittest.mock import patch
    # Patch only in locations where load_config is imported and used
    with patch('odor_plume_nav.api.load_config') as api_mock, \
         patch('odor_plume_nav.config.utils.load_config') as config_mock:
        # Create valid config with numpy arrays since that's what the validator expects
        positions = np.array([[10, 20], [30, 40]])
        orientations = np.array([45, 90])
        speeds = np.array([0.5, 0.7])
        max_speeds = np.array([1.0, 1.0])
        dummy_config = {
            "positions": positions,
            "orientations": orientations,
            "speeds": speeds,
            "max_speeds": max_speeds,
            "video_plume": {
                "flip": True,
                "kernel_size": 5,
                "kernel_sigma": 1.0
            }
        }
        api_mock.return_value = dummy_config
        config_mock.return_value = dummy_config
        yield api_mock


def test_create_navigator_from_config(mock_config_load):
    """Test creating a navigator from a configuration file."""
    # Create a navigator from a configuration file
    navigator = create_navigator(config_path="test_config.yaml")
    
    # Verify config was loaded
    mock_config_load.assert_called_once_with("test_config.yaml")
    
    # Check that the navigator has the correct properties with multi-agent protocol
    assert len(navigator.positions) == 2  # Two agents from config
    assert navigator.orientations[0] == 45  # First agent's orientation
    assert navigator.speeds[0] == 0.5  # First agent's speed
    assert np.allclose(navigator.positions[0], [10, 20])  # First agent's position


def test_create_navigator_from_config_single_agent(mock_config_load):
    """Test creating a single-agent navigator from config."""
    # Override the mock to return a single-agent config
    mock_config_load.return_value = {
        "position": (10, 20),
        "orientation": 45,
        "speed": 0.5,
        "max_speed": 1.0,
        "video_plume": {
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.0
        }
    }
    
    # Create a navigator from the single-agent config
    navigator = create_navigator(config_path="single_agent_config.yaml")
    
    # Verify config was loaded
    mock_config_load.assert_called_once_with("single_agent_config.yaml")
    
    # Check that the navigator has the correct properties with single-agent protocol
    assert len(navigator.positions) == 1  # Single agent
    assert navigator.orientations[0] == 45
    assert navigator.speeds[0] == 0.5
    assert navigator.max_speeds[0] == 1.0
    assert np.allclose(navigator.positions[0], [10, 20])


@pytest.fixture
def mock_video_capture():
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
def mock_exists(monkeypatch):
    """Mock the Path.exists method to return True for all paths."""
    def patched_exists(self):
        return True
    
    monkeypatch.setattr(Path, "exists", patched_exists)
    return patched_exists


def test_create_video_plume(mock_video_capture, mock_exists):
    """Test creating a video plume with the API function."""
    # Create a video plume
    plume = create_video_plume("test_video.mp4", flip=True, kernel_size=5)
    
    # Check that the plume has the correct properties
    assert plume.video_path == Path("test_video.mp4")
    assert plume.flip is True
    assert plume.kernel_size == 5


def test_create_video_plume_with_config(mock_video_capture, mock_exists, mock_config_load):
    """Test creating a video plume with a configuration file."""
    # Create a video plume from a configuration file
    plume = create_video_plume("test_video.mp4", config_path="test_config.yaml")
    
    # Verify config was loaded
    mock_config_load.assert_called_once_with("test_config.yaml")


@pytest.fixture
def mock_run_simulation():
    """Mock the run_simulation function."""
    # We need to patch the function where it's imported, not where it's defined
    with patch('odor_plume_nav.api.run_simulation') as mock_run:
        # Configure mock to return synthetic data
        positions_history = np.array([[[0, 0], [1, 1], [2, 2]]])
        orientations_history = np.array([[0, 45, 90]])
        odor_readings = np.array([[0.1, 0.2, 0.3]])
        
        mock_run.return_value = (positions_history, orientations_history, odor_readings)
        yield mock_run


def test_run_plume_simulation(mock_run_simulation, mock_video_capture, mock_exists):
    """Test running a plume simulation with the API function."""
    # Create a navigator and video plume
    navigator = create_navigator(positions=(10, 20), orientations=45)
    plume = create_video_plume("test_video.mp4")
    
    # Run the simulation
    positions, orientations, readings = run_plume_simulation(
        navigator, plume, num_steps=100, dt=0.5
    )
    
    # Check that the simulation function was called with the correct parameters
    mock_run_simulation.assert_called_once()
    args, kwargs = mock_run_simulation.call_args
    assert args[0] == navigator
    assert args[1] == plume
    assert kwargs["num_steps"] == 100
    assert kwargs["dt"] == 0.5
    
    # Check that the results were returned correctly
    assert positions.shape == (1, 3, 2)  # (num_agents, num_steps, 2)
    assert orientations.shape == (1, 3)  # (num_agents, num_steps)
    assert readings.shape == (1, 3)    # (num_agents, num_steps)


@pytest.fixture
def mock_visualize_trajectory():
    """Mock the visualize_trajectory function as imported in the API module."""
    with patch('odor_plume_nav.api.visualize_trajectory') as mock_viz:
        yield mock_viz


def test_visualize_simulation_results(mock_visualize_trajectory):
    """Test visualizing simulation results with the API function."""
    # Create synthetic simulation results
    positions = np.array([[[0, 0], [1, 1], [2, 2]]])
    orientations = np.array([[0, 45, 90]])
    
    # Visualize the results
    from odor_plume_nav.api import visualize_simulation_results
    visualize_simulation_results(
        positions, orientations, output_path="test_output.png", show_plot=False
    )
    
    # Check that the visualization function was called with the correct parameters
    mock_visualize_trajectory.assert_called_once()
    _, kwargs = mock_visualize_trajectory.call_args
    assert np.array_equal(kwargs["positions"], positions)
    assert np.array_equal(kwargs["orientations"], orientations)
    assert kwargs["output_path"] == "test_output.png"
    assert kwargs["show_plot"] is False


def test_create_navigator_config_and_override(tmp_path, mock_config_load):
    """Test that direct arguments override config file values when both are provided."""
    # Mock config: positions = [[10, 20], [30, 40]], orientations = [45, 90]
    config_path = tmp_path / "test_config.yaml"
    # The mock_config_load fixture patches the config loader, so we only need to call with the path
    # Direct argument overrides orientation for first agent
    override_orientations = [180, 90]
    navigator = create_navigator(
        config_path=str(config_path),
        orientations=override_orientations
    )
    # Direct argument should take precedence
    assert np.allclose(navigator.orientations, override_orientations)
    # Positions should still be loaded from config
    assert np.allclose(navigator.positions, np.array([[10, 20], [30, 40]]))


def test_create_video_plume_config_override(mock_video_capture, mock_exists, mock_config_load):
    """Direct argument should override config value."""
    mock_config_load.return_value = {
        "video_path": "test_video.mp4",
        "flip": False,
        "kernel_size": 3
    }
    plume = create_video_plume("test_video.mp4", config_path="test_config.yaml", flip=True, kernel_size=5)
    assert plume.flip is True
    assert plume.kernel_size == 5


def test_create_video_plume_partial_config(mock_video_capture, mock_exists, mock_config_load):
    """Direct argument supplies missing config field."""
    mock_config_load.return_value = {
        "video_path": "test_video.mp4",
        "flip": True
    }
    plume = create_video_plume("test_video.mp4", config_path="test_config.yaml", kernel_size=7)
    assert plume.kernel_size == 7
    assert plume.flip is True


def test_create_video_plume_invalid_kernel_size(mock_video_capture, mock_exists):
    """Invalid kernel_size (negative/int as string) raises ValueError."""
    with pytest.raises(ValueError):
        create_video_plume("test_video.mp4", kernel_size=-1)
    with pytest.raises(ValueError):
        create_video_plume("test_video.mp4", kernel_size="five")


def test_create_video_plume_invalid_flip(mock_video_capture, mock_exists):
    """Non-bool flip raises ValueError."""
    with pytest.raises(ValueError):
        create_video_plume("test_video.mp4", flip="yes")


def test_create_video_plume_missing_video_path(mock_video_capture, mock_exists):
    """Missing video_path should raise TypeError or ValueError."""
    with pytest.raises((TypeError, ValueError)):
        create_video_plume()


def test_create_video_plume_unknown_config_field(mock_video_capture, mock_exists, mock_config_load):
    """Unknown config field is ignored or raises error (depending on implementation)."""
    mock_config_load.return_value = {
        "video_path": "test_video.mp4",
        "flip": True,
        "kernel_size": 3,
        "unknown_field": 42
    }
    # Accept either: ignore unknown field, or raise ValueError
    with contextlib.suppress(ValueError):
        plume = create_video_plume("test_video.mp4", config_path="test_config.yaml")
        assert hasattr(plume, "video_path")


def test_create_video_plume_conflicting_fields(mock_video_capture, mock_exists, mock_config_load):
    """Direct arg and config provide different values for same field; direct arg wins."""
    mock_config_load.return_value = {
        "video_path": "test_video.mp4",
        "flip": False,
        "kernel_size": 3
    }
    plume = create_video_plume("test_video.mp4", config_path="test_config.yaml", flip=True)
    assert plume.flip is True


# Edge case: invalid file path (if path validation is present)
def test_create_video_plume_invalid_path(mock_video_capture, monkeypatch):
    """Non-existent video file path raises error if validated."""
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises((FileNotFoundError, ValueError)):
        create_video_plume("nonexistent.mp4")


import pytest
@pytest.mark.parametrize(
    "positions,expected_exception",
    [
        ([(1, 2), (3, 4)], None),  # valid
        ([1, 2], None),            # valid single-agent: treat as (x, y)
        ([(1, 2, 3), (4, 5, 6)], ValueError),  # wrong shape
        (["a", "b"], ValueError),  # not numeric
        ([[1], [2]], ValueError),   # wrong length
        ([(1, 2), (3,)], ValueError),  # one valid, one invalid
        (np.array([[1, 2], [3, 4]]), None),  # valid np.ndarray
        (np.array([[1], [2]]), ValueError),  # invalid np.ndarray
    ]
)


@pytest.mark.parametrize(
    "i,expected_pos",
    [(0, (10, 20)), (1, (30, 40)), (2, (50, 60))]
)
def test_create_navigator_position_index(i, expected_pos):
    positions = [(10, 20), (30, 40), (50, 60)]
    navigator = create_navigator(
        positions=positions
    )
    assert np.allclose(navigator.positions[i], expected_pos)


@pytest.mark.parametrize(
    "positions,expected_shape",
    [
        ([(1, 2), (3, 4)], (2, 2)),
        ([1, 2], (1, 2)),
        ((1, 2), (1, 2)),
        (np.array([[1, 2], [3, 4]]), (2, 2)),
    ]
)
def test_create_navigator_positions_shape_valid(positions, expected_shape):
    navigator = create_navigator(positions=positions)
    assert navigator.positions.shape == expected_shape

@pytest.mark.parametrize(
    "positions",
    [
        (np.array([[1], [2]])),
    ]
)
def test_create_navigator_positions_shape_invalid(positions):
    with pytest.raises(ValueError):
        create_navigator(positions=positions)


def test_run_plume_simulation_valid_dt():
    """run_plume_simulation should accept dt and produce correct output."""
    from odor_plume_nav.api import create_navigator, create_video_plume, run_plume_simulation
    import numpy as np
    class DummyVideoCapture:
        def __init__(self, *a, **kw): pass
        def isOpened(self): return True
        def read(self): return True, np.zeros((10, 10, 3), dtype=np.uint8)
        def release(self): pass
        def get(self, prop): return 1  # Plausible dummy value for any property
        def set(self, prop, value): pass  # Accept any set call, do nothing
    with patch("pathlib.Path.exists", return_value=True), \
         patch("cv2.VideoCapture", DummyVideoCapture):
        nav = create_navigator(positions=[(0.0, 0.0), (1.0, 1.0)])  # ensure float positions
        plume = create_video_plume("test_video.mp4")
        pos, ori, read = run_plume_simulation(nav, plume, num_steps=3, dt=0.5)
        assert pos.shape == (2, 3, 2)
        assert ori.shape == (2, 3)
        assert read.shape == (2, 3)


def test_run_plume_simulation_config_override(mock_run_simulation, mock_video_capture, mock_exists, mock_config_load):
    """Direct argument should override config value."""
    mock_config_load.return_value = {
        "num_steps": 50,
        "dt": 0.1
    }
    navigator = create_navigator(positions=(0, 0))
    plume = create_video_plume("test_video.mp4")
    positions, orientations, readings = run_plume_simulation(
        navigator, plume, num_steps=100, dt=0.5, config_path="test_config.yaml"
    )
    mock_run_simulation.assert_called_once()
    args, kwargs = mock_run_simulation.call_args
    assert kwargs["num_steps"] == 100
    assert kwargs["dt"] == 0.5


def test_run_plume_simulation_partial_config(mock_run_simulation, mock_video_capture, mock_exists, mock_config_load):
    """Direct argument supplies missing config field."""
    mock_config_load.return_value = {
        "num_steps": 25
    }
    navigator = create_navigator(positions=(0, 0))
    plume = create_video_plume("test_video.mp4")
    positions, orientations, readings = run_plume_simulation(
        navigator, plume, config_path="test_config.yaml", dt=0.2
    )
    mock_run_simulation.assert_called_once()
    args, kwargs = mock_run_simulation.call_args
    assert kwargs["num_steps"] == 25
    assert kwargs["dt"] == 0.2


def test_run_plume_simulation_direct_args_only(mock_run_simulation, mock_video_capture, mock_exists):
    """Test running simulation with only direct arguments."""
    navigator = create_navigator(positions=(0, 0))
    plume = create_video_plume("test_video.mp4")
    positions, orientations, readings = run_plume_simulation(
        navigator, plume, num_steps=5, dt=1.0
    )
    mock_run_simulation.assert_called_once()
    args, kwargs = mock_run_simulation.call_args
    assert kwargs["num_steps"] == 5
    assert kwargs["dt"] == 1.0


@pytest.mark.parametrize("bad", [0, -1, "ten"])
def test_run_plume_simulation_invalid_num_steps_param(mock_run_simulation, mock_video_capture, mock_exists, bad):
    """Negative or zero num_steps raises ValueError."""
    navigator = create_navigator(positions=(0, 0))
    plume = create_video_plume("test_video.mp4")
    with pytest.raises(ValueError):
        run_plume_simulation(navigator, plume, num_steps=bad, dt=1.0)


@pytest.mark.parametrize("bad", [0, -0.1, "small"])
def test_run_plume_simulation_invalid_dt_param(mock_run_simulation, mock_video_capture, mock_exists, bad):
    """Non-positive or non-float dt raises ValueError."""
    navigator = create_navigator(positions=(0, 0))
    plume = create_video_plume("test_video.mp4")
    with pytest.raises(ValueError):
        run_plume_simulation(navigator, plume, num_steps=10, dt=bad)


def test_run_plume_simulation_missing_required(mock_run_simulation, mock_video_capture, mock_exists):
    """Missing navigator or plume raises TypeError or ValueError."""
    with pytest.raises((TypeError, ValueError)):
        run_plume_simulation(None, None, num_steps=5, dt=1.0)


def test_run_plume_simulation_unknown_config_field(mock_run_simulation, mock_video_capture, mock_exists, mock_config_load):
    """Unknown config field is ignored or raises error."""
    mock_config_load.return_value = {
        "num_steps": 10,
        "dt": 1.0,
        "unknown_field": 42
    }
    navigator = create_navigator(positions=(0, 0))
    plume = create_video_plume("test_video.mp4")
    with contextlib.suppress(ValueError):
        run_plume_simulation(navigator, plume, config_path="test_config.yaml")


def test_run_plume_simulation_conflicting_fields(mock_run_simulation, mock_video_capture, mock_exists, mock_config_load):
    """Direct arg and config provide different values for same field; direct arg wins."""
    mock_config_load.return_value = {
        "num_steps": 10,
        "dt": 1.0
    }
    navigator = create_navigator(positions=(0, 0))
    plume = create_video_plume("test_video.mp4")
    positions, orientations, readings = run_plume_simulation(
        navigator, plume, config_path="test_config.yaml", num_steps=20
    )
    mock_run_simulation.assert_called_once()
    args, kwargs = mock_run_simulation.call_args
    assert kwargs["num_steps"] == 20
    assert kwargs["dt"] == 1.0


def test_run_plume_simulation_output_shapes(mock_run_simulation, mock_video_capture, mock_exists):
    """Output arrays have correct shapes for single- and multi-agent."""
    # Single agent
    navigator = create_navigator(positions=(0, 0))
    plume = create_video_plume("test_video.mp4")
    positions, orientations, readings = run_plume_simulation(
        navigator, plume, num_steps=3, dt=1.0
    )
    assert positions.shape == (1, 3, 2)
    assert orientations.shape == (1, 3)
    assert readings.shape == (1, 3)
    # Multi-agent
    navigator = create_navigator(positions=[(0, 0), (1, 1)])
    positions, orientations, readings = run_plume_simulation(
        navigator, plume, num_steps=3, dt=1.0
    )
    assert positions.shape == (2, 3, 2)
    assert orientations.shape == (2, 3)
    assert readings.shape == (2, 3)


def test_run_plume_simulation_mismatched_types(mock_run_simulation, mock_video_capture, mock_exists):
    """Navigator and plume from incompatible protocols raises error (if enforced)."""
    # Here, just pass wrong types
    with pytest.raises((TypeError, ValueError)):
        run_plume_simulation("not_a_navigator", "not_a_plume", num_steps=5, dt=1.0)


def test_run_plume_simulation_edge_output_cases(mock_run_simulation, mock_video_capture, mock_exists):
    """Edge output cases: 1 agent, 1 step; output shapes correct."""
    navigator = create_navigator(positions=(0, 0))
    plume = create_video_plume("test_video.mp4")
    positions, orientations, readings = run_plume_simulation(
        navigator, plume, num_steps=1, dt=1.0
    )
    assert positions.shape == (1, 1, 2)
    assert orientations.shape == (1, 1)
    assert readings.shape == (1, 1)

def test_load_navigator_from_config_raises_on_unknown_keys(mock_config_load):
    """Unknown keys in config should raise ValueError."""
    from odor_plume_nav.utils.navigator_utils import load_navigator_from_config
    config = {"positions": [(0, 0), (1, 1)], "orientations": [0, 90], "unknown": 42}
    with pytest.raises(ValueError, match="Unknown keys in config: {'unknown'}"):
        load_navigator_from_config(config)

def test_load_navigator_from_config_raises_on_missing_positions_multi(mock_config_load):
    """Missing 'positions' in multi-agent config should raise ValueError."""
    from odor_plume_nav.utils.navigator_utils import load_navigator_from_config
    config = {"orientations": [0, 90]}
    with pytest.raises(ValueError, match="Config must include either 'positions' \(multi-agent\) or 'position' \(single-agent\) key."):
        load_navigator_from_config(config)

def test_load_navigator_from_config_raises_on_missing_position_single(mock_config_load):
    """Missing 'position' in single-agent config should raise ValueError."""
    from odor_plume_nav.utils.navigator_utils import load_navigator_from_config
    config = {"orientation": 0}
    with pytest.raises(ValueError, match="Config must include either 'positions' \(multi-agent\) or 'position' \(single-agent\) key."):
        load_navigator_from_config(config)

def test_load_navigator_from_config_raises_on_invalid_positions_type(mock_config_load):
    """Invalid positions type in multi-agent config should raise ValueError."""
    from odor_plume_nav.utils.navigator_utils import load_navigator_from_config
    config = {"positions": 123, "orientations": [0, 90]}
    with pytest.raises(ValueError, match="positions must be a single \(x, y\) or a sequence of \(x, y\) pairs \(shape \(2,\) or \(N, 2\)\)"):
        load_navigator_from_config(config)

def test_load_navigator_from_config_raises_on_invalid_position_type(mock_config_load):
    """Invalid position type in single-agent config should raise ValueError."""
    from odor_plume_nav.utils.navigator_utils import load_navigator_from_config
    config = {"position": [1, 2, 3], "orientation": 0}
    with pytest.raises(ValueError, match="positions must be a single \(x, y\) or a sequence of \(x, y\) pairs \(shape \(2,\) or \(N, 2\)\)"):
        load_navigator_from_config(config)

def test_load_navigator_from_config_valid_multi_agent(mock_config_load):
    """Valid multi-agent config should construct Navigator without error."""
    from odor_plume_nav.utils.navigator_utils import load_navigator_from_config
    config = {"positions": [(0, 0), (1, 1)], "orientations": [0, 90]}
    nav = load_navigator_from_config(config)
    assert hasattr(nav, "positions")
    assert len(nav.positions) == 2

def test_load_navigator_from_config_valid_single_agent(mock_config_load):
    """Valid single-agent config should construct Navigator without error."""
    from odor_plume_nav.utils.navigator_utils import load_navigator_from_config
    config = {"position": (0, 0), "orientation": 0}
    nav = load_navigator_from_config(config)
    assert hasattr(nav, "positions")
    assert len(nav.positions) == 1

@pytest.mark.parametrize(
    "i,expected_pos",
    [(0, (10, 20)), (1, (30, 40)), (2, (50, 60))]
)
def test_create_navigator_position_index(i, expected_pos):
    positions = [(10, 20), (30, 40), (50, 60)]
    orientations = [45, 90, 135]
    speeds = [0.5, 0.7, 0.9]
    navigator = create_navigator(
        positions=positions,
        orientations=orientations,
        speeds=speeds
    )
    assert np.allclose(navigator.positions[i], expected_pos)

@pytest.mark.parametrize(
    "positions,expected_shape",
    [
        ([(1, 2), (3, 4)], (2, 2)),
        ([1, 2], (1, 2)),
        ((1, 2), (1, 2)),
        (np.array([[1, 2], [3, 4]]), (2, 2)),
    ]
)
def test_create_navigator_positions_shape_valid(positions, expected_shape):
    navigator = create_navigator(positions=positions)
    assert navigator.positions.shape == expected_shape

@pytest.mark.parametrize(
    "positions",
    [
        (np.array([[1], [2]])),
    ]
)
def test_create_navigator_positions_shape_invalid(positions):
    with pytest.raises(ValueError):
        create_navigator(positions=positions)
