"""
Shared fixtures for tests in the odor_plume_nav package.

This file contains pytest fixtures that are shared across multiple test files
to reduce duplication and ensure consistency.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


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
def temp_config_files(tmp_path):
    """
    Create temporary configuration files for testing.
    
    This fixture creates actual YAML files from the config_files fixture.
    
    Args:
        tmp_path: Pytest built-in fixture that provides a temporary directory
    
    Returns:
        Dictionary containing paths to the created files and the config objects
    """
    import yaml
    
    # Get configurations
    configs = config_files()
    
    # Create config directory
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    
    # Create default config file
    default_path = config_dir / "default.yaml"
    with open(default_path, 'w') as f:
        yaml.dump(configs["default_config"], f)
    
    # Create user config file
    user_path = config_dir / "user.yaml"
    with open(user_path, 'w') as f:
        yaml.dump(configs["user_config"], f)
    
    return {
        "config_dir": config_dir,
        "default_path": default_path,
        "user_path": user_path,
        "default_config": configs["default_config"],
        "user_config": configs["user_config"]
    }
