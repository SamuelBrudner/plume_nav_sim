"""Tests for the VideoPlume class."""

import pytest
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from odor_plume_nav.video_plume import VideoPlume


@pytest.fixture
def mock_exists(monkeypatch):
    """Mock the Path.exists method to return True for all paths except 'nonexistent_file.mp4'."""
    def patched_exists(self):
        return str(self) != "nonexistent_file.mp4"
    
    monkeypatch.setattr(Path, "exists", patched_exists)
    return patched_exists


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
def failed_video_capture():
    """Create a mock for cv2.VideoCapture that fails to open."""
    with patch('cv2.VideoCapture') as mock_cap:
        # Configure the mock to return a failed instance
        mock_instance = MagicMock()
        mock_cap.return_value = mock_instance
        
        # Mock isOpened to return False
        mock_instance.isOpened.return_value = False
        
        yield mock_cap


def test_video_plume_loading(mock_video_capture, mock_exists):
    """Test that VideoPlume can be initialized with a valid path."""
    # Create a VideoPlume instance
    video_path = "dummy_video.mp4"
    plume = VideoPlume(video_path)
    
    # Check that cv2.VideoCapture was called with the correct path
    mock_video_capture.assert_called_once_with(video_path)
    
    # Check that plume properties were set correctly
    assert plume.video_path == Path(video_path)
    assert plume.frame_count == 100
    
    # Check observable behavior instead of implementation detail
    # A newly created plume should be able to get frames
    assert plume.get_frame(0) is not None


def test_nonexistent_file(mock_exists):
    """Test that VideoPlume raises IOError when file doesn't exist."""
    with pytest.raises(IOError, match="Video file does not exist"):
        VideoPlume("nonexistent_file.mp4")


def test_failed_open(failed_video_capture, mock_exists):
    """Test that VideoPlume raises IOError when video can't be opened."""
    with pytest.raises(IOError, match="Failed to open video file"):
        VideoPlume("failed_video.mp4")


def test_get_frame_valid_index(mock_video_capture, mock_exists):
    """Test that get_frame returns a frame for valid indices."""
    plume = VideoPlume("dummy_video.mp4")
    
    # Get a frame at index 50
    frame = plume.get_frame(50)
    
    # Check that the frame was retrieved and converted to grayscale
    assert frame is not None
    mock_video_capture.return_value.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 50)
    mock_video_capture.return_value.read.assert_called_once()


def test_get_frame_invalid_index(mock_video_capture, mock_exists):
    """Test that get_frame returns None for invalid indices."""
    plume = VideoPlume("dummy_video.mp4")
    
    # Try to get frames with invalid indices
    negative_frame = plume.get_frame(-1)
    assert negative_frame is None
    
    too_large_frame = plume.get_frame(200)  # Beyond frame_count
    assert too_large_frame is None


def test_get_frame_after_close(mock_video_capture, mock_exists):
    """Test that get_frame raises ValueError after VideoPlume is closed."""
    plume = VideoPlume("dummy_video.mp4")
    
    # Close the plume
    plume.close()
    
    # Try to get a frame after closing
    with pytest.raises(ValueError, match="Cannot get frame from closed VideoPlume"):
        plume.get_frame(0)


def test_close_idempotent(mock_video_capture, mock_exists):
    """Test that calling close() multiple times is safe."""
    plume = VideoPlume("dummy_video.mp4")
    
    # Call close multiple times
    plume.close()
    
    # Verify closed state through behavior instead of internal state
    with pytest.raises(ValueError, match="Cannot get frame from closed VideoPlume"):
        plume.get_frame(0)
    
    # Second close should not raise any errors
    plume.close()  
    
    # Still closed after second close
    with pytest.raises(ValueError, match="Cannot get frame from closed VideoPlume"):
        plume.get_frame(0)


def test_frame_metadata(mock_video_capture, mock_exists):
    """Test that frame metadata properties are correctly exposed."""
    plume = VideoPlume("dummy_video.mp4")
    
    # Check metadata properties
    assert plume.width == 640
    assert plume.height == 480
    assert plume.fps == 30.0
    assert plume.shape == (480, 640)
    assert plume.duration == pytest.approx(100 / 30.0)
    
    # Test the get_metadata method
    metadata = plume.get_metadata()
    assert metadata["width"] == 640
    assert metadata["height"] == 480
    assert metadata["fps"] == 30.0
    assert metadata["frame_count"] == 100
    assert metadata["duration"] == pytest.approx(100 / 30.0)
    assert metadata["shape"] == (480, 640)


def test_frame_flip(mock_video_capture, mock_exists):
    """Test that frames can be flipped horizontally."""
    # Create a synthetic colored frame (BGR)
    colored_frame = np.zeros((2, 3, 3), dtype=np.uint8)
    colored_frame[0, 0] = [255, 0, 0]  # Set pixel at (0,0) to blue in BGR
    
    # Expected resulting frames
    grayscale_frame = np.zeros((2, 3), dtype=np.uint8)
    grayscale_frame[0, 0] = 255  # White at (0,0)
    
    flipped_colored = np.zeros((2, 3, 3), dtype=np.uint8)
    flipped_colored[0, 2] = [255, 0, 0]  # Blue at (0,2) after flip
    
    flipped_grayscale = np.zeros((2, 3), dtype=np.uint8)
    flipped_grayscale[0, 2] = 255  # White at (0,2) after flip
    
    # Configure the mock to return the colored frame
    mock_instance = mock_video_capture.return_value
    mock_instance.read.return_value = (True, colored_frame)
    
    # Test with flip=False (default)
    with patch('cv2.cvtColor', return_value=grayscale_frame):
        video_plume = VideoPlume("test.mp4")
        frame_no_flip = video_plume.get_frame(0)
        assert frame_no_flip[0, 0] == 255
    
    # Test with flip=True
    video_plume_flip = VideoPlume("test.mp4", flip=True)
    
    # Create a chained mock that simulates both operations
    # First the flip occurs, then the color conversion
    with patch('cv2.flip', return_value=flipped_colored) as mock_flip:
        with patch('cv2.cvtColor', return_value=flipped_grayscale):
            frame_flip = video_plume_flip.get_frame(0)
            mock_flip.assert_called_once_with(colored_frame, 1)
            assert frame_flip[0, 2] == 255  # Pixel moved from (0,0) to (0,2) due to flip
            assert frame_flip[0, 0] == 0    # Original position is now empty


@pytest.fixture
def mock_config_file():
    """Mock the load_config function to return a test configuration."""
    with patch('odor_plume_nav.video_plume.load_config') as mock_load:
        mock_load.return_value = {
            "video_plume": {
                "flip": True,
                "kernel_size": 3,
                "kernel_sigma": 1.5
            }
        }
        yield mock_load


def test_from_config_with_dict(mock_video_capture, mock_exists):
    """Test creating VideoPlume with the from_config method using a dictionary."""
    config_dict = {
        "flip": True,
        "kernel_size": 5,
        "kernel_sigma": 2.0
    }
    
    # Create VideoPlume using from_config with a dictionary
    video_plume = VideoPlume.from_config("test.mp4", config_dict=config_dict)
    
    # Verify configuration was applied
    assert video_plume.flip is True
    assert video_plume.kernel_size == 5
    assert video_plume.kernel_sigma == 2.0
    assert video_plume.video_path == Path("test.mp4")


def test_from_config_with_file(mock_video_capture, mock_exists, mock_config_file):
    """Test creating VideoPlume with the from_config method using a config file."""
    # Create VideoPlume using from_config with a config file path
    video_plume = VideoPlume.from_config("test.mp4", config_path="config.yaml")
    
    # Verify configuration from the file was applied
    assert video_plume.flip is True
    assert video_plume.kernel_size == 3
    assert video_plume.kernel_sigma == 1.5
    
    # Verify the config file was loaded
    mock_config_file.assert_called_once_with("config.yaml")


def test_from_config_with_kwargs_override(mock_video_capture, mock_exists, mock_config_file):
    """Test creating VideoPlume with from_config where kwargs override config."""
    # Create config dictionary
    config_dict = {
        "flip": False,
        "kernel_size": 5,
        "kernel_sigma": 2.0
    }
    
    # Create VideoPlume with config_dict and overriding kwargs
    video_plume = VideoPlume.from_config(
        "test.mp4", 
        config_dict=config_dict,
        config_path="config.yaml",  # Should load but be overridden by config_dict and kwargs
        flip=True,  # Should override config_dict
        kernel_size=7  # Should override config_dict
    )
    
    # Verify kwargs took precedence over config_dict and file
    assert video_plume.flip is True  # From kwargs
    assert video_plume.kernel_size == 7  # From kwargs
    assert video_plume.kernel_sigma == 2.0  # From config_dict (not overridden)


def test_from_config_validation(mock_video_capture, mock_exists):
    """Test validation in from_config method."""
    # Test with invalid kernel_size (negative)
    with pytest.raises(ValueError, match="kernel_size must be non-negative"):
        VideoPlume.from_config("test.mp4", config_dict={"kernel_size": -1})
    
    # Test with invalid kernel_sigma (zero)
    with pytest.raises(ValueError, match="kernel_sigma must be positive"):
        VideoPlume.from_config("test.mp4", config_dict={"kernel_sigma": 0})
    
    # Test with invalid video_path type
    with pytest.raises(ValueError):
        VideoPlume.from_config(123)  # video_path should be string or Path
