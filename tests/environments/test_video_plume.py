"""Tests for the VideoPlume environment."""

import pytest
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

# Import from the new environments module
from odor_plume_nav.environments import VideoPlume


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
    
    # Close the video plume
    plume.close()
    
    # Try to get a frame after closing, should raise ValueError
    with pytest.raises(ValueError, match="VideoPlume is closed"):
        plume.get_frame(0)


def test_close_idempotent(mock_video_capture, mock_exists):
    """Test that calling close() multiple times is safe."""
    plume = VideoPlume("dummy_video.mp4")
    
    # Close once
    plume.close()
    
    # Verify closed state through behavior instead of internal state
    with pytest.raises(ValueError, match="VideoPlume is closed"):
        plume.get_frame(0)
    
    # Close again, should not raise any errors
    plume.close()
    
    # Still closed after second close
    with pytest.raises(ValueError, match="VideoPlume is closed"):
        plume.get_frame(0)


def test_frame_metadata(mock_video_capture, mock_exists):
    """Test that frame metadata properties are correctly exposed."""
    plume = VideoPlume("dummy_video.mp4")
    
    # Check basic metadata
    assert plume.frame_count == 100
    assert plume.width == 640
    assert plume.height == 480
    assert plume.fps == 30.0
    
    # Test format string conversion
    metadata_str = plume.get_metadata_string()
    assert "100 frames" in metadata_str
    assert "640x480" in metadata_str
    assert "30.0 fps" in metadata_str


@pytest.fixture
def mock_config_file():
    """Mock the load_config function to return a test configuration."""
    # Use the new config utils path
    with patch('odor_plume_nav.config.utils.load_config') as mock_load:
        mock_load.return_value = {
            "video_plume": {
                "flip": True,
                "kernel_size": 3,
                "kernel_sigma": 1.5
            }
        }
        yield mock_load
