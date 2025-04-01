"""Tests for VideoPlume factory functions."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from odor_plume_nav.video_plume_factory import create_video_plume_from_config


def test_create_video_plume_with_default_config(config_files, mock_exists):
    """Test creating a VideoPlume with default configuration."""
    with patch('odor_plume_nav.video_plume_factory.load_config', 
               return_value=config_files["default_config"]), \
         patch('odor_plume_nav.video_plume.cv2.VideoCapture') as mock_cv:
        
        # Set up the mock video capture to return valid video properties
        mock_instance = MagicMock()
        mock_cv.return_value = mock_instance
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Create a VideoPlume with default config
        video_path = "test_video.mp4"
        plume = create_video_plume_from_config(video_path)
        
        # Check that the VideoPlume was created with default settings
        assert plume.video_path == Path(video_path)
        assert plume.flip is False
        assert plume.kernel_size == 0
        assert plume.kernel_sigma == 1.0


def test_create_video_plume_with_user_config(config_files, mock_exists):
    """Test creating a VideoPlume with user configuration overrides."""
    with patch('odor_plume_nav.video_plume_factory.load_config', 
               return_value=config_files["user_config"]), \
         patch('odor_plume_nav.video_plume.cv2.VideoCapture') as mock_cv:
        
        # Set up the mock video capture to return valid video properties
        mock_instance = MagicMock()
        mock_cv.return_value = mock_instance
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Create a VideoPlume with user config
        video_path = "test_video.mp4"
        plume = create_video_plume_from_config(video_path)
        
        # Check that the VideoPlume was created with user settings
        assert plume.video_path == Path(video_path)
        assert plume.flip is True  # Overridden in user config
        assert plume.kernel_size == 5  # Overridden in user config
        assert plume.kernel_sigma == 1.0  # Not overridden, should use default


def test_create_video_plume_with_merged_config(config_files, mock_exists):
    """Test creating a VideoPlume with merged configuration."""
    # Create a merged config by combining default and parts of user config
    merged_config = config_files["default_config"].copy()
    merged_config["video_plume"]["flip"] = True  # Override just the flip parameter
    
    with patch('odor_plume_nav.video_plume_factory.load_config', 
               return_value=merged_config), \
         patch('odor_plume_nav.video_plume.cv2.VideoCapture') as mock_cv:
        
        # Set up the mock video capture to return valid video properties
        mock_instance = MagicMock()
        mock_cv.return_value = mock_instance
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Create a VideoPlume with merged config
        video_path = "test_video.mp4"
        plume = create_video_plume_from_config(video_path)
        
        # Check that the VideoPlume was created with merged settings
        assert plume.video_path == Path(video_path)
        assert plume.flip is True  # Overridden
        assert plume.kernel_size == 0  # Not overridden, using default
        assert plume.kernel_sigma == 1.0  # Not overridden, using default


def test_create_video_plume_with_additional_params(config_files, mock_exists):
    """Test creating a VideoPlume with additional parameters."""
    with patch('odor_plume_nav.video_plume_factory.load_config', 
               return_value=config_files["default_config"]), \
         patch('odor_plume_nav.video_plume.cv2.VideoCapture') as mock_cv:
        
        # Set up the mock video capture to return valid video properties
        mock_instance = MagicMock()
        mock_cv.return_value = mock_instance
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Create a VideoPlume with default config but override some params
        video_path = "test_video.mp4"
        plume = create_video_plume_from_config(
            video_path,
            flip=True,  # Override the config
            additional_param="test"  # Parameter not in config
        )
        
        # Check that the explicitly provided parameters override config
        assert plume.video_path == Path(video_path)
        assert plume.flip is True  # Explicitly overridden
        assert plume.kernel_size == 0  # From config
        
        # This should test that additional_param was passed to VideoPlume
        # But we'd need to modify VideoPlume to accept and store additional_param
        # For now, we're just testing the explicitly overridden parameters
