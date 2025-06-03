"""Tests for VideoPlume factory functions with Hydra configuration integration."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_store
from hydra.core.config_store import ConfigStore

from {{cookiecutter.project_slug}}.api.navigation import create_video_plume_from_config
from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
from {{cookiecutter.project_slug}}.config.schemas import VideoPlumeConfig


@pytest.fixture
def hydra_config():
    """Create a Hydra DictConfig for testing."""
    config_dict = {
        "video_plume": {
            "flip": False,
            "kernel_size": 0,
            "kernel_sigma": 1.0,
        }
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def hydra_user_config():
    """Create a Hydra DictConfig with user overrides."""
    config_dict = {
        "video_plume": {
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.5,
        }
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def mock_cv2_videocapture():
    """Mock cv2.VideoCapture for testing."""
    with patch('{{cookiecutter.project_slug}}.data.video_plume.cv2.VideoCapture') as mock_cv:
        mock_instance = MagicMock()
        mock_cv.return_value = mock_instance
        
        # Configure video properties
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: {
            0: 640,  # CAP_PROP_FRAME_WIDTH
            1: 480,  # CAP_PROP_FRAME_HEIGHT
            5: 30.0,  # CAP_PROP_FPS
            7: 300,  # CAP_PROP_FRAME_COUNT
        }.get(prop, 0)
        
        # Configure frame reading
        mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        yield mock_cv


class TestVideoPlumeFactor:
    """Test suite for VideoPlume factory methods with Hydra integration."""

    def test_create_video_plume_with_hydra_config(self, hydra_config, mock_cv2_videocapture, mock_exists):
        """Test creating a VideoPlume with Hydra configuration."""
        video_path = "test_video.mp4"
        
        # Test the factory method with Hydra config
        plume = create_video_plume_from_config(video_path, config=hydra_config)
        
        # Verify the VideoPlume was created with correct settings
        assert plume.video_path == Path(video_path)
        assert plume.flip is False
        assert plume.kernel_size == 0
        assert plume.kernel_sigma == 1.0

    def test_create_video_plume_with_user_overrides(self, hydra_user_config, mock_cv2_videocapture, mock_exists):
        """Test creating a VideoPlume with user configuration overrides."""
        video_path = "test_video.mp4"
        
        plume = create_video_plume_from_config(video_path, config=hydra_user_config)
        
        # Check that user overrides were applied
        assert plume.video_path == Path(video_path)
        assert plume.flip is True  # Overridden in user config
        assert plume.kernel_size == 5  # Overridden in user config
        assert plume.kernel_sigma == 1.5  # Overridden in user config

    def test_create_video_plume_with_parameter_overrides(self, hydra_config, mock_cv2_videocapture, mock_exists):
        """Test creating a VideoPlume with explicit parameter overrides."""
        video_path = "test_video.mp4"
        
        # Override parameters directly in factory call
        plume = create_video_plume_from_config(
            video_path,
            config=hydra_config,
            flip=True,  # Override config
            kernel_size=3  # Override config
        )
        
        # Verify explicit parameters override config
        assert plume.video_path == Path(video_path)
        assert plume.flip is True  # Explicitly overridden
        assert plume.kernel_size == 3  # Explicitly overridden
        assert plume.kernel_sigma == 1.0  # From config (not overridden)

    def test_create_video_plume_from_config_dict(self, mock_cv2_videocapture, mock_exists):
        """Test creating VideoPlume from raw config dictionary."""
        video_path = "test_video.mp4"
        config_dict = {
            "video_plume": {
                "flip": True,
                "kernel_size": 7,
                "kernel_sigma": 2.0,
            }
        }
        
        plume = create_video_plume_from_config(
            video_path,
            config_dict=config_dict
        )
        
        assert plume.video_path == Path(video_path)
        assert plume.flip is True
        assert plume.kernel_size == 7
        assert plume.kernel_sigma == 2.0

    def test_video_plume_config_validation(self, mock_cv2_videocapture, mock_exists):
        """Test configuration validation using Pydantic schemas."""
        video_path = "test_video.mp4"
        
        # Test valid configuration
        valid_config = {
            "flip": False,
            "kernel_size": 5,
            "kernel_sigma": 1.5,
        }
        
        # Should pass validation
        validated_config = VideoPlumeConfig.model_validate(valid_config)
        assert validated_config.flip is False
        assert validated_config.kernel_size == 5
        assert validated_config.kernel_sigma == 1.5

    def test_video_plume_config_validation_errors(self):
        """Test configuration validation with invalid parameters."""
        # Test invalid kernel_size (negative)
        with pytest.raises(ValueError, match="kernel_size must be non-negative"):
            VideoPlumeConfig.model_validate({
                "flip": False,
                "kernel_size": -1,
                "kernel_sigma": 1.0,
            })
        
        # Test invalid kernel_sigma (non-positive)
        with pytest.raises(ValueError, match="kernel_sigma must be positive"):
            VideoPlumeConfig.model_validate({
                "flip": False,
                "kernel_size": 0,
                "kernel_sigma": 0.0,
            })

    def test_create_video_plume_with_merged_config(self, mock_cv2_videocapture, mock_exists):
        """Test creating VideoPlume with configuration merging."""
        video_path = "test_video.mp4"
        
        # Base configuration
        base_config = OmegaConf.create({
            "video_plume": {
                "flip": False,
                "kernel_size": 0,
                "kernel_sigma": 1.0,
            }
        })
        
        # User overrides (partial)
        user_overrides = OmegaConf.create({
            "video_plume": {
                "flip": True,
                # kernel_size not specified, should use base
                "kernel_sigma": 2.0,
            }
        })
        
        # Merge configurations
        merged_config = OmegaConf.merge(base_config, user_overrides)
        
        plume = create_video_plume_from_config(video_path, config=merged_config)
        
        # Verify merged settings
        assert plume.video_path == Path(video_path)
        assert plume.flip is True  # From user overrides
        assert plume.kernel_size == 0  # From base (not overridden)
        assert plume.kernel_sigma == 2.0  # From user overrides

    def test_video_plume_from_config_method(self, mock_cv2_videocapture, mock_exists):
        """Test VideoPlume.from_config class method with Hydra integration."""
        video_path = "test_video.mp4"
        
        config_dict = {
            "flip": True,
            "kernel_size": 3,
            "kernel_sigma": 1.5,
        }
        
        plume = VideoPlume.from_config(
            video_path=video_path,
            config_dict=config_dict
        )
        
        assert plume.video_path == Path(video_path)
        assert plume.flip is True
        assert plume.kernel_size == 3
        assert plume.kernel_sigma == 1.5

    def test_create_video_plume_parameter_precedence(self, hydra_config, mock_cv2_videocapture, mock_exists):
        """Test parameter precedence: kwargs > config > defaults."""
        video_path = "test_video.mp4"
        
        # Config has flip=False, kernel_size=0
        # We'll override flip via kwargs
        plume = create_video_plume_from_config(
            video_path,
            config=hydra_config,
            flip=True  # This should override config
        )
        
        assert plume.flip is True  # kwargs override
        assert plume.kernel_size == 0  # from config
        assert plume.kernel_sigma == 1.0  # from config

    def test_create_video_plume_with_environment_interpolation(self, mock_cv2_videocapture, mock_exists):
        """Test configuration with environment variable interpolation."""
        import os
        
        # Set environment variable
        os.environ['TEST_KERNEL_SIZE'] = '7'
        
        try:
            # Create config with environment interpolation
            config_dict = {
                "video_plume": {
                    "flip": False,
                    "kernel_size": "${oc.env:TEST_KERNEL_SIZE}",
                    "kernel_sigma": 1.0,
                }
            }
            config = OmegaConf.create(config_dict)
            
            video_path = "test_video.mp4"
            plume = create_video_plume_from_config(video_path, config=config)
            
            # Verify environment variable was interpolated
            assert plume.kernel_size == 7
            assert plume.flip is False
            assert plume.kernel_sigma == 1.0
            
        finally:
            # Clean up environment
            del os.environ['TEST_KERNEL_SIZE']

    def test_create_video_plume_error_handling(self, mock_cv2_videocapture):
        """Test error handling in factory method."""
        video_path = "nonexistent_video.mp4"
        
        # Mock file existence check to return False
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(IOError, match="Video file does not exist"):
                create_video_plume_from_config(video_path)

    def test_create_video_plume_opencv_failure(self, mock_exists):
        """Test handling of OpenCV failure."""
        video_path = "test_video.mp4"
        
        with patch('{{cookiecutter.project_slug}}.data.video_plume.cv2.VideoCapture') as mock_cv:
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = False  # Simulate failure
            
            with pytest.raises(IOError, match="Failed to open video file"):
                create_video_plume_from_config(video_path)

    def test_hydra_structured_config_integration(self):
        """Test integration with Hydra structured configs."""
        # This test verifies that our configuration can be used with Hydra's
        # structured config system for type safety
        
        config = VideoPlumeConfig(
            flip=True,
            kernel_size=5,
            kernel_sigma=1.5
        )
        
        # Convert to OmegaConf for Hydra compatibility
        omega_config = OmegaConf.structured(config)
        
        assert omega_config.flip is True
        assert omega_config.kernel_size == 5
        assert omega_config.kernel_sigma == 1.5
        
        # Verify it can be converted back
        reconstructed = VideoPlumeConfig(**omega_config)
        assert reconstructed.flip is True
        assert reconstructed.kernel_size == 5
        assert reconstructed.kernel_sigma == 1.5

    def test_factory_with_none_config(self, mock_cv2_videocapture, mock_exists):
        """Test factory method with None config (should use defaults)."""
        video_path = "test_video.mp4"
        
        plume = create_video_plume_from_config(video_path, config=None)
        
        # Should use default values
        assert plume.video_path == Path(video_path)
        assert plume.flip is False  # default
        assert plume.kernel_size == 0  # default
        assert plume.kernel_sigma == 1.0  # default

    def test_factory_performance_timing(self, hydra_config, mock_cv2_videocapture, mock_exists):
        """Test that factory method meets performance requirements."""
        import time
        
        video_path = "test_video.mp4"
        
        start_time = time.time()
        plume = create_video_plume_from_config(video_path, config=hydra_config)
        end_time = time.time()
        
        # Should complete in reasonable time (< 2 seconds as per requirements)
        assert (end_time - start_time) < 2.0
        assert plume is not None