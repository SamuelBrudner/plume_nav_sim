"""
Tests for VideoPlume factory functions with enhanced Hydra configuration support.

This test module validates the video plume factory functionality within the new
plume_nav_sim.api.navigation module, including Hydra-based configuration
composition, parameter validation, and factory method patterns for video plume creation.

The tests cover:
- Legacy create_video_plume_from_config functionality
- Enhanced Hydra DictConfig integration
- Configuration validation and parameter merging
- Error handling for invalid configurations
- Factory method patterns for different use cases
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Import the new API structure
from plume_nav_sim.api.navigation import (
    create_video_plume,
    create_video_plume_from_config, 
    ConfigurationError
)
from plume_nav_sim.data.video_plume import VideoPlume
from plume_nav_sim.config.schemas import VideoPlumeConfig

# Try to import Hydra components for advanced testing
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict


class TestCreateVideoPlume:
    """Test suite for the create_video_plume function with direct parameters."""
    
    def test_create_video_plume_basic(self, mock_exists):
        """Test creating a VideoPlume with basic parameters."""
        with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture') as mock_cv:
            # Set up the mock video capture
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: {
                0: 640,  # CAP_PROP_FRAME_WIDTH
                1: 480,  # CAP_PROP_FRAME_HEIGHT 
                5: 30.0, # CAP_PROP_FPS
                7: 300,  # CAP_PROP_FRAME_COUNT
            }.get(prop, 0)
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Create a VideoPlume with basic parameters
            video_path = "test_video.mp4"
            plume = create_video_plume(video_path=video_path)
            
            # Verify the VideoPlume was created with default settings
            assert isinstance(plume, VideoPlume)
            assert plume.video_path == Path(video_path)
            assert plume.flip is False
            assert plume.kernel_size == 0
            assert plume.kernel_sigma == 1.0
    
    def test_create_video_plume_with_parameters(self, mock_exists):
        """Test creating a VideoPlume with custom parameters."""
        with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture') as mock_cv:
            # Set up the mock video capture
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: {
                0: 640, 1: 480, 5: 30.0, 7: 300
            }.get(prop, 0)
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Create a VideoPlume with custom parameters
            video_path = "test_video.mp4"
            plume = create_video_plume(
                video_path=video_path,
                flip=True,
                kernel_size=5,
                kernel_sigma=2.0
            )
            
            # Verify the VideoPlume was created with custom settings
            assert plume.video_path == Path(video_path)
            assert plume.flip is True
            assert plume.kernel_size == 5
            assert plume.kernel_sigma == 2.0
    
    def test_create_video_plume_missing_path(self):
        """Test that create_video_plume requires video_path."""
        with pytest.raises(ConfigurationError, match="video_path is required"):
            create_video_plume()
    
    def test_create_video_plume_file_not_found(self):
        """Test error handling for non-existent video file."""
        with pytest.raises(FileNotFoundError, match="Video file does not exist"):
            create_video_plume(video_path="nonexistent_video.mp4")
    
    def test_create_video_plume_invalid_kernel_size(self, mock_exists):
        """Test validation of kernel_size parameter."""
        with pytest.raises(ConfigurationError, match="kernel_size must be a positive integer"):
            create_video_plume(video_path="test_video.mp4", kernel_size=-1)
    
    def test_create_video_plume_invalid_kernel_sigma(self, mock_exists):
        """Test validation of kernel_sigma parameter."""
        with pytest.raises(ConfigurationError, match="kernel_sigma must be a positive number"):
            create_video_plume(video_path="test_video.mp4", kernel_sigma=-1.0)


class TestCreateVideoPlumeFromConfig:
    """Test suite for the create_video_plume_from_config function (legacy compatibility)."""
    
    def test_create_video_plume_with_default_config(self, config_files, mock_exists):
        """Test creating a VideoPlume with default configuration."""
        with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture') as mock_cv:
            # Set up the mock video capture
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: {
                0: 640, 1: 480, 5: 30.0, 7: 300
            }.get(prop, 0)
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Create configuration with video_path
            config = config_files["default_config"]["video_plume"].copy()
            config["video_path"] = "test_video.mp4"
            
            # Create a VideoPlume with default config
            plume = create_video_plume_from_config(config)
            
            # Check that the VideoPlume was created with default settings
            assert plume.video_path == Path("test_video.mp4")
            assert plume.flip is False
            assert plume.kernel_size == 0
            assert plume.kernel_sigma == 1.0
    
    def test_create_video_plume_with_user_config(self, config_files, mock_exists):
        """Test creating a VideoPlume with user configuration overrides."""
        with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture') as mock_cv:
            # Set up the mock video capture
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: {
                0: 640, 1: 480, 5: 30.0, 7: 300
            }.get(prop, 0)
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Create configuration with overrides and video_path
            config = config_files["user_config"]["video_plume"].copy()
            config["video_path"] = "test_video.mp4"
            
            # Create a VideoPlume with user config
            plume = create_video_plume_from_config(config)
            
            # Check that the VideoPlume was created with user settings
            assert plume.video_path == Path("test_video.mp4")
            assert plume.flip is True  # Overridden in user config
            assert plume.kernel_size == 5  # Overridden in user config
            assert plume.kernel_sigma == 1.0  # Default value since not overridden
    
    def test_create_video_plume_with_parameter_overrides(self, config_files, mock_exists):
        """Test creating a VideoPlume with config and direct parameter overrides."""
        with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture') as mock_cv:
            # Set up the mock video capture
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: {
                0: 640, 1: 480, 5: 30.0, 7: 300
            }.get(prop, 0)
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Create configuration
            config = config_files["default_config"]["video_plume"].copy()
            config["video_path"] = "test_video.mp4"
            
            # Create a VideoPlume with config and parameter overrides
            plume = create_video_plume_from_config(
                config,
                flip=True,  # Override the config
                kernel_sigma=3.0  # Override the config
            )
            
            # Check that the explicitly provided parameters override config
            assert plume.video_path == Path("test_video.mp4")
            assert plume.flip is True  # Explicitly overridden
            assert plume.kernel_size == 0  # From config
            assert plume.kernel_sigma == 3.0  # Explicitly overridden


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
class TestHydraConfigurationIntegration:
    """Test suite for enhanced Hydra-based factory method patterns."""
    
    def test_create_video_plume_with_dictconfig(self, mock_exists):
        """Test creating VideoPlume with Hydra DictConfig object."""
        with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture') as mock_cv:
            # Set up the mock video capture
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: {
                0: 640, 1: 480, 5: 30.0, 7: 300
            }.get(prop, 0)
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Create DictConfig
            config_dict = {
                "video_path": "test_video.mp4",
                "flip": True,
                "kernel_size": 5,
                "kernel_sigma": 1.5
            }
            cfg = DictConfig(config_dict)
            
            # Create VideoPlume using DictConfig
            plume = create_video_plume(cfg=cfg)
            
            # Verify configuration was applied correctly
            assert plume.video_path == Path("test_video.mp4")
            assert plume.flip is True
            assert plume.kernel_size == 5
            assert plume.kernel_sigma == 1.5
    
    def test_create_video_plume_with_dictconfig_overrides(self, mock_exists):
        """Test DictConfig with parameter overrides."""
        with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture') as mock_cv:
            # Set up the mock video capture
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: {
                0: 640, 1: 480, 5: 30.0, 7: 300
            }.get(prop, 0)
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Create DictConfig
            config_dict = {
                "video_path": "test_video.mp4",
                "flip": False,
                "kernel_size": 3
            }
            cfg = DictConfig(config_dict)
            
            # Create VideoPlume with overrides
            plume = create_video_plume(
                cfg=cfg,
                flip=True,  # Override config
                kernel_sigma=2.5  # Add new parameter
            )
            
            # Verify overrides took precedence
            assert plume.flip is True  # Overridden
            assert plume.kernel_size == 3  # From config
            assert plume.kernel_sigma == 2.5  # Direct parameter
    
    def test_create_video_plume_with_environment_interpolation(self, mock_exists, monkeypatch):
        """Test DictConfig with environment variable interpolation."""
        with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture') as mock_cv:
            # Set up the mock video capture
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: {
                0: 640, 1: 480, 5: 30.0, 7: 300
            }.get(prop, 0)
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Set environment variable
            monkeypatch.setenv("TEST_VIDEO_PATH", "test_video.mp4")
            
            # Create DictConfig with interpolation
            config_dict = {
                "video_path": "${oc.env:TEST_VIDEO_PATH}",
                "flip": True
            }
            cfg = DictConfig(config_dict)
            
            # Resolve interpolation manually for testing
            resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
            
            # Create VideoPlume
            plume = create_video_plume(cfg=resolved_cfg)
            
            # Verify interpolation worked
            assert plume.video_path == Path("test_video.mp4")
            assert plume.flip is True


class TestConfigurationValidationAndMerging:
    """Test suite for configuration validation and parameter merging functionality."""
    
    def test_configuration_schema_validation(self, mock_exists):
        """Test that configurations are validated using Pydantic schemas."""
        with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture') as mock_cv:
            # Set up the mock video capture
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: {
                0: 640, 1: 480, 5: 30.0, 7: 300
            }.get(prop, 0)
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Valid configuration
            config = {
                "video_path": "test_video.mp4",
                "flip": True,
                "kernel_size": 5,
                "kernel_sigma": 1.5
            }
            
            # This should not raise an error
            plume = create_video_plume(cfg=config)
            assert isinstance(plume, VideoPlume)
    
    def test_configuration_validation_failure(self):
        """Test that invalid configurations raise appropriate errors."""
        # Invalid kernel_size (even number)
        invalid_config = {
            "video_path": "test_video.mp4",
            "kernel_size": 4  # Invalid: must be odd
        }
        
        with pytest.raises(ConfigurationError):
            create_video_plume(cfg=invalid_config)
    
    def test_parameter_merging_precedence(self, mock_exists):
        """Test that direct parameters take precedence over configuration."""
        with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture') as mock_cv:
            # Set up the mock video capture
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: {
                0: 640, 1: 480, 5: 30.0, 7: 300
            }.get(prop, 0)
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Configuration
            config = {
                "video_path": "test_video.mp4",
                "flip": False,
                "kernel_size": 3,
                "kernel_sigma": 1.0
            }
            
            # Create with overrides
            plume = create_video_plume(
                cfg=config,
                flip=True,  # Should override config
                kernel_sigma=2.0  # Should override config
                # kernel_size not overridden, should use config value
            )
            
            # Verify precedence
            assert plume.flip is True  # Overridden
            assert plume.kernel_size == 3  # From config
            assert plume.kernel_sigma == 2.0  # Overridden
    
    def test_none_values_handling(self, mock_exists):
        """Test that None values in parameters don't override config."""
        with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture') as mock_cv:
            # Set up the mock video capture
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: {
                0: 640, 1: 480, 5: 30.0, 7: 300
            }.get(prop, 0)
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Configuration
            config = {
                "video_path": "test_video.mp4",
                "flip": True,
                "kernel_size": 5
            }
            
            # Create with None values (should not override)
            plume = create_video_plume(
                cfg=config,
                flip=None,  # Should not override
                kernel_sigma=None  # Should not override
            )
            
            # Verify None values didn't override config
            assert plume.flip is True  # From config, not overridden
            assert plume.kernel_size == 5  # From config
            assert plume.kernel_sigma == 1.0  # Default value


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge case scenarios."""
    
    def test_create_video_plume_opencv_failure(self, mock_exists):
        """Test handling of OpenCV VideoCapture failure."""
        with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture') as mock_cv:
            # Set up mock to simulate failure
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = False  # Simulate failure
            
            # This should raise an error during VideoPlume creation
            with pytest.raises(ConfigurationError, match="Failed to create VideoPlume"):
                create_video_plume(video_path="test_video.mp4")
    
    def test_create_video_plume_with_additional_kwargs(self, mock_exists):
        """Test that additional kwargs are passed through to VideoPlume."""
        with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture') as mock_cv:
            # Set up the mock video capture
            mock_instance = MagicMock()
            mock_cv.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: {
                0: 640, 1: 480, 5: 30.0, 7: 300
            }.get(prop, 0)
            mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Create VideoPlume with additional parameters
            plume = create_video_plume(
                video_path="test_video.mp4",
                flip=True,
                grayscale=False,  # Additional parameter
                normalize=False   # Additional parameter
            )
            
            # Verify additional parameters were set
            assert plume.flip is True
            assert plume.grayscale is False
            assert plume.normalize is False
    
    def test_legacy_config_path_handling(self):
        """Test that string/Path config arguments raise appropriate errors."""
        # Test string path (deprecated)
        with pytest.raises(ConfigurationError, match="File path configuration loading is deprecated"):
            create_video_plume_from_config("path/to/config.yaml")
        
        # Test Path object (deprecated)
        with pytest.raises(ConfigurationError, match="File path configuration loading is deprecated"):
            create_video_plume_from_config(Path("path/to/config.yaml"))


@pytest.fixture
def mock_hydra_config():
    """Fixture providing mock Hydra configuration for testing."""
    if HYDRA_AVAILABLE:
        config_dict = {
            "video_path": "test_plume.mp4",
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.5,
            "grayscale": True,
            "normalize": True
        }
        return DictConfig(config_dict)
    else:
        return {
            "video_path": "test_plume.mp4",
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.5,
            "grayscale": True,
            "normalize": True
        }


@pytest.fixture
def extended_config_files(config_files):
    """Extended configuration fixtures for video plume testing."""
    extended = config_files.copy()
    
    # Add video plume specific configurations
    extended["video_plume_basic"] = {
        "video_path": "basic_plume.mp4",
        "flip": False,
        "kernel_size": 0,
        "kernel_sigma": 1.0
    }
    
    extended["video_plume_advanced"] = {
        "video_path": "advanced_plume.mp4", 
        "flip": True,
        "kernel_size": 7,
        "kernel_sigma": 2.0,
        "grayscale": True,
        "normalize": True,
        "threshold": 0.5
    }
    
    return extended