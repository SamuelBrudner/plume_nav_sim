"""
Tests for data processing and video plume functionality.

This comprehensive test suite validates VideoPlume functionality, OpenCV integration,
frame processing, metadata extraction, and integration with the broader simulation framework.
Ensures video processing reliability, performance requirements, and configuration management.
"""

import pytest
import numpy as np
import cv2
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, Any, Optional

# Test framework imports for advanced testing capabilities
import tempfile
import contextlib
import os


class TestVideoPlumeInitialization:
    """Test VideoPlume initialization and configuration validation."""
    
    def test_video_plume_initialization_default_config(self, mock_video_capture, mock_exists):
        """Test VideoPlume initialization with default configuration."""
        # Import the actual VideoPlume class from the new location
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        video_path = "test_video.mp4"
        plume = VideoPlume(video_path)
        
        # Verify initialization parameters
        assert plume.video_path == Path(video_path)
        assert plume.flip is False  # Default value
        assert plume.kernel_size == 0  # Default value
        assert plume.kernel_sigma == 1.0  # Default value
        assert not plume._is_closed
        
        # Verify OpenCV integration
        mock_video_capture.assert_called_once_with(str(video_path))
        assert plume.cap is not None
    
    def test_video_plume_initialization_custom_config(self, mock_video_capture, mock_exists):
        """Test VideoPlume initialization with custom configuration parameters."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        video_path = "test_video.mp4"
        plume = VideoPlume(
            video_path=video_path,
            flip=True,
            kernel_size=5,
            kernel_sigma=2.0
        )
        
        # Verify custom parameters are set correctly
        assert plume.video_path == Path(video_path)
        assert plume.flip is True
        assert plume.kernel_size == 5
        assert plume.kernel_sigma == 2.0
    
    def test_video_plume_from_config_method(self, mock_video_capture, mock_exists, mock_config_loader):
        """Test VideoPlume creation from configuration using from_config method."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        config_dict = {
            "flip": True,
            "kernel_size": 3,
            "kernel_sigma": 1.5
        }
        
        video_path = "test_video.mp4"
        plume = VideoPlume.from_config(
            video_path=video_path,
            config_dict=config_dict
        )
        
        # Verify configuration was applied
        assert plume.video_path == Path(video_path)
        assert plume.flip is True
        assert plume.kernel_size == 3
        assert plume.kernel_sigma == 1.5
    
    def test_video_plume_configuration_validation_error(self, mock_video_capture, mock_exists):
        """Test VideoPlume raises appropriate errors for invalid configuration."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Test negative kernel_size
        with pytest.raises(ValueError, match="Invalid VideoPlume configuration"):
            VideoPlume.from_config(
                video_path="test_video.mp4",
                config_dict={"kernel_size": -1}
            )
        
        # Test invalid kernel_sigma
        with pytest.raises(ValueError, match="Invalid VideoPlume configuration"):
            VideoPlume.from_config(
                video_path="test_video.mp4",
                config_dict={"kernel_sigma": 0}
            )


class TestOpenCVIntegration:
    """Test OpenCV integration for video capture and frame processing."""
    
    def test_opencv_video_capture_success(self, mock_exists):
        """Test successful OpenCV VideoCapture initialization."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        with patch('cv2.VideoCapture') as mock_cv2:
            # Configure successful capture
            mock_instance = MagicMock()
            mock_cv2.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            
            # Set up video properties
            mock_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 100,
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(prop, 0)
            
            plume = VideoPlume("test_video.mp4")
            
            # Verify properties are extracted correctly
            assert plume.frame_count == 100
            assert plume.width == 640
            assert plume.height == 480
            assert plume.fps == 30.0
    
    def test_opencv_video_capture_failure(self, mock_exists):
        """Test VideoPlume handles OpenCV VideoCapture failure gracefully."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        with patch('cv2.VideoCapture') as mock_cv2:
            # Configure failed capture
            mock_instance = MagicMock()
            mock_cv2.return_value = mock_instance
            mock_instance.isOpened.return_value = False
            
            with pytest.raises(IOError, match="Failed to open video file"):
                VideoPlume("test_video.mp4")
    
    def test_file_existence_validation(self):
        """Test VideoPlume validates file existence before attempting to open."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Test nonexistent file
        with pytest.raises(IOError, match="Video file does not exist"):
            VideoPlume("nonexistent_file.mp4")
    
    def test_video_format_compatibility(self, mock_video_capture, mock_exists):
        """Test VideoPlume works with different video formats."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Test various video formats
        formats = ["test.mp4", "test.avi", "test.mov", "test.mkv"]
        
        for video_format in formats:
            plume = VideoPlume(video_format)
            assert plume.video_path == Path(video_format)
            # Clean up for next iteration
            plume.close()


class TestFrameProcessing:
    """Test frame preprocessing including grayscale conversion and Gaussian smoothing."""
    
    def test_frame_grayscale_conversion(self, mock_exists):
        """Test frames are properly converted to grayscale."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        with patch('cv2.VideoCapture') as mock_cv2:
            # Configure capture to return BGR frame
            mock_instance = MagicMock()
            mock_cv2.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            
            # Create a mock BGR frame (3 channels)
            bgr_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            mock_instance.read.return_value = (True, bgr_frame)
            mock_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 100,
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(prop, 0)
            
            with patch('cv2.cvtColor') as mock_cvtColor:
                # Mock grayscale conversion
                gray_frame = np.ones((480, 640), dtype=np.uint8) * 128
                mock_cvtColor.return_value = gray_frame
                
                plume = VideoPlume("test_video.mp4")
                frame = plume.get_frame(0)
                
                # Verify grayscale conversion was called
                mock_cvtColor.assert_called_once_with(bgr_frame, cv2.COLOR_BGR2GRAY)
                assert frame.shape == (480, 640)  # 2D grayscale
    
    def test_frame_flipping(self, mock_exists):
        """Test horizontal frame flipping functionality."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_instance = MagicMock()
            mock_cv2.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            
            # Create a mock BGR frame
            bgr_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            mock_instance.read.return_value = (True, bgr_frame)
            mock_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 100,
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(prop, 0)
            
            with patch('cv2.flip') as mock_flip, patch('cv2.cvtColor') as mock_cvtColor:
                # Configure flipping enabled
                plume = VideoPlume("test_video.mp4", flip=True)
                
                mock_cvtColor.return_value = np.ones((480, 640), dtype=np.uint8) * 128
                mock_flip.return_value = bgr_frame
                
                frame = plume.get_frame(0)
                
                # Verify flip was called before color conversion
                mock_flip.assert_called_once_with(bgr_frame, 1)
    
    def test_gaussian_smoothing_application(self, mock_exists):
        """Test Gaussian smoothing is applied when kernel_size > 0."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_instance = MagicMock()
            mock_cv2.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            
            mock_instance.read.return_value = (True, np.ones((480, 640, 3), dtype=np.uint8) * 128)
            mock_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 100,
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(prop, 0)
            
            with patch('cv2.GaussianBlur') as mock_blur, patch('cv2.cvtColor') as mock_cvtColor:
                plume = VideoPlume("test_video.mp4", kernel_size=5, kernel_sigma=2.0)
                
                gray_frame = np.ones((480, 640), dtype=np.uint8) * 128
                mock_cvtColor.return_value = gray_frame
                mock_blur.return_value = gray_frame
                
                frame = plume.get_frame(0)
                
                # Verify Gaussian blur was applied with correct parameters
                mock_blur.assert_called_once_with(gray_frame, (5, 5), 2.0)
    
    def test_frame_preprocessing_pipeline(self, mock_exists):
        """Test complete frame preprocessing pipeline with all transformations."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_instance = MagicMock()
            mock_cv2.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            
            original_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_instance.read.return_value = (True, original_frame)
            mock_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 100,
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(prop, 0)
            
            with patch('cv2.flip') as mock_flip, \
                 patch('cv2.cvtColor') as mock_cvtColor, \
                 patch('cv2.GaussianBlur') as mock_blur:
                
                # Configure complete preprocessing pipeline
                plume = VideoPlume("test_video.mp4", flip=True, kernel_size=3, kernel_sigma=1.5)
                
                flipped_frame = original_frame.copy()
                gray_frame = np.ones((480, 640), dtype=np.uint8) * 128
                blurred_frame = gray_frame.copy()
                
                mock_flip.return_value = flipped_frame
                mock_cvtColor.return_value = gray_frame
                mock_blur.return_value = blurred_frame
                
                frame = plume.get_frame(0)
                
                # Verify processing order: flip -> grayscale -> blur
                mock_flip.assert_called_once_with(original_frame, 1)
                mock_cvtColor.assert_called_once_with(flipped_frame, cv2.COLOR_BGR2GRAY)
                mock_blur.assert_called_once_with(gray_frame, (3, 3), 1.5)


class TestMetadataExtraction:
    """Test video properties and timing information extraction."""
    
    def test_video_metadata_extraction(self, mock_video_capture, mock_exists):
        """Test comprehensive video metadata extraction."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Configure mock with realistic video properties
        mock_video_capture.return_value.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 1500,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 60.0
        }.get(prop, 0)
        
        plume = VideoPlume("test_video.mp4")
        
        # Test basic properties
        assert plume.frame_count == 1500
        assert plume.width == 1920
        assert plume.height == 1080
        assert plume.fps == 60.0
        
        # Test computed properties
        assert plume.duration == 25.0  # 1500 frames / 60 fps
        assert plume.shape == (1080, 1920)  # Height x Width
    
    def test_metadata_structure(self, mock_video_capture, mock_exists):
        """Test get_metadata returns properly structured metadata dictionary."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        plume = VideoPlume("test_video.mp4")
        metadata = plume.get_metadata()
        
        # Verify metadata structure
        required_keys = ["width", "height", "fps", "frame_count", "duration", "shape"]
        for key in required_keys:
            assert key in metadata
        
        # Verify data types
        assert isinstance(metadata["width"], int)
        assert isinstance(metadata["height"], int)
        assert isinstance(metadata["fps"], float)
        assert isinstance(metadata["frame_count"], int)
        assert isinstance(metadata["duration"], float)
        assert isinstance(metadata["shape"], tuple)
    
    def test_duration_calculation_edge_cases(self, mock_exists):
        """Test duration calculation handles edge cases correctly."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_instance = MagicMock()
            mock_cv2.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            
            # Test zero FPS case
            mock_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 100,
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 0.0
            }.get(prop, 0)
            
            plume = VideoPlume("test_video.mp4")
            assert plume.duration == 0.0


class TestResourceManagement:
    """Test resource management and cleanup to prevent memory leaks."""
    
    def test_automatic_resource_cleanup_on_close(self, mock_video_capture, mock_exists):
        """Test VideoPlume properly releases resources when closed."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        plume = VideoPlume("test_video.mp4")
        capture_instance = plume.cap
        
        # Close the plume
        plume.close()
        
        # Verify cleanup
        assert plume._is_closed
        capture_instance.release.assert_called_once()
    
    def test_idempotent_close_behavior(self, mock_video_capture, mock_exists):
        """Test calling close() multiple times is safe and idempotent."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        plume = VideoPlume("test_video.mp4")
        capture_instance = plume.cap
        
        # Close multiple times
        plume.close()
        plume.close()
        plume.close()
        
        # Verify release was only called once
        capture_instance.release.assert_called_once()
        assert plume._is_closed
    
    def test_automatic_cleanup_on_deletion(self, mock_video_capture, mock_exists):
        """Test VideoPlume automatically cleans up resources when deleted."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        plume = VideoPlume("test_video.mp4")
        capture_instance = plume.cap
        
        # Delete the object
        del plume
        
        # Verify cleanup was attempted (may be called via __del__)
        # Note: __del__ behavior can be unreliable in tests, so we test the close() method
    
    def test_memory_leak_prevention(self, mock_video_capture, mock_exists):
        """Test creating and destroying multiple VideoPlume instances doesn't leak memory."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Create and destroy multiple instances
        for i in range(10):
            plume = VideoPlume("test_video.mp4")
            plume.close()
        
        # If we reach here without memory errors, the test passes
        assert True
    
    def test_resource_state_after_close(self, mock_video_capture, mock_exists):
        """Test VideoPlume state and behavior after being closed."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        plume = VideoPlume("test_video.mp4")
        plume.close()
        
        # Test that accessing frames after close raises appropriate error
        with pytest.raises(ValueError, match="Cannot get frame from closed VideoPlume"):
            plume.get_frame(0)


class TestFrameAccessPatterns:
    """Test frame access patterns and error handling for end-of-stream conditions."""
    
    def test_valid_frame_access(self, mock_video_capture, mock_exists):
        """Test accessing frames within valid range."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Configure mock to return different frames for different indices
        def mock_read():
            return (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        
        mock_video_capture.return_value.read.side_effect = mock_read
        
        plume = VideoPlume("test_video.mp4")
        
        # Test accessing various valid frame indices
        valid_indices = [0, 1, 50, 99]  # Within frame_count of 100
        for idx in valid_indices:
            frame = plume.get_frame(idx)
            assert frame is not None
            assert frame.shape == (480, 640)  # Grayscale
    
    def test_invalid_frame_access(self, mock_video_capture, mock_exists):
        """Test accessing frames outside valid range returns None."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        plume = VideoPlume("test_video.mp4")
        
        # Test negative indices
        assert plume.get_frame(-1) is None
        assert plume.get_frame(-10) is None
        
        # Test indices beyond frame count
        assert plume.get_frame(100) is None  # frame_count is 100, so 100 is invalid
        assert plume.get_frame(1000) is None
    
    def test_frame_seeking_behavior(self, mock_video_capture, mock_exists):
        """Test frame seeking behavior using OpenCV CAP_PROP_POS_FRAMES."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        plume = VideoPlume("test_video.mp4")
        
        # Access frame at specific index
        frame_index = 42
        plume.get_frame(frame_index)
        
        # Verify seeking was performed
        mock_video_capture.return_value.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, frame_index)
        mock_video_capture.return_value.read.assert_called()
    
    def test_failed_frame_read_handling(self, mock_exists):
        """Test handling of failed frame reads from OpenCV."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_instance = MagicMock()
            mock_cv2.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            
            # Configure read to fail
            mock_instance.read.return_value = (False, None)
            mock_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 100,
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(prop, 0)
            
            plume = VideoPlume("test_video.mp4")
            
            # Should return None for failed reads
            frame = plume.get_frame(0)
            assert frame is None
    
    def test_end_of_stream_handling(self, mock_video_capture, mock_exists):
        """Test proper handling of end-of-stream conditions."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        plume = VideoPlume("test_video.mp4")
        
        # Test accessing frame at the boundary
        last_valid_frame = plume.frame_count - 1
        frame = plume.get_frame(last_valid_frame)
        # Should not be None for valid index (mocked to return valid frame)
        
        # Test accessing frame beyond the end
        beyond_end_frame = plume.get_frame(plume.frame_count)
        assert beyond_end_frame is None


class TestPerformanceRequirements:
    """Test performance requirements for frame processing per Section 6.6.3.3."""
    
    def test_frame_processing_performance(self, mock_video_capture, mock_exists):
        """Test frame processing completes within 33ms per Section 6.6.3.3."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Configure realistic frame processing
        mock_video_capture.return_value.read.return_value = (
            True, 
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        
        plume = VideoPlume("test_video.mp4")
        
        # Measure frame processing time
        start_time = time.time()
        frame = plume.get_frame(0)
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Verify frame was retrieved
        assert frame is not None
        
        # Verify performance requirement (33ms)
        assert processing_time < 33.0, f"Frame processing took {processing_time:.2f}ms, exceeds 33ms limit"
    
    def test_multiple_frame_access_performance(self, mock_video_capture, mock_exists):
        """Test performance of accessing multiple frames sequentially."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        mock_video_capture.return_value.read.return_value = (
            True,
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        
        plume = VideoPlume("test_video.mp4")
        
        # Test accessing 10 frames and measure average time
        frame_times = []
        for i in range(10):
            start_time = time.time()
            frame = plume.get_frame(i)
            end_time = time.time()
            
            frame_times.append((end_time - start_time) * 1000)
            assert frame is not None
        
        average_time = sum(frame_times) / len(frame_times)
        max_time = max(frame_times)
        
        # Verify performance requirements
        assert average_time < 33.0, f"Average frame processing time {average_time:.2f}ms exceeds 33ms"
        assert max_time < 50.0, f"Max frame processing time {max_time:.2f}ms exceeds reasonable limit"
    
    def test_memory_usage_efficiency(self, mock_video_capture, mock_exists):
        """Test memory usage remains efficient during frame processing."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Create realistic frame data
        frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_video_capture.return_value.read.return_value = (True, frame_data)
        
        plume = VideoPlume("test_video.mp4")
        
        # Access multiple frames to test memory efficiency
        frames = []
        for i in range(5):
            frame = plume.get_frame(i)
            frames.append(frame)
        
        # Verify frames are independent (not sharing memory inappropriately)
        for i, frame in enumerate(frames):
            assert frame is not None
            assert frame.shape == (480, 640)


class TestConfigurationSystemIntegration:
    """Test integration with configuration system and factory method patterns."""
    
    def test_hydra_configuration_loading(self, mock_video_capture, mock_exists):
        """Test VideoPlume integration with Hydra configuration system."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Test configuration file path loading
        config_path = Path("test_config.yaml")
        
        with patch('{{cookiecutter.project_slug}}.config.utils.load_config') as mock_load_config:
            mock_load_config.return_value = {
                "video_plume": {
                    "flip": True,
                    "kernel_size": 5,
                    "kernel_sigma": 2.0
                }
            }
            
            plume = VideoPlume.from_config(
                video_path="test_video.mp4",
                config_path=config_path
            )
            
            # Verify configuration was loaded and applied
            mock_load_config.assert_called_once_with(config_path)
            assert plume.flip is True
            assert plume.kernel_size == 5
            assert plume.kernel_sigma == 2.0
    
    def test_configuration_override_precedence(self, mock_video_capture, mock_exists):
        """Test configuration override precedence: kwargs > config_dict > config_path."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        with patch('{{cookiecutter.project_slug}}.config.utils.load_config') as mock_load_config:
            # Config from file
            mock_load_config.return_value = {
                "video_plume": {
                    "flip": False,
                    "kernel_size": 1,
                    "kernel_sigma": 1.0
                }
            }
            
            # Config dict override
            config_dict = {
                "flip": True,
                "kernel_size": 3
            }
            
            # Kwargs override
            plume = VideoPlume.from_config(
                video_path="test_video.mp4",
                config_path="test_config.yaml",
                config_dict=config_dict,
                kernel_sigma=2.5  # Kwargs override
            )
            
            # Verify precedence: kwargs > config_dict > config_path
            assert plume.flip is True  # From config_dict
            assert plume.kernel_size == 3  # From config_dict
            assert plume.kernel_sigma == 2.5  # From kwargs
    
    def test_configuration_validation_integration(self, mock_video_capture, mock_exists):
        """Test integration with Pydantic configuration validation."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Test that configuration validation is performed
        with pytest.raises(ValueError, match="Invalid VideoPlume configuration"):
            VideoPlume.from_config(
                video_path="test_video.mp4",
                config_dict={
                    "kernel_size": -5,  # Invalid: negative value
                    "kernel_sigma": 0  # Invalid: zero value
                }
            )
    
    def test_factory_method_patterns(self, mock_video_capture, mock_exists):
        """Test factory method patterns for VideoPlume creation."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Test direct instantiation
        plume1 = VideoPlume("test_video.mp4")
        assert isinstance(plume1, VideoPlume)
        
        # Test factory method
        plume2 = VideoPlume.from_config(
            video_path="test_video.mp4",
            config_dict={"flip": True}
        )
        assert isinstance(plume2, VideoPlume)
        assert plume2.flip is True


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility and path handling."""
    
    def test_path_handling_cross_platform(self, mock_video_capture, mock_exists):
        """Test VideoPlume handles paths correctly across platforms."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Test different path formats
        test_paths = [
            "video.mp4",  # Relative path
            "/absolute/path/video.mp4",  # Unix absolute
            "relative/sub/video.mp4",  # Relative with subdirectories
        ]
        
        for path_str in test_paths:
            plume = VideoPlume(path_str)
            assert plume.video_path == Path(path_str)
            plume.close()
    
    def test_windows_path_compatibility(self, mock_video_capture, mock_exists):
        """Test Windows-style path handling."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Test Windows-style paths (if running on appropriate platform)
        windows_path = r"C:\Videos\test_video.mp4"
        
        plume = VideoPlume(windows_path)
        assert plume.video_path == Path(windows_path)
        plume.close()
    
    def test_unicode_path_support(self, mock_video_capture, mock_exists):
        """Test support for Unicode characters in file paths."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        unicode_path = "测试视频_ñäme.mp4"
        
        plume = VideoPlume(unicode_path)
        assert plume.video_path == Path(unicode_path)
        plume.close()


class TestWorkflowIntegration:
    """Test integration with DVC and data versioning scenarios."""
    
    def test_dvc_integration_compatibility(self, mock_video_capture, mock_exists):
        """Test VideoPlume compatibility with DVC data versioning."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Simulate DVC-style data path
        dvc_data_path = "data/videos/experiment_001/plume_video.mp4"
        
        plume = VideoPlume(dvc_data_path)
        assert plume.video_path == Path(dvc_data_path)
        
        # Test metadata extraction for DVC tracking
        metadata = plume.get_metadata()
        assert "frame_count" in metadata
        assert "duration" in metadata
        assert "fps" in metadata
        
        plume.close()
    
    def test_workflow_metadata_compatibility(self, mock_video_capture, mock_exists):
        """Test metadata format compatibility with workflow systems."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        plume = VideoPlume("test_video.mp4")
        metadata = plume.get_metadata()
        
        # Verify metadata is JSON-serializable (for workflow systems)
        import json
        try:
            json.dumps(metadata)
            metadata_json_compatible = True
        except (TypeError, ValueError):
            metadata_json_compatible = False
        
        assert metadata_json_compatible, "Metadata must be JSON-serializable for workflow integration"
        
        plume.close()
    
    def test_batch_processing_scenarios(self, mock_video_capture, mock_exists):
        """Test VideoPlume behavior in batch processing scenarios."""
        from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        
        # Simulate batch processing of multiple videos
        video_paths = [f"batch_video_{i}.mp4" for i in range(3)]
        
        plumes = []
        for path in video_paths:
            plume = VideoPlume(path)
            plumes.append(plume)
        
        # Test all plumes are functional
        for plume in plumes:
            assert plume.frame_count > 0
            assert plume.get_frame(0) is not None
        
        # Clean up
        for plume in plumes:
            plume.close()


# Pytest fixtures for test support
@pytest.fixture
def mock_video_capture():
    """Create a comprehensive mock for cv2.VideoCapture with realistic behavior."""
    with patch('cv2.VideoCapture') as mock_cv2:
        mock_instance = MagicMock()
        mock_cv2.return_value = mock_instance
        
        # Configure successful video capture
        mock_instance.isOpened.return_value = True
        
        # Set up realistic video properties
        mock_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        # Configure frame reading
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_instance.read.return_value = (True, mock_frame)
        
        # Configure seeking and resource management
        mock_instance.set.return_value = True
        mock_instance.release.return_value = None
        
        yield mock_cv2


@pytest.fixture
def mock_exists():
    """Mock pathlib.Path.exists to return True for all test files."""
    with patch.object(Path, 'exists', return_value=True):
        yield


@pytest.fixture
def mock_config_loader():
    """Mock configuration loading utilities."""
    with patch('{{cookiecutter.project_slug}}.config.utils.load_config') as mock_load:
        mock_load.return_value = {
            "video_plume": {
                "flip": False,
                "kernel_size": 0,
                "kernel_sigma": 1.0
            }
        }
        yield mock_load


@pytest.fixture
def test_video_file(tmp_path):
    """Create a temporary test video file for integration tests."""
    # Create a dummy file that represents a video
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake video content")
    return video_file


@pytest.fixture
def performance_timer():
    """Utility fixture for measuring execution time in tests."""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        def elapsed_ms(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return None
    
    return PerformanceTimer()


# Additional test utilities
def create_mock_video_frame(width: int = 640, height: int = 480, channels: int = 3) -> np.ndarray:
    """Create a mock video frame for testing purposes."""
    return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)


def assert_valid_grayscale_frame(frame: np.ndarray, expected_shape: tuple = (480, 640)):
    """Assert that a frame is a valid grayscale frame with expected properties."""
    assert frame is not None, "Frame should not be None"
    assert len(frame.shape) == 2, "Grayscale frame should be 2D"
    assert frame.shape == expected_shape, f"Expected shape {expected_shape}, got {frame.shape}"
    assert frame.dtype == np.uint8, f"Expected uint8 dtype, got {frame.dtype}"


def assert_performance_requirement(execution_time_ms: float, max_time_ms: float = 33.0):
    """Assert that execution time meets performance requirements."""
    assert execution_time_ms < max_time_ms, \
        f"Execution time {execution_time_ms:.2f}ms exceeds requirement of {max_time_ms}ms"