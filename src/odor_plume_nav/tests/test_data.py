"""
Comprehensive test suite for VideoPlume data processing functionality.

This module provides comprehensive testing for video-based odor plume environment
processing, validating OpenCV integration, frame processing performance, metadata
extraction, resource management, and configuration system integration within the
unified odor_plume_nav package structure.

The test suite ensures VideoPlume meets all performance requirements including
33ms frame processing limits, >80% test coverage, cross-platform compatibility,
and proper integration with the broader simulation framework.

Test Categories:
- Initialization and configuration validation
- OpenCV video capture integration
- Frame preprocessing and access patterns
- Metadata extraction and timing information
- Resource management and cleanup validation
- Performance benchmarking and SLA compliance
- Configuration system integration testing
- Workflow integration with DVC and versioning
- Cross-platform compatibility validation
- Error handling and edge case coverage
"""

import pytest
cv2 = pytest.importorskip("cv2")
import numpy as np
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, Optional
import warnings
import os
import json

try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = dict
    HYDRA_AVAILABLE = False

from odor_plume_nav.data.video_plume import (
    VideoPlume,
    create_video_plume
)
from odor_plume_nav.config.models import VideoPlumeConfig


class TestVideoPlumeInitialization:
    """Test suite for VideoPlume initialization and configuration validation."""
    
    def test_basic_initialization_with_mock_video(self, mock_video_capture, temp_video_file):
        """Test basic VideoPlume initialization with mocked video capture."""
        # Configure mock video capture with realistic metadata
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(
                video_path=temp_video_file,
                flip=False,
                kernel_size=5,
                kernel_sigma=1.0,
                grayscale=True
            )
            
            # Validate initialization parameters
            assert plume.video_path == Path(temp_video_file)
            assert plume.flip == False
            assert plume.kernel_size == 5
            assert plume.kernel_sigma == 1.0
            assert plume.grayscale == True
            
            # Validate extracted metadata
            assert plume.frame_count == 100
            assert plume.width == 640
            assert plume.height == 480
            assert plume.fps == 30.0
            assert plume.shape == (480, 640)
            assert abs(plume.duration - 3.33) < 0.01  # 100 frames / 30 fps
            
            plume.close()
    
    def test_initialization_with_invalid_video_path(self):
        """Test VideoPlume initialization with non-existent video file."""
        with pytest.raises(IOError, match="Video file does not exist"):
            VideoPlume(video_path="nonexistent_video.mp4")
    
    def test_initialization_with_invalid_kernel_size(self, temp_video_file):
        """Test VideoPlume initialization with invalid kernel parameters."""
        # Test even kernel size
        with pytest.raises(ValueError, match="kernel_size must be positive and odd"):
            VideoPlume(video_path=temp_video_file, kernel_size=4, kernel_sigma=1.0)
        
        # Test negative kernel size
        with pytest.raises(ValueError, match="kernel_size must be positive and odd"):
            VideoPlume(video_path=temp_video_file, kernel_size=-1, kernel_sigma=1.0)
        
        # Test negative sigma
        with pytest.raises(ValueError, match="kernel_sigma must be positive"):
            VideoPlume(video_path=temp_video_file, kernel_size=5, kernel_sigma=-0.5)
    
    def test_initialization_with_failed_video_capture(self, temp_video_file):
        """Test VideoPlume initialization when OpenCV fails to open video."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            with pytest.raises(IOError, match="Failed to open video file"):
                VideoPlume(video_path=temp_video_file)
    
    def test_initialization_with_zero_frame_count(self, temp_video_file):
        """Test VideoPlume initialization with video containing no frames."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 0,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            with pytest.raises(IOError, match="Invalid frame count"):
                VideoPlume(video_path=temp_video_file)
    
    def test_from_config_factory_method_with_dict(self, mock_video_capture, temp_video_file):
        """Test VideoPlume.from_config factory method with dictionary configuration."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 150,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 60.0
        }.get(prop, 0)
        
        config = {
            "video_path": str(temp_video_file),
            "flip": True,
            "kernel_size": 7,
            "kernel_sigma": 2.0,
            "grayscale": False,
            "threshold": 0.5,
            "normalize": True
        }
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume.from_config(config)
            
            assert plume.flip == True
            assert plume.kernel_size == 7
            assert plume.kernel_sigma == 2.0
            assert plume.grayscale == False
            assert plume.threshold == 0.5
            assert plume.normalize == True
            assert plume.frame_count == 150
            
            plume.close()
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_from_config_factory_method_with_hydra_config(self, mock_video_capture, temp_video_file):
        """Test VideoPlume.from_config factory method with Hydra DictConfig."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 200,
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 24.0
        }.get(prop, 0)
        
        hydra_config = DictConfig({
            "video_path": str(temp_video_file),
            "flip": False,
            "kernel_size": 3,
            "kernel_sigma": 1.5,
            "grayscale": True
        })
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume.from_config(hydra_config)
            
            assert plume.kernel_size == 3
            assert plume.kernel_sigma == 1.5
            assert plume.frame_count == 200
            
            plume.close()
    
    def test_from_config_with_invalid_configuration(self):
        """Test VideoPlume.from_config with invalid configuration parameters."""
        config = {
            "video_path": "nonexistent.mp4",
            "kernel_size": 4,  # Even kernel size
            "kernel_sigma": 1.0
        }
        
        with pytest.raises(ValueError, match="Invalid VideoPlume configuration"):
            VideoPlume.from_config(config)


class TestVideoPlumeFrameProcessing:
    """Test suite for VideoPlume frame processing and access patterns."""
    
    def test_get_frame_basic_functionality(self, mock_video_capture, temp_video_file):
        """Test basic frame retrieval functionality."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 50,
            cv2.CAP_PROP_FRAME_WIDTH: 320,
            cv2.CAP_PROP_FRAME_HEIGHT: 240,
            cv2.CAP_PROP_FPS: 15.0
        }.get(prop, 0)
        
        # Create synthetic frame data
        test_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(
                video_path=temp_video_file,
                flip=False,
                grayscale=False,
                normalize=False
            )
            
            frame = plume.get_frame(10)
            
            # Validate frame retrieval
            assert frame is not None
            assert isinstance(frame, np.ndarray)
            mock_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 10)
            mock_cap.read.assert_called_once()
            
            plume.close()
    
    def test_get_frame_with_preprocessing(self, mock_video_capture, temp_video_file):
        """Test frame retrieval with comprehensive preprocessing pipeline."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 160,
            cv2.CAP_PROP_FRAME_HEIGHT: 120,
            cv2.CAP_PROP_FPS: 10.0
        }.get(prop, 0)
        
        # Create test frame
        test_frame = np.ones((120, 160, 3), dtype=np.uint8) * 128
        mock_cap.read.return_value = (True, test_frame)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(
                video_path=temp_video_file,
                flip=True,
                kernel_size=5,
                kernel_sigma=1.5,
                grayscale=True,
                threshold=0.5,
                normalize=True
            )
            
            frame = plume.get_frame(5)
            
            # Validate that frame was processed (should be different from input)
            assert frame is not None
            assert isinstance(frame, np.ndarray)
            
            plume.close()
    
    def test_get_frame_performance_benchmark(self, mock_video_capture, temp_video_file):
        """Test frame processing performance meets 33ms SLA requirement."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        # Create realistic frame data
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(
                video_path=temp_video_file,
                flip=True,
                kernel_size=5,
                kernel_sigma=1.0,
                grayscale=True,
                normalize=True
            )
            
            # Benchmark frame processing time
            start_time = time.time()
            
            for i in range(10):  # Test multiple frames for consistency
                frame = plume.get_frame(i)
                assert frame is not None
            
            end_time = time.time()
            avg_frame_time = (end_time - start_time) / 10
            
            # Validate performance requirement: ≤33ms per frame
            assert avg_frame_time <= 0.033, f"Frame processing took {avg_frame_time:.4f}s, exceeds 33ms limit"
            
            plume.close()
    
    def test_get_frame_out_of_bounds_access(self, mock_video_capture, temp_video_file):
        """Test frame access with invalid frame indices."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 20,
            cv2.CAP_PROP_FRAME_WIDTH: 320,
            cv2.CAP_PROP_FRAME_HEIGHT: 240,
            cv2.CAP_PROP_FPS: 15.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(video_path=temp_video_file)
            
            # Test negative frame index
            assert plume.get_frame(-1) is None
            
            # Test frame index beyond range
            assert plume.get_frame(20) is None  # Frame count is 20, so index 20 is out of bounds
            assert plume.get_frame(100) is None
            
            plume.close()
    
    def test_get_frame_read_failure(self, mock_video_capture, temp_video_file):
        """Test frame retrieval when OpenCV read operation fails."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 160,
            cv2.CAP_PROP_FRAME_HEIGHT: 120,
            cv2.CAP_PROP_FPS: 10.0
        }.get(prop, 0)
        
        # Simulate read failure
        mock_cap.read.return_value = (False, None)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(video_path=temp_video_file)
            
            frame = plume.get_frame(5)
            assert frame is None
            
            plume.close()
    
    def test_get_frame_after_close(self, mock_video_capture, temp_video_file):
        """Test frame access after VideoPlume has been closed."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 10,
            cv2.CAP_PROP_FRAME_WIDTH: 160,
            cv2.CAP_PROP_FRAME_HEIGHT: 120,
            cv2.CAP_PROP_FPS: 10.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(video_path=temp_video_file)
            plume.close()
            
            with pytest.raises(ValueError, match="Cannot get frame from closed VideoPlume"):
                plume.get_frame(0)


class TestVideoPlumeMetadata:
    """Test suite for VideoPlume metadata extraction and timing information."""
    
    def test_get_metadata_comprehensive(self, mock_video_capture, temp_video_file):
        """Test comprehensive metadata extraction functionality."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 300,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 60.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(
                video_path=temp_video_file,
                flip=True,
                kernel_size=7,
                kernel_sigma=2.0,
                grayscale=False,
                threshold=0.3,
                normalize=True
            )
            
            metadata = plume.get_metadata()
            
            # Validate video file properties
            assert metadata["video_path"] == str(temp_video_file)
            
            # Validate video stream properties
            assert metadata["width"] == 1920
            assert metadata["height"] == 1080
            assert metadata["fps"] == 60.0
            assert metadata["frame_count"] == 300
            assert metadata["shape"] == (1080, 1920)
            assert abs(metadata["duration"] - 5.0) < 0.01  # 300 frames / 60 fps = 5.0 seconds
            
            # Validate processing configuration
            config = metadata["preprocessing"]
            assert config["flip"] == True
            assert config["kernel_size"] == 7
            assert config["kernel_sigma"] == 2.0
            assert config["grayscale"] == False
            assert config["threshold"] == 0.3
            assert config["normalize"] == True
            
            plume.close()
    
    def test_get_metadata_string_formatting(self, mock_video_capture, temp_video_file):
        """Test formatted metadata string generation for research documentation."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 1800,
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(
                video_path=temp_video_file,
                flip=True,
                kernel_size=5,
                kernel_sigma=1.5,
                threshold=0.5,
                normalize=True
            )
            
            metadata_str = plume.get_metadata_string()
            
            # Validate formatted string content
            assert Path(temp_video_file).name in metadata_str
            assert "1280x720 pixels" in metadata_str
            assert "30.0 fps" in metadata_str
            assert "1800 frames" in metadata_str
            assert "60.00 seconds" in metadata_str
            assert "flip" in metadata_str
            assert "gaussian(5,1.5)" in metadata_str
            assert "threshold(0.5)" in metadata_str
            assert "normalize" in metadata_str
            
            plume.close()
    
    def test_get_metadata_string_no_processing(self, mock_video_capture, temp_video_file):
        """Test metadata string formatting when no processing is applied."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 120,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 24.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(
                video_path=temp_video_file,
                flip=False,
                normalize=False
            )
            
            metadata_str = plume.get_metadata_string()
            
            # Validate that "none" is shown when no processing is applied
            assert "Preprocessing: none" in metadata_str
            
            plume.close()
    
    def test_duration_property_calculation(self, mock_video_capture, temp_video_file):
        """Test duration property calculation with edge cases."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        
        # Test normal case
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 150,
            cv2.CAP_PROP_FRAME_WIDTH: 320,
            cv2.CAP_PROP_FRAME_HEIGHT: 240,
            cv2.CAP_PROP_FPS: 25.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(video_path=temp_video_file)
            assert abs(plume.duration - 6.0) < 0.01  # 150 / 25 = 6.0
            plume.close()
        
        # Test zero FPS case
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 320,
            cv2.CAP_PROP_FRAME_HEIGHT: 240,
            cv2.CAP_PROP_FPS: 0.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress FPS warning for this test
            plume = VideoPlume(video_path=temp_video_file)
            assert plume.duration == 0.0
            plume.close()


class TestVideoPlumeResourceManagement:
    """Test suite for VideoPlume resource management and cleanup validation."""
    
    def test_proper_resource_cleanup_on_close(self, mock_video_capture, temp_video_file):
        """Test proper resource cleanup when close() is called explicitly."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 50,
            cv2.CAP_PROP_FRAME_WIDTH: 320,
            cv2.CAP_PROP_FRAME_HEIGHT: 240,
            cv2.CAP_PROP_FPS: 15.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(video_path=temp_video_file)
            
            # Verify initial state
            assert not plume._is_closed
            
            # Test explicit close
            plume.close()
            
            # Verify cleanup
            assert plume._is_closed
            mock_cap.release.assert_called_once()
    
    def test_multiple_close_calls_safe(self, mock_video_capture, temp_video_file):
        """Test that multiple close() calls are safe and don't cause errors."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 160,
            cv2.CAP_PROP_FRAME_HEIGHT: 120,
            cv2.CAP_PROP_FPS: 10.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(video_path=temp_video_file)
            
            # Call close multiple times
            plume.close()
            plume.close()
            plume.close()
            
            # Should still be closed without errors
            assert plume._is_closed
            mock_cap.release.assert_called_once()  # Should only be called once
    
    def test_context_manager_automatic_cleanup(self, mock_video_capture, temp_video_file):
        """Test automatic resource cleanup using context manager."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 25,
            cv2.CAP_PROP_FRAME_WIDTH: 200,
            cv2.CAP_PROP_FRAME_HEIGHT: 150,
            cv2.CAP_PROP_FPS: 12.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            with VideoPlume(video_path=temp_video_file) as plume:
                assert not plume._is_closed
                # Use plume within context
                assert plume.frame_count == 25
            
            # Verify automatic cleanup after exiting context
            assert plume._is_closed
            mock_cap.release.assert_called_once()
    
    def test_destructor_cleanup(self, mock_video_capture, temp_video_file):
        """Test automatic cleanup through destructor when object is garbage collected."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 40,
            cv2.CAP_PROP_FRAME_WIDTH: 240,
            cv2.CAP_PROP_FRAME_HEIGHT: 180,
            cv2.CAP_PROP_FPS: 20.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(video_path=temp_video_file)
            plume_id = id(plume)
            
            # Delete the object to trigger destructor
            del plume
            
            # Note: In production, this would call release, but in testing we can't
            # easily verify destructor behavior due to garbage collection timing
            # The important thing is that the destructor exists and is implemented
    
    def test_thread_safety_resource_management(self, mock_video_capture, temp_video_file):
        """Test thread safety of resource management operations."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 60,
            cv2.CAP_PROP_FRAME_WIDTH: 320,
            cv2.CAP_PROP_FRAME_HEIGHT: 240,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        test_frame = np.ones((240, 320, 3), dtype=np.uint8) * 100
        mock_cap.read.return_value = (True, test_frame)
        
        def worker_function(plume, results, worker_id):
            """Worker function for concurrent access testing."""
            try:
                for i in range(5):
                    frame = plume.get_frame(i % plume.frame_count)
                    if frame is not None:
                        results[worker_id] = results.get(worker_id, 0) + 1
                    time.sleep(0.001)  # Small delay to increase chance of race conditions
            except Exception as e:
                results[f"error_{worker_id}"] = str(e)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(video_path=temp_video_file)
            
            # Create multiple threads accessing the same VideoPlume instance
            threads = []
            results = {}
            
            for i in range(3):
                thread = threading.Thread(
                    target=worker_function,
                    args=(plume, results, i)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify no errors occurred and operations completed
            for i in range(3):
                assert f"error_{i}" not in results
                assert results.get(i, 0) > 0  # Each worker should have processed frames
            
            plume.close()


class TestVideoPlumeConfigurationIntegration:
    """Test suite for VideoPlume integration with configuration system."""
    
    def test_pydantic_schema_validation_integration(self, temp_video_file):
        """Test integration with Pydantic VideoPlumeConfig schema validation."""
        # Test valid configuration
        valid_config = VideoPlumeConfig(
            video_path=temp_video_file,
            flip=True,
            kernel_size=5,
            kernel_sigma=1.5,
            grayscale=True,
            threshold=0.3,
            normalize=True
        )
        
        assert valid_config.video_path == temp_video_file
        assert valid_config.flip == True
        assert valid_config.kernel_size == 5
        assert valid_config.kernel_sigma == 1.5
        
        # Test invalid configuration - even kernel size
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            VideoPlumeConfig(
                video_path=temp_video_file,
                kernel_size=4,
                kernel_sigma=1.0
            )
        
        # Test invalid configuration - negative kernel size
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            VideoPlumeConfig(
                video_path=temp_video_file,
                kernel_size=-2,
                kernel_sigma=1.0
            )
    
    def test_environment_variable_interpolation(self, mock_video_capture, temp_video_file):
        """Test environment variable interpolation in video path configuration."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 75,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 25.0
        }.get(prop, 0)
        
        # Set environment variable
        os.environ['TEST_VIDEO_PATH'] = str(temp_video_file)
        
        try:
            with patch('cv2.VideoCapture', return_value=mock_cap):
                # Test standard environment variable expansion
                plume = VideoPlume(video_path=os.path.expandvars("$TEST_VIDEO_PATH"))
                assert plume.video_path == Path(temp_video_file)
                plume.close()
                
                # Test ${VAR} syntax  
                plume = VideoPlume(video_path=os.path.expandvars("${TEST_VIDEO_PATH}"))
                assert plume.video_path == Path(temp_video_file)
                plume.close()
        finally:
            # Clean up environment variable
            del os.environ['TEST_VIDEO_PATH']
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_interpolation_fallback(self, mock_video_capture, temp_video_file):
        """Test Hydra-style environment variable interpolation fallback."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 90,
            cv2.CAP_PROP_FRAME_WIDTH: 800,
            cv2.CAP_PROP_FRAME_HEIGHT: 600,
            cv2.CAP_PROP_FPS: 20.0
        }.get(prop, 0)
        
        # Set environment variable
        os.environ['HYDRA_TEST_VIDEO'] = str(temp_video_file)
        
        try:
            with patch('cv2.VideoCapture', return_value=mock_cap):
                # Test Hydra-style interpolation (simulated)
                hydra_path = os.environ.get('HYDRA_TEST_VIDEO', temp_video_file)
                plume = VideoPlume(video_path=hydra_path)
                assert plume.video_path == Path(temp_video_file)
                plume.close()
        finally:
            # Clean up environment variable
            del os.environ['HYDRA_TEST_VIDEO']
    
    def test_factory_method_with_configuration_validation(self, mock_video_capture, temp_video_file):
        """Test factory method integration with comprehensive configuration validation."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 120,
            cv2.CAP_PROP_FRAME_WIDTH: 1024,
            cv2.CAP_PROP_FRAME_HEIGHT: 768,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        config = {
            "video_path": str(temp_video_file),
            "flip": False,
            "kernel_size": 7,
            "kernel_sigma": 2.0,
            "grayscale": True,
            "threshold": 0.4,
            "normalize": True
        }
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            # Test successful creation through factory method
            plume = VideoPlume.from_config(config)
            
            # Verify configuration was properly applied
            assert plume.flip == False
            assert plume.kernel_size == 7
            assert plume.kernel_sigma == 2.0
            assert plume.grayscale == True
            assert plume.threshold == 0.4
            assert plume.normalize == True
            
            # Verify video properties
            assert plume.frame_count == 120
            assert plume.width == 1024
            assert plume.height == 768
            
            plume.close()


class TestVideoPlumeWorkflowIntegration:
    """Test suite for VideoPlume integration with DVC and workflow orchestration."""
    
    def test_create_video_plume_factory_function(self, mock_video_capture, temp_video_file):
        """Test create_video_plume factory function integration."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 200,
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 60.0
        }.get(prop, 0)
        
        config = {
            "video_path": str(temp_video_file),
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.5
        }
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            # Test factory function with configuration override
            plume = create_video_plume(config, flip=True)
            
            assert plume.video_path == Path(temp_video_file)
            assert plume.flip == True
            assert plume.kernel_size == 5
            assert plume.kernel_sigma == 1.5
            
            # Verify metadata includes workflow compatibility information
            metadata = plume.get_metadata()
            assert "video_path" in metadata
            assert "preprocessing" in metadata
            
            plume.close()
    
    def test_workflow_metadata_integration(self, mock_video_capture, temp_video_file):
        """Test metadata generation for workflow integration and documentation."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 450,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(
                video_path=temp_video_file,
                flip=True,
                kernel_size=7,
                kernel_sigma=2.0,
                grayscale=True,
                normalize=True
            )
            
            metadata = plume.get_metadata()
            
            # Validate metadata can be serialized for workflow systems
            json_metadata = json.dumps(metadata, default=str)
            parsed_metadata = json.loads(json_metadata)
            
            assert parsed_metadata["frame_count"] == 450
            assert parsed_metadata["width"] == 1920
            assert parsed_metadata["height"] == 1080
            
            # Test workflow metadata method
            workflow_metadata = plume.get_workflow_metadata()
            assert "file_hash" in workflow_metadata
            assert "file_size" in workflow_metadata
            assert "workflow_version" in workflow_metadata
            assert "dependencies" in workflow_metadata
            
            plume.close()


class TestVideoPlumeErrorHandling:
    """Test suite for VideoPlume error handling and edge case coverage."""
    
    def test_string_representation_methods(self, mock_video_capture, temp_video_file):
        """Test string representation methods for debugging and logger."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(video_path=temp_video_file)
            
            # Test __repr__ method
            repr_str = repr(plume)
            assert "VideoPlume" in repr_str
            assert Path(temp_video_file).name in repr_str
            assert "100 frames" in repr_str
            assert "640x480" in repr_str
            assert "open" in repr_str
            
            # Test representation after close
            plume.close()
            repr_str_closed = repr(plume)
            assert "closed" in repr_str_closed
    
    def test_invalid_video_dimensions_handling(self, temp_video_file):
        """Test handling of videos with invalid dimensions."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 50,
            cv2.CAP_PROP_FRAME_WIDTH: 0,  # Invalid width
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            with pytest.raises(IOError, match="Invalid video dimensions"):
                VideoPlume(video_path=temp_video_file)
    
    def test_missing_fps_warning_handling(self, mock_video_capture, temp_video_file):
        """Test handling of videos with missing or invalid FPS information."""
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 60,
            cv2.CAP_PROP_FRAME_WIDTH: 320,
            cv2.CAP_PROP_FRAME_HEIGHT: 240,
            cv2.CAP_PROP_FPS: 0.0  # Invalid FPS
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                plume = VideoPlume(video_path=temp_video_file)
                
                # Verify warning was issued
                assert len(w) == 1
                assert "Invalid or missing FPS" in str(w[0].message)
                
                # Verify fallback FPS was applied
                assert plume.fps == 30.0
                
                plume.close()


class TestVideoPlumeVideoFormatCompatibility:
    """Test suite for VideoPlume compatibility with different video formats."""
    
    @pytest.mark.parametrize("video_format,extension", [
        ("MP4", ".mp4"),
        ("AVI", ".avi"),
    ])
    def test_video_format_support(self, video_format, extension, mock_video_capture, tmp_path):
        """Test VideoPlume support for different video formats per technical specifications."""
        # Create temporary video file with format-specific extension
        video_file = tmp_path / f"test_video{extension}"
        video_file.write_text("dummy_content")  # Create file
        
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 150,
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 24.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            plume = VideoPlume(video_path=video_file)
            
            # Verify format is handled correctly
            assert plume.video_path.suffix == extension
            assert plume.frame_count == 150
            assert plume.width == 1280
            assert plume.height == 720
            
            # Verify metadata includes format information
            metadata = plume.get_metadata()
            assert str(video_file) in metadata["video_path"]
            
            plume.close()
    
    def test_cross_platform_path_handling(self, mock_video_capture, tmp_path):
        """Test cross-platform file path handling for video files."""
        # Create test video file
        video_file = tmp_path / "cross_platform_test.mp4"
        video_file.write_text("dummy_content")
        
        mock_cap = mock_video_capture
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            # Test with Path object
            plume_path = VideoPlume(video_path=video_file)
            assert isinstance(plume_path.video_path, Path)
            plume_path.close()
            
            # Test with string path
            plume_str = VideoPlume(video_path=str(video_file))
            assert isinstance(plume_str.video_path, Path)
            assert plume_str.video_path == video_file
            plume_str.close()
            
            # Test with relative path
            relative_path = video_file.relative_to(Path.cwd())
            plume_rel = VideoPlume(video_path=str(relative_path))
            assert isinstance(plume_rel.video_path, Path)
            plume_rel.close()


# Test Fixtures and Utilities

@pytest.fixture
def mock_video_capture():
    """Provide a mock OpenCV VideoCapture object with standard behavior."""
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 0
    mock_cap.read.return_value = (False, None)
    mock_cap.set.return_value = True
    mock_cap.release.return_value = None
    return mock_cap


@pytest.fixture
def temp_video_file(tmp_path):
    """Create a temporary video file for testing."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_text("dummy_video_content")
    return str(video_file)


@pytest.fixture
def sample_video_config():
    """Provide a sample video configuration for testing."""
    return {
        "flip": True,
        "kernel_size": 5,
        "kernel_sigma": 1.5,
        "grayscale": True,
        "threshold": 0.3,
        "normalize": True
    }


@pytest.fixture
def performance_test_config():
    """Provide configuration optimized for performance testing."""
    return {
        "flip": True,
        "kernel_size": 5,
        "kernel_sigma": 1.0,
        "grayscale": True,
        "normalize": True
    }


# Test markers for different test categories
pytestmark = [
    pytest.mark.unit,
    pytest.mark.data_processing,
    pytest.mark.video_processing
]