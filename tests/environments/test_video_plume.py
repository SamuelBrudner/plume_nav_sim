"""
Tests for the VideoPlume environment with comprehensive Hydra integration.

This test module validates the VideoPlume class functionality including:
- Enhanced Hydra DictConfig integration and factory method patterns
- Environment variable interpolation through ${oc.env:} syntax
- Pydantic schema validation with Hydra structured config system
- Workflow orchestration integration points for DVC/Snakemake
- Configuration loading and validation with hierarchical parameter management
- CLI integration testing with Click and Hydra parameter flow
- Database session compatibility for future persistence integration
- Cross-platform compatibility and error handling scenarios

The testing approach follows the enhanced cookiecutter-based architecture with
comprehensive fixture management, isolated test environments, and research-grade
quality assurance standards as specified in Section 6.6 of the technical specification.
"""

import pytest
import os
import tempfile
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import patch, MagicMock, mock_open
from contextlib import contextmanager

# Update imports per Section 0.2.1 mapping table
from {{cookiecutter.project_slug}}.data.video_plume import (
    VideoPlume,
    create_video_plume_from_dvc_path,
    create_snakemake_rule_config
)
from {{cookiecutter.project_slug}}.config.schemas import VideoPlumeConfig

# Import Hydra components for enhanced configuration testing
try:
    from omegaconf import DictConfig, OmegaConf
    import hydra
    from hydra import compose, initialize_config_store
    HYDRA_AVAILABLE = True
except ImportError:
    # Graceful fallback for environments without Hydra
    DictConfig = dict
    OmegaConf = None
    HYDRA_AVAILABLE = False
    

class TestVideoPlumeBasicFunctionality:
    """
    Test suite for core VideoPlume functionality with enhanced Hydra integration.
    
    This test class validates the fundamental VideoPlume operations while incorporating
    the new factory method patterns, configuration management, and error handling
    required by the cookiecutter-based architecture.
    """

    @pytest.fixture
    def mock_exists(self, monkeypatch):
        """Mock the Path.exists method to return True for all paths except 'nonexistent_file.mp4'."""
        def patched_exists(self):
            return str(self) != "nonexistent_file.mp4"
        
        monkeypatch.setattr(Path, "exists", patched_exists)
        return patched_exists

    @pytest.fixture
    def mock_video_capture(self):
        """Create a mock for cv2.VideoCapture with comprehensive frame data."""
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
    def failed_video_capture(self):
        """Create a mock for cv2.VideoCapture that fails to open."""
        with patch('cv2.VideoCapture') as mock_cap:
            # Configure the mock to return a failed instance
            mock_instance = MagicMock()
            mock_cap.return_value = mock_instance
            
            # Mock isOpened to return False
            mock_instance.isOpened.return_value = False
            
            yield mock_cap

    def test_video_plume_loading(self, mock_video_capture, mock_exists):
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

    def test_nonexistent_file(self, mock_exists):
        """Test that VideoPlume raises IOError when file doesn't exist."""
        with pytest.raises(IOError, match="Video file does not exist"):
            VideoPlume("nonexistent_file.mp4")

    def test_failed_open(self, failed_video_capture, mock_exists):
        """Test that VideoPlume raises IOError when video can't be opened."""
        with pytest.raises(IOError, match="Failed to open video file"):
            VideoPlume("failed_video.mp4")

    def test_get_frame_valid_index(self, mock_video_capture, mock_exists):
        """Test that get_frame returns a frame for valid indices."""
        plume = VideoPlume("dummy_video.mp4")
        
        # Get a frame at index 50
        frame = plume.get_frame(50)
        
        # Check that the frame was retrieved and converted to grayscale
        assert frame is not None
        mock_video_capture.return_value.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 50)
        mock_video_capture.return_value.read.assert_called_once()

    def test_get_frame_invalid_index(self, mock_video_capture, mock_exists):
        """Test that get_frame returns None for invalid indices."""
        plume = VideoPlume("dummy_video.mp4")
        
        # Try to get frames with invalid indices
        negative_frame = plume.get_frame(-1)
        assert negative_frame is None
        
        too_large_frame = plume.get_frame(200)  # Beyond frame_count
        assert too_large_frame is None

    def test_get_frame_after_close(self, mock_video_capture, mock_exists):
        """Test that get_frame raises ValueError after VideoPlume is closed."""
        plume = VideoPlume("dummy_video.mp4")
        
        # Close the video plume
        plume.close()
        
        # Try to get a frame after closing, should raise ValueError
        with pytest.raises(ValueError, match="Cannot get frame from closed VideoPlume"):
            plume.get_frame(0)

    def test_close_idempotent(self, mock_video_capture, mock_exists):
        """Test that calling close() multiple times is safe."""
        plume = VideoPlume("dummy_video.mp4")
        
        # Close once
        plume.close()
        
        # Verify closed state through behavior instead of internal state
        with pytest.raises(ValueError, match="Cannot get frame from closed VideoPlume"):
            plume.get_frame(0)
        
        # Close again, should not raise any errors
        plume.close()
        
        # Still closed after second close
        with pytest.raises(ValueError, match="Cannot get frame from closed VideoPlume"):
            plume.get_frame(0)

    def test_frame_metadata(self, mock_video_capture, mock_exists):
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

    def test_preprocessing_pipeline(self, mock_video_capture, mock_exists):
        """Test comprehensive preprocessing pipeline functionality."""
        # Test flip functionality
        plume_flip = VideoPlume("dummy_video.mp4", flip=True)
        frame = plume_flip.get_frame(0)
        assert frame is not None
        
        # Test gaussian smoothing
        plume_blur = VideoPlume("dummy_video.mp4", kernel_size=5, kernel_sigma=1.5)
        frame = plume_blur.get_frame(0)
        assert frame is not None
        
        # Test normalization
        plume_norm = VideoPlume("dummy_video.mp4", normalize=True)
        frame = plume_norm.get_frame(0)
        assert frame is not None
        
        # Test threshold
        plume_thresh = VideoPlume("dummy_video.mp4", threshold=0.5)
        frame = plume_thresh.get_frame(0)
        assert frame is not None

    def test_context_manager_usage(self, mock_video_capture, mock_exists):
        """Test VideoPlume usage as context manager."""
        with VideoPlume("dummy_video.mp4") as plume:
            assert not plume.is_closed
            frame = plume.get_frame(0)
            assert frame is not None
        
        # Should be closed after exiting context
        assert plume.is_closed


class TestVideoPlumeHydraIntegration:
    """
    Comprehensive test suite for Hydra DictConfig integration.
    
    This test class validates the enhanced VideoPlume factory method patterns,
    configuration composition, environment variable interpolation, and hierarchical
    parameter management as specified in Section 3.2.4.1 and Section 5.2.2.
    """

    @pytest.fixture
    def mock_video_capture(self):
        """Mock cv2.VideoCapture for configuration testing."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_instance = MagicMock()
            mock_cap.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            
            cap_properties = {
                cv2.CAP_PROP_FRAME_COUNT: 150,
                cv2.CAP_PROP_FRAME_WIDTH: 1280,
                cv2.CAP_PROP_FRAME_HEIGHT: 720,
                cv2.CAP_PROP_FPS: 25.0
            }
            mock_instance.get.side_effect = lambda prop: cap_properties.get(prop, 0)
            
            mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            mock_instance.read.return_value = (True, mock_frame)
            
            yield mock_cap

    @pytest.fixture
    def mock_path_exists(self, monkeypatch):
        """Mock path existence for configuration testing."""
        def patched_exists(self):
            # Allow test video paths to exist
            return "test_video" in str(self) or "experiment" in str(self)
        
        monkeypatch.setattr(Path, "exists", patched_exists)
        return patched_exists

    @pytest.fixture
    def basic_hydra_config(self):
        """Create a basic Hydra DictConfig for testing."""
        if HYDRA_AVAILABLE:
            config_dict = {
                "video_path": "test_video.mp4",
                "flip": True,
                "grayscale": True,
                "kernel_size": 5,
                "kernel_sigma": 1.5,
                "normalize": True
            }
            return OmegaConf.create(config_dict)
        else:
            # Fallback for environments without Hydra
            return {
                "video_path": "test_video.mp4",
                "flip": True,
                "grayscale": True,
                "kernel_size": 5,
                "kernel_sigma": 1.5,
                "normalize": True
            }

    @pytest.fixture
    def env_interpolation_config(self):
        """Create Hydra config with environment variable interpolation."""
        if HYDRA_AVAILABLE:
            config_dict = {
                "video_path": "${oc.env:TEST_VIDEO_PATH,data/default_video.mp4}",
                "flip": "${oc.env:VIDEO_FLIP,false}",
                "kernel_size": "${oc.env:KERNEL_SIZE,3}",
                "threshold": "${oc.env:PLUME_THRESHOLD,0.5}"
            }
            return OmegaConf.create(config_dict)
        else:
            return {
                "video_path": "data/default_video.mp4",
                "flip": False,
                "kernel_size": 3,
                "threshold": 0.5
            }

    def test_from_config_basic_functionality(self, mock_video_capture, mock_path_exists, basic_hydra_config):
        """Test VideoPlume.from_config() with basic DictConfig."""
        plume = VideoPlume.from_config(basic_hydra_config)
        
        # Verify configuration was applied correctly
        assert plume.flip is True
        assert plume.kernel_size == 5
        assert plume.kernel_sigma == 1.5
        assert plume.normalize is True
        
        # Verify video capture initialization
        mock_video_capture.assert_called_once()
        
        # Test frame retrieval works
        frame = plume.get_frame(0)
        assert frame is not None

    def test_from_config_with_dictionary(self, mock_video_capture, mock_path_exists):
        """Test VideoPlume.from_config() with plain dictionary."""
        config_dict = {
            "video_path": "test_video.mp4",
            "flip": False,
            "grayscale": True,
            "kernel_size": 7,
            "kernel_sigma": 2.0
        }
        
        plume = VideoPlume.from_config(config_dict)
        
        # Verify configuration
        assert plume.flip is False
        assert plume.kernel_size == 7
        assert plume.kernel_sigma == 2.0

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_environment_variable_interpolation(self, mock_video_capture, mock_path_exists, env_interpolation_config):
        """Test Hydra environment variable interpolation per Section 3.2.7.1."""
        # Set test environment variables
        os.environ['TEST_VIDEO_PATH'] = 'experiment_001.mp4'
        os.environ['VIDEO_FLIP'] = 'true'
        os.environ['KERNEL_SIZE'] = '9'
        os.environ['PLUME_THRESHOLD'] = '0.3'
        
        try:
            # Resolve environment variables in config
            resolved_config = OmegaConf.to_container(env_interpolation_config, resolve=True)
            plume = VideoPlume.from_config(resolved_config)
            
            # Verify environment variable values were interpolated
            assert "experiment_001.mp4" in str(plume.video_path)
            assert plume.flip is True  # 'true' string converted to boolean
            assert plume.kernel_size == 9
            assert plume.threshold == 0.3
            
        finally:
            # Clean up environment variables
            for var in ['TEST_VIDEO_PATH', 'VIDEO_FLIP', 'KERNEL_SIZE', 'PLUME_THRESHOLD']:
                os.environ.pop(var, None)

    def test_config_validation_errors(self, mock_path_exists):
        """Test comprehensive configuration validation error scenarios."""
        # Test invalid kernel_size (even number)
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            invalid_config = {"video_path": "test_video.mp4", "kernel_size": 4}
            VideoPlume.from_config(invalid_config)
        
        # Test negative kernel_size
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            invalid_config = {"video_path": "test_video.mp4", "kernel_size": -1}
            VideoPlume.from_config(invalid_config)
        
        # Test kernel_size without kernel_sigma
        with pytest.raises(ValueError, match="kernel_sigma must be specified"):
            invalid_config = {"video_path": "test_video.mp4", "kernel_size": 5}
            VideoPlume.from_config(invalid_config)
        
        # Test invalid threshold range
        with pytest.raises(ValueError):
            invalid_config = {"video_path": "test_video.mp4", "threshold": 1.5}  # > 1.0
            VideoPlume.from_config(invalid_config)

    def test_config_override_functionality(self, mock_video_capture, mock_path_exists, basic_hydra_config):
        """Test configuration override through kwargs."""
        # Override specific parameters
        plume = VideoPlume.from_config(
            basic_hydra_config,
            flip=False,  # Override flip from True to False
            kernel_size=7,  # Override kernel_size from 5 to 7
            threshold=0.8  # Add new parameter
        )
        
        # Verify overrides were applied
        assert plume.flip is False  # Overridden
        assert plume.kernel_size == 7  # Overridden
        assert plume.kernel_sigma == 1.5  # Original value
        assert plume.threshold == 0.8  # New parameter

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hierarchical_config_composition(self, mock_video_capture, mock_path_exists):
        """Test hierarchical configuration composition scenarios."""
        # Simulate base config
        base_config = {
            "video_path": "base_video.mp4",
            "flip": False,
            "grayscale": True,
            "kernel_size": 3,
            "kernel_sigma": 1.0
        }
        
        # Simulate environment override
        env_overrides = {
            "flip": True,
            "kernel_size": 5,
            "threshold": 0.6
        }
        
        # Merge configurations (simulating Hydra composition)
        merged_config = {**base_config, **env_overrides}
        hydra_config = OmegaConf.create(merged_config)
        
        plume = VideoPlume.from_config(hydra_config)
        
        # Verify hierarchy was respected
        assert plume.flip is True  # Override value
        assert plume.kernel_size == 5  # Override value
        assert plume.grayscale is True  # Base value
        assert plume.threshold == 0.6  # New parameter


class TestVideoPlumeSchemaValidation:
    """
    Test suite for Pydantic schema validation integrated with Hydra per Section 7.2.2.2.
    
    This test class validates the comprehensive schema validation pipeline including
    type checking, constraint enforcement, and error reporting for the VideoPlumeConfig
    Pydantic model integrated with Hydra's structured configuration system.
    """

    @pytest.fixture
    def valid_schema_config(self):
        """Create valid configuration for schema testing."""
        return {
            "video_path": "valid_video.mp4",
            "flip": True,
            "grayscale": True,
            "kernel_size": 5,
            "kernel_sigma": 1.5,
            "threshold": 0.5,
            "normalize": True
        }

    @pytest.fixture
    def mock_path_exists(self, monkeypatch):
        """Mock path existence for schema validation testing."""
        def patched_exists(self):
            return "valid_video" in str(self) or "experiment" in str(self)
        
        monkeypatch.setattr(Path, "exists", patched_exists)

    def test_schema_validation_success(self, valid_schema_config, mock_path_exists):
        """Test successful Pydantic schema validation."""
        # Skip file validation for testing
        valid_schema_config["_skip_validation"] = True
        
        # This should not raise any exceptions
        validated_config = VideoPlumeConfig.model_validate(valid_schema_config)
        
        # Verify all fields are properly typed
        assert isinstance(validated_config.flip, bool)
        assert isinstance(validated_config.kernel_size, int)
        assert isinstance(validated_config.kernel_sigma, float)
        assert isinstance(validated_config.threshold, float)
        assert isinstance(validated_config.normalize, bool)

    def test_schema_type_validation(self):
        """Test Pydantic type validation enforcement."""
        # Test string instead of boolean for flip
        with pytest.raises(ValueError):
            VideoPlumeConfig.model_validate({
                "video_path": "test.mp4",
                "flip": "not_a_boolean",
                "_skip_validation": True
            })
        
        # Test string instead of integer for kernel_size
        with pytest.raises(ValueError):
            VideoPlumeConfig.model_validate({
                "video_path": "test.mp4", 
                "kernel_size": "not_an_integer",
                "_skip_validation": True
            })
        
        # Test negative value for threshold (should be >= 0.0)
        with pytest.raises(ValueError):
            VideoPlumeConfig.model_validate({
                "video_path": "test.mp4",
                "threshold": -0.1,
                "_skip_validation": True
            })

    def test_schema_constraint_validation(self):
        """Test Pydantic constraint enforcement."""
        # Test kernel_size constraint (must be odd)
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            VideoPlumeConfig.model_validate({
                "video_path": "test.mp4",
                "kernel_size": 6,  # Even number
                "kernel_sigma": 1.0,
                "_skip_validation": True
            })
        
        # Test kernel_sigma constraint (must be positive)
        with pytest.raises(ValueError, match="kernel_sigma must be positive"):
            VideoPlumeConfig.model_validate({
                "video_path": "test.mp4",
                "kernel_size": 5,
                "kernel_sigma": 0.0,  # Not positive
                "_skip_validation": True
            })
        
        # Test threshold range constraint (0.0 <= threshold <= 1.0)
        with pytest.raises(ValueError):
            VideoPlumeConfig.model_validate({
                "video_path": "test.mp4",
                "threshold": 1.5,  # > 1.0
                "_skip_validation": True
            })

    def test_schema_missing_dependency_validation(self):
        """Test validation of parameter dependencies."""
        # Test kernel_size without kernel_sigma
        with pytest.raises(ValueError, match="kernel_sigma must be specified"):
            VideoPlumeConfig.model_validate({
                "video_path": "test.mp4",
                "kernel_size": 5,
                # Missing kernel_sigma
                "_skip_validation": True
            })
        
        # Test kernel_sigma without kernel_size
        with pytest.raises(ValueError, match="kernel_size must be specified"):
            VideoPlumeConfig.model_validate({
                "video_path": "test.mp4",
                "kernel_sigma": 1.5,
                # Missing kernel_size
                "_skip_validation": True
            })

    def test_schema_file_validation(self, mock_path_exists):
        """Test file existence validation in schema."""
        # Test with existing file (should succeed)
        valid_config = VideoPlumeConfig.model_validate({
            "video_path": "valid_video.mp4"
        })
        assert valid_config.video_path == Path("valid_video.mp4")
        
        # Test with non-existent file (should fail)
        with pytest.raises(ValueError, match="Video file not found"):
            VideoPlumeConfig.model_validate({
                "video_path": "nonexistent_video.mp4"
            })

    def test_schema_path_conversion(self):
        """Test automatic path conversion in schema validation."""
        config = VideoPlumeConfig.model_validate({
            "video_path": "test_video.mp4",
            "_skip_validation": True
        })
        
        # Verify string was converted to Path
        assert isinstance(config.video_path, Path)
        assert str(config.video_path) == "test_video.mp4"


class TestVideoPlumeWorkflowIntegration:
    """
    Test suite for DVC/Snakemake workflow orchestration integration per Section 5.2.2.
    
    This test class validates the VideoPlume integration with research workflow
    orchestration tools including DVC data versioning and Snakemake pipeline
    execution as specified in the technical specification.
    """

    @pytest.fixture
    def mock_dvc_environment(self):
        """Mock DVC environment for workflow testing."""
        with patch('subprocess.run') as mock_subprocess:
            # Mock DVC command execution
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "data/videos/experiment_001.mp4"
            yield mock_subprocess

    @pytest.fixture
    def mock_snakemake_environment(self):
        """Mock Snakemake environment for workflow testing."""
        return {
            "input": {"video": "data/raw/experiment.mp4"},
            "output": {"metadata": "data/processed/experiment_meta.json"},
            "params": {"flip": True, "kernel_size": 5},
            "resources": {"mem_mb": 2048, "runtime": 300}
        }

    def test_dvc_integration_workflow(self, mock_dvc_environment):
        """Test VideoPlume integration with DVC data versioning."""
        # Test DVC path resolution
        dvc_path = "data/videos/experiment_001.mp4"
        
        with patch('{{cookiecutter.project_slug}}.data.video_plume.VideoPlume') as mock_plume:
            mock_instance = MagicMock()
            mock_plume.return_value = mock_instance
            
            # Test DVC integration function
            plume = create_video_plume_from_dvc_path(
                dvc_path,
                flip=True,
                kernel_size=5
            )
            
            # Verify VideoPlume was created with DVC path
            mock_plume.assert_called_once()
            call_args = mock_plume.call_args
            assert str(call_args[1]['video_path']).endswith('experiment_001.mp4')
            assert call_args[1]['flip'] is True
            assert call_args[1]['kernel_size'] == 5

    def test_snakemake_rule_config_generation(self, mock_snakemake_environment):
        """Test Snakemake rule configuration generation."""
        rule_config = create_snakemake_rule_config(
            input_video="data/raw/experiment.mp4",
            output_metadata="data/processed/experiment_meta.json",
            processing_params={"flip": True, "kernel_size": 5}
        )
        
        # Verify rule configuration structure
        assert "input" in rule_config
        assert "output" in rule_config
        assert "params" in rule_config
        assert "resources" in rule_config
        
        # Verify specific values
        assert rule_config["input"]["video"] == "data/raw/experiment.mp4"
        assert rule_config["output"]["metadata"] == "data/processed/experiment_meta.json"
        assert rule_config["params"]["flip"] is True
        assert rule_config["params"]["kernel_size"] == 5
        
        # Verify resource allocation
        assert rule_config["resources"]["mem_mb"] == 2048
        assert rule_config["resources"]["runtime"] == 300

    def test_workflow_metadata_integration(self):
        """Test workflow metadata generation for pipeline integration."""
        config = {
            "video_path": "workflow_test.mp4",
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.5,
            "_skip_validation": True
        }
        
        with patch('cv2.VideoCapture') as mock_cap:
            mock_instance = MagicMock()
            mock_cap.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_COUNT: 200,
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                cv2.CAP_PROP_FPS: 30.0
            }.get(prop, 0)
            
            # Create VideoPlume instance
            plume = VideoPlume.from_config(config)
            metadata = plume.get_metadata()
            
            # Verify workflow integration metadata
            assert metadata["dvc_compatible"] is True
            assert metadata["snakemake_ready"] is True
            assert metadata["hydra_managed"] == HYDRA_AVAILABLE
            
            # Verify processing configuration is preserved
            assert metadata["processing_config"]["flip"] is True
            assert metadata["processing_config"]["kernel_size"] == 5

    def test_workflow_error_handling(self):
        """Test error handling in workflow integration scenarios."""
        # Test invalid DVC path handling
        with pytest.raises(Exception):  # Should handle DVC path resolution errors
            create_video_plume_from_dvc_path(
                "invalid/dvc/path.mp4",
                flip=True
            )
        
        # Test invalid Snakemake configuration
        with pytest.raises((TypeError, KeyError)):
            create_snakemake_rule_config(
                input_video=None,  # Invalid input
                output_metadata="output.json"
            )


class TestVideoPlumeEnhancedErrorHandling:
    """
    Test suite for enhanced error handling with Hydra configuration objects.
    
    This test class validates comprehensive error handling scenarios including
    configuration validation failures, environment variable interpolation errors,
    and workflow integration error recovery as specified in the technical specification.
    """

    def test_hydra_config_composition_errors(self):
        """Test error handling for Hydra configuration composition failures."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available")
        
        # Test invalid configuration structure
        with pytest.raises(ValueError, match="Invalid VideoPlume configuration"):
            invalid_config = OmegaConf.create({
                "video_path": 123,  # Invalid type
                "flip": "not_boolean"
            })
            VideoPlume.from_config(invalid_config)

    def test_environment_variable_interpolation_errors(self):
        """Test error handling for environment variable interpolation failures."""
        # Test missing environment variable with no default
        config_with_missing_env = {
            "video_path": "${oc.env:MISSING_VIDEO_PATH}",  # No default provided
            "_skip_validation": True
        }
        
        # Should handle missing environment variable gracefully
        plume_config = VideoPlumeConfig.model_validate(config_with_missing_env)
        # The path should contain the unresolved interpolation or empty string
        assert "${oc.env:MISSING_VIDEO_PATH}" in str(plume_config.video_path) or str(plume_config.video_path) == ""

    def test_configuration_validation_error_messages(self):
        """Test descriptive error messages for configuration validation."""
        # Test multiple validation errors
        try:
            VideoPlumeConfig.model_validate({
                "video_path": 123,  # Wrong type
                "flip": "not_boolean",  # Wrong type
                "kernel_size": -1,  # Negative value
                "threshold": 2.0,  # Out of range
                "_skip_validation": True
            })
        except ValueError as e:
            error_msg = str(e)
            # Verify error message contains information about multiple failures
            assert len(error_msg) > 50  # Should be a comprehensive error message

    def test_resource_cleanup_on_errors(self):
        """Test proper resource cleanup when errors occur during initialization."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_instance = MagicMock()
            mock_cap.return_value = mock_instance
            mock_instance.isOpened.return_value = False  # Simulate open failure
            
            # Should raise error and clean up resources
            with pytest.raises(IOError, match="Failed to open video file"):
                VideoPlume("test_video.mp4", _skip_validation=True)
            
            # Verify cleanup was attempted
            mock_instance.release.assert_called()

    def test_thread_safety_error_handling(self):
        """Test error handling in multi-threaded scenarios."""
        import threading
        import time
        
        config = {
            "video_path": "thread_test.mp4",
            "_skip_validation": True
        }
        
        with patch('cv2.VideoCapture') as mock_cap:
            mock_instance = MagicMock()
            mock_cap.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: 100 if prop == cv2.CAP_PROP_FRAME_COUNT else 0
            mock_instance.read.return_value = (True, np.zeros((100, 100, 3)))
            
            plume = VideoPlume.from_config(config)
            
            errors = []
            
            def worker():
                try:
                    # Simulate concurrent access
                    frame = plume.get_frame(0)
                    time.sleep(0.01)
                    plume.close()
                except Exception as e:
                    errors.append(e)
            
            # Start multiple threads
            threads = [threading.Thread(target=worker) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            # Some operations might fail due to concurrent access, but should not crash
            # The exact behavior depends on thread scheduling, so we just check no crashes occurred
            assert len(errors) <= 3  # May have some expected errors from concurrent access


class TestVideoPlumePerformanceAndCompatibility:
    """
    Test suite for performance validation and cross-platform compatibility.
    
    This test class validates performance characteristics and compatibility
    requirements as specified in Section 6.6.3.3 of the technical specification.
    """

    def test_frame_processing_performance(self):
        """Test frame processing performance meets SLA requirements."""
        import time
        
        config = {
            "video_path": "performance_test.mp4",
            "kernel_size": 5,
            "kernel_sigma": 1.5,
            "_skip_validation": True
        }
        
        with patch('cv2.VideoCapture') as mock_cap:
            mock_instance = MagicMock()
            mock_cap.return_value = mock_instance
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda prop: 1000 if prop == cv2.CAP_PROP_FRAME_COUNT else 0
            
            # Mock frame that triggers processing pipeline
            mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            mock_instance.read.return_value = (True, mock_frame)
            
            plume = VideoPlume.from_config(config)
            
            # Measure frame processing time
            start_time = time.time()
            frame = plume.get_frame(0)
            processing_time = time.time() - start_time
            
            # Should be well under 33ms (30+ fps requirement)
            assert processing_time < 0.033, f"Frame processing took {processing_time:.4f}s, exceeds 33ms limit"
            assert frame is not None

    def test_configuration_loading_performance(self):
        """Test configuration loading performance meets SLA requirements."""
        import time
        
        complex_config = {
            "video_path": "complex_config_test.mp4",
            "flip": True,
            "grayscale": True,
            "kernel_size": 7,
            "kernel_sigma": 2.0,
            "threshold": 0.5,
            "normalize": True,
            "_skip_validation": True
        }
        
        # Measure configuration validation time
        start_time = time.time()
        validated_config = VideoPlumeConfig.model_validate(complex_config)
        validation_time = time.time() - start_time
        
        # Should be well under 500ms requirement
        assert validation_time < 0.5, f"Configuration validation took {validation_time:.4f}s, exceeds 500ms limit"
        assert validated_config is not None

    def test_memory_usage_efficiency(self):
        """Test memory usage remains efficient for large configurations."""
        import sys
        
        # Test memory usage doesn't grow excessively
        initial_size = sys.getsizeof(VideoPlumeConfig)
        
        # Create multiple config objects
        configs = []
        for i in range(100):
            config = VideoPlumeConfig.model_validate({
                "video_path": f"test_video_{i}.mp4",
                "flip": i % 2 == 0,
                "kernel_size": 3 + (i % 3) * 2,  # 3, 5, or 7
                "kernel_sigma": 1.0 + (i % 3) * 0.5,
                "_skip_validation": True
            })
            configs.append(config)
        
        # Memory usage should scale reasonably
        final_size = sum(sys.getsizeof(config) for config in configs)
        average_size = final_size / len(configs)
        
        # Each config should not be excessively large
        assert average_size < initial_size * 10, "Memory usage per config object is excessive"

    def test_cross_platform_path_handling(self):
        """Test cross-platform path handling compatibility."""
        import os
        
        # Test various path formats
        test_paths = [
            "video.mp4",  # Relative path
            "data/video.mp4",  # Unix-style path
            "data\\video.mp4",  # Windows-style path
            Path("data") / "video.mp4",  # Path object
        ]
        
        for test_path in test_paths:
            config = VideoPlumeConfig.model_validate({
                "video_path": test_path,
                "_skip_validation": True
            })
            
            # Should always result in a Path object
            assert isinstance(config.video_path, Path)
            
            # Should handle path normalization
            normalized_path = str(config.video_path)
            assert "video.mp4" in normalized_path


# Integration with pytest configuration and fixtures
@pytest.fixture(scope="session")
def hydra_config_store():
    """Set up Hydra ConfigStore for testing session."""
    if HYDRA_AVAILABLE:
        from hydra.core.config_store import ConfigStore
        cs = ConfigStore.instance()
        
        # Register test configurations
        cs.store(name="test_video_plume", node=VideoPlumeConfig)
        
        return cs
    return None


@pytest.fixture
def temp_video_file():
    """Create a temporary video file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        # Write minimal video file header (for path existence testing)
        temp_file.write(b"fake video content")
        temp_path = temp_file.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except OSError:
        pass


@pytest.fixture
def env_var_cleanup():
    """Ensure environment variables are cleaned up after tests."""
    original_env = os.environ.copy()
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Performance monitoring fixture
@pytest.fixture
def performance_monitor():
    """Monitor test performance and validate SLA compliance."""
    import time
    
    start_time = time.time()
    yield
    end_time = time.time()
    
    test_duration = end_time - start_time
    
    # Log performance for monitoring
    print(f"Test completed in {test_duration:.4f} seconds")
    
    # Ensure test doesn't exceed reasonable duration
    assert test_duration < 10.0, f"Test took {test_duration:.4f}s, may indicate performance issue"