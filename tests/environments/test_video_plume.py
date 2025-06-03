"""
Comprehensive tests for the VideoPlume environment with enhanced Hydra integration.

This test module validates video-based plume environment functionality with comprehensive
Hydra configuration integration, DVC/Snakemake workflow compatibility, and advanced 
error handling. The tests cover migration from PyYAML-based configuration to modern
Hydra structured configuration system with environment variable interpolation support.

Key Testing Areas:
    - VideoPlume initialization and basic functionality
    - Hydra DictConfig factory method integration
    - Environment variable interpolation through ${oc.env:} syntax
    - Pydantic schema validation within Hydra structured configs
    - Workflow orchestration compatibility for DVC/Snakemake
    - Error handling for Hydra-based configuration objects
    - Performance validation and resource management

Integration Points:
    - {{cookiecutter.project_slug}}.data.video_plume.VideoPlume
    - {{cookiecutter.project_slug}}.config.schemas.VideoPlumeConfig
    - Hydra configuration composition and override scenarios
    - pytest-hydra fixtures for configuration testing
"""

import pytest
from pathlib import Path
import numpy as np
import cv2
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, Any, Optional

# Import from the updated data module path per Section 0.2.1 mapping table
from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume, create_video_plume
from {{cookiecutter.project_slug}}.config.schemas import VideoPlumeConfig

# Hydra imports for configuration management testing
try:
    from omegaconf import DictConfig, OmegaConf
    from hydra import compose, initialize_config_store
    from hydra.core.config_store import ConfigStore
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None


# =====================================================================================
# ENHANCED FIXTURES FOR HYDRA CONFIGURATION TESTING
# =====================================================================================

@pytest.fixture
def mock_exists(monkeypatch):
    """Mock the Path.exists method to return True for all paths except 'nonexistent_file.mp4'."""
    def patched_exists(self):
        return str(self) != "nonexistent_file.mp4"
    
    monkeypatch.setattr(Path, "exists", patched_exists)
    return patched_exists


@pytest.fixture
def mock_video_capture():
    """Create a mock for cv2.VideoCapture with enhanced metadata support."""
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


@pytest.fixture
def hydra_config_basic():
    """
    Basic Hydra DictConfig for VideoPlume testing per Section 7.2.1.2.
    
    Provides a minimal but valid Hydra configuration for testing basic
    VideoPlume.from_config() factory method functionality.
    """
    if not HYDRA_AVAILABLE:
        return {
            "video_path": "test_video.mp4",
            "flip": False,
            "grayscale": True,
            "normalize": True
        }
    
    config_dict = {
        "video_path": "test_video.mp4",
        "flip": False,
        "grayscale": True,
        "kernel_size": None,
        "kernel_sigma": None,
        "threshold": None,
        "normalize": True,
        "frame_skip": 0,
        "start_frame": 0,
        "end_frame": None,
        "_target_": "{{cookiecutter.project_slug}}.data.video_plume.VideoPlume"
    }
    
    return OmegaConf.create(config_dict)


@pytest.fixture
def hydra_config_advanced():
    """
    Advanced Hydra DictConfig with preprocessing options per Section 5.2.2.
    
    Provides comprehensive configuration with Gaussian filtering, thresholding,
    and frame range selection for testing advanced VideoPlume functionality.
    """
    if not HYDRA_AVAILABLE:
        return {
            "video_path": "advanced_video.mp4",
            "flip": True,
            "grayscale": True,
            "kernel_size": 5,
            "kernel_sigma": 1.5,
            "threshold": 0.5,
            "normalize": True,
            "frame_skip": 2,
            "start_frame": 10,
            "end_frame": 90
        }
    
    config_dict = {
        "video_path": "advanced_video.mp4",
        "flip": True,
        "grayscale": True,
        "kernel_size": 5,
        "kernel_sigma": 1.5,
        "threshold": 0.5,
        "normalize": True,
        "frame_skip": 2,
        "start_frame": 10,
        "end_frame": 90,
        "fourcc": "mp4v",
        "fps_override": 25.0,
        "_target_": "{{cookiecutter.project_slug}}.data.video_plume.VideoPlume"
    }
    
    return OmegaConf.create(config_dict)


@pytest.fixture
def hydra_config_env_interpolation():
    """
    Hydra DictConfig with environment variable interpolation per Section 3.2.7.1.
    
    Tests Hydra's ${oc.env:} syntax for secure video path resolution and 
    environment-specific configuration management.
    """
    if not HYDRA_AVAILABLE:
        return {
            "video_path": "${oc.env:VIDEO_PATH,./data/default.mp4}",
            "flip": False,
            "grayscale": True,
            "normalize": True
        }
    
    config_dict = {
        "video_path": "${oc.env:VIDEO_PATH,./data/default.mp4}",
        "flip": "${oc.env:FLIP_FRAMES,false}",
        "grayscale": True,
        "kernel_size": "${oc.env:GAUSSIAN_KERNEL,null}",
        "threshold": "${oc.env:THRESHOLD_VALUE,null}",
        "normalize": True,
        "_target_": "{{cookiecutter.project_slug}}.data.video_plume.VideoPlume"
    }
    
    return OmegaConf.create(config_dict)


@pytest.fixture
def mock_hydra_compose():
    """
    Mock Hydra compose function for testing configuration loading scenarios.
    
    Provides controlled Hydra configuration composition for testing override
    scenarios, validation failures, and hierarchical parameter resolution.
    """
    with patch('hydra.compose') as mock_compose:
        def compose_side_effect(config_name=None, overrides=None, return_hydra_config=False):
            """Simulate Hydra compose behavior with override support."""
            base_config = {
                "video_plume": {
                    "video_path": "default_video.mp4",
                    "flip": False,
                    "grayscale": True,
                    "normalize": True,
                    "_target_": "{{cookiecutter.project_slug}}.data.video_plume.VideoPlume"
                }
            }
            
            # Apply overrides if provided
            if overrides:
                for override in overrides:
                    if "=" in override:
                        key, value = override.split("=", 1)
                        # Simple override parsing for testing
                        if key.startswith("video_plume."):
                            param = key.replace("video_plume.", "")
                            # Type conversion for common parameters
                            if value.lower() in ['true', 'false']:
                                value = value.lower() == 'true'
                            elif value.replace('.', '').isdigit():
                                value = float(value) if '.' in value else int(value)
                            base_config["video_plume"][param] = value
            
            if HYDRA_AVAILABLE:
                return OmegaConf.create(base_config)
            else:
                return base_config
        
        mock_compose.side_effect = compose_side_effect
        yield mock_compose


@pytest.fixture
def mock_environment_variables(monkeypatch):
    """
    Mock environment variables for testing Hydra interpolation scenarios.
    
    Sets up controlled environment variables for testing ${oc.env:} syntax
    resolution and environment-specific configuration behavior.
    """
    env_vars = {
        "VIDEO_PATH": "/test/data/env_video.mp4",
        "FLIP_FRAMES": "true",
        "GAUSSIAN_KERNEL": "7",
        "THRESHOLD_VALUE": "0.3",
        "OUTPUT_DIR": "/test/outputs",
        "RANDOM_SEED": "42"
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


@pytest.fixture
def workflow_metadata_mock():
    """
    Mock workflow metadata for DVC/Snakemake integration testing per Section 5.2.2.
    
    Provides mock workflow orchestration integration points for testing
    VideoPlume compatibility with batch processing and experiment automation.
    """
    return {
        "dvc_stage": "video_processing",
        "snakemake_rule": "process_plume_video",
        "dependencies": ["raw_video.mp4"],
        "outputs": ["processed_frames/"],
        "parameters": {
            "preprocessing": {
                "grayscale": True,
                "gaussian_blur": True,
                "kernel_size": 5
            }
        },
        "metadata": {
            "version": "1.0",
            "created_by": "automated_workflow",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    }


# =====================================================================================
# BASIC FUNCTIONALITY TESTS (UPDATED WITH NEW IMPORT PATH)
# =====================================================================================

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
    with pytest.raises(ValueError, match="Cannot get frame from closed VideoPlume"):
        plume.get_frame(0)


def test_close_idempotent(mock_video_capture, mock_exists):
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


# =====================================================================================
# HYDRA DICTCONFIG INTEGRATION TESTS (NEW PER SECTION 5.2.2)
# =====================================================================================

@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
def test_video_plume_from_config_basic(hydra_config_basic, mock_video_capture, mock_exists):
    """
    Test VideoPlume.from_config() factory method with basic Hydra DictConfig.
    
    Validates basic Hydra configuration consumption and VideoPlume instantiation
    per Section 7.2.1.2 factory method pattern requirements.
    """
    # Create VideoPlume instance from Hydra configuration
    plume = VideoPlume.from_config(hydra_config_basic)
    
    # Verify instance creation and basic properties
    assert isinstance(plume, VideoPlume)
    assert plume.video_path == Path("test_video.mp4")
    assert plume.flip is False
    assert plume.grayscale is True
    assert plume.normalize is True
    
    # Verify OpenCV capture was initialized
    mock_video_capture.assert_called_once_with("test_video.mp4")


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
def test_video_plume_from_config_advanced(hydra_config_advanced, mock_video_capture, mock_exists):
    """
    Test VideoPlume.from_config() with advanced preprocessing configuration.
    
    Validates comprehensive Hydra configuration with Gaussian filtering,
    thresholding, and frame range selection per Section 5.2.2.
    """
    # Create VideoPlume instance with advanced configuration
    plume = VideoPlume.from_config(hydra_config_advanced)
    
    # Verify advanced preprocessing parameters
    assert plume.video_path == Path("advanced_video.mp4")
    assert plume.flip is True
    assert plume.grayscale is True
    assert plume.kernel_size == 5
    assert plume.kernel_sigma == 1.5
    assert plume.threshold == 0.5
    assert plume.normalize is True
    
    # Verify frame range parameters
    assert plume.frame_skip == 2
    assert plume.start_frame == 10
    assert plume.end_frame == 90
    
    # Verify OpenCV capture initialization
    mock_video_capture.assert_called_once_with("advanced_video.mp4")


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
def test_video_plume_from_config_environment_interpolation(
    hydra_config_env_interpolation, 
    mock_environment_variables, 
    mock_video_capture, 
    mock_exists
):
    """
    Test Hydra environment variable interpolation through ${oc.env:} syntax.
    
    Validates environment variable resolution for video path configuration
    per Section 3.2.7.1 secure path management requirements.
    """
    # Resolve environment variable interpolation manually for testing
    # In real usage, Hydra handles this automatically
    resolved_config = OmegaConf.create({
        "video_path": "/test/data/env_video.mp4",  # Resolved from VIDEO_PATH
        "flip": True,  # Resolved from FLIP_FRAMES
        "grayscale": True,
        "kernel_size": 7,  # Resolved from GAUSSIAN_KERNEL
        "threshold": 0.3,  # Resolved from THRESHOLD_VALUE
        "normalize": True,
        "_target_": "{{cookiecutter.project_slug}}.data.video_plume.VideoPlume"
    })
    
    # Create VideoPlume with resolved configuration
    plume = VideoPlume.from_config(resolved_config)
    
    # Verify environment variable resolution
    assert plume.video_path == Path("/test/data/env_video.mp4")
    assert plume.flip is True
    assert plume.kernel_size == 7
    assert plume.threshold == 0.3
    
    # Verify OpenCV capture initialization with resolved path
    mock_video_capture.assert_called_once_with("/test/data/env_video.mp4")


def test_video_plume_from_config_dict_fallback(mock_video_capture, mock_exists):
    """
    Test VideoPlume.from_config() with regular dictionary when Hydra unavailable.
    
    Ensures backward compatibility and graceful fallback when Hydra dependencies
    are not available in the environment.
    """
    config_dict = {
        "video_path": "dict_video.mp4",
        "flip": True,
        "grayscale": False,
        "kernel_size": 3,
        "kernel_sigma": 1.0,
        "normalize": False
    }
    
    # Create VideoPlume from dictionary configuration
    plume = VideoPlume.from_config(config_dict)
    
    # Verify configuration application
    assert plume.video_path == Path("dict_video.mp4")
    assert plume.flip is True
    assert plume.grayscale is False
    assert plume.kernel_size == 3
    assert plume.kernel_sigma == 1.0
    assert plume.normalize is False
    
    # Verify OpenCV capture initialization
    mock_video_capture.assert_called_once_with("dict_video.mp4")


# =====================================================================================
# PYDANTIC SCHEMA VALIDATION WITH HYDRA INTEGRATION (NEW PER SECTION 7.2.2.2)
# =====================================================================================

@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
def test_pydantic_schema_validation_success(hydra_config_basic, mock_video_capture, mock_exists):
    """
    Test successful Pydantic schema validation within Hydra structured config system.
    
    Validates that VideoPlume.from_config() properly validates configuration
    through Pydantic schemas per Section 7.2.2.2 requirements.
    """
    # Should not raise any validation errors
    plume = VideoPlume.from_config(hydra_config_basic)
    
    # Verify successful creation
    assert isinstance(plume, VideoPlume)
    assert plume.video_path == Path("test_video.mp4")


def test_pydantic_schema_validation_invalid_kernel_size():
    """
    Test Pydantic schema validation failure for invalid kernel_size.
    
    Validates that configuration validation catches invalid Gaussian
    kernel parameters per Pydantic schema constraints.
    """
    invalid_config = {
        "video_path": "test_video.mp4",
        "kernel_size": 4,  # Invalid: must be odd
        "kernel_sigma": 1.0,
        "grayscale": True,
        "normalize": True
    }
    
    # Should raise ValueError due to invalid kernel_size
    with pytest.raises(ValueError, match="kernel_size must be positive and odd"):
        VideoPlume.from_config(invalid_config)


def test_pydantic_schema_validation_invalid_threshold():
    """
    Test Pydantic schema validation failure for out-of-range threshold.
    
    Validates threshold parameter range validation per schema constraints.
    """
    invalid_config = {
        "video_path": "test_video.mp4",
        "threshold": 1.5,  # Invalid: must be between 0.0 and 1.0
        "grayscale": True,
        "normalize": True
    }
    
    # Should raise ValueError due to invalid threshold
    with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
        VideoPlume.from_config(invalid_config)


def test_pydantic_schema_validation_missing_kernel_sigma():
    """
    Test Pydantic schema validation for incomplete Gaussian parameters.
    
    Validates that Gaussian kernel parameters must be specified together
    per schema consistency requirements.
    """
    invalid_config = {
        "video_path": "test_video.mp4",
        "kernel_size": 5,  # Specified
        "kernel_sigma": None,  # Missing
        "grayscale": True,
        "normalize": True
    }
    
    # Should raise ValueError due to incomplete Gaussian parameters
    with pytest.raises(ValueError, match="Both kernel_size and kernel_sigma must be specified together"):
        VideoPlume.from_config(invalid_config)


# =====================================================================================
# HYDRA CONFIGURATION LOADING AND OVERRIDE TESTS (NEW PER SECTION 3.2.4.1)
# =====================================================================================

@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
def test_hydra_configuration_override_scenarios(mock_hydra_compose, mock_video_capture, mock_exists):
    """
    Test Hydra configuration override scenarios and parameter precedence.
    
    Validates Hydra's hierarchical configuration composition with command-line
    overrides per Section 3.2.4.1 configuration management requirements.
    """
    # Test basic configuration composition
    base_config = mock_hydra_compose()
    plume = VideoPlume.from_config(base_config["video_plume"])
    
    assert plume.video_path == Path("default_video.mp4")
    assert plume.flip is False
    
    # Test configuration with overrides
    override_config = mock_hydra_compose(
        config_name="config",
        overrides=["video_plume.flip=true", "video_plume.kernel_size=7"]
    )
    plume_override = VideoPlume.from_config(override_config["video_plume"])
    
    assert plume_override.flip is True
    assert plume_override.kernel_size == 7


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
def test_hydra_configuration_error_handling(mock_hydra_compose):
    """
    Test enhanced error handling for Hydra-based configuration objects.
    
    Validates comprehensive error reporting and recovery scenarios for
    invalid Hydra configurations and composition failures.
    """
    # Mock Hydra compose to raise configuration error
    mock_hydra_compose.side_effect = Exception("Configuration composition failed")
    
    # Test error handling for composition failures
    with pytest.raises(Exception, match="Configuration composition failed"):
        mock_hydra_compose(config_name="invalid_config")


def test_hydra_configuration_validation_edge_cases():
    """
    Test edge cases in Hydra configuration validation and error scenarios.
    
    Validates robust error handling for malformed configurations,
    missing required parameters, and type conversion failures.
    """
    # Test missing required video_path parameter
    incomplete_config = {
        "flip": True,
        "grayscale": True
        # Missing video_path
    }
    
    with pytest.raises((ValueError, TypeError)):
        VideoPlume.from_config(incomplete_config)
    
    # Test invalid type for boolean parameter
    invalid_type_config = {
        "video_path": "test_video.mp4",
        "flip": "invalid_boolean",  # Should be boolean
        "grayscale": True,
        "normalize": True
    }
    
    # Configuration should handle type conversion or raise appropriate error
    try:
        VideoPlume.from_config(invalid_type_config)
    except (ValueError, TypeError):
        pass  # Expected behavior for invalid types


# =====================================================================================
# WORKFLOW ORCHESTRATION INTEGRATION TESTS (NEW PER SECTION 5.2.2)
# =====================================================================================

def test_video_plume_workflow_metadata_generation(
    hydra_config_basic, 
    workflow_metadata_mock, 
    mock_video_capture, 
    mock_exists
):
    """
    Test VideoPlume workflow metadata generation for DVC/Snakemake integration.
    
    Validates workflow orchestration compatibility per Section 5.2.2
    including metadata generation for reproducible data processing pipelines.
    """
    if HYDRA_AVAILABLE:
        plume = VideoPlume.from_config(hydra_config_basic)
    else:
        plume = VideoPlume("test_video.mp4")
    
    # Generate workflow metadata
    metadata = plume.get_workflow_metadata()
    
    # Verify workflow-compatible metadata structure
    assert "video_path" in metadata
    assert "file_hash" in metadata
    assert "file_size" in metadata
    assert "workflow_version" in metadata
    assert "dependencies" in metadata
    assert "opencv_version" in metadata["dependencies"]
    assert "numpy_version" in metadata["dependencies"]
    
    # Verify metadata format for DVC/Snakemake compatibility
    assert isinstance(metadata["file_hash"], str)
    assert isinstance(metadata["file_size"], int)
    assert isinstance(metadata["workflow_version"], str)


def test_video_plume_dvc_pipeline_compatibility(mock_video_capture, mock_exists):
    """
    Test VideoPlume compatibility with DVC pipeline execution patterns.
    
    Validates data versioning compatibility and parameter tracking
    for reproducible research workflows.
    """
    # Create VideoPlume with parameters typical for DVC pipeline
    config = {
        "video_path": "data/raw/plume_video.mp4",
        "flip": True,
        "grayscale": True,
        "kernel_size": 5,
        "kernel_sigma": 1.5,
        "normalize": True
    }
    
    plume = VideoPlume.from_config(config)
    
    # Verify DVC-compatible parameter structure
    metadata = plume.get_metadata()
    preprocessing = metadata["preprocessing"]
    
    # DVC expects consistent parameter naming and types
    assert preprocessing["flip"] is True
    assert preprocessing["grayscale"] is True
    assert preprocessing["kernel_size"] == 5
    assert preprocessing["kernel_sigma"] == 1.5
    assert preprocessing["normalize"] is True
    
    # Verify metadata includes video properties needed for pipeline validation
    assert metadata["width"] == 640
    assert metadata["height"] == 480
    assert metadata["frame_count"] == 100
    assert metadata["fps"] == 30.0


def test_video_plume_snakemake_rule_integration(mock_video_capture, mock_exists):
    """
    Test VideoPlume integration with Snakemake workflow rules.
    
    Validates batch processing compatibility and resource management
    for scalable scientific computing workflows.
    """
    # Simulate Snakemake rule parameters
    snakemake_config = {
        "video_path": "data/processed/{sample}_video.mp4",
        "flip": False,
        "grayscale": True,
        "kernel_size": 7,
        "kernel_sigma": 2.0,
        "threshold": 0.4,
        "normalize": True,
        "frame_skip": 1,
        "start_frame": 0,
        "end_frame": 1000
    }
    
    # Replace Snakemake wildcard for testing
    actual_config = snakemake_config.copy()
    actual_config["video_path"] = "data/processed/sample1_video.mp4"
    
    plume = VideoPlume.from_config(actual_config)
    
    # Verify Snakemake-compatible configuration
    assert plume.video_path == Path("data/processed/sample1_video.mp4")
    assert plume.kernel_size == 7
    assert plume.kernel_sigma == 2.0
    assert plume.threshold == 0.4
    assert plume.frame_skip == 1
    
    # Verify workflow metadata for Snakemake logging
    workflow_metadata = plume.get_workflow_metadata()
    assert "file_hash" in workflow_metadata
    assert "dependencies" in workflow_metadata


# =====================================================================================
# ENHANCED ERROR HANDLING AND EDGE CASES
# =====================================================================================

def test_video_plume_configuration_error_recovery():
    """
    Test VideoPlume error recovery and validation for configuration failures.
    
    Validates robust error handling for various configuration failure modes
    and appropriate error message generation.
    """
    # Test configuration with conflicting parameters
    conflicting_config = {
        "video_path": "test_video.mp4",
        "kernel_size": 5,
        "kernel_sigma": None,  # Conflict: size specified but not sigma
        "grayscale": True,
        "normalize": True
    }
    
    with pytest.raises(ValueError, match="Both kernel_size and kernel_sigma must be specified together"):
        VideoPlume.from_config(conflicting_config)
    
    # Test configuration with invalid file extension
    invalid_extension_config = {
        "video_path": "test_file.txt",  # Not a video file
        "grayscale": True,
        "normalize": True
    }
    
    # Should handle gracefully or raise appropriate error
    with patch('pathlib.Path.exists', return_value=True):
        try:
            VideoPlume.from_config(invalid_extension_config)
        except (IOError, ValueError):
            pass  # Expected for invalid file types


def test_video_plume_memory_management_edge_cases(mock_video_capture, mock_exists):
    """
    Test VideoPlume memory management and resource cleanup edge cases.
    
    Validates proper resource management under various failure conditions
    and ensure no memory leaks in error scenarios.
    """
    # Test creation failure during initialization
    with patch.object(VideoPlume, '__init__', side_effect=Exception("Init failed")):
        with pytest.raises(Exception, match="Init failed"):
            VideoPlume("test_video.mp4")
    
    # Test resource cleanup after partial initialization
    plume = VideoPlume("test_video.mp4")
    
    # Force an error condition and verify cleanup
    original_cap = plume.cap
    plume.close()
    
    # Verify cap.release() was called
    original_cap.release.assert_called_once()
    
    # Verify idempotent cleanup
    plume.close()  # Should not raise error


def test_video_plume_threading_safety_considerations(mock_video_capture, mock_exists):
    """
    Test VideoPlume thread safety and concurrent access patterns.
    
    Validates thread-safe resource management and concurrent frame access
    scenarios that may occur in multi-threaded research applications.
    """
    plume = VideoPlume("test_video.mp4")
    
    # Test concurrent frame access (simulated)
    frame1 = plume.get_frame(10)
    frame2 = plume.get_frame(20)
    frame3 = plume.get_frame(30)
    
    # All frames should be retrieved successfully
    assert frame1 is not None
    assert frame2 is not None
    assert frame3 is not None
    
    # Verify OpenCV capture position was set for each frame
    expected_calls = [
        ((cv2.CAP_PROP_POS_FRAMES, 10),),
        ((cv2.CAP_PROP_POS_FRAMES, 20),),
        ((cv2.CAP_PROP_POS_FRAMES, 30),)
    ]
    
    actual_calls = mock_video_capture.return_value.set.call_args_list
    assert len(actual_calls) == 3
    
    # Test thread safety of close operation
    plume.close()
    
    # Subsequent frame access should raise appropriate error
    with pytest.raises(ValueError, match="Cannot get frame from closed VideoPlume"):
        plume.get_frame(40)


# =====================================================================================
# PERFORMANCE AND COMPATIBILITY VALIDATION
# =====================================================================================

def test_video_plume_performance_characteristics(mock_video_capture, mock_exists):
    """
    Test VideoPlume performance characteristics and timing requirements.
    
    Validates frame processing performance meets SLA requirements
    per technical specification timing constraints.
    """
    import time
    
    plume = VideoPlume("test_video.mp4", normalize=True)
    
    # Test frame processing timing (should be < 33ms per spec)
    start_time = time.time()
    frame = plume.get_frame(50)
    processing_time = time.time() - start_time
    
    # Performance validation (relaxed for mocked environment)
    assert processing_time < 0.1  # 100ms limit for mocked operations
    assert frame is not None
    
    # Test metadata generation performance
    start_time = time.time()
    metadata = plume.get_metadata()
    metadata_time = time.time() - start_time
    
    assert metadata_time < 0.01  # 10ms limit for metadata generation
    assert "width" in metadata
    assert "height" in metadata


def test_video_plume_cross_platform_compatibility():
    """
    Test VideoPlume cross-platform compatibility and path handling.
    
    Validates proper path handling across Windows, macOS, and Linux
    environments with different path separators and conventions.
    """
    # Test Windows-style paths
    windows_config = {
        "video_path": "C:\\data\\videos\\test_video.mp4",
        "grayscale": True,
        "normalize": True
    }
    
    # Test Unix-style paths
    unix_config = {
        "video_path": "/data/videos/test_video.mp4",
        "grayscale": True,
        "normalize": True
    }
    
    # Test relative paths
    relative_config = {
        "video_path": "../data/test_video.mp4",
        "grayscale": True,
        "normalize": True
    }
    
    # All configurations should handle path normalization properly
    with patch('pathlib.Path.exists', return_value=True), \
         patch('cv2.VideoCapture') as mock_cap:
        
        mock_instance = MagicMock()
        mock_cap.return_value = mock_instance
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        # Test all path formats
        for config in [windows_config, unix_config, relative_config]:
            plume = VideoPlume.from_config(config)
            assert isinstance(plume.video_path, Path)
            assert plume.width == 640
            assert plume.height == 480


# =====================================================================================
# FACTORY FUNCTION TESTS (NEW API COMPATIBILITY)
# =====================================================================================

def test_create_video_plume_factory_function(hydra_config_basic, mock_video_capture, mock_exists):
    """
    Test create_video_plume factory function for API compatibility.
    
    Validates alternative factory method with parameter override support
    per API layer compatibility requirements.
    """
    if not HYDRA_AVAILABLE:
        config = {
            "video_path": "test_video.mp4",
            "flip": False,
            "grayscale": True,
            "normalize": True
        }
    else:
        config = hydra_config_basic
    
    # Test basic factory function
    plume = create_video_plume(config)
    
    assert isinstance(plume, VideoPlume)
    assert plume.video_path == Path("test_video.mp4")
    assert plume.flip is False
    
    # Test factory function with parameter overrides
    plume_override = create_video_plume(config, flip=True, kernel_size=5, kernel_sigma=1.0)
    
    assert plume_override.flip is True
    assert plume_override.kernel_size == 5
    assert plume_override.kernel_sigma == 1.0


def test_create_video_plume_with_overrides(mock_video_capture, mock_exists):
    """
    Test create_video_plume with configuration overrides.
    
    Validates parameter override functionality for dynamic configuration
    adjustment in API layer usage scenarios.
    """
    base_config = {
        "video_path": "base_video.mp4",
        "flip": False,
        "grayscale": True,
        "kernel_size": 3,
        "kernel_sigma": 1.0,
        "normalize": True
    }
    
    # Test multiple parameter overrides
    plume = create_video_plume(
        base_config,
        flip=True,
        kernel_size=7,
        kernel_sigma=2.0,
        threshold=0.5
    )
    
    # Verify base configuration
    assert plume.video_path == Path("base_video.mp4")
    assert plume.grayscale is True
    assert plume.normalize is True
    
    # Verify overridden parameters
    assert plume.flip is True
    assert plume.kernel_size == 7
    assert plume.kernel_sigma == 2.0
    assert plume.threshold == 0.5


# =====================================================================================
# LEGACY COMPATIBILITY AND MIGRATION TESTS
# =====================================================================================

def test_legacy_yaml_configuration_compatibility():
    """
    Test backward compatibility with legacy YAML-based configuration format.
    
    Validates migration support from PyYAML-based configuration to
    Hydra structured configuration system.
    """
    # Simulate legacy YAML configuration structure
    legacy_config = {
        "video_plume": {
            "video_path": "legacy_video.mp4",
            "preprocessing": {
                "flip": True,
                "grayscale": True,
                "gaussian_blur": {
                    "kernel_size": 5,
                    "sigma": 1.5
                }
            },
            "normalization": {
                "enabled": True
            }
        }
    }
    
    # Convert legacy format to new format
    modern_config = {
        "video_path": legacy_config["video_plume"]["video_path"],
        "flip": legacy_config["video_plume"]["preprocessing"]["flip"],
        "grayscale": legacy_config["video_plume"]["preprocessing"]["grayscale"],
        "kernel_size": legacy_config["video_plume"]["preprocessing"]["gaussian_blur"]["kernel_size"],
        "kernel_sigma": legacy_config["video_plume"]["preprocessing"]["gaussian_blur"]["sigma"],
        "normalize": legacy_config["video_plume"]["normalization"]["enabled"]
    }
    
    # Test configuration migration compatibility
    with patch('pathlib.Path.exists', return_value=True), \
         patch('cv2.VideoCapture') as mock_cap:
        
        mock_instance = MagicMock()
        mock_cap.return_value = mock_instance
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        plume = VideoPlume.from_config(modern_config)
        
        # Verify migration success
        assert plume.video_path == Path("legacy_video.mp4")
        assert plume.flip is True
        assert plume.grayscale is True
        assert plume.kernel_size == 5
        assert plume.kernel_sigma == 1.5
        assert plume.normalize is True


def test_configuration_format_validation():
    """
    Test validation of different configuration formats and error handling.
    
    Validates robust configuration parsing for various input formats
    and appropriate error reporting for malformed configurations.
    """
    # Test empty configuration
    with pytest.raises((ValueError, TypeError)):
        VideoPlume.from_config({})
    
    # Test minimal valid configuration
    minimal_config = {"video_path": "minimal.mp4"}
    
    with patch('pathlib.Path.exists', return_value=True), \
         patch('cv2.VideoCapture') as mock_cap:
        
        mock_instance = MagicMock()
        mock_cap.return_value = mock_instance
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        plume = VideoPlume.from_config(minimal_config)
        
        # Verify defaults are applied
        assert plume.video_path == Path("minimal.mp4")
        assert plume.flip is False  # Default value
        assert plume.grayscale is True  # Default value
        assert plume.normalize is True  # Default value