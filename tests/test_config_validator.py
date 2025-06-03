"""Tests for the configuration validation module using Pydantic schemas."""

import pytest
from pydantic import ValidationError
from {{cookiecutter.project_slug}}.config.schemas import (
    NavigatorConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    VideoPlumeConfig,
)


class TestVideoPlumeConfigValidation:
    """Test validation of VideoPlumeConfig Pydantic model."""

    def test_missing_required_fields(self):
        """Test error when required fields are missing."""
        # Missing video_path
        with pytest.raises(ValidationError, match="video_path"):
            VideoPlumeConfig()

    def test_valid_minimal_config(self):
        """Test that a minimal valid config passes validation."""
        config = VideoPlumeConfig(video_path="/path/to/video.mp4")
        assert config.video_path == "/path/to/video.mp4"
        assert config.flip is False  # default value
        assert config.grayscale is True  # default value

    def test_invalid_field_types(self):
        """Test error when field types are invalid."""
        # Invalid flip type
        with pytest.raises(ValidationError, match="flip"):
            VideoPlumeConfig(video_path="/path/to/video.mp4", flip="not-a-bool")
            
        # Invalid kernel_size type
        with pytest.raises(ValidationError, match="kernel_size"):
            VideoPlumeConfig(video_path="/path/to/video.mp4", kernel_size=2.5)
            
        # Invalid kernel_sigma type
        with pytest.raises(ValidationError, match="kernel_sigma"):
            VideoPlumeConfig(video_path="/path/to/video.mp4", kernel_sigma="1.0")

    def test_invalid_kernel_size_values(self):
        """Test error when kernel_size values are invalid."""
        # kernel_size must be positive
        with pytest.raises(ValidationError, match="kernel_size must be positive"):
            VideoPlumeConfig(video_path="/path/to/video.mp4", kernel_size=-1)
            
        # kernel_size must be odd
        with pytest.raises(ValidationError, match="kernel_size must be odd"):
            VideoPlumeConfig(video_path="/path/to/video.mp4", kernel_size=2)
            
        # Zero kernel_size should fail
        with pytest.raises(ValidationError, match="kernel_size must be positive"):
            VideoPlumeConfig(video_path="/path/to/video.mp4", kernel_size=0)

    def test_valid_kernel_size_values(self):
        """Test that valid kernel_size values pass validation."""
        # Odd positive values should work
        config = VideoPlumeConfig(video_path="/path/to/video.mp4", kernel_size=3)
        assert config.kernel_size == 3
        
        config = VideoPlumeConfig(video_path="/path/to/video.mp4", kernel_size=5)
        assert config.kernel_size == 5

    def test_valid_complete_config(self):
        """Test that a complete valid config passes validation."""
        config = VideoPlumeConfig(
            video_path="/path/to/video.mp4",
            flip=True,
            grayscale=False,
            kernel_size=3,
            kernel_sigma=1.5,
            threshold=0.5,
            normalize=False
        )
        assert config.video_path == "/path/to/video.mp4"
        assert config.flip is True
        assert config.grayscale is False
        assert config.kernel_size == 3
        assert config.kernel_sigma == 1.5
        assert config.threshold == 0.5
        assert config.normalize is False

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed due to extra='allow' configuration."""
        config = VideoPlumeConfig(
            video_path="/path/to/video.mp4",
            extra_field="extra_value"
        )
        assert config.video_path == "/path/to/video.mp4"
        # Extra fields should be accessible via model_extra or __dict__ in Pydantic v2


class TestNavigatorConfigValidation:
    """Test validation of NavigatorConfig Pydantic model."""

    def test_default_single_agent_config(self):
        """Test that default single agent config is valid."""
        config = NavigatorConfig()
        assert config.orientation == 0.0
        assert config.speed == 0.0
        assert config.max_speed == 1.0
        assert config.angular_velocity == 0.0

    def test_invalid_field_types(self):
        """Test error when field types are invalid."""
        # Invalid orientation type
        with pytest.raises(ValidationError, match="orientation"):
            NavigatorConfig(orientation="invalid")
            
        # Invalid speed type
        with pytest.raises(ValidationError, match="speed"):
            NavigatorConfig(speed="invalid")
            
        # Invalid max_speed type
        with pytest.raises(ValidationError, match="max_speed"):
            NavigatorConfig(max_speed="invalid")

    def test_speed_exceeds_max_speed_validation(self):
        """Test error when speed exceeds max_speed."""
        with pytest.raises(ValidationError, match="speed.*cannot exceed max_speed"):
            NavigatorConfig(speed=2.0, max_speed=1.0)
            
        with pytest.raises(ValidationError, match="speed.*cannot exceed max_speed"):
            NavigatorConfig(speed=-2.0, max_speed=1.0)

    def test_valid_single_agent_config(self):
        """Test that a valid single agent config passes validation."""
        config = NavigatorConfig(
            position=(10.0, 20.0),
            orientation=45.0,
            speed=0.5,
            max_speed=2.0,
            angular_velocity=10.0
        )
        assert config.position == (10.0, 20.0)
        assert config.orientation == 45.0
        assert config.speed == 0.5
        assert config.max_speed == 2.0
        assert config.angular_velocity == 10.0

    def test_multi_agent_positions_validation(self):
        """Test validation of multi-agent positions."""
        # Valid multi-agent config
        config = NavigatorConfig(
            positions=[[0.0, 0.0], [1.0, 1.0]],
            orientations=[0.0, 90.0],
            num_agents=2
        )
        assert len(config.positions) == 2
        assert len(config.orientations) == 2

    def test_conflicting_single_and_multi_agent_params(self):
        """Test error when both single and multi-agent parameters are specified."""
        with pytest.raises(ValidationError, match="Cannot specify both single-agent and multi-agent parameters"):
            NavigatorConfig(
                position=(0.0, 0.0),  # single-agent
                positions=[[0.0, 0.0], [1.0, 1.0]]  # multi-agent
            )

    def test_mismatched_multi_agent_param_lengths(self):
        """Test error when multi-agent parameter arrays have mismatched lengths."""
        with pytest.raises(ValidationError, match="orientations length.*does not match number of agents"):
            NavigatorConfig(
                positions=[[0.0, 0.0], [1.0, 1.0]],  # 2 agents
                orientations=[0.0, 90.0, 180.0]  # 3 orientations
            )

    def test_valid_mixed_numeric_types(self):
        """Test that mixed integer and float types are valid."""
        config = NavigatorConfig(
            orientation=45,  # int
            speed=0.5,  # float
            max_speed=1  # int
        )
        assert config.orientation == 45
        assert config.speed == 0.5
        assert config.max_speed == 1


class TestSingleAgentConfigValidation:
    """Test validation of SingleAgentConfig Pydantic model."""

    def test_empty_config_valid(self):
        """Test that an empty SingleAgentConfig is valid."""
        config = SingleAgentConfig()
        assert config.position is None
        assert config.orientation is None
        assert config.speed is None
        assert config.max_speed is None
        assert config.angular_velocity is None

    def test_valid_partial_config(self):
        """Test that partial configuration is valid."""
        config = SingleAgentConfig(
            position=(5.0, 10.0),
            orientation=30.0
        )
        assert config.position == (5.0, 10.0)
        assert config.orientation == 30.0
        assert config.speed is None

    def test_valid_complete_config(self):
        """Test that complete configuration is valid."""
        config = SingleAgentConfig(
            position=(0.0, 0.0),
            orientation=0.0,
            speed=1.0,
            max_speed=2.0,
            angular_velocity=5.0
        )
        assert config.position == (0.0, 0.0)
        assert config.orientation == 0.0
        assert config.speed == 1.0
        assert config.max_speed == 2.0
        assert config.angular_velocity == 5.0


class TestMultiAgentConfigValidation:
    """Test validation of MultiAgentConfig Pydantic model."""

    def test_empty_config_valid(self):
        """Test that an empty MultiAgentConfig is valid."""
        config = MultiAgentConfig()
        assert config.positions is None
        assert config.orientations is None
        assert config.num_agents is None

    def test_positions_validation(self):
        """Test validation of positions field."""
        # Valid positions
        config = MultiAgentConfig(positions=[[0.0, 0.0], [1.0, 1.0]])
        assert len(config.positions) == 2
        
        # Invalid positions - not a list
        with pytest.raises(ValidationError, match="positions must be a list"):
            MultiAgentConfig(positions="not-a-list")
        
        # Invalid positions - wrong format
        with pytest.raises(ValidationError, match="Each position must be a list of"):
            MultiAgentConfig(positions=[[0.0], [1.0, 1.0]])
        
        # Invalid positions - wrong element type
        with pytest.raises(ValidationError, match="Each position must be a list of"):
            MultiAgentConfig(positions=[(0.0, 0.0), [1.0, 1.0]])

    def test_valid_complete_multi_agent_config(self):
        """Test that complete multi-agent configuration is valid."""
        config = MultiAgentConfig(
            positions=[[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]],
            orientations=[0.0, 90.0, 180.0],
            speeds=[0.5, 1.0, 1.5],
            max_speeds=[1.0, 2.0, 3.0],
            angular_velocities=[5.0, 10.0, 15.0],
            num_agents=3
        )
        assert len(config.positions) == 3
        assert len(config.orientations) == 3
        assert len(config.speeds) == 3
        assert len(config.max_speeds) == 3
        assert len(config.angular_velocities) == 3
        assert config.num_agents == 3


class TestConfigIntegrationWithHydra:
    """Test configuration integration scenarios with Hydra patterns."""

    def test_config_dict_to_pydantic_conversion(self):
        """Test conversion from dictionary (typical Hydra output) to Pydantic models."""
        # Simulate Hydra configuration dictionary
        hydra_config = {
            "video_path": "/data/experiment.mp4",
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 2.0
        }
        
        # Convert to Pydantic model
        config = VideoPlumeConfig(**hydra_config)
        assert config.video_path == "/data/experiment.mp4"
        assert config.flip is True
        assert config.kernel_size == 5
        assert config.kernel_sigma == 2.0

    def test_navigator_config_from_hydra_dict(self):
        """Test NavigatorConfig creation from Hydra-style configuration."""
        hydra_navigator_config = {
            "position": [10.0, 20.0],
            "orientation": 45.0,
            "speed": 1.5,
            "max_speed": 3.0,
            "angular_velocity": 12.0
        }
        
        config = NavigatorConfig(**hydra_navigator_config)
        assert config.position == [10.0, 20.0]  # Pydantic converts to tuple
        assert config.orientation == 45.0
        assert config.speed == 1.5
        assert config.max_speed == 3.0

    def test_multi_agent_config_from_hydra_dict(self):
        """Test MultiAgentConfig creation from Hydra-style configuration."""
        hydra_multi_config = {
            "positions": [[0.0, 0.0], [10.0, 10.0]],
            "orientations": [0.0, 180.0],
            "speeds": [1.0, 1.5],
            "max_speeds": [2.0, 3.0],
            "num_agents": 2
        }
        
        config = MultiAgentConfig(**hydra_multi_config)
        assert len(config.positions) == 2
        assert config.orientations == [0.0, 180.0]
        assert config.num_agents == 2

    def test_config_validation_with_environment_overrides(self):
        """Test configuration validation with environment variable patterns."""
        # Simulate environment variable values that might come through Hydra
        env_config = {
            "video_path": "${oc.env:VIDEO_PATH,/default/path.mp4}",  # Hydra interpolation syntax
            "flip": "${oc.env:VIDEO_FLIP,false}",
            "kernel_size": 3,
            "kernel_sigma": 1.0
        }
        
        # In real scenarios, Hydra would resolve interpolations before Pydantic validation
        # For testing, we'll use resolved values
        resolved_config = {
            "video_path": "/resolved/path.mp4",
            "flip": False,
            "kernel_size": 3,
            "kernel_sigma": 1.0
        }
        
        config = VideoPlumeConfig(**resolved_config)
        assert config.video_path == "/resolved/path.mp4"
        assert config.flip is False

    def test_schema_error_reporting_for_hydra_integration(self):
        """Test comprehensive error reporting for Hydra configuration issues."""
        # Test detailed error messages that would help debug Hydra config issues
        try:
            NavigatorConfig(
                speed=5.0,
                max_speed=2.0,  # speed > max_speed
                positions=[[0.0, 0.0]],  # conflicting with single-agent params
                position=(1.0, 1.0)
            )
        except ValidationError as e:
            error_dict = e.errors()
            # Should contain multiple validation errors
            assert len(error_dict) >= 1
            
            # Check for specific error types that would help with Hydra debugging
            error_messages = [error["msg"] for error in error_dict]
            assert any("Cannot specify both single-agent and multi-agent" in msg for msg in error_messages)

    def test_config_serialization_for_hydra_output(self):
        """Test configuration serialization compatible with Hydra output formats."""
        config = NavigatorConfig(
            position=(5.0, 10.0),
            orientation=30.0,
            speed=1.0,
            max_speed=2.0
        )
        
        # Test model dump (equivalent to dict conversion)
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["position"] == (5.0, 10.0)
        assert config_dict["orientation"] == 30.0
        
        # Test JSON serialization
        config_json = config.model_dump_json()
        assert isinstance(config_json, str)
        assert "5.0" in config_json
        assert "30.0" in config_json