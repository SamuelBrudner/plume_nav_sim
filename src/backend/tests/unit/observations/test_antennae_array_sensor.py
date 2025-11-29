"""
Concrete tests for AntennaeArraySensor implementation.

This test suite inherits all universal tests from TestObservationModelInterface
and adds AntennaeArraySensor-specific tests for multi-sensor behavior.

Contract: src/backend/contracts/observation_model_interface.md
"""

import gymnasium as gym
import numpy as np
import pytest

from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.observations import AntennaeArraySensor
from tests.contracts.test_observation_model_interface import (
    TestObservationModelInterface,
    create_simple_env_state,
)


class TestAntennaeArraySensor(TestObservationModelInterface):
    __test__ = True
    """Concrete tests for AntennaeArraySensor.

    Inherits all 13 universal tests from TestObservationModelInterface.
    Adds implementation-specific tests for multi-sensor, orientation-aware behavior.
    """

    # ==============================================================================
    # Fixture Override
    # ==============================================================================

    @pytest.fixture
    def observation_model(self):
        """Provide AntennaeArraySensor for testing.

        Uses default configuration: 2 sensors (left/right antennae) at ±45° angles.

        Returns:
            AntennaeArraySensor instance with default config
        """
        return AntennaeArraySensor(
            n_sensors=2,
            sensor_angles=[45.0, -45.0],  # Left and right relative to heading
            sensor_distance=1.0,
        )

    # ==============================================================================
    # Implementation-Specific Tests
    # ==============================================================================

    def test_observation_space_is_box_with_correct_shape(self, observation_model):
        """AntennaeArraySensor uses Box space with shape (n_sensors,)."""
        space = observation_model.observation_space

        assert isinstance(space, gym.spaces.Box), "Must use Box space"
        assert space.shape == (
            2,
        ), f"Shape must be (2,) for 2 sensors, got {space.shape}"
        assert space.dtype == np.float32, f"Dtype must be float32, got {space.dtype}"
        assert np.all(space.low == 0.0), "Low bound must be 0.0"
        assert np.all(space.high == 1.0), "High bound must be 1.0"

    def test_custom_sensor_count(self):
        """AntennaeArraySensor accepts custom sensor count."""
        # 4 sensors at cardinal directions relative to heading
        sensor = AntennaeArraySensor(
            n_sensors=4,
            sensor_angles=[0.0, 90.0, 180.0, 270.0],
            sensor_distance=1.0,
        )

        assert sensor.observation_space.shape == (4,)

        grid = GridSize(width=20, height=20)
        agent_state = AgentState(position=Coordinates(10, 10), orientation=0.0)
        plume_field = np.random.rand(grid.height, grid.width).astype(np.float32)
        env_state = {
            "agent_state": agent_state,
            "plume_field": plume_field,
            "grid_size": grid,
        }

        observation = sensor.get_observation(env_state)
        assert observation.shape == (4,), "Should return 4 sensor readings"

    def test_sensors_sample_at_offset_positions(self):
        """AntennaeArraySensor samples at orientation-relative positions."""
        sensor = AntennaeArraySensor(
            n_sensors=2,
            sensor_angles=[0.0, 180.0],  # Front and back
            sensor_distance=2.0,
        )

        # Create plume with known gradient
        grid = GridSize(width=20, height=20)
        plume_field = np.zeros((grid.height, grid.width), dtype=np.float32)

        # Agent at (10, 10) facing East (0°)
        agent_pos = Coordinates(10, 10)
        plume_field[agent_pos.y, agent_pos.x + 2] = 0.8  # Front sensor position
        plume_field[agent_pos.y, agent_pos.x - 2] = 0.2  # Back sensor position

        agent_state = AgentState(position=agent_pos, orientation=0.0)
        env_state = {
            "agent_state": agent_state,
            "plume_field": plume_field,
            "grid_size": grid,
        }

        observation = sensor.get_observation(env_state)

        # Front sensor (0°) should read ~0.8, back sensor (180°) should read ~0.2
        assert np.isclose(
            observation[0], 0.8, atol=0.1
        ), f"Front sensor should read ~0.8, got {observation[0]}"
        assert np.isclose(
            observation[1], 0.2, atol=0.1
        ), f"Back sensor should read ~0.2, got {observation[1]}"

    def test_orientation_affects_sensor_positions(self):
        """AntennaeArraySensor rotates sensor positions with agent orientation."""
        sensor = AntennaeArraySensor(
            n_sensors=1,
            sensor_angles=[0.0],  # Directly ahead
            sensor_distance=1.0,
        )

        grid = GridSize(width=20, height=20)
        plume_field = np.zeros((grid.height, grid.width), dtype=np.float32)

        # Place markers at different positions
        agent_pos = Coordinates(10, 10)
        plume_field[agent_pos.y, agent_pos.x + 1] = 0.9  # East
        plume_field[agent_pos.y - 1, agent_pos.x] = 0.7  # North

        # Agent facing East (0°) - sensor should read 0.9
        state_east = AgentState(position=agent_pos, orientation=0.0)
        env_state_east = {
            "agent_state": state_east,
            "plume_field": plume_field,
            "grid_size": grid,
        }
        obs_east = sensor.get_observation(env_state_east)

        # Agent facing North (90°) - sensor should read 0.7
        state_north = AgentState(position=agent_pos, orientation=90.0)
        env_state_north = {
            "agent_state": state_north,
            "plume_field": plume_field,
            "grid_size": grid,
        }
        obs_north = sensor.get_observation(env_state_north)

        assert np.isclose(
            obs_east[0], 0.9, atol=0.1
        ), f"East-facing should read 0.9, got {obs_east[0]}"
        assert np.isclose(
            obs_north[0], 0.7, atol=0.1
        ), f"North-facing should read 0.7, got {obs_north[0]}"

    def test_handles_out_of_bounds_sensors_gracefully(self):
        """AntennaeArraySensor returns 0.0 for sensors outside grid."""
        sensor = AntennaeArraySensor(
            n_sensors=1,
            sensor_angles=[0.0],
            sensor_distance=50.0,  # Very far away
        )

        # Small grid
        grid = GridSize(width=10, height=10)
        plume_field = np.ones((grid.height, grid.width), dtype=np.float32)

        # Agent near edge
        agent_state = AgentState(position=Coordinates(1, 1), orientation=0.0)
        env_state = {
            "agent_state": agent_state,
            "plume_field": plume_field,
            "grid_size": grid,
        }

        observation = sensor.get_observation(env_state)

        # Should return 0.0 for out-of-bounds sensor
        assert (
            observation[0] == 0.0
        ), f"Out-of-bounds sensor should return 0.0, got {observation[0]}"

    def test_multiple_sensors_return_different_values(self):
        """AntennaeArraySensor returns independent readings for each sensor."""
        sensor = AntennaeArraySensor(
            n_sensors=3,
            sensor_angles=[-45.0, 0.0, 45.0],  # Left, center, right
            sensor_distance=1.0,
        )

        # Create plume with spatial variation
        grid = GridSize(width=20, height=20)
        x, y = np.meshgrid(np.arange(20), np.arange(20))
        plume_field = ((x + y) / 40.0).astype(np.float32)  # Gradient

        agent_state = AgentState(position=Coordinates(10, 10), orientation=0.0)
        env_state = {
            "agent_state": agent_state,
            "plume_field": plume_field,
            "grid_size": grid,
        }

        observation = sensor.get_observation(env_state)

        # All three sensors should read different values (gradient field)
        assert (
            len(np.unique(observation)) >= 2
        ), "Multiple sensors should return different values in gradient field"

    def test_metadata_structure(self, observation_model):
        """AntennaeArraySensor metadata has expected structure."""
        metadata = observation_model.get_metadata()

        assert metadata["type"] == "antennae_array_sensor"
        assert "modality" in metadata
        assert metadata["modality"] == "olfactory"
        assert "parameters" in metadata
        assert metadata["parameters"]["n_sensors"] == 2
        assert "sensor_angles" in metadata["parameters"]
        assert "sensor_distance" in metadata["parameters"]
        assert "required_state_keys" in metadata
        assert "agent_state" in metadata["required_state_keys"]
        assert "plume_field" in metadata["required_state_keys"]

    def test_observation_dtype_is_float32(self, observation_model):
        """AntennaeArraySensor returns float32 observations."""
        env_state = create_simple_env_state()
        observation = observation_model.get_observation(env_state)

        assert (
            observation.dtype == np.float32
        ), f"Expected float32, got {observation.dtype}"

    def test_clamps_concentrations_to_valid_range(self, observation_model):
        """AntennaeArraySensor clamps all sensor readings to [0, 1]."""
        # Create plume with out-of-range values
        grid = GridSize(width=20, height=20)
        plume_field = np.full((grid.height, grid.width), 1.5, dtype=np.float32)

        agent_state = AgentState(position=Coordinates(10, 10), orientation=0.0)
        env_state = {
            "agent_state": agent_state,
            "plume_field": plume_field,
            "grid_size": grid,
        }

        observation = observation_model.get_observation(env_state)

        # All values should be clamped to 1.0
        assert np.all(observation <= 1.0), "All readings should be <= 1.0"
        assert np.all(observation >= 0.0), "All readings should be >= 0.0"
