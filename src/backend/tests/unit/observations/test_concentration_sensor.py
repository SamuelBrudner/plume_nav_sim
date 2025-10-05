"""
Concrete tests for ConcentrationSensor implementation.

This test suite inherits all universal tests from TestObservationModelInterface
and adds ConcentrationSensor-specific tests.

Contract: src/backend/contracts/observation_model_interface.md
"""

import numpy as np
import pytest

import gymnasium as gym
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.observations import ConcentrationSensor
from tests.contracts.test_observation_model_interface import (
    TestObservationModelInterface,
    create_simple_env_state,
)


class TestConcentrationSensor(TestObservationModelInterface):
    """Concrete tests for ConcentrationSensor.

    Inherits all 13 universal tests from TestObservationModelInterface.
    Adds implementation-specific tests for single-sensor behavior.
    """

    # ==============================================================================
    # Fixture Override (provides ConcentrationSensor for universal tests)
    # ==============================================================================

    @pytest.fixture
    def observation_model(self):
        """Provide ConcentrationSensor for testing.

        Returns:
            ConcentrationSensor instance
        """
        return ConcentrationSensor()

    # ==============================================================================
    # Implementation-Specific Tests
    # ==============================================================================

    def test_observation_space_is_box(self, observation_model):
        """ConcentrationSensor uses Box space with shape (1,)."""
        space = observation_model.observation_space

        assert isinstance(space, gym.spaces.Box), "Must use Box space"
        assert space.shape == (1,), f"Shape must be (1,), got {space.shape}"
        assert space.dtype == np.float32, f"Dtype must be float32, got {space.dtype}"
        assert np.all(space.low == 0.0), "Low bound must be 0.0"
        assert np.all(space.high == 1.0), "High bound must be 1.0"

    def test_samples_concentration_at_agent_position(self):
        """ConcentrationSensor samples plume at agent's exact position."""
        sensor = ConcentrationSensor()

        # Create plume field with known values
        grid = GridSize(width=10, height=10)
        plume_field = np.zeros((grid.height, grid.width), dtype=np.float32)
        plume_field[5, 7] = 0.75  # Known concentration at (7, 5)

        # Agent at position (7, 5)
        agent_state = AgentState(position=Coordinates(7, 5))
        env_state = {
            "agent_state": agent_state,
            "plume_field": plume_field,
            "grid_size": grid,
        }

        observation = sensor.get_observation(env_state)

        # Should return concentration at agent position
        assert observation.shape == (1,)
        assert np.isclose(
            observation[0], 0.75
        ), f"Expected 0.75 at position (7,5), got {observation[0]}"

    def test_returns_zero_for_empty_plume(self):
        """ConcentrationSensor returns 0.0 for zero concentration."""
        sensor = ConcentrationSensor()

        # Create empty plume field
        grid = GridSize(width=10, height=10)
        plume_field = np.zeros((grid.height, grid.width), dtype=np.float32)

        agent_state = AgentState(position=Coordinates(5, 5))
        env_state = {
            "agent_state": agent_state,
            "plume_field": plume_field,
            "grid_size": grid,
        }

        observation = sensor.get_observation(env_state)

        assert observation.shape == (1,)
        assert observation[0] == 0.0

    def test_clamps_to_valid_range(self):
        """ConcentrationSensor clamps values to [0, 1]."""
        sensor = ConcentrationSensor()

        # Create plume field with out-of-range value
        grid = GridSize(width=10, height=10)
        plume_field = np.zeros((grid.height, grid.width), dtype=np.float32)
        plume_field[3, 2] = 1.5  # Over 1.0

        agent_state = AgentState(position=Coordinates(2, 3))
        env_state = {
            "agent_state": agent_state,
            "plume_field": plume_field,
            "grid_size": grid,
        }

        observation = sensor.get_observation(env_state)

        # Should be clamped to 1.0
        assert observation[0] <= 1.0, "Concentration should be clamped to 1.0"

    def test_different_positions_return_different_concentrations(self):
        """ConcentrationSensor returns different values at different positions."""
        sensor = ConcentrationSensor()

        # Create plume with gradient
        grid = GridSize(width=10, height=10)
        plume_field = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)

        # Sample at two different positions
        pos1 = Coordinates(0, 0)
        state1 = AgentState(position=pos1)
        env_state1 = {
            "agent_state": state1,
            "plume_field": plume_field,
            "grid_size": grid,
        }
        obs1 = sensor.get_observation(env_state1)

        pos2 = Coordinates(9, 9)
        state2 = AgentState(position=pos2)
        env_state2 = {
            "agent_state": state2,
            "plume_field": plume_field,
            "grid_size": grid,
        }
        obs2 = sensor.get_observation(env_state2)

        # Should be different
        assert not np.isclose(obs1[0], obs2[0]), (
            f"Different positions should yield different concentrations: "
            f"pos1={obs1[0]}, pos2={obs2[0]}"
        )

    def test_metadata_structure(self, observation_model):
        """ConcentrationSensor metadata has expected structure."""
        metadata = observation_model.get_metadata()

        assert metadata["type"] == "concentration_sensor"
        assert "modality" in metadata
        assert metadata["modality"] == "olfactory"
        assert "parameters" in metadata
        assert "required_state_keys" in metadata
        assert "plume_field" in metadata["required_state_keys"]
        assert "agent_state" in metadata["required_state_keys"]

    def test_observation_dtype_is_float32(self, observation_model):
        """ConcentrationSensor returns float32 observations."""
        env_state = create_simple_env_state()
        observation = observation_model.get_observation(env_state)

        assert (
            observation.dtype == np.float32
        ), f"Expected float32, got {observation.dtype}"
