"""
Universal test suite for ObservationModel protocol implementations.

Contract: src/backend/contracts/observation_model_interface.md

All observation model implementations MUST pass these tests.
Concrete test classes should inherit from TestObservationModelInterface
and provide an observation_model fixture.

Usage:
    class TestConcentrationSensor(TestObservationModelInterface):
        @pytest.fixture
        def observation_model(self):
            return ConcentrationSensor()
"""

import copy

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import gymnasium as gym
from plume_nav_sim.core.geometry import GridSize
from plume_nav_sim.interfaces import ObservationModel
from tests.strategies import (
    agent_state_strategy,
    env_state_strategy,
    grid_size_strategy,
)


class TestObservationModelInterface:
    """Universal test suite for ObservationModel implementations.

    Contract: observation_model_interface.md

    All implementations must pass these tests to be considered valid.
    Concrete test classes should inherit this and provide observation_model fixture.
    """

    # ==============================================================================
    # Fixtures (Override in concrete test classes)
    # ==============================================================================

    @pytest.fixture
    def observation_model(self) -> ObservationModel:
        """Override this fixture to provide the observation model to test.

        Returns:
            ObservationModel implementation to test

        Raises:
            NotImplementedError: If not overridden in subclass
        """
        raise NotImplementedError(
            "Concrete test classes must override observation_model fixture"
        )

    @pytest.fixture
    def grid_size(self) -> GridSize:
        """Default grid size for tests.

        Returns:
            GridSize(128, 128)
        """
        return GridSize(width=128, height=128)

    # ==============================================================================
    # Property 1: Space Containment (UNIVERSAL)
    # ==============================================================================

    @given(env_state=env_state_strategy(include_plume_field=True))
    @settings(
        deadline=None,
        max_examples=50,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.differing_executors,
        ],
    )
    def test_space_containment(self, observation_model, env_state):
        """Property: Observation always in observation_space.

        Contract: observation_model_interface.md - Property 1: Space Containment

        ∀ env_state: observation_space.contains(get_observation(env_state))
        """
        observation = observation_model.get_observation(env_state)

        assert observation_model.observation_space.contains(observation), (
            f"Observation not in observation_space.\n"
            f"Observation: {observation}\n"
            f"Space: {observation_model.observation_space}"
        )

    # ==============================================================================
    # Property 2: Determinism (UNIVERSAL)
    # ==============================================================================

    @given(env_state=env_state_strategy(include_plume_field=True))
    @settings(
        deadline=None,
        max_examples=50,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.differing_executors,
        ],
    )
    def test_determinism(self, observation_model, env_state):
        """Property: Same env_state produces same observation.

        Contract: observation_model_interface.md - Property 2: Determinism

        ∀ env_state: get_observation(env_state) = get_observation(env_state)
        """
        # Get observation twice with same env_state
        obs1 = observation_model.get_observation(env_state)
        obs2 = observation_model.get_observation(env_state)

        # Must be equal
        if isinstance(obs1, np.ndarray):
            assert np.array_equal(obs1, obs2), "Observation is not deterministic"
        elif isinstance(obs1, dict):
            for key in obs1.keys():
                if isinstance(obs1[key], np.ndarray):
                    assert np.array_equal(
                        obs1[key], obs2[key]
                    ), f"Observation['{key}'] is not deterministic"
                else:
                    assert (
                        obs1[key] == obs2[key]
                    ), f"Observation['{key}'] is not deterministic"
        elif isinstance(obs1, tuple):
            for i, (val1, val2) in enumerate(zip(obs1, obs2)):
                if isinstance(val1, np.ndarray):
                    assert np.array_equal(
                        val1, val2
                    ), f"Observation[{i}] is not deterministic"
                else:
                    assert val1 == val2, f"Observation[{i}] is not deterministic"
        else:
            assert obs1 == obs2, "Observation is not deterministic"

    # ==============================================================================
    # Property 3: Purity (UNIVERSAL)
    # ==============================================================================

    def test_purity_no_state_mutation(self, observation_model, grid_size):
        """Property: get_observation does not mutate env_state.

        Contract: observation_model_interface.md - Property 3: Purity

        No modification of env_state dictionary or its contents.
        """
        from plume_nav_sim.core.geometry import Coordinates
        from plume_nav_sim.core.state import AgentState

        # Create env_state
        agent_state = AgentState(position=Coordinates(10, 10), orientation=45.0)
        plume_field = np.random.rand(32, 32).astype(np.float32)

        env_state = {
            "agent_state": agent_state,
            "plume_field": plume_field,
            "grid_size": grid_size,
            "time_step": 0,
        }

        # Deep copy for comparison
        agent_state_copy = copy.deepcopy(agent_state)
        plume_field_copy = plume_field.copy()

        # Get observation
        _ = observation_model.get_observation(env_state)

        # Verify no mutations
        assert agent_state.position == agent_state_copy.position
        assert agent_state.orientation == agent_state_copy.orientation
        assert agent_state.step_count == agent_state_copy.step_count

        assert np.array_equal(plume_field, plume_field_copy), "plume_field was mutated"

        # env_state dict itself should not be modified
        assert "agent_state" in env_state
        assert "plume_field" in env_state

    # ==============================================================================
    # Property 4: Shape Consistency (UNIVERSAL)
    # ==============================================================================

    def test_shape_consistency(self, observation_model, grid_size):
        """Property: Observation shape matches observation_space.

        Contract: observation_model_interface.md - Property 4: Shape Consistency

        For Box spaces: observation.shape matches space.shape
        For Dict spaces: all keys present, each value matches subspace
        """
        from plume_nav_sim.core.geometry import Coordinates
        from plume_nav_sim.core.state import AgentState

        # Create simple env_state
        agent_state = AgentState(position=Coordinates(10, 10))
        env_state = {
            "agent_state": agent_state,
            "plume_field": np.zeros((32, 32), dtype=np.float32),
            "grid_size": grid_size,
        }

        observation = observation_model.get_observation(env_state)
        space = observation_model.observation_space

        # Check shape based on space type
        if isinstance(space, gym.spaces.Box):
            assert hasattr(observation, "shape"), "Box observation must have shape"
            assert (
                observation.shape == space.shape
            ), f"Shape mismatch: observation {observation.shape} vs space {space.shape}"
        elif isinstance(space, gym.spaces.Dict):
            assert isinstance(observation, dict), "Dict space requires dict observation"
            # Check all keys present
            for key in space.spaces.keys():
                assert key in observation, f"Missing key '{key}' in observation"
        elif isinstance(space, gym.spaces.Tuple):
            assert isinstance(
                observation, tuple
            ), "Tuple space requires tuple observation"
            assert len(observation) == len(
                space.spaces
            ), f"Tuple length mismatch: {len(observation)} vs {len(space.spaces)}"

    # ==============================================================================
    # Observation Space Tests
    # ==============================================================================

    def test_has_observation_space_property(self, observation_model):
        """Test: observation_model has observation_space property.

        Contract: observation_model_interface.md - ObservationModel protocol
        """
        assert hasattr(observation_model, "observation_space")
        # Should be a property, not a method
        assert isinstance(observation_model.observation_space, gym.Space)

    def test_observation_space_is_immutable(self, observation_model):
        """Test: observation_space returns same instance.

        Contract: observation_model_interface.md - Postcondition C2

        observation_space should be computed once and cached.
        """
        space1 = observation_model.observation_space
        space2 = observation_model.observation_space

        # Should be same instance
        assert space1 is space2, "observation_space should return same instance"

    def test_observation_space_is_valid_gym_space(self, observation_model):
        """Test: observation_space is a valid Gymnasium Space."""
        space = observation_model.observation_space

        assert isinstance(
            space, gym.Space
        ), f"observation_space must be gym.Space, got {type(space)}"

        # Should be able to sample from it
        try:
            sample = space.sample()
            assert space.contains(sample), "Space sample not contained in space"
        except Exception as e:
            pytest.fail(f"observation_space.sample() failed: {e}")

    # ==============================================================================
    # Get Observation Tests
    # ==============================================================================

    def test_has_get_observation_method(self, observation_model):
        """Test: observation_model has get_observation() method.

        Contract: observation_model_interface.md - ObservationModel protocol
        """
        assert hasattr(observation_model, "get_observation")
        assert callable(observation_model.get_observation)

    def test_get_observation_accepts_env_state_dict(self, observation_model, grid_size):
        """Test: get_observation accepts env_state dictionary."""
        from plume_nav_sim.core.geometry import Coordinates
        from plume_nav_sim.core.state import AgentState

        agent_state = AgentState(position=Coordinates(10, 10))
        env_state = {
            "agent_state": agent_state,
            "plume_field": np.zeros((32, 32), dtype=np.float32),
            "grid_size": grid_size,
        }

        # Should not raise
        try:
            observation = observation_model.get_observation(env_state)
            assert observation is not None
        except Exception as e:
            pytest.fail(f"get_observation raised unexpected error: {e}")

    # ==============================================================================
    # Metadata Tests
    # ==============================================================================

    def test_has_get_metadata_method(self, observation_model):
        """Test: observation_model has get_metadata() method.

        Contract: observation_model_interface.md - ObservationModel protocol
        """
        assert hasattr(observation_model, "get_metadata")
        assert callable(observation_model.get_metadata)

    def test_metadata_has_required_keys(self, observation_model):
        """Test: Metadata contains required keys.

        Contract: observation_model_interface.md - get_metadata() return value
        """
        metadata = observation_model.get_metadata()

        assert isinstance(metadata, dict), "Metadata must be a dictionary"
        assert "type" in metadata, "Metadata must contain 'type' key"
        assert isinstance(metadata["type"], str), "'type' must be a string"

    # ==============================================================================
    # Protocol Conformance
    # ==============================================================================

    def test_conforms_to_observation_model_protocol(self, observation_model):
        """Test: Implementation satisfies ObservationModel protocol.

        Contract: observation_model_interface.md - ObservationModel protocol
        """
        assert isinstance(
            observation_model, ObservationModel
        ), f"{type(observation_model).__name__} does not satisfy ObservationModel protocol"


# ==============================================================================
# Helper Functions for Concrete Tests
# ==============================================================================


def create_simple_env_state():
    """Helper to create a simple env_state for manual tests.

    Returns:
        Dictionary with basic env_state structure
    """
    from plume_nav_sim.core.geometry import Coordinates
    from plume_nav_sim.core.state import AgentState

    agent_state = AgentState(position=Coordinates(16, 16), orientation=0.0)
    plume_field = np.zeros((32, 32), dtype=np.float32)
    grid_size = GridSize(width=32, height=32)

    return {
        "agent_state": agent_state,
        "plume_field": plume_field,
        "grid_size": grid_size,
        "time_step": 0,
    }
