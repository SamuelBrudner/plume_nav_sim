"""
Integration tests for ComponentBasedEnvironment.

Tests the new component-based environment with actual component implementations
to ensure proper integration and contract compliance.

Contract: environment_state_machine.md
"""

import numpy as np
import pytest

from plume_nav_sim.actions import DiscreteGridActions
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.envs import ComponentBasedEnvironment, EnvironmentState
from plume_nav_sim.observations import ConcentrationSensor
from plume_nav_sim.plume.concentration_field import ConcentrationField
from plume_nav_sim.rewards import SparseGoalReward


@pytest.fixture
def grid_size():
    """Standard grid size for tests."""
    return GridSize(width=64, height=64)


@pytest.fixture
def goal_location():
    """Goal location away from starting position."""
    return Coordinates(50, 50)


@pytest.fixture
def concentration_field(grid_size, goal_location):
    """Create a simple concentration field centered at goal."""
    import numpy as np

    # Create a mock field with numpy array directly
    field = ConcentrationField(grid_size=grid_size, enable_caching=False)

    # Manually create a simple Gaussian field
    x = np.arange(grid_size.width)
    y = np.arange(grid_size.height)
    xx, yy = np.meshgrid(x, y)

    # Gaussian centered at goal
    sigma = 10.0
    dx = xx - goal_location.x
    dy = yy - goal_location.y
    field_array = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    # Manually set the field array
    field.field_array = field_array.astype(np.float32)
    field.is_generated = True

    return field


@pytest.fixture
def action_processor():
    """Create action processor."""
    return DiscreteGridActions(step_size=1)


@pytest.fixture
def observation_model():
    """Create observation model."""
    return ConcentrationSensor()


@pytest.fixture
def reward_function(goal_location):
    """Create reward function."""
    return SparseGoalReward(goal_position=goal_location, goal_radius=5.0)


@pytest.fixture
def component_env(
    action_processor,
    observation_model,
    reward_function,
    concentration_field,
    grid_size,
    goal_location,
):
    """Create fully-configured component-based environment."""
    return ComponentBasedEnvironment(
        action_processor=action_processor,
        observation_model=observation_model,
        reward_function=reward_function,
        concentration_field=concentration_field,
        grid_size=grid_size,
        max_steps=100,
        goal_location=goal_location,
        goal_radius=5.0,
        start_location=Coordinates(0, 0),
    )


class TestComponentBasedEnvironment:
    """Integration tests for ComponentBasedEnvironment."""

    def test_initialization(self, component_env):
        """Test: Environment initializes in CREATED state."""
        assert component_env._state == EnvironmentState.CREATED
        assert component_env._step_count == 0
        assert component_env._agent_state is None

    def test_spaces_from_components(
        self, component_env, action_processor, observation_model
    ):
        """Test: action_space and observation_space from components."""
        assert component_env.action_space is action_processor.action_space
        assert component_env.observation_space is observation_model.observation_space

    def test_reset_transitions_to_ready(self, component_env):
        """Test: reset() transitions from CREATED to READY."""
        assert component_env._state == EnvironmentState.CREATED

        obs, info = component_env.reset()

        assert component_env._state == EnvironmentState.READY
        assert component_env._step_count == 0
        assert component_env._agent_state is not None

    def test_reset_returns_observation_and_info(self, component_env):
        """Test: reset() returns observation and info dict."""
        obs, info = component_env.reset()

        # Observation should be from observation_model (numpy array for ConcentrationSensor)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (1,)  # ConcentrationSensor returns shape (1,)

        # Info should have required keys
        assert "agent_position" in info

    def test_step_before_reset_raises_error(self, component_env):
        """Test: step() before reset() raises StateError."""
        from plume_nav_sim.utils.exceptions import StateError

        with pytest.raises(StateError, match="Must call reset"):
            component_env.step(0)

    def test_step_after_reset(self, component_env):
        """Test: step() works after reset()."""
        component_env.reset()

        # Sample valid action
        action = component_env.action_space.sample()
        obs, reward, terminated, truncated, info = component_env.step(action)

        # May be READY or TERMINATED depending on random action
        assert component_env._state in [
            EnvironmentState.READY,
            EnvironmentState.TERMINATED,
            EnvironmentState.TRUNCATED,
        ]
        assert component_env._step_count == 1
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_increments_count(self, component_env):
        """Test: step() increments step_count."""
        component_env.reset()

        initial_count = component_env._step_count
        component_env.step(0)  # UP

        assert component_env._step_count == initial_count + 1

    def test_episode_truncation(self, component_env):
        """Test: Episode truncates at max_steps."""
        component_env.reset()

        # Run to max_steps
        truncated = False
        for _ in range(component_env.max_steps):
            _, _, _, truncated, _ = component_env.step(0)
            if truncated:
                break

        assert truncated
        assert component_env._state == EnvironmentState.TRUNCATED

    def test_goal_termination(self, component_env, goal_location):
        """Test: Episode terminates when goal reached."""
        component_env.reset()

        # Manually position agent near goal (hack for testing)
        component_env._agent_state.position = Coordinates(
            goal_location.x + 2, goal_location.y
        )

        # Move towards goal
        terminated = False
        for _ in range(10):
            _, _, terminated, _, _ = component_env.step(3)  # LEFT
            if terminated:
                break

        assert terminated
        assert component_env._state == EnvironmentState.TERMINATED
        assert component_env._agent_state.goal_reached

    def test_reset_after_termination(self, component_env, goal_location):
        """Test: Can reset() after termination."""
        component_env.reset()

        # Force termination
        component_env._agent_state.position = goal_location
        component_env.step(0)

        assert component_env._state == EnvironmentState.TERMINATED

        # Reset again
        obs, info = component_env.reset()

        assert component_env._state == EnvironmentState.READY
        assert component_env._step_count == 0

    def test_close_transitions_to_closed(self, component_env):
        """Test: close() transitions to CLOSED."""
        component_env.reset()

        component_env.close()

        assert component_env._state == EnvironmentState.CLOSED

    def test_cannot_step_after_close(self, component_env):
        """Test: step() after close() raises StateError."""
        from plume_nav_sim.utils.exceptions import StateError

        component_env.reset()
        component_env.close()

        with pytest.raises(StateError, match="Cannot step closed environment"):
            component_env.step(0)

    def test_cannot_reset_after_close(self, component_env):
        """Test: reset() after close() raises StateError."""
        from plume_nav_sim.utils.exceptions import StateError

        component_env.reset()
        component_env.close()

        with pytest.raises(StateError, match="Cannot reset closed environment"):
            component_env.reset()

    def test_close_is_idempotent(self, component_env):
        """Test: close() can be called multiple times safely."""
        component_env.reset()

        component_env.close()
        component_env.close()  # Should not raise
        component_env.close()  # Should not raise

        assert component_env._state == EnvironmentState.CLOSED

    def test_deterministic_reset(self, component_env):
        """Test: reset(seed=X) produces deterministic results."""
        obs1, _ = component_env.reset(seed=42)
        component_env.close()

        # Create new environment
        from plume_nav_sim.actions import DiscreteGridActions
        from plume_nav_sim.observations import ConcentrationSensor
        from plume_nav_sim.rewards import SparseGoalReward

        env2 = ComponentBasedEnvironment(
            action_processor=DiscreteGridActions(step_size=1),
            observation_model=ConcentrationSensor(),
            reward_function=SparseGoalReward(
                goal_position=component_env.goal_location, goal_radius=5.0
            ),
            concentration_field=component_env._concentration_field,
            grid_size=component_env.grid_size,
            max_steps=100,
            goal_location=component_env.goal_location,
            goal_radius=5.0,
            start_location=Coordinates(0, 0),
        )

        obs2, _ = env2.reset(seed=42)

        # Observations should match (both are numpy arrays)
        np.testing.assert_array_equal(obs1, obs2)

    def test_component_delegation(self, component_env, action_processor):
        """Test: Environment delegates to components correctly."""
        component_env.reset(seed=42)  # Seed for deterministic agent placement

        initial_pos = component_env._agent_state.position
        
        # Choose action based on position to avoid boundary
        # If at right edge, use different action
        if initial_pos.x >= 62:
            action = 3  # LEFT instead of RIGHT
        else:
            action = 1  # RIGHT

        obs, reward, terminated, truncated, info = component_env.step(action)

        # Position should have changed (delegated to ActionProcessor)
        assert component_env._agent_state.position != initial_pos

        # Observation should come from ObservationModel (numpy array)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (1,)

        # Reward should come from RewardFunction
        assert isinstance(reward, (int, float))

    def test_invalid_action_raises_error(self, component_env):
        """Test: Invalid action raises ValidationError."""
        from plume_nav_sim.utils.exceptions import ValidationError

        component_env.reset()

        with pytest.raises(ValidationError, match="Invalid action"):
            component_env.step(999)  # Invalid action

    def test_gymnasium_api_compliance(self, component_env):
        """Test: Environment follows Gymnasium API."""
        # Should have required attributes
        assert hasattr(component_env, "action_space")
        assert hasattr(component_env, "observation_space")
        assert hasattr(component_env, "reset")
        assert hasattr(component_env, "step")
        assert hasattr(component_env, "close")
        assert hasattr(component_env, "render")
        assert hasattr(component_env, "metadata")

        # reset() should return (obs, info)
        result = component_env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

        # step() should return (obs, reward, terminated, truncated, info)
        result = component_env.step(0)
        assert isinstance(result, tuple)
        assert len(result) == 5
