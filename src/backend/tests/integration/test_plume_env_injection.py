"""Integration tests for direct component injection into PlumeEnv."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from plume_nav_sim._compat import StateError, ValidationError
from plume_nav_sim.actions import DiscreteGridActions
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.envs import PlumeEnv
from plume_nav_sim.envs.state import EnvironmentState
from plume_nav_sim.interfaces.action import ActionType
from plume_nav_sim.observations import ConcentrationSensor
from plume_nav_sim.plume.gaussian import GaussianPlume
from plume_nav_sim.rewards import SparseGoalReward


@pytest.fixture
def grid_size() -> GridSize:
    return GridSize(width=64, height=64)


@pytest.fixture
def goal_location() -> Coordinates:
    return Coordinates(50, 50)


@pytest.fixture
def concentration_field(
    grid_size: GridSize, goal_location: Coordinates
) -> GaussianPlume:
    return GaussianPlume(
        grid_size=grid_size,
        source_location=goal_location,
        sigma=10.0,
    )


@pytest.fixture
def action_model() -> DiscreteGridActions:
    return DiscreteGridActions(step_size=1)


@pytest.fixture
def sensor_model() -> ConcentrationSensor:
    return ConcentrationSensor()


@pytest.fixture
def reward_fn(goal_location: Coordinates) -> SparseGoalReward:
    return SparseGoalReward(goal_position=goal_location, goal_radius=5.0)


@pytest.fixture
def vector_action_model():
    class VectorBoxActions:
        def __init__(self, step_size: float = 1.0) -> None:
            self.step_size = step_size
            self._action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            )

        @property
        def action_space(self) -> gym.Space:
            return self._action_space

        def process_action(
            self, action: ActionType, current_state: AgentState, grid_size: GridSize
        ) -> AgentState:
            if not self.validate_action(action):
                raise ValueError(f"Invalid action for Box space: {action}")

            dx = int(np.round(float(action[0]) * self.step_size))
            dy = int(np.round(float(action[1]) * self.step_size))

            new_x = int(np.clip(current_state.position.x + dx, 0, grid_size.width - 1))
            new_y = int(np.clip(current_state.position.y + dy, 0, grid_size.height - 1))
            return AgentState(
                position=Coordinates(new_x, new_y),
                orientation=current_state.orientation,
                step_count=current_state.step_count,
                total_reward=current_state.total_reward,
                goal_reached=current_state.goal_reached,
            )

        def validate_action(self, action: ActionType) -> bool:
            return self._action_space.contains(action)

        def get_metadata(self) -> dict[str, object]:
            return {
                "type": "vector_box_actions",
                "parameters": {"step_size": self.step_size, "shape": (2,)},
            }

    return VectorBoxActions(step_size=1.0)


@pytest.fixture
def plume_env(
    action_model: DiscreteGridActions,
    sensor_model: ConcentrationSensor,
    reward_fn: SparseGoalReward,
    concentration_field: GaussianPlume,
    grid_size: GridSize,
    goal_location: Coordinates,
) -> PlumeEnv:
    return PlumeEnv(
        grid_size=grid_size,
        source_location=goal_location,
        start_location=Coordinates(0, 0),
        max_steps=100,
        goal_radius=5.0,
        plume=concentration_field,
        sensor_model=sensor_model,
        action_model=action_model,
        reward_fn=reward_fn,
    )


@pytest.fixture
def vector_plume_env(
    vector_action_model,
    sensor_model: ConcentrationSensor,
    reward_fn: SparseGoalReward,
    concentration_field: GaussianPlume,
    grid_size: GridSize,
    goal_location: Coordinates,
) -> PlumeEnv:
    return PlumeEnv(
        grid_size=grid_size,
        source_location=goal_location,
        start_location=Coordinates(0, 0),
        max_steps=25,
        goal_radius=5.0,
        plume=concentration_field,
        sensor_model=sensor_model,
        action_model=vector_action_model,
        reward_fn=reward_fn,
    )


def test_initialization_uses_injected_components(
    plume_env: PlumeEnv,
    action_model: DiscreteGridActions,
    sensor_model: ConcentrationSensor,
    reward_fn: SparseGoalReward,
    concentration_field: GaussianPlume,
) -> None:
    assert plume_env._state == EnvironmentState.CREATED
    assert plume_env._step_count == 0
    assert plume_env._agent_state is None
    assert plume_env.action_space is action_model.action_space
    assert plume_env.observation_space is sensor_model.observation_space
    assert plume_env._action_processor is action_model
    assert plume_env._observation_model is sensor_model
    assert plume_env._reward_function is reward_fn
    assert plume_env._concentration_field is concentration_field


def test_reset_transitions_to_ready(plume_env: PlumeEnv) -> None:
    assert plume_env._state == EnvironmentState.CREATED

    obs, info = plume_env.reset(seed=7)

    assert plume_env._state == EnvironmentState.READY
    assert plume_env._step_count == 0
    assert plume_env._agent_state is not None
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (1,)
    assert info["agent_position"] == (0, 0)
    assert info["goal_location"] == plume_env.source_location


def test_step_before_reset_raises_error(plume_env: PlumeEnv) -> None:
    with pytest.raises(StateError, match="Must call reset"):
        plume_env.step(0)


def test_step_after_reset_updates_count(plume_env: PlumeEnv) -> None:
    plume_env.reset(seed=0)

    obs, reward, terminated, truncated, info = plume_env.step(0)

    assert plume_env._state in {
        EnvironmentState.READY,
        EnvironmentState.TERMINATED,
        EnvironmentState.TRUNCATED,
    }
    assert plume_env._step_count == 1
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["step_count"] == 1


def test_episode_truncates_at_max_steps(plume_env: PlumeEnv) -> None:
    plume_env.reset(seed=0)

    truncated = False
    for _ in range(plume_env.max_steps):
        _, _, _, truncated, _ = plume_env.step(0)
        if truncated:
            break

    assert truncated is True
    assert plume_env._state == EnvironmentState.TRUNCATED


def test_goal_termination_sets_goal_reached(
    plume_env: PlumeEnv, goal_location: Coordinates
) -> None:
    plume_env.reset(seed=0)
    assert plume_env._agent_state is not None
    plume_env._agent_state.position = Coordinates(goal_location.x + 2, goal_location.y)

    terminated = False
    for _ in range(10):
        _, _, terminated, _, _ = plume_env.step(3)
        if terminated:
            break

    assert terminated is True
    assert plume_env._state == EnvironmentState.TERMINATED
    assert plume_env._agent_state is not None
    assert plume_env._agent_state.goal_reached is True


def test_box_action_models_are_supported(
    vector_plume_env: PlumeEnv, grid_size: GridSize
) -> None:
    vector_plume_env.reset(seed=0)

    action = vector_plume_env.action_space.sample()
    obs, reward, terminated, truncated, _ = vector_plume_env.step(action)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert vector_plume_env._agent_state is not None
    assert 0 <= vector_plume_env._agent_state.position.x < grid_size.width
    assert 0 <= vector_plume_env._agent_state.position.y < grid_size.height


def test_reset_after_termination_restarts_episode(
    plume_env: PlumeEnv, goal_location: Coordinates
) -> None:
    plume_env.reset(seed=0)
    assert plume_env._agent_state is not None
    plume_env._agent_state.position = goal_location
    plume_env.step(0)

    assert plume_env._state == EnvironmentState.TERMINATED

    obs, info = plume_env.reset(seed=1)

    assert plume_env._state == EnvironmentState.READY
    assert plume_env._step_count == 0
    assert obs.shape == (1,)
    assert info["seed"] == 1


def test_close_is_idempotent_and_blocks_use(plume_env: PlumeEnv) -> None:
    plume_env.reset(seed=0)
    plume_env.close()
    plume_env.close()

    assert plume_env._state == EnvironmentState.CLOSED

    with pytest.raises(StateError, match="Cannot step closed environment"):
        plume_env.step(0)
    with pytest.raises(StateError, match="Cannot reset closed environment"):
        plume_env.reset()


def test_reset_with_same_seed_is_deterministic(
    grid_size: GridSize, goal_location: Coordinates
) -> None:
    env1 = PlumeEnv(
        grid_size=grid_size,
        source_location=goal_location,
        max_steps=25,
        plume=GaussianPlume(
            grid_size=grid_size,
            source_location=goal_location,
            sigma=10.0,
        ),
        sensor_model=ConcentrationSensor(),
        action_model=DiscreteGridActions(step_size=1),
        reward_fn=SparseGoalReward(goal_position=goal_location, goal_radius=5.0),
    )
    env2 = PlumeEnv(
        grid_size=grid_size,
        source_location=goal_location,
        max_steps=25,
        plume=GaussianPlume(
            grid_size=grid_size,
            source_location=goal_location,
            sigma=10.0,
        ),
        sensor_model=ConcentrationSensor(),
        action_model=DiscreteGridActions(step_size=1),
        reward_fn=SparseGoalReward(goal_position=goal_location, goal_radius=5.0),
    )

    try:
        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)

        assert np.array_equal(obs1, obs2)
        assert info1["agent_position"] == info2["agent_position"]
        assert info1["seed"] == info2["seed"] == 42
    finally:
        env1.close()
        env2.close()


def test_invalid_action_raises_validation_error(plume_env: PlumeEnv) -> None:
    plume_env.reset(seed=0)

    with pytest.raises(ValidationError, match="Invalid action"):
        plume_env.step(999)


def test_gymnasium_api_surface_present(plume_env: PlumeEnv) -> None:
    assert hasattr(plume_env, "action_space")
    assert hasattr(plume_env, "observation_space")
    assert hasattr(plume_env, "reset")
    assert hasattr(plume_env, "step")
    assert hasattr(plume_env, "close")
    assert hasattr(plume_env, "render")
    assert hasattr(plume_env, "metadata")

    reset_result = plume_env.reset(seed=0)
    assert isinstance(reset_result, tuple)
    assert len(reset_result) == 2

    step_result = plume_env.step(0)
    assert isinstance(step_result, tuple)
    assert len(step_result) == 5
