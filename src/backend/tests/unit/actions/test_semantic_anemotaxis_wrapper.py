import gymnasium as gym
import numpy as np
import pytest

from plume_nav_sim.actions.semantic_anemotaxis_wrapper import (
    SemanticAnemotaxisActionWrapper,
)
from plume_nav_sim.core.enums import Action
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.wind_field import ConstantWindField


class _DummyWindEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, wind_vector=(1.0, 0.0), seed: int | None = None):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._wind_field = ConstantWindField(vector=wind_vector)
        self._agent_state = AgentState(position=Coordinates(5, 5))
        self.grid_size = GridSize(10, 10)
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._agent_state = AgentState(position=Coordinates(5, 5))
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        return obs, {"seed": seed}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        info = {
            "agent_position": (
                self._agent_state.position.x,
                self._agent_state.position.y,
            )
        }
        return obs, 0.0, False, False, info


class _BadActionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def reset(self, **kwargs):
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}


def test_requires_discrete_four_space():
    env = _BadActionEnv()
    with pytest.raises(ValueError):
        SemanticAnemotaxisActionWrapper(env)


def test_surge_moves_upwind():
    env = _DummyWindEnv(wind_vector=(1.0, 0.0))
    wrapper = SemanticAnemotaxisActionWrapper(env)
    wrapper.reset()

    mapped = wrapper.action(SemanticAnemotaxisActionWrapper.SURGE)

    assert mapped == int(Action.LEFT)


def test_cast_draws_from_non_upwind_actions_with_env_rng():
    seed = 7
    env = _DummyWindEnv(wind_vector=(1.0, 0.0), seed=seed)
    wrapper = SemanticAnemotaxisActionWrapper(env)
    wrapper.reset()

    # Upwind for this vector is LEFT, so CAST should pull from [UP, RIGHT, DOWN]
    candidates = [int(Action.UP), int(Action.RIGHT), int(Action.DOWN)]
    expected_rng = np.random.default_rng(seed)
    expected = [
        candidates[int(expected_rng.integers(0, len(candidates)))] for _ in range(3)
    ]

    mapped = [wrapper.action(SemanticAnemotaxisActionWrapper.CAST) for _ in range(3)]

    assert mapped == expected
    assert all(choice != int(Action.LEFT) for choice in mapped)


def test_zero_wind_defaults_to_up_and_casts_elsewhere():
    env = _DummyWindEnv(wind_vector=(0.0, 0.0))
    wrapper = SemanticAnemotaxisActionWrapper(env)
    wrapper.reset()

    surge_action = wrapper.action(SemanticAnemotaxisActionWrapper.SURGE)
    cast_action = wrapper.action(SemanticAnemotaxisActionWrapper.CAST)

    assert surge_action == int(Action.UP)
    assert cast_action in {int(Action.RIGHT), int(Action.DOWN), int(Action.LEFT)}


def test_invalid_semantic_action_rejected():
    env = _DummyWindEnv()
    wrapper = SemanticAnemotaxisActionWrapper(env)
    wrapper.reset()

    with pytest.raises(ValueError):
        wrapper.action(5)
