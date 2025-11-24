from __future__ import annotations

import gymnasium as gym
import numpy as np

from plume_nav_sim.runner.runner import run_episode


class _DictObservationEnv:
    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32)}
        )
        self._step = 0

    def reset(self, seed=None):
        self._step = 0
        return {"state": np.zeros(2, dtype=np.float32)}, {}

    def step(self, action):
        self._step += 1
        next_obs = {"state": np.full(2, float(self._step), dtype=np.float32)}
        reward = float(np.sum(action))
        terminated = self._step >= 2
        truncated = False
        info: dict[str, object] = {}
        return next_obs, reward, terminated, truncated, info


class _DictObservationPolicy:
    def __init__(self, action_space: gym.Space) -> None:
        self.action_space = action_space
        self._seed = None

    def reset(self, *, seed=None):
        self._seed = seed

    def select_action(self, observation, *, explore: bool = True):  # noqa: ARG002
        assert isinstance(observation, dict)
        base = observation["state"]
        action = np.asarray(base, dtype=np.float32) + 0.5
        return np.clip(action, self.action_space.low, self.action_space.high)


def test_runner_accepts_dict_observation_and_box_action():
    env = _DictObservationEnv()
    policy = _DictObservationPolicy(env.action_space)

    events = []
    result = run_episode(env, policy, max_steps=2, seed=42, on_step=events.append)

    assert result.steps == 2
    assert len(events) == 2
    assert all(isinstance(ev.obs, dict) for ev in events)
    assert all(isinstance(ev.action, np.ndarray) for ev in events)
    for ev in events:
        assert env.action_space.contains(ev.action)
        assert ev.obs["state"].shape == (2,)
