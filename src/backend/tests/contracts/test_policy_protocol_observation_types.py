"""Runtime protocol tests covering composite observations and non-discrete actions."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from plume_nav_sim.interfaces import Policy as PolicyProtocol


class _DictObservationPolicy:
    def __init__(self) -> None:
        self._action_space = gym.spaces.Discrete(2)
        self._seed = None

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def reset(self, *, seed: int | None = None) -> None:
        self._seed = seed

    def select_action(self, observation, *, explore: bool = True):  # noqa: ARG002
        assert isinstance(observation, dict)
        return 0


class _TupleObservationPolicy:
    def __init__(self) -> None:
        self._action_space = gym.spaces.MultiDiscrete([2, 2])

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def reset(self, *, seed: int | None = None) -> None:  # noqa: ARG002
        return None

    def select_action(self, observation, *, explore: bool = True):  # noqa: ARG002
        assert isinstance(observation, tuple)
        return np.array([1, 0], dtype=np.int64)


def test_policy_protocol_accepts_dict_observation():
    policy = _DictObservationPolicy()
    assert isinstance(policy, PolicyProtocol)
    action = policy.select_action({"x": 1.0}, explore=True)
    assert policy.action_space.contains(action)


def test_policy_protocol_accepts_tuple_observation():
    policy = _TupleObservationPolicy()
    assert isinstance(policy, PolicyProtocol)
    action = policy.select_action((0.0, 1.0), explore=False)
    assert policy.action_space.contains(action)
