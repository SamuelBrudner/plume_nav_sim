import numpy as np
import pytest
from unittest.mock import patch

from plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv


def make_env(**kwargs):
    return PlumeNavigationEnv(video_path="nonexistent.mp4", **kwargs)


def test_use_gymnasium_api_true_returns_gymnasium_tuples():
    env = make_env(use_gymnasium_api=True)
    result = env.reset()
    assert isinstance(result, tuple) and len(result) == 2
    action = env.action_space.sample()
    step_result = env.step(action)
    assert len(step_result) == 5


def test_use_gymnasium_api_false_returns_legacy_tuples():
    env = make_env(use_gymnasium_api=False)
    result = env.reset()
    assert not isinstance(result, tuple)
    action = env.action_space.sample()
    step_result = env.step(action)
    assert len(step_result) == 4


def test_force_legacy_overrides_gymnasium_param():
    env = make_env(use_gymnasium_api=True, _force_legacy_api=True)
    result = env.reset()
    assert not isinstance(result, tuple)
    action = env.action_space.sample()
    step_result = env.step(action)
    assert len(step_result) == 4


def test_detect_legacy_caller_used_when_param_absent():
    with patch("plume_nav_sim.envs.plume_navigation_env._detect_legacy_gym_caller", return_value=True) as mock_detect:
        env = make_env()
        mock_detect.assert_called_once()
        result = env.reset()
        assert not isinstance(result, tuple)
        action = env.action_space.sample()
        step_result = env.step(action)
        assert len(step_result) == 4
