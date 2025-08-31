import logging
import types
from collections import namedtuple
import pytest

# The compat module will be created later; we still import to show tests expecting it
from plume_nav_sim.envs import compat as compat_module

# We expect the module to provide APIVersionResult, CompatibilityMode, detect_api_version, wrap_environment


def make_dummy_gym_module():
    dummy = types.SimpleNamespace(__name__='gym', __version__='0.26.0')
    return dummy


def test_detect_api_version_legacy(monkeypatch):
    dummy_gym = make_dummy_gym_module()

    def legacy_caller():
        gym = dummy_gym  # noqa: F841 local variable named gym
        return compat_module.detect_api_version()

    result = legacy_caller()
    assert result.is_legacy is True
    assert result.detection_method == 'stack_inspection'


def test_detect_api_version_modern():
    result = compat_module.detect_api_version()
    assert result.is_legacy is False
    assert result.detection_method == 'stack_inspection'


class ModernEnv:
    def reset(self):
        return 'obs', {'meta': 1}

    def step(self, action):  # pragma: no cover - action unused
        return 'obs', 1.0, False, False, {'meta': 2}


class LegacyEnv:
    def reset(self):
        return 'obs'

    def step(self, action):  # pragma: no cover - action unused
        return 'obs', 1.0, True, {'meta': 3}


def test_wrap_environment_to_legacy(caplog):
    caplog.set_level(logging.DEBUG)
    dummy_detection = compat_module.APIVersionResult(True, 1.0, 'test')
    mode = compat_module.CompatibilityMode(True, dummy_detection, False, 'abc')
    env = compat_module.wrap_environment(ModernEnv(), mode)
    obs = env.reset()
    assert obs == 'obs'
    assert 'Converting reset output' in caplog.text
    obs2, reward, done, info = env.step(0)
    assert (obs2, reward, done, info) == ('obs', 1.0, False, {'meta': 2})
    assert 'Converting step output' in caplog.text


def test_wrap_environment_to_modern(caplog):
    caplog.set_level(logging.DEBUG)
    dummy_detection = compat_module.APIVersionResult(False, 1.0, 'test')
    mode = compat_module.CompatibilityMode(False, dummy_detection, False, 'def')
    env = compat_module.wrap_environment(LegacyEnv(), mode)
    obs, info = env.reset()
    assert (obs, info) == ('obs', {})
    assert 'Converting reset output' in caplog.text
    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert (obs2, reward, terminated, truncated, info2) == (
        'obs', 1.0, True, False, {'meta': 3}
    )
    assert 'Converting step output' in caplog.text
