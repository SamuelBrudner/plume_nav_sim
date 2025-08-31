import warnings
import gymnasium as gym
import pytest

from plume_nav_sim.envs.compat import APIVersionResult


def test_gym_make_normalizes_and_wraps(monkeypatch, caplog):
    """gym_make should normalize env IDs, emit deprecation, and wrap for legacy API."""
    import plume_nav_sim.shims.gym_make as gm

    # Dummy environment returning modern 5-tuple results
    class DummyEnv(gym.Env):
        def reset(self, *, seed=None, options=None):
            return 0, {}

        def step(self, action):
            return 0, 0.0, False, False, {}

    called = {}

    def fake_make(env_id, **kwargs):
        called['env_id'] = env_id
        called['kwargs'] = kwargs
        return DummyEnv()

    monkeypatch.setattr(gym, "make", fake_make)

    def fake_detect():
        return APIVersionResult(True, 1.0, "test")

    monkeypatch.setattr(gm, "detect_api_version", fake_detect)

    caplog.set_level("INFO")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        env = gm.gym_make("OdorPlumeNavigation-v1")
    assert called['env_id'] == "PlumeNavSim-v0"
    assert any(isinstance(warn.message, DeprecationWarning) for warn in w)
    assert len(env.step(0)) == 4
    assert any("gym_make" in rec.message for rec in caplog.records)
