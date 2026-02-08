from __future__ import annotations

import builtins

import numpy as np
import pytest

from plume_nav_sim.utils.notebook import display_frame, live_display


class _DummyPolicy:
    def reset(self, *, seed: int | None = None) -> None:
        _ = seed

    def select_action(self, observation, *, explore: bool = True):  # noqa: ANN001
        _ = (observation, explore)
        return 0


class _DummyEnv:
    def __init__(self, horizon: int = 3) -> None:
        self._horizon = int(horizon)
        self._steps = 0

    def reset(self, seed: int | None = None):
        _ = seed
        self._steps = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):  # noqa: ANN001
        _ = action
        self._steps += 1
        terminated = self._steps >= self._horizon
        obs = np.array([float(self._steps)], dtype=np.float32)
        return obs, 0.0, terminated, False, {}

    def render(self, mode: str = "rgb_array"):
        _ = mode
        return np.full((4, 4, 3), self._steps, dtype=np.uint8)


def _block_ipython_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def _import(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if name == "IPython" or name.startswith("IPython."):
            raise ImportError("IPython unavailable for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


def test_display_frame_returns_without_ipython(monkeypatch: pytest.MonkeyPatch):
    _block_ipython_imports(monkeypatch)

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert display_frame(frame) is None


def test_live_display_headless_fallback(monkeypatch: pytest.MonkeyPatch):
    _block_ipython_imports(monkeypatch)

    env = _DummyEnv(horizon=3)
    policy = _DummyPolicy()

    frames = live_display(env, policy, seed=123, max_steps=10, fps=0)

    assert len(frames) == 3
    assert all(isinstance(frame, np.ndarray) for frame in frames)
    assert all(frame.shape == (4, 4, 3) for frame in frames)
