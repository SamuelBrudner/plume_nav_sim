from __future__ import annotations

import gymnasium as gym
import numpy as np

from plume_nav_sim.observations.history_wrappers import ConcentrationNBackWrapper


class _SequenceEnv(gym.Env):
    """Minimal Env yielding a fixed sequence of scalar concentration values.

    - observation_space: Box(1,)
    - action_space: Discrete(1) (ignored)
    - reset() returns the first value; step() advances through the sequence
    """

    metadata = {"render_modes": []}

    def __init__(self, seq: list[float], *, low: float = 0.0, high: float = 1.0):
        assert len(seq) >= 1, "Sequence must contain at least one value"
        self._seq = [float(x) for x in seq]
        self._i = 0
        self.observation_space = gym.spaces.Box(
            low=np.array([low], dtype=np.float32),
            high=np.array([high], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed: int | None = None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        self._i = 0
        c = np.array([self._seq[self._i]], dtype=self.observation_space.dtype)
        return c, {}

    def step(self, action):  # type: ignore[override]
        self._i = min(self._i + 1, len(self._seq) - 1)
        c = np.array([self._seq[self._i]], dtype=self.observation_space.dtype)
        terminated = self._i == len(self._seq) - 1
        truncated = False
        reward = 0.0
        info = {}
        return c, reward, terminated, truncated, info


def test_nback_observation_space_shape_and_dtype():
    base = _SequenceEnv([0.1, 0.2, 0.3])
    wrapped = ConcentrationNBackWrapper(base, n=4)
    space = wrapped.observation_space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (4,)
    assert space.dtype == np.float32
    # Bounds repeat original
    assert np.allclose(space.low, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(space.high, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))


def test_nback_initial_padding_and_updates():
    seq = [0.1, 0.2, 0.3, 0.4]
    base = _SequenceEnv(seq)
    n = 3
    wrapped = ConcentrationNBackWrapper(base, n=n)

    obs, info = wrapped.reset(seed=123)
    # Initial padding: [c0, c0, c0]
    np.testing.assert_allclose(obs, np.array([seq[0]] * n, dtype=np.float32))

    # First step -> [c0, c0, c1]
    obs, r, term, trunc, inf = wrapped.step(0)
    np.testing.assert_allclose(
        obs, np.array([seq[0], seq[0], seq[1]], dtype=np.float32)
    )

    # Second step -> [c0, c1, c2]
    obs, r, term, trunc, inf = wrapped.step(0)
    np.testing.assert_allclose(
        obs, np.array([seq[0], seq[1], seq[2]], dtype=np.float32)
    )

    # Third step -> [c1, c2, c3]
    obs, r, term, trunc, inf = wrapped.step(0)
    np.testing.assert_allclose(
        obs, np.array([seq[1], seq[2], seq[3]], dtype=np.float32)
    )
