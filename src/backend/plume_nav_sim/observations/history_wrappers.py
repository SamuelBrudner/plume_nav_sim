from __future__ import annotations

from collections import deque
from typing import Deque

import gymnasium as gym
import numpy as np


class ConcentrationNBackWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, n: int = 3, *, dtype: np.dtype | None = None):
        super().__init__(env)
        if n <= 0:
            raise ValueError(f"n must be >= 1, got {n}")
        orig = env.observation_space
        if not isinstance(orig, gym.spaces.Box) or orig.shape != (1,):
            raise ValueError(
                "ConcentrationNBackWrapper expects a Box observation of shape (1,)"
            )
        self.n = n
        self._buf: Deque[float] = deque(maxlen=self.n)
        low0 = float(np.asarray(orig.low).reshape(-1)[0])
        high0 = float(np.asarray(orig.high).reshape(-1)[0])
        out_dtype = dtype if dtype is not None else (orig.dtype or np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.full(self.n, low0, dtype=out_dtype),
            high=np.full(self.n, high0, dtype=out_dtype),
            dtype=out_dtype,
        )

    def reset(self, **kwargs):  # type: ignore[override]
        self._buf.clear()
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation: np.ndarray) -> np.ndarray:  # type: ignore[override]
        c_now = float(observation[0])
        if not self._buf:
            # Pad with initial value
            self._buf.extend([c_now] * self.n)
        else:
            self._buf.append(c_now)
        return np.array(list(self._buf), dtype=self.observation_space.dtype)
