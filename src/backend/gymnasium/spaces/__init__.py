"""Minimal spaces module compatible with the subset of Gymnasium API used in tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


class Space:
    def contains(self, x: Any) -> bool:  # pragma: no cover - very basic shim
        return True

    def __contains__(self, x: Any) -> bool:  # Allow `x in space` checks
        return self.contains(x)


@dataclass
class Box(Space):
    low: Any
    high: Any
    shape: Tuple[int, ...]
    dtype: Any = np.float32

    def sample(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=self.dtype)


@dataclass
class Discrete(Space):
    n: int

    def __post_init__(self) -> None:
        # Local RNG to support deterministic sampling when seeding via env
        self._rng = np.random.default_rng()

    def seed(self, seed: Optional[int] = None) -> None:  # type: ignore[override]
        # Gymnasium seeds spaces via a seed() method, not constructor kwargs
        self._rng = np.random.default_rng(None if seed is None else int(seed))

    def sample(self) -> int:  # pragma: no cover - trivial
        return int(self._rng.integers(0, self.n))

    def contains(self, x: Any) -> bool:
        try:
            xi = int(x)
            return 0 <= xi < self.n
        except Exception:
            return False


class Dict(Space):  # noqa: N801 - match API name
    def __init__(self, spaces: Mapping[str, Space]):
        self.spaces: Dict[str, Space] = dict(spaces)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, Mapping):
            return False
        for k, sp in self.spaces.items():
            if k not in x or not sp.contains(x[k]):
                return False
        return True


__all__ = ["Space", "Box", "Discrete", "Dict"]
