from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[assignment]

from ._compat import ValidationError
from .core.geometry import Coordinates
from .interfaces.fields import VectorField

__all__ = ["ConstantWindField"]


class ConstantWindField(VectorField):
    def __init__(
        self,
        *,
        direction_deg: float = 0.0,
        speed: float = 1.0,
        vector: Optional[Iterable[float]] = None,
    ):
        self.vector = self._resolve_vector(direction_deg, speed, vector)
        self.direction_deg = float(
            (math.degrees(math.atan2(-self.vector[1], self.vector[0]))) % 360.0
        )
        self.speed = float(np.linalg.norm(self.vector))

    def sample(self, position: Coordinates) -> NDArray[np.floating]:
        """Return the constant wind vector (position ignored)."""
        return self.vector.copy()

    @staticmethod
    def _resolve_vector(
        direction_deg: float, speed: float, vector: Optional[Iterable[float]]
    ) -> NDArray[np.floating]:
        if vector is not None:
            try:
                vx, vy = vector
            except Exception as exc:  # pragma: no cover - defensive path
                raise ValidationError("vector must be an iterable of length 2") from exc
            vector_arr = np.array([vx, vy], dtype=np.float32)
        else:
            theta = math.radians(direction_deg)
            vx = float(speed) * math.cos(theta)
            vy = -float(speed) * math.sin(theta)
            vector_arr = np.array([vx, vy], dtype=np.float32)

        if not np.all(np.isfinite(vector_arr)):
            raise ValidationError(
                f"Wind vector must be finite, got {vector_arr.tolist()}"
            )

        if vector_arr.shape != (2,):  # pragma: no cover - shape guard
            raise ValidationError(
                f"Wind vector must have shape (2,), got {vector_arr.shape}"
            )

        return vector_arr
