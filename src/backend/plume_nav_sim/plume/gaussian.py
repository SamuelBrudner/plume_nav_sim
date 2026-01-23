"""Static Gaussian plume implementation with precomputed field array."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..constants import DEFAULT_PLUME_SIGMA
from ..core.geometry import Coordinates, GridSize
from ..utils.exceptions import ValidationError

__all__ = ["GaussianPlume"]


def _coerce_grid_size(grid_size: GridSize | tuple[int, int]) -> GridSize:
    if isinstance(grid_size, GridSize):
        return grid_size
    if isinstance(grid_size, tuple) and len(grid_size) == 2:
        return GridSize(width=int(grid_size[0]), height=int(grid_size[1]))
    raise ValidationError(
        f"grid_size must be GridSize or tuple[int, int], got {grid_size}"
    )


def _coerce_source_location(
    source_location: Coordinates | tuple[int, int] | None, grid_size: GridSize
) -> Coordinates:
    if source_location is None:
        return grid_size.center()
    if isinstance(source_location, Coordinates):
        location = source_location
    elif isinstance(source_location, tuple) and len(source_location) == 2:
        location = Coordinates(x=int(source_location[0]), y=int(source_location[1]))
    else:
        raise ValidationError(
            "source_location must be Coordinates, tuple[int, int], or None"
        )
    if not grid_size.contains(location):
        raise ValidationError(
            f"source_location {location.to_tuple()} outside grid {grid_size.to_tuple()}"
        )
    return location


class GaussianPlume:
    """Static Gaussian plume precomputing a concentration field."""

    def __init__(
        self,
        *,
        grid_size: GridSize | tuple[int, int],
        source_location: Coordinates | tuple[int, int] | None = None,
        sigma: float = DEFAULT_PLUME_SIGMA,
    ) -> None:
        if sigma <= 0:
            raise ValidationError(f"sigma must be positive, got {sigma}")
        self.grid_size = _coerce_grid_size(grid_size)
        self.source_location = _coerce_source_location(source_location, self.grid_size)
        self.sigma = float(sigma)
        self.field_array = self._generate_field()
        self._seed: Optional[int] = None

    def _generate_field(self) -> np.ndarray:
        x = np.arange(self.grid_size.width, dtype=np.float32)
        y = np.arange(self.grid_size.height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        dx = xx - float(self.source_location.x)
        dy = yy - float(self.source_location.y)
        denom = 2.0 * (self.sigma**2)
        field = np.exp(-(dx * dx + dy * dy) / denom).astype(np.float32)
        return np.clip(field, 0.0, 1.0)

    def reset(self, seed: int | None = None) -> None:
        self._seed = None if seed is None else int(seed)

    def sample(self, x: float, y: float, t: float | None = None) -> float:
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < self.grid_size.width and 0 <= iy < self.grid_size.height:
            return float(self.field_array[iy, ix])
        return 0.0
