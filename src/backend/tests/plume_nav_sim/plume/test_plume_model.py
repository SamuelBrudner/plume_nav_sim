"""Tests for simplified plume models."""

import numpy as np

from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.plume.gaussian import GaussianPlume
from plume_nav_sim.plume.protocol import ConcentrationField


def test_gaussian_plume_protocol() -> None:
    plume = GaussianPlume(grid_size=(32, 32), source_location=(16, 16), sigma=2.0)
    assert isinstance(plume, ConcentrationField)


def test_gaussian_field_shape_and_dtype() -> None:
    grid = GridSize(width=64, height=32)
    plume = GaussianPlume(grid_size=grid, source_location=Coordinates(10, 5), sigma=3.0)
    assert plume.field_array.shape == (grid.height, grid.width)
    assert plume.field_array.dtype == np.float32


def test_gaussian_sample_at_source() -> None:
    plume = GaussianPlume(grid_size=(64, 64), source_location=(12, 21), sigma=5.0)
    assert abs(plume.sample(12, 21) - 1.0) < 1e-6


def test_gaussian_sample_out_of_bounds() -> None:
    plume = GaussianPlume(grid_size=(16, 16), source_location=(8, 8), sigma=2.0)
    assert plume.sample(-1, 0) == 0.0
    assert plume.sample(0, -1) == 0.0
    assert plume.sample(100, 0) == 0.0
    assert plume.sample(0, 100) == 0.0


def test_gaussian_reset_is_idempotent() -> None:
    plume = GaussianPlume(grid_size=(16, 16), source_location=(8, 8), sigma=2.0)
    before = plume.field_array.copy()
    plume.reset(seed=123)
    assert np.array_equal(before, plume.field_array)
