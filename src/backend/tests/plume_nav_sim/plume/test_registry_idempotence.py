"""Regression test for Gaussian plume construction idempotence."""

import numpy as np

from plume_nav_sim.plume.gaussian import GaussianPlume


def test_gaussian_plume_recreation_is_deterministic() -> None:
    plume1 = GaussianPlume(grid_size=(32, 32), source_location=(16, 16), sigma=4.0)
    plume2 = GaussianPlume(grid_size=(32, 32), source_location=(16, 16), sigma=4.0)
    assert np.array_equal(plume1.field_array, plume2.field_array)
