import numpy as np
import pytest

from src.odor_plume_nav.coordinate_frame import (
    normalize_angle,
    rotate,
    compose_angles,
)
from src.odor_plume_nav.utils.navigator_utils import update_positions_and_orientations


def test_normalize_angle_wraps():
    assert normalize_angle(360.0) == pytest.approx(0.0)
    assert normalize_angle(-90.0) == pytest.approx(270.0)
    assert normalize_angle(720.0) == pytest.approx(0.0)


def test_compose_angles_normalizes():
    assert compose_angles(350.0, 20.0) == pytest.approx(10.0)


def test_rotation_composition():
    point = np.array([1.0, 0.0])
    r1 = rotate(point, 90.0)
    r2 = rotate(r1, 180.0)
    r_total = rotate(point, compose_angles(90.0, 180.0))
    assert np.allclose(r2, r_total)


def test_update_positions_and_orientations_negative_dt():
    positions = np.zeros((1, 2))
    orientations = np.zeros(1)
    speeds = np.zeros(1)
    ang_vel = np.zeros(1)
    with pytest.raises(ValueError):
        update_positions_and_orientations(positions, orientations, speeds, ang_vel, dt=-0.1)


def test_small_dt_consistency():
    positions = np.zeros((1, 2))
    orientations = np.zeros(1)
    speeds = np.array([1.0])
    ang_vel = np.array([0.0])
    dt = 0.001
    steps = 1000
    for _ in range(steps):
        update_positions_and_orientations(positions, orientations, speeds, ang_vel, dt=dt)
    assert positions[0, 0] == pytest.approx(1.0, rel=1e-6)
    assert positions[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert orientations[0] == pytest.approx(0.0)
