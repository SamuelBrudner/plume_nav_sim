"""Coordinate frame utilities with consistent angle handling.

This module centralizes basic geometric operations used across the
navigation stack.  All rotations use degrees as the public API while the
internal computations operate in radians with double precision to avoid
numerical drift.  Angles are always normalised to the ``[0, 360)`` range
so that ``0`` and ``360`` represent the same orientation.
"""

from __future__ import annotations

from typing import Union

import numpy as np


Number = Union[float, np.ndarray]


def normalize_angle(angle: Number) -> Number:
    """Normalise ``angle`` to the ``[0, 360)`` range.

    Parameters
    ----------
    angle:
        A single angle or array of angles expressed in degrees.

    Returns
    -------
    The normalised angle with the same shape as the input.
    """
    return np.mod(angle, 360.0)


def rotation_matrix(angle_deg: float) -> np.ndarray:
    """Return a 2×2 rotation matrix for ``angle_deg`` degrees.

    The computation uses ``float64`` precision to keep successive
    compositions stable when simulations run with very small time steps.
    """
    angle_rad = np.deg2rad(normalize_angle(float(angle_deg)))
    c, s = np.cos(angle_rad, dtype=np.float64), np.sin(angle_rad, dtype=np.float64)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate(point: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate ``point`` by ``angle_deg`` degrees around the origin.

    ``point`` is converted to ``float64`` to preserve precision.
    """
    return rotation_matrix(angle_deg) @ np.asarray(point, dtype=np.float64)

