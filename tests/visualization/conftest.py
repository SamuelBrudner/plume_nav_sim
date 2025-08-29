"""Fixtures for visualization tests.

These fixtures generate lightweight sample data used across the
visualization tests.  The data itself is intentionally simple – the
tests only verify that the visualization functions make the expected
matplotlib calls rather than producing visually meaningful figures.

The fixtures live in ``tests/visualization`` so they are only loaded for
the visualization test suite and avoid polluting the global fixture
namespace.  Each fixture aims to be explicit about the shape of the
arrays it returns in order to catch accidental misuse during test
development.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for file based tests.

    The directory is created inside pytest's ``tmp_path`` so it is cleaned
    up automatically after the test session.
    """

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return out_dir


@pytest.fixture
def sample_single_agent_data() -> Dict[str, np.ndarray]:
    """Provide a small trajectory for a single agent.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with ``positions`` of shape ``(T, 2)`` and
        ``orientations`` of shape ``(T,)``.  The ``num_agents`` key is
        included for convenience in a few tests.
    """

    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )
    orientations = np.linspace(0.0, 180.0, len(positions))
    return {"positions": positions, "orientations": orientations, "num_agents": 1}


@pytest.fixture
def sample_multi_agent_data() -> Dict[str, np.ndarray]:
    """Provide trajectory data for a small multi‑agent scenario."""

    positions = np.array(
        [
            [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]],
            [[5.0, 5.0], [6.0, 5.5], [7.0, 6.0]],
        ]
    )
    orientations = np.array(
        [
            [0.0, 45.0, 90.0],
            [180.0, 225.0, 270.0],
        ]
    )
    return {
        "positions": positions,
        "orientations": orientations,
        "num_agents": positions.shape[0],
    }


@pytest.fixture
def sample_plume_data() -> np.ndarray:
    """Return a simple 2‑D plume used for background rendering tests."""

    plume = np.zeros((10, 10), dtype=float)
    plume[4:7, 4:7] = 1.0
    return plume


@pytest.fixture
def hydra_visualization_config():  # type: ignore[override]
    """Minimal Hydra configuration object for visualization tests."""

    try:  # Hydra/OmegaConf may not be installed in minimal environments
        from omegaconf import OmegaConf
    except Exception:  # pragma: no cover - test environments always have it
        pytest.skip("OmegaConf not available")

    config = {
        "resolution": "720p",
        "animation": {"fps": 30, "format": "mp4"},
        "static": {"dpi": 150, "format": "png"},
        "agents": {"max_agents": 100, "color_scheme": "scientific"},
        "headless": False,
    }
    return OmegaConf.create(config)

