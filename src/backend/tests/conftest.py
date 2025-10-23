"""
Pytest configuration for visualization tests to normalize Matplotlib behavior
and harden CI stability.

Applies a deterministic, headless backend and rcParams, and guards logging
from emitting exceptions during teardown.
"""

import logging

import matplotlib
import numpy as np

import pytest  # noqa: E402
from matplotlib import rcParams  # noqa: E402

from plume_nav_sim.core.types import Coordinates, GridSize
from plume_nav_sim.render.base_renderer import create_render_context
from tests.test_rendering import create_dual_mode_test_environment

# Skip archived tests - implementation details may be outdated
collect_ignore = ["archived"]

# Use headless backend consistently across tests before importing pyplot
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)


@pytest.fixture(scope="session", autouse=True)
def _normalize_matplotlib_and_logging():
    """Normalize Matplotlib rcParams and guard logging exceptions for CI."""
    # Prevent logging module from printing handler/stream errors during teardown
    logging.raiseExceptions = False

    # Normalize core rcParams for consistent sizing and rendering
    rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 100,
            # Use a deterministic font family available in Matplotlib wheels
            "font.family": ["DejaVu Sans"],
            # Keep grids off by default; tests enable grids explicitly when needed
            "axes.grid": False,
            # Ensure antialiasing for consistent visuals (does not affect Agg determinism)
            "text.antialiased": True,
        }
    )

    yield

    # Best-effort cleanup of any remaining figures at end of session
    try:
        plt.close("all")
    except Exception:
        pass


DUAL_MODE_TEST_GRID_SIZES = [(32, 32), (64, 64), (128, 128)]
INTEGRATION_TEST_SEEDS = [42, 123, 456]


@pytest.fixture(scope="session")
def matplotlib_available():
    """Session-wide check for matplotlib availability used by multiple suites."""

    try:
        import matplotlib.backends.backend_agg  # noqa: F401  # type: ignore

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        plt.close(fig)
        return True
    except Exception:
        pytest.skip("Matplotlib not available for rendering tests")


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    """Ensure figures are closed after each test to avoid cross-test leakage."""
    yield
    try:
        plt.close("all")
    except Exception:
        pass


@pytest.fixture
def edge_case_test_env():
    """Provide a reusable environment configured for edge-case testing suites."""

    env = create_dual_mode_test_environment(
        grid_size=(32, 32),
        initial_render_mode="rgb_array",
        test_config={"enable_edge_case_testing": True},
    )

    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def integration_test_env():
    env = create_dual_mode_test_environment(
        grid_size=(32, 32),
        initial_render_mode="rgb_array",
        enable_performance_monitoring=True,
    )
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def unit_test_env():
    env = create_dual_mode_test_environment(
        grid_size=(32, 32),
        initial_render_mode="rgb_array",
        enable_performance_monitoring=False,
    )
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def performance_test_env():
    env = create_dual_mode_test_environment(
        grid_size=(64, 64),
        initial_render_mode="rgb_array",
        enable_performance_monitoring=True,
    )
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def test_render_context(integration_test_env):
    env = integration_test_env
    env.reset()

    concentration_field = np.random.rand(32, 32).astype(np.float32)
    agent_pos = Coordinates(x=5, y=5)
    source_pos = Coordinates(x=16, y=16)
    grid_size = GridSize(width=32, height=32)

    return create_render_context(
        concentration_field=concentration_field,
        agent_position=agent_pos,
        source_position=source_pos,
        grid_size=grid_size,
    )


@pytest.fixture
def test_coordinates():
    return {
        "agent_positions": [Coordinates(x=5, y=5), Coordinates(x=10, y=15)],
        "source_positions": [Coordinates(x=16, y=16), Coordinates(x=20, y=20)],
    }


@pytest.fixture
def test_grid_sizes():
    return list(DUAL_MODE_TEST_GRID_SIZES)


@pytest.fixture
def test_seeds():
    return list(INTEGRATION_TEST_SEEDS)
