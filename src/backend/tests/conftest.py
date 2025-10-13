"""
Pytest configuration for visualization tests to normalize Matplotlib behavior
and harden CI stability.

Applies a deterministic, headless backend and rcParams, and guards logging
from emitting exceptions during teardown.
"""

import logging

import matplotlib

# Use headless backend consistently across tests before importing pyplot
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import pytest  # noqa: E402
from matplotlib import rcParams  # noqa: E402


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


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    """Ensure figures are closed after each test to avoid cross-test leakage."""
    yield
    try:
        plt.close("all")
    except Exception:
        pass
