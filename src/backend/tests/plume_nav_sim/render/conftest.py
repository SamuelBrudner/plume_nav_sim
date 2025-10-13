"""
Render test-specific pytest configuration and helpers.

Provides a stub for colormap resolution to decouple tests from Matplotlib
internals when the exact colormap identity is irrelevant.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def stub_colormap(monkeypatch):
    """Stub color map resolution to a stable colormap (viridis).

    Use in tests that do not validate specific colormap semantics but need a
    functioning colormap object without touching Matplotlib deprecation paths.
    """
    import matplotlib

    from plume_nav_sim.render import color_schemes as cs

    def _fixed_cmap(_name: str):
        try:
            return matplotlib.colormaps.get_cmap("viridis")
        except Exception:
            return cs.cm.get_cmap("viridis")

    monkeypatch.setattr(cs, "_safe_get_cmap", _fixed_cmap, raising=True)
    yield
