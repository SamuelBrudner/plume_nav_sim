import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "missing_module",
    [
        "plume_nav_sim.models.plume.gaussian_plume",
        "plume_nav_sim.models.plume.turbulent_plume",
        "plume_nav_sim.models.plume.video_plume_adapter",
    ],
)
def test_plume_package_requires_core_components(monkeypatch, missing_module):
    """Ensure ImportError is raised when a core plume model module is absent."""
    # Ensure fresh import attempt for the plume package
    monkeypatch.delitem(sys.modules, "plume_nav_sim.models.plume", raising=False)
    # Simulate missing core component
    monkeypatch.setitem(sys.modules, missing_module, None)

    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.models.plume")
