import builtins
import importlib.util
from pathlib import Path
import sys

import pytest


def test_turbulent_wind_requires_scipy(monkeypatch):
    """Turbulent wind module should raise ImportError when SciPy is missing."""
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "plume_nav_sim"
        / "models"
        / "wind"
        / "turbulent_wind.py"
    )

    # Simulate missing SciPy
    for mod in list(sys.modules):
        if mod.startswith("scipy"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("No module named 'scipy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location(
        "turbulent_wind_missing_scipy", module_path
    )
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(ImportError) as excinfo:
        spec.loader.exec_module(module)
    assert "scipy" in str(excinfo.value).lower()
