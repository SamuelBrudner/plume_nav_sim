import importlib.util
import builtins
import sys
import types
from pathlib import Path
import pytest


def test_loguru_absence_raises_import_error(monkeypatch):
    """Wind registry should raise ImportError if loguru is missing."""
    monkeypatch.delitem(sys.modules, "loguru", raising=False)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "loguru":
            raise ImportError("No module named 'loguru'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Stub dependent modules to isolate loguru behavior
    pkg = types.ModuleType("plume_nav_sim")
    models_pkg = types.ModuleType("plume_nav_sim.models")
    wind_pkg = types.ModuleType("plume_nav_sim.models.wind")
    protocols_pkg = types.ModuleType("plume_nav_sim.protocols")
    wind_field_mod = types.ModuleType("plume_nav_sim.protocols.wind_field")
    setattr(wind_field_mod, "WindFieldProtocol", object)

    monkeypatch.setitem(sys.modules, "plume_nav_sim", pkg)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.models", models_pkg)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.models.wind", wind_pkg)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.protocols", protocols_pkg)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.protocols.wind_field", wind_field_mod)

    for name in ["constant_wind", "turbulent_wind", "time_varying_wind"]:
        mod = types.ModuleType(f"plume_nav_sim.models.wind.{name}")
        monkeypatch.setitem(sys.modules, f"plume_nav_sim.models.wind.{name}", mod)

    wind_path = Path(__file__).resolve().parents[2] / "src/plume_nav_sim/models/wind/__init__.py"
    spec = importlib.util.spec_from_file_location("plume_nav_sim.models.wind", wind_path)

    with pytest.raises(ImportError):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
