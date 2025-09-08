import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src"
WIND_INIT = ROOT / "plume_nav_sim" / "models" / "wind" / "__init__.py"


def test_import_fails_with_loguru_stub(monkeypatch):
    """Wind module import should error if loguru lacks configure."""
    loguru_stub = types.ModuleType("loguru")
    loguru_stub.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None,
        info=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "loguru", loguru_stub)

    plume_pkg = types.ModuleType("plume_nav_sim")
    plume_pkg.__path__ = [str(ROOT / "plume_nav_sim")]
    models_pkg = types.ModuleType("plume_nav_sim.models")
    models_pkg.__path__ = [str(ROOT / "plume_nav_sim" / "models")]
    monkeypatch.setitem(sys.modules, "plume_nav_sim", plume_pkg)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.models", models_pkg)

    const_stub = types.ModuleType("plume_nav_sim.models.wind.constant_wind")
    const_stub.ConstantWindField = object
    const_stub.ConstantWindFieldConfig = object
    const_stub.create_constant_wind_field = lambda *a, **k: None
    turb_stub = types.ModuleType("plume_nav_sim.models.wind.turbulent_wind")
    turb_stub.TurbulentWindField = object
    turb_stub.TurbulentWindFieldConfig = object
    time_stub = types.ModuleType("plume_nav_sim.models.wind.time_varying_wind")
    time_stub.TimeVaryingWindField = object
    time_stub.TimeVaryingWindFieldConfig = object
    monkeypatch.setitem(sys.modules, "plume_nav_sim.models.wind.constant_wind", const_stub)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.models.wind.turbulent_wind", turb_stub)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.models.wind.time_varying_wind", time_stub)

    spec = importlib.util.spec_from_file_location(
        "plume_nav_sim.models.wind", WIND_INIT, submodule_search_locations=[str(WIND_INIT.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.models.wind", module)

    with pytest.raises(ImportError, match="loguru"):
        spec.loader.exec_module(module)
