import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "plume_nav_sim"
    / "models"
    / "wind"
    / "time_varying_wind.py"
)


@pytest.fixture(autouse=True)
def stub_protocols(monkeypatch):
    proto_module = types.ModuleType("plume_nav_sim.protocols.wind_field")
    class DummyProtocol:  # minimal stand-in for protocol
        pass
    proto_module.WindFieldProtocol = DummyProtocol
    protocols_module = types.ModuleType("plume_nav_sim.protocols")
    protocols_module.wind_field = proto_module
    package_module = types.ModuleType("plume_nav_sim")
    package_module.protocols = protocols_module
    monkeypatch.setitem(sys.modules, "plume_nav_sim", package_module)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.protocols", protocols_module)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.protocols.wind_field", proto_module)
    yield


@pytest.mark.parametrize("missing_module", ["scipy", "omegaconf", "pandas"])
def test_import_errors_when_dependency_missing(monkeypatch, missing_module):
    monkeypatch.setitem(sys.modules, missing_module, None)
    spec = importlib.util.spec_from_file_location("time_varying_wind", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["time_varying_wind"] = module
    with pytest.raises(ImportError):
        spec.loader.exec_module(module)
