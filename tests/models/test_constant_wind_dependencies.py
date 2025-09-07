import importlib.util
from loguru import logger
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
    / "constant_wind.py"
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


def test_import_error_without_hydra(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "omegaconf", None)
    spec = importlib.util.spec_from_file_location("constant_wind", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["constant_wind"] = module
    with caplog.at_level(logger.ERROR):
        with pytest.raises(ImportError):
            spec.loader.exec_module(module)
        assert "omegaconf" in caplog.text
