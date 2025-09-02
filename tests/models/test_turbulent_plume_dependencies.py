import builtins
import importlib.util
import sys
from pathlib import Path
import types

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "plume_nav_sim" / "models" / "plume" / "turbulent_plume.py"


def stub_protocols(monkeypatch):
    plume_nav_sim = types.ModuleType("plume_nav_sim")
    protocols = types.ModuleType("plume_nav_sim.protocols")
    plume_model = types.ModuleType("plume_nav_sim.protocols.plume_model")
    wind_field = types.ModuleType("plume_nav_sim.protocols.wind_field")

    class PlumeModelProtocol:  # pragma: no cover
        pass

    class WindFieldProtocol:  # pragma: no cover
        pass

    plume_model.PlumeModelProtocol = PlumeModelProtocol
    wind_field.WindFieldProtocol = WindFieldProtocol

    monkeypatch.setitem(sys.modules, "plume_nav_sim", plume_nav_sim)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.protocols", protocols)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.protocols.plume_model", plume_model)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.protocols.wind_field", wind_field)


@pytest.mark.parametrize("missing_module", ["scipy", "numba", "omegaconf"])
def test_turbulent_plume_requires_dependencies(monkeypatch, missing_module):
    stub_protocols(monkeypatch)
    monkeypatch.delitem(sys.modules, missing_module, raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing_module or name.startswith(missing_module + "."):
            raise ImportError(f"No module named {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location("turbulent_plume", MODULE_PATH)
    with pytest.raises(ImportError):
        module = importlib.util.module_from_spec(spec)
        sys.modules["turbulent_plume"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
