import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src"
WIND_INIT = ROOT / "plume_nav_sim" / "models" / "wind" / "__init__.py"


def test_lightweight_stub_logs_warning(monkeypatch, capsys):
    """Importing wind module with package stubs should emit a warning."""
    # Stub dependencies required during import
    omegaconf_stub = types.ModuleType("omegaconf")
    class DummyConfig: ...
    omegaconf_stub.DictConfig = DummyConfig
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf_stub)

    numpy_stub = types.ModuleType("numpy")
    numpy_typing = types.ModuleType("numpy.typing")
    numpy_typing.NDArray = object
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)
    monkeypatch.setitem(sys.modules, "numpy.typing", numpy_typing)

    numba_stub = types.ModuleType("numba")
    numba_stub.jit = lambda *a, **k: (lambda f: f)
    numba_stub.prange = range
    monkeypatch.setitem(sys.modules, "numba", numba_stub)

    scipy_stub = types.ModuleType("scipy")
    scipy_interpolate = types.ModuleType("scipy.interpolate")
    scipy_signal = types.ModuleType("scipy.signal")
    scipy_signal.periodogram = lambda *a, **k: ([], [])
    scipy_interpolate.griddata = lambda *a, **k: None
    scipy_interpolate.RBFInterpolator = object
    scipy_stub.interpolate = scipy_interpolate
    scipy_stub.signal = scipy_signal
    monkeypatch.setitem(sys.modules, "scipy", scipy_stub)
    monkeypatch.setitem(sys.modules, "scipy.interpolate", scipy_interpolate)
    monkeypatch.setitem(sys.modules, "scipy.signal", scipy_signal)

    hydra_stub = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")
    hydra_utils.instantiate = lambda cfg: cfg
    hydra_stub.utils = hydra_utils
    monkeypatch.setitem(sys.modules, "hydra", hydra_stub)
    monkeypatch.setitem(sys.modules, "hydra.utils", hydra_utils)

    pandas_stub = types.ModuleType("pandas")
    monkeypatch.setitem(sys.modules, "pandas", pandas_stub)

    # Create lightweight package hierarchy to avoid heavy __init__
    plume_pkg = types.ModuleType("plume_nav_sim")
    plume_pkg.__path__ = [str(ROOT / "plume_nav_sim")]
    models_pkg = types.ModuleType("plume_nav_sim.models")
    models_pkg.__path__ = [str(ROOT / "plume_nav_sim" / "models")]
    monkeypatch.setitem(sys.modules, "plume_nav_sim", plume_pkg)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.models", models_pkg)

    spec = importlib.util.spec_from_file_location(
        "plume_nav_sim.models.wind", WIND_INIT, submodule_search_locations=[str(WIND_INIT.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.models.wind", module)

    spec.loader.exec_module(module)

    captured = capsys.readouterr()
    assert "limited mode" in captured.err.lower()
