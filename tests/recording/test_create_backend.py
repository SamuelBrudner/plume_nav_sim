import importlib
import sys
import types
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
PACKAGE_ROOT = SRC_ROOT / "plume_nav_sim"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_backends_module(monkeypatch):
    root_pkg = types.ModuleType("plume_nav_sim")
    root_pkg.__path__ = [str(PACKAGE_ROOT)]
    monkeypatch.setitem(sys.modules, "plume_nav_sim", root_pkg)

    recording_pkg = types.ModuleType("plume_nav_sim.recording")
    recording_pkg.__path__ = [str(PACKAGE_ROOT / "recording")]
    recording_pkg.BaseRecorder = type("BaseRecorder", (), {})
    monkeypatch.setitem(sys.modules, "plume_nav_sim.recording", recording_pkg)

    stubs = {
        "parquet": "ParquetRecorder",
        "hdf5": "HDF5Recorder",
        "sqlite": "SQLiteRecorder",
    }
    for name, cls_name in stubs.items():
        module = types.ModuleType(f"plume_nav_sim.recording.backends.{name}")
        setattr(module, cls_name, type(cls_name, (), {"__init__": lambda self, config: None}))
        monkeypatch.setitem(sys.modules, f"plume_nav_sim.recording.backends.{name}", module)

    monkeypatch.delitem(sys.modules, "plume_nav_sim.recording.backends", raising=False)
    return importlib.import_module("plume_nav_sim.recording.backends")


def test_unknown_backend_raises_value_error(monkeypatch):
    backends = load_backends_module(monkeypatch)
    with pytest.raises(ValueError):
        backends.create_backend({'backend': 'does_not_exist'})


def test_backend_dependency_error_raises_import_error(monkeypatch):
    backends = load_backends_module(monkeypatch)

    class DummyRecorder:
        def __init__(self, config):
            raise ImportError('missing dependency')

    monkeypatch.setitem(backends.BACKEND_REGISTRY, 'dummy', DummyRecorder)
    try:
        with pytest.raises(ImportError):
            backends.create_backend({'backend': 'dummy'})
    finally:
        backends.BACKEND_REGISTRY.pop('dummy', None)
