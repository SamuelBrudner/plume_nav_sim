import importlib
import sys
import types
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
PACKAGE_ROOT = SRC_ROOT / "plume_nav_sim"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_backends_module():
    root_pkg = types.ModuleType("plume_nav_sim")
    root_pkg.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["plume_nav_sim"] = root_pkg

    recording_pkg = types.ModuleType("plume_nav_sim.recording")
    recording_pkg.__path__ = [str(PACKAGE_ROOT / "recording")]
    recording_pkg.BaseRecorder = type("BaseRecorder", (), {})
    sys.modules["plume_nav_sim.recording"] = recording_pkg

    # Provide lightweight backend stubs to avoid heavy dependencies
    stubs = {
        "parquet": "ParquetRecorder",
        "hdf5": "HDF5Recorder",
        "sqlite": "SQLiteRecorder",
    }
    for name, cls_name in stubs.items():
        mod = types.ModuleType(f"plume_nav_sim.recording.backends.{name}")
        recorder_cls = type(cls_name, (), {"__init__": lambda self, config: None})
        setattr(mod, cls_name, recorder_cls)
        sys.modules[f"plume_nav_sim.recording.backends.{name}"] = mod

    return importlib.import_module("plume_nav_sim.recording.backends")


def test_create_backend_invalid_name_raises_value_error(caplog):
    backends = _load_backends_module()
    config = {"backend": "does-not-exist"}
    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError):
            backends.create_backend(config)
    assert "Unknown backend" in caplog.text


class _MissingDependencyRecorder:
    def __init__(self, config):
        raise ImportError("missing dependency")


def test_create_backend_missing_dependency_raises_import_error(monkeypatch, caplog):
    backends = _load_backends_module()
    monkeypatch.setitem(backends.BACKEND_REGISTRY, "missing", _MissingDependencyRecorder)
    config = {"backend": "missing"}
    with caplog.at_level("ERROR"):
        with pytest.raises(ImportError):
            backends.create_backend(config)
    assert "missing dependency" in caplog.text
