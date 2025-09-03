import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

import pytest

# Ensure src directory on path
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
PACKAGE_ROOT = SRC_ROOT / "plume_nav_sim"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_backend_registry_import_error_on_missing_dependency(monkeypatch):
    # Provide lightweight package stubs to avoid heavy top-level imports
    root_pkg = types.ModuleType("plume_nav_sim")
    root_pkg.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["plume_nav_sim"] = root_pkg

    recording_pkg = types.ModuleType("plume_nav_sim.recording")
    recording_pkg.__path__ = [str(PACKAGE_ROOT / "recording")]
    recording_pkg.BaseRecorder = type("BaseRecorder", (), {})
    sys.modules["plume_nav_sim.recording"] = recording_pkg

    original_find_spec = importlib.machinery.PathFinder.find_spec

    def fake_find_spec(fullname, path=None, target=None):
        missing = {
            "plume_nav_sim.recording.backends.parquet",
            "plume_nav_sim.recording.backends.hdf5",
            "plume_nav_sim.recording.backends.sqlite",
        }
        if fullname in missing:
            return None
        return original_find_spec(fullname, path=path, target=target)

    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec", staticmethod(fake_find_spec))
    sys.modules.pop("plume_nav_sim.recording.backends", None)

    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.recording.backends")
