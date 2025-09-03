import builtins
import importlib
import sys
import types
from pathlib import Path

import pytest

# Ensure src directory on path
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
PACKAGE_ROOT = SRC_ROOT / "plume_nav_sim"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _stub_package(monkeypatch):
    """Create lightweight stubs for package hierarchy to avoid heavy imports."""
    root_pkg = types.ModuleType("plume_nav_sim")
    root_pkg.__path__ = [str(PACKAGE_ROOT)]
    monkeypatch.setitem(sys.modules, "plume_nav_sim", root_pkg)

    recording_pkg = types.ModuleType("plume_nav_sim.recording")
    recording_pkg.__path__ = [str(PACKAGE_ROOT / "recording")]
    recording_pkg.BaseRecorder = type("BaseRecorder", (), {})
    monkeypatch.setitem(sys.modules, "plume_nav_sim.recording", recording_pkg)


def _reload_parquet(monkeypatch, missing: str):
    _stub_package(monkeypatch)
    monkeypatch.delitem(sys.modules, "plume_nav_sim.recording.backends.parquet", raising=False)
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith(missing):
            raise ImportError(f"No module named {missing}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.recording.backends.parquet")


def test_missing_pyarrow_raises(monkeypatch):
    _reload_parquet(monkeypatch, "pyarrow")


def test_missing_pandas_raises(monkeypatch):
    _reload_parquet(monkeypatch, "pandas")


def test_initializes_when_dependencies_present(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    _stub_package(monkeypatch)
    module = importlib.import_module("plume_nav_sim.recording.backends.parquet")
    config = module.ParquetConfig(file_path=tmp_path / "test.parquet")
    recorder = module.ParquetRecorder(config)
    assert recorder.parquet_config.file_path == tmp_path / "test.parquet"
