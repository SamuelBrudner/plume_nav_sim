"""Tests for ParquetRecorder pandas dependency handling."""

import builtins
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def test_parquet_import_requires_pandas(monkeypatch):
    """ParquetRecorder import should fail when pandas is missing."""
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "plume_nav_sim"
        / "recording"
        / "backends"
        / "parquet.py"
    )

    # Replace package structure with minimal dummy modules
    pkg_root = types.ModuleType("plume_nav_sim")
    recording_pkg = types.ModuleType("plume_nav_sim.recording")
    backends_pkg = types.ModuleType("plume_nav_sim.recording.backends")
    class DummyBaseRecorder: ...
    class DummyRecorderConfig: ...
    recording_pkg.BaseRecorder = DummyBaseRecorder
    recording_pkg.RecorderConfig = DummyRecorderConfig
    backends_pkg.__path__ = []
    pkg_root.recording = recording_pkg
    recording_pkg.backends = backends_pkg
    monkeypatch.setitem(sys.modules, "plume_nav_sim", pkg_root)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.recording", recording_pkg)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.recording.backends", backends_pkg)

    # Provide dummy pyarrow so pandas is the only missing dependency
    dummy_pyarrow = types.ModuleType("pyarrow")
    dummy_pyarrow.Table = object
    dummy_pyarrow.Schema = object
    dummy_pyarrow.compute = types.SimpleNamespace(is_in=lambda *a, **k: None)
    dummy_pyarrow.array = lambda x: x
    dummy_pyarrow.parquet = types.ModuleType("pyarrow.parquet")
    dummy_pyarrow.dataset = types.ModuleType("pyarrow.dataset")
    monkeypatch.setitem(sys.modules, "pyarrow", dummy_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", dummy_pyarrow.parquet)
    monkeypatch.setitem(sys.modules, "pyarrow.dataset", dummy_pyarrow.dataset)

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("pandas missing")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location(
        "plume_nav_sim.recording.backends.parquet", module_path
    )
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(ImportError):
        spec.loader.exec_module(module)  # type: ignore[union-attr]
