import importlib
import builtins
import sys

import pytest

MODULE_PATH = "plume_nav_sim.recording.backends.parquet"


def _block_import(monkeypatch, package_name):
    """Remove package from sys.modules and block future imports."""
    for mod in list(sys.modules):
        if mod == package_name or mod.startswith(package_name + "."):
            sys.modules.pop(mod)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == package_name or name.startswith(package_name + "."):
            raise ImportError(f"Missing dependency: {package_name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop(MODULE_PATH, None)


def test_pyarrow_required(monkeypatch):
    _block_import(monkeypatch, "pyarrow")
    with pytest.raises(ImportError):
        importlib.import_module(MODULE_PATH)


def test_pandas_required(monkeypatch):
    _block_import(monkeypatch, "pandas")
    with pytest.raises(ImportError):
        importlib.import_module(MODULE_PATH)
