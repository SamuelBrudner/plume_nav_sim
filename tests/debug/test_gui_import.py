import importlib
import builtins
import sys
import types
from pathlib import Path

import pytest


def test_import_requires_pyside6(monkeypatch):
    """Debug GUI import should raise ImportError when PySide6 is unavailable."""
    monkeypatch.delitem(sys.modules, "PySide6", raising=False)
    monkeypatch.delitem(sys.modules, "plume_nav_sim.debug.gui", raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("PySide6"):
            raise ImportError("PySide6 not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.debug.gui")


def test_import_requires_logging_setup(monkeypatch):
    """Debug GUI import should fail without logging setup module."""
    monkeypatch.delitem(sys.modules, "plume_nav_sim.utils.logging_setup", raising=False)
    monkeypatch.delitem(sys.modules, "plume_nav_sim.debug.gui", raising=False)

    root = Path(__file__).resolve().parents[2] / "src"
    plume_pkg = types.ModuleType("plume_nav_sim")
    plume_pkg.__path__ = [str(root / "plume_nav_sim")]
    monkeypatch.setitem(sys.modules, "plume_nav_sim", plume_pkg)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "plume_nav_sim.utils.logging_setup":
            raise ImportError("logging setup missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    pyside_stub = types.ModuleType("PySide6")
    monkeypatch.setitem(sys.modules, "PySide6", pyside_stub)
    for sub in ["QtCore", "QtWidgets", "QtGui"]:
        monkeypatch.setitem(sys.modules, f"PySide6.{sub}", types.ModuleType(f"PySide6.{sub}"))

    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.use = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_stub)
    monkeypatch.setitem(sys.modules, "matplotlib.backends.backend_qt5agg", types.ModuleType("backend_qt5agg"))
    monkeypatch.setitem(sys.modules, "matplotlib.figure", types.ModuleType("figure"))

    with pytest.raises(ImportError, match="logging setup"):
        importlib.import_module("plume_nav_sim.debug.gui")
