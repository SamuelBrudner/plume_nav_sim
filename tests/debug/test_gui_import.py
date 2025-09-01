import importlib
import builtins
import sys

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
