import importlib
import builtins
import sys

import pytest


def test_missing_numba_raises_import_error(monkeypatch):
    """Importing turbulent_wind without numba should raise ImportError."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("numba"):
            raise ImportError("No module named 'numba'")
        return real_import(name, *args, **kwargs)

    for mod in list(sys.modules):
        if mod.startswith("numba"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.delitem(sys.modules, "plume_nav_sim.models.wind.turbulent_wind", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.models.wind.turbulent_wind")


def test_missing_hydra_raises_import_error(monkeypatch):
    """Importing turbulent_wind without hydra should raise ImportError."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("hydra"):
            raise ImportError("No module named 'hydra'")
        return real_import(name, *args, **kwargs)

    for mod in list(sys.modules):
        if mod.startswith("hydra"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.delitem(sys.modules, "plume_nav_sim.models.wind.turbulent_wind", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.models.wind.turbulent_wind")
