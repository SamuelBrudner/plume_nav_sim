import importlib
import builtins
import sys
import pytest


def test_import_error_when_core_protocols_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "plume_nav_sim.core.protocols":
            raise ImportError("protocols missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "plume_nav_sim.envs.plume_navigation_env", raising=False)

    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.envs.plume_navigation_env")
