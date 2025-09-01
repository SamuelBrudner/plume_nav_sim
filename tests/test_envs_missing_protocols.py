import builtins
import importlib
import sys

import pytest


def test_envs_import_raises_when_protocols_missing(monkeypatch):
    """Env package should fail to import when core protocols are missing."""
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "plume_nav_sim.core.protocols":
            raise ImportError("protocols missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "plume_nav_sim.envs", raising=False)

    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.envs")
