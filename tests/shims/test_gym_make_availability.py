import builtins
import importlib
import sys

import pytest


def test_missing_gym_make_raises_import_error(monkeypatch):
    """Importing plume_nav_sim.shims should fail when gym_make is unavailable."""
    sys.modules.pop("plume_nav_sim.shims", None)
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "plume_nav_sim.shims.gym_make":
            raise ImportError("mocked missing gym_make")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.shims")
