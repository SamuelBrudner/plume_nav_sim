import importlib
import builtins
import sys
import pytest

MISSING_MODULES = [
    "hydra",
    "plume_nav_sim.envs.video_plume",
    "plume_nav_sim.utils.logging_setup",
    "gymnasium",
]

@pytest.mark.parametrize("missing", MISSING_MODULES)
def test_import_error_when_dependency_missing(monkeypatch, missing):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == missing and globals and globals.get("__name__", "").startswith("plume_nav_sim"):
            raise ImportError(f"{missing} missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "plume_nav_sim.envs", raising=False)

    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.envs")
