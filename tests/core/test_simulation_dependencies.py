import importlib
import builtins
import sys

import pytest

DEPENDENCIES = [
    "gymnasium",
    "plume_nav_sim.utils.frame_cache",
    "plume_nav_sim.utils.visualization",
    "plume_nav_sim.db.session_manager",
    "psutil",
    "loguru",
]


@pytest.mark.parametrize("module_name", DEPENDENCIES)
def test_import_error_when_dependency_missing(monkeypatch, module_name):
    """Simulation module should fail to import when core dependency is missing."""
    monkeypatch.delitem(sys.modules, "plume_nav_sim.core.simulation", raising=False)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module_name:
            raise ImportError(f"No module named {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.core.simulation")
