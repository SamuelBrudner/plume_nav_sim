import builtins
import importlib.util
import sys
import types
from pathlib import Path
import pytest


def test_import_error_when_logging_setup_missing(monkeypatch):
    """Frame cache should fail to import if logging helpers are unavailable."""
    real_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "plume_nav_sim.utils.logging_setup":
            raise ImportError("mocked missing logging_setup")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Create lightweight package placeholders to avoid importing full package
    pkg = types.ModuleType("plume_nav_sim")
    pkg.__path__ = []
    utils_pkg = types.ModuleType("plume_nav_sim.utils")
    utils_pkg.__path__ = []
    sys.modules["plume_nav_sim"] = pkg
    sys.modules["plume_nav_sim.utils"] = utils_pkg

    module_path = Path("src/plume_nav_sim/utils/frame_cache.py")
    spec = importlib.util.spec_from_file_location(
        "plume_nav_sim.utils.frame_cache", module_path
    )

    with pytest.raises(ImportError):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
