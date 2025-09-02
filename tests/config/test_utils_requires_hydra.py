import builtins
import importlib.util
import pathlib
import sys
import types

import pytest


def test_utils_import_requires_hydra(monkeypatch):
    """Utilities module should raise ImportError when Hydra is missing."""
    package_root = pathlib.Path(__file__).resolve().parents[2] / "src/plume_nav_sim"
    pkg = types.ModuleType("plume_nav_sim")
    pkg.__path__ = [str(package_root)]
    sys.modules["plume_nav_sim"] = pkg
    config_pkg = types.ModuleType("plume_nav_sim.config")
    config_pkg.__path__ = [str(package_root / "config")]
    sys.modules["plume_nav_sim.config"] = config_pkg

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("hydra") or name.startswith("omegaconf"):
            raise ImportError("Hydra stack missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location(
        "plume_nav_sim.config.utils", package_root / "config" / "utils.py"
    )
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(ImportError):
        spec.loader.exec_module(module)
