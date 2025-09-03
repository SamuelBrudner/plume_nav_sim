import builtins
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _setup_stub_packages():
    root = Path("src/plume_nav_sim")
    packages = {
        "plume_nav_sim": root,
        "plume_nav_sim.core": root / "core",
        "plume_nav_sim.config": root / "config",
        "plume_nav_sim.envs": root / "envs",
    }
    for name, path in packages.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _import_protocols(monkeypatch):
    _setup_stub_packages()
    spec = importlib.util.spec_from_file_location(
        "plume_nav_sim.core.protocols",
        Path("src/plume_nav_sim/core/protocols.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def test_missing_config_schemas_raises_import_error(monkeypatch):
    sys.modules.pop("plume_nav_sim.core.protocols", None)

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "plume_nav_sim.config.schemas":
            raise ImportError("mock missing schemas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="mock missing schemas"):
        _import_protocols(monkeypatch)


def test_missing_spaces_factory_raises_import_error(monkeypatch):
    sys.modules.pop("plume_nav_sim.core.protocols", None)

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "plume_nav_sim.envs.spaces":
            raise ImportError("mock missing spaces factory")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="mock missing spaces factory"):
        _import_protocols(monkeypatch)
