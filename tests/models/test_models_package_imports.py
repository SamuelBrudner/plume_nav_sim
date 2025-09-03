import builtins
import importlib
import sys

import pytest


def _reload_models(monkeypatch):
    """Helper to reload plume_nav_sim.models after clearing it from sys.modules."""
    monkeypatch.delitem(sys.modules, "plume_nav_sim.models", raising=False)
    return importlib.import_module("plume_nav_sim.models")


@pytest.fixture(autouse=True)
def restore_imports(monkeypatch):
    """Ensure patched modules are restored after each test."""
    yield
    # Reload actual modules to reset state
    importlib.reload(importlib.import_module("plume_nav_sim.models"))


def test_import_models_requires_hydra(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("hydra") or name.startswith("omegaconf"):
            raise ImportError("hydra missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        _reload_models(monkeypatch)


def test_import_models_requires_loguru(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("loguru"):
            raise ImportError("loguru missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        _reload_models(monkeypatch)
