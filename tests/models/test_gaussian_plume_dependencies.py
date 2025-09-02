import importlib
import sys
import types

import pytest

hydra_stub = types.ModuleType("hydra")
hydra_stub.initialize_config_store = lambda *args, **kwargs: None
hydra_stub.compose = lambda *args, **kwargs: None
sys.modules["hydra"] = hydra_stub

import plume_nav_sim.models.plume.gaussian_plume as gaussian_plume


def _reload():
    return importlib.reload(gaussian_plume)


def test_import_error_without_scipy(monkeypatch):
    for name in list(sys.modules):
        if name.startswith("scipy"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(ImportError):
        _reload()


def test_import_error_without_hydra(monkeypatch):
    for name in list(sys.modules):
        if name.startswith("omegaconf") or name.startswith("hydra"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(ImportError):
        _reload()
