import builtins
import importlib
import sys

import pytest

MODULE = "plume_nav_sim.core.controllers"


def expect_import_error(monkeypatch, missing_name: str) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing_name:
            raise ImportError(f"mock missing {missing_name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop(MODULE, None)
    with pytest.raises(ImportError):
        importlib.import_module(MODULE)


def test_missing_hydra(monkeypatch):
    expect_import_error(monkeypatch, "hydra.core.hydra_config")


def test_missing_gymnasium(monkeypatch):
    expect_import_error(monkeypatch, "gymnasium")


def test_missing_frame_cache(monkeypatch):
    expect_import_error(monkeypatch, "plume_nav_sim.utils.frame_cache")


def test_missing_spaces_factory(monkeypatch):
    expect_import_error(monkeypatch, "plume_nav_sim.envs.spaces")
