import importlib
import sys
import builtins

import pytest


def _assert_import_error(monkeypatch, missing_module: str) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith(missing_module):
            raise ModuleNotFoundError(f"No module named '{missing_module}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("odor_plume_nav.config", None)

    with pytest.raises(ImportError):
        importlib.import_module("odor_plume_nav.config")


def test_missing_hydra(monkeypatch):
    _assert_import_error(monkeypatch, "hydra")


def test_missing_dotenv(monkeypatch):
    _assert_import_error(monkeypatch, "dotenv")
