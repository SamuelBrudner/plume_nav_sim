import importlib
import builtins
import sys
import pathlib

import pytest


def test_import_requires_loguru_and_hydra(monkeypatch):
    """Importing the API without loguru or Hydra must fail."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "loguru":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="loguru.*required"):
        importlib.import_module("odor_plume_nav.api")
