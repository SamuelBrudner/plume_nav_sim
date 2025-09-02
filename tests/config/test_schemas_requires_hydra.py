import importlib.util
import builtins
import pathlib

import pytest


def test_schemas_requires_hydra(monkeypatch):
    """Ensure config schemas import fails loudly when Hydra is missing."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("hydra"):
            raise ModuleNotFoundError("No module named 'hydra'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    schemas_path = pathlib.Path("src/plume_nav_sim/config/schemas.py")
    spec = importlib.util.spec_from_file_location("plume_nav_sim.config.schemas", schemas_path)
    with pytest.raises(ImportError):
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
