import importlib
import sys
import types
import builtins
import pytest

def test_missing_spaces_module_raises_import_error(monkeypatch):
    module_name = "plume_nav_sim.envs.plume_navigation_env"
    sys.modules.pop(module_name, None)

    # Mock hydra to avoid configuration side effects
    hydra_module = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")
    hydra_utils.instantiate = lambda cfg, *a, **kw: cfg
    sys.modules["hydra"] = hydra_module
    sys.modules["hydra.utils"] = hydra_utils

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "plume_nav_sim.envs.spaces":
            raise ImportError("spaces module missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        importlib.import_module(module_name)
