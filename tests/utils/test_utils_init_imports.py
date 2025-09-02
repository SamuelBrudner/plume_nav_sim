import builtins
import importlib
import sys
import types

import pytest

CORE_SUBMODULES = [
    "frame_cache",
    "logging_setup",
    "seed_manager",
    "visualization",
    "io",
    "navigator_utils",
]


@pytest.mark.parametrize("missing", CORE_SUBMODULES)
def test_utils_import_raises_on_missing_submodule(monkeypatch, missing):
    dummy_hydra = types.ModuleType("hydra")
    dummy_hydra.initialize_config_store = lambda *a, **k: None

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "hydra":
            return dummy_hydra
        if name == f"plume_nav_sim.utils.{missing}":
            raise ImportError(f"No module named '{missing}'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("plume_nav_sim.utils", None)
    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.utils")
