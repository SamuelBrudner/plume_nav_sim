import importlib
import builtins
import sys

import pytest

MODULE = "plume_nav_sim.models.plume"


@pytest.mark.parametrize(
    "missing",
    [
        "plume_nav_sim.models.plume.gaussian_plume",
        "plume_nav_sim.models.plume.turbulent_plume",
        "plume_nav_sim.models.plume.video_plume_adapter",
    ],
)
def test_import_error_on_missing_dependency(monkeypatch, missing):
    import types

    # Stub hydra module to satisfy package imports
    hydra_stub = types.ModuleType("hydra")
    hydra_stub.compose = lambda *args, **kwargs: None
    hydra_stub.initialize_config_store = lambda *args, **kwargs: None
    sys.modules.setdefault("hydra", hydra_stub)

    sys.modules.pop(MODULE, None)
    sys.modules.pop(missing, None)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing:
            raise ImportError(f"No module named {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        importlib.import_module(MODULE)
