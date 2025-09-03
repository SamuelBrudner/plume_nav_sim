import importlib
import sys
import types
import builtins
import pytest

MISSING_MODULES = [
    "plume_nav_sim.envs.spaces",
    "plume_nav_sim.utils.seed_utils",
    "plume_nav_sim.hooks",
    "plume_nav_sim.utils.logging_setup",
    "omegaconf",
    "hydra.utils",
    "plume_nav_sim.core.sources",
    "plume_nav_sim.core.initialization",
    "plume_nav_sim.core.boundaries",
    "plume_nav_sim.core.actions",
    "plume_nav_sim.recording",
    "plume_nav_sim.analysis",
    "matplotlib",
]


@pytest.mark.parametrize("missing", MISSING_MODULES)
def test_missing_dependency_raises_import_error(monkeypatch, missing):
    module_name = "plume_nav_sim.envs.plume_navigation_env"
    sys.modules.pop(module_name, None)

    if missing not in {"omegaconf", "hydra.utils"}:
        hydra_module = types.ModuleType("hydra")
        hydra_utils = types.ModuleType("hydra.utils")
        hydra_utils.instantiate = lambda cfg, *a, **kw: cfg
        sys.modules["hydra"] = hydra_module
        sys.modules["hydra.utils"] = hydra_utils
        omegaconf_module = types.ModuleType("omegaconf")
        class DictConfig(dict):
            pass
        class OmegaConf:
            @staticmethod
            def to_container(config, resolve=True):
                return dict(config)
        omegaconf_module.DictConfig = DictConfig
        omegaconf_module.OmegaConf = OmegaConf
        sys.modules["omegaconf"] = omegaconf_module

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing:
            raise ImportError(f"{missing} missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        importlib.import_module(module_name)
