import importlib
import builtins
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = 'plume_nav_sim.utils.navigator_utils'


def test_import_fails_without_navigator(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'plume_nav_sim.core.navigator':
            raise ImportError('Navigator missing')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    monkeypatch.delitem(sys.modules, MODULE_PATH, raising=False)
    monkeypatch.delitem(sys.modules, 'plume_nav_sim.core.navigator', raising=False)

    with pytest.raises(ImportError):
        importlib.import_module(MODULE_PATH)


def test_import_fails_without_config_models(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'plume_nav_sim.config.models':
            raise ImportError('Config models missing')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    monkeypatch.delitem(sys.modules, MODULE_PATH, raising=False)
    monkeypatch.delitem(sys.modules, 'plume_nav_sim.config.models', raising=False)

    with pytest.raises(ImportError):
        importlib.import_module(MODULE_PATH)


def test_availability_flags_removed(monkeypatch):
    package_path = Path('src/plume_nav_sim')

    pkg = types.ModuleType('plume_nav_sim')
    pkg.__path__ = [str(package_path)]
    utils_pkg = types.ModuleType('plume_nav_sim.utils')
    utils_pkg.__path__ = [str(package_path / 'utils')]

    dummy_modules = {
        'plume_nav_sim': pkg,
        'plume_nav_sim.utils': utils_pkg,
        'plume_nav_sim.protocols.navigator': types.ModuleType('plume_nav_sim.protocols.navigator'),
        'plume_nav_sim.core.navigator': types.ModuleType('plume_nav_sim.core.navigator'),
        'plume_nav_sim.config.models': types.ModuleType('plume_nav_sim.config.models'),
        'plume_nav_sim.utils.seed_manager': types.ModuleType('plume_nav_sim.utils.seed_manager'),
        'plume_nav_sim.utils.logging_setup': types.ModuleType('plume_nav_sim.utils.logging_setup'),
        'omegaconf': types.ModuleType('omegaconf'),
        'hydra': types.ModuleType('hydra'),
        'hydra.core': types.ModuleType('hydra.core'),
        'hydra.core.hydra_config': types.ModuleType('hydra.core.hydra_config'),
    }

    dummy_modules['plume_nav_sim.protocols.navigator'].NavigatorProtocol = object
    dummy_modules['plume_nav_sim.core.navigator'].Navigator = object
    dummy_modules['plume_nav_sim.config.models'].NavigatorConfig = object
    dummy_modules['plume_nav_sim.config.models'].SingleAgentConfig = object
    dummy_modules['plume_nav_sim.config.models'].MultiAgentConfig = object
    dummy_modules['plume_nav_sim.utils.seed_manager'].SeedManager = object
    dummy_modules['plume_nav_sim.utils.seed_manager'].set_global_seed = lambda *a, **k: None
    dummy_modules['plume_nav_sim.utils.seed_manager'].get_global_seed_manager = lambda: None
    dummy_modules['plume_nav_sim.utils.logging_setup'].get_enhanced_logger = lambda name: None
    dummy_modules['omegaconf'].DictConfig = dict
    dummy_modules['omegaconf'].OmegaConf = object
    dummy_modules['hydra.core.hydra_config'].HydraConfig = object

    for name, mod in dummy_modules.items():
        monkeypatch.setitem(sys.modules, name, mod)

    module = importlib.import_module(MODULE_PATH)
    assert not hasattr(module, 'HYDRA_AVAILABLE')
    assert not hasattr(module, 'SEED_MANAGER_AVAILABLE')
    assert not hasattr(module, 'ENHANCED_LOGGING_AVAILABLE')
