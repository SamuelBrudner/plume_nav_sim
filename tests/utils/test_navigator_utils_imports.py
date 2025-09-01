import importlib
import builtins
import sys

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
