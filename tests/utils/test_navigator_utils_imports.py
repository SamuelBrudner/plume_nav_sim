import importlib
import builtins
from loguru import logger
import sys
from pathlib import Path
import types

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


def test_import_fails_without_hydra(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {'omegaconf', 'hydra.core.hydra_config'}:
            raise ImportError('Hydra missing')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    monkeypatch.delitem(sys.modules, MODULE_PATH, raising=False)
    monkeypatch.delitem(sys.modules, 'omegaconf', raising=False)
    monkeypatch.delitem(sys.modules, 'hydra.core.hydra_config', raising=False)

    with pytest.raises(ImportError):
        importlib.import_module(MODULE_PATH)


def test_import_fails_without_logging_utils(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'plume_nav_sim.utils.logging_setup':
            raise ImportError('Logging setup missing')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    monkeypatch.delitem(sys.modules, MODULE_PATH, raising=False)
    monkeypatch.delitem(sys.modules, 'plume_nav_sim.utils.logging_setup', raising=False)

    with pytest.raises(ImportError):
        importlib.import_module(MODULE_PATH)


def test_logs_successful_dependency_initialization(monkeypatch, caplog):
    package_path = Path(__file__).resolve().parents[2] / 'src' / 'plume_nav_sim'

    fake_pkg = types.ModuleType('plume_nav_sim')
    fake_pkg.__path__ = [str(package_path)]
    monkeypatch.setitem(sys.modules, 'plume_nav_sim', fake_pkg)

    fake_utils_pkg = types.ModuleType('plume_nav_sim.utils')
    fake_utils_pkg.__path__ = [str(package_path / 'utils')]
    monkeypatch.setitem(sys.modules, 'plume_nav_sim.utils', fake_utils_pkg)

    monkeypatch.setitem(
        sys.modules,
        'plume_nav_sim.protocols.navigator',
        types.SimpleNamespace(NavigatorProtocol=object),
    )

    monkeypatch.setitem(
        sys.modules,
        'plume_nav_sim.core.navigator',
        types.SimpleNamespace(Navigator=object),
    )

    monkeypatch.setitem(
        sys.modules,
        'plume_nav_sim.config.models',
        types.SimpleNamespace(
            NavigatorConfig=object,
            SingleAgentConfig=object,
            MultiAgentConfig=object,
        ),
    )

    def get_enhanced_logger(name):
        return logger

    monkeypatch.setitem(
        sys.modules,
        'plume_nav_sim.utils.logging_setup',
        types.SimpleNamespace(get_enhanced_logger=get_enhanced_logger),
    )

    monkeypatch.setitem(
        sys.modules,
        'plume_nav_sim.utils.seed_manager',
        types.SimpleNamespace(
            SeedManager=object,
            set_global_seed=lambda *a, **k: None,
            get_global_seed_manager=lambda: None,
        ),
    )

    module_file = package_path / 'utils' / 'navigator_utils.py'
    top_lines = ''.join(module_file.read_text().splitlines(keepends=True)[:170])
    with caplog.at_level(logger.INFO):
        exec(compile(top_lines, str(module_file), 'exec'), {})

    assert 'Navigator utilities dependencies initialized' in caplog.text
