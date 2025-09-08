import importlib
import builtins
import os
import sys
import pytest

def test_protocols_requires_hydra(monkeypatch):
    os.environ["PLUME_NAV_SIM_SKIP_INSTALL_CHECK"] = "1"
    sys.modules.pop('plume_nav_sim.core.protocols', None)
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith('omegaconf'):
            raise ImportError('missing omegaconf')
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(ImportError):
        importlib.import_module('plume_nav_sim.core.protocols')

def test_protocols_requires_gymnasium(monkeypatch):
    os.environ["PLUME_NAV_SIM_SKIP_INSTALL_CHECK"] = "1"
    sys.modules.pop('plume_nav_sim.core.protocols', None)
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ('gymnasium', 'gym'):
            raise ImportError('missing gymnasium')
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(ImportError):
        importlib.import_module('plume_nav_sim.core.protocols')
