import builtins
import importlib
import sys

import pytest


def test_import_requires_hydra(monkeypatch):
    sys.modules.pop('odor_plume_nav.data', None)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'omegaconf':
            raise ModuleNotFoundError('omegaconf missing')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    with pytest.raises(ImportError):
        importlib.import_module('odor_plume_nav.data')
