import importlib
import os
import sys

import pytest

os.environ.setdefault("PLUME_NAV_SIM_SKIP_INSTALL_CHECK", "1")

import plume_nav_sim.models as models


def test_auto_discover_models_raises_on_failed_import(monkeypatch):
    """auto_discover_models should raise when a module import fails."""
    sys.modules.pop('plume_nav_sim.models.plume.gaussian_plume', None)
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name.endswith('.plume.gaussian_plume'):
            raise ImportError("boom")
        return real_import_module(name, package)

    monkeypatch.setattr(models.importlib, 'import_module', fake_import_module)

    with pytest.raises(ImportError):
        models.auto_discover_models()
