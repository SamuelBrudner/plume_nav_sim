import builtins
import importlib
import sys

import pytest


def test_missing_protocols_module_raises_import_error(monkeypatch):
    """Package import should fail when protocol module is missing."""
    sys.modules.pop("plume_nav_sim", None)
    sys.modules.pop("plume_nav_sim.core.protocols", None)

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "plume_nav_sim.core.protocols":
            raise ImportError("mock missing protocol module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim")
