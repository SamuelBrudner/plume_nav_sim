import builtins
import importlib
import sys

import pytest


def test_protocol_import_requires_hydra(monkeypatch):
    """Protocols module should raise ImportError when Hydra (omegaconf) is missing."""

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "omegaconf" or name.startswith("omegaconf."):
            raise ImportError("No module named 'omegaconf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "odor_plume_nav.core.protocols", raising=False)

    with pytest.raises(ImportError):
        importlib.import_module("odor_plume_nav.core.protocols")
