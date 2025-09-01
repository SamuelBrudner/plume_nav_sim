import builtins
import importlib
import sys

import pytest


@pytest.mark.parametrize("missing_module", [
    "gymnasium",
    "odor_plume_nav.environments.compat",
])
def test_import_error_when_dependency_missing(monkeypatch, missing_module):
    """Expect ImportError when required RL dependencies are missing."""
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing_module:
            raise ImportError(f"No module named {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("odor_plume_nav.environments", None)

    with pytest.raises(ImportError):
        importlib.import_module("odor_plume_nav.environments")
