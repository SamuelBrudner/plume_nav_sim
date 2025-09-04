import builtins
import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "module_path",
    [
        "plume_nav_sim.utils.visualization",
        "odor_plume_nav.utils.visualization",
    ],
)
def test_visualization_requires_dataclasses(monkeypatch, module_path):
    """Importing visualization modules should require dataclasses."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dataclasses":
            raise ImportError("mock missing dataclasses")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "dataclasses", raising=False)

    if module_path in sys.modules:
        del sys.modules[module_path]

    with pytest.raises(ImportError):
        importlib.import_module(module_path)
