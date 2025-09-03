import builtins
import importlib
import sys

import pytest

try:  # pragma: no cover - skip if package cannot import
    import plume_nav_sim  # noqa: F401
except Exception:  # pragma: no cover
    pytest.skip("plume_nav_sim import failed", allow_module_level=True)


@pytest.mark.parametrize("missing_module", [
    "plume_nav_sim.models.plume.gaussian_plume",
    "plume_nav_sim.core.sensors.binary_sensor",
])
def test_plume_navigation_env_import_error_missing_dependency(monkeypatch, missing_module):
    """PlumeNavigationEnv import should fail when dependencies are absent."""
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing_module:
            raise ImportError(f"No module named {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("plume_nav_sim.envs.plume_navigation_env", None)

    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.envs.plume_navigation_env")
