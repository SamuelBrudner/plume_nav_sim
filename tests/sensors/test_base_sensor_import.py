import importlib
import importlib.machinery
import sys
import pytest


def test_base_sensor_import(monkeypatch):
    """BaseSensor import should fail loudly when the module is missing."""
    # Import succeeds when module is available
    module = importlib.import_module('plume_nav_sim.core.sensors.base_sensor')
    assert module is not None

    real_find_spec = importlib.machinery.PathFinder.find_spec

    def fake_find_spec(name, path=None, target=None):
        if name == 'plume_nav_sim.core.sensors.base_sensor':
            return None
        return real_find_spec(name, path, target)

    # Remove access to base_sensor
    monkeypatch.setattr(importlib.machinery.PathFinder, 'find_spec', fake_find_spec)
    monkeypatch.delitem(sys.modules, 'plume_nav_sim.core.sensors.base_sensor', raising=False)
    monkeypatch.delitem(sys.modules, 'plume_nav_sim.core.sensors.concentration_sensor', raising=False)

    # Direct import now fails
    with pytest.raises(ImportError):
        importlib.import_module('plume_nav_sim.core.sensors.base_sensor')

    # Dependent modules should also fail to import
    with pytest.raises(ImportError):
        importlib.import_module('plume_nav_sim.core.sensors.concentration_sensor')
