import builtins
import pytest


def test_sensor_config_import_and_missing(monkeypatch):
    # Import should succeed when module is present
    from plume_nav_sim.config.schemas import SensorConfig  # noqa: F401

    original_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == "plume_nav_sim.config.schemas":
            raise ImportError("Mocked missing module")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)

    with pytest.raises(ImportError):
        from plume_nav_sim.config.schemas import SensorConfig  # noqa: F401
