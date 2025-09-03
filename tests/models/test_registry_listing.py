import importlib.util
from pathlib import Path
from unittest.mock import patch

# Load models module directly to avoid package import side effects
models_path = Path(__file__).resolve().parents[2] / "src/plume_nav_sim/models/__init__.py"
spec = importlib.util.spec_from_file_location("plume_nav_sim.models", models_path)
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)


def test_list_available_plume_models_no_fallback_and_logging(monkeypatch):
    monkeypatch.setattr(models, "_PLUME_MODEL_REGISTRY", {"FooModel": {"class": object}})
    with patch.object(models.logger, "debug") as mock_debug:
        assert models.list_available_plume_models() == ["FooModel"]
        mock_debug.assert_called_once()
        args, kwargs = mock_debug.call_args
        assert args[0] == "Available plume models: {}"
        assert args[1] == ["FooModel"]


def test_list_available_wind_fields_no_fallback_and_logging(monkeypatch):
    monkeypatch.setattr(models, "_WIND_FIELD_REGISTRY", {"FooField": {"class": object}})
    with patch.object(models.logger, "debug") as mock_debug:
        assert models.list_available_wind_fields() == ["FooField"]
        mock_debug.assert_called_once()
        args, kwargs = mock_debug.call_args
        assert args[0] == "Available wind fields: {}"
        assert args[1] == ["FooField"]


def test_list_available_sensors_no_fallback_and_logging(monkeypatch):
    monkeypatch.setattr(models, "_SENSOR_REGISTRY", {"FooSensor": {"class": object}})
    with patch.object(models.logger, "debug") as mock_debug:
        assert models.list_available_sensors() == ["FooSensor"]
        mock_debug.assert_called_once()
        args, kwargs = mock_debug.call_args
        assert args[0] == "Available sensors: {}"
        assert args[1] == ["FooSensor"]
