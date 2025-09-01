"""Tests for explicit guard functions in plume_nav_sim.utils."""
import importlib
import plume_nav_sim.utils as utils
import pytest

def test_require_frame_cache(monkeypatch):
    monkeypatch.setattr(utils, "CACHE_AVAILABLE", False)
    with pytest.raises(ImportError):
        utils.require_frame_cache()

def test_require_logging(monkeypatch):
    monkeypatch.setattr(utils, "LOGGING_AVAILABLE", False)
    with pytest.raises(ImportError):
        utils.require_logging()

def test_require_visualization(monkeypatch):
    monkeypatch.setattr(utils, "VISUALIZATION_AVAILABLE", False)
    with pytest.raises(ImportError):
        utils.require_visualization()

def test_require_seed_manager(monkeypatch):
    monkeypatch.setattr(utils, "SEED_MANAGER_AVAILABLE", False)
    with pytest.raises(ImportError):
        utils.require_seed_manager()

def test_require_navigator_utils(monkeypatch):
    monkeypatch.setattr(utils, "NAVIGATOR_UTILS_AVAILABLE", False)
    with pytest.raises(ImportError):
        utils.require_navigator_utils()
