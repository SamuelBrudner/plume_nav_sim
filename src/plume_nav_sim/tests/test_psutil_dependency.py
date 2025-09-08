import importlib
import importlib.metadata
import psutil
import pytest


def test_frame_cache_requires_psutil_process(monkeypatch):
    """Ensure FrameCache fails initialization when psutil.Process is unavailable."""
    real_distribution = importlib.metadata.distribution

    class FakeDist:
        def locate_file(self, path):
            return path

    def fake_distribution(name):
        if name == "plume_nav_sim":
            return FakeDist()
        return real_distribution(name)

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)

    from plume_nav_sim.utils.frame_cache import FrameCache, CacheMode

    def boom(*args, **kwargs):  # pragma: no cover - test shim
        raise psutil.Error("psutil unavailable")

    monkeypatch.setattr(psutil, "Process", boom)

    with pytest.raises(psutil.Error):
        FrameCache(mode=CacheMode.LRU)
