"""Tests for VideoPlume data handling."""

from pathlib import Path
import types
import sys
import pytest

# Provide stub for ConfigurationError to avoid circular import during testing
stub_api_nav = types.ModuleType("plume_nav_sim.api.navigation")
class ConfigurationError(Exception):
    pass
stub_api_nav.ConfigurationError = ConfigurationError
sys.modules.setdefault("plume_nav_sim.api.navigation", stub_api_nav)

from plume_nav_sim.data.video_plume import VideoPlume


class DummyVideoPlume(VideoPlume):
    """VideoPlume subclass that skips video capture for testing."""
    def _init_video_capture(self):  # type: ignore[override]
        self._cap = None
        self.frame_count = 1
        self.width = 1
        self.height = 1
        self.fps = 1.0


def test_get_concentration_not_implemented():
    """get_concentration should raise until implemented."""
    video_path = Path('test_video.mp4').resolve()
    vp = DummyVideoPlume(video_path=video_path)
    with pytest.raises(NotImplementedError):
        vp.get_concentration((0, 0))
