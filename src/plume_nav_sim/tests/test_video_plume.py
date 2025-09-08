"""Tests for VideoPlume data handling."""

from pathlib import Path
import pytest

from plume_nav_sim.api.navigation import ConfigurationError
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


def test_real_navigation_module_loaded():
    import plume_nav_sim.api.navigation as nav
    assert hasattr(nav, "create_navigator")
