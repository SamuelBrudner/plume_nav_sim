import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from plume_nav_sim.data.video_plume import VideoPlume


def test_logs_when_kernel_smoothing_disabled(caplog):
    video_path = Path(__file__).resolve().parent.parent / "test_video.mp4"
    with patch.object(VideoPlume, "_init_video_capture", return_value=None):
        with caplog.at_level(logging.INFO):
            VideoPlume(video_path=video_path, kernel_size=0)
    assert "Kernel smoothing disabled" in caplog.text
