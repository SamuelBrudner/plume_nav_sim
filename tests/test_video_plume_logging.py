import logging
from unittest.mock import MagicMock, patch

import numpy as np
import cv2
import pytest

from plume_nav_sim.data.video_plume import VideoPlume


class DummyCapture:
    def __init__(self):
        self.position = 0
        self.opened = True
        self.props = {
            cv2.CAP_PROP_FRAME_WIDTH: 10,
            cv2.CAP_PROP_FRAME_HEIGHT: 8,
            cv2.CAP_PROP_FPS: 20.0,
            cv2.CAP_PROP_FRAME_COUNT: 30,
        }

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.props.get(prop, 0)

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = value

    def read(self):
        frame = np.full((8, 10, 3), self.position, dtype=np.uint8)
        return True, frame

    def release(self):
        self.opened = False


def test_logs_when_kernel_smoothing_disabled(mock_exists, caplog):
    dummy = DummyCapture()
    with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture', return_value=dummy):
        with caplog.at_level(logging.INFO):
            VideoPlume(video_path="video.mp4", kernel_size=0)
    assert "Kernel smoothing disabled" in caplog.text


def test_binds_context_on_init(mock_exists):
    dummy = DummyCapture()
    with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture', return_value=dummy):
        with patch('plume_nav_sim.data.video_plume.get_correlation_context') as mock_ctx:
            ctx = MagicMock()
            mock_ctx.return_value = ctx
            VideoPlume(video_path="video.mp4")
            assert ctx.bind_context.called


def test_binds_context_in_workflow_metadata(tmp_path, mock_exists):
    dummy = DummyCapture()
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"data")
    with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture', return_value=dummy):
        with patch('plume_nav_sim.data.video_plume.get_correlation_context') as mock_ctx:
            ctx = MagicMock()
            mock_ctx.return_value = ctx
            plume = VideoPlume(video_path=video_file)
            ctx.bind_context.reset_mock()
            plume.get_workflow_metadata()
            assert ctx.bind_context.called
