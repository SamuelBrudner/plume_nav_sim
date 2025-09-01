import pytest
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import numpy as np
cv2 = pytest.importorskip("cv2")

from odor_plume_nav.data.video_plume import VideoPlume


class DummyLogger:
    def __init__(self):
        self.timer_called_with = None
        self.metrics = None

    @contextmanager
    def performance_timer(self, operation: str):
        self.timer_called_with = operation
        metrics = SimpleNamespace(operation_name=operation, to_dict=lambda: {})
        self.metrics = metrics
        yield metrics

    def bind(self, **kwargs):
        return self

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


@pytest.fixture
def dummy_logger(monkeypatch):
    logger = DummyLogger()
    monkeypatch.setattr('odor_plume_nav.data.video_plume.logger', logger)
    return logger


@pytest.fixture(autouse=True)
def mock_video_capture(monkeypatch):
    class MockCapture:
        def __init__(self, path):
            self.path = path
            self.position = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 1
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 10
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 10
            if prop == cv2.CAP_PROP_FPS:
                return 30
            return 0

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                self.position = value

        def read(self):
            return True, np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr('cv2.VideoCapture', MockCapture)


def test_frame_retrieval_uses_performance_timer(dummy_logger):
    plume = VideoPlume(video_path=Path('test_video.mp4'))
    frame = plume.get_frame(0)
    assert frame is not None
    assert dummy_logger.timer_called_with == 'video_frame_processing'
    assert dummy_logger.metrics.operation_name == 'video_frame_processing'
