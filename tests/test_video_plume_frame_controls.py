import numpy as np
from unittest.mock import patch
from plume_nav_sim.data.video_plume import VideoPlume
import pytest
cv2 = pytest.importorskip("cv2")


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


def test_frame_selection_and_bounds(mock_exists):
    dummy = DummyCapture()
    with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture', return_value=dummy):
        plume = VideoPlume('video.mp4', frame_skip=2, start_frame=1, end_frame=6)
        assert plume.get_frame(0) is not None
        assert dummy.position == 1
        plume.get_frame(1)
        assert dummy.position == 3
        plume.get_frame(2)
        assert dummy.position == 5
        assert plume.get_frame(3) is None


def test_metadata_includes_frame_controls(mock_exists):
    dummy = DummyCapture()
    with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture', return_value=dummy):
        plume = VideoPlume('video.mp4', frame_skip=3, start_frame=2, end_frame=10)
        metadata = plume.get_metadata()
        assert metadata['frame_skip'] == 3
        assert metadata['start_frame'] == 2
        assert metadata['end_frame'] == 10
