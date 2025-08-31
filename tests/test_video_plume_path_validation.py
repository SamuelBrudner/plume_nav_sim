import logging
from unittest.mock import patch

import pytest

from plume_nav_sim.data.video_plume import VideoPlume


class DummyCapture:
    def __init__(self):
        self.opened = True
        self.props = {
            3: 10,  # cv2.CAP_PROP_FRAME_WIDTH placeholder
            4: 10,  # cv2.CAP_PROP_FRAME_HEIGHT placeholder
            5: 30.0,  # cv2.CAP_PROP_FPS placeholder
            7: 1,  # cv2.CAP_PROP_FRAME_COUNT placeholder
        }

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.props.get(prop, 0)

    def set(self, prop, value):
        pass

    def read(self):
        return True, None

    def release(self):
        self.opened = False


def test_resolves_path_and_within_root(tmp_path):
    allowed_root = tmp_path
    subdir = allowed_root / "sub"
    subdir.mkdir()
    video_file = subdir / "video.mp4"
    video_file.write_bytes(b"data")

    path_with_parent = subdir / ".." / "sub" / "video.mp4"
    dummy = DummyCapture()
    with patch('plume_nav_sim.data.video_plume.cv2.VideoCapture', return_value=dummy):
        plume = VideoPlume(video_path=path_with_parent, allowed_root=allowed_root)
    assert plume.video_path == video_file.resolve()


def test_invalid_suffix_logs_and_raises(tmp_path, caplog):
    allowed_root = tmp_path
    bad_file = allowed_root / "video.txt"
    bad_file.write_text("data")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="Unsupported video file extension"):
            VideoPlume(video_path=bad_file, allowed_root=allowed_root)
    assert "Unsupported video file extension" in caplog.text


def test_outside_allowed_root_logs_and_raises(tmp_path, caplog):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "video.mp4"
    outside_file.write_text("data")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="outside allowed root"):
            VideoPlume(video_path=outside_file, allowed_root=allowed_root)
    assert "outside allowed root" in caplog.text
