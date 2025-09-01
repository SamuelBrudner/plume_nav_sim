import logging
from pathlib import Path
import pytest

from plume_nav_sim.config.schemas import VideoPlumeConfig
from pydantic import BaseModel


def test_invalid_video_path_raises_and_logs(caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="Video file not found"):
            VideoPlumeConfig(video_path="nonexistent_file.mp4")
    assert "Video file not found" in caplog.text


def test_video_path_exposed_as_string():
    path = Path("test_video.mp4")
    cfg = VideoPlumeConfig(video_path=str(path))
    assert isinstance(cfg.video_path, Path)
    assert cfg.model_dump()["video_path"] == str(path)


def test_env_pattern_kept_as_string():
    pattern = "${oc.env:VAR}"
    cfg = VideoPlumeConfig(video_path=pattern)
    assert cfg.video_path == pattern
    assert cfg.model_dump()["video_path"] == pattern


def test_base_model_dump_serializes_to_string():
    path = Path("test_video.mp4")
    cfg = VideoPlumeConfig(video_path=path)
    data = BaseModel.model_dump(cfg)
    assert data["video_path"] == "test_video.mp4"


@pytest.mark.parametrize("flag_name", ["skip_validation", "_skip_validation"])
def test_skip_flags_bypass_existence_check(tmp_path, flag_name):
    """Ensure both skip flags bypass path existence checks."""
    missing = tmp_path / "missing.mp4"
    cfg = VideoPlumeConfig(video_path=missing, **{flag_name: True})
    assert cfg.video_path == missing
