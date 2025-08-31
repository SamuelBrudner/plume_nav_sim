import pytest

from plume_nav_sim.config import VideoPlumeConfig


def test_kernel_size_zero_allows_disable_smoothing():
    config = VideoPlumeConfig(
        video_path="test.mp4",
        kernel_size=0,
        kernel_sigma=1.0,
        _skip_validation=True,
    )
    assert config.kernel_size == 0

@pytest.mark.parametrize("value, message", [(-1, "kernel_size must be positive"), (2, "kernel_size must be odd")])
def test_kernel_size_invalid(value, message):
    with pytest.raises(ValueError, match=message):
        VideoPlumeConfig(
            video_path="test.mp4",
            kernel_size=value,
            kernel_sigma=1.0,
            _skip_validation=True,
        )
