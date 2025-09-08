import pytest
from plume_nav_sim.recording import RecorderManager, RecorderFactory


def test_recorder_manager_requires_recorder():
    manager = RecorderManager()
    with pytest.raises(RuntimeError):
        manager.start_recording()


def test_validate_unknown_target_errors():
    with pytest.raises(ValueError):
        RecorderFactory.validate_config({'_target_': 'plume_nav_sim.recording.backends.UnknownRecorder'})
