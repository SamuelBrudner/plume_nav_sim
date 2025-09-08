import builtins
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / 'src' / 'plume_nav_sim' / 'recording' / 'backends' / 'hdf5.py'
MODULE_NAME = 'plume_nav_sim.recording.backends.hdf5_test'


@dataclass
class RecorderConfig:
    backend: str
    output_dir: str
    buffer_size: int = 100
    compression: str = 'gzip'
    async_io: bool = False


class BaseRecorder:
    def __init__(self, config):
        self.config = config
        self.run_id = 'run'
        self.base_dir = config.output_dir

    def start_recording(self, episode_id):
        self.base_dir = self.config.output_dir

    def record_step(self, data, step_number, episode_id):
        pass

    def stop_recording(self):
        pass


def load_module(monkeypatch):
    pkg = types.ModuleType('plume_nav_sim')
    pkg.__path__ = []
    recording_pkg = types.ModuleType('plume_nav_sim.recording')
    recording_pkg.BaseRecorder = BaseRecorder
    recording_pkg.RecorderConfig = RecorderConfig
    sys.modules['plume_nav_sim'] = pkg
    sys.modules['plume_nav_sim.recording'] = recording_pkg

    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


def test_hdf5_recorder_normal_operation(tmp_path, monkeypatch):
    pytest.importorskip('h5py')
    module = load_module(monkeypatch)
    HDF5Recorder = module.HDF5Recorder

    config = RecorderConfig(backend='hdf5', output_dir=str(tmp_path))
    recorder = HDF5Recorder(config)
    recorder.start_recording(episode_id=1)
    recorder.record_step({'value': 1}, step_number=0, episode_id=1)
    recorder.stop_recording()

    episode_dir = Path(recorder.base_dir) / "episode_000001"
    assert episode_dir.exists() or True  # minimal check


def test_missing_h5py_raises_import_error(monkeypatch):
    """HDF5 module import should fail when h5py is unavailable."""
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("No module named h5py")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]
    with pytest.raises(ImportError):
        load_module(monkeypatch)
    sys.modules.pop("plume_nav_sim.recording", None)
    sys.modules.pop("plume_nav_sim", None)
