import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / 'src' / 'plume_nav_sim' / 'recording' / 'backends' / 'sqlite.py'
MODULE_NAME = 'plume_nav_sim.recording.backends.sqlite_test'


@dataclass
class RecorderConfig:
    backend: str
    output_dir: str
    buffer_size: int = 100
    compression: str = 'none'
    async_io: bool = False
    database_path: str = 'recording.db'

    def __post_init__(self):
        pass


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


def load_module():
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
    spec.loader.exec_module(module)
    return module


def test_sqlite_serializer_unsupported_type(tmp_path):
    module = load_module()
    SQLiteRecorder = module.SQLiteRecorder
    SQLiteConfig = module.SQLiteConfig
    config = SQLiteConfig(backend='sqlite', output_dir=str(tmp_path), database_path=str(tmp_path / 'test.db'))
    recorder = SQLiteRecorder(config)

    with pytest.raises(TypeError):
        recorder._json_serializer(object())
