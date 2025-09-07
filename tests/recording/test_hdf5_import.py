"""Tests for HDF5Recorder dependency handling."""

import builtins
import importlib.util
import importlib.metadata
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / 'src' / 'plume_nav_sim' / 'recording' / 'backends' / 'hdf5.py'
MODULE_NAME = 'plume_nav_sim.recording.backends.hdf5_test'


class _DummyDistribution:
    version = "0.0"


importlib.metadata.distribution = lambda name: _DummyDistribution()


@dataclass
class RecorderConfig:
    backend: str
    output_dir: str


class BaseRecorder:
    def __init__(self, config):
        self.config = config


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
    spec.loader.exec_module(module)  # type: ignore
    return module


def test_hdf5_import_requires_h5py(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'h5py':
            raise ImportError('No module named h5py')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    sys.modules.pop(MODULE_NAME, None)
    with pytest.raises(ImportError):
        load_module()
