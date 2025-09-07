import importlib
import sys
import types

import pytest


def load_env_module():
    fake_numba = types.ModuleType("numba")
    fake_numba.jit = lambda *a, **k: (lambda f: f)
    fake_numba.prange = range
    sys.modules.setdefault("numba", fake_numba)
    fake_video_plume = types.ModuleType("plume_nav_sim.models.plume.video_plume")
    fake_video_plume.VideoPlume = object
    sys.modules.setdefault("plume_nav_sim.models.plume.video_plume", fake_video_plume)
    fake_video_plume_adapter = types.ModuleType("plume_nav_sim.models.plume.video_plume_adapter")
    fake_video_plume_adapter.VideoPlumeAdapter = object
    sys.modules.setdefault("plume_nav_sim.models.plume.video_plume_adapter", fake_video_plume_adapter)
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    fake_db = types.ModuleType("plume_nav_sim.db.session_manager")
    fake_db.DatabaseSessionManager = object
    sys.modules.setdefault("plume_nav_sim.db.session_manager", fake_db)
    import importlib.metadata as importlib_metadata
    class _FakeDist:
        def locate_file(self, path):
            return path
    importlib_metadata.distribution = lambda name: _FakeDist()
    fake_hydra = types.ModuleType("hydra")
    fake_hydra.instantiate = lambda *a, **k: None
    fake_hydra.compose = lambda *a, **k: None
    fake_hydra.initialize_config_dir = lambda *a, **k: None
    fake_hydra.utils = types.SimpleNamespace(instantiate=lambda *a, **k: None)
    sys.modules["hydra"] = fake_hydra
    sys.modules["hydra.utils"] = fake_hydra.utils
    fake_debug = types.ModuleType("plume_nav_sim.debug")
    fake_debug.DebugGUI = object
    fake_debug.plot_initial_state = lambda *a, **k: None
    sys.modules["plume_nav_sim.debug"] = fake_debug
    return importlib.import_module("plume_nav_sim.envs.plume_navigation_env")


def test_env_requires_real_navigator(monkeypatch):
    env_module = load_env_module()
    env = object.__new__(env_module.PlumeNavigationEnv)
    env.env_width = 10
    env.env_height = 10
    monkeypatch.setattr(env_module, "NAVIGATOR_AVAILABLE", False)
    with pytest.raises(env_module.DependencyNotInstalled):
        env._init_navigator((0.0, 0.0), 0.0, 1.0, 1.0)


class DummySensor:
    def detect(self, plume_state, positions):
        return [False] * len(list(positions))

    def measure(self, plume_state, positions):
        return [0.0] * len(list(positions))

    def compute_gradient(self, plume_state, positions):
        return [(0.0, 0.0)] * len(list(positions))

    def configure(self, **kwargs):
        pass




def test_unknown_sensor_type_raises(monkeypatch):
    env_module = load_env_module()
    env = object.__new__(env_module.PlumeNavigationEnv)
    env.max_speed = 1.0
    env.max_angular_velocity = 1.0
    env.env_width = 10
    env.env_height = 10
    env.sensors = [DummySensor()]
    env._wind_enabled = False
    monkeypatch.setattr(env_module, "SPACES_AVAILABLE", False)
    with pytest.raises(ValueError):
        env._init_spaces()


def test_missing_video_file_raises(monkeypatch, tmp_path):
    env_module = load_env_module()
    env = object.__new__(env_module.PlumeNavigationEnv)
    env._plume_model_config = None
    env._video_path = tmp_path / "missing_video.mp4"
    env._video_frames = None
    with pytest.raises(RuntimeError) as exc:
        env._init_plume_model()
    assert isinstance(exc.value.__cause__, FileNotFoundError)
