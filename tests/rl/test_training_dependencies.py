import builtins
import importlib
import sys
import types

import importlib.util
import pytest

MODULE = "odor_plume_nav.rl.training"


def _simulate_missing_module(monkeypatch, module_name):
    def ensure_dependencies():
        # create stubs for required modules when absent
        if module_name != "stable_baselines3":
            try:
                import stable_baselines3  # noqa: F401
            except ImportError:
                sb3 = types.ModuleType("stable_baselines3")
                sb3.__path__ = []  # type: ignore[attr-defined]
                sb3.PPO = sb3.SAC = sb3.TD3 = sb3.A2C = sb3.DQN = object
                common = types.ModuleType("stable_baselines3.common")
                common.__path__ = []  # type: ignore[attr-defined]
                base_class = types.ModuleType("stable_baselines3.common.base_class")
                setattr(base_class, "BaseAlgorithm", object)
                vec_env = types.ModuleType("stable_baselines3.common.vec_env")
                for attr in ["VecEnv", "DummyVecEnv", "SubprocVecEnv", "SyncVectorEnv", "AsyncVectorEnv"]:
                    setattr(vec_env, attr, object)
                callbacks = types.ModuleType("stable_baselines3.common.callbacks")
                for attr in [
                    "BaseCallback",
                    "CheckpointCallback",
                    "EvalCallback",
                    "StopTrainingOnRewardThreshold",
                    "CallbackList",
                ]:
                    setattr(callbacks, attr, object)
                monitor = types.ModuleType("stable_baselines3.common.monitor")
                setattr(monitor, "Monitor", object)
                evaluation = types.ModuleType("stable_baselines3.common.evaluation")
                setattr(evaluation, "evaluate_policy", lambda *a, **kw: (0.0, 0.0))
                env_util = types.ModuleType("stable_baselines3.common.env_util")
                setattr(env_util, "make_vec_env", lambda *a, **kw: None)
                utils = types.ModuleType("stable_baselines3.common.utils")
                setattr(utils, "safe_mean", lambda *a, **kw: 0.0)
                monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
                monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
                monkeypatch.setitem(sys.modules, "stable_baselines3.common.base_class", base_class)
                monkeypatch.setitem(sys.modules, "stable_baselines3.common.vec_env", vec_env)
                monkeypatch.setitem(sys.modules, "stable_baselines3.common.callbacks", callbacks)
                monkeypatch.setitem(sys.modules, "stable_baselines3.common.monitor", monitor)
                monkeypatch.setitem(sys.modules, "stable_baselines3.common.evaluation", evaluation)
                monkeypatch.setitem(sys.modules, "stable_baselines3.common.env_util", env_util)
                monkeypatch.setitem(sys.modules, "stable_baselines3.common.utils", utils)
        if module_name != "gymnasium":
            try:
                import gymnasium  # noqa: F401
            except ImportError:
                gym = types.ModuleType("gymnasium")
                gym.__path__ = []  # type: ignore[attr-defined]
                vector = types.ModuleType("gymnasium.vector")
                setattr(vector, "SyncVectorEnv", object)
                setattr(vector, "AsyncVectorEnv", object)
                gym.vector = vector
                monkeypatch.setitem(sys.modules, "gymnasium", gym)
                monkeypatch.setitem(sys.modules, "gymnasium.vector", vector)
        if module_name != "tensorboard":
            try:
                import tensorboard  # noqa: F401
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401
            except ImportError:
                tb = types.ModuleType("tensorboard")
                monkeypatch.setitem(sys.modules, "tensorboard", tb)
                torch_mod = types.ModuleType("torch")
                torch_mod.__path__ = []  # type: ignore[attr-defined]
                utils_mod = types.ModuleType("torch.utils")
                utils_mod.__path__ = []  # type: ignore[attr-defined]
                tb_mod = types.ModuleType("torch.utils.tensorboard")
                class DummyWriter:
                    pass
                tb_mod.SummaryWriter = DummyWriter
                utils_mod.tensorboard = tb_mod
                torch_mod.utils = utils_mod
                monkeypatch.setitem(sys.modules, "torch", torch_mod)
                monkeypatch.setitem(sys.modules, "torch.utils", utils_mod)
                monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tb_mod)
        if module_name != "wandb":
            try:
                import wandb  # noqa: F401
            except ImportError:
                monkeypatch.setitem(sys.modules, "wandb", types.ModuleType("wandb"))

    ensure_dependencies()
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == module_name or name.startswith(module_name + "."):
            raise ImportError(f"No module named '{module_name}'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.mark.parametrize(
    "missing_module, expected", [
        ("stable_baselines3", "stable-baselines3"),
        ("gymnasium", "(?i)gymnasium"),
        ("tensorboard", "(?i)tensorboard"),
        ("wandb", "wandb"),
    ]
)
def test_training_import_requires_dependencies(monkeypatch, missing_module, expected):
    if missing_module != "stable_baselines3" and importlib.util.find_spec("stable_baselines3") is None:
        pytest.skip("stable-baselines3 missing; cannot test other dependencies")
    _simulate_missing_module(monkeypatch, missing_module)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    with pytest.raises(ImportError, match=expected):
        importlib.import_module(MODULE)
