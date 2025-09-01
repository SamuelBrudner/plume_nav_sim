import importlib
import builtins
import sys

import pytest

MODULE = "plume_nav_sim.models.plume.video_plume_adapter"


@pytest.mark.parametrize(
    "missing",
    [
        "cv2",
        "plume_nav_sim.envs.video_plume",
        "plume_nav_sim.utils.frame_cache",
    ],
)
def test_import_error_on_missing_dependency(monkeypatch, missing):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing:
            raise ImportError(f"No module named {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    if MODULE in sys.modules:
        del sys.modules[MODULE]

    with pytest.raises(ImportError):
        importlib.import_module(MODULE)
