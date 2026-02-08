from __future__ import annotations

import os

import pytest

# Ensure Qt can run in headless CI (must be set before importing QtWidgets).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from plume_nav_debugger.env_driver import DebuggerConfig  # noqa: E402
from plume_nav_debugger.main_window import LiveConfigWidget  # noqa: E402
from plume_nav_debugger.replay_driver import ReplayDriver  # noqa: E402
from plume_nav_sim.data_capture.schemas import RunMeta  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_live_config_enables_registry_controls_only_with_dataset_id(
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    cfg = DebuggerConfig(plume="movie")
    w = LiveConfigWidget(cfg)
    w.plume_combo.setCurrentText("movie")
    w._on_fields_changed()

    # With no dataset id, registry-only controls are disabled.
    assert not w.movie_auto_download_check.isEnabled()
    assert not w.movie_cache_root_edit.isEnabled()

    w.movie_dataset_edit.setText("colorado_jet_v1")
    w._on_fields_changed()

    assert w.movie_auto_download_check.isEnabled()
    assert w.movie_cache_root_edit.isEnabled()


def test_live_config_apply_emits_movie_registry_fields(
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    cfg = DebuggerConfig(plume="movie")
    w = LiveConfigWidget(cfg)
    w.plume_combo.setCurrentText("movie")
    w.movie_dataset_edit.setText("colorado_jet_v1")
    w._on_fields_changed()

    w.movie_auto_download_check.setChecked(True)
    w.movie_cache_root_edit.setText("/tmp/plume-cache")

    captured: list[DebuggerConfig] = []
    w.apply_requested.connect(lambda obj: captured.append(obj))
    w._emit_apply()

    assert captured, "expected apply_requested to emit a DebuggerConfig"
    emitted = captured[-1]
    assert emitted.movie_dataset_id == "colorado_jet_v1"
    assert emitted.movie_auto_download is True
    assert emitted.movie_cache_root == "/tmp/plume-cache"


def test_replay_driver_resolves_env_kwargs_from_run_meta_extra(
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    meta = RunMeta(
        run_id="run-1",
        experiment="exp",
        start_time="2026-02-08T00:00:00Z",
        env_config={
            "grid_size": [8, 8],
            "source_location": [4, 4],
            "max_steps": 3,
            "goal_radius": 1.0,
            "plume_params": {"sigma": 12.0, "source_location": [4, 4]},
        },
        extra={
            "env": {
                "plume": "movie",
                "action_type": "run_tumble",
                "observation_type": "concentration",
                "reward_type": "step_penalty",
            },
            "movie": {
                "dataset_id": "colorado_jet_v1",
                "auto_download": False,
                "cache_root": "/tmp/cache",
            },
        },
    )

    class DummyEngine:
        run_meta = meta

    d = ReplayDriver()
    d._engine = DummyEngine()  # type: ignore[assignment]
    kwargs = d.get_resolved_env_kwargs(render=False)

    assert kwargs["grid_size"] == (8, 8)
    assert kwargs["source_location"] == (4, 4)
    assert kwargs["plume"] == "movie"
    assert kwargs["action_type"] == "run_tumble"
    assert kwargs["movie_dataset_id"] == "colorado_jet_v1"
    assert kwargs["movie_auto_download"] is False
    assert kwargs["movie_cache_root"] == "/tmp/cache"

