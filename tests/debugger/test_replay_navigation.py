from __future__ import annotations

import os
from pathlib import Path

import pytest

# Ensure Qt can run in headless CI (must be set before importing QtWidgets).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from plume_nav_debugger.replay_driver import ReplayDriver  # noqa: E402
from plume_nav_debugger.widgets.control_bar import ControlBar  # noqa: E402
from plume_nav_sim.cli import capture  # noqa: E402
from plume_nav_sim.data_capture import load_replay_engine  # noqa: E402
from plume_nav_sim.data_capture.loader import load_replay_artifacts  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _Ev:
    def __init__(
        self,
        *,
        terminated: bool = False,
        truncated: bool = False,
        info: dict | None = None,
    ) -> None:
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.info = info or {}


def _latest_run_dir(output_root: Path, experiment: str) -> Path:
    exp_dir = output_root / experiment
    run_dirs = sorted(path for path in exp_dir.iterdir() if path.is_dir())
    assert run_dirs, f"no run directories under {exp_dir}"
    return run_dirs[-1]


def test_replay_driver_markers_and_semantic_jumps(
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    d = ReplayDriver()
    d._engine = object()  # type: ignore[assignment]
    d._events = [
        _Ev(),
        _Ev(terminated=True, info={"goal_reached": True}),
        _Ev(),
        _Ev(truncated=True),
        _Ev(),
    ]
    d._episode_starts = [0, 3]
    d._index = 0

    markers = d.get_timeline_markers()
    assert markers["episode"] == [0, 3]
    assert markers["terminated"] == [1]
    assert markers["truncated"] == [3]
    assert markers["goal"] == [1]

    assert d.jump_next_done() is True
    assert d.current_index() == 1

    # No additional goal event after index=1.
    assert d.jump_next_goal() is False
    assert d.current_index() == 1

    assert d.jump_next_episode() is True
    assert d.current_index() == 3

    assert d.jump_prev_episode() is True
    assert d.current_index() == 0


def test_control_bar_replay_jump_signals_and_markers(
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    c = ControlBar()
    c.set_timeline(5, 0, total_episodes=2, current_episode=0)

    got: list[str] = []
    c.jump_prev_episode_requested.connect(lambda: got.append("prev_ep"))
    c.jump_next_episode_requested.connect(lambda: got.append("next_ep"))
    c.jump_next_done_requested.connect(lambda: got.append("next_done"))
    c.jump_next_goal_requested.connect(lambda: got.append("next_goal"))

    c.prev_episode_btn.click()
    c.next_episode_btn.click()
    c.next_done_btn.click()
    c.next_goal_btn.click()

    assert got == ["prev_ep", "next_ep", "next_done", "next_goal"]

    c.set_replay_markers({"episode": [0, 2], "truncated": [2]})
    assert c.timeline_slider.get_markers()["episode"] == [0, 2]


def test_replay_driver_loads_real_capture_run(
    tmp_path: Path,
    qapp: QtWidgets.QApplication,  # noqa: ARG001
) -> None:
    output_root = tmp_path / "results"
    rc = capture.main(
        [
            "--output",
            str(output_root),
            "--experiment",
            "replay-e2e",
            "--episodes",
            "1",
            "--grid",
            "8x8",
            "--max-steps",
            "4",
            "--seed",
            "123",
        ]
    )
    assert rc == 0

    run_dir = _latest_run_dir(output_root, "replay-e2e")
    artifacts = load_replay_artifacts(run_dir)

    driver = ReplayDriver()
    errors: list[str] = []
    driver.error_occurred.connect(lambda msg: errors.append(str(msg)))

    driver.load_run(run_dir)

    assert not errors
    assert driver.is_loaded()
    assert driver.total_steps() == len(artifacts.steps)
    assert driver.total_episodes() == len(artifacts.episodes)
    assert driver.current_index() == 0


def test_replay_engine_exposes_episode_boundaries(tmp_path: Path) -> None:
    output_root = tmp_path / "results"
    rc = capture.main(
        [
            "--output",
            str(output_root),
            "--experiment",
            "engine-boundaries",
            "--episodes",
            "2",
            "--grid",
            "8x8",
            "--max-steps",
            "2",
            "--seed",
            "50",
        ]
    )
    assert rc == 0

    run_dir = _latest_run_dir(output_root, "engine-boundaries")
    engine = load_replay_engine(str(run_dir))

    assert engine.steps
    assert len(engine.episode_starts) == engine.total_episodes()
    assert engine.episode_starts[0] == 0
    assert engine.run_dir == str(run_dir)
