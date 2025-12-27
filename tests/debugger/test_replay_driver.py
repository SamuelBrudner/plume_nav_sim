from __future__ import annotations

import gzip
import importlib
import json
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip(
    "plume_nav_debugger",
    reason="Debugger package not importable; ensure PYTHONPATH=src for local runs",
)


def _pyqt5_present() -> bool:
    try:
        if "PyQt5" in sys.modules:
            return True
        return importlib.util.find_spec("PyQt5") is not None
    except Exception:
        return False


def _write_fake_run(
    tmp_path: Path, *, episodes: int = 1, steps_per_episode: int = 2
) -> tuple[Path, list[dict]]:
    from plume_nav_sim.data_capture.schemas import SCHEMA_VERSION

    run_dir = tmp_path / "run-replay"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = "run-test"
    env_cfg = {
        "grid_size": [4, 4],
        "max_steps": 3,
        "goal_radius": 1.0,
        "enable_rendering": False,
    }
    run_meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "start_time": datetime.utcnow().isoformat(),
        "env_config": env_cfg,
        "base_seed": 42,
        "system": {"hostname": "local"},
    }
    with open(run_dir / "run.json", "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh)

    steps: list[dict] = []
    for ep_idx in range(episodes):
        ep_id = f"ep-{ep_idx + 1}"
        for step_idx in range(steps_per_episode):
            reward = 0.1 * (step_idx + 1 + ep_idx)
            steps.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "ts": time.time(),
                    "run_id": run_id,
                    "episode_id": ep_id,
                    "step": step_idx + 1,
                    "action": (step_idx + ep_idx) % 2,
                    "reward": reward,
                    "terminated": step_idx == steps_per_episode - 1,
                    "truncated": False,
                    "agent_position": {"x": step_idx, "y": ep_idx},
                    "distance_to_goal": 5.0 - 0.5 * step_idx,
                    "observation_summary": [float(step_idx)],
                    "seed": 42 + ep_idx,
                }
            )
    with gzip.open(run_dir / "steps.jsonl.gz", "wt", encoding="utf-8") as fh:
        for rec in steps:
            fh.write(json.dumps(rec) + "\n")

    episode_records: list[dict] = []
    for ep_idx in range(episodes):
        ep_id = f"ep-{ep_idx + 1}"
        total_reward = sum(rec["reward"] for rec in steps if rec["episode_id"] == ep_id)
        episode_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "episode_id": ep_id,
                "terminated": True,
                "truncated": False,
                "total_steps": steps_per_episode,
                "total_reward": total_reward,
                "final_position": {
                    "x": steps_per_episode - 1,
                    "y": ep_idx,
                },
                "final_distance_to_goal": 5.0 - 0.5 * (steps_per_episode - 1),
                "duration_ms": float(steps_per_episode),
                "avg_step_time_ms": 1.0,
            }
        )
    with gzip.open(run_dir / "episodes.jsonl.gz", "wt", encoding="utf-8") as fh:
        for rec in episode_records:
            fh.write(json.dumps(rec) + "\n")

    return run_dir, steps


@pytest.mark.skipif(
    _pyqt5_present(),
    reason="PySide6 not available or PyQt5 present (binding conflict)",
)
def test_replay_driver_loads_and_seeks(tmp_path) -> None:
    QtWidgets = pytest.importorskip(
        "PySide6.QtWidgets",
        reason="PySide6 not available or Qt stack incomplete",
    )
    from plume_nav_debugger.replay_driver import ReplayDriver

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    run_dir, steps = _write_fake_run(Path(tmp_path))

    driver = ReplayDriver()
    timeline_updates: list[tuple[int, int]] = []
    driver.timeline_changed.connect(
        lambda cur, total: timeline_updates.append((cur, total))
    )
    driver.load_run(run_dir)

    assert driver.total_steps() == len(steps)
    assert driver.total_episodes() == 1
    assert driver.current_index() == 0  # first event is emitted on load
    assert timeline_updates and timeline_updates[-1][1] == len(steps)

    events = []
    driver.step_done.connect(lambda ev: events.append(ev))
    driver.step_once()

    assert driver.current_index() == 1
    assert events, "Replay driver should emit events on step"

    driver.seek_to(len(steps) - 1)
    assert driver.current_index() == len(steps) - 1


@pytest.mark.skipif(
    _pyqt5_present(),
    reason="PySide6 not available or PyQt5 present (binding conflict)",
)
def test_replay_driver_wires_inspector_signals(tmp_path) -> None:
    QtWidgets = pytest.importorskip(
        "PySide6.QtWidgets",
        reason="PySide6 not available or Qt stack incomplete",
    )
    from plume_nav_debugger.main_window import InspectorWidget
    from plume_nav_debugger.replay_driver import ReplayDriver

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    run_dir, _ = _write_fake_run(Path(tmp_path))

    driver = ReplayDriver()
    insp = InspectorWidget()
    insp.set_strict_provider_only(True)

    run_meta: list[tuple[int, object]] = []
    driver.run_meta_changed.connect(lambda seed, start: run_meta.append((seed, start)))
    driver.provider_mux_changed.connect(insp.on_mux_changed)
    driver.action_space_changed.connect(insp.on_action_names)
    driver.step_done.connect(insp.on_step_event)

    driver.load_run(run_dir)
    driver.step_once()

    assert run_meta and run_meta[-1][0] == 42
    assert run_meta[-1][1] == (0, 0)
    assert insp._obs_model.summary is not None  # type: ignore[attr-defined]
    assert insp._act_model.state.action_label != "-"  # type: ignore[attr-defined]


@pytest.mark.skipif(
    _pyqt5_present(),
    reason="PySide6 not available or PyQt5 present (binding conflict)",
)
def test_replay_driver_caches_and_seeks_by_episode(tmp_path) -> None:
    QtWidgets = pytest.importorskip(
        "PySide6.QtWidgets",
        reason="PySide6 not available or Qt stack incomplete",
    )
    from plume_nav_debugger.replay_driver import ReplayDriver

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    run_dir, steps = _write_fake_run(Path(tmp_path), episodes=2, steps_per_episode=3)

    driver = ReplayDriver()
    driver.load_run(run_dir)
    assert driver.total_episodes() == 2
    assert driver.current_episode_index() == 0

    # Patch iter_events to track start_step calls for cache reuse
    calls: list[int] = []
    engine = driver._engine
    assert engine is not None
    orig_iter = engine.iter_events

    def traced_iter(self, *, render=False, validate=False, start_step=0):
        calls.append(int(start_step))
        yield from orig_iter(render=render, validate=validate, start_step=start_step)

    engine.iter_events = types.MethodType(traced_iter, engine)

    # Rebuild iterator using patched method and advance cache
    driver.seek_to(driver.current_index())
    driver.step_once()

    driver.seek_to(0)
    assert calls and 0 not in calls  # cache avoids restarting from step 0
    assert min(calls) >= 1

    driver.seek_to_episode(1)
    assert driver.current_episode_index() == 1
    assert driver.current_index() == 3
