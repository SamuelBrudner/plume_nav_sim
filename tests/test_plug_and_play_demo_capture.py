"""Tests for plug-and-play demo capture mode (bead 166).

Exercises argparse flags: --capture-root, --experiment, --episodes, --validate.
Asserts capture artifacts exist: run.json, steps.jsonl.gz, episodes.jsonl.gz.
"""
from __future__ import annotations

import json
import gzip
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_MAIN = PROJECT_ROOT / "plug-and-play-demo" / "main.py"


def _run_demo(tmp_path: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    capture_root = tmp_path / "capture"
    cmd = [
        sys.executable,
        str(DEMO_MAIN),
        "--max-steps", "20",
        "--grid", "32x32",
        "--capture-root", str(capture_root),
        "--experiment", "test_exp",
        "--episodes", "1",
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    return result


def _find_run_dir(capture_root: Path) -> Path:
    """Locate the generated run directory under capture_root/test_exp/."""
    exp_dir = capture_root / "test_exp"
    assert exp_dir.exists(), f"experiment dir missing: {exp_dir}"
    run_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) >= 1, f"no run directories under {exp_dir}"
    return run_dirs[0]


def test_capture_produces_run_json(tmp_path: Path) -> None:
    result = _run_demo(tmp_path)
    assert result.returncode == 0, (
        f"demo failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    run_dir = _find_run_dir(tmp_path / "capture")
    meta_path = run_dir / "run.json"
    assert meta_path.exists(), f"run.json missing in {run_dir}"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "run_id" in meta


def test_capture_produces_steps_jsonl(tmp_path: Path) -> None:
    result = _run_demo(tmp_path)
    assert result.returncode == 0
    run_dir = _find_run_dir(tmp_path / "capture")
    steps_files = list(run_dir.glob("steps*.jsonl.gz"))
    assert len(steps_files) >= 1, f"no steps.jsonl.gz in {run_dir}"
    with gzip.open(steps_files[0], "rt", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) > 0, "steps.jsonl.gz is empty"
    first = json.loads(lines[0])
    assert "step" in first or "action" in first


def test_capture_produces_episodes_jsonl(tmp_path: Path) -> None:
    result = _run_demo(tmp_path)
    assert result.returncode == 0
    run_dir = _find_run_dir(tmp_path / "capture")
    ep_files = list(run_dir.glob("episodes*.jsonl.gz"))
    assert len(ep_files) >= 1, f"no episodes.jsonl.gz in {run_dir}"
    with gzip.open(ep_files[0], "rt", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) >= 1, "episodes.jsonl.gz has no records"


def test_capture_multi_episode(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    cmd = [
        sys.executable, str(DEMO_MAIN),
        "--max-steps", "10",
        "--grid", "32x32",
        "--capture-root", str(capture_root),
        "--experiment", "multi",
        "--episodes", "3",
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"multi-episode failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    exp_dir = capture_root / "multi"
    assert exp_dir.exists()
    run_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) >= 1
    run_dir = run_dirs[0]
    ep_files = list(run_dir.glob("episodes*.jsonl.gz"))
    assert len(ep_files) >= 1
    total_episodes = 0
    for ep_file in ep_files:
        with gzip.open(ep_file, "rt", encoding="utf-8") as f:
            total_episodes += sum(1 for _ in f)
    assert total_episodes >= 3, f"expected >=3 episode records, got {total_episodes}"


def test_capture_validate_flag(tmp_path: Path) -> None:
    result = _run_demo(tmp_path, extra_args=["--validate"])
    assert result.returncode == 0, (
        f"validate failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "Validation" in result.stdout or "Capture complete" in result.stdout


def test_non_capture_default_unchanged(tmp_path: Path) -> None:
    """Without capture flags, demo runs a single episode and prints summary."""
    cmd = [
        sys.executable, str(DEMO_MAIN),
        "--max-steps", "10",
        "--grid", "32x32",
        "--no-render",
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Episode summary" in result.stdout
    assert "frames_captured" in result.stdout
