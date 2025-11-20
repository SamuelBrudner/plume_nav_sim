from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from plume_nav_sim.data_capture.validate import validate_run_artifacts

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MAIN_PY = _REPO_ROOT / "plug-and-play-demo" / "main.py"


def _have_pyarrow() -> bool:
    """Best-effort check for pyarrow availability.

    Mirrors the import pattern used by RunRecorder._have_pyarrow.
    """

    try:
        import importlib

        importlib.import_module("pyarrow")
        importlib.import_module("pyarrow.parquet")
        return True
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.regression
def test_plug_and_play_main_capture_workflow(tmp_path: Path) -> None:
    """Exercise plug-and-play demo capture flags end-to-end.

    Invokes plug-and-play-demo/main.py via subprocess with a tiny grid and
    small episode count, writing into a temporary capture root. Asserts that
    core JSONL artifacts exist under <root>/<experiment>/<run_id>/ and that
    validate_run_artifacts reports OK for steps and episodes. When pyarrow is
    available, also asserts that Parquet exports are created.
    """

    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    experiment = "plugdemo"

    cmd = [
        sys.executable,
        str(_MAIN_PY),
        "--grid",
        "8x8",
        "--capture-root",
        str(capture_root),
        "--experiment",
        experiment,
        "--episodes",
        "2",
        "--parquet",
        "--validate",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        "plug-and-play demo main.py failed\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    exp_dir = capture_root / experiment
    assert exp_dir.is_dir(), f"Experiment directory missing: {exp_dir}"

    run_dirs = [
        p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("run-")
    ]
    assert run_dirs, f"No run directory found under {exp_dir}"

    # Use the most recently modified run directory (in case multiple runs exist).
    run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)

    run_json = run_dir / "run.json"
    steps_jsonl = run_dir / "steps.jsonl.gz"
    episodes_jsonl = run_dir / "episodes.jsonl.gz"

    assert run_json.exists(), "run.json missing from capture output"
    assert steps_jsonl.exists(), "steps.jsonl.gz missing from capture output"
    assert episodes_jsonl.exists(), "episodes.jsonl.gz missing from capture output"

    report = validate_run_artifacts(run_dir)
    assert report["steps"]["ok"] is True
    assert report["episodes"]["ok"] is True

    if _have_pyarrow():
        assert (run_dir / "steps.parquet").exists(), "steps.parquet missing"
        assert (run_dir / "episodes.parquet").exists(), "episodes.parquet missing"
