from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from plume_nav_sim.data_capture.validate import validate_run_artifacts

from .helpers import run_small_capture


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def test_capture_pipeline_validates_and_exports(tmp_path: Path) -> None:
    # Run a tiny deterministic capture
    run_dir = run_small_capture(
        tmp_path,
        experiment="ci_guard",
        episodes=2,
        grid=(8, 8),
        max_steps=20,
        seed=123,
        export_parquet=True,
    )

    # Core artifacts exist
    steps_gz = run_dir / "steps.jsonl.gz"
    episodes_gz = run_dir / "episodes.jsonl.gz"
    assert steps_gz.exists(), "steps.jsonl.gz not found"
    assert episodes_gz.exists(), "episodes.jsonl.gz not found"

    # Batch validate with Pandera schemas
    report = validate_run_artifacts(run_dir)
    assert isinstance(report, dict)
    assert report.get("steps", {}).get("ok") is True, f"Steps failed: {report}"
    assert report.get("episodes", {}).get("ok") is True, f"Episodes failed: {report}"
    assert report["steps"]["rows"] >= 1, "Expected at least one step record"
    assert report["episodes"]["rows"] >= 1, "Expected at least one episode record"

    # If pyarrow is available (installed via [data] extra), verify Parquet export matches schema fields
    if _have("pyarrow"):
        import pandas as pd  # type: ignore

        from plume_nav_sim.data_capture.validate import _schemas

        steps_parquet = run_dir / "steps.parquet"
        episodes_parquet = run_dir / "episodes.parquet"
        assert steps_parquet.exists(), "steps.parquet not found"
        assert episodes_parquet.exists(), "episodes.parquet not found"

        sdf = pd.read_parquet(steps_parquet)
        edf = pd.read_parquet(episodes_parquet)
        steps_schema, episodes_schema = _schemas()

        # Expected columns from schema should be present in Parquet columns
        expected_steps_cols = set(steps_schema.columns.keys())
        expected_episodes_cols = set(episodes_schema.columns.keys())
        assert expected_steps_cols.issubset(set(sdf.columns)), (
            f"steps.parquet columns missing schema fields: "
            f"expected {expected_steps_cols}, got {set(sdf.columns)}"
        )
        assert expected_episodes_cols.issubset(set(edf.columns)), (
            f"episodes.parquet columns missing schema fields: "
            f"expected {expected_episodes_cols}, got {set(edf.columns)}"
        )
