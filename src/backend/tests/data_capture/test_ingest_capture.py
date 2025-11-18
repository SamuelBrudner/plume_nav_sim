from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from .test_helpers import run_small_capture


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


@pytest.mark.parametrize(
    "export_parquet, remove_manifest, force_jsonl_only",
    [
        # JSONL-only path: disable parquet generation explicitly
        (False, False, True),
        # Parquet path (if pyarrow available): keep manifest
        (True, False, False),
        # Missing manifest should be auto-generated during ingest
        (False, True, True),
    ],
)
def test_ingest_capture_copies_and_indexes(
    tmp_path: Path,
    export_parquet: bool,
    remove_manifest: bool,
    force_jsonl_only: bool,
) -> None:
    # 1) Manufacture a tiny run directory via the capture CLI helper
    run_dir = run_small_capture(
        tmp_path / "results",
        experiment="ingest_test",
        episodes=2,
        grid=(8, 8),
        max_steps=20,
        seed=123,
        export_parquet=export_parquet,
    )

    # Optionally remove manifest to exercise auto-generation in ingest script
    manifest_path = run_dir / "manifest.json"
    if remove_manifest and manifest_path.exists():
        manifest_path.unlink()

    # 2) Import the ingest script module directly from its file path
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "ingest_capture.py"
    spec = importlib.util.spec_from_file_location("ingest_capture_mod", script_path)
    assert spec and spec.loader, "Failed to resolve ingest_capture script path"
    ingest_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(ingest_mod)  # type: ignore[union-attr]

    # Optionally force JSONL-only by disabling parquet generation in the ingest script
    if force_jsonl_only:
        orig_ensure = getattr(ingest_mod, "_ensure_parquet")

        def _noop(_run_dir: Path) -> None:
            return None

        setattr(ingest_mod, "_ensure_parquet", _noop)

    out_root = tmp_path / "data" / "captures"
    dest = ingest_mod.ingest_capture(run_dir, out_root)

    # Restore original _ensure_parquet if it was monkeypatched
    if force_jsonl_only:
        setattr(ingest_mod, "_ensure_parquet", orig_ensure)

    # 3) Assert destination structure matches DVC expectations: out_root/<experiment>/<run_id>
    assert dest == out_root / run_dir.parent.name / run_dir.name
    assert dest.is_dir()

    # 4) Assert core artifacts copied
    assert (dest / "run.json").exists(), "run.json not copied"
    assert (dest / "steps.jsonl.gz").exists(), "steps.jsonl.gz not copied"
    assert (dest / "episodes.jsonl.gz").exists(), "episodes.jsonl.gz not copied"

    # 5) Manifest should exist at destination (copied or auto-generated)
    dest_manifest = dest / "manifest.json"
    assert dest_manifest.exists(), "manifest.json missing at destination"
    with open(dest_manifest, "r", encoding="utf-8") as fh:
        manifest_obj = json.load(fh)
        assert manifest_obj.get("experiment") == run_dir.parent.name
        assert manifest_obj.get("run_id") == run_dir.name

    # 6) Dataset index written
    idx_path = dest / "DATASET_INDEX.json"
    assert idx_path.exists(), "DATASET_INDEX.json not written"
    with open(idx_path, "r", encoding="utf-8") as fh:
        idx = json.load(fh)
        assert idx.get("experiment") == run_dir.parent.name
        assert idx.get("run_id") == run_dir.name
        # schema_version may be None in minimal runs; ensure key exists
        assert "schema_version" in idx
        assert idx.get("source")

    # 7) Parquet behavior
    steps_parquet = dest / "steps.parquet"
    episodes_parquet = dest / "episodes.parquet"
    if force_jsonl_only:
        # Explicitly disabled parquet generation for this parameterization
        assert not steps_parquet.exists(), "Unexpected steps.parquet in JSONL-only mode"
        assert (
            not episodes_parquet.exists()
        ), "Unexpected episodes.parquet in JSONL-only mode"
    else:
        # Parquet expected only if pyarrow is available
        if _have("pyarrow") and _have("pyarrow.parquet"):
            assert steps_parquet.exists(), "steps.parquet not found in Parquet mode"
            assert (
                episodes_parquet.exists()
            ), "episodes.parquet not found in Parquet mode"
        else:
            pytest.skip("pyarrow not available; Parquet path not asserted")
