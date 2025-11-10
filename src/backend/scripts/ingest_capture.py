from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


def _ensure_parquet(run_dir: Path) -> None:
    """If pyarrow is available, generate Parquet files for steps/episodes.

    This mirrors the logic from RunRecorder._export_parquet but works standalone.
    """
    try:
        import pyarrow as _pa  # type: ignore
        import pyarrow.parquet as _pq  # type: ignore
    except Exception:
        return

    for stem in ("steps", "episodes"):
        jsonl_paths: List[Path] = sorted(run_dir.glob(f"{stem}.part*.jsonl.gz"))
        base = run_dir / f"{stem}.jsonl.gz"
        if base.exists():
            jsonl_paths.insert(0, base)
        if not jsonl_paths:
            continue

        rows = []
        for p in jsonl_paths:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
        if not rows:
            continue

        table = _pa.Table.from_pylist(rows)
        _pq.write_table(table, run_dir / f"{stem}.parquet")


def _collect_schema_version(run_dir: Path) -> str | None:
    try:
        with open(run_dir / "run.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
            return meta.get("schema_version")
    except Exception:
        return None


def _generate_manifest_if_missing(run_dir: Path) -> None:
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        return
    # Minimal manifest; recorder will usually create a richer one
    try:
        # Prefer import via installed package name if available
        from plume_nav_sim.data_capture.validate import (  # type: ignore
            validate_run_artifacts,
        )
    except Exception:
        try:
            # Fallback to source layout import if PYTHONPATH includes repo root
            from src.backend.plume_nav_sim.data_capture.validate import (  # type: ignore
                validate_run_artifacts,
            )
        except Exception:
            validate_run_artifacts = None  # type: ignore

    report: Dict[str, Any] = {}
    if validate_run_artifacts is not None:
        try:
            report = validate_run_artifacts(run_dir)
        except Exception as e:
            report = {"error": str(e)}

    try:
        with open(run_dir / "run.json", "r", encoding="utf-8") as fh:
            run_meta = json.load(fh)
    except Exception:
        run_meta = {}

    schema_version = _collect_schema_version(run_dir)

    data: Dict[str, Any] = {
        "run_id": run_meta.get("run_id"),
        "experiment": run_meta.get("experiment"),
        "schema_version": schema_version,
        "validation": report,
        "files": [f.name for f in (run_dir.glob("*.jsonl.gz"))],
    }
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def ingest_capture(run_dir: Path, out_root: Path) -> Path:
    run_dir = run_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Produce parquet in-place if libraries are available
    _ensure_parquet(run_dir)

    # Ensure a manifest exists
    _generate_manifest_if_missing(run_dir)

    # Determine destination path from run.json (experiment/run_id)
    try:
        with open(run_dir / "run.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        experiment = meta.get("experiment") or run_dir.parent.name
        run_id = meta.get("run_id") or run_dir.name
    except Exception:
        experiment = run_dir.parent.name
        run_id = run_dir.name

    dest = out_root / str(experiment) / str(run_id)
    dest.mkdir(parents=True, exist_ok=True)

    # Copy core artifacts
    to_copy = [
        "run.json",
        "manifest.json",
        "steps.jsonl.gz",
        "episodes.jsonl.gz",
        "steps.parquet",
        "episodes.parquet",
    ]
    for name in to_copy:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)

    # Copy sharded parts if any
    for p in sorted(run_dir.glob("steps.part*.jsonl.gz")):
        shutil.copy2(p, dest / p.name)
    for p in sorted(run_dir.glob("episodes.part*.jsonl.gz")):
        shutil.copy2(p, dest / p.name)

    # Write a small dataset-level index
    index: Dict[str, Any] = {
        "experiment": experiment,
        "run_id": run_id,
        "schema_version": _collect_schema_version(run_dir),
        "source": str(run_dir),
    }
    with open(dest / "DATASET_INDEX.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)

    return dest


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest capture artifacts into versioned dataset"
    )
    ap.add_argument(
        "--run-dir", required=True, help="Path to results/<experiment>/<run_id>"
    )
    ap.add_argument(
        "--out-dir",
        default="data/captures",
        help="Dataset root where experiment/run_id will be created",
    )
    # Parquet export happens automatically if pyarrow is available
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_root = Path(args.out_dir)
    dest = ingest_capture(run_dir, out_root)
    print(str(dest))


if __name__ == "__main__":
    main()
