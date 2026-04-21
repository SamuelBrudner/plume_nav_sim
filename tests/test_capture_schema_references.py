from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

from plume_nav_sim.data_capture.schemas import SCHEMA_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "data" / "replay_fixture"


def _first_jsonl_record(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.loads(next(fh))


def test_active_capture_docs_and_fixture_match_schema_version() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    capture_catalog = (
        REPO_ROOT / "src" / "backend" / "docs" / "data_catalog_capture.md"
    ).read_text(encoding="utf-8")
    capture_schema_doc = (
        REPO_ROOT / "src" / "backend" / "docs" / "data_capture_schemas.md"
    ).read_text(encoding="utf-8")

    readme_match = re.search(
        r"Loader hard-validates schema_version `([^`]+)`", readme
    )
    catalog_match = re.search(r"Current version: `([^`]+)`", capture_catalog)
    schema_doc_match = re.search(r"Schema version: ([0-9.]+)", capture_schema_doc)

    assert readme_match is not None
    assert readme_match.group(1) == SCHEMA_VERSION
    assert catalog_match is not None
    assert catalog_match.group(1) == SCHEMA_VERSION
    assert schema_doc_match is not None
    assert schema_doc_match.group(1) == SCHEMA_VERSION

    run_meta = json.loads((FIXTURE_DIR / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    step = _first_jsonl_record(FIXTURE_DIR / "steps.jsonl.gz")
    episode = _first_jsonl_record(FIXTURE_DIR / "episodes.jsonl.gz")

    assert run_meta["schema_version"] == SCHEMA_VERSION
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert step["schema_version"] == SCHEMA_VERSION
    assert episode["schema_version"] == SCHEMA_VERSION
