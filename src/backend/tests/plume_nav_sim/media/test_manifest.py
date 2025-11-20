from __future__ import annotations

import json
import sys
from pathlib import Path

from plume_nav_sim import PACKAGE_VERSION
from plume_nav_sim.media import (
    MANIFEST_FILENAME,
    ProvenanceManifest,
    build_provenance_manifest,
    get_default_manifest_path,
    load_manifest,
    write_manifest,
)


def test_manifest_constants_and_path(tmp_path: Path):
    assert MANIFEST_FILENAME == "manifest.json"
    root = tmp_path / "dataset"
    p = get_default_manifest_path(root)
    assert p.name == MANIFEST_FILENAME
    assert p.parent == root


def test_manifest_roundtrip_write_and_load(tmp_path: Path):
    root = tmp_path / "zarr_root"
    m = build_provenance_manifest(
        source_dtype="uint8",
        cli_args=["video_ingest", "--input", "x.mp4", "--output", str(root)],
        git_sha="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        config_hash="abc123cfg",
        package_version=PACKAGE_VERSION,
    )

    out_path = write_manifest(root, m)
    assert out_path.exists()
    assert out_path.name == MANIFEST_FILENAME

    # Validate JSON structure manually
    data = json.loads(out_path.read_text())
    # Required fields
    assert data["source_dtype"] == "uint8"
    assert data["package_version"] == PACKAGE_VERSION
    assert data["git_sha"] == "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    assert data["config_hash"] == "abc123cfg"
    assert isinstance(data["created_at"], str)
    # CLI args recorded exactly
    assert data["cli_args"][:2] == ["video_ingest", "--input"]

    # Pydantic validation on load
    loaded = load_manifest(root)
    assert isinstance(loaded, ProvenanceManifest)
    assert loaded.source_dtype == "uint8"
    assert loaded.package_version == PACKAGE_VERSION


def test_manifest_env_metadata_present_by_default(tmp_path: Path):
    root = tmp_path / "dataset"
    m = build_provenance_manifest(source_dtype="float32")
    write_manifest(root, m)

    data = json.loads((root / MANIFEST_FILENAME).read_text())
    # env is present with non-sensitive hints; keys may be None
    assert "env" in data
    assert set(data["env"].keys()) == {"hostname", "platform", "python_version"}
