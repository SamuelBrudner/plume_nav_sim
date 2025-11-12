from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from plume_nav_sim import PACKAGE_VERSION

# Public constants
MANIFEST_FILENAME = "manifest.json"


class SystemEnv(BaseModel):
    """Optional environment metadata for provenance (kept minimal).

    Avoid including sensitive information; intended for reproducibility hints.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    hostname: Optional[str] = None
    platform: Optional[str] = None
    python_version: Optional[str] = None


class ProvenanceManifest(BaseModel):
    """Provenance manifest stored alongside video-derived plume datasets.

    This record captures information needed to reproduce a dataset on a given
    commit and configuration, and how it was invoked at the CLI level.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Reproducibility & versioning
    git_sha: Optional[str] = None
    package_version: Optional[str] = Field(default=PACKAGE_VERSION)
    config_hash: Optional[str] = None

    # Invocation details
    cli_args: Optional[List[str]] = None

    # Source/media details
    source_dtype: str

    # Optional environment metadata
    env: SystemEnv = Field(default_factory=SystemEnv)


def build_provenance_manifest(
    *,
    source_dtype: str,
    cli_args: Optional[List[str]] = None,
    git_sha: Optional[str] = None,
    config_hash: Optional[str] = None,
    package_version: Optional[str] = PACKAGE_VERSION,
    include_env: bool = True,
    env: Optional[SystemEnv] = None,
) -> ProvenanceManifest:
    """Assemble a provenance manifest with sensible defaults.

    Args:
        source_dtype: Numpy-like dtype string of the source frames (e.g. "uint8").
        cli_args: Exact CLI argv list used to invoke the ingest.
        git_sha: Git commit SHA for reproducibility.
        config_hash: Stable hash/fingerprint of the resolved config (if any).
        package_version: Package version recorded for traceability.
        include_env: When True, include a best-effort SystemEnv snapshot.
        env: Optional explicit SystemEnv to include; overrides include_env.

    Returns:
        ProvenanceManifest instance.
    """

    system_env = (
        env if env is not None else (SystemEnv() if include_env else SystemEnv())
    )

    return ProvenanceManifest(
        git_sha=git_sha,
        package_version=package_version,
        config_hash=config_hash,
        cli_args=list(cli_args) if cli_args is not None else None,
        source_dtype=source_dtype,
        env=system_env,
    )


def get_default_manifest_path(root: Path | str) -> Path:
    """Return the canonical path for the manifest under a dataset root."""
    root_path = Path(root)
    return root_path / MANIFEST_FILENAME


def write_manifest(root: Path | str, manifest: ProvenanceManifest) -> Path:
    """Write the manifest JSON to the dataset root and return the path.

    Datetime fields are serialized via Pydantic's JSON encoder.
    """
    path = get_default_manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use model_dump_json to ensure proper datetime encoding
    json_text = manifest.model_dump_json(indent=2)
    path.write_text(json_text)
    return path


def load_manifest(path_or_root: Path | str) -> ProvenanceManifest:
    """Load and validate a manifest from a path or dataset root."""
    p = Path(path_or_root)
    manifest_path = p if p.name.endswith(".json") else get_default_manifest_path(p)
    text = manifest_path.read_text()
    return ProvenanceManifest.model_validate_json(text)
