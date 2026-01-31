from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from plume_nav_sim import PACKAGE_VERSION

# Public constants
MANIFEST_FILENAME = "manifest.json"


class SystemEnv(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hostname: Optional[str] = None
    platform: Optional[str] = None
    python_version: Optional[str] = None


class ProvenanceManifest(BaseModel):
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
