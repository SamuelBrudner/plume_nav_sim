"""Media I/O and dataset utilities (manifests, loaders).

This module houses helpers for working with external media-backed plume
datasets, such as video-derived Zarr stores. It provides a provenance
manifest schema and utilities to read/write it alongside datasets.
"""

from .manifest import (
    MANIFEST_FILENAME,
    ProvenanceManifest,
    build_provenance_manifest,
    get_default_manifest_path,
    load_manifest,
    write_manifest,
)
from .mapping import (
    DEFAULT_ROUNDING,
    DEFAULT_STEP_POLICY,
    ROUND_CEIL,
    ROUND_FLOOR,
    ROUND_NEAREST,
    STEP_POLICY_CLAMP,
    STEP_POLICY_INDEX,
    STEP_POLICY_WRAP,
    VideoTimebase,
    map_step_to_frame,
)

__all__ = [
    "MANIFEST_FILENAME",
    "ProvenanceManifest",
    "build_provenance_manifest",
    "get_default_manifest_path",
    "load_manifest",
    "write_manifest",
    # Mapping contract
    "VideoTimebase",
    "map_step_to_frame",
    "STEP_POLICY_INDEX",
    "STEP_POLICY_CLAMP",
    "STEP_POLICY_WRAP",
    "DEFAULT_STEP_POLICY",
    "ROUND_FLOOR",
    "ROUND_NEAREST",
    "ROUND_CEIL",
    "DEFAULT_ROUNDING",
]
