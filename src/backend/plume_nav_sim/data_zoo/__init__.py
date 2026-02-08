"""Curated plume data registry and related helpers."""

from __future__ import annotations

from .download import (
    ChecksumMismatchError,
    DatasetDownloadError,
    LayoutValidationError,
    ProvenanceSidecar,
    ensure_dataset_available,
    generate_provenance,
)
from .loader import load_plume
from .registry import (
    DATASET_REGISTRY,
    DEFAULT_CACHE_ROOT,
    REGISTRY_SCHEMA_VERSION,
    Creator,
    CrimaldiFluorescenceIngest,
    DatasetArtifact,
    DatasetMetadata,
    DatasetRegistryEntry,
    EmonetSmokeIngest,
    H5ToZarrSpec,
    RegistryValidationError,
    RelatedIdentifier,
    RigolliDNSIngest,
    SimulationMetadata,
    describe_dataset,
    get_dataset_registry,
    validate_registry,
)

__all__ = [
    # Registry
    "DATASET_REGISTRY",
    "DEFAULT_CACHE_ROOT",
    "REGISTRY_SCHEMA_VERSION",
    "DatasetArtifact",
    "DatasetMetadata",
    "DatasetRegistryEntry",
    "RegistryValidationError",
    # DataCite metadata
    "Creator",
    "RelatedIdentifier",
    "SimulationMetadata",
    # Ingest specs
    "CrimaldiFluorescenceIngest",
    "RigolliDNSIngest",
    "EmonetSmokeIngest",
    "H5ToZarrSpec",  # Backwards compat alias
    # Download
    "ChecksumMismatchError",
    "DatasetDownloadError",
    "LayoutValidationError",
    "ensure_dataset_available",
    # Provenance
    "ProvenanceSidecar",
    "generate_provenance",
    # Helpers
    "describe_dataset",
    "get_dataset_registry",
    "validate_registry",
    # Loader
    "load_plume",
]
