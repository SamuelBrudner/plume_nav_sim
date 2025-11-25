"""Curated plume data registry and related helpers."""

from __future__ import annotations

from .download import (
    ChecksumMismatchError,
    DatasetDownloadError,
    LayoutValidationError,
    ensure_dataset_available,
)
from .registry import (
    DATASET_REGISTRY,
    DEFAULT_CACHE_ROOT,
    REGISTRY_SCHEMA_VERSION,
    DatasetArtifact,
    DatasetMetadata,
    DatasetRegistryEntry,
    RegistryValidationError,
    describe_dataset,
    get_dataset_registry,
    validate_registry,
)

__all__ = [
    "DATASET_REGISTRY",
    "DEFAULT_CACHE_ROOT",
    "REGISTRY_SCHEMA_VERSION",
    "DatasetArtifact",
    "DatasetMetadata",
    "DatasetRegistryEntry",
    "RegistryValidationError",
    "ChecksumMismatchError",
    "DatasetDownloadError",
    "LayoutValidationError",
    "ensure_dataset_available",
    "describe_dataset",
    "get_dataset_registry",
    "validate_registry",
]
