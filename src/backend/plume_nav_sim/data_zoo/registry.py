"""Dataset registry definitions for curated plume data assets.

This module defines the schema for the plume data "zoo" along with a small set
of seed entries. Future downloader utilities can rely on the typed structure to
resolve cache locations, integrity expectations, and citation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# Schema and registry bookkeeping
REGISTRY_SCHEMA_VERSION = "1.0.0"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "plume_nav_sim" / "data_zoo"


@dataclass(frozen=True)
class DatasetArtifact:
    """Source artifact details for a dataset entry."""

    url: str
    checksum: str
    checksum_type: str = "sha256"
    archive_type: str = "zip"  # e.g., "zip", "tar.gz"
    archive_member: Optional[str] = None  # For archives containing multiple files
    layout: str = "zarr"  # Expected on-disk structure after unpack


@dataclass(frozen=True)
class DatasetMetadata:
    """Human-facing metadata used for attribution and documentation."""

    title: str
    description: str
    citation: str
    license: str
    doi: Optional[str] = None
    contact: Optional[str] = None


@dataclass(frozen=True)
class DatasetRegistryEntry:
    """Complete registry entry covering artifacts, layout, and cache hints."""

    dataset_id: str
    version: str
    artifact: DatasetArtifact
    cache_subdir: str
    expected_root: str
    metadata: DatasetMetadata

    def cache_path(self, cache_root: Path = DEFAULT_CACHE_ROOT) -> Path:
        """Return the canonical cache path for this dataset version."""

        return cache_root / self.cache_subdir / self.version


DATASET_REGISTRY: Dict[str, DatasetRegistryEntry] = {
    "colorado_jet_v1": DatasetRegistryEntry(
        dataset_id="colorado_jet_v1",
        version="1.0.0",
        cache_subdir="colorado_jet",
        expected_root="colorado_jet_v1.zarr",
        artifact=DatasetArtifact(
            url="https://zenodo.org/record/1234567/files/colorado_jet_v1.zarr.zip",
            checksum="4c1ba02f650b40d58c8d0edb2ed7c2f8b2e74c6a7a7c316a987f3ab5c2c7e428",
            archive_type="zip",
            layout="zarr",
        ),
        metadata=DatasetMetadata(
            title="Colorado wind-tunnel jet plume (baseline)",
            description=(
                "Steady jet dispersion with laminar inflow boundary conditions, "
                "128x128 spatial grid, 2000 time steps; emitted as a consolidated "
                "Zarr store."
            ),
            citation="Doe et al. 2023, Journal of Field Robotics (supplemental plume dataset)",
            doi="10.1234/zenodo.1234567",
            license="CC-BY-4.0",
            contact="data-curation@example.org",
        ),
    ),
    "moffett_field_dispersion_v0": DatasetRegistryEntry(
        dataset_id="moffett_field_dispersion_v0",
        version="0.9.0",
        cache_subdir="moffett_field",
        expected_root="moffett_field_dispersion.h5",
        artifact=DatasetArtifact(
            url="https://huggingface.co/datasets/example-lab/plume-nav-sim/resolve/main/moffett_field_dispersion_v0.tar.gz",
            checksum="5d3c1af4f8aa31c6c0b51c0a1d6b703fb2d9a8d664e0b96c4e9c2dabba87c2d1",
            archive_type="tar.gz",
            archive_member="plume_movie.h5",
            layout="hdf5",
        ),
        metadata=DatasetMetadata(
            title="Moffett Field open-air dispersion trial (alpha)",
            description=(
                "Tracer-release field experiment subset with 64x64 spatial grid and "
                "downsampled 10 Hz sampling; packaged as HDF5 movie with accompanying "
                "attrs for FPS and spatial scaling."
            ),
            citation="Acme Environmental Lab 2022, Internal field release report",
            doi=None,
            license="CC-BY-NC-4.0",
            contact="plume-datasets@example.org",
        ),
    ),
}


def get_dataset_registry() -> Dict[str, DatasetRegistryEntry]:
    """Return the registry mapping for downstream consumers."""

    return DATASET_REGISTRY


def describe_dataset(dataset_id: str) -> DatasetRegistryEntry:
    """Fetch a dataset entry or raise a keyed error for clearer caller handling."""

    if dataset_id not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset id: {dataset_id}")
    return DATASET_REGISTRY[dataset_id]
