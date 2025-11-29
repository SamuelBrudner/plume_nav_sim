"""Dataset registry definitions for curated plume data assets.

This module defines the schema for the plume data "zoo" along with a small set
of seed entries. Future downloader utilities can rely on the typed structure to
resolve cache locations, integrity expectations, and citation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# Schema and registry bookkeeping
REGISTRY_SCHEMA_VERSION = "1.0.0"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "plume_nav_sim" / "data_zoo"


class RegistryValidationError(ValueError):
    """Raised when the registry contains invalid or incomplete entries."""


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
    ingest: Optional["H5ToZarrSpec"] = None

    def cache_path(self, cache_root: Path = DEFAULT_CACHE_ROOT) -> Path:
        """Return the canonical cache path for this dataset version."""

        return cache_root / self.cache_subdir / self.version


@dataclass(frozen=True)
class H5ToZarrSpec:
    """Optional post-processing hook for HDF5 → Zarr ingestion."""

    dataset: str
    fps: float
    pixel_to_grid: Tuple[float, float]
    origin: Tuple[float, float]
    extent: Tuple[float, float]
    normalize: bool = True
    chunk_t: Optional[int] = None
    output_layout: str = "zarr"


DATASET_REGISTRY: Dict[str, DatasetRegistryEntry] = {
    "colorado_jet_v1": DatasetRegistryEntry(
        dataset_id="colorado_jet_v1",
        version="1.0.0",
        cache_subdir="zenodo_6538177",
        expected_root="a0004_nearbed_10cm_s.zarr",
        artifact=DatasetArtifact(
            url=(
                "https://zenodo.org/records/6538177/files/"
                "a0004_air_stationarySource_isokineticNearbedRelease_10cm_s.h5"
            ),
            checksum="0d712585798102e49b49fe9bcf41da33222fff1b889f4b772c038086eb171512",
            archive_type="none",
            layout="hdf5",
        ),
        metadata=DatasetMetadata(
            title="CU Boulder PLIF odor plume (a0004 near-bed, 10 cm/s)",
            description=(
                "150-frame acetone vapor plume measured via planar laser-induced "
                "fluorescence (PLIF) at 15 FPS. Near-bed isokinetic release at "
                "0.10 m/s with 406x216 px field of view (~300x160 mm), normalized "
                "concentration values, and embedded experiment metadata."
            ),
            citation=(
                "Connor, E. G., McHugh, M. K., & Crimaldi, J. P. (2018). "
                "Quantification of airborne odor plumes using planar laser-induced "
                "fluorescence. Experiments in Fluids, 59(9), 137. "
                "Data from Zenodo record 6538177."
            ),
            doi="10.5281/zenodo.6538177",
            license="CC-BY-4.0",
            contact="Prof. John Crimaldi <crimaldi@colorado.edu>",
        ),
        ingest=H5ToZarrSpec(
            dataset="/Plume Data/dataset_001",
            fps=15.0,
            pixel_to_grid=(159.84 / 216.0, 300.44 / 406.0),
            origin=(0.0, 0.0),
            extent=(159.84, 300.44),
            normalize=False,
            chunk_t=50,
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


def validate_registry(
    registry: Optional[Dict[str, DatasetRegistryEntry]] = None,
) -> None:
    """Validate registry entries for completeness and internal consistency."""

    entries = registry or DATASET_REGISTRY
    if not isinstance(entries, dict):
        raise RegistryValidationError("Registry must be a mapping of dataset entries")

    for key, entry in entries.items():
        if not isinstance(entry, DatasetRegistryEntry):
            raise RegistryValidationError(
                f"Registry entry for '{key}' is not a DatasetRegistryEntry"
            )
        if not entry.dataset_id or key != entry.dataset_id:
            raise RegistryValidationError(
                f"Registry key '{key}' must match entry dataset_id '{entry.dataset_id}'"
            )
        if not entry.version:
            raise RegistryValidationError(f"Dataset '{key}' is missing a version")
        if not entry.cache_subdir:
            raise RegistryValidationError(f"Dataset '{key}' is missing cache_subdir")
        if not entry.expected_root:
            raise RegistryValidationError(f"Dataset '{key}' is missing expected_root")

        artifact = entry.artifact
        if not artifact.url:
            raise RegistryValidationError(f"Dataset '{key}' is missing artifact.url")
        if not artifact.checksum:
            raise RegistryValidationError(
                f"Dataset '{key}' is missing artifact.checksum"
            )
        if not artifact.archive_type:
            raise RegistryValidationError(
                f"Dataset '{key}' is missing artifact.archive_type"
            )
        if not artifact.layout:
            raise RegistryValidationError(f"Dataset '{key}' is missing artifact.layout")

        metadata = entry.metadata
        if not metadata.title.strip():
            raise RegistryValidationError(f"Dataset '{key}' is missing metadata.title")
        if not metadata.description.strip():
            raise RegistryValidationError(
                f"Dataset '{key}' is missing metadata.description"
            )
        if not metadata.citation.strip():
            raise RegistryValidationError(
                f"Dataset '{key}' is missing metadata.citation"
            )
        if not metadata.license.strip():
            raise RegistryValidationError(
                f"Dataset '{key}' is missing metadata.license"
            )

        ingest = entry.ingest
        if ingest:
            if not ingest.dataset:
                raise RegistryValidationError(
                    f"Dataset '{key}' has an ingest spec but no source dataset path"
                )
            if ingest.fps <= 0:
                raise RegistryValidationError(
                    f"Dataset '{key}' ingest fps must be positive"
                )
            py, px = ingest.pixel_to_grid
            if py <= 0 or px <= 0:
                raise RegistryValidationError(
                    f"Dataset '{key}' ingest pixel_to_grid entries must be positive"
                )
            ey, ex = ingest.extent
            if ey <= 0 or ex <= 0:
                raise RegistryValidationError(
                    f"Dataset '{key}' ingest extent entries must be positive"
                )
            if not ingest.output_layout.strip():
                raise RegistryValidationError(
                    f"Dataset '{key}' ingest output_layout must be provided"
                )
            if ingest.output_layout.lower() not in {"zarr", "hdf5", "h5"}:
                raise RegistryValidationError(
                    f"Dataset '{key}' ingest output_layout '{ingest.output_layout}' "
                    "is not supported"
                )
