"""Dataset registry definitions for curated plume data assets.

This module defines the schema for the plume data "zoo" along with a small set
of seed entries. Future downloader utilities can rely on the typed structure to
resolve cache locations, integrity expectations, and citation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

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
    ingest: Optional["IngestSpec"] = None

    def cache_path(self, cache_root: Path = DEFAULT_CACHE_ROOT) -> Path:
        """Return the canonical cache path for this dataset version."""

        return cache_root / self.cache_subdir / self.version


@dataclass(frozen=True)
class CrimaldiFluorescenceIngest:
    """Ingest spec for CU Boulder PLIF fluorescence datasets (Crimaldi lab).

    Handles the specific HDF5 structure used in Zenodo record 6538177 and similar
    PLIF plume movies from the Crimaldi lab.
    """

    dataset: str
    fps: float
    pixel_to_grid: Tuple[float, float]
    origin: Tuple[float, float]
    extent: Tuple[float, float]
    normalize: bool = True
    chunk_t: Optional[int] = None
    output_layout: str = "zarr"


@dataclass(frozen=True)
class RigolliDNSIngest:
    """Ingest spec for Rigolli et al. DNS turbulent plume datasets.

    Handles the specific MATLAB v7.3 structure with separate concentration and
    coordinate files from Zenodo record 15469831.
    """

    concentration_key: str  # Key in .mat file for concentration array
    coords_url: str  # URL to download coordinates.mat
    coords_checksum: str  # MD5 checksum for coordinates file
    x_key: str = "x"  # Key for X coordinates in coords file
    y_key: str = "y"  # Key for Y coordinates in coords file
    fps: Optional[float] = None  # Frames per second (None if unknown)
    normalize: bool = False  # Whether to normalize concentration to [0,1]
    chunk_t: Optional[int] = 100  # Chunk size for time dimension
    output_layout: str = "zarr"


@dataclass(frozen=True)
class EmonetSmokeIngest:
    """Ingest spec for Emonet lab smoke plume video datasets.

    Handles the flyWalk MATLAB video format from Dryad record 10.5061/dryad.4j0zpc87z.
    Smoke intensity serves as proxy for odor concentration.
    """

    frames_key: str = "frames"  # Key in .mat file for video frames array
    metadata_url: str = ""  # URL to download metadata .mat file
    metadata_checksum: str = ""  # MD5 checksum for metadata file
    fps: float = 90.0  # Frame rate (90 Hz for intermittent plume experiments)
    px_per_mm: float = 1.0  # Pixel to mm conversion factor
    arena_size_mm: Tuple[float, float] = (300.0, 180.0)  # Arena dimensions (x, y) in mm
    normalize: bool = True  # Normalize smoke intensity to [0,1]
    chunk_t: Optional[int] = 100  # Chunk size for time dimension
    output_layout: str = "zarr"


# Type alias for ingest specifications
IngestSpec = Union[CrimaldiFluorescenceIngest, RigolliDNSIngest, EmonetSmokeIngest]

# Backwards compatibility aliases
H5ToZarrSpec = CrimaldiFluorescenceIngest
MatToZarrSpec = RigolliDNSIngest


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
        ingest=CrimaldiFluorescenceIngest(
            dataset="/Plume Data/dataset_001",
            fps=15.0,
            pixel_to_grid=(159.84 / 216.0, 300.44 / 406.0),
            origin=(0.0, 0.0),
            extent=(159.84, 300.44),
            normalize=False,
            chunk_t=50,
        ),
    ),
    "rigolli_dns_nose_v1": DatasetRegistryEntry(
        dataset_id="rigolli_dns_nose_v1",
        version="1.0.0",
        cache_subdir="zenodo_15469831",
        expected_root="rigolli_nose.zarr",
        artifact=DatasetArtifact(
            url="https://zenodo.org/records/15469831/files/nose_data.mat?download=1",
            checksum="ce390644cf1b075fac06360a6dbe1881",
            checksum_type="md5",
            archive_type="none",
            layout="hdf5",
        ),
        metadata=DatasetMetadata(
            title="Rigolli et al. DNS turbulent plume - nose level (z~50cm)",
            description=(
                "2D time-series of odor concentration at nose height (~50 cm) from 3D "
                "direct numerical simulation of turbulent channel flow. Auto-ingested "
                "from MATLAB v7.3 format with spatial coordinates into standardized Zarr. "
                "Source: ~6.8 GB .mat file."
            ),
            citation=(
                "Rigolli, N., Reddy, G., Seminara, A., & Vergassola, M. (2022). "
                "Alternation emerges as a multi-modal strategy for turbulent odor "
                "navigation. eLife, 11, e76989. https://doi.org/10.7554/eLife.76989"
            ),
            doi="10.5281/zenodo.15469831",
            license="CC-BY-4.0",
            contact="Nicola Rigolli <nicola.rigolli@edu.unige.it>",
        ),
        ingest=RigolliDNSIngest(
            concentration_key="nose_data",
            coords_url="https://zenodo.org/records/15469831/files/coordinates.mat?download=1",
            coords_checksum="1e264919fb96574ecfb1981eaccc7d20",
            x_key="x",
            y_key="y",
            normalize=False,
            chunk_t=100,
        ),
    ),
    "rigolli_dns_ground_v1": DatasetRegistryEntry(
        dataset_id="rigolli_dns_ground_v1",
        version="1.0.0",
        cache_subdir="zenodo_15469831",
        expected_root="rigolli_ground.zarr",
        artifact=DatasetArtifact(
            url="https://zenodo.org/records/15469831/files/ground_data.mat?download=1",
            checksum="5e674b6a30890f2de0dc720809baa936",
            checksum_type="md5",
            archive_type="none",
            layout="hdf5",
        ),
        metadata=DatasetMetadata(
            title="Rigolli et al. DNS turbulent plume - ground level (z=0)",
            description=(
                "2D time-series of odor concentration at ground level (z=0) from 3D "
                "direct numerical simulation of turbulent channel flow. Auto-ingested "
                "from MATLAB v7.3 format with spatial coordinates into standardized Zarr. "
                "Source: ~6.8 GB .mat file."
            ),
            citation=(
                "Rigolli, N., Reddy, G., Seminara, A., & Vergassola, M. (2022). "
                "Alternation emerges as a multi-modal strategy for turbulent odor "
                "navigation. eLife, 11, e76989. https://doi.org/10.7554/eLife.76989"
            ),
            doi="10.5281/zenodo.15469831",
            license="CC-BY-4.0",
            contact="Nicola Rigolli <nicola.rigolli@edu.unige.it>",
        ),
        ingest=RigolliDNSIngest(
            concentration_key="ground_data",
            coords_url="https://zenodo.org/records/15469831/files/coordinates.mat?download=1",
            coords_checksum="1e264919fb96574ecfb1981eaccc7d20",
            x_key="x",
            y_key="y",
            normalize=False,
            chunk_t=100,
        ),
    ),
    "emonet_smoke_v1": DatasetRegistryEntry(
        dataset_id="emonet_smoke_v1",
        version="1.0.0",
        cache_subdir="dryad_4j0zpc87z",
        expected_root="emonet_smoke.zarr",
        artifact=DatasetArtifact(
            url="https://datadryad.org/stash/downloads/file_stream/852661",
            # NOTE: Checksum needs verification - file is 28.73 GB
            checksum="placeholder_needs_computation",
            checksum_type="md5",
            archive_type="none",
            layout="hdf5",
        ),
        metadata=DatasetMetadata(
            title="Emonet lab smoke plume video (walking Drosophila study)",
            description=(
                "High-speed (90 Hz) smoke plume video from wind tunnel walking arena. "
                "Smoke intensity serves as proxy for odor concentration. 300×180 mm arena, "
                "~29 GB raw frames. Auto-ingested to standardized Zarr. "
                "WARNING: Large download!"
            ),
            citation=(
                "Demir, M., Kadakia, N., Anderson, H. D., Clark, D. A., & Emonet, T. (2020). "
                "Walking Drosophila navigate complex plumes using stochastic decisions "
                "biased by the timing of odor encounters. eLife, 9, e57524. "
                "https://doi.org/10.7554/eLife.57524"
            ),
            doi="10.5061/dryad.4j0zpc87z",
            license="CC0-1.0",
            contact="Thierry Emonet <thierry.emonet@yale.edu>",
        ),
        ingest=EmonetSmokeIngest(
            frames_key="frames",
            metadata_url="https://datadryad.org/stash/downloads/file_stream/852662",
            metadata_checksum="placeholder_needs_computation",
            fps=90.0,
            arena_size_mm=(300.0, 180.0),
            normalize=True,
            chunk_t=100,
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
            if not ingest.output_layout.strip():
                raise RegistryValidationError(
                    f"Dataset '{key}' ingest output_layout must be provided"
                )
            if ingest.output_layout.lower() not in {"zarr", "hdf5", "h5"}:
                raise RegistryValidationError(
                    f"Dataset '{key}' ingest output_layout '{ingest.output_layout}' "
                    "is not supported"
                )
            # Validate CrimaldiFluorescenceIngest-specific fields
            if isinstance(ingest, CrimaldiFluorescenceIngest):
                if not ingest.dataset:
                    raise RegistryValidationError(
                        f"Dataset '{key}' CrimaldiFluorescenceIngest missing dataset path"
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
            # Validate RigolliDNSIngest-specific fields
            elif isinstance(ingest, RigolliDNSIngest):
                if not ingest.concentration_key:
                    raise RegistryValidationError(
                        f"Dataset '{key}' RigolliDNSIngest missing concentration_key"
                    )
                if not ingest.coords_url:
                    raise RegistryValidationError(
                        f"Dataset '{key}' RigolliDNSIngest missing coords_url"
                    )
                if not ingest.coords_checksum:
                    raise RegistryValidationError(
                        f"Dataset '{key}' RigolliDNSIngest missing coords_checksum"
                    )
            # Validate EmonetSmokeIngest-specific fields
            elif isinstance(ingest, EmonetSmokeIngest):
                if ingest.fps <= 0:
                    raise RegistryValidationError(
                        f"Dataset '{key}' EmonetSmokeIngest fps must be positive"
                    )
                ax, ay = ingest.arena_size_mm
                if ax <= 0 or ay <= 0:
                    raise RegistryValidationError(
                        f"Dataset '{key}' EmonetSmokeIngest arena_size_mm must be positive"
                    )
