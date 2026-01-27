from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# Schema and registry bookkeeping
REGISTRY_SCHEMA_VERSION = "1.0.0"

# Cache root: configurable via environment variable for external storage
_ENV_CACHE_ROOT = os.environ.get("PLUME_DATA_ZOO_CACHE")
DEFAULT_CACHE_ROOT = (
    Path(_ENV_CACHE_ROOT)
    if _ENV_CACHE_ROOT
    else Path.home() / ".cache" / "plume_nav_sim" / "data_zoo"
)


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
class Creator:
    """DataCite-compatible creator/author information."""

    name: str  # Format: "Family, Given" or organization name
    name_type: str = "Personal"  # "Personal" or "Organizational"
    orcid: Optional[str] = None
    affiliation: Optional[str] = None

    def to_datacite(self) -> Dict:
        """Export to DataCite JSON format."""
        result: Dict = {"name": self.name, "nameType": self.name_type}
        if self.orcid:
            result["nameIdentifiers"] = [
                {"nameIdentifier": self.orcid, "nameIdentifierScheme": "ORCID"}
            ]
        if self.affiliation:
            result["affiliation"] = [{"name": self.affiliation}]
        return result


@dataclass(frozen=True)
class RelatedIdentifier:
    """DataCite-compatible related identifier for provenance tracking."""

    identifier: str
    relation_type: str  # e.g., "IsDerivedFrom", "IsSupplementTo", "Cites"
    identifier_type: str = "DOI"  # "DOI", "URL", "ARK", etc.

    def to_datacite(self) -> Dict:
        """Export to DataCite JSON format."""
        return {
            "relatedIdentifier": self.identifier,
            "relatedIdentifierType": self.identifier_type,
            "relationType": self.relation_type,
        }


@dataclass(frozen=True)
class DatasetMetadata:
    # DataCite required fields
    title: str
    creators: Tuple[Creator, ...] = ()
    publisher: str = "plume-nav-sim Data Zoo"
    publication_year: Optional[int] = None
    resource_type: str = "Dataset"

    # DataCite recommended fields
    description: str = ""
    license: str = ""  # Maps to DataCite "rights"
    subjects: Tuple[str, ...] = ()  # Keywords
    related_identifiers: Tuple[RelatedIdentifier, ...] = ()

    # DataCite optional fields
    doi: Optional[str] = None
    version: Optional[str] = None
    language: str = "en"
    formats: Tuple[str, ...] = ("application/x-zarr",)

    # Legacy / convenience fields
    citation: str = ""  # Human-readable citation string
    contact: Optional[str] = None

    def to_datacite(self) -> Dict:
        """Export to DataCite JSON format for Zenodo/repository upload."""
        result: Dict = {
            "titles": [{"title": self.title}],
            "publisher": self.publisher,
            "resourceType": {"resourceTypeGeneral": self.resource_type},
            "language": self.language,
        }

        if self.creators:
            result["creators"] = [c.to_datacite() for c in self.creators]

        if self.publication_year:
            result["publicationYear"] = self.publication_year

        if self.description:
            result["descriptions"] = [
                {"description": self.description, "descriptionType": "Abstract"}
            ]

        if self.license:
            result["rightsList"] = [{"rights": self.license}]

        if self.subjects:
            result["subjects"] = [{"subject": s} for s in self.subjects]

        if self.related_identifiers:
            result["relatedIdentifiers"] = [
                ri.to_datacite() for ri in self.related_identifiers
            ]

        if self.doi:
            result["identifiers"] = [{"identifier": self.doi, "identifierType": "DOI"}]

        if self.version:
            result["version"] = self.version

        if self.formats:
            result["formats"] = list(self.formats)

        return result

    def generate_citation(self) -> str:
        """Generate a citation string from structured metadata."""
        if self.citation:
            return self.citation
        if not self.creators:
            return f"{self.title}. {self.publisher}."

        # Format authors
        if len(self.creators) == 1:
            authors = self.creators[0].name
        elif len(self.creators) == 2:
            authors = f"{self.creators[0].name} & {self.creators[1].name}"
        else:
            authors = f"{self.creators[0].name} et al."

        year = f" ({self.publication_year})" if self.publication_year else ""
        doi_str = f" https://doi.org/{self.doi}" if self.doi else ""

        return f"{authors}{year}. {self.title}. {self.publisher}.{doi_str}"


@dataclass(frozen=True)
class SimulationMetadata(DatasetMetadata):
    # Software provenance
    software_name: str = "plume-nav-sim"
    software_version: Optional[str] = None
    software_doi: Optional[str] = None  # e.g., Zenodo DOI for the software release

    # Simulation parameters
    config_hash: Optional[str] = None  # Hash of Hydra/config used
    random_seed: Optional[int] = None
    parameters: Tuple[Tuple[str, str], ...] = ()  # Key-value pairs

    # Computational context
    runtime_seconds: Optional[float] = None
    platform: Optional[str] = None  # e.g., "Linux-5.4.0-x86_64"
    generated_at: Optional[str] = None  # ISO 8601 timestamp

    def __post_init__(self) -> None:
        # Override resource_type for simulations
        if self.resource_type == "Dataset":
            object.__setattr__(self, "resource_type", "Dataset/Simulation")

    def to_datacite(self) -> Dict:
        """Export to DataCite JSON with simulation-specific extensions."""
        result = super().to_datacite()

        # Add software as contributor
        if self.software_name:
            result.setdefault("contributors", []).append(
                {
                    "name": self.software_name,
                    "nameType": "Organizational",
                    "contributorType": "Producer",
                }
            )

        # Add software DOI as related identifier
        if self.software_doi:
            result.setdefault("relatedIdentifiers", []).append(
                {
                    "relatedIdentifier": self.software_doi,
                    "relatedIdentifierType": "DOI",
                    "relationType": "IsCompiledBy",
                }
            )

        # Add simulation parameters as additional descriptions
        if self.parameters or self.random_seed is not None or self.config_hash:
            param_lines = []
            if self.config_hash:
                param_lines.append(f"config_hash: {self.config_hash}")
            if self.random_seed is not None:
                param_lines.append(f"random_seed: {self.random_seed}")
            for key, val in self.parameters:
                param_lines.append(f"{key}: {val}")

            result.setdefault("descriptions", []).append(
                {
                    "description": "Simulation parameters:\n" + "\n".join(param_lines),
                    "descriptionType": "TechnicalInfo",
                }
            )

        return result

    @classmethod
    def from_config(
        cls,
        title: str,
        creator_name: str,
        config: Dict,
        seed: Optional[int] = None,
        software_version: Optional[str] = None,
        **kwargs,
    ) -> "SimulationMetadata":
        import hashlib
        import json
        from datetime import datetime, timezone

        # Hash the config
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

        # Extract key parameters (flatten top-level)
        params = tuple(
            (str(k), str(v))
            for k, v in config.items()
            if not isinstance(v, dict) and v is not None
        )

        return cls(
            title=title,
            creators=(Creator(name=creator_name),),
            publisher="plume-nav-sim",
            publication_year=datetime.now().year,
            resource_type="Dataset/Simulation",
            software_version=software_version,
            config_hash=config_hash,
            random_seed=seed,
            parameters=params,
            generated_at=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )


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
    dataset: str
    fps: float
    pixel_to_grid: Tuple[float, float]
    origin: Tuple[float, float]
    extent: Tuple[float, float]
    source_location_px: Optional[Tuple[int, int]] = None
    normalize: bool = True
    chunk_t: Optional[int] = None
    output_layout: str = "zarr"


@dataclass(frozen=True)
class RigolliDNSIngest:
    concentration_key: str  # Key in .mat file for concentration array
    coords_url: str  # URL to download coordinates.mat
    coords_checksum: str  # MD5 checksum for coordinates file
    x_key: str = "x"  # Key for X coordinates in coords file
    y_key: str = "y"  # Key for Y coordinates in coords file
    source_location_px: Optional[Tuple[int, int]] = None
    fps: Optional[float] = None  # Frames per second (None if unknown)
    normalize: bool = False  # Whether to normalize concentration to [0,1]
    chunk_t: Optional[int] = 100  # Chunk size for time dimension
    output_layout: str = "zarr"


@dataclass(frozen=True)
class EmonetSmokeIngest:
    frames_key: str = "frames"  # Key in .mat file for video frames array
    metadata_url: str = ""  # URL to download metadata .mat file
    metadata_checksum: str = ""  # MD5 checksum for metadata file
    fps: float = 90.0  # Frame rate (90 Hz for intermittent plume experiments)
    px_per_mm: float = 1.0  # Pixel to mm conversion factor
    arena_size_mm: Tuple[float, float] = (300.0, 180.0)  # Arena dimensions (x, y) in mm
    source_location_px: Optional[Tuple[int, int]] = None
    background_subtract: bool = True
    background_n_frames: int = 200
    auto_trim_start: bool = True
    skip_initial_frames: int = 0
    trim_abs_threshold: Optional[float] = None
    trim_sigma: float = 5.0
    trim_consecutive: int = 10
    trim_max_scan: int = 5000
    auto_trim_end: bool = False
    end_abs_threshold: Optional[float] = None
    end_sigma: float = 5.0
    end_consecutive: int = 900
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
        cache_subdir="zenodo_4971113",
        expected_root="a0004_nearbed_10cm_s.zarr",
        artifact=DatasetArtifact(
            url=(
                "https://zenodo.org/api/records/4971113/files/"
                "10302017_10cms_bounded_2.h5/content"
            ),
            checksum="cf3cadd8a9b53c2a3f6d3aca37f71a62",
            checksum_type="md5",
            archive_type="none",
            layout="hdf5",
        ),
        metadata=DatasetMetadata(
            title="CU Boulder PLIF odor plume (a0004 near-bed, 10 cm/s)",
            creators=(
                Creator(
                    name="Connor, Esteban G.",
                    affiliation="University of Colorado Boulder",
                ),
                Creator(
                    name="McHugh, Matthew K.",
                    affiliation="University of Colorado Boulder",
                ),
                Creator(
                    name="Crimaldi, John P.",
                    orcid="https://orcid.org/0000-0002-5731-792X",
                    affiliation="University of Colorado Boulder",
                ),
            ),
            publisher="Dryad",
            publication_year=2018,
            description=(
                "150-frame acetone vapor plume measured via planar laser-induced "
                "fluorescence (PLIF) at 15 FPS. Near-bed isokinetic release at "
                "0.10 m/s with 406x216 px field of view (~300x160 mm), normalized "
                "concentration values, and embedded experiment metadata. "
                "Source dataset DOI: 10.5061/dryad.g27mq71 (mirrored on Zenodo record 4971113)."
            ),
            subjects=("plume", "PLIF", "odor", "turbulence", "fluorescence"),
            related_identifiers=(
                RelatedIdentifier(
                    identifier="10.1007/s00348-018-2591-3",
                    relation_type="IsSupplementTo",
                ),
                RelatedIdentifier(
                    identifier="https://zenodo.org/records/4971113",
                    relation_type="IsIdenticalTo",
                    identifier_type="URL",
                ),
            ),
            doi="10.5061/dryad.g27mq71",
            license="CC-BY-4.0",
            citation=(
                "Connor, E. G., McHugh, M. K., & Crimaldi, J. P. (2018). "
                "Quantification of airborne odor plumes using planar laser-induced "
                "fluorescence. Experiments in Fluids, 59(9), 137. "
                "Dryad dataset: https://doi.org/10.5061/dryad.g27mq71 "
                "(download mirror: Zenodo record 4971113)."
            ),
            contact="Prof. John Crimaldi <crimaldi@colorado.edu>",
        ),
        ingest=CrimaldiFluorescenceIngest(
            dataset="dataset2",
            fps=15.0,
            pixel_to_grid=(159.84 / 406.0, 300.44 / 216.0),
            origin=(0.0, 0.0),
            extent=(159.84, 300.44),
            source_location_px=(107, 0),
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
            creators=(
                Creator(
                    name="Rigolli, Nicola",
                    affiliation="University of Genoa",
                ),
                Creator(
                    name="Reddy, Gautam",
                    orcid="https://orcid.org/0000-0002-8254-9034",
                    affiliation="Harvard University",
                ),
                Creator(
                    name="Seminara, Agnese",
                    orcid="https://orcid.org/0000-0002-6966-1032",
                    affiliation="University of Genoa",
                ),
                Creator(
                    name="Vergassola, Massimo",
                    orcid="https://orcid.org/0000-0002-4163-1090",
                    affiliation="École Normale Supérieure",
                ),
            ),
            publisher="Zenodo",
            publication_year=2022,
            description=(
                "2D time-series of odor concentration at nose height (~50 cm) from 3D "
                "direct numerical simulation of turbulent channel flow. Auto-ingested "
                "from MATLAB v7.3 format with spatial coordinates into standardized Zarr. "
                "Source: ~6.8 GB .mat file."
            ),
            subjects=("DNS", "turbulence", "odor", "plume", "navigation", "simulation"),
            related_identifiers=(
                RelatedIdentifier(
                    identifier="10.7554/eLife.76989",
                    relation_type="IsSupplementTo",
                ),
            ),
            doi="10.5281/zenodo.15469831",
            license="CC-BY-4.0",
            citation=(
                "Rigolli, N., Reddy, G., Seminara, A., & Vergassola, M. (2022). "
                "Alternation emerges as a multi-modal strategy for turbulent odor "
                "navigation. eLife, 11, e76989. https://doi.org/10.7554/eLife.76989"
            ),
            contact="Nicola Rigolli <nicola.rigolli@edu.unige.it>",
        ),
        ingest=RigolliDNSIngest(
            concentration_key="nose",
            coords_url="https://zenodo.org/records/15469831/files/coordinates.mat?download=1",
            coords_checksum="1e264919fb96574ecfb1981eaccc7d20",
            x_key="X",
            y_key="Y",
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
            creators=(
                Creator(
                    name="Rigolli, Nicola",
                    affiliation="University of Genoa",
                ),
                Creator(
                    name="Reddy, Gautam",
                    orcid="https://orcid.org/0000-0002-8254-9034",
                    affiliation="Harvard University",
                ),
                Creator(
                    name="Seminara, Agnese",
                    orcid="https://orcid.org/0000-0002-6966-1032",
                    affiliation="University of Genoa",
                ),
                Creator(
                    name="Vergassola, Massimo",
                    orcid="https://orcid.org/0000-0002-4163-1090",
                    affiliation="École Normale Supérieure",
                ),
            ),
            publisher="Zenodo",
            publication_year=2022,
            description=(
                "2D time-series of odor concentration at ground level (z=0) from 3D "
                "direct numerical simulation of turbulent channel flow. Auto-ingested "
                "from MATLAB v7.3 format with spatial coordinates into standardized Zarr. "
                "Source: ~6.8 GB .mat file."
            ),
            subjects=("DNS", "turbulence", "odor", "plume", "navigation", "simulation"),
            related_identifiers=(
                RelatedIdentifier(
                    identifier="10.7554/eLife.76989",
                    relation_type="IsSupplementTo",
                ),
            ),
            doi="10.5281/zenodo.15469831",
            license="CC-BY-4.0",
            citation=(
                "Rigolli, N., Reddy, G., Seminara, A., & Vergassola, M. (2022). "
                "Alternation emerges as a multi-modal strategy for turbulent odor "
                "navigation. eLife, 11, e76989. https://doi.org/10.7554/eLife.76989"
            ),
            contact="Nicola Rigolli <nicola.rigolli@edu.unige.it>",
        ),
        ingest=RigolliDNSIngest(
            concentration_key="ground",
            coords_url="https://zenodo.org/records/15469831/files/coordinates.mat?download=1",
            coords_checksum="1e264919fb96574ecfb1981eaccc7d20",
            x_key="X",
            y_key="Y",
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
            url="https://datadryad.org/api/v2/files/852661/download",
            checksum="6f87df24e4a5146c49c56979aca0fd78",
            checksum_type="md5",
            archive_type="none",
            layout="hdf5",
        ),
        metadata=DatasetMetadata(
            title="Emonet lab smoke plume video (walking Drosophila study)",
            creators=(
                Creator(
                    name="Demir, Mahmut",
                    affiliation="Yale University",
                ),
                Creator(
                    name="Kadakia, Nirag",
                    orcid="https://orcid.org/0000-0003-0538-7539",
                    affiliation="Yale University",
                ),
                Creator(
                    name="Anderson, Hope D.",
                    affiliation="Yale University",
                ),
                Creator(
                    name="Clark, Damon A.",
                    orcid="https://orcid.org/0000-0001-5698-9094",
                    affiliation="Yale University",
                ),
                Creator(
                    name="Emonet, Thierry",
                    orcid="https://orcid.org/0000-0002-7746-3527",
                    affiliation="Yale University",
                ),
            ),
            publisher="Dryad",
            publication_year=2020,
            description=(
                "High-speed (90 Hz) smoke plume video from wind tunnel walking arena. "
                "Smoke intensity serves as proxy for odor concentration. 300×180 mm arena, "
                "~29 GB raw frames. Auto-ingested to standardized Zarr. "
                "WARNING: Large download!"
            ),
            subjects=(
                "Drosophila",
                "plume",
                "navigation",
                "smoke",
                "olfaction",
                "behavior",
            ),
            related_identifiers=(
                RelatedIdentifier(
                    identifier="10.7554/eLife.57524",
                    relation_type="IsSupplementTo",
                ),
            ),
            doi="10.5061/dryad.4j0zpc87z",
            license="CC0-1.0",
            citation=(
                "Demir, M., Kadakia, N., Anderson, H. D., Clark, D. A., & Emonet, T. (2020). "
                "Walking Drosophila navigate complex plumes using stochastic decisions "
                "biased by the timing of odor encounters. eLife, 9, e57524. "
                "https://doi.org/10.7554/eLife.57524"
            ),
            contact="Thierry Emonet <thierry.emonet@yale.edu>",
        ),
        ingest=EmonetSmokeIngest(
            frames_key="frames",
            metadata_url="https://datadryad.org/api/v2/files/852662/download",
            metadata_checksum="",  # Optional - metadata parsing is best-effort
            fps=90.0,  # 90 Hz camera (can be refined from metadata.p.framerate)
            px_per_mm=2048.0 / 300.0,  # ~6.83 px/mm based on 300mm arena width
            arena_size_mm=(300.0, 180.0),  # 2048x1200 px arena
            background_subtract=True,
            background_n_frames=200,
            auto_trim_start=True,
            trim_abs_threshold=1.3,
            trim_sigma=5.0,
            trim_consecutive=10,
            trim_max_scan=5000,
            normalize=True,
            chunk_t=100,
        ),
    ),
    "emonet_smoke_trimmed_v1": DatasetRegistryEntry(
        dataset_id="emonet_smoke_trimmed_v1",
        version="1.0.0",
        cache_subdir="dryad_4j0zpc87z",
        expected_root="emonet_smoke_trimmed.zarr",
        artifact=DatasetArtifact(
            url="https://datadryad.org/api/v2/files/852661/download",
            checksum="6f87df24e4a5146c49c56979aca0fd78",
            checksum_type="md5",
            archive_type="none",
            layout="hdf5",
        ),
        metadata=DatasetMetadata(
            title="Emonet lab smoke plume video (walking Drosophila study) - tail-trimmed",
            creators=(
                Creator(
                    name="Demir, Mahmut",
                    affiliation="Yale University",
                ),
                Creator(
                    name="Kadakia, Nirag",
                    orcid="https://orcid.org/0000-0003-0538-7539",
                    affiliation="Yale University",
                ),
                Creator(
                    name="Anderson, Hope D.",
                    affiliation="Yale University",
                ),
                Creator(
                    name="Clark, Damon A.",
                    orcid="https://orcid.org/0000-0001-5698-9094",
                    affiliation="Yale University",
                ),
                Creator(
                    name="Emonet, Thierry",
                    orcid="https://orcid.org/0000-0002-7746-3527",
                    affiliation="Yale University",
                ),
            ),
            publisher="Dryad",
            publication_year=2020,
            description=(
                "High-speed (90 Hz) smoke plume video from wind tunnel walking arena. "
                "Smoke intensity serves as proxy for odor concentration. 300×180 mm arena, "
                "~29 GB raw frames. This entry trims the low-intensity tail using "
                "background-subtracted mean intensity over time and then ingests to standardized Zarr. "
                "WARNING: Large download!"
            ),
            subjects=(
                "Drosophila",
                "plume",
                "navigation",
                "smoke",
                "olfaction",
                "behavior",
            ),
            related_identifiers=(
                RelatedIdentifier(
                    identifier="10.7554/eLife.57524",
                    relation_type="IsSupplementTo",
                ),
            ),
            doi="10.5061/dryad.4j0zpc87z",
            license="CC0-1.0",
            citation=(
                "Demir, M., Kadakia, N., Anderson, H. D., Clark, D. A., & Emonet, T. (2020). "
                "Walking Drosophila navigate complex plumes using stochastic decisions "
                "biased by the timing of odor encounters. eLife, 9, e57524. "
                "https://doi.org/10.7554/eLife.57524"
            ),
            contact="Thierry Emonet <thierry.emonet@yale.edu>",
        ),
        ingest=EmonetSmokeIngest(
            frames_key="frames",
            metadata_url="https://datadryad.org/api/v2/files/852662/download",
            metadata_checksum="",
            fps=90.0,
            px_per_mm=2048.0 / 300.0,
            arena_size_mm=(300.0, 180.0),
            background_subtract=True,
            background_n_frames=200,
            auto_trim_start=True,
            trim_abs_threshold=1.3,
            trim_sigma=5.0,
            trim_consecutive=10,
            trim_max_scan=5000,
            auto_trim_end=True,
            end_abs_threshold=1.3,
            end_sigma=5.0,
            end_consecutive=900,
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


def _validate_registry_mapping(
    entries: Optional[Dict[str, DatasetRegistryEntry]],
) -> Dict[str, DatasetRegistryEntry]:
    registry = entries or DATASET_REGISTRY
    if not isinstance(registry, dict):
        raise RegistryValidationError("Registry must be a mapping of dataset entries")
    return registry


def _validate_entry_identity(key: str, entry: DatasetRegistryEntry) -> None:
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


def _validate_artifact(key: str, artifact: DatasetArtifact) -> None:
    if not artifact.url:
        raise RegistryValidationError(f"Dataset '{key}' is missing artifact.url")
    if not artifact.checksum:
        raise RegistryValidationError(f"Dataset '{key}' is missing artifact.checksum")
    if not artifact.archive_type:
        raise RegistryValidationError(
            f"Dataset '{key}' is missing artifact.archive_type"
        )
    if not artifact.layout:
        raise RegistryValidationError(f"Dataset '{key}' is missing artifact.layout")


def _validate_metadata(key: str, metadata: DatasetMetadata) -> None:
    if not metadata.title.strip():
        raise RegistryValidationError(f"Dataset '{key}' is missing metadata.title")
    if not metadata.description.strip():
        raise RegistryValidationError(
            f"Dataset '{key}' is missing metadata.description"
        )
    # DataCite requires either structured creators OR legacy citation
    if not metadata.creators and not metadata.citation.strip():
        raise RegistryValidationError(
            f"Dataset '{key}' must have either creators or citation"
        )
    if not metadata.license.strip():
        raise RegistryValidationError(f"Dataset '{key}' is missing metadata.license")


def _validate_creators(key: str, creators: tuple[Creator, ...]) -> None:
    for i, creator in enumerate(creators):
        if not creator.name.strip():
            raise RegistryValidationError(f"Dataset '{key}' creator {i} has empty name")


def _validate_ingest_layout(key: str, ingest: IngestSpec) -> None:
    if not ingest.output_layout.strip():
        raise RegistryValidationError(
            f"Dataset '{key}' ingest output_layout must be provided"
        )
    if ingest.output_layout.lower() not in {"zarr", "hdf5", "h5"}:
        raise RegistryValidationError(
            f"Dataset '{key}' ingest output_layout '{ingest.output_layout}' "
            "is not supported"
        )


def _validate_crimaldi_ingest(key: str, ingest: CrimaldiFluorescenceIngest) -> None:
    if not ingest.dataset:
        raise RegistryValidationError(
            f"Dataset '{key}' CrimaldiFluorescenceIngest missing dataset path"
        )
    if ingest.fps <= 0:
        raise RegistryValidationError(f"Dataset '{key}' ingest fps must be positive")
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


def _validate_rigolli_ingest(key: str, ingest: RigolliDNSIngest) -> None:
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


def _validate_emonet_ingest(key: str, ingest: EmonetSmokeIngest) -> None:
    if ingest.fps <= 0:
        raise RegistryValidationError(
            f"Dataset '{key}' EmonetSmokeIngest fps must be positive"
        )
    ax, ay = ingest.arena_size_mm
    if ax <= 0 or ay <= 0:
        raise RegistryValidationError(
            f"Dataset '{key}' EmonetSmokeIngest arena_size_mm must be positive"
        )
    if ingest.background_n_frames < 0:
        raise RegistryValidationError(
            f"Dataset '{key}' EmonetSmokeIngest background_n_frames must be >= 0"
        )
    if ingest.trim_sigma < 0:
        raise RegistryValidationError(
            f"Dataset '{key}' EmonetSmokeIngest trim_sigma must be >= 0"
        )
    if ingest.trim_consecutive < 1:
        raise RegistryValidationError(
            f"Dataset '{key}' EmonetSmokeIngest trim_consecutive must be >= 1"
        )
    if ingest.trim_max_scan < 0:
        raise RegistryValidationError(
            f"Dataset '{key}' EmonetSmokeIngest trim_max_scan must be >= 0"
        )
    if ingest.trim_abs_threshold is not None and ingest.trim_abs_threshold < 0:
        raise RegistryValidationError(
            f"Dataset '{key}' EmonetSmokeIngest trim_abs_threshold must be >= 0"
        )
    if ingest.end_sigma < 0:
        raise RegistryValidationError(
            f"Dataset '{key}' EmonetSmokeIngest end_sigma must be >= 0"
        )
    if ingest.end_consecutive < 1:
        raise RegistryValidationError(
            f"Dataset '{key}' EmonetSmokeIngest end_consecutive must be >= 1"
        )
    if ingest.end_abs_threshold is not None and ingest.end_abs_threshold < 0:
        raise RegistryValidationError(
            f"Dataset '{key}' EmonetSmokeIngest end_abs_threshold must be >= 0"
        )


def _validate_ingest(key: str, ingest: IngestSpec) -> None:
    _validate_ingest_layout(key, ingest)
    if isinstance(ingest, CrimaldiFluorescenceIngest):
        _validate_crimaldi_ingest(key, ingest)
    elif isinstance(ingest, RigolliDNSIngest):
        _validate_rigolli_ingest(key, ingest)
    elif isinstance(ingest, EmonetSmokeIngest):
        _validate_emonet_ingest(key, ingest)


def validate_registry(
    registry: Optional[Dict[str, DatasetRegistryEntry]] = None,
) -> None:
    """Validate registry entries for completeness and internal consistency."""

    entries = _validate_registry_mapping(registry)
    for key, entry in entries.items():
        _validate_entry_identity(key, entry)
        _validate_artifact(key, entry.artifact)
        _validate_metadata(key, entry.metadata)
        _validate_creators(key, entry.metadata.creators)
        if entry.ingest:
            _validate_ingest(key, entry.ingest)
