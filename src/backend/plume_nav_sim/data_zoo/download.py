"""Downloader utilities for curated plume datasets.

These helpers resolve registry entries, manage cache locations, download and
verify artifacts, unpack archives, and perform lightweight layout validation for
Zarr and HDF5 payloads. Callers can opt-in to automatic downloads or operate in
an offline mode when the cache already satisfies integrity checks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .registry import (
    DATASET_REGISTRY,
    DEFAULT_CACHE_ROOT,
    CrimaldiFluorescenceIngest,
    DatasetRegistryEntry,
    EmonetSmokeIngest,
    RigolliDNSIngest,
    describe_dataset,
)
from .stats import compute_concentration_stats, store_stats_in_zarr

LOG = logging.getLogger(__name__)


class DatasetDownloadError(RuntimeError):
    """Base error for data zoo download failures."""


@dataclass
class ChecksumMismatchError(DatasetDownloadError):
    """Raised when a downloaded artifact fails checksum validation."""

    expected: str
    actual: str
    path: Path

    def __str__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Checksum mismatch for {self.path}: expected {self.expected}, "
            f"got {self.actual}"
        )


class LayoutValidationError(DatasetDownloadError):
    """Raised when unpacked data does not match expected layout."""


try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[misc,assignment]

    def Field(**kwargs: Any) -> Any:  # type: ignore[misc]  # noqa: N802
        """Stub for pydantic.Field when pydantic is not available."""
        return kwargs.get("default")


class ProvenanceSidecar(BaseModel if PYDANTIC_AVAILABLE else object):  # type: ignore[misc]
    """Provenance metadata for a processed dataset.

    This sidecar documents the full lineage from raw source to processed Zarr,
    enabling reproducibility and citation tracking.
    """

    # Dataset identification
    dataset_id: str
    version: str
    processed_at: str  # ISO 8601 timestamp

    # Processing info
    processor: str = "plume_nav_sim"
    processor_version: str

    # Source information
    source_url: str
    source_doi: Optional[str] = None
    source_filename: str
    source_checksum_md5: str
    source_size_bytes: Optional[int] = None

    # Ingest parameters (serialized from spec)
    ingest_spec_type: str
    ingest_params: Dict[str, Any] = Field(default_factory=dict)

    # Output information
    output_format: str
    output_shape: List[int] = Field(default_factory=list)
    output_dtype: str = "float32"
    output_checksum_md5: Optional[str] = None

    # Citation
    citation: Optional[str] = None
    citation_doi: Optional[str] = None

    if PYDANTIC_AVAILABLE:
        model_config = {"extra": "forbid"}

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON format."""
        if PYDANTIC_AVAILABLE:
            return self.model_dump_json(indent=indent)
        return json.dumps(self.__dict__, indent=indent)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        if PYDANTIC_AVAILABLE:
            return self.model_dump()
        return dict(self.__dict__)

    def write(self, output_path: Path, format: str = "json") -> Path:
        """Write sidecar to file next to the dataset.

        Args:
            output_path: Path to the Zarr directory or output file
            format: 'json' (recommended) or 'yaml'

        Returns:
            Path to the written sidecar file
        """
        sidecar_path = output_path.parent / f"{output_path.name}.provenance.{format}"

        if format == "json":
            content = self.to_json(indent=2)
        else:
            # Simple YAML-like output for human readability
            content = self._to_yaml()

        sidecar_path.write_text(content)
        LOG.info("Wrote provenance sidecar: %s", sidecar_path)
        return sidecar_path

    def _to_yaml(self) -> str:
        """Simple YAML serialization without pyyaml dependency."""
        lines = ["# Provenance sidecar for plume_nav_sim Data Zoo", ""]
        data = self.to_dict()

        def _format_value(v: Any, indent: int = 0) -> str:
            prefix = "  " * indent
            if v is None:
                return "null"
            elif isinstance(v, bool):
                return "true" if v else "false"
            elif isinstance(v, (int, float)):
                return str(v)
            elif isinstance(v, str):
                if "\n" in v or ":" in v or '"' in v:
                    return f'"{v}"'
                return v
            elif isinstance(v, list):
                if not v:
                    return "[]"
                return "".join(
                    f"\n{prefix}  - {_format_value(item, indent + 1)}" for item in v
                )
            elif isinstance(v, dict):
                if not v:
                    return "{}"
                return "".join(
                    f"\n{prefix}  {k}: {_format_value(val, indent + 1)}"
                    for k, val in v.items()
                )
            return str(v)

        for key, value in data.items():
            formatted = _format_value(value)
            lines.append(
                f"{key}:{formatted}" if "\n" in formatted else f"{key}: {formatted}"
            )

        return "\n".join(lines) + "\n"

    @classmethod
    def json_schema(cls) -> Dict[str, Any]:
        """Get JSON Schema for validation."""
        if PYDANTIC_AVAILABLE:
            return cls.model_json_schema()
        return {
            "type": "object",
            "description": "Provenance sidecar (schema unavailable)",
        }


def _get_version() -> str:
    """Get plume_nav_sim version string."""
    try:
        from plume_nav_sim import __version__

        return __version__
    except (ImportError, AttributeError):
        return "unknown"


def _compute_zarr_checksum(zarr_path: Path) -> str:
    """Compute a checksum for a Zarr directory by hashing .zarray and .zattrs files."""
    hasher = hashlib.md5(usedforsecurity=False)
    # Hash metadata files in sorted order for reproducibility
    for meta_file in sorted(zarr_path.rglob(".z*")):
        if meta_file.is_file():
            hasher.update(meta_file.read_bytes())
    return hasher.hexdigest()


def generate_provenance(
    entry: DatasetRegistryEntry,
    source_path: Path,
    output_path: Path,
    output_shape: List[int],
    output_dtype: str = "float32",
) -> ProvenanceSidecar:
    """Generate a provenance sidecar for a processed dataset.

    Args:
        entry: The registry entry for the dataset
        source_path: Path to the downloaded source file
        output_path: Path to the output Zarr directory
        output_shape: Shape of the main data array
        output_dtype: Data type of the main array

    Returns:
        ProvenanceSidecar instance (also writes to disk)
    """
    # Get source file size
    source_size = source_path.stat().st_size if source_path.exists() else None

    # Serialize ingest params
    ingest_params: Dict[str, Any] = {}
    ingest_type = "none"
    if entry.ingest:
        ingest_type = type(entry.ingest).__name__
        ingest_params = asdict(entry.ingest)

    sidecar = ProvenanceSidecar(
        dataset_id=entry.dataset_id,
        version=entry.version,
        processed_at=datetime.now(timezone.utc).isoformat(),
        processor="plume_nav_sim",
        processor_version=_get_version(),
        source_url=entry.artifact.url,
        source_doi=entry.metadata.doi if entry.metadata.doi else None,
        source_filename=source_path.name,
        source_checksum_md5=entry.artifact.checksum,
        source_size_bytes=source_size,
        ingest_spec_type=ingest_type,
        ingest_params=ingest_params,
        output_format=(
            entry.ingest.output_layout if entry.ingest else entry.artifact.layout
        ),
        output_shape=output_shape,
        output_dtype=output_dtype,
        output_checksum_md5=(
            _compute_zarr_checksum(output_path) if output_path.is_dir() else None
        ),
        citation=entry.metadata.citation if entry.metadata else None,
        citation_doi=entry.metadata.doi if entry.metadata else None,
    )

    # Write sidecar (JSON preferred for Pydantic schema compatibility)
    sidecar.write(output_path, format="json")

    return sidecar


def _dataset_layout(entry: DatasetRegistryEntry) -> str:
    """Return expected layout after any post-download processing."""

    if entry.ingest:
        return entry.ingest.output_layout
    return entry.artifact.layout


def ensure_dataset_available(
    dataset_id: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    auto_download: bool = False,
    force_download: bool = False,
    verify_checksum: bool = True,
    confirm_download: Optional[Callable[[DatasetRegistryEntry], bool]] = None,
) -> Path:
    """Ensure a dataset is present locally and return the resolved path.

    Args:
        dataset_id: Registry key for the dataset.
        cache_root: Base cache directory for artifacts and unpacked data.
        auto_download: When True, download the artifact if it is missing or the
            existing checksum fails validation. When False, raises if download is
            required unless the caller confirms via confirm_download.
        force_download: When True, re-download and re-unpack even if the cache
            appears valid.
        verify_checksum: When True, validate the downloaded artifact against the
            registry checksum before unpacking.
        confirm_download: Optional callable that returns True when the user
            approves a download. If omitted, an interactive prompt is used when
            auto_download is False and a download is required in a TTY context.

    Returns:
        Path to the unpacked dataset root (e.g., Zarr store directory or HDF5
        file).
    """

    entry = describe_dataset(dataset_id)
    cache_dir = entry.cache_path(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected_path = cache_dir / entry.expected_root
    artifact_path = cache_dir / _artifact_filename(entry)
    layout = _dataset_layout(entry)

    if force_download:
        _purge_cached(artifact_path, expected_path, cache_dir)
        if entry.ingest and entry.artifact.archive_member:
            stale_source = cache_dir / entry.artifact.archive_member
            if stale_source.exists():
                if stale_source.is_dir():
                    shutil.rmtree(stale_source)
                else:
                    stale_source.unlink()

    # Fast path: already unpacked and checksum (if available) is trusted
    if expected_path.exists() and not force_download:
        _validate_layout(expected_path, layout)
        if verify_checksum and artifact_path.is_file():
            try:
                _verify_checksum(artifact_path, entry)
                return expected_path
            except ChecksumMismatchError:
                if not auto_download:
                    raise
                LOG.warning(
                    "Checksum mismatch for cached %s; purging cache and re-downloading",
                    dataset_id,
                )
                _purge_cached(artifact_path, expected_path, cache_dir)
        else:
            if verify_checksum and not artifact_path.is_file():
                LOG.info(
                    "Skipping checksum verification for %s; cached data present but "
                    "artifact %s is missing or is a directory",
                    dataset_id,
                    artifact_path,
                )
            return expected_path

    if not artifact_path.exists() and not auto_download:
        should_download = (
            confirm_download(entry)
            if confirm_download is not None
            else _default_confirm_download(entry)
        )
        if not should_download:
            raise DatasetDownloadError(
                f"Dataset '{dataset_id}' missing from cache; set auto_download=True "
                "or approve the download to fetch from registry."
            )
        LOG.info("User approved download for %s", dataset_id)

    if auto_download or not artifact_path.exists():
        _download_artifact(entry, artifact_path)

    if verify_checksum:
        _verify_checksum(artifact_path, entry)

    source_path = _unpack_artifact(
        entry,
        artifact_path,
        cache_dir,
        _ingest_source_target(entry, cache_dir, artifact_path, expected_path),
    )
    final_path = _maybe_run_ingest(entry, source_path, expected_path)
    _validate_layout(final_path, layout)
    return final_path


def _ingest_source_target(
    entry: DatasetRegistryEntry,
    cache_dir: Path,
    artifact_path: Path,
    expected_path: Path,
) -> Path:
    """Determine where the unpacked artifact should live before ingest."""

    if entry.ingest is None:
        return expected_path

    archive_type = entry.artifact.archive_type.lower()
    if archive_type in ("none", "file", "raw"):
        return artifact_path
    if entry.artifact.archive_member:
        return cache_dir / entry.artifact.archive_member
    raise DatasetDownloadError(
        f"Dataset '{entry.dataset_id}' requires an archive_member to ingest "
        f"{entry.artifact.archive_type} payloads"
    )


def _maybe_run_ingest(
    entry: DatasetRegistryEntry, source_path: Path, expected_path: Path
) -> Path:
    """Optionally ingest an intermediate artifact into the final layout."""

    if entry.ingest is None:
        return source_path

    target_layout = entry.ingest.output_layout.lower()
    if target_layout in ("hdf5", "h5"):
        return source_path
    if target_layout != "zarr":
        raise DatasetDownloadError(
            f"Unsupported ingest output layout: {entry.ingest.output_layout}"
        )

    if expected_path.exists():
        return expected_path

    # Dispatch to appropriate ingest handler
    if isinstance(entry.ingest, CrimaldiFluorescenceIngest):
        result = _ingest_hdf5_to_zarr(entry.ingest, source_path, expected_path)
    elif isinstance(entry.ingest, RigolliDNSIngest):
        result = _ingest_mat_to_zarr(entry.ingest, source_path, expected_path)
    elif isinstance(entry.ingest, EmonetSmokeIngest):
        result = _ingest_emonet_to_zarr(entry.ingest, source_path, expected_path)
    else:
        raise DatasetDownloadError(
            f"Unknown ingest spec type: {type(entry.ingest).__name__}"
        )

    # Generate provenance sidecar after successful ingest
    _generate_provenance_for_zarr(entry, source_path, result)

    return result


def _generate_provenance_for_zarr(
    entry: DatasetRegistryEntry, source_path: Path, output_path: Path
) -> None:
    """Generate provenance sidecar for an ingested Zarr dataset."""
    try:
        import zarr

        store = zarr.open(str(output_path), mode="r")
        # Get shape from the main data array (concentration or similar)
        if "concentration" in store:
            shape = list(store["concentration"].shape)
            dtype = str(store["concentration"].dtype)
        else:
            # Fallback: use first array found
            for key in store.keys():
                arr = store[key]
                if hasattr(arr, "shape") and len(arr.shape) >= 2:
                    shape = list(arr.shape)
                    dtype = str(arr.dtype)
                    break
            else:
                shape = []
                dtype = "unknown"

        generate_provenance(entry, source_path, output_path, shape, dtype)

    except Exception as exc:
        # Don't fail the ingest if provenance generation fails
        LOG.warning("Failed to generate provenance sidecar: %s", exc)


def _artifact_filename(entry: DatasetRegistryEntry) -> str:
    """Extract clean filename from artifact URL, stripping query parameters."""
    from urllib.parse import unquote, urlparse

    parsed = urlparse(entry.artifact.url)
    if parsed.netloc.endswith("zenodo.org"):
        parts = [p for p in parsed.path.split("/") if p]
        if parts and parts[-1] == "content":
            try:
                files_index = parts.index("files")
            except ValueError:
                files_index = -1
            if 0 <= files_index < len(parts) - 2:
                candidate = parts[files_index + 1]
                if candidate:
                    return unquote(candidate)
    name = Path(parsed.path).name
    if name:
        return unquote(name)
    sanitized = entry.artifact.archive_type.replace(".", "_")
    return f"{entry.dataset_id}.{sanitized}"


def _download_artifact(entry: DatasetRegistryEntry, dest: Path) -> None:
    env_override = os.environ.get(f"PLUME_DATA_ZOO_URL_OVERRIDE_{entry.dataset_id}")
    url = env_override or os.environ.get("PLUME_DATA_ZOO_URL_OVERRIDE")
    if url:
        LOG.info(
            "Using override URL for %s via PLUME_DATA_ZOO_URL_OVERRIDE%s",
            entry.dataset_id,
            f"_{entry.dataset_id}" if env_override else "",
        )
    else:
        url = entry.artifact.url

    dest.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading %s -> %s", entry.dataset_id, dest)
    try:
        with (
            urllib.request.urlopen(url) as response,
            dest.open("wb") as fh,
        ):
            shutil.copyfileobj(response, fh)
    except Exception as exc:
        raise DatasetDownloadError(
            f"Failed to download {entry.dataset_id} from {url}: {exc}"
        ) from exc


def _verify_checksum(artifact_path: Path, entry: DatasetRegistryEntry) -> None:
    algo = entry.artifact.checksum_type.lower()
    try:
        hasher = hashlib.new(algo)
    except Exception as exc:  # pragma: no cover - unexpected algorithm names
        raise DatasetDownloadError(f"Unsupported checksum algorithm: {algo}") from exc

    with artifact_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    expected = entry.artifact.checksum.lower()
    if digest.lower() != expected:
        raise ChecksumMismatchError(
            expected=expected, actual=digest, path=artifact_path
        )


def _unpack_artifact(
    entry: DatasetRegistryEntry,
    artifact_path: Path,
    cache_dir: Path,
    target_path: Path,
) -> Path:
    archive_type = entry.artifact.archive_type.lower()

    # For non-archive types, handle the case where artifact_path == target_path
    # (the downloaded file IS the final target, so don't delete it)
    if archive_type in ("none", "file", "raw"):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Use resolve() for robust path comparison
        if artifact_path.resolve() != target_path.resolve():
            if target_path.exists():
                target_path.unlink()
            shutil.copy2(artifact_path, target_path)
        return target_path

    # For archives, clean up any stale target before extraction
    if target_path.exists():
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()
    if archive_type == "zip":
        with zipfile.ZipFile(artifact_path, "r") as zf:
            _safe_extract_zip(zf, cache_dir, entry.artifact.archive_member)
            _maybe_rename_member(cache_dir, entry.artifact.archive_member, target_path)
    elif archive_type in ("tar.gz", "tgz", "tar"):
        mode = "r:gz" if archive_type in ("tar.gz", "tgz") else "r:"
        with tarfile.open(artifact_path, mode) as tf:
            _safe_extract_tar(tf, cache_dir, entry.artifact.archive_member)
            _maybe_rename_member(cache_dir, entry.artifact.archive_member, target_path)
    else:
        raise DatasetDownloadError(f"Unsupported archive type: {archive_type}")

    if not target_path.exists():
        raise DatasetDownloadError(
            f"Expected unpacked root {target_path} missing after extracting {artifact_path}"
        )

    return target_path


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path, member: Optional[str]) -> None:
    if member:
        # Extract all members matching the prefix (handles directories stored with trailing slash)
        prefix = member.rstrip("/")
        members = [
            n
            for n in zf.namelist()
            if n == prefix or n.rstrip("/") == prefix or n.startswith(prefix + "/")
        ]
        if not members:
            raise DatasetDownloadError(
                f"Archive member '{member}' not found in archive. "
                f"Available members: {zf.namelist()[:10]}..."
            )
    else:
        members = zf.namelist()
    _validate_archive_members(members, dest)
    for name in members:
        zf.extract(name, path=dest)


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path, member: Optional[str]) -> None:
    members = [m for m in tf.getmembers() if member is None or m.name == member]
    _validate_archive_members([m.name for m in members], dest)
    for tar_member in members:
        tf.extract(tar_member, path=dest)


def _validate_archive_members(members: list[str], dest: Path) -> None:
    base = dest.resolve()
    for name in members:
        resolved = (base / name).resolve()
        if base not in resolved.parents and resolved != base:
            raise DatasetDownloadError(f"Unsafe archive member path: {name}")


def _infer_time_y_x_axes_for_rigolli(
    shape: tuple[int, ...], *, x_len: int, y_len: int
) -> tuple[int, int, int]:
    """Infer which axes correspond to (t, y, x) using coordinate lengths.

    Rigolli DNS MATLAB payloads are typically stored as either:
      - (x, y, t)  (common MATLAB convention)
      - (t, y, x)  (already in desired order)
      - (t, x, y)  (spatial axes swapped)

    When the spatial dimensions cannot be inferred from coordinate lengths, this
    falls back to treating the largest dimension as time.
    """

    if len(shape) != 3:
        raise DatasetDownloadError(
            f"Expected 3D concentration array, got shape {shape}"
        )

    candidates: list[tuple[int, int, int]] = []
    for time_axis in (0, 1, 2):
        other_axes = [ax for ax in (0, 1, 2) if ax != time_axis]
        for y_axis, x_axis in (other_axes, other_axes[::-1]):
            if shape[y_axis] == y_len and shape[x_axis] == x_len:
                candidates.append((time_axis, y_axis, x_axis))

    if len(candidates) == 1:
        return candidates[0]

    if candidates:
        # Prefer MATLAB-like ordering when ambiguous: (x, y, t) -> (t, y, x)
        for candidate in candidates:
            if candidate == (2, 1, 0):
                return candidate
        # Otherwise prefer time-last when possible.
        for time_axis, y_axis, x_axis in candidates:
            if time_axis == 2:
                return (time_axis, y_axis, x_axis)
        return candidates[0]

    time_axis = max(range(3), key=lambda ax: shape[ax])
    other_axes = [ax for ax in (0, 1, 2) if ax != time_axis]
    return (time_axis, other_axes[0], other_axes[1])


def _maybe_rename_member(dest: Path, member: Optional[str], target_path: Path) -> None:
    if member:
        extracted = dest / member.rstrip("/")
        if extracted != target_path and extracted.exists():
            if target_path.exists():
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
            extracted.rename(target_path)


def _ingest_hdf5_to_zarr(
    spec: CrimaldiFluorescenceIngest,
    source_path: Path,
    output_path: Path,
    compute_stats: bool = True,
) -> Path:
    """Convert an HDF5 plume movie into a Zarr dataset.

    Args:
        spec: Ingestion specification from registry.
        source_path: Path to source HDF5 file.
        output_path: Path for output Zarr store.
        compute_stats: If True, compute and store concentration statistics.

    Returns:
        Path to the created Zarr store.
    """
    try:
        from plume_nav_sim.media.h5_movie import H5MovieIngestConfig, ingest_h5_movie
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DatasetDownloadError(
            "HDF5 ingest requires optional dependencies (h5py, zarr, numcodecs). "
            "Install media extras or provide a pre-ingested dataset."
        ) from exc

    cfg = H5MovieIngestConfig(
        input=source_path,
        dataset=spec.dataset,
        output=output_path,
        t_start=0,
        t_stop=None,
        fps=spec.fps,
        pixel_to_grid=spec.pixel_to_grid,
        origin=spec.origin,
        extent=spec.extent,
        normalize=spec.normalize,
        chunk_t=spec.chunk_t,
    )

    try:
        result_path = ingest_h5_movie(cfg)
        if compute_stats:
            LOG.info("Computing concentration stats for %s...", output_path.name)
            stats = compute_concentration_stats(result_path)
            if spec.normalize:
                stats["normalized_during_ingest"] = True
            store_stats_in_zarr(result_path, stats)
        return result_path
    except Exception as exc:  # pragma: no cover - ingestion errors bubble up
        raise DatasetDownloadError(
            f"Failed to ingest HDF5 movie {source_path} into Zarr: {exc}"
        ) from exc


def _ingest_mat_to_zarr(
    spec: RigolliDNSIngest,
    source_path: Path,
    output_path: Path,
    compute_stats: bool = True,
) -> Path:
    """Convert a MATLAB v7.3 plume dataset with separate coordinates into Zarr.

    Downloads the coordinates file, loads both files using h5py (for MATLAB v7.3
    HDF5 format), and creates a standardized Zarr dataset with proper spatial
    dimension coordinates.

    Args:
        spec: Ingestion specification from registry.
        source_path: Path to source MATLAB file.
        output_path: Path for output Zarr store.
        compute_stats: If True, compute and store concentration statistics.

    Returns:
        Path to the created Zarr store.
    """
    try:
        import h5py
        import numpy as np
        import zarr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DatasetDownloadError(
            "MATLAB ingest requires optional dependencies (h5py, zarr, numpy). "
            "Install media extras or provide a pre-ingested dataset."
        ) from exc

    # Download coordinates file to cache directory
    coords_path = source_path.parent / "coordinates.mat"
    if not coords_path.exists():
        LOG.info("Downloading coordinates file for %s", source_path.name)
        coords_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with (
                urllib.request.urlopen(spec.coords_url) as response,
                coords_path.open("wb") as fh,
            ):
                shutil.copyfileobj(response, fh)
        except Exception as exc:
            raise DatasetDownloadError(
                f"Failed to download coordinates from {spec.coords_url}: {exc}"
            ) from exc

        # Verify coordinates checksum
        hasher = hashlib.md5(usedforsecurity=False)
        with coords_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest().lower() != spec.coords_checksum.lower():
            coords_path.unlink()
            raise DatasetDownloadError(
                f"Coordinates file checksum mismatch for {spec.coords_url}"
            )

    try:
        # Load coordinates - SciPy is optional (v5). Fall back to h5py (v7.3) when
        # SciPy is missing or the payload is MATLAB v7.3.
        coords_errors: list[Exception] = []
        x_coords: np.ndarray
        y_coords: np.ndarray

        try:
            from scipy.io import loadmat  # type: ignore[import-untyped]
        except Exception as exc:  # pragma: no cover - scipy is optional
            coords_errors.append(exc)
        else:
            try:
                coords_mat = loadmat(str(coords_path))
                x_coords = (
                    np.asarray(coords_mat[spec.x_key]).flatten().astype(np.float32)
                )
                y_coords = (
                    np.asarray(coords_mat[spec.y_key]).flatten().astype(np.float32)
                )
            except NotImplementedError as exc:
                coords_errors.append(exc)
            except Exception as exc:
                coords_errors.append(exc)

        if "x_coords" not in locals() or "y_coords" not in locals():
            try:
                with h5py.File(coords_path, "r") as coords_file:
                    x_data = coords_file[spec.x_key][:]
                    y_data = coords_file[spec.y_key][:]
                    x_coords = np.asarray(x_data).flatten().astype(np.float32)
                    y_coords = np.asarray(y_data).flatten().astype(np.float32)
            except Exception as exc:
                coords_errors.append(exc)
                details = "; ".join(type(e).__name__ for e in coords_errors)
                raise DatasetDownloadError(
                    "Failed to load Rigolli DNS coordinates file. "
                    "If this is a legacy MATLAB v5 file, install SciPy; "
                    f"load attempts failed with: {details}"
                ) from exc

        # Load concentration data from source MATLAB file (streaming in time)
        with h5py.File(source_path, "r") as mat_file:
            if spec.concentration_key not in mat_file:
                raise DatasetDownloadError(
                    f"Could not find '{spec.concentration_key}' in {source_path.name}. "
                    f"Available keys: {list(mat_file.keys())}"
                )
            conc_dataset = mat_file[spec.concentration_key]
            src_shape = tuple(int(s) for s in conc_dataset.shape)
            time_axis, y_axis, x_axis = _infer_time_y_x_axes_for_rigolli(
                src_shape, x_len=len(x_coords), y_len=len(y_coords)
            )

            n_frames = src_shape[time_axis]
            ny = src_shape[y_axis]
            nx = src_shape[x_axis]

            # Sanity: match inferred spatial dims to coordinate arrays when possible.
            if ny != len(y_coords) or nx != len(x_coords):
                LOG.warning(
                    "Rigolli DNS coordinate length mismatch: inferred (ny=%d,nx=%d) "
                    "but coords are (len(y)=%d,len(x)=%d). Proceeding with inferred dims.",
                    ny,
                    nx,
                    len(y_coords),
                    len(x_coords),
                )

            # Choose chunk size to keep Zarr chunks well below codec limits.
            bytes_per_frame = ny * nx * 4  # float32
            max_chunk_bytes = 500 * 1024 * 1024  # 500MB target
            chunk_t = min(
                spec.chunk_t or 100, max(1, max_chunk_bytes // max(1, bytes_per_frame))
            )

            store = zarr.DirectoryStore(str(output_path))
            root = zarr.group(store, overwrite=True)
            conc = root.create_dataset(
                "concentration",
                shape=(n_frames, ny, nx),
                chunks=(chunk_t, ny, nx),
                dtype=np.float32,
            )
            root.create_dataset("x", data=x_coords, dtype=np.float32)
            root.create_dataset("y", data=y_coords, dtype=np.float32)

            # Process in time chunks and transpose into (t, y, x)
            global_min, global_max = np.inf, -np.inf
            for t_start in range(0, n_frames, chunk_t):
                t_end = min(t_start + chunk_t, n_frames)
                slicer = [slice(None), slice(None), slice(None)]
                slicer[time_axis] = slice(t_start, t_end)
                chunk = conc_dataset[tuple(slicer)]
                chunk = np.transpose(chunk, (time_axis, y_axis, x_axis)).astype(
                    np.float32
                )
                conc[t_start:t_end] = chunk

                if spec.normalize:
                    global_min = min(global_min, float(chunk.min()))
                    global_max = max(global_max, float(chunk.max()))

                if (t_start // chunk_t) % 10 == 0:
                    LOG.info("  Processed %d / %d frames...", t_end, n_frames)

        # Normalize in a second pass if needed
        if spec.normalize and global_max > global_min:
            LOG.info("Normalizing data (min=%.4f, max=%.4f)...", global_min, global_max)
            for t_start in range(0, n_frames, chunk_t):
                t_end = min(t_start + chunk_t, n_frames)
                chunk = conc[t_start:t_end]
                conc[t_start:t_end] = (chunk - global_min) / (global_max - global_min)

        # Store metadata in .zattrs
        root.attrs["fps"] = spec.fps if spec.fps else 0.0
        root.attrs["n_frames"] = int(n_frames)
        root.attrs["shape"] = [int(n_frames), int(ny), int(nx)]
        root.attrs["source_format"] = "matlab_v73"
        root.attrs["normalized"] = bool(spec.normalize)

        LOG.info(
            "Ingested MATLAB file to Zarr: %s -> %s (shape=[%d, %d, %d])",
            source_path.name,
            output_path.name,
            n_frames,
            ny,
            nx,
        )

        # Compute and store concentration statistics
        if compute_stats:
            LOG.info("Computing concentration stats for %s...", output_path.name)
            stats = compute_concentration_stats(output_path)
            if spec.normalize:
                stats["normalized_during_ingest"] = True
            store_stats_in_zarr(output_path, stats)

        return output_path

    except Exception as exc:
        # Clean up partial output on failure
        if output_path.exists():
            shutil.rmtree(output_path)
        raise DatasetDownloadError(
            f"Failed to ingest MATLAB file {source_path} into Zarr: {exc}"
        ) from exc


def _ingest_emonet_to_zarr(
    spec: EmonetSmokeIngest,
    source_path: Path,
    output_path: Path,
    compute_stats: bool = True,
) -> Path:
    """Convert Emonet lab flyWalk video frames into standardized Zarr.

    Downloads optional metadata file, loads video frames from MATLAB format,
    and creates a Zarr dataset with spatial coordinates derived from arena size.

    Args:
        spec: Ingestion specification from registry.
        source_path: Path to source MATLAB file.
        output_path: Path for output Zarr store.
        compute_stats: If True, compute and store concentration statistics.
            For large datasets, set to False and compute stats separately.

    Returns:
        Path to the created Zarr store.
    """
    try:
        import h5py
        import numpy as np
        import zarr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DatasetDownloadError(
            "Emonet ingest requires optional dependencies (h5py, zarr, numpy). "
            "Install media extras or provide a pre-ingested dataset."
        ) from exc

    # Optionally download metadata file
    metadata = {}
    if spec.metadata_url:
        meta_path = source_path.parent / "metadata.mat"
        if not meta_path.exists():
            LOG.info("Downloading metadata file for %s", source_path.name)
            try:
                with (
                    urllib.request.urlopen(spec.metadata_url) as response,
                    meta_path.open("wb") as fh,
                ):
                    shutil.copyfileobj(response, fh)
            except Exception as exc:
                LOG.warning("Failed to download metadata: %s (continuing without)", exc)

        if meta_path.exists():
            try:
                with h5py.File(meta_path, "r") as mf:
                    # Extract any useful metadata (fps, px_per_mm, etc.)
                    for key in mf.keys():
                        try:
                            metadata[key] = np.asarray(mf[key]).item()
                        except (ValueError, TypeError):
                            pass
            except Exception as exc:
                LOG.warning("Failed to parse metadata file: %s", exc)

    try:
        # Open HDF5 file and get shape without loading data
        with h5py.File(source_path, "r") as mat_file:
            # Try common keys for video frames
            frames_key = spec.frames_key
            if frames_key not in mat_file:
                # Try to find the frames array
                for key in mat_file.keys():
                    if "frame" in key.lower() or mat_file[key].ndim == 3:
                        frames_key = key
                        break

            if frames_key not in mat_file:
                raise DatasetDownloadError(
                    f"Could not find frames array in {source_path}. "
                    f"Available keys: {list(mat_file.keys())}"
                )

            frames_dataset = mat_file[frames_key]
            src_shape = frames_dataset.shape
            LOG.info(
                "Found frames at key '%s' with shape %s, processing in chunks...",
                frames_key,
                src_shape,
            )

            # Emonet format per README: (Height x Width x n_frames) = (H, W, T)
            # But actual file has (T, X, Y) = (n_frames, Width, Height)
            # Arena: 300mm x 180mm, px_per_mm ~6.83
            # 300mm * 6.83 = 2048px (X/Width), 180mm * 6.83 ≈ 1200px (Y/Height)
            # We want (T, Y, X) -> swap axes 1 and 2 during chunk processing
            n_frames, n_x, n_y = src_shape

            # Generate spatial coordinates from arena size
            arena_x, arena_y = spec.arena_size_mm
            x_coords = np.linspace(0, arena_x, n_x, dtype=np.float32)
            y_coords = np.linspace(0, arena_y, n_y, dtype=np.float32)

            # Create Zarr store with pre-allocated empty array
            # Use smaller time chunks to stay under 2GB codec limit
            # Each frame is n_y * n_x * 4 bytes = ~10 MB for Emonet
            bytes_per_frame = n_y * n_x * 4
            max_chunk_bytes = 500 * 1024 * 1024  # 500 MB target
            chunk_t = min(
                spec.chunk_t or 100, max(1, max_chunk_bytes // bytes_per_frame)
            )

            LOG.info(
                "Using chunks: (%d, %d, %d) for shape (%d, %d, %d), ~%.0f MB/chunk",
                chunk_t,
                n_y,
                n_x,
                n_frames,
                n_y,
                n_x,
                chunk_t * bytes_per_frame / 1e6,
            )

            store = zarr.DirectoryStore(str(output_path))
            root = zarr.group(store, overwrite=True)

            conc = root.create_dataset(
                "concentration",
                shape=(n_frames, n_y, n_x),
                chunks=(chunk_t, n_y, n_x),
                dtype=np.float32,
            )
            root.create_dataset("x", data=x_coords, dtype=np.float32)
            root.create_dataset("y", data=y_coords, dtype=np.float32)

            # Process in chunks to avoid memory issues
            # Source is (T, X, Y), we need (T, Y, X) - swap spatial axes
            batch_size = chunk_t
            global_min, global_max = np.inf, -np.inf

            for t_start in range(0, n_frames, batch_size):
                t_end = min(t_start + batch_size, n_frames)
                # Read time slice: (batch, X, Y)
                chunk = frames_dataset[t_start:t_end]
                # Transpose to (batch, Y, X) - swap axes 1 and 2
                chunk = np.transpose(chunk, (0, 2, 1)).astype(np.float32)
                conc[t_start:t_end] = chunk

                if spec.normalize:
                    global_min = min(global_min, chunk.min())
                    global_max = max(global_max, chunk.max())

                if (t_start // batch_size) % 10 == 0:
                    LOG.info("  Processed %d / %d frames...", t_end, n_frames)

        # Normalize in a second pass if needed
        if spec.normalize and global_max > global_min:
            LOG.info("Normalizing data (min=%.2f, max=%.2f)...", global_min, global_max)
            for t_start in range(0, n_frames, batch_size):
                t_end = min(t_start + batch_size, n_frames)
                chunk = conc[t_start:t_end]
                conc[t_start:t_end] = (chunk - global_min) / (global_max - global_min)

        # Store metadata
        root.attrs["fps"] = metadata.get("fps", spec.fps)
        root.attrs["n_frames"] = n_frames
        root.attrs["shape"] = [n_frames, n_y, n_x]
        root.attrs["arena_size_mm"] = list(spec.arena_size_mm)
        root.attrs["source_format"] = "emonet_flywalk"
        root.attrs["normalized"] = spec.normalize

        LOG.info(
            "Ingested Emonet smoke video to Zarr: %s -> %s (shape=[%d, %d, %d])",
            source_path.name,
            output_path.name,
            n_frames,
            n_y,
            n_x,
        )

        # Compute and store concentration statistics
        # For large Emonet datasets, consider compute_stats=False and run separately
        if compute_stats:
            LOG.info("Computing concentration stats for %s...", output_path.name)
            stats = compute_concentration_stats(output_path)
            if spec.normalize:
                stats["normalized_during_ingest"] = True
                # Store original range for reference
                stats["original_min"] = (
                    float(global_min) if global_max > global_min else None
                )
                stats["original_max"] = (
                    float(global_max) if global_max > global_min else None
                )
            store_stats_in_zarr(output_path, stats)

        return output_path

    except Exception as exc:
        if output_path.exists():
            shutil.rmtree(output_path)
        raise DatasetDownloadError(
            f"Failed to ingest Emonet video {source_path} into Zarr: {exc}"
        ) from exc


def _validate_layout(target_path: Path, layout: str) -> None:
    layout = layout.lower()
    if layout == "zarr":
        if not target_path.is_dir():
            raise LayoutValidationError(f"Expected Zarr directory at {target_path}")
        zattrs = target_path / ".zattrs"
        if not zattrs.exists():
            raise LayoutValidationError(
                f"Missing Zarr metadata (.zattrs) in {target_path}"
            )
    elif layout in ("hdf5", "h5"):
        if not target_path.is_file():
            raise LayoutValidationError(f"Expected HDF5 file at {target_path}")
        if target_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise LayoutValidationError(
                f"Unexpected HDF5 extension for {target_path.name}"
            )
    else:
        raise DatasetDownloadError(f"Unknown layout type: {layout}")


def _purge_cached(artifact_path: Path, expected_path: Path, cache_dir: Path) -> None:
    if artifact_path.exists():
        artifact_path.unlink()
    if expected_path.exists():
        if expected_path.is_dir():
            shutil.rmtree(expected_path)
        else:
            expected_path.unlink()
    cache_dir.mkdir(parents=True, exist_ok=True)


def _default_confirm_download(entry: DatasetRegistryEntry) -> bool:
    """Default confirmation handler that prompts when interactive."""

    if not sys.stdin.isatty():
        LOG.info(
            "Cannot prompt for download of %s in non-interactive context",
            entry.dataset_id,
        )
        return False

    prompt = (
        f"Dataset '{entry.dataset_id}' ({entry.metadata.title}) is not cached.\n"
        f"Download from {entry.artifact.url} into {entry.cache_subdir}/{entry.version}? [y/N]: "
    )
    try:
        answer = input(prompt)
    except EOFError:
        return False

    return answer.strip().lower() in {"y", "yes"}


__all__ = [
    "DatasetDownloadError",
    "ChecksumMismatchError",
    "LayoutValidationError",
    "ensure_dataset_available",
]
