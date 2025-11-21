"""Downloader utilities for curated plume datasets.

These helpers resolve registry entries, manage cache locations, download and
verify artifacts, unpack archives, and perform lightweight layout validation for
Zarr and HDF5 payloads. Callers can opt-in to automatic downloads or operate in
an offline mode when the cache already satisfies integrity checks.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .registry import (
    DATASET_REGISTRY,
    DEFAULT_CACHE_ROOT,
    DatasetRegistryEntry,
    describe_dataset,
)

LOG = logging.getLogger(__name__)


class DatasetDownloadError(RuntimeError):
    """Base error for data zoo download failures."""


@dataclass
class ChecksumMismatchError(DatasetDownloadError):
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


def ensure_dataset_available(
    dataset_id: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    auto_download: bool = False,
    force_download: bool = False,
    verify_checksum: bool = True,
) -> Path:
    """Ensure a dataset is present locally and return the resolved path.

    Args:
        dataset_id: Registry key for the dataset.
        cache_root: Base cache directory for artifacts and unpacked data.
        auto_download: When True, download the artifact if it is missing or the
            existing checksum fails validation. When False, raises if download is
            required.
        force_download: When True, re-download and re-unpack even if the cache
            appears valid.
        verify_checksum: When True, validate the downloaded artifact against the
            registry checksum before unpacking.

    Returns:
        Path to the unpacked dataset root (e.g., Zarr store directory or HDF5
        file).
    """

    entry = describe_dataset(dataset_id)
    cache_dir = entry.cache_path(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected_path = cache_dir / entry.expected_root
    artifact_path = cache_dir / _artifact_filename(entry)

    if force_download:
        _purge_cached(artifact_path, expected_path, cache_dir)

    # Fast path: already unpacked and checksum (if available) is trusted
    if expected_path.exists() and not force_download:
        if verify_checksum and artifact_path.exists():
            _verify_checksum(artifact_path, entry)
        return expected_path

    if not artifact_path.exists() and not auto_download:
        raise DatasetDownloadError(
            f"Dataset '{dataset_id}' missing from cache; set auto_download=True "
            "to fetch from registry."
        )

    if auto_download:
        _download_artifact(entry, artifact_path)
    elif not artifact_path.exists():
        raise DatasetDownloadError(
            f"Dataset '{dataset_id}' artifact not found at {artifact_path}"  # pragma: no cover - guarded above
        )

    if verify_checksum:
        _verify_checksum(artifact_path, entry)

    target_path = _unpack_artifact(entry, artifact_path, cache_dir)
    _validate_layout(target_path, entry)
    return target_path


def _artifact_filename(entry: DatasetRegistryEntry) -> str:
    name = Path(entry.artifact.url).name
    if name:
        return name
    sanitized = entry.artifact.archive_type.replace(".", "_")
    return f"{entry.dataset_id}.{sanitized}"


def _download_artifact(entry: DatasetRegistryEntry, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading %s -> %s", entry.dataset_id, dest)
    try:
        with (
            urllib.request.urlopen(entry.artifact.url) as response,
            dest.open("wb") as fh,
        ):
            shutil.copyfileobj(response, fh)
    except Exception as exc:
        raise DatasetDownloadError(
            f"Failed to download {entry.dataset_id} from {entry.artifact.url}: {exc}"
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
    entry: DatasetRegistryEntry, artifact_path: Path, cache_dir: Path
) -> Path:
    archive_type = entry.artifact.archive_type.lower()
    target_path = cache_dir / entry.expected_root

    if target_path.exists():
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()

    if archive_type == "zip":
        with zipfile.ZipFile(artifact_path, "r") as zf:
            if entry.artifact.archive_member:
                member = entry.artifact.archive_member
                zf.extract(member, path=cache_dir)
                extracted = cache_dir / member
                extracted.rename(target_path)
            else:
                zf.extractall(path=cache_dir)
    elif archive_type in ("tar.gz", "tgz", "tar"):
        mode = "r:gz" if archive_type in ("tar.gz", "tgz") else "r:"
        with tarfile.open(artifact_path, mode) as tf:
            if entry.artifact.archive_member:
                member = entry.artifact.archive_member
                tf.extract(member, path=cache_dir)
                extracted = cache_dir / member
                extracted.rename(target_path)
            else:
                tf.extractall(path=cache_dir)
    else:
        raise DatasetDownloadError(f"Unsupported archive type: {archive_type}")

    if not target_path.exists():
        raise DatasetDownloadError(
            f"Expected unpacked root {target_path} missing after extracting {artifact_path}"
        )

    return target_path


def _validate_layout(target_path: Path, entry: DatasetRegistryEntry) -> None:
    layout = entry.artifact.layout.lower()
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


__all__ = [
    "DatasetDownloadError",
    "ChecksumMismatchError",
    "LayoutValidationError",
    "ensure_dataset_available",
]
