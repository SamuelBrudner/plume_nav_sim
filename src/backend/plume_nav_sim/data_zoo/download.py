from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

from .downloader import (  # noqa: F401
    ChecksumMismatchError,
    DatasetDownloadError,
    LayoutValidationError,
    _artifact_filename,
    _download_artifact,
    _http_request,
    _maybe_rename_member,
    _resolve_dryad_presigned_url,
    _safe_extract_tar,
    _safe_extract_zip,
    _unpack_artifact,
    _validate_archive_members,
    _verify_checksum,
)
from .ingest import _compute_chunk_t, _normalize_concentration, _require_ingest_deps  # noqa: F401
from .ingest.crimaldi import _ingest_hdf5_to_zarr  # noqa: F401
from .ingest.emonet import (  # noqa: F401
    _compute_emonet_background,
    _download_emonet_metadata,
    _emonet_frame_signal,
    _estimate_emonet_end_frame,
    _estimate_emonet_start_frame,
    _init_emonet_store,
    _ingest_emonet_to_zarr,
    _load_emonet_metadata,
    _parse_emonet_metadata,
    _resolve_frames_key,
    _stream_emonet_chunks,
    _write_emonet_attrs,
)
from .ingest.rigolli import (  # noqa: F401
    _ensure_coords_file,
    _get_mat_concentration_dataset,
    _infer_time_y_x_axes_for_rigolli,
    _init_rigolli_store,
    _ingest_mat_to_zarr,
    _load_rigolli_coords,
    _pick_rigolli_axes,
    _rigolli_axis_candidates,
    _stream_rigolli_chunks,
    _try_load_coords_with_h5py,
    _try_load_coords_with_scipy,
    _write_rigolli_attrs,
)
from .registry import (
    DEFAULT_CACHE_ROOT,
    CrimaldiFluorescenceIngest,
    DatasetRegistryEntry,
    EmonetSmokeIngest,
    RigolliDNSIngest,
    describe_dataset,
)
from .sidecar import (  # noqa: F401
    PYDANTIC_AVAILABLE,
    BaseModel,
    Field,
    ProvenanceSidecar,
    _compute_zarr_checksum,
    _format_yaml_dict,
    _format_yaml_list,
    _format_yaml_string,
    _format_yaml_value,
    _generate_provenance_for_zarr,
    _get_version,
    generate_provenance,
)

LOG = logging.getLogger(__name__)


def _dataset_layout(entry: DatasetRegistryEntry) -> str:
    """Return expected layout after any post-download processing."""

    if entry.ingest:
        return entry.ingest.output_layout
    return entry.artifact.layout


def _purge_for_force_download(
    entry: DatasetRegistryEntry,
    *,
    cache_dir: Path,
    artifact_path: Path,
    expected_path: Path,
) -> None:
    _purge_cached(artifact_path, expected_path, cache_dir)
    if entry.ingest and entry.artifact.archive_member:
        stale_source = cache_dir / entry.artifact.archive_member
        if stale_source.exists():
            if stale_source.is_dir():
                shutil.rmtree(stale_source)
            else:
                stale_source.unlink()


def _try_cached_dataset(
    entry: DatasetRegistryEntry,
    *,
    expected_path: Path,
    artifact_path: Path,
    cache_dir: Path,
    layout: str,
    auto_download: bool,
    verify_checksum: bool,
) -> Path | None:
    if not expected_path.exists():
        return None

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
                entry.dataset_id,
            )
            _purge_cached(artifact_path, expected_path, cache_dir)
            return None

    if verify_checksum and not artifact_path.is_file():
        LOG.info(
            "Skipping checksum verification for %s; cached data present but "
            "artifact %s is missing or is a directory",
            entry.dataset_id,
            artifact_path,
        )
    return expected_path


def _confirm_download_if_needed(
    entry: DatasetRegistryEntry,
    *,
    artifact_path: Path,
    auto_download: bool,
    confirm_download: Optional[Callable[[DatasetRegistryEntry], bool]],
) -> None:
    if artifact_path.exists() or auto_download:
        return

    should_download = (
        confirm_download(entry)
        if confirm_download is not None
        else _default_confirm_download(entry)
    )
    if not should_download:
        raise DatasetDownloadError(
            f"Dataset '{entry.dataset_id}' missing from cache; set auto_download=True "
            "or approve the download to fetch from registry."
        )
    LOG.info("User approved download for %s", entry.dataset_id)


def ensure_dataset_available(
    dataset_id: str,
    *,
    cache_root: Path | None = DEFAULT_CACHE_ROOT,
    auto_download: bool = False,
    force_download: bool = False,
    verify_checksum: bool = True,
    confirm_download: Optional[Callable[[DatasetRegistryEntry], bool]] = None,
) -> Path:
    entry = describe_dataset(dataset_id)
    resolved_cache_root = DEFAULT_CACHE_ROOT if cache_root is None else cache_root
    cache_dir = entry.cache_path(resolved_cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected_path = cache_dir / entry.expected_root
    artifact_path = cache_dir / _artifact_filename(entry)
    layout = _dataset_layout(entry)

    if force_download:
        _purge_for_force_download(
            entry,
            cache_dir=cache_dir,
            artifact_path=artifact_path,
            expected_path=expected_path,
        )

    # Fast path: already unpacked and checksum (if available) is trusted
    if not force_download:
        cached_path = _try_cached_dataset(
            entry,
            expected_path=expected_path,
            artifact_path=artifact_path,
            cache_dir=cache_dir,
            layout=layout,
            auto_download=auto_download,
            verify_checksum=verify_checksum,
        )
        if cached_path is not None:
            return cached_path

    _confirm_download_if_needed(
        entry,
        artifact_path=artifact_path,
        auto_download=auto_download,
        confirm_download=confirm_download,
    )

    if not artifact_path.exists():
        _download_artifact(entry, artifact_path)

    if verify_checksum:
        try:
            _verify_checksum(artifact_path, entry)
        except ChecksumMismatchError:
            if artifact_path.exists():
                artifact_path.unlink()
            if not auto_download:
                raise
            LOG.warning(
                "Checksum mismatch for %s; re-downloading because auto_download=True",
                entry.dataset_id,
            )
            _download_artifact(entry, artifact_path)
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

