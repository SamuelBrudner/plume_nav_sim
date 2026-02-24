from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .registry import DatasetRegistryEntry

LOG = logging.getLogger("plume_nav_sim.data_zoo.download")


class DatasetDownloadError(RuntimeError):
    """Base error for data zoo download failures."""


def _http_request(url: str) -> urllib.request.Request:
    user_agent = os.environ.get(
        "PLUME_DATA_ZOO_USER_AGENT",
        "plume_nav_sim-data_zoo/1.0 (urllib; https://github.com/SamuelBrudner/plume_nav_sim)",
    )
    headers: dict[str, str] = {"User-Agent": user_agent}
    bearer = os.environ.get("PLUME_DRYAD_BEARER_TOKEN")
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    return urllib.request.Request(url, headers=headers)


def _resolve_dryad_presigned_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc != "datadryad.org":
        return url
    if not parsed.path.startswith("/api/v2/files/"):
        return url
    if not parsed.path.endswith("/download"):
        return url

    # Dryad API v2 download endpoints may reject plain GET/HEAD with 401 but will
    # issue a redirect when a small ranged GET is used.
    req = _http_request(url)
    user_agent = (
        req.headers.get("User-agent")
        or req.headers.get("User-Agent")
        or os.environ.get("PLUME_DATA_ZOO_USER_AGENT")
        or "plume_nav_sim-data_zoo/1.0 (urllib)"
    )
    auth = req.headers.get("Authorization")
    headers: dict[str, str] = {
        "User-Agent": user_agent,
        "Range": "bytes=0-0",
        "Accept": "*/*",
    }
    if auth:
        headers["Authorization"] = auth
    range_req = urllib.request.Request(
        url,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(range_req, timeout=30) as response:
            resolved = response.geturl()
    except Exception as exc:
        raise DatasetDownloadError(
            f"Failed to resolve Dryad presigned URL for {url}: {exc}"
        ) from exc

    if not resolved or resolved == url:
        raise DatasetDownloadError(
            f"Dryad download URL did not resolve to a presigned Location URL: {url}"
        )
    return resolved


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


def _artifact_filename(entry: DatasetRegistryEntry) -> str:
    """Extract clean filename from artifact URL, stripping query parameters."""
    from urllib.parse import unquote, urlparse

    parsed = urlparse(entry.artifact.url)
    if parsed.netloc == "datadryad.org" and parsed.path.startswith("/api/v2/files/"):
        parts = [p for p in parsed.path.split("/") if p]
        if (
            len(parts) >= 4
            and parts[0] == "api"
            and parts[1] == "v2"
            and parts[2] == "files"
        ):
            file_id = parts[3]
            try:
                meta_url = f"https://datadryad.org/api/v2/files/{file_id}"
                with urllib.request.urlopen(
                    _http_request(meta_url), timeout=30
                ) as resp:
                    meta = json.load(resp)
                candidate = meta.get("path")
                if isinstance(candidate, str) and candidate:
                    return candidate
            except Exception:
                return f"dryad_file_{file_id}"
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
        tmp_dest = dest.with_name(dest.name + ".part")
        if tmp_dest.exists():
            tmp_dest.unlink()

        resolved_url = _resolve_dryad_presigned_url(url)

        with urllib.request.urlopen(_http_request(resolved_url)) as response:
            headers = getattr(response, "headers", None)
            content_length = (
                headers.get("Content-Length") if hasattr(headers, "get") else None
            )
            expected_bytes = int(content_length) if content_length else None

            with tmp_dest.open("wb") as fh:
                shutil.copyfileobj(response, fh)

        actual_bytes = tmp_dest.stat().st_size
        if expected_bytes is not None and actual_bytes != expected_bytes:
            tmp_dest.unlink()
            raise DatasetDownloadError(
                f"Partial download for {entry.dataset_id} from {url}: expected "
                f"{expected_bytes} bytes, got {actual_bytes} bytes"
            )

        tmp_dest.replace(dest)
    except urllib.error.HTTPError as exc:
        raise DatasetDownloadError(
            f"Failed to download {entry.dataset_id} from {url}: "
            f"HTTP {exc.code} {exc.reason}"
        ) from exc
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

