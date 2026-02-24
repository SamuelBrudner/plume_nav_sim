"""Tests for the Data Zoo downloader module.

All tests are offline-safe -- no real network calls are made.
Every HTTP interaction is mocked, and filesystem operations use tmp_path.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from plume_nav_sim.data_zoo.downloader import (
    ChecksumMismatchError,
    DatasetDownloadError,
    LayoutValidationError,
    _artifact_filename,
    _download_artifact,
    _resolve_dryad_presigned_url,
    _safe_extract_tar,
    _safe_extract_zip,
    _unpack_artifact,
    _validate_archive_members,
    _verify_checksum,
)
from plume_nav_sim.data_zoo.registry import (
    DatasetArtifact,
    DatasetMetadata,
    DatasetRegistryEntry,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_entry(
    *,
    url: str = "https://example.com/data.zip",
    checksum: str = "abc123",
    checksum_type: str = "sha256",
    archive_type: str = "zip",
    archive_member: str | None = None,
    layout: str = "zarr",
    dataset_id: str = "test_ds",
    version: str = "1.0.0",
    cache_subdir: str = "test_cache",
    expected_root: str = "test_root",
) -> DatasetRegistryEntry:
    """Build a minimal DatasetRegistryEntry for testing."""
    return DatasetRegistryEntry(
        dataset_id=dataset_id,
        version=version,
        artifact=DatasetArtifact(
            url=url,
            checksum=checksum,
            checksum_type=checksum_type,
            archive_type=archive_type,
            archive_member=archive_member,
            layout=layout,
        ),
        cache_subdir=cache_subdir,
        expected_root=expected_root,
        metadata=DatasetMetadata(title="Test dataset"),
    )


def _write_file(path: Path, content: bytes = b"hello world") -> Path:
    """Write bytes to *path* and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _md5_hex(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _make_zip_bytes(members: dict[str, bytes]) -> bytes:
    """Return raw bytes of a ZIP archive containing *members*."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _make_tar_gz_bytes(members: dict[str, bytes]) -> bytes:
    """Return raw bytes of a tar.gz archive containing *members*."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _make_tar_bytes(members: dict[str, bytes]) -> bytes:
    """Return raw bytes of a plain tar archive containing *members*."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# ChecksumMismatchError
# ---------------------------------------------------------------------------


class TestChecksumMismatchError:
    def test_is_dataset_download_error(self):
        err = ChecksumMismatchError(expected="aaa", actual="bbb", path=Path("/f"))
        assert isinstance(err, DatasetDownloadError)

    def test_fields(self):
        p = Path("/some/file.bin")
        err = ChecksumMismatchError(expected="aaa", actual="bbb", path=p)
        assert err.expected == "aaa"
        assert err.actual == "bbb"
        assert err.path == p

    def test_str_contains_info(self):
        err = ChecksumMismatchError(
            expected="aaa", actual="bbb", path=Path("/x/y.bin")
        )
        text = str(err)
        assert "aaa" in text
        assert "bbb" in text
        assert "y.bin" in text


# ---------------------------------------------------------------------------
# LayoutValidationError
# ---------------------------------------------------------------------------


class TestLayoutValidationError:
    def test_is_dataset_download_error(self):
        assert issubclass(LayoutValidationError, DatasetDownloadError)


# ---------------------------------------------------------------------------
# _verify_checksum
# ---------------------------------------------------------------------------


class TestVerifyChecksum:
    def test_sha256_pass(self, tmp_path: Path):
        data = b"checksum test payload"
        digest = _sha256_hex(data)
        artifact = _write_file(tmp_path / "artifact.bin", data)
        entry = _make_entry(checksum=digest, checksum_type="sha256")
        # Should not raise
        _verify_checksum(artifact, entry)

    def test_sha256_fail(self, tmp_path: Path):
        data = b"checksum test payload"
        artifact = _write_file(tmp_path / "artifact.bin", data)
        entry = _make_entry(checksum="0" * 64, checksum_type="sha256")
        with pytest.raises(ChecksumMismatchError) as exc_info:
            _verify_checksum(artifact, entry)
        assert exc_info.value.expected == "0" * 64
        assert exc_info.value.actual == _sha256_hex(data)
        assert exc_info.value.path == artifact

    def test_md5_pass(self, tmp_path: Path):
        data = b"md5 payload"
        digest = _md5_hex(data)
        artifact = _write_file(tmp_path / "artifact.bin", data)
        entry = _make_entry(checksum=digest, checksum_type="md5")
        _verify_checksum(artifact, entry)

    def test_md5_fail(self, tmp_path: Path):
        data = b"md5 payload"
        artifact = _write_file(tmp_path / "artifact.bin", data)
        entry = _make_entry(checksum="d" * 32, checksum_type="md5")
        with pytest.raises(ChecksumMismatchError):
            _verify_checksum(artifact, entry)

    def test_case_insensitive_match(self, tmp_path: Path):
        data = b"case test"
        digest = _sha256_hex(data).upper()
        artifact = _write_file(tmp_path / "artifact.bin", data)
        entry = _make_entry(checksum=digest, checksum_type="SHA256")
        # Should pass despite uppercase
        _verify_checksum(artifact, entry)


# ---------------------------------------------------------------------------
# _validate_archive_members
# ---------------------------------------------------------------------------


class TestValidateArchiveMembers:
    def test_safe_members_pass(self, tmp_path: Path):
        members = ["data/file.txt", "data/sub/file2.csv"]
        # Should not raise
        _validate_archive_members(members, tmp_path)

    def test_path_traversal_dotdot_rejected(self, tmp_path: Path):
        members = ["../../../etc/passwd"]
        with pytest.raises(DatasetDownloadError, match="Unsafe archive member path"):
            _validate_archive_members(members, tmp_path)

    def test_path_traversal_absolute_rejected(self, tmp_path: Path):
        members = ["/etc/passwd"]
        with pytest.raises(DatasetDownloadError, match="Unsafe archive member path"):
            _validate_archive_members(members, tmp_path)

    def test_mixed_safe_and_unsafe_rejected(self, tmp_path: Path):
        members = ["safe_file.txt", "../../escape.txt"]
        with pytest.raises(DatasetDownloadError, match="Unsafe archive member path"):
            _validate_archive_members(members, tmp_path)

    def test_empty_members_pass(self, tmp_path: Path):
        _validate_archive_members([], tmp_path)


# ---------------------------------------------------------------------------
# _safe_extract_zip
# ---------------------------------------------------------------------------


class TestSafeExtractZip:
    def test_extracts_all_members(self, tmp_path: Path):
        archive_data = _make_zip_bytes(
            {"root/a.txt": b"aaa", "root/b.txt": b"bbb"}
        )
        archive_path = _write_file(tmp_path / "archive.zip", archive_data)
        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(archive_path, "r") as zf:
            _safe_extract_zip(zf, dest, member=None)
        assert (dest / "root" / "a.txt").read_bytes() == b"aaa"
        assert (dest / "root" / "b.txt").read_bytes() == b"bbb"

    def test_extracts_specific_member_prefix(self, tmp_path: Path):
        archive_data = _make_zip_bytes(
            {
                "wanted/data.csv": b"csv",
                "wanted/info.json": b"json",
                "other/skip.txt": b"skip",
            }
        )
        archive_path = _write_file(tmp_path / "archive.zip", archive_data)
        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(archive_path, "r") as zf:
            _safe_extract_zip(zf, dest, member="wanted")
        assert (dest / "wanted" / "data.csv").read_bytes() == b"csv"
        assert not (dest / "other").exists()

    def test_missing_member_raises(self, tmp_path: Path):
        archive_data = _make_zip_bytes({"a.txt": b"aaa"})
        archive_path = _write_file(tmp_path / "archive.zip", archive_data)
        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(archive_path, "r") as zf:
            with pytest.raises(DatasetDownloadError, match="not found in archive"):
                _safe_extract_zip(zf, dest, member="nonexistent_dir")


# ---------------------------------------------------------------------------
# _safe_extract_tar
# ---------------------------------------------------------------------------


class TestSafeExtractTar:
    def test_extracts_all_members(self, tmp_path: Path):
        archive_data = _make_tar_gz_bytes(
            {"root/a.txt": b"aaa", "root/b.txt": b"bbb"}
        )
        archive_path = _write_file(tmp_path / "archive.tar.gz", archive_data)
        dest = tmp_path / "out"
        dest.mkdir()
        with tarfile.open(archive_path, "r:gz") as tf:
            _safe_extract_tar(tf, dest, member=None)
        assert (dest / "root" / "a.txt").read_bytes() == b"aaa"
        assert (dest / "root" / "b.txt").read_bytes() == b"bbb"

    def test_extracts_specific_member(self, tmp_path: Path):
        archive_data = _make_tar_gz_bytes(
            {"wanted.txt": b"wanted", "skip.txt": b"skip"}
        )
        archive_path = _write_file(tmp_path / "archive.tar.gz", archive_data)
        dest = tmp_path / "out"
        dest.mkdir()
        with tarfile.open(archive_path, "r:gz") as tf:
            _safe_extract_tar(tf, dest, member="wanted.txt")
        assert (dest / "wanted.txt").read_bytes() == b"wanted"
        assert not (dest / "skip.txt").exists()


# ---------------------------------------------------------------------------
# _unpack_artifact
# ---------------------------------------------------------------------------


class TestUnpackArtifact:
    def test_zip_extraction(self, tmp_path: Path):
        content = b"zip inner content"
        archive_bytes = _make_zip_bytes({"test_root/data.bin": content})
        artifact_path = _write_file(tmp_path / "dl" / "data.zip", archive_bytes)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        target_path = cache_dir / "test_root"
        entry = _make_entry(archive_type="zip")

        result = _unpack_artifact(entry, artifact_path, cache_dir, target_path)

        assert result == target_path
        assert target_path.exists()
        assert (target_path / "data.bin").read_bytes() == content

    def test_tar_gz_extraction(self, tmp_path: Path):
        content = b"tar gz content"
        archive_bytes = _make_tar_gz_bytes({"test_root/data.bin": content})
        artifact_path = _write_file(tmp_path / "dl" / "data.tar.gz", archive_bytes)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        target_path = cache_dir / "test_root"
        entry = _make_entry(archive_type="tar.gz")

        result = _unpack_artifact(entry, artifact_path, cache_dir, target_path)

        assert result == target_path
        assert target_path.exists()
        assert (target_path / "data.bin").read_bytes() == content

    def test_plain_tar_extraction(self, tmp_path: Path):
        content = b"plain tar content"
        archive_bytes = _make_tar_bytes({"test_root/info.txt": content})
        artifact_path = _write_file(tmp_path / "dl" / "data.tar", archive_bytes)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        target_path = cache_dir / "test_root"
        entry = _make_entry(archive_type="tar")

        result = _unpack_artifact(entry, artifact_path, cache_dir, target_path)

        assert result == target_path
        assert (target_path / "info.txt").read_bytes() == content

    def test_raw_copy(self, tmp_path: Path):
        data = b"raw file contents"
        artifact_path = _write_file(tmp_path / "dl" / "raw.h5", data)
        target_path = tmp_path / "cache" / "raw_copy.h5"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        entry = _make_entry(archive_type="none")

        result = _unpack_artifact(entry, artifact_path, cache_dir, target_path)

        assert result == target_path
        assert target_path.read_bytes() == data

    def test_file_archive_type_copy(self, tmp_path: Path):
        data = b"file type"
        artifact_path = _write_file(tmp_path / "dl" / "raw.h5", data)
        target_path = tmp_path / "cache" / "copy.h5"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        entry = _make_entry(archive_type="file")

        result = _unpack_artifact(entry, artifact_path, cache_dir, target_path)

        assert result == target_path
        assert target_path.read_bytes() == data

    def test_raw_archive_type_copy(self, tmp_path: Path):
        data = b"raw type"
        artifact_path = _write_file(tmp_path / "dl" / "raw.h5", data)
        target_path = tmp_path / "cache" / "copy.h5"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        entry = _make_entry(archive_type="raw")

        result = _unpack_artifact(entry, artifact_path, cache_dir, target_path)

        assert result == target_path
        assert target_path.read_bytes() == data

    def test_raw_same_path_no_error(self, tmp_path: Path):
        """When artifact_path IS the target_path, no copy should occur."""
        data = b"same path"
        target_path = tmp_path / "cache" / "same.h5"
        _write_file(target_path, data)
        cache_dir = tmp_path / "cache"
        entry = _make_entry(archive_type="none")

        result = _unpack_artifact(entry, target_path, cache_dir, target_path)

        assert result == target_path
        assert target_path.read_bytes() == data

    def test_unsupported_archive_type_raises(self, tmp_path: Path):
        artifact_path = _write_file(tmp_path / "dl" / "data.rar", b"rar stuff")
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        target_path = cache_dir / "test_root"
        entry = _make_entry(archive_type="rar")

        with pytest.raises(DatasetDownloadError, match="Unsupported archive type"):
            _unpack_artifact(entry, artifact_path, cache_dir, target_path)

    def test_missing_expected_root_raises(self, tmp_path: Path):
        """If archive doesn't produce the expected root directory, raise."""
        archive_bytes = _make_zip_bytes({"wrong_name/data.bin": b"x"})
        artifact_path = _write_file(tmp_path / "dl" / "data.zip", archive_bytes)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        target_path = cache_dir / "test_root"
        entry = _make_entry(archive_type="zip")

        with pytest.raises(DatasetDownloadError, match="Expected unpacked root"):
            _unpack_artifact(entry, artifact_path, cache_dir, target_path)


# ---------------------------------------------------------------------------
# _artifact_filename
# ---------------------------------------------------------------------------


class TestArtifactFilename:
    def test_zenodo_content_url(self):
        entry = _make_entry(
            url="https://zenodo.org/api/records/4971113/files/10302017_10cms_bounded_2.h5/content"
        )
        name = _artifact_filename(entry)
        assert name == "10302017_10cms_bounded_2.h5"

    def test_zenodo_url_with_query(self):
        entry = _make_entry(
            url="https://zenodo.org/records/15469831/files/nose_data.mat?download=1"
        )
        name = _artifact_filename(entry)
        assert name == "nose_data.mat"

    @patch("plume_nav_sim.data_zoo.downloader.urllib.request.urlopen")
    def test_dryad_api_filename_from_meta(self, mock_urlopen):
        """Dryad API URLs should fetch metadata to discover the filename."""
        meta_response = MagicMock()
        meta_response.__enter__ = MagicMock(return_value=meta_response)
        meta_response.__exit__ = MagicMock(return_value=False)
        meta_response.read = MagicMock(
            return_value=json.dumps({"path": "experiment_video.mat"}).encode()
        )
        # json.load reads from the file-like object
        mock_urlopen.return_value = meta_response
        # Make json.load work with the mock
        meta_response.readline = MagicMock(side_effect=[b"", b""])

        # We need to make json.load work -- patch it through the context manager
        # Actually, json.load calls read() on the response. Let's use a BytesIO.
        payload = json.dumps({"path": "experiment_video.mat"}).encode()
        fake_resp = io.BytesIO(payload)
        mock_urlopen.return_value.__enter__.return_value = fake_resp

        entry = _make_entry(
            url="https://datadryad.org/api/v2/files/852661/download"
        )
        name = _artifact_filename(entry)
        assert name == "experiment_video.mat"

    @patch("plume_nav_sim.data_zoo.downloader.urllib.request.urlopen")
    def test_dryad_api_fallback_on_error(self, mock_urlopen):
        """If metadata fetch fails, fallback to dryad_file_<id>."""
        mock_urlopen.side_effect = Exception("network error")
        entry = _make_entry(
            url="https://datadryad.org/api/v2/files/852661/download"
        )
        name = _artifact_filename(entry)
        assert name == "dryad_file_852661"

    def test_simple_url_filename(self):
        entry = _make_entry(url="https://example.com/datasets/my_data.tar.gz")
        name = _artifact_filename(entry)
        assert name == "my_data.tar.gz"

    def test_url_with_encoded_characters(self):
        entry = _make_entry(
            url="https://zenodo.org/api/records/123/files/my%20file.h5/content"
        )
        name = _artifact_filename(entry)
        assert name == "my file.h5"

    def test_empty_path_fallback(self):
        entry = _make_entry(
            url="https://example.com/",
            archive_type="tar.gz",
            dataset_id="fallback_ds",
        )
        name = _artifact_filename(entry)
        # Fallback: dataset_id + sanitized archive type
        assert name == "fallback_ds.tar_gz"


# ---------------------------------------------------------------------------
# _resolve_dryad_presigned_url
# ---------------------------------------------------------------------------


class TestResolveDryadPresignedUrl:
    def test_non_dryad_url_returned_unchanged(self):
        url = "https://zenodo.org/records/123/files/data.h5"
        assert _resolve_dryad_presigned_url(url) == url

    def test_dryad_non_api_url_returned_unchanged(self):
        url = "https://datadryad.org/some/other/path"
        assert _resolve_dryad_presigned_url(url) == url

    def test_dryad_api_without_download_suffix_returned_unchanged(self):
        url = "https://datadryad.org/api/v2/files/852661"
        assert _resolve_dryad_presigned_url(url) == url

    @patch("plume_nav_sim.data_zoo.downloader.urllib.request.urlopen")
    def test_dryad_resolves_redirect(self, mock_urlopen):
        original = "https://datadryad.org/api/v2/files/852661/download"
        presigned = "https://s3.amazonaws.com/datadryad/bucket/file?sig=abc"
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.geturl.return_value = presigned
        mock_urlopen.return_value = mock_resp

        result = _resolve_dryad_presigned_url(original)
        assert result == presigned

    @patch("plume_nav_sim.data_zoo.downloader.urllib.request.urlopen")
    def test_dryad_no_redirect_raises(self, mock_urlopen):
        original = "https://datadryad.org/api/v2/files/852661/download"
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        # Returns the same URL (no redirect)
        mock_resp.geturl.return_value = original
        mock_urlopen.return_value = mock_resp

        with pytest.raises(DatasetDownloadError, match="did not resolve"):
            _resolve_dryad_presigned_url(original)

    @patch("plume_nav_sim.data_zoo.downloader.urllib.request.urlopen")
    def test_dryad_network_error_raises(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("timeout")
        url = "https://datadryad.org/api/v2/files/852661/download"
        with pytest.raises(DatasetDownloadError, match="Failed to resolve"):
            _resolve_dryad_presigned_url(url)


# ---------------------------------------------------------------------------
# _download_artifact
# ---------------------------------------------------------------------------


class TestDownloadArtifact:
    @patch("plume_nav_sim.data_zoo.downloader._resolve_dryad_presigned_url")
    @patch("plume_nav_sim.data_zoo.downloader.urllib.request.urlopen")
    def test_downloads_to_dest(self, mock_urlopen, mock_resolve, tmp_path: Path):
        mock_resolve.side_effect = lambda u: u  # pass-through
        payload = b"downloaded content"

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        # Make shutil.copyfileobj work by providing a readable stream
        mock_resp.read = MagicMock(side_effect=[payload, b""])
        # Headers for Content-Length
        mock_headers = MagicMock()
        mock_headers.get.return_value = str(len(payload))
        mock_resp.headers = mock_headers

        mock_urlopen.return_value = mock_resp

        dest = tmp_path / "downloads" / "artifact.bin"
        entry = _make_entry(url="https://example.com/data.bin")

        _download_artifact(entry, dest)

        assert dest.exists()
        assert dest.read_bytes() == payload

    @patch("plume_nav_sim.data_zoo.downloader._resolve_dryad_presigned_url")
    @patch("plume_nav_sim.data_zoo.downloader.urllib.request.urlopen")
    def test_download_failure_raises(self, mock_urlopen, mock_resolve, tmp_path: Path):
        mock_resolve.side_effect = lambda u: u
        mock_urlopen.side_effect = Exception("connection refused")

        dest = tmp_path / "downloads" / "artifact.bin"
        entry = _make_entry(url="https://example.com/data.bin")

        with pytest.raises(DatasetDownloadError, match="Failed to download"):
            _download_artifact(entry, dest)

    @patch("plume_nav_sim.data_zoo.downloader._resolve_dryad_presigned_url")
    @patch("plume_nav_sim.data_zoo.downloader.urllib.request.urlopen")
    def test_partial_download_raises(self, mock_urlopen, mock_resolve, tmp_path: Path):
        """If Content-Length doesn't match actual bytes, should raise."""
        mock_resolve.side_effect = lambda u: u
        payload = b"short"

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(side_effect=[payload, b""])
        mock_headers = MagicMock()
        # Claim the file should be much larger than what we deliver
        mock_headers.get.return_value = "999999"
        mock_resp.headers = mock_headers
        mock_urlopen.return_value = mock_resp

        dest = tmp_path / "downloads" / "artifact.bin"
        entry = _make_entry(url="https://example.com/data.bin")

        with pytest.raises(DatasetDownloadError, match="Partial download|Failed to download"):
            _download_artifact(entry, dest)

    @patch("plume_nav_sim.data_zoo.downloader._resolve_dryad_presigned_url")
    @patch("plume_nav_sim.data_zoo.downloader.urllib.request.urlopen")
    def test_env_override_url(self, mock_urlopen, mock_resolve, tmp_path: Path, monkeypatch):
        """PLUME_DATA_ZOO_URL_OVERRIDE_<id> should override the artifact URL."""
        override_url = "https://mirror.example.com/data.bin"
        monkeypatch.setenv("PLUME_DATA_ZOO_URL_OVERRIDE_test_ds", override_url)
        mock_resolve.side_effect = lambda u: u

        payload = b"overridden content"
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(side_effect=[payload, b""])
        mock_headers = MagicMock()
        mock_headers.get.return_value = str(len(payload))
        mock_resp.headers = mock_headers
        mock_urlopen.return_value = mock_resp

        dest = tmp_path / "downloads" / "artifact.bin"
        entry = _make_entry(url="https://example.com/original.bin")

        _download_artifact(entry, dest)

        # Verify _resolve_dryad_presigned_url was called with the override URL
        mock_resolve.assert_called_once_with(override_url)


# ---------------------------------------------------------------------------
# Integration-style: checksum + unpack round-trip
# ---------------------------------------------------------------------------


class TestChecksumAndUnpackRoundTrip:
    def test_sha256_zip_roundtrip(self, tmp_path: Path):
        """Create a ZIP, verify its checksum, unpack, and confirm contents."""
        inner = b"round trip content"
        archive_bytes = _make_zip_bytes({"test_root/payload.dat": inner})
        digest = _sha256_hex(archive_bytes)

        artifact_path = _write_file(tmp_path / "dl" / "data.zip", archive_bytes)
        entry = _make_entry(checksum=digest, checksum_type="sha256", archive_type="zip")

        # Checksum passes
        _verify_checksum(artifact_path, entry)

        # Unpack succeeds
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        target = cache_dir / "test_root"
        result = _unpack_artifact(entry, artifact_path, cache_dir, target)
        assert (result / "payload.dat").read_bytes() == inner

    def test_md5_tar_gz_roundtrip(self, tmp_path: Path):
        inner = b"tar gz round trip"
        archive_bytes = _make_tar_gz_bytes({"test_root/file.txt": inner})
        digest = _md5_hex(archive_bytes)

        artifact_path = _write_file(tmp_path / "dl" / "data.tar.gz", archive_bytes)
        entry = _make_entry(
            checksum=digest, checksum_type="md5", archive_type="tar.gz"
        )

        _verify_checksum(artifact_path, entry)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        target = cache_dir / "test_root"
        result = _unpack_artifact(entry, artifact_path, cache_dir, target)
        assert (result / "file.txt").read_bytes() == inner
