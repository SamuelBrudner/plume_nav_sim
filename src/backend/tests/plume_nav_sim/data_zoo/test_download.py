import hashlib
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest

from plume_nav_sim.data_zoo import download as download_module
from plume_nav_sim.data_zoo.download import (
    DatasetDownloadError,
    ensure_dataset_available,
)
from plume_nav_sim.data_zoo.registry import (
    DATASET_REGISTRY,
    DatasetArtifact,
    DatasetMetadata,
    DatasetRegistryEntry,
    H5ToZarrSpec,
)


def _artifact_filename(entry: DatasetRegistryEntry) -> str:
    name = Path(entry.artifact.url).name
    if name:
        return name
    sanitized = entry.artifact.archive_type.replace(".", "_")
    return f"{entry.dataset_id}.{sanitized}"


def _make_zarr_archive(tmp_path: Path, root_name: str = "demo.zarr") -> Path:
    dataset_root = tmp_path / root_name
    dataset_root.mkdir(parents=True)
    (dataset_root / ".zattrs").write_text("{}", encoding="utf-8")
    archive_path = tmp_path / f"{root_name}.zip"

    with zipfile.ZipFile(archive_path, "w") as zf:
        # Explicitly record the root directory entry so tests that request the
        # directory member (e.g., archive_member="demo.zarr") can resolve it.
        zf.write(dataset_root, Path(root_name).as_posix())
        for path in dataset_root.rglob("*"):
            arcname = Path(root_name) / path.relative_to(dataset_root)
            zf.write(path, arcname.as_posix())

    return archive_path


def _make_invalid_zarr_archive(tmp_path: Path, root_name: str = "broken.zarr") -> Path:
    dataset_root = tmp_path / root_name
    dataset_root.mkdir(parents=True)
    archive_path = tmp_path / f"{root_name}.zip"

    with zipfile.ZipFile(archive_path, "w") as zf:
        # Capture the empty directory structure without .zattrs
        zf.write(dataset_root, Path(root_name).as_posix())

    return archive_path


def _make_hdf5_tar(tmp_path: Path, filename: str = "movie.h5") -> Path:
    payload = tmp_path / filename
    payload.write_bytes(b"dummy h5 payload")
    tar_path = tmp_path / f"{filename}.tar.gz"

    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(payload, arcname=filename)

    return tar_path


def _register_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset_id: str = "demo_dataset"
) -> DatasetRegistryEntry:
    archive_path = _make_zarr_archive(tmp_path)
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    entry = DatasetRegistryEntry(
        dataset_id=dataset_id,
        version="1.0.0",
        cache_subdir=dataset_id,
        expected_root="demo.zarr",
        artifact=DatasetArtifact(
            url=archive_path.as_uri(),
            checksum=checksum,
            archive_type="zip",
            archive_member="demo.zarr",
            layout="zarr",
        ),
        metadata=DatasetMetadata(
            title="Demo dataset",
            description="Test entry for downloader",
            citation="N/A",
            license="MIT",
        ),
    )
    monkeypatch.setitem(DATASET_REGISTRY, dataset_id, entry)
    return entry


def test_downloads_and_validates_zarr_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    entry = _register_entry(tmp_path, monkeypatch)

    resolved = ensure_dataset_available(
        entry.dataset_id,
        cache_root=cache_root,
        auto_download=False,
        confirm_download=lambda _: True,
    )

    assert (
        resolved
        == cache_root / entry.cache_subdir / entry.version / entry.expected_root
    )
    assert (resolved / ".zattrs").exists()
    assert (
        cache_root / entry.cache_subdir / entry.version / _artifact_filename(entry)
    ).exists()


def test_skips_download_when_cache_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "cache"
    entry = _register_entry(tmp_path, monkeypatch)

    first = ensure_dataset_available(
        entry.dataset_id,
        cache_root=cache_root,
        confirm_download=lambda _: True,
    )

    def _fail_download(*_: object) -> None:
        raise AssertionError("Download should not be invoked for cached dataset")

    monkeypatch.setattr(download_module, "_download_artifact", _fail_download)

    second = ensure_dataset_available(
        entry.dataset_id,
        cache_root=cache_root,
        auto_download=True,
    )

    assert first == second


def test_checksum_mismatch_triggers_redownload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    entry = _register_entry(tmp_path, monkeypatch)

    # Seed cache with corrupt artifact and unpacked directory
    cache_dir = entry.cache_path(cache_root)
    cache_dir.mkdir(parents=True)
    artifact_path = cache_dir / _artifact_filename(entry)
    artifact_path.write_bytes(b"corrupted")
    unpacked = cache_dir / entry.expected_root
    unpacked.mkdir()
    (unpacked / ".zattrs").write_text("{}", encoding="utf-8")

    resolved = ensure_dataset_available(
        entry.dataset_id,
        cache_root=cache_root,
        auto_download=True,
    )

    assert resolved.exists()
    assert artifact_path.read_bytes() == Path(entry.artifact.url).read_bytes()


def test_download_decline_raises_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    entry = _register_entry(tmp_path, monkeypatch)

    # Ensure cache is empty
    cache_dir = entry.cache_path(cache_root)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    with pytest.raises(DatasetDownloadError):
        ensure_dataset_available(
            entry.dataset_id,
            cache_root=cache_root,
            auto_download=False,
            confirm_download=lambda _: False,
        )


def test_checksum_mismatch_without_auto_download_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    entry = _register_entry(tmp_path, monkeypatch)

    def _write_corrupt_artifact(_: object, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"not the expected payload")

    monkeypatch.setattr(
        download_module, "_download_artifact", _write_corrupt_artifact, raising=True
    )

    with pytest.raises(download_module.ChecksumMismatchError):
        ensure_dataset_available(
            entry.dataset_id, cache_root=cache_root, auto_download=True
        )


def test_cache_hit_returns_when_artifact_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    entry = _register_entry(tmp_path, monkeypatch)

    cache_dir = entry.cache_path(cache_root)
    expected_path = cache_dir / entry.expected_root
    expected_path.mkdir(parents=True)
    (expected_path / ".zattrs").write_text("{}", encoding="utf-8")

    download_calls = {"count": 0}

    def _fail_if_called(*_: object, **__: object) -> None:
        download_calls["count"] += 1
        raise AssertionError("Download should not be invoked for cached dataset")

    monkeypatch.setattr(download_module, "_download_artifact", _fail_if_called)

    resolved = ensure_dataset_available(
        entry.dataset_id,
        cache_root=cache_root,
        auto_download=True,
        verify_checksum=True,
    )

    assert resolved == expected_path
    assert download_calls["count"] == 0


def test_invalid_zarr_layout_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    archive_path = _make_invalid_zarr_archive(tmp_path, "broken.zarr")
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    entry = DatasetRegistryEntry(
        dataset_id="broken",
        version="0.0.1",
        cache_subdir="broken",
        expected_root="broken.zarr",
        artifact=DatasetArtifact(
            url=archive_path.as_uri(),
            checksum=checksum,
            archive_type="zip",
            archive_member="broken.zarr",
            layout="zarr",
        ),
        metadata=DatasetMetadata(
            title="Broken dataset",
            description="Missing zarr metadata",
            citation="N/A",
            license="MIT",
        ),
    )
    monkeypatch.setitem(DATASET_REGISTRY, entry.dataset_id, entry)

    with pytest.raises(download_module.LayoutValidationError):
        ensure_dataset_available(
            entry.dataset_id, cache_root=cache_root, auto_download=True
        )


def test_hdf5_layout_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    archive_path = _make_hdf5_tar(tmp_path, "movie.h5")
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    entry = DatasetRegistryEntry(
        dataset_id="hdf5_dataset",
        version="0.1.0",
        cache_subdir="hdf5",
        expected_root="movie.h5",
        artifact=DatasetArtifact(
            url=archive_path.as_uri(),
            checksum=checksum,
            archive_type="tar.gz",
            archive_member="movie.h5",
            layout="hdf5",
        ),
        metadata=DatasetMetadata(
            title="HDF5 dataset",
            description="Test hdf5 layout validation",
            citation="N/A",
            license="MIT",
        ),
    )
    monkeypatch.setitem(DATASET_REGISTRY, entry.dataset_id, entry)

    resolved = ensure_dataset_available(
        entry.dataset_id, cache_root=cache_root, auto_download=True
    )

    assert resolved.exists()
    assert resolved.is_file()
    assert resolved.name == "movie.h5"


def test_hdf5_layout_rejects_wrong_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    archive_path = _make_hdf5_tar(tmp_path, "movie.txt")
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    entry = DatasetRegistryEntry(
        dataset_id="bad_hdf5",
        version="0.1.0",
        cache_subdir="bad_hdf5",
        expected_root="movie.txt",
        artifact=DatasetArtifact(
            url=archive_path.as_uri(),
            checksum=checksum,
            archive_type="tar.gz",
            archive_member="movie.txt",
            layout="hdf5",
        ),
        metadata=DatasetMetadata(
            title="Bad HDF5",
            description="Should fail validation",
            citation="N/A",
            license="MIT",
        ),
    )
    monkeypatch.setitem(DATASET_REGISTRY, entry.dataset_id, entry)

    with pytest.raises(download_module.LayoutValidationError):
        ensure_dataset_available(
            entry.dataset_id, cache_root=cache_root, auto_download=True
        )


def test_direct_file_downloads_without_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    payload = tmp_path / "movie.h5"
    payload.write_bytes(b"real h5 payload")
    checksum = hashlib.sha256(payload.read_bytes()).hexdigest()

    entry = DatasetRegistryEntry(
        dataset_id="direct_h5",
        version="0.0.1",
        cache_subdir="direct",
        expected_root="movie.h5",
        artifact=DatasetArtifact(
            url=payload.as_uri(),
            checksum=checksum,
            archive_type="none",
            layout="hdf5",
        ),
        metadata=DatasetMetadata(
            title="Direct file download",
            description="No archive wrapper",
            citation="N/A",
            license="MIT",
        ),
    )
    monkeypatch.setitem(DATASET_REGISTRY, entry.dataset_id, entry)

    resolved = ensure_dataset_available(
        entry.dataset_id, cache_root=cache_root, auto_download=True
    )

    assert (
        resolved
        == cache_root / entry.cache_subdir / entry.version / entry.expected_root
    )
    assert resolved.read_bytes() == payload.read_bytes()


def test_ingest_spec_invokes_postprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    payload = tmp_path / "movie.h5"
    payload.write_bytes(b"hdf5 bytes")
    checksum = hashlib.sha256(payload.read_bytes()).hexdigest()

    entry = DatasetRegistryEntry(
        dataset_id="ingest_h5",
        version="0.0.2",
        cache_subdir="ingest",
        expected_root="movie.zarr",
        artifact=DatasetArtifact(
            url=payload.as_uri(),
            checksum=checksum,
            archive_type="none",
            layout="hdf5",
        ),
        metadata=DatasetMetadata(
            title="Ingest example",
            description="Checks ingest hook",
            citation="N/A",
            license="MIT",
        ),
        ingest=H5ToZarrSpec(
            dataset="/Plume Data/dataset_001",
            fps=10.0,
            pixel_to_grid=(1.0, 1.0),
            origin=(0.0, 0.0),
            extent=(1.0, 1.0),
            normalize=False,
            chunk_t=None,
        ),
    )
    monkeypatch.setitem(DATASET_REGISTRY, entry.dataset_id, entry)

    calls: dict[str, tuple[H5ToZarrSpec, Path, Path]] = {}

    def _fake_ingest(spec: H5ToZarrSpec, source: Path, output: Path) -> Path:
        calls["args"] = (spec, source, output)
        output.mkdir(parents=True, exist_ok=True)
        (output / ".zattrs").write_text("{}", encoding="utf-8")
        return output

    monkeypatch.setattr(download_module, "_ingest_hdf5_to_zarr", _fake_ingest)

    resolved = ensure_dataset_available(
        entry.dataset_id, cache_root=cache_root, auto_download=True
    )

    assert (
        resolved
        == cache_root / entry.cache_subdir / entry.version / entry.expected_root
    )
    assert "args" in calls
    assert calls["args"][1].name == "movie.h5"
    assert (resolved / ".zattrs").exists()
