import pytest

from plume_nav_sim.data_zoo.registry import (
    DATASET_REGISTRY,
    DatasetArtifact,
    DatasetMetadata,
    DatasetRegistryEntry,
    RegistryValidationError,
    describe_dataset,
    validate_registry,
)


def test_describe_dataset_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        describe_dataset("does_not_exist")


def test_validate_registry_accepts_seed_entries() -> None:
    # Should not raise for bundled registry data
    validate_registry(DATASET_REGISTRY)


def test_validate_registry_flags_missing_metadata() -> None:
    entry = DatasetRegistryEntry(
        dataset_id="incomplete",
        version="0.0.1",
        cache_subdir="incomplete",
        expected_root="data.zarr",
        artifact=DatasetArtifact(
            url="file:///tmp/incomplete.zip",
            checksum="deadbeef",
            archive_type="zip",
            layout="zarr",
        ),
        metadata=DatasetMetadata(
            title="",
            description="Some description",
            citation="Some citation",
            license="",
        ),
    )

    with pytest.raises(RegistryValidationError) as excinfo:
        validate_registry({entry.dataset_id: entry})

    msg = str(excinfo.value)
    assert "metadata.title" in msg or "metadata.license" in msg
