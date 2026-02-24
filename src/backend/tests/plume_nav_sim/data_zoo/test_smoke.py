"""CI smoke tests for the Data Zoo — no network access required.

These tests verify that the registry, CLI, and public API surface are
internally consistent and fail cleanly when the cache is empty. They
run in every CI job to catch regressions.
"""

from __future__ import annotations

import pytest

from plume_nav_sim.data_zoo.cli import main as cli_main
from plume_nav_sim.data_zoo.download import DatasetDownloadError, ensure_dataset_available
from plume_nav_sim.data_zoo.registry import (
    DATASET_REGISTRY,
    describe_dataset,
    validate_registry,
)


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


class TestRegistryIntegrity:
    """Validate the registry schema without network access."""

    def test_validate_registry_passes(self):
        """The full registry passes schema validation."""
        validate_registry()

    def test_all_ids_resolvable(self):
        """Every registered dataset_id resolves via describe_dataset."""
        for did in DATASET_REGISTRY:
            entry = describe_dataset(did)
            assert entry.dataset_id == did
            assert entry.version
            assert entry.metadata.title

    def test_all_entries_have_checksum(self):
        for did, entry in DATASET_REGISTRY.items():
            assert entry.artifact.checksum, f"{did} missing checksum"
            assert entry.artifact.checksum_type in (
                "sha256",
                "md5",
            ), f"{did} unexpected checksum_type: {entry.artifact.checksum_type}"

    def test_all_entries_have_url(self):
        for did, entry in DATASET_REGISTRY.items():
            url = entry.artifact.url
            assert url.startswith("https://"), f"{did} URL not HTTPS: {url}"

    def test_no_duplicate_cache_subdirs_with_same_version(self):
        """No two datasets should map to the same cache path."""
        seen: dict[str, str] = {}
        for did, entry in DATASET_REGISTRY.items():
            key = f"{entry.cache_subdir}/{entry.version}/{entry.expected_root}"
            if key in seen:
                pytest.fail(
                    f"{did} and {seen[key]} share cache path {key}"
                )
            seen[key] = did

    def test_unknown_dataset_raises_keyerror(self):
        with pytest.raises(KeyError):
            describe_dataset("this_dataset_does_not_exist_xyzzy")


# ---------------------------------------------------------------------------
# Download gating (offline)
# ---------------------------------------------------------------------------


class TestDownloadGating:
    """Ensure ensure_dataset_available fails cleanly when offline."""

    def test_missing_cache_auto_download_false_raises(self, tmp_path):
        """With empty cache and auto_download=False, should raise."""
        with pytest.raises(DatasetDownloadError, match="missing from cache"):
            ensure_dataset_available(
                "colorado_jet_v1",
                cache_root=tmp_path,
                auto_download=False,
            )

    def test_unknown_dataset_raises_keyerror(self, tmp_path):
        with pytest.raises(KeyError):
            ensure_dataset_available(
                "nonexistent_dataset_zzz",
                cache_root=tmp_path,
                auto_download=False,
            )


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestCLISmoke:
    """Smoke test the CLI subcommands without network access."""

    def test_list_exits_zero(self, capsys):
        rc = cli_main(["list"])
        out = capsys.readouterr().out
        assert rc == 0
        # Should list at least one dataset
        assert "colorado_jet_v1" in out

    def test_describe_exits_zero(self, capsys):
        rc = cli_main(["describe", "colorado_jet_v1"])
        assert rc == 0

    def test_describe_unknown_exits_one(self, capsys):
        rc = cli_main(["describe", "no_such_dataset"])
        assert rc == 1

    def test_cache_status_exits_zero(self, capsys):
        rc = cli_main(["cache-status"])
        assert rc == 0

    def test_validate_exits_zero(self, capsys):
        rc = cli_main(["validate"])
        assert rc == 0


# ---------------------------------------------------------------------------
# load_plume gating (offline)
# ---------------------------------------------------------------------------


class TestLoadPlumeGating:
    """load_plume should fail cleanly when cache is empty."""

    def test_load_plume_raises_when_no_cache(self, tmp_path):
        from plume_nav_sim.data_zoo.loader import load_plume

        with pytest.raises((DatasetDownloadError, KeyError, Exception)):
            load_plume(
                "colorado_jet_v1",
                cache_root=tmp_path,
                auto_download=False,
            )
