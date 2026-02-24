"""Tests for the Data Zoo CLI.

All tests are offline-safe — no network access required.
"""

from __future__ import annotations

import pytest

from plume_nav_sim.data_zoo.cli import main
from plume_nav_sim.data_zoo.registry import DATASET_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(argv: list[str], capsys) -> tuple[int, str, str]:
    """Run the CLI and capture output. Returns (exit_code, stdout, stderr)."""
    rc = main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


# ---------------------------------------------------------------------------
# list subcommand
# ---------------------------------------------------------------------------


class TestList:
    def test_exits_zero(self, capsys):
        rc, out, _err = _run(["list"], capsys)
        assert rc == 0

    def test_prints_all_dataset_ids(self, capsys):
        _rc, out, _err = _run(["list"], capsys)
        for did in DATASET_REGISTRY:
            assert did in out

    def test_prints_header(self, capsys):
        _rc, out, _err = _run(["list"], capsys)
        assert "DATASET_ID" in out
        assert "VERSION" in out


# ---------------------------------------------------------------------------
# describe subcommand
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_known_dataset(self, capsys):
        rc, out, _err = _run(["describe", "colorado_jet_v1"], capsys)
        assert rc == 0
        assert "colorado_jet_v1" in out
        assert "10.5061/dryad.g27mq71" in out
        assert "PLIF" in out or "plume" in out.lower()

    def test_shows_creators(self, capsys):
        _rc, out, _err = _run(["describe", "rigolli_dns_nose_v1"], capsys)
        assert "Rigolli" in out

    def test_shows_citation(self, capsys):
        _rc, out, _err = _run(["describe", "colorado_jet_v1"], capsys)
        assert "Citation:" in out

    def test_unknown_dataset(self, capsys):
        rc, _out, err = _run(["describe", "nonexistent_dataset"], capsys)
        assert rc == 1
        assert "nonexistent_dataset" in err

    def test_unknown_lists_available(self, capsys):
        _rc, _out, err = _run(["describe", "nonexistent_dataset"], capsys)
        assert "colorado_jet_v1" in err


# ---------------------------------------------------------------------------
# download subcommand
# ---------------------------------------------------------------------------


class TestDownload:
    def test_unknown_dataset(self, capsys):
        rc, _out, _err = _run(["download", "nonexistent_dataset"], capsys)
        assert rc == 1

    def test_download_fails_gracefully_no_network(self, capsys, tmp_path, monkeypatch):
        """Download with empty cache should fail with exit code 2 (no network)."""
        import urllib.error

        def _fake_urlopen(*args, **kwargs):
            raise urllib.error.URLError("mocked network failure")

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

        rc, _out, _err = _run(
            ["--cache-root", str(tmp_path), "download", "colorado_jet_v1"],
            capsys,
        )
        # Should fail because there's no cached data and network is mocked out.
        # Exit code 2 = download failure.
        assert rc == 2


# ---------------------------------------------------------------------------
# cache-status subcommand
# ---------------------------------------------------------------------------


class TestCacheStatus:
    def test_exits_zero(self, capsys):
        rc, out, _err = _run(["cache-status"], capsys)
        assert rc == 0

    def test_shows_all_datasets(self, capsys):
        _rc, out, _err = _run(["cache-status"], capsys)
        for did in DATASET_REGISTRY:
            assert did in out

    def test_empty_cache_shows_no(self, capsys, tmp_path):
        _rc, out, _err = _run(
            ["--cache-root", str(tmp_path), "cache-status"], capsys
        )
        # With a fresh tmp cache, nothing should be cached
        lines = out.strip().split("\n")
        # Skip header and separator
        data_lines = lines[2:]
        for line in data_lines:
            assert "no" in line


# ---------------------------------------------------------------------------
# Top-level / help
# ---------------------------------------------------------------------------


class TestTopLevel:
    def test_no_command_exits_zero(self, capsys):
        rc, out, _err = _run([], capsys)
        assert rc == 0

    def test_no_command_prints_usage(self, capsys):
        _rc, out, _err = _run([], capsys)
        assert "plume-nav-data-zoo" in out or "usage" in out.lower()

    def test_help_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# validate subcommand
# ---------------------------------------------------------------------------


class TestValidate:
    def test_exits_zero_on_valid_registry(self, capsys):
        rc, out, _err = _run(["validate"], capsys)
        assert rc == 0

    def test_prints_ok_message(self, capsys):
        rc, out, _err = _run(["validate"], capsys)
        assert "ok" in out.lower()
        assert str(len(DATASET_REGISTRY)) in out

    def test_reports_failure_on_bad_registry(self, monkeypatch, capsys):
        """Inject a broken entry and confirm validate reports FAIL."""
        from plume_nav_sim.data_zoo.registry import (
            DatasetArtifact,
            DatasetMetadata,
            DatasetRegistryEntry,
        )

        bad_entry = DatasetRegistryEntry(
            dataset_id="bad_id",
            version="0.0.0",
            cache_subdir="test",
            expected_root="test",
            artifact=DatasetArtifact(
                url="https://example.com/test.zip",
                checksum="abc123",
            ),
            metadata=DatasetMetadata(
                title="",  # empty title triggers validation error
                description="test",
                license="MIT",
            ),
        )
        patched = dict(DATASET_REGISTRY)
        patched["bad_id"] = bad_entry
        monkeypatch.setattr(
            "plume_nav_sim.data_zoo.cli.DATASET_REGISTRY", patched
        )
        monkeypatch.setattr(
            "plume_nav_sim.data_zoo.registry.DATASET_REGISTRY", patched
        )

        rc, _out, err = _run(["validate"], capsys)
        assert rc == 1
        assert "FAIL" in err
