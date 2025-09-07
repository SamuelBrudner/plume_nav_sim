"""Tests for ParquetRecorder dependency handling."""

import importlib
import importlib.metadata
import sys
import pytest


class _DummyDistribution:
    version = "0.0"


importlib.metadata.distribution = lambda name: _DummyDistribution()


def test_parquet_import_requires_pyarrow():
    sys.modules.pop("plume_nav_sim.recording.backends.parquet", None)
    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.recording.backends.parquet")
