import csv
from loguru import logger
import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Minimal pandas stub to satisfy TimeVaryingWindField import
class Series:
    def __init__(self, data):
        self.values = np.array(data)


class DataFrame:
    def __init__(self, data):
        self._data = data
        self.columns = list(data.keys())

    def __getitem__(self, key):
        if isinstance(key, list):
            return DataFrame({k: self._data[k] for k in key})
        return Series(self._data[key])

    @property
    def values(self):
        return np.column_stack([self._data[col] for col in self.columns])

def read_csv(path):
    data = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return DataFrame(data)

pandas_stub = types.ModuleType("pandas")
pandas_stub.DataFrame = DataFrame
pandas_stub.read_csv = read_csv
sys.modules.setdefault("pandas", pandas_stub)

# Load module directly to avoid heavy package imports
module_path = Path(__file__).resolve().parents[2] / "src/plume_nav_sim/models/wind/time_varying_wind.py"
spec = importlib.util.spec_from_file_location("time_varying_wind", module_path)
wind_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wind_module
spec.loader.exec_module(wind_module)
TimeVaryingWindField = wind_module.TimeVaryingWindField


def test_interpolation_failure_raises(tmp_path, caplog):
    """Interpolation failures should propagate exceptions with context."""
    data_file = tmp_path / "wind.csv"
    with open(data_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "u_wind", "v_wind"])
        writer.writerow([0, 0.0, 0.0])
        writer.writerow([1, 1.0, 1.0])

    with caplog.at_level(logger.ERROR, logger="time_varying_wind"):
        with pytest.raises(Exception):
            TimeVaryingWindField(
                temporal_pattern="measured",
                data_file=str(data_file),
                interpolation_method="cubic",
                temporal_column="time",
                velocity_columns=["u_wind", "v_wind"],
            )

    log_messages = " ".join(record.message for record in caplog.records)
    assert "cubic" in log_messages
    assert "[0. 1.]" in log_messages
