from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# Candidate keys (in priority order) used to locate a concentration reading when
# the observation is a dict. These mirror common observation adapters in the
# codebase.
_DEFAULT_CONCENTRATION_KEYS = (
    "sensor_reading",
    "concentration",
    "observation",
    "odor",
    "odor_concentration",
)

_NUMERIC_TYPES = (int, float, np.integer, np.floating)


def _is_flat_numeric_sequence(seq: Sequence[Any]) -> bool:
    return all(isinstance(v, _NUMERIC_TYPES) for v in seq)


def extract_concentration(
    observation: Any,
    *,
    policy_name: str,
    concentration_key: str | None,
    modality_index: int,
    sensor_index: int | None,
) -> float:
    """Extract a scalar concentration value from heterogeneous observations.

    Supports:
    - numpy arrays or numeric scalars (shape () or (1,))
    - dict observations with a concentration-bearing key
    - tuple/list observations treated as multi-modal containers (selects
      modality_index and recurses)

    Raises:
        ValueError: If no scalar concentration can be extracted, with guidance
        on how to adapt the observation (e.g., set sensor_index or
        concentration_key).
    """

    def _coerce(value: Any) -> float:
        if isinstance(value, Mapping):
            return _from_mapping(value)
        if isinstance(value, np.ndarray):
            return _from_array(value)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            if len(value) == 0:
                raise ValueError(
                    f"{policy_name} received an empty observation sequence; cannot "
                    "extract a concentration value."
                )
            if _is_flat_numeric_sequence(value):
                return _from_array(np.asarray(value, dtype=np.float32))
            selected = _select_sequence_value(value)
            return _coerce(selected)
        return _from_array(np.asarray(value, dtype=np.float32))

    def _from_mapping(mapping: Mapping[str, Any]) -> float:
        if concentration_key is not None:
            if concentration_key not in mapping:
                raise ValueError(
                    f"{policy_name} concentration_key '{concentration_key}' not found "
                    f"in observation keys {sorted(mapping.keys())}."
                )
            return _coerce(mapping[concentration_key])

        for key in _DEFAULT_CONCENTRATION_KEYS:
            if key in mapping:
                return _coerce(mapping[key])

        raise ValueError(
            f"{policy_name} expected a concentration entry in the observation dict. "
            f"Provide concentration_key or include one of "
            f"{list(_DEFAULT_CONCENTRATION_KEYS)}; received keys {sorted(mapping.keys())}."
        )

    def _from_array(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape == ():
            return float(arr)
        if arr.shape == (1,):
            return float(arr[0])
        if arr.ndim == 1:
            if sensor_index is None:
                raise ValueError(
                    f"{policy_name} requires a scalar concentration but received a "
                    f"1D observation of length {arr.shape[0]}. "
                    "Provide sensor_index to select one element or wrap the "
                    "environment to emit a scalar concentration."
                )
            if sensor_index < 0 or sensor_index >= arr.shape[0]:
                raise ValueError(
                    f"{policy_name} sensor_index {sensor_index} is out of bounds for "
                    f"observation of length {arr.shape[0]}."
                )
            return float(arr[int(sensor_index)])

        raise ValueError(
            f"{policy_name} requires a 1D scalar concentration; received array with "
            f"shape {arr.shape}. Provide a preprocessing wrapper or adapter that "
            "returns a single concentration value."
        )

    def _select_sequence_value(seq: Sequence[Any]) -> Any:
        if modality_index < 0 or modality_index >= len(seq):
            raise ValueError(
                f"{policy_name} modality_index {modality_index} is out of bounds for "
                f"observation of length {len(seq)}."
            )
        return seq[modality_index]

    return _coerce(observation)
