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


def _select_sequence_value(
    seq: Sequence[Any], *, policy_name: str, modality_index: int
) -> Any:
    if modality_index < 0 or modality_index >= len(seq):
        raise ValueError(
            f"{policy_name} modality_index {modality_index} is out of bounds for "
            f"observation of length {len(seq)}."
        )
    return seq[modality_index]


def _from_array(
    arr: np.ndarray, *, policy_name: str, sensor_index: int | None
) -> float:
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


def _coerce_concentration(
    value: Any,
    *,
    policy_name: str,
    concentration_key: str | None,
    modality_index: int,
    sensor_index: int | None,
) -> float:
    if isinstance(value, Mapping):
        return _from_mapping(
            value,
            policy_name=policy_name,
            concentration_key=concentration_key,
            modality_index=modality_index,
            sensor_index=sensor_index,
        )
    if isinstance(value, np.ndarray):
        return _from_array(value, policy_name=policy_name, sensor_index=sensor_index)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _from_sequence(
            value,
            policy_name=policy_name,
            concentration_key=concentration_key,
            modality_index=modality_index,
            sensor_index=sensor_index,
        )
    return _from_array(
        np.asarray(value, dtype=np.float32),
        policy_name=policy_name,
        sensor_index=sensor_index,
    )


def _from_mapping(
    mapping: Mapping[str, Any],
    *,
    policy_name: str,
    concentration_key: str | None,
    modality_index: int,
    sensor_index: int | None,
) -> float:
    if concentration_key is not None:
        if concentration_key not in mapping:
            raise ValueError(
                f"{policy_name} concentration_key '{concentration_key}' not found "
                f"in observation keys {sorted(mapping.keys())}."
            )
        return _coerce_concentration(
            mapping[concentration_key],
            policy_name=policy_name,
            concentration_key=concentration_key,
            modality_index=modality_index,
            sensor_index=sensor_index,
        )

    for key in _DEFAULT_CONCENTRATION_KEYS:
        if key in mapping:
            return _coerce_concentration(
                mapping[key],
                policy_name=policy_name,
                concentration_key=concentration_key,
                modality_index=modality_index,
                sensor_index=sensor_index,
            )

    raise ValueError(
        f"{policy_name} expected a concentration entry in the observation dict. "
        f"Provide concentration_key or include one of "
        f"{list(_DEFAULT_CONCENTRATION_KEYS)}; received keys {sorted(mapping.keys())}."
    )


def _from_sequence(
    seq: Sequence[Any],
    *,
    policy_name: str,
    concentration_key: str | None,
    modality_index: int,
    sensor_index: int | None,
) -> float:
    if len(seq) == 0:
        raise ValueError(
            f"{policy_name} received an empty observation sequence; cannot "
            "extract a concentration value."
        )
    if _is_flat_numeric_sequence(seq):
        return _from_array(
            np.asarray(seq, dtype=np.float32),
            policy_name=policy_name,
            sensor_index=sensor_index,
        )
    selected = _select_sequence_value(
        seq, policy_name=policy_name, modality_index=modality_index
    )
    return _coerce_concentration(
        selected,
        policy_name=policy_name,
        concentration_key=concentration_key,
        modality_index=modality_index,
        sensor_index=sensor_index,
    )


def extract_concentration(
    observation: Any,
    *,
    policy_name: str,
    concentration_key: str | None,
    modality_index: int,
    sensor_index: int | None,
) -> float:
    return _coerce_concentration(
        observation,
        policy_name=policy_name,
        concentration_key=concentration_key,
        modality_index=modality_index,
        sensor_index=sensor_index,
    )
