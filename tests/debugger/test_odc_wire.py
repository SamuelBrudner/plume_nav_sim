from __future__ import annotations

from plume_nav_debugger.odc.models import ActionInfo, ObservationInfo, PipelineInfo
from plume_nav_debugger.odc.wire import (
    action_info_from_schema,
    action_info_to_schema,
    distribution_from_schema,
    distribution_to_schema,
    observation_info_from_schema,
    observation_info_to_schema,
    pipeline_info_from_schema,
    pipeline_info_to_schema,
)


def test_action_info_roundtrip():
    model = ActionInfo(names=["RUN", "TUMBLE"])
    schema = action_info_to_schema(model)
    assert schema.names == ["RUN", "TUMBLE"]
    restored = action_info_from_schema(schema)
    assert restored == model


def test_observation_info_roundtrip():
    model = ObservationInfo(kind="vector", label="obs")
    schema = observation_info_to_schema(model)
    assert schema.kind == "vector"
    assert schema.label == "obs"
    restored = observation_info_from_schema(schema)
    assert restored == model


def test_pipeline_info_roundtrip():
    model = PipelineInfo(names=["WrapperA", "CoreEnv"])
    schema = pipeline_info_to_schema(model)
    assert schema.names == ["WrapperA", "CoreEnv"]
    restored = pipeline_info_from_schema(schema)
    assert restored == model


def test_distribution_schema_roundtrip():
    dist = {"probs": [0.2, 0.8]}
    schema = distribution_to_schema(dist)
    assert schema.oneof_key() == "probs"
    restored = distribution_from_schema(schema)
    assert restored == dist
