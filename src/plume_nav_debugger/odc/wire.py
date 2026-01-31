from __future__ import annotations

"""ODC JSON wire helpers.

This module provides:
- Schema models for one-of policy distribution payloads
- Converters between runtime dataclasses and Pydantic schemas
- Minimal request/response envelope types for future remote attach

All helpers are dependency-light and safe to import in headless contexts.
"""

from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as _np
from pydantic import BaseModel, ConfigDict, Field

from .models import ActionInfo, ObservationInfo, PipelineInfo
from .schemas import ActionInfoSchema, ObservationInfoSchema, PipelineInfoSchema


# ----------------------------
# Distribution (one-of) schema
# ----------------------------
class PolicyDistributionSchema(BaseModel):
    """One-of distribution payload used by providers.

    Exactly one of `probs`, `q_values`, or `logits` must be provided and must be
    a non-empty 1D list of floats. Validation is kept minimal; callers should
    still enforce length equals action count when known.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    probs: Optional[List[float]] = None
    q_values: Optional[List[float]] = None
    logits: Optional[List[float]] = None

    def oneof_key(self) -> Optional[str]:
        keys = [
            k for k in ("probs", "q_values", "logits") if getattr(self, k) is not None
        ]
        if len(keys) == 1:
            return keys[0]
        return None


# ----------------------------
# Request/Response envelopes
# ----------------------------
class ODCRequest(BaseModel):
    """Minimal RPC request envelope for remote attach groundwork.

    Current methods:
    - get_action_info: no params
    - get_pipeline: no params
    - policy_distribution: params = { observation: list | nested lists }
    - describe_observation: params = { observation, context? }
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal[
        "get_action_info",
        "get_pipeline",
        "policy_distribution",
        "describe_observation",
    ]
    params: Dict[str, Any] = Field(default_factory=dict)
    id: Optional[str] = None


class ODCResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool = True
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    id: Optional[str] = None


# ---------------------------------------
# Dataclass <-> Schema conversion helpers
# ---------------------------------------
def action_info_to_schema(model: ActionInfo) -> ActionInfoSchema:
    return ActionInfoSchema(names=list(model.names))


def action_info_from_schema(schema: ActionInfoSchema) -> ActionInfo:
    return ActionInfo(names=list(schema.names))


def observation_info_to_schema(model: ObservationInfo) -> ObservationInfoSchema:
    return ObservationInfoSchema(kind=str(model.kind), label=model.label)


def observation_info_from_schema(schema: ObservationInfoSchema) -> ObservationInfo:
    return ObservationInfo(kind=str(schema.kind), label=schema.label)


def pipeline_info_to_schema(model: PipelineInfo) -> PipelineInfoSchema:
    return PipelineInfoSchema(names=list(model.names))


def pipeline_info_from_schema(schema: PipelineInfoSchema) -> PipelineInfo:
    return PipelineInfo(names=list(schema.names))


def distribution_to_schema(
    dist: Dict[str, Sequence[float]]
) -> PolicyDistributionSchema:
    # Trust caller to pass exactly one key; schema will enforce shape minimally
    return PolicyDistributionSchema(**{k: list(map(float, v)) for k, v in dist.items()})


def distribution_from_schema(
    schema: PolicyDistributionSchema,
) -> Dict[str, List[float]]:
    k = schema.oneof_key()
    if k is None:
        return {}
    v = getattr(schema, k)
    return {k: list(v or [])}


# ----------------------------
# Numpy helpers for observations
# ----------------------------
def _to_list(o: Any) -> Any:
    try:
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return o


__all__ = [
    "PolicyDistributionSchema",
    "ODCRequest",
    "ODCResponse",
    "action_info_to_schema",
    "action_info_from_schema",
    "observation_info_to_schema",
    "observation_info_from_schema",
    "pipeline_info_to_schema",
    "pipeline_info_from_schema",
    "distribution_to_schema",
    "distribution_from_schema",
]
