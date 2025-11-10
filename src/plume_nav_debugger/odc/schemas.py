from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ActionInfoSchema(BaseModel):
    """Pydantic schema for action space presentation info."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    names: List[str] = Field(default_factory=list)


class ObservationInfoSchema(BaseModel):
    """Pydantic schema for observation presentation hints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["vector", "image", "scalar", "unknown"] = "unknown"
    label: Optional[str] = None


class PipelineInfoSchema(BaseModel):
    """Pydantic schema for observation pipeline composition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    names: List[str] = Field(default_factory=list)
