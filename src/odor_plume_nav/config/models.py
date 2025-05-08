"""
Alias Pydantic configuration models from the domain layer.
"""
from odor_plume_nav.domain.models import (
    NavigatorConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    VideoPlumeConfig,
)

__all__ = [
    "NavigatorConfig",
    "SingleAgentConfig",
    "MultiAgentConfig",
    "VideoPlumeConfig",
]
