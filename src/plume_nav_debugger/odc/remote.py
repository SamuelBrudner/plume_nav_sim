from __future__ import annotations

"""Remote attach groundwork for ODC providers.

Defines a minimal Transport interface and a RemoteProvider that speaks the ODC
JSON wire format (see wire.py). This is intentionally light-weight and does not
open sockets or threads; concrete transports can be provided by applications.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from .models import ActionInfo, PipelineInfo
from .provider import DebuggerProvider
from .schemas import ActionInfoSchema, PipelineInfoSchema
from .wire import (
    ODCRequest,
    ODCResponse,
    PolicyDistributionSchema,
    action_info_from_schema,
    distribution_from_schema,
    pipeline_info_from_schema,
)


class Transport:
    """Abstract request/response transport for remote ODC calls."""

    def request(self, req: ODCRequest) -> ODCResponse:  # pragma: no cover - interface
        raise NotImplementedError


class CallableTransport(Transport):
    """Transport backed by a callable func(ODCRequest) -> ODCResponse.

    Useful for embedding a trivial in-process handler or tests.
    """

    def __init__(self, handler: Callable[[ODCRequest], ODCResponse]) -> None:
        self._handler = handler

    def request(self, req: ODCRequest) -> ODCResponse:
        return self._handler(req)


class RemoteProvider(DebuggerProvider):
    """DebuggerProvider that proxies to a remote endpoint via a Transport.

    The remote endpoint is expected to implement the same ODC semantics and
    respond with JSON that validates against the Pydantic schemas.
    """

    def __init__(self, transport: Transport) -> None:
        self._tx = transport

    # Actions ---------------------------------------------------------------
    def get_action_info(
        self, env: Any
    ) -> Optional[ActionInfo]:  # noqa: ARG002 - env unused for remote
        req = ODCRequest(method="get_action_info", params={})
        resp = self._tx.request(req)
        if not resp.ok or not isinstance(resp.result, dict):
            return None
        try:
            schema = ActionInfoSchema(**resp.result)
            return action_info_from_schema(schema)
        except Exception:
            return None

    # Distribution ----------------------------------------------------------
    def policy_distribution(
        self, policy: Any, observation: Any
    ) -> Optional[dict]:  # noqa: ARG002 - policy unused for remote
        try:
            obs = (
                observation.tolist()
                if isinstance(observation, np.ndarray)
                else observation
            )
        except Exception:
            obs = observation
        req = ODCRequest(method="policy_distribution", params={"observation": obs})
        resp = self._tx.request(req)
        if not resp.ok or not isinstance(resp.result, dict):
            return None
        try:
            schema = PolicyDistributionSchema(**resp.result)
            return distribution_from_schema(schema)
        except Exception:
            return None

    # Pipeline --------------------------------------------------------------
    def get_pipeline(
        self, env: Any
    ) -> Optional[PipelineInfo]:  # noqa: ARG002 - env unused for remote
        req = ODCRequest(method="get_pipeline", params={})
        resp = self._tx.request(req)
        if not resp.ok or not isinstance(resp.result, dict):
            return None
        try:
            schema = PipelineInfoSchema(**resp.result)
            return pipeline_info_from_schema(schema)
        except Exception:
            return None


__all__ = ["Transport", "CallableTransport", "RemoteProvider"]
