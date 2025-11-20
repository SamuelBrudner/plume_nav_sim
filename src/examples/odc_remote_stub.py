from __future__ import annotations

"""Minimal example of RemoteProvider with a callable transport.

This demonstrates how a debugger process could proxy ODC methods to a remote
provider process. Here we simulate the remote side with an in-process handler.
"""

from typing import Any, Dict

from plume_nav_debugger.odc.models import ActionInfo, PipelineInfo
from plume_nav_debugger.odc.remote import CallableTransport, RemoteProvider
from plume_nav_debugger.odc.wire import ODCRequest, ODCResponse


def _handler(req: ODCRequest) -> ODCResponse:
    if req.method == "get_action_info":
        return ODCResponse(ok=True, result={"names": ["FORWARD", "LEFT", "RIGHT"]})
    if req.method == "get_pipeline":
        return ODCResponse(ok=True, result={"names": ["TopEnv", "Wrapper", "CoreEnv"]})
    if req.method == "policy_distribution":
        # Always return a peaked distribution on index 0 for demo purposes
        obs = req.params.get("observation", None)
        _ = obs  # unused
        return ODCResponse(ok=True, result={"probs": [1.0, 0.0, 0.0]})
    if req.method == "describe_observation":
        return ODCResponse(ok=True, result={"kind": "vector", "label": "demo"})
    return ODCResponse(ok=False, error=f"unknown method: {req.method}")


def build_remote_provider() -> RemoteProvider:
    return RemoteProvider(CallableTransport(_handler))


if __name__ == "__main__":
    prov = build_remote_provider()
    # Methods ignore env/policy on remote side
    print(prov.get_action_info(env=None))
    print(prov.get_pipeline(env=None))
    print(prov.policy_distribution(policy=None, observation=[0.1, 0.2, 0.3]))
