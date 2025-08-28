import logging
from typing import Dict, Any

import pytest
from mypy import api as mypy_api

from plume_nav_sim import protocols as sim_protocols

logger = logging.getLogger(__name__)

PROTOCOL_SPECS: Dict[str, Dict[str, Any]] = {
    "NavigatorProtocol": {
        "methods": {
            "reset": lambda self: None,
            "step": lambda self, dt: None,
            "get_state": lambda self: {},
        },
        "missing": "step",
    },
    "SensorProtocol": {
        "methods": {
            "detect": lambda self, plume_state, positions: [False],
            "measure": lambda self, plume_state, positions: [0.0],
        },
        "missing": "measure",
    },
    "PlumeModelProtocol": {
        "methods": {
            "concentration_at": lambda self, positions: [0.0],
            "step": lambda self, dt: None,
            "reset": lambda self: None,
        },
        "missing": "step",
    },
    "WindFieldProtocol": {
        "methods": {
            "velocity_at": lambda self, positions: [(0.0, 0.0)],
            "step": lambda self, dt: None,
            "reset": lambda self: None,
        },
        "missing": "step",
    },
    "PerformanceMonitorProtocol": {
        "methods": {
            "record_step": lambda self, duration_ms, label=None: None,
            "export": lambda self: {"dummy": 0.0},
        },
        "missing": "export",
    },
}


@pytest.mark.parametrize("protocol_name,spec", PROTOCOL_SPECS.items())
def test_protocol_structural_compatibility(protocol_name: str, spec: Dict[str, Any]) -> None:
    logger.info("Validating positive case for %s", protocol_name)
    protocol_cls = getattr(sim_protocols, protocol_name)
    dummy_cls = type(f"Dummy{protocol_name.replace('Protocol', '')}", (), spec["methods"])
    instance = dummy_cls()
    assert isinstance(instance, protocol_cls)


@pytest.mark.parametrize("protocol_name,spec", PROTOCOL_SPECS.items())
def test_protocol_missing_method_runtime(protocol_name: str, spec: Dict[str, Any]) -> None:
    logger.info("Validating runtime failure for %s", protocol_name)
    protocol_cls = getattr(sim_protocols, protocol_name)
    methods = {name: fn for name, fn in spec["methods"].items() if name != spec["missing"]}
    incomplete_cls = type(f"Incomplete{protocol_name.replace('Protocol', '')}", (), methods)
    instance = incomplete_cls()
    assert not isinstance(instance, protocol_cls)


@pytest.mark.parametrize("protocol_name", PROTOCOL_SPECS.keys())
def test_protocol_missing_method_mypy(protocol_name: str) -> None:
    logger.info("Running mypy negative test for %s", protocol_name)
    class_name = f"Incomplete{protocol_name.replace('Protocol', '')}"
    snippet = f"""
from plume_nav_sim.protocols import {protocol_name}
class {class_name}:
    pass

def accept(obj: {protocol_name}) -> None: ...
accept({class_name}())
"""
    stdout, stderr, exit_status = mypy_api.run(["--strict", "-c", snippet])
    logger.info("mypy output for %s: %s", protocol_name, stdout.strip())
    assert exit_status != 0
