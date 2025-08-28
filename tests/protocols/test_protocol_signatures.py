import logging
from typing import Dict, Any

import numpy as np
import pytest
from mypy import api as mypy_api

# Prefer modern protocol exports but retain core import for legacy types
import plume_nav_sim.protocols as protocols_pkg
from plume_nav_sim.core import protocols as core_protocols

logger = logging.getLogger(__name__)

PROTOCOL_SPECS: Dict[str, Dict[str, Any]] = {
    "SourceProtocol": {
        "methods": {
            "get_emission_rate": lambda self: 1.0,
            "get_position": lambda self: (0.0, 0.0),
            "update_state": lambda self, dt: None,
        },
        "missing": "update_state",
    },
    "BoundaryPolicyProtocol": {
        "methods": {
            "apply_policy": lambda self, positions, velocities=None: (
                positions if velocities is None else (positions, velocities)
            ),
            "check_violations": lambda self, positions: np.zeros(1, dtype=bool),
            "get_termination_status": lambda self: "continue",
        },
        "missing": "check_violations",
    },
    "ActionInterfaceProtocol": {
        "methods": {
            "translate_action": lambda self, action: {
                "linear_velocity": 0.0,
                "angular_velocity": 0.0,
            },
            "validate_action": lambda self, action: True,
            "get_action_space": lambda self: None,
        },
        "missing": "validate_action",
    },
    "RecorderProtocol": {
        "methods": {
            "record_step": lambda self, step_data, step_number, episode_id=None, **metadata: None,
            "record_episode": lambda self, episode_data, episode_id, **metadata: None,
            "export_data": lambda self, output_path, format="parquet", compression=None, filter_episodes=None, **export_options: True,
        },
        "missing": "record_episode",
    },
    "StatsAggregatorProtocol": {
        "methods": {
            "calculate_episode_stats": lambda self, trajectory_data, episode_id, custom_metrics=None: {},
            "calculate_run_stats": lambda self, episode_data_list, run_id, statistical_tests=None: {},
            "export_summary": lambda self, output_path, run_data=None, include_distributions=False, format="json": True,
        },
        "missing": "calculate_run_stats",
    },
    "AgentInitializerProtocol": {
        "methods": {
            "initialize_positions": lambda self, num_agents, **kwargs: np.zeros((num_agents, 2)),
            "validate_domain": lambda self, positions, domain_bounds: True,
            "reset": lambda self, seed=None, **kwargs: None,
            "get_strategy_name": lambda self: "dummy",
        },
        "missing": "validate_domain",
    },
    "PlumeModelProtocol": {
        "methods": {
            "concentration_at": lambda self, positions: np.zeros(1),
            "step": lambda self, dt: None,
            "reset": lambda self, **kwargs: None,
        },
        "missing": "step",
    },
    "WindFieldProtocol": {
        "methods": {
            "velocity_at": lambda self, positions: np.zeros_like(positions),
            "step": lambda self, dt: None,
            "reset": lambda self, **kwargs: None,
        },
        "missing": "step",
    },
    "SensorProtocol": {
        "methods": {
            "detect": lambda self, plume_state, positions: np.zeros(1, dtype=bool),
            "measure": lambda self, plume_state, positions: np.zeros(1),
            "compute_gradient": lambda self, plume_state, positions: np.zeros((1, 2)),
            "configure": lambda self, **kwargs: None,
        },
        "missing": "measure",
    },
    "AgentObservationProtocol": {
        "methods": {
            "construct_observation": lambda self, agent_state, plume_state, **kwargs: {},
            "get_observation_space": lambda self: None,
        },
        "missing": "get_observation_space",
    },
    "AgentActionProtocol": {
        "methods": {
            "validate_action": lambda self, action: action,
            "process_action": lambda self, action: {"target_speed": 0.0, "target_angular_velocity": 0.0},
            "get_action_space": lambda self: None,
        },
        "missing": "process_action",
    },
}


def _get_protocol_cls(name: str):
    """Return protocol class preferring modern export path."""
    if hasattr(protocols_pkg, name):
        return getattr(protocols_pkg, name)
    return getattr(core_protocols, name)


@pytest.mark.parametrize("protocol_name,spec", PROTOCOL_SPECS.items())
def test_protocol_structural_compatibility(protocol_name: str, spec: Dict[str, Any]) -> None:
    logger.info("Validating positive case for %s", protocol_name)
    protocol_cls = _get_protocol_cls(protocol_name)
    dummy_cls = type(f"Dummy{protocol_name.replace('Protocol', '')}", (), spec["methods"])
    instance = dummy_cls()
    assert isinstance(instance, protocol_cls)


@pytest.mark.parametrize("protocol_name,spec", PROTOCOL_SPECS.items())
def test_protocol_missing_method_runtime(protocol_name: str, spec: Dict[str, Any]) -> None:
    logger.info("Validating runtime failure for %s", protocol_name)
    protocol_cls = _get_protocol_cls(protocol_name)
    methods = {name: fn for name, fn in spec["methods"].items() if name != spec["missing"]}
    incomplete_cls = type(f"Incomplete{protocol_name.replace('Protocol', '')}", (), methods)
    instance = incomplete_cls()
    assert not isinstance(instance, protocol_cls)


@pytest.mark.parametrize("protocol_name", PROTOCOL_SPECS.keys())
def test_protocol_missing_method_mypy(protocol_name: str) -> None:
    logger.info("Running mypy negative test for %s", protocol_name)
    class_name = f"Incomplete{protocol_name.replace('Protocol', '')}"
    module = (
        "plume_nav_sim.protocols"
        if hasattr(protocols_pkg, protocol_name)
        else "plume_nav_sim.core.protocols"
    )
    snippet = f"""
from {module} import {protocol_name}
class {class_name}:
    pass

def accept(obj: {protocol_name}) -> None: ...
accept({class_name}())
"""
    stdout, stderr, exit_status = mypy_api.run(["--strict", "-c", snippet])
    logger.info("mypy output for %s: %s", protocol_name, stdout.strip())
    assert exit_status != 0


def test_core_protocol_reexports() -> None:
    """Ensure legacy access via core.protocols remains available."""
    assert protocols_pkg.PlumeModelProtocol is core_protocols.PlumeModelProtocol
    assert protocols_pkg.SensorProtocol is core_protocols.SensorProtocol
