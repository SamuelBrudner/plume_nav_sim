import importlib.util
import logging
import sys
import types

import numpy as np
import pytest


def load_single_agent_controller():
    """Load SingleAgentController with minimal dependency stubs."""
    # Stub required modules to avoid heavy optional dependencies
    stubs = {
        'plume_nav_sim.protocols.navigator': types.SimpleNamespace(NavigatorProtocol=object),
        'plume_nav_sim.core.protocols': types.SimpleNamespace(
            BoundaryPolicyProtocol=object,
            SourceProtocol=object,
        ),
        'plume_nav_sim.protocols.sensor': types.SimpleNamespace(SensorProtocol=object),
        'plume_nav_sim.core.boundaries': types.SimpleNamespace(
            create_boundary_policy=lambda *a, **k: object()
        ),
        'plume_nav_sim.envs.spaces': types.SimpleNamespace(SpacesFactory=object),
        'plume_nav_sim.utils.frame_cache': types.SimpleNamespace(
            FrameCache=object,
            CacheMode=object,
        ),
        'plume_nav_sim.utils.logging_setup': types.SimpleNamespace(
            get_enhanced_logger=logging.getLogger
        ),
        'plume_nav_sim.utils.seed_manager': types.ModuleType('seed_manager'),
        'plume_nav_sim.utils.visualization': types.ModuleType('visualization'),
        'plume_nav_sim.utils.io': types.ModuleType('io'),
        'plume_nav_sim.utils.navigator_utils': types.ModuleType('navigator_utils'),
        'plume_nav_sim.envs.video_plume': types.SimpleNamespace(VideoPlume=object),
        'plume_nav_sim.config.schemas': types.SimpleNamespace(
            NavigatorConfig=object,
            SingleAgentConfig=object,
            MultiAgentConfig=object,
        ),
    }
    for name, module in stubs.items():
        sys.modules.setdefault(name, module)

    # Ensure package parents exist for relative imports
    for pkg in [
        'plume_nav_sim',
        'plume_nav_sim.core',
        'plume_nav_sim.envs',
        'plume_nav_sim.utils',
        'plume_nav_sim.protocols',
        'plume_nav_sim.config',
    ]:
        sys.modules.setdefault(pkg, types.ModuleType(pkg))

    spec = importlib.util.spec_from_file_location(
        'plume_nav_sim.core.controllers',
        'src/plume_nav_sim/core/controllers.py',
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['plume_nav_sim.core.controllers'] = module
    spec.loader.exec_module(module)
    return module.SingleAgentController


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
def test_read_single_antenna_odor_invalid_sensor_values_raise(bad_value):
    SingleAgentController = load_single_agent_controller()
    controller = SingleAgentController(position=(0, 0))
    environment = np.array([[bad_value]], dtype=float)
    with pytest.raises(RuntimeError):
        controller.read_single_antenna_odor(environment)
