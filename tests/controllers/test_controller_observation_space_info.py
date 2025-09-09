import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load_controllers_module():
    module_path = Path(__file__).resolve().parents[2] / 'src' / 'plume_nav_sim' / 'core' / 'controllers.py'

    pkg_plume = types.ModuleType('plume_nav_sim'); pkg_plume.__path__ = []
    pkg_core = types.ModuleType('plume_nav_sim.core'); pkg_core.__path__ = []
    pkg_config = types.ModuleType('plume_nav_sim.config'); pkg_config.__path__ = []
    pkg_schemas = types.ModuleType('plume_nav_sim.config.schemas'); pkg_schemas.__path__ = []
    class NavigatorConfig: ...
    class SingleAgentConfig: ...
    class MultiAgentConfig: ...
    pkg_schemas.NavigatorConfig = NavigatorConfig
    pkg_schemas.SingleAgentConfig = SingleAgentConfig
    pkg_schemas.MultiAgentConfig = MultiAgentConfig

    pkg_utils = types.ModuleType('plume_nav_sim.utils'); pkg_utils.__path__ = []
    pkg_frame_cache = types.ModuleType('plume_nav_sim.utils.frame_cache'); pkg_frame_cache.__path__ = []
    class FrameCache: ...
    class CacheMode: ...
    pkg_frame_cache.FrameCache = FrameCache
    pkg_frame_cache.CacheMode = CacheMode

    pkg_envs = types.ModuleType('plume_nav_sim.envs'); pkg_envs.__path__ = []
    pkg_envs_spaces = types.ModuleType('plume_nav_sim.envs.spaces'); pkg_envs_spaces.__path__ = []
    class SpacesFactory: ...
    pkg_envs_spaces.SpacesFactory = SpacesFactory

    pkg_protocols = types.ModuleType('plume_nav_sim.protocols'); pkg_protocols.__path__ = []
    pkg_protocols_nav = types.ModuleType('plume_nav_sim.protocols.navigator')
    class NavigatorProtocol: ...
    pkg_protocols_nav.NavigatorProtocol = NavigatorProtocol
    pkg_protocols_sensor = types.ModuleType('plume_nav_sim.protocols.sensor')
    class SensorProtocol: ...
    pkg_protocols_sensor.SensorProtocol = SensorProtocol

    pkg_core_protocols = types.ModuleType('plume_nav_sim.core.protocols')
    class BoundaryPolicyProtocol: ...
    class SourceProtocol: ...
    pkg_core_protocols.BoundaryPolicyProtocol = BoundaryPolicyProtocol
    pkg_core_protocols.SourceProtocol = SourceProtocol

    pkg_core_boundaries = types.ModuleType('plume_nav_sim.core.boundaries')
    def create_boundary_policy(*args, **kwargs): ...
    pkg_core_boundaries.create_boundary_policy = create_boundary_policy

    sys.modules.update({
        'plume_nav_sim': pkg_plume,
        'plume_nav_sim.core': pkg_core,
        'plume_nav_sim.config': pkg_config,
        'plume_nav_sim.config.schemas': pkg_schemas,
        'plume_nav_sim.utils': pkg_utils,
        'plume_nav_sim.utils.frame_cache': pkg_frame_cache,
        'plume_nav_sim.envs': pkg_envs,
        'plume_nav_sim.envs.spaces': pkg_envs_spaces,
        'plume_nav_sim.protocols': pkg_protocols,
        'plume_nav_sim.protocols.navigator': pkg_protocols_nav,
        'plume_nav_sim.protocols.sensor': pkg_protocols_sensor,
        'plume_nav_sim.core.protocols': pkg_core_protocols,
        'plume_nav_sim.core.boundaries': pkg_core_boundaries,
    })

    spec = importlib.util.spec_from_file_location('plume_nav_sim.core.controllers', module_path)
    controllers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = controllers
    spec.loader.exec_module(controllers)
    return controllers


def test_base_controller_observation_space_info_not_implemented():
    controllers = load_controllers_module()
    BaseController = controllers.BaseController
    controller = BaseController(enable_logging=False)
    with pytest.raises(NotImplementedError):
        controller.get_observation_space_info()
