import pytest
from unittest.mock import MagicMock

from src.plume_nav_sim.core.controllers import SingleAgentController, MultiAgentController


class TestControllerHookDiscovery:
    def test_single_agent_controller_hooks(self):
        controller = SingleAgentController(enable_logging=True, enable_extensibility_hooks=True)
        # Ensure hooks exist
        assert hasattr(controller, "compute_additional_obs")
        assert hasattr(controller, "compute_extra_reward")
        assert hasattr(controller, "on_episode_end")

        controller._logger = MagicMock()
        controller.compute_additional_obs({})
        controller.compute_extra_reward(0.0, {})
        controller.on_episode_end({})

        debug_calls = [c.args[0] for c in controller._logger.debug.call_args_list]
        assert any("compute_additional_obs" in msg for msg in debug_calls)
        assert any("compute_extra_reward" in msg for msg in debug_calls)
        assert any("on_episode_end" in msg for msg in debug_calls)

    def test_multi_agent_controller_hooks(self):
        controller = MultiAgentController(enable_logging=True, enable_extensibility_hooks=True)
        # Ensure hooks exist
        assert hasattr(controller, "compute_additional_obs")
        assert hasattr(controller, "compute_extra_reward")
        assert hasattr(controller, "on_episode_end")

        controller._logger = MagicMock()
        controller.compute_additional_obs({})
        controller.compute_extra_reward(0.0, {})
        controller.on_episode_end({})

        debug_calls = [c.args[0] for c in controller._logger.debug.call_args_list]
        assert any("compute_additional_obs" in msg for msg in debug_calls)
        assert any("compute_extra_reward" in msg for msg in debug_calls)
        assert any("on_episode_end" in msg for msg in debug_calls)
