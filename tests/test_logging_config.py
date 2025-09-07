import plume_nav_sim
from plume_nav_sim.utils.logging_setup import EnhancedLogger

def test_package_uses_enhanced_logger():
    assert isinstance(plume_nav_sim.logger, EnhancedLogger)
    assert getattr(plume_nav_sim, "_BOOTSTRAP_COMPLETED", False)
