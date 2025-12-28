def test_space_validator_uses_canonical_logger_name():
    # Lazy import to avoid heavy module import cost during test collection
    from plume_nav_sim.utils.spaces import SpaceValidator

    sv = SpaceValidator()
    # ComponentLogger exposes the component_name it was created with
    assert hasattr(sv.logger, "component_name")
    assert sv.logger.component_name == "validation"
