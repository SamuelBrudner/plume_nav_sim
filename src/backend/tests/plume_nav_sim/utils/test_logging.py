import logging

import pytest


def test_get_component_logger_dotted_module_names_no_unknown_warning(caplog):
    # Import locally to ensure logging package init occurs within test context
    from plume_nav_sim.utils.logging import ComponentType, get_component_logger

    caplog.clear()
    caplog.set_level(logging.WARNING)

    # Use a fully qualified module path typical of __name__ usage
    dotted_name = "plume_nav_sim.utils.validation"
    get_component_logger(dotted_name, ComponentType.UTILS)

    # Ensure no "Unknown component name" warning is emitted for dotted names
    assert not any(
        "Unknown component name" in rec.getMessage() for rec in caplog.records
    )


def test_get_component_logger_package_prefix_stripped_no_unknown_warning(caplog):
    from plume_nav_sim.utils.logging import ComponentType, get_component_logger

    caplog.clear()
    caplog.set_level(logging.WARNING)

    # Direct package-qualified component (prefix should be stripped to 'utils')
    dotted_name = "plume_nav_sim.utils"
    get_component_logger(dotted_name, ComponentType.UTILS)

    assert not any(
        "Unknown component name" in rec.getMessage() for rec in caplog.records
    )


def test_get_component_logger_arbitrary_name_no_unknown_warning(caplog):
    from plume_nav_sim.utils.logging import ComponentType, get_component_logger

    caplog.clear()
    caplog.set_level(logging.WARNING)

    # Arbitrary tool/script name should not warn
    get_component_logger("pytest_cache_cleaner", ComponentType.UTILS)

    assert not any(
        "Unknown component name" in rec.getMessage() for rec in caplog.records
    )


def test_get_component_logger_core_component_id_validation():
    from plume_nav_sim.utils.exceptions import ValidationError
    from plume_nav_sim.utils.logging import ComponentType, get_component_logger

    with pytest.raises(ValidationError):
        get_component_logger(
            "some_logger", ComponentType.UTILS, core_component_id="not_a_core"
        )


def test_get_component_logger_core_component_id_metadata():
    from plume_nav_sim.utils.logging import ComponentType, get_component_logger

    logger = get_component_logger(
        "render", ComponentType.RENDERING, core_component_id="render"
    )
    ctx = getattr(logger, "component_context", {})
    assert isinstance(ctx, dict) and ctx.get("core_component_id") == "render"
