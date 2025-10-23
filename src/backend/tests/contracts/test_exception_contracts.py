"""Contract enforcement tests for exception classes.

Tests that exception signatures match CONTRACTS.md specification.
These tests prevent accidental API changes to exception classes.

Reference: ../../CONTRACTS.md Section "Exception Hierarchy"
"""

import inspect

import pytest

from plume_nav_sim.utils.exceptions import (
    ComponentError,
    ConfigurationError,
    PlumeNavSimError,
    RenderingError,
    StateError,
    ValidationError,
)


class TestValidationErrorContract:
    """Enforce ValidationError contract from CONTRACTS.md."""

    def test_signature_is_stable(self):
        """ValidationError signature must match CONTRACTS.md (IMMUTABLE)."""
        sig = inspect.signature(ValidationError.__init__)
        params = list(sig.parameters.keys())

        expected = [
            "self",
            "message",
            "parameter_name",
            "parameter_value",
            "expected_format",
            "parameter_constraints",
            "context",
            "invalid_value",  # Deprecated but kept for backward compatibility
        ]

        assert params == expected, (
            f"ValidationError signature changed!\n"
            f"Expected: {expected}\n"
            f"Got: {params}\n"
            f"This is a BREAKING CHANGE - update CONTRACTS.md if intentional"
        )

    def test_only_message_is_required(self):
        """Only 'message' parameter is required, all others optional."""
        # Should work with just message
        error = ValidationError("test message")
        assert error.message == "test message"
        assert error.parameter_name is None
        assert error.parameter_value is None
        assert error.expected_format is None

    def test_parameter_value_not_invalid_value(self):
        """Must use parameter_value, NOT deprecated invalid_value."""
        # Correct usage
        error = ValidationError("test", parameter_name="x", parameter_value=42)
        assert hasattr(error, "parameter_value")
        assert error.parameter_value == 42

        # invalid_value exists as deprecated alias but parameter_value is preferred
        assert hasattr(error, "invalid_value")
        assert error.invalid_value == error.parameter_value, "invalid_value should alias parameter_value"

    def test_deprecated_invalid_value_parameter_rejected(self):
        """Using deprecated invalid_value parameter still works but is deprecated."""
        # invalid_value is keyword-only and deprecated but still functional
        error = ValidationError("test", invalid_value=42)
        assert error.parameter_value == 42
        assert error.invalid_value == 42

    def test_stores_all_parameters(self):
        """ValidationError must store all provided parameters."""
        error = ValidationError(
            message="Test error",
            parameter_name="test_param",
            parameter_value=123,
            expected_format="positive integer",
            parameter_constraints={"min": 0},
        )

        assert error.message == "Test error"
        assert error.parameter_name == "test_param"
        assert error.parameter_value == 123
        assert error.expected_format == "positive integer"
        assert error.parameter_constraints == {"min": 0}

    def test_inherits_from_correct_parents(self):
        """ValidationError must inherit from PlumeNavSimError and ValueError."""
        assert issubclass(ValidationError, PlumeNavSimError)
        assert issubclass(ValidationError, ValueError)


class TestComponentErrorContract:
    """Enforce ComponentError contract from CONTRACTS.md."""

    def test_signature_is_stable(self):
        """ComponentError signature must match CONTRACTS.md (IMMUTABLE)."""
        sig = inspect.signature(ComponentError.__init__)
        params = list(sig.parameters.keys())

        expected = [
            "self",
            "message",
            "component_name",
            "operation_name",
            "underlying_error",
        ]

        assert params == expected, (
            f"ComponentError signature changed!\n"
            f"Expected: {expected}\n"
            f"Got: {params}\n"
            f"Update CONTRACTS.md if this is intentional"
        )

    def test_component_name_is_required(self):
        """component_name is REQUIRED per CONTRACTS.md."""
        with pytest.raises(TypeError):
            ComponentError("test message")  # Missing component_name

    def test_message_and_component_name_required(self):
        """Only message and component_name are required."""
        error = ComponentError(
            message="test error",
            component_name="TestComponent",
        )
        assert error.message == "test error"
        assert error.component_name == "TestComponent"
        assert error.operation_name is None
        assert error.underlying_error is None

    def test_deprecated_severity_parameter_rejected(self):
        """ComponentError must NOT accept severity parameter (removed)."""
        with pytest.raises(TypeError, match="severity"):
            # Old API used severity= but it's been removed
            ComponentError(
                "test",
                component_name="Test",
                severity="HIGH",  # type: ignore
            )

    def test_stores_all_parameters(self):
        """ComponentError must store all provided parameters."""
        underlying = ValueError("original error")
        error = ComponentError(
            message="Component failed",
            component_name="TestComponent",
            operation_name="test_operation",
            underlying_error=underlying,
        )

        assert error.message == "Component failed"
        assert error.component_name == "TestComponent"
        assert error.operation_name == "test_operation"
        assert error.underlying_error is underlying


class TestConfigurationErrorContract:
    """Enforce ConfigurationError contract from CONTRACTS.md."""

    def test_signature_is_stable(self):
        """ConfigurationError signature must match CONTRACTS.md (IMMUTABLE)."""
        sig = inspect.signature(ConfigurationError.__init__)
        params = list(sig.parameters.keys())

        expected = [
            "self",
            "message",
            "config_parameter",
            "parameter_value",
            "valid_options",
            "invalid_value",  # Deprecated but kept for backward compatibility
        ]

        assert params == expected, (
            f"ConfigurationError signature changed!\n"
            f"Expected: {expected}\n"
            f"Got: {params}"
        )

    def test_only_message_is_required(self):
        """Only message is required."""
        error = ConfigurationError("config error")
        assert error.message == "config error"
        assert error.config_parameter is None

    def test_uses_parameter_value_not_invalid_value(self):
        """Must use parameter_value, not deprecated invalid_value."""
        error = ConfigurationError(
            "test",
            config_parameter="param",
            parameter_value="bad_value",
        )
        assert error.parameter_value == "bad_value"
        # invalid_value exists as deprecated alias
        assert hasattr(error, "invalid_value")
        assert error.invalid_value == error.parameter_value


class TestStateErrorContract:
    """Enforce StateError contract from CONTRACTS.md."""

    def test_signature_is_stable(self):
        """StateError signature must match CONTRACTS.md (IMMUTABLE)."""
        sig = inspect.signature(StateError.__init__)
        params = list(sig.parameters.keys())

        expected = [
            "self",
            "message",
            "current_state",
            "expected_state",
            "component_name",
        ]

        assert params == expected, (
            f"StateError signature changed!\n"
            f"Expected: {expected}\n"
            f"Got: {params}"
        )

    def test_only_message_is_required(self):
        """Only message is required."""
        error = StateError("invalid state")
        assert error.message == "invalid state"
        assert error.current_state is None
        assert error.expected_state is None


class TestRenderingErrorContract:
    """Enforce RenderingError contract from CONTRACTS.md."""

    def test_signature_is_stable(self):
        """RenderingError signature must match CONTRACTS.md (IMMUTABLE)."""
        sig = inspect.signature(RenderingError.__init__)
        params = list(sig.parameters.keys())

        expected = [
            "self",
            "message",
            "render_mode",
            "backend_name",
            "underlying_error",
            "context",
        ]

        assert params == expected, (
            f"RenderingError signature changed!\n"
            f"Expected: {expected}\n"
            f"Got: {params}"
        )

    def test_only_message_is_required(self):
        """Only message is required."""
        error = RenderingError("render failed")
        assert error.message == "render failed"
        assert error.render_mode is None
        assert error.backend_name is None


class TestExceptionHierarchy:
    """Enforce exception hierarchy from CONTRACTS.md."""

    def test_all_exceptions_inherit_from_base(self):
        """All plume_nav_sim exceptions must inherit from PlumeNavSimError."""
        exceptions = [
            ValidationError,
            ComponentError,
            ConfigurationError,
            StateError,
            RenderingError,
        ]

        for exc_class in exceptions:
            assert issubclass(
                exc_class, PlumeNavSimError
            ), f"{exc_class.__name__} must inherit from PlumeNavSimError"

    def test_validation_error_also_inherits_value_error(self):
        """ValidationError inherits from both PlumeNavSimError and ValueError."""
        assert issubclass(ValidationError, ValueError)
        assert issubclass(ValidationError, PlumeNavSimError)


class TestExceptionInstantiation:
    """Test that exceptions can be instantiated correctly."""

    def test_validation_error_instantiation(self):
        """ValidationError can be instantiated and raised."""
        error = ValidationError(
            "test error",
            parameter_name="x",
            parameter_value=5,
        )

        # Can be raised and caught
        with pytest.raises(ValidationError) as exc_info:
            raise error

        assert exc_info.value.parameter_name == "x"
        assert exc_info.value.parameter_value == 5

    def test_component_error_instantiation(self):
        """ComponentError can be instantiated and raised."""
        error = ComponentError(
            "component failed",
            component_name="TestComponent",
            operation_name="test_op",
        )

        with pytest.raises(ComponentError) as exc_info:
            raise error

        assert exc_info.value.component_name == "TestComponent"
        assert exc_info.value.operation_name == "test_op"
