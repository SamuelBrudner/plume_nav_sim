# External imports with version comments
import dataclasses
import enum
import inspect  # >=3.10 - Frame inspection for automatic error context extraction and caller information
import logging  # >=3.10 - Logger creation and error logging integration for development debugging and monitoring
import re  # >=3.10 - Regular expressions for sanitizing untrusted error messages
import threading  # >=3.10 - Thread-safe error handling and context management for multi-threaded exception scenarios
import time  # >=3.10 - Timestamp generation for error tracking and performance context in exception handling
import traceback  # >=3.10 - Stack trace formatting for detailed error context and development debugging
import uuid
from typing import (  # >=3.10 - Type hints for exception parameters, error contexts, and recovery suggestion functions
    Any,
    Dict,
    List,
    Optional,
    Union,
)

# Global constants for error handling configuration
ERROR_CONTEXT_MAX_LENGTH = 1000
MAX_STACK_TRACE_DEPTH = 10
SANITIZATION_PLACEHOLDER = "<sanitized>"
SENSITIVE_KEYS = [
    "password",
    "token",
    "key",
    "secret",
    "credential",
    "internal",
    "debug",
    "stack_trace",
    "private",
]
RECOVERY_SUGGESTION_MAX_LENGTH = 500
ERROR_HISTORY_MAX_SIZE = 50

# Module exports - comprehensive exception handling interface
__all__ = [
    "PlumeNavSimError",
    "ValidationError",
    "StateError",
    "RenderingError",
    "ConfigurationError",
    "ComponentError",
    "ResourceError",
    "IntegrationError",
    "ErrorSeverity",
    "ErrorContext",
    "handle_component_error",
    "sanitize_error_context",
    "format_error_details",
    "create_error_context",
    "log_exception_with_recovery",
]


class ErrorSeverity(enum.IntEnum):
    """Enumeration class defining error severity levels for exception classification and handling priority in development and monitoring contexts.

    This enumeration provides standardized severity levels that enable consistent error classification,
    appropriate logging levels, and escalation decisions throughout the plume_nav_sim system.
    """

    # Severity level definitions with integer values for ordering
    LOW = 1  # Minor issues like validation warnings
    MEDIUM = 2  # Recoverable errors like rendering fallbacks
    HIGH = 3  # Significant errors like component failures
    CRITICAL = 4  # System-level failures requiring immediate attention

    def get_description(self) -> str:
        """Get human-readable description of error severity level.

        Returns:
            str: Description of severity level for logging and user display
        """
        severity_descriptions = {
            ErrorSeverity.LOW: "Minor issue with suggested improvements",
            ErrorSeverity.MEDIUM: "Recoverable error with fallback available",
            ErrorSeverity.HIGH: "Significant error requiring attention",
            ErrorSeverity.CRITICAL: "Critical system failure requiring immediate action",
        }
        return severity_descriptions.get(self, "Unknown severity level")

    def should_escalate(self) -> bool:
        """Check if error severity requires escalation to higher-level error handling.

        Returns:
            bool: True if error should be escalated based on severity level
        """
        return self in (ErrorSeverity.HIGH, ErrorSeverity.CRITICAL)


@dataclasses.dataclass
class ErrorContext:
    """Data class for structured error context information including timestamps, component details, operation context, and debugging information for comprehensive error tracking.

    This class provides a standardized structure for capturing and managing error context
    across all plume_nav_sim components, enabling consistent debugging and error analysis.
    """

    # Required fields for error context identification
    component_name: str
    operation_name: str
    timestamp: float

    # Optional fields for additional debugging context
    function_name: Optional[str] = None
    line_number: Optional[int] = None
    thread_id: Optional[str] = None
    additional_data: Dict[str, Any] = dataclasses.field(default_factory=dict)
    is_sanitized: bool = False

    def add_caller_info(self, stack_depth: int = 2) -> None:
        """Add caller function and line information using stack inspection.

        Args:
            stack_depth (int): Stack depth to inspect for caller information
        """
        try:
            # Use inspect.stack() to get caller information at specified depth
            frame_info = inspect.stack()[stack_depth]
            self.function_name = frame_info.function
            self.line_number = frame_info.lineno
        except (IndexError, AttributeError):
            # Handle exceptions gracefully if stack inspection fails
            self.function_name = "<unknown>"
            self.line_number = 0

    def add_system_info(self) -> None:
        """Add system and runtime information for debugging context."""
        try:
            # Get current thread ID using threading.current_thread().ident
            self.thread_id = str(threading.current_thread().ident)

            # Add Python version and platform information
            import platform
            import sys

            self.additional_data.update(
                {
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    "thread_name": threading.current_thread().name,
                }
            )
        except Exception:
            # Store system information in additional_data dictionary
            self.additional_data.setdefault(
                "system_info_error", "Failed to collect system information"
            )

    def sanitize(self, additional_sensitive_keys: Optional[List[str]] = None) -> None:
        """Sanitize error context removing sensitive information while preserving debugging data.

        Args:
            additional_sensitive_keys (Optional[List[str]]): Additional sensitive keys to sanitize
        """
        if self.is_sanitized:
            return

        # Apply sanitize_error_context to additional_data
        sensitive_keys = SENSITIVE_KEYS.copy()
        if additional_sensitive_keys:
            sensitive_keys.extend(additional_sensitive_keys)

        self.additional_data = sanitize_error_context(
            self.additional_data, sensitive_keys
        )

        # Remove or mask sensitive information from all context fields
        if self.function_name and self.function_name.startswith("_"):
            # Preserve debugging information like timestamps and component names
            pass  # Function name is useful for debugging even if private

        # Set is_sanitized to True to indicate sanitization completion
        self.is_sanitized = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary for serialization and logging.

        Returns:
            dict: Dictionary representation of error context with auto-sanitization
        """
        # Auto-sanitize if not already sanitized to prevent sensitive data exposure
        if not self.is_sanitized:
            self.sanitize()

        # Create dictionary with all context fields
        result = {
            "component_name": self.component_name,
            "operation_name": self.operation_name,
            "timestamp": self.timestamp,
        }

        # Add optional fields if they are not None
        if self.function_name is not None:
            result["function_name"] = self.function_name
        if self.line_number is not None:
            result["line_number"] = self.line_number
        if self.thread_id is not None:
            result["thread_id"] = self.thread_id

        # Include additional_data and sanitization status
        result["additional_data"] = self.additional_data
        result["is_sanitized"] = self.is_sanitized

        return result


class PlumeNavSimError(Exception):
    """Base exception class for all plume_nav_sim package errors providing consistent error handling interface, logging integration, recovery suggestions, and secure error reporting with development debugging support.

    This base class establishes the foundation for hierarchical error handling throughout
    the plume_nav_sim system, ensuring consistent error reporting, logging, and recovery patterns.
    """

    def __init__(
        self,
        message: str,
        context: Optional[Union[ErrorContext, Dict[str, Any]]] = None,
        severity: Union[ErrorSeverity, str] = ErrorSeverity.MEDIUM,
        **kwargs: Any,
    ):
        """Initialize base exception with message, context, severity level, and error tracking.

        Args:
            message (str): Primary error description
            context (Optional[ErrorContext]): Error context for debugging and analysis
            severity (ErrorSeverity): Error severity level for classification and handling priority
        """
        super().__init__(message)

        # Store message as primary error description
        self.message = message
        # Initialize context containers and normalize provided context
        self._context_obj: Optional[ErrorContext] = None
        self._context_dict: Optional[Dict[str, Any]] = None
        self.context = context
        # Normalize severity level for error classification and handling priority
        if isinstance(severity, str):
            try:
                self.severity = ErrorSeverity[severity.upper()]
            except Exception:
                self.severity = ErrorSeverity.MEDIUM
        else:
            self.severity = severity
        # Generate timestamp using time.time() for error tracking
        self.timestamp = time.time()
        # Create unique error_id for tracking and correlation
        self.error_id = str(uuid.uuid4())
        # Initialize recovery_suggestion to None (set by subclasses)
        self.recovery_suggestion: Optional[str] = None
        # Initialize empty error_details dictionary
        self.error_details: Dict[str, Any] = {}
        # Capture optional, forward-compatible kwargs into error_details to avoid constructor failures
        # Common extras used by tests: component, suggestions
        for k, v in kwargs.items():
            # Avoid overriding core fields
            if k not in {"message", "context", "severity"}:
                self.error_details[k] = v
        # Set logged to False initially
        self.logged = False

    @property
    def context(self) -> Optional[Union[Dict[str, Any], ErrorContext]]:
        if self._context_dict is not None:
            return dict(self._context_dict)
        return self._context_obj

    @context.setter
    def context(self, value: Optional[Union[ErrorContext, Dict[str, Any]]]) -> None:
        if value is None:
            self._context_obj = None
            self._context_dict = None
        elif isinstance(value, ErrorContext):
            self._context_obj = value
            self._context_dict = None
        elif isinstance(value, dict):
            self._context_dict = dict(value)
            self._context_obj = ErrorContext(
                component_name=self.__class__.__module__,
                operation_name=self.__class__.__name__,
                timestamp=time.time(),
                additional_data=dict(value),
            )
        else:
            raise TypeError(
                "context must be an ErrorContext instance, a dictionary, or None"
            )

    def get_error_details(self) -> Dict[str, Any]:
        """Get comprehensive error details including context, timestamp, and recovery information for debugging and logging.

        Returns:
            dict: Dictionary containing all error details and metadata
        """
        # Create base error details with message, error_id, and timestamp
        details = {
            "error_id": self.error_id,
            "message": self.message,
            "timestamp": self.timestamp,
            "severity": self.severity.name,
            "severity_description": self.severity.get_description(),
            "exception_type": self.__class__.__name__,
            "module": self.__class__.__module__,
        }

        # Include context information if available using stored context data
        if self._context_obj:
            details["context"] = self._context_obj.to_dict()
        elif self._context_dict is not None:
            details["context"] = dict(self._context_dict)

        # Add recovery_suggestion if available
        if self.recovery_suggestion:
            details["recovery_suggestion"] = self.recovery_suggestion

        # Include error_details dictionary for additional information
        details["error_details"] = self.error_details

        return details

    def format_for_user(self, include_suggestions: bool = True) -> str:
        """Format error message for user display removing technical details and sensitive information.

        Args:
            include_suggestions (bool): Whether to include recovery suggestions

        Returns:
            str: User-friendly error message with optional recovery suggestions
        """
        # Extract user-readable message removing technical jargon
        user_message = self.message

        # Remove potentially dangerous scripting content and injection attempts
        if user_message:
            # Remove script tags
            script_pattern = re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL)
            user_message = script_pattern.sub("", user_message)

            # Remove dangerous function calls
            for dangerous in ("javascript:", "eval(", "exec("):
                user_message = user_message.replace(dangerous, "")

            # Sanitize common injection patterns
            injection_patterns = [
                (r"';?\s*DROP\s+TABLE", SANITIZATION_PLACEHOLDER),  # SQL injection
                (r"\$\{jndi:", SANITIZATION_PLACEHOLDER),  # Log4j injection
                (r"\.\./\.\./", SANITIZATION_PLACEHOLDER),  # Path traversal
                (r"\\x00", SANITIZATION_PLACEHOLDER),  # Null bytes
            ]
            for pattern, replacement in injection_patterns:
                user_message = re.sub(pattern, replacement, user_message, flags=re.IGNORECASE)

            # Escape remaining HTML
            user_message = user_message.replace("<", "&lt;").replace(">", "&gt;")

        # Remove stack traces and internal system information
        if user_message.lower().startswith(("traceback", "exception", "error in")):
            user_message = "An error occurred in the plume navigation environment."

        # Remove internal implementation details that shouldn't be exposed to users
        internal_disclosure_terms = [
            ("file path disclosure:", "File disclosure:"),
            ("class name", "component"),
            ("method name", "operation"),
            ("variable name", "parameter"),
        ]
        user_message_lower = user_message.lower()
        for term, replacement in internal_disclosure_terms:
            if term in user_message_lower:
                # Replace the term case-insensitively
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                user_message = pattern.sub(replacement, user_message)

        # Include recovery suggestions if include_suggestions is True and available
        if include_suggestions and self.recovery_suggestion:
            user_message += f"\n\nSuggestion: {self.recovery_suggestion}"

        # Ensure no sensitive information is disclosed
        for sensitive_key in SENSITIVE_KEYS:
            if sensitive_key.lower() in user_message.lower():
                user_message = re.sub(
                    sensitive_key, SANITIZATION_PLACEHOLDER, user_message, flags=re.IGNORECASE
                )

        return user_message

    def log_error(
        self, logger: Optional[logging.Logger] = None, include_stack_trace: bool = False
    ) -> None:
        """Log error with appropriate logger including context, recovery suggestions, and debugging information.

        Args:
            logger (Optional[logging.Logger]): Logger instance or None for default
            include_stack_trace (bool): Whether to include stack trace in log
        """
        if self.logged:
            return  # Prevent duplicate logging

        # Get logger from parameter or create default logger
        if logger is None:
            logger = logging.getLogger("plume_nav_sim.exceptions")

        # Create comprehensive log message with error details (details available via get_error_details())

        # Include sanitized context information for debugging
        context_str = ""
        context_obj = self._context_obj
        if context_obj:
            if not context_obj.is_sanitized:
                context_obj.sanitize()
            context_str = (
                f" | Context: {context_obj.component_name}.{context_obj.operation_name}"
            )

        # Add stack trace if include_stack_trace is True
        stack_trace_str = ""
        if include_stack_trace:
            stack_trace_str = f"\n{traceback.format_exc()}"

        log_message = f"[{self.error_id}] {self.message}{context_str}{stack_trace_str}"

        # Format component.operation info if context available
        component_operation = ""
        if self.context:
            comp = self.context.component_name
            op = self.context.operation_name
            if comp and op:
                component_operation = f" [{comp}.{op}]"

        final_message = f"{component_operation}{log_message}" if component_operation else log_message
        sanitized_log_message = final_message
        for sensitive_key in SENSITIVE_KEYS:
            pattern = re.compile(re.escape(sensitive_key), re.IGNORECASE)
            sanitized_log_message = pattern.sub(SANITIZATION_PLACEHOLDER, sanitized_log_message)

        # Log using appropriate severity level with sanitized content
        if self.severity == ErrorSeverity.LOW:
            logger.info(sanitized_log_message)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(sanitized_log_message)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(sanitized_log_message)
        else:  # CRITICAL
            logger.critical(sanitized_log_message)

        # Set logged flag to True to prevent duplicate logging
        self.logged = True

    def set_recovery_suggestion(self, suggestion: str) -> None:
        """Set recovery suggestion for automated error handling and user guidance.

        Args:
            suggestion (str): Recovery suggestion text
        """
        # Validate suggestion length against RECOVERY_SUGGESTION_MAX_LENGTH
        if len(suggestion) > RECOVERY_SUGGESTION_MAX_LENGTH:
            suggestion = suggestion[: RECOVERY_SUGGESTION_MAX_LENGTH - 3] + "..."

        # Store suggestion in recovery_suggestion field
        self.recovery_suggestion = suggestion

        # Update error_details with recovery information
        self.error_details["has_recovery_guidance"] = True

    def add_context(self, key: str, value: Any) -> None:
        """Add additional context information to error for enhanced debugging.

        Args:
            key (str): Context key
            value (Any): Context value
        """
        # Validate key is non-empty string
        if not key or not isinstance(key, str):
            raise ValueError("Context key must be a non-empty string")

        # Sanitize value to prevent sensitive information disclosure
        if isinstance(value, str) and any(
            sensitive in value.lower() for sensitive in SENSITIVE_KEYS
        ):
            value = SANITIZATION_PLACEHOLDER

        # Add key-value pair to error_details dictionary
        self.error_details[key] = value


class ValidationError(PlumeNavSimError, ValueError):
    """Exception class for input parameter and action validation failures with specific validation context, detailed error reporting, and parameter-specific recovery suggestions for development debugging.

    This exception handles all input validation failures with detailed parameter information
    and specific recovery suggestions for resolving validation issues.
    """

    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_value: Optional[Any] = None,
        expected_format: Optional[str] = None,
        parameter_constraints: Optional[Dict[str, Any]] = None,
        context: Optional[Union[ErrorContext, Dict[str, Any]]] = None,
        *,
        invalid_value: Optional[Any] = None,
    ):
        """Initialize validation error with parameter details and validation context.

        Args:
            message (str): Primary error description
            parameter_name (Optional[str]): Name of parameter that failed validation
            parameter_value (Optional[Any]): The value that was provided for the parameter
            expected_format (Optional[str]): Expected format or constraint description
        """
        # Call parent constructor with message, context, and MEDIUM severity
        super().__init__(message, context=context, severity=ErrorSeverity.MEDIUM)

        # Store parameter_name for parameter-specific error handling
        self.parameter_name = parameter_name
        # Choose invalid value from explicit invalid_value or parameter_value, sanitize for safety
        if invalid_value is not None:
            self.invalid_value = self._sanitize_value(invalid_value)
            # Maintain backward-compat alias
            self.parameter_value = self.invalid_value
        else:
            self.parameter_value = self._sanitize_value(parameter_value)
            self.invalid_value = self.parameter_value
        # Store expected_format for user guidance
        self.expected_format = expected_format
        # Initialize empty validation_errors list
        self.validation_errors: List[Dict[str, str]] = []
        # Initialize parameter_constraints dictionary with optional caller-supplied data
        self.parameter_constraints: Dict[str, Any] = {}
        if parameter_constraints:
            self.parameter_constraints.update(parameter_constraints)

        # Set default recovery suggestion for validation failures
        self.set_recovery_suggestion(
            "Check input parameters and ensure they meet the expected format and constraints."
        )

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize value to prevent sensitive information disclosure."""
        if isinstance(value, str):
            for sensitive_key in SENSITIVE_KEYS:
                if sensitive_key.lower() in value.lower():
                    return SANITIZATION_PLACEHOLDER
        return value

    def get_validation_details(self) -> Dict[str, Any]:
        """Get comprehensive validation error details including parameter information and expected formats.

        Returns:
            dict: Dictionary containing validation-specific error details
        """
        # Get base error details from parent class
        details = self.get_error_details()

        # Add parameter_name and expected_format to details
        details.update(
            {
                "parameter_name": self.parameter_name,
                "invalid_value": self.invalid_value,
                "expected_format": self.expected_format,
                "validation_errors": self.validation_errors,
                "parameter_constraints": self.parameter_constraints,
            }
        )

        return details

    def add_validation_error(
        self, error_message: str, field_name: Optional[str] = None
    ) -> None:
        """Add additional validation error for compound validation failures.

        Args:
            error_message (str): Validation error description
            field_name (Optional[str]): Field name associated with error
        """
        # Validate error_message is non-empty string
        if not error_message or not isinstance(error_message, str):
            raise ValueError("Error message must be a non-empty string")

        # Add error message and optional field_name to validation_errors list
        error_entry = {"message": error_message}
        if field_name:
            error_entry["field_name"] = field_name

        self.validation_errors.append(error_entry)

        # Update recovery suggestion if multiple validation errors exist
        if len(self.validation_errors) > 1:
            self.set_recovery_suggestion(
                "Multiple validation errors detected. Please review all parameter constraints."
            )

    def set_parameter_constraints(self, constraints: Dict[str, Any]) -> None:
        """Set parameter constraints that were violated for detailed error reporting.

        Args:
            constraints (dict): Dictionary of parameter constraints
        """
        # Validate constraints is dictionary
        if not isinstance(constraints, dict):
            raise TypeError("Constraints must be a dictionary")

        # Store constraints in parameter_constraints field
        self.parameter_constraints = constraints

        # Generate recovery suggestion based on constraints
        if constraints:
            constraint_desc = ", ".join(f"{k}={v}" for k, v in constraints.items())
            self.set_recovery_suggestion(
                f"Ensure parameter meets constraints: {constraint_desc}"
            )


class StateError(PlumeNavSimError):
    """Exception class for invalid environment state transitions and inconsistent component states with state analysis, transition validation, and automated recovery action suggestions.

    This exception handles state-related errors with comprehensive state analysis
    and specific recovery actions for resolving state inconsistencies.
    """

    def __init__(
        self,
        message: str,
        current_state: Optional[str] = None,
        expected_state: Optional[str] = None,
        component_name: Optional[str] = None,
    ):
        """Initialize state error with current state, expected state, and component information.

        Args:
            message (str): Primary error description
            current_state (Optional[str]): Current state description
            expected_state (Optional[str]): Expected state description
            component_name (Optional[str]): Component name where state error occurred
        """
        # Call parent constructor with message and HIGH severity
        super().__init__(message, severity=ErrorSeverity.HIGH)

        # Store current_state for state analysis
        self.current_state = current_state
        # Store expected_state for recovery guidance
        self.expected_state = expected_state
        # Store component_name for component-specific recovery
        self.component_name = component_name
        # Initialize empty state_details dictionary
        self.state_details: Dict[str, Any] = {}
        # Initialize empty state_transition_history list
        self.state_transition_history: List[Dict[str, Any]] = []

        # Set recovery suggestion based on state transition analysis
        recovery_action = self.suggest_recovery_action()
        self.set_recovery_suggestion(recovery_action)

    def suggest_recovery_action(self) -> str:
        """Suggest specific recovery actions based on state transition analysis and component type.

        Returns:
            str: Recovery action suggestion for resolving state error
        """
        # Check for specific state conditions first (uninitialized, terminated, error)
        # These are high-priority indicators that override component-specific logic
        if self.current_state:
            current_lower = self.current_state.lower()
            if "uninitialized" in current_lower:
                return "Initialize component or environment before use"
            elif "terminated" in current_lower:
                return "Reset environment to begin new episode"
            elif "error" in current_lower and self.component_name:
                # For error states, prefer component-specific recovery
                component_lower = self.component_name.lower()
                if "plume" in component_lower:
                    return "Reinitialize plume model with valid parameters"
                else:
                    return "Clear error state and reinitialize component"

        # Consider component_name for component-specific recovery strategies
        if self.component_name:
            component_lower = self.component_name.lower()
            if "plume" in component_lower:
                return "Reinitialize plume model with valid parameters"
            elif "render" in component_lower:
                return "Reset rendering pipeline or switch to fallback mode"
            elif "env" in component_lower:
                return "Reset environment to initial state"

        # Return most appropriate recovery action based on error context
        return "Verify component state and reinitialize if necessary"

    def add_state_details(self, details: Dict[str, Any]) -> None:
        """Add detailed state information for debugging and recovery analysis.

        Args:
            details (dict): State details dictionary
        """
        # Validate details is dictionary
        if not isinstance(details, dict):
            raise TypeError("State details must be a dictionary")

        # Sanitize details to remove sensitive state information
        sanitized_details = sanitize_error_context(details)

        # Merge details into state_details dictionary
        self.state_details.update(sanitized_details)

        # Update recovery suggestion based on detailed state information
        if "episode_active" in details and not details["episode_active"]:
            self.set_recovery_suggestion(
                "Call reset() to start a new episode before stepping"
            )


class RenderingError(PlumeNavSimError):
    """Exception class for visualization and display failures including matplotlib backend issues, rendering pipeline problems, and display fallback strategies with backend compatibility analysis.

    This exception handles rendering and visualization errors with specific fallback
    suggestions and backend compatibility analysis.
    """

    def __init__(
        self,
        message: str,
        render_mode: Optional[str] = None,
        backend_name: Optional[str] = None,
        underlying_error: Optional[Exception] = None,
        context: Optional[Any] = None,
    ):
        """Initialize rendering error with render mode, backend information, and underlying error details.

        Args:
            message (str): Primary error description
            render_mode (Optional[str]): Rendering mode that failed
            backend_name (Optional[str]): Backend that caused the error
            underlying_error (Optional[Exception]): Underlying exception that caused rendering failure
        """
        # Prepare optional context: accept either ErrorContext or a plain dict
        error_context = None
        if context is not None:
            try:
                if isinstance(context, ErrorContext):
                    error_context = context
                elif isinstance(context, dict):
                    error_context = create_error_context(
                        operation_name="rendering_error",
                        additional_context=context,
                        include_caller_info=True,
                        include_system_info=False,
                    )
            except Exception:
                error_context = None

        # Call parent constructor with message, optional context and MEDIUM severity
        super().__init__(message, context=error_context, severity=ErrorSeverity.MEDIUM)

        # Store render_mode for mode-specific error handling
        self.render_mode = render_mode
        # Store backend_name for backend-specific recovery
        self.backend_name = backend_name
        # Store underlying_error for root cause analysis
        self.underlying_error = underlying_error
        # Initialize empty available_fallbacks list
        self.available_fallbacks: List[str] = []
        # Initialize empty rendering_context dictionary
        self.rendering_context: Dict[str, Any] = {}

        # Set recovery suggestion for rendering fallback options
        fallbacks = self.get_fallback_suggestions()
        if fallbacks:
            self.set_recovery_suggestion(
                f"Try fallback options: {', '.join(fallbacks[:2])}"
            )
        else:
            self.set_recovery_suggestion(
                "Switch to rgb_array mode or check matplotlib installation"
            )

    def get_fallback_suggestions(self) -> List[str]:
        """Get list of available rendering fallback options based on error context and system capabilities.

        Returns:
            list: List of fallback rendering options with implementation details
        """
        fallbacks = []

        # Analyze render_mode and backend_name for fallback options
        if self.render_mode == "human":
            fallbacks.append("rgb_array")  # Always available fallback

        # Check system capabilities for alternative backends
        if self.backend_name != "Agg":
            fallbacks.append("Agg backend")

        # Include 'Agg' backend fallback for matplotlib display issues
        if self.backend_name in ["TkAgg", "Qt5Agg"]:
            fallbacks.extend(["TkAgg", "Qt5Agg", "Agg"])

        # Return prioritized list of available fallback options
        self.available_fallbacks = list(set(fallbacks))  # Remove duplicates
        return self.available_fallbacks

    def set_rendering_context(self, context: Dict[str, Any]) -> None:
        """Set rendering context information for detailed error analysis and debugging.

        Args:
            context (dict): Rendering context information
        """
        # Validate context contains rendering-specific information
        if not isinstance(context, dict):
            raise TypeError("Rendering context must be a dictionary")

        # Store context in rendering_context dictionary
        self.rendering_context = context

        # Update available_fallbacks based on context
        if "headless" in context and context["headless"]:
            self.available_fallbacks = ["rgb_array", "Agg backend"]
            self.set_recovery_suggestion("Use rgb_array mode in headless environment")


class ConfigurationError(PlumeNavSimError):
    """Exception class for environment setup, registration issues, and invalid configuration parameters with configuration validation, parameter analysis, and valid option suggestions.

    This exception handles configuration-related errors with comprehensive parameter
    validation and suggestions for valid configuration options.
    """

    def __init__(
        self,
        message: str,
        config_parameter: Optional[str] = None,
        parameter_value: Optional[Any] = None,
        valid_options: Optional[Dict[str, Any]] = None,
        *,
        invalid_value: Optional[Any] = None,
    ):
        """Initialize configuration error with parameter details and valid options.

        Args:
            message (str): Primary error description
            config_parameter (Optional[str]): Configuration parameter that is invalid
            parameter_value (Optional[Any]): Value that was provided for the configuration parameter
            valid_options (Optional[dict]): Dictionary of valid configuration options
        """
        # Call parent constructor with message and HIGH severity
        super().__init__(message, severity=ErrorSeverity.HIGH)

        # Store config_parameter for parameter-specific error handling
        self.config_parameter = config_parameter
        # Store sanitized invalid value, allow keyword 'invalid_value' for compatibility
        if invalid_value is not None:
            self.invalid_value = self._sanitize_config_value(invalid_value)
            self.parameter_value = self.invalid_value
        else:
            self.parameter_value = self._sanitize_config_value(parameter_value)
            self.invalid_value = self.parameter_value
        # Store valid_options for recovery guidance
        self.valid_options = valid_options or {}
        # Initialize empty configuration_context dictionary
        self.configuration_context: Dict[str, Any] = {}
        # Initialize empty validation_errors list
        self.validation_errors: List[str] = []

        # Set recovery suggestion based on valid options
        if self.valid_options and self.config_parameter:
            options_str = str(
                list(self.valid_options.keys())[:3]
            )  # Show first 3 options
            self.set_recovery_suggestion(
                f"Use valid options for {self.config_parameter}: {options_str}"
            )
        else:
            self.set_recovery_suggestion(
                "Check configuration parameters against documentation"
            )

    def _sanitize_config_value(self, value: Any) -> Any:
        """Sanitize configuration value to prevent sensitive information disclosure."""
        if isinstance(value, str):
            for sensitive_key in SENSITIVE_KEYS:
                if sensitive_key.lower() in value.lower():
                    return SANITIZATION_PLACEHOLDER
        return value

    def get_valid_options(self) -> Dict[str, Any]:
        """Get dictionary of valid configuration options for error parameter with descriptions and examples.

        Returns:
            dict: Dictionary containing valid configuration options and usage examples
        """
        # Return valid_options if available
        if self.valid_options:
            return self.valid_options

        # Generate standard options for common config parameters
        standard_options = {}
        if self.config_parameter:
            param_lower = self.config_parameter.lower()
            if "grid_size" in param_lower:
                standard_options = {
                    "grid_size": "(width, height) tuple with positive integers"
                }
            elif "render_mode" in param_lower:
                standard_options = {"render_mode": ["human", "rgb_array"]}
            elif "source_location" in param_lower:
                standard_options = {
                    "source_location": "(x, y) tuple within grid bounds"
                }

        return standard_options

    def add_configuration_context(self, context: Dict[str, Any]) -> None:
        """Add configuration context for detailed error analysis and debugging.

        Args:
            context (dict): Configuration context information
        """
        # Sanitize context to remove sensitive configuration data
        sanitized_context = sanitize_error_context(context)

        # Merge context into configuration_context dictionary
        self.configuration_context.update(sanitized_context)

        # Update valid_options based on context information
        if "available_backends" in context:
            self.valid_options["backends"] = context["available_backends"]

        # Enhance recovery suggestions with context-aware guidance
        if "grid_size" in context:
            grid_info = context["grid_size"]
            self.set_recovery_suggestion(
                f"Use grid_size format like {grid_info} with positive integers"
            )


class ComponentError(PlumeNavSimError):
    """Exception class for general component-level failures in plume model, rendering pipeline, or state management with component diagnostic information and failure analysis.

    This exception handles component-specific errors with detailed diagnostic
    information and component-specific recovery strategies.
    """

    def __init__(
        self,
        message: str,
        component_name: str,
        operation_name: Optional[str] = None,
        underlying_error: Optional[Exception] = None,
    ):
        """Initialize component error with component identification and operation context.

        Args:
            message (str): Primary error description
            component_name (str): Name of component that failed
            operation_name (Optional[str]): Operation that was being performed
            underlying_error (Optional[Exception]): Underlying exception that caused component failure
        """
        # Call parent constructor with message and HIGH severity
        super().__init__(message, severity=ErrorSeverity.HIGH)

        # Store component_name for component identification
        self.component_name = component_name
        # Store operation_name for operation-specific debugging
        self.operation_name = operation_name
        # Store underlying_error for root cause analysis
        self.underlying_error = underlying_error
        # Initialize empty component_state dictionary
        self.component_state: Dict[str, Any] = {}
        # Initialize empty diagnostic_info list
        self.diagnostic_info: List[str] = []

        # Set component-specific recovery suggestion
        self.set_recovery_suggestion(self._generate_component_recovery_suggestion())

    def _generate_component_recovery_suggestion(self) -> str:
        """Generate component-specific recovery suggestion."""
        component_lower = self.component_name.lower()
        if "plume" in component_lower:
            return "Reinitialize plume model with valid grid_size and source_location"
        elif "render" in component_lower:
            return "Check matplotlib installation or switch to rgb_array mode"
        elif "env" in component_lower:
            return "Reset environment or check initialization parameters"
        else:
            return f"Restart {self.component_name} component or check configuration"

    def diagnose_failure(self) -> Dict[str, Any]:
        """Perform component-specific failure diagnosis and generate detailed diagnostic report.

        Returns:
            dict: Diagnostic report with component analysis and failure details
        """
        # Analyze component_name for component-specific diagnostics
        diagnostic_report = {
            "component_name": self.component_name,
            "operation_name": self.operation_name,
            "failure_timestamp": self.timestamp,
            "diagnostic_info": self.diagnostic_info.copy(),
        }

        # Check operation_name for operation-specific failure patterns
        if self.operation_name:
            diagnostic_report["operation_analysis"] = (
                f"Failure during {self.operation_name} operation"
            )

        # Analyze underlying_error for root cause identification
        if self.underlying_error:
            diagnostic_report["root_cause"] = {
                "error_type": type(self.underlying_error).__name__,
                "error_message": str(self.underlying_error),
            }

        # Generate component state analysis from component_state
        if self.component_state:
            diagnostic_report["component_state_analysis"] = self.component_state.copy()

        return diagnostic_report

    def set_component_state(self, state: Dict[str, Any]) -> None:
        """Set component state information for diagnostic analysis.

        Args:
            state (dict): Component state information
        """
        # Sanitize state to remove sensitive component information
        sanitized_state = sanitize_error_context(state)

        # Store state in component_state dictionary
        self.component_state = sanitized_state

        # Update diagnostic_info with state analysis
        if "initialized" in state and not state["initialized"]:
            self.diagnostic_info.append("Component not properly initialized")

        # Enhance recovery suggestions based on component state
        if state.get("memory_usage", 0) > 100 * 1024 * 1024:  # 100MB
            self.set_recovery_suggestion(
                "High memory usage detected - consider reducing grid size or clearing cache"
            )


class ResourceError(PlumeNavSimError):
    """Exception class for resource-related failures including memory exhaustion, cleanup issues, and system resource constraints with resource analysis and cleanup action recommendations.

    This exception handles resource-related errors with detailed resource usage
    analysis and specific cleanup action recommendations.
    """

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit_exceeded: Optional[float] = None,
    ):
        """Initialize resource error with resource type and usage information.

        Args:
            message (str): Primary error description
            resource_type (Optional[str]): Type of resource (memory, disk, etc.)
            current_usage (Optional[float]): Current resource usage value
            limit_exceeded (Optional[float]): Resource limit that was exceeded
        """
        # Call parent constructor with message and HIGH severity
        super().__init__(message, severity=ErrorSeverity.HIGH)

        # Store resource_type for resource-specific error handling
        self.resource_type = resource_type
        # Store current_usage and limit_exceeded for analysis
        self.current_usage = current_usage
        self.limit_exceeded = limit_exceeded
        # Initialize empty resource_details dictionary
        self.resource_details: Dict[str, Any] = {}
        # Initialize empty cleanup_actions list
        self.cleanup_actions: List[str] = []

        # Set resource-specific recovery suggestions
        cleanup_suggestions = self.suggest_cleanup_actions()
        if cleanup_suggestions:
            actions_str = ", ".join(cleanup_suggestions[:2])
            self.set_recovery_suggestion(f"Try cleanup actions: {actions_str}")
        else:
            self.set_recovery_suggestion(
                "Free up system resources or reduce operation scope"
            )

    def suggest_cleanup_actions(self) -> List[str]:
        """Suggest cleanup actions based on resource diagnostics."""
        actions: List[str] = []

        # Analyze resource_type for resource-specific cleanup strategies
        if self.resource_type:
            resource_lower = self.resource_type.lower()
            if "memory" in resource_lower:
                actions.extend(
                    ["clear_cache", "reduce_grid_size", "cleanup_matplotlib"]
                )
            elif "disk" in resource_lower:
                actions.extend(["cleanup_temp_files", "remove_old_logs"])
            elif "cpu" in resource_lower:
                actions.extend(["reduce_computation_complexity", "increase_timeout"])

        # Compare current_usage with limit_exceeded for severity assessment
        if self.current_usage and self.limit_exceeded:
            if self.current_usage > self.limit_exceeded * 0.9:  # >90% of limit
                actions.insert(0, "immediate_cleanup_required")

        # Return ordered list of cleanup actions with implementation details
        self.cleanup_actions = actions
        return actions

    def set_resource_details(self, details: Dict[str, Any]) -> None:
        """Set detailed resource usage information for analysis and cleanup planning.

        Args:
            details (dict): Resource usage details
        """
        # Store details in resource_details dictionary
        self.resource_details = details

        # Update cleanup_actions based on detailed resource information
        if "memory_breakdown" in details:
            memory_info = details["memory_breakdown"]
            if isinstance(memory_info, dict):
                # Find largest memory consumers
                if (
                    "plume_field" in memory_info
                    and memory_info["plume_field"] > 50 * 1024 * 1024
                ):
                    self.cleanup_actions.insert(0, "reduce_plume_field_size")

        # Calculate resource optimization recommendations
        if "optimization_suggestions" not in self.resource_details:
            self.resource_details["optimization_suggestions"] = self.cleanup_actions


class IntegrationError(PlumeNavSimError):
    """Exception class for external dependency failures including Gymnasium, NumPy, or Matplotlib integration issues with dependency compatibility checking and version analysis.

    This exception handles integration errors with external dependencies including
    detailed compatibility analysis and version checking.
    """

    def __init__(
        self,
        message: str,
        dependency_name: str,
        required_version: Optional[str] = None,
        current_version: Optional[str] = None,
    ):
        """Initialize integration error with dependency information and version details.

        Args:
            message (str): Primary error description
            dependency_name (str): Name of dependency that failed
            required_version (Optional[str]): Required version specification
            current_version (Optional[str]): Currently installed version
        """
        # Call parent constructor with message and HIGH severity
        super().__init__(message, severity=ErrorSeverity.HIGH)

        # Store dependency_name for dependency-specific error handling
        self.dependency_name = dependency_name
        # Store required_version and current_version for compatibility analysis
        self.required_version = required_version
        self.current_version = current_version
        # Initialize empty compatibility_info dictionary
        self.compatibility_info: Dict[str, Any] = {}
        # Set version_mismatch flag based on version comparison
        self.version_mismatch = self._check_version_mismatch()

        # Set dependency-specific recovery suggestions
        if self.version_mismatch:
            self.set_recovery_suggestion(
                f"Update {dependency_name} to version {required_version or 'latest'}"
            )
        else:
            self.set_recovery_suggestion(
                f"Verify {dependency_name} installation and configuration"
            )

    def _check_version_mismatch(self) -> bool:
        """Check if there's a version mismatch between required and current versions."""
        if not self.required_version or not self.current_version:
            return False

        # Simple version comparison (could be enhanced with packaging.version)
        try:
            # Remove version specifiers like >=, ==, etc.
            required_clean = (
                self.required_version.replace(">=", "")
                .replace("==", "")
                .replace(">", "")
                .strip()
            )
            current_clean = self.current_version.strip()

            return required_clean != current_clean
        except Exception:
            return True  # Assume mismatch if comparison fails

    def check_compatibility(self) -> Dict[str, Any]:
        """Check dependency compatibility and generate detailed compatibility report.

        Returns:
            dict: Compatibility analysis with version details and upgrade recommendations
        """
        # Compare required_version with current_version if both available
        compatibility_report = {
            "dependency_name": self.dependency_name,
            "required_version": self.required_version,
            "current_version": self.current_version,
            "version_mismatch": self.version_mismatch,
            "compatibility_status": (
                "incompatible" if self.version_mismatch else "unknown"
            ),
        }

        # Analyze compatibility based on dependency_name and version requirements
        if self.dependency_name.lower() in ["numpy", "matplotlib", "gymnasium"]:
            compatibility_report["dependency_type"] = "critical"
            compatibility_report["installation_priority"] = "high"

        # Include upgrade recommendations and installation instructions
        if self.version_mismatch and self.required_version:
            compatibility_report["upgrade_recommendation"] = {
                "action": "upgrade",
                "command": f"pip install {self.dependency_name}>={self.required_version}",
                "urgency": (
                    "high"
                    if self.dependency_name in ["gymnasium", "numpy"]
                    else "medium"
                ),
            }

        # Return comprehensive compatibility analysis dictionary
        self.compatibility_info = compatibility_report
        return compatibility_report

    def set_compatibility_info(self, info: Dict[str, Any]) -> None:
        """Store extended compatibility information with sanitization and guidance."""

        if not isinstance(info, dict):
            raise TypeError("Compatibility info must be provided as a dictionary")

        sanitized_info = sanitize_error_context(info)
        self.compatibility_info.update(sanitized_info)

        if "version_compatible" in sanitized_info:
            self.version_mismatch = not bool(sanitized_info["version_compatible"])

        upgrade_path = sanitized_info.get("upgrade_path")
        if isinstance(upgrade_path, list) and upgrade_path:
            path_str = " -> ".join(str(step) for step in upgrade_path[:3])
            self.set_recovery_suggestion(f"Follow upgrade path: {path_str}")

        breaking_changes = sanitized_info.get("breaking_changes")
        if breaking_changes:
            self.error_details["breaking_changes"] = breaking_changes

        migration_guide = sanitized_info.get("migration_guide")
        if migration_guide:
            self.error_details["migration_guide"] = migration_guide


class PlumeModelError(ComponentError):
    """Model-specific error with recovery suggestion helpers used by plume model tests."""

    def get_recovery_suggestions(self) -> List[str]:
        suggestions = [
            "Reinitialize plume model with valid parameters",
            "Check grid_size and source_location bounds",
            "Validate sigma is within acceptable range",
        ]
        return suggestions

    def set_compatibility_info(self, info: Dict[str, Any]) -> None:
        """Set detailed compatibility information for comprehensive error analysis.

        Args:
            info (dict): Compatibility information dictionary
        """
        # Store info in compatibility_info dictionary
        self.compatibility_info.update(info)

        # Update version_mismatch flag based on compatibility information
        if "version_compatible" in info:
            self.version_mismatch = not info["version_compatible"]

        # Generate upgrade path recommendations
        if "upgrade_path" in info:
            upgrade_path = info["upgrade_path"]
            if isinstance(upgrade_path, list):
                path_str = " -> ".join(upgrade_path[:3])  # Show first 3 steps
                self.set_recovery_suggestion(f"Follow upgrade path: {path_str}")


# Utility functions for centralized error handling


def handle_component_error(
    error: Exception,
    component_name: str,
    error_context: Optional[dict] = None,
    recovery_action: Optional[str] = None,
) -> str:
    """Centralized error handling function with component-specific recovery strategies, logging integration, and secure error reporting for all plume_nav_sim components.

    Args:
        error (Exception): Exception that occurred
        component_name (str): Name of component where error occurred
        error_context (Optional[dict]): Additional error context information
        recovery_action (Optional[str]): Suggested recovery action

    Returns:
        str: Recovery strategy identifier or error escalation status for automated error handling
    """
    # Validate error is Exception instance and component_name is non-empty string
    if not isinstance(error, Exception):
        raise TypeError("Error parameter must be an Exception instance")
    if not component_name or not isinstance(component_name, str):
        raise ValueError("Component name must be a non-empty string")

    # Get logger for component-specific logging
    logger = logging.getLogger(f"plume_nav_sim.{component_name}")

    # Classify error type using isinstance checks for plume_nav_sim exception hierarchy
    if isinstance(error, ValidationError):
        # Sanitize error_context to prevent sensitive information disclosure
        sanitized_context = sanitize_error_context(error_context or {})

        # Log error details using component-specific logger with sanitized context
        logger.error(f"Validation error in {component_name}: {error.message}")
        if sanitized_context:
            logger.debug(f"Validation context: {sanitized_context}")

        # Apply automatic recovery actions for recoverable error types
        return "validation_failed"

    elif isinstance(error, RenderingError):
        logger.warning(f"Rendering error in {component_name}: {error.message}")

        # Check for fallback options
        fallbacks = error.get_fallback_suggestions()
        if fallbacks:
            logger.info(f"Available fallbacks for {component_name}: {fallbacks}")

        return "fallback_mode"

    elif isinstance(
        error, (StateError, ComponentError, ResourceError, IntegrationError)
    ):
        logger.error(f"System error in {component_name}: {error.message}")

        # Escalate critical errors that require immediate attention
        if hasattr(error, "severity") and error.severity == ErrorSeverity.CRITICAL:
            logger.critical(
                f"Critical error in {component_name} requires immediate attention"
            )
            return "system_error"

        return "component_error"

    else:
        # Handle non-plume_nav_sim errors
        logger.critical(f"Unexpected error in {component_name}: {str(error)}")

        # Create PlumeNavSimError wrapper for unknown errors
        wrapped_error = PlumeNavSimError(
            f"Unexpected error in {component_name}: {str(error)}",
            severity=ErrorSeverity.CRITICAL,
        )
        wrapped_error.add_context("original_error_type", type(error).__name__)
        wrapped_error.log_error(logger)

        return "system_error"


def sanitize_error_context(  # noqa: C901
    context: dict, additional_sensitive_keys: Optional[List[str]] = None
) -> dict:
    """Sanitize error context dictionary to prevent sensitive information disclosure while preserving debugging information for secure error logging.

    Args:
        context (dict): Context dictionary to sanitize
        additional_sensitive_keys (Optional[List[str]]): Additional sensitive keys to sanitize

    Returns:
        dict: Sanitized context dictionary safe for logging and error reporting
    """
    if not isinstance(context, dict):
        return {}

    # Create copy of context dictionary to avoid modifying original
    sanitized = context.copy()

    # Combine SENSITIVE_KEYS with additional_sensitive_keys if provided
    sensitive_keys = SENSITIVE_KEYS.copy()
    if additional_sensitive_keys:
        sensitive_keys.extend(additional_sensitive_keys)

    # Iterate through all context keys and nested dictionary values
    for key, value in list(sanitized.items()):
        key_lower = str(key).lower()
        # Remove null bytes from keys
        if isinstance(key, str) and "\x00" in key:
            del sanitized[key]
            continue

        # Replace sensitive values with SANITIZATION_PLACEHOLDER
        # Use word boundary matching to avoid false positives (e.g., "normal_key" should not match "key")
        is_sensitive = False
        for sensitive in sensitive_keys:
            sensitive_lower = sensitive.lower()
            # For very short sensitive terms (<=3 chars), only match if at start or exact match
            # to avoid false positives like "normal_key" matching "key"
            if len(sensitive_lower) <= 3:
                if key_lower == sensitive_lower or key_lower.startswith(sensitive_lower + "_"):
                    is_sensitive = True
                    break
            else:
                # For longer terms, match as whole word or with common separators
                if (key_lower == sensitive_lower or  # Exact match
                    key_lower.startswith(sensitive_lower + "_") or  # prefix_
                    key_lower.endswith("_" + sensitive_lower) or  # _suffix
                    f"_{sensitive_lower}_" in key_lower):  # _middle_
                    is_sensitive = True
                    break

        if is_sensitive:
            sanitized[key] = SANITIZATION_PLACEHOLDER
            continue

        # Handle nested dictionaries
        if isinstance(value, dict):
            sanitized[key] = sanitize_error_context(value, additional_sensitive_keys)
            continue

        # Handle string values: truncate large strings and check for sensitive content
        if isinstance(value, str):
            # Remove null bytes
            if "\x00" in value:
                value = value.replace("\x00", "")
            # Check string content for sensitive information BEFORE truncating
            value_lower = value.lower()
            if any(sensitive in value_lower for sensitive in sensitive_keys):
                sanitized[key] = SANITIZATION_PLACEHOLDER
            elif len(value) > ERROR_CONTEXT_MAX_LENGTH:
                sanitized[key] = value[: ERROR_CONTEXT_MAX_LENGTH - 3] + "..."
            else:
                sanitized[key] = value
            continue

        # Remove private attributes (starting with underscore) from context
        if isinstance(key, str) and key.startswith("_"):
            del sanitized[key]
            continue

    # Return sanitized context dictionary safe for external logging
    return sanitized


def format_error_details(
    error: Exception,
    context: Optional[dict] = None,
    recovery_suggestion: Optional[str] = None,
    include_stack_trace: bool = False,
) -> str:
    """Format comprehensive error details including exception info, context, and recovery suggestions for development debugging and user-friendly error messages.

    Args:
        error (Exception): Exception to format
        context (Optional[dict]): Error context information
        recovery_suggestion (Optional[str]): Recovery suggestion text
        include_stack_trace (bool): Whether to include stack trace

    Returns:
        str: Formatted error details string with context and recovery information
    """
    # Extract basic error information (type, message, args)
    error_type = type(error).__name__
    error_message = str(error)

    # Format exception details with timestamp and error classification
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    details_lines = [
        f"Error Report - {timestamp}",
        f"Error Type: {error_type}",
        f"Message: {error_message}",
    ]

    # Add error ID if available (for PlumeNavSimError instances)
    if hasattr(error, "error_id"):
        details_lines.append(f"Error ID: {error.error_id}")

    # Add severity information if available
    if hasattr(error, "severity"):
        details_lines.append(
            f"Severity: {error.severity.name} - {error.severity.get_description()}"
        )

    # Add component identification if available (for ComponentError, StateError, etc.)
    if hasattr(error, "component_name") and error.component_name:
        details_lines.append(f"Component: {error.component_name}")
        if hasattr(error, "operation_name") and error.operation_name:
            details_lines.append(f"Operation: {error.operation_name}")
    # Also check context for component/operation info
    elif hasattr(error, "context") and error.context:
        if hasattr(error.context, "component_name") and error.context.component_name:
            details_lines.append(f"Component: {error.context.component_name}")
        if hasattr(error.context, "operation_name") and error.context.operation_name:
            details_lines.append(f"Operation: {error.context.operation_name}")

    # Add sanitized context information if provided
    if context:
        sanitized_context = sanitize_error_context(context)
        if sanitized_context:
            details_lines.append("\nContext:")
            for key, value in sanitized_context.items():
                details_lines.append(f"  {key}: {value}")

    # Include stack trace if include_stack_trace is True and safe for disclosure
    if include_stack_trace:
        # Limit stack trace depth to MAX_STACK_TRACE_DEPTH for readability
        stack_trace = traceback.format_exc()
        trace_lines = stack_trace.split("\n")
        if len(trace_lines) > MAX_STACK_TRACE_DEPTH:
            trace_lines = trace_lines[:MAX_STACK_TRACE_DEPTH] + ["... (truncated)"]

        details_lines.extend(["\nStack Trace:", "\n".join(trace_lines)])

    # Add recovery_suggestion if provided with length validation
    if recovery_suggestion:
        if len(recovery_suggestion) > RECOVERY_SUGGESTION_MAX_LENGTH:
            recovery_suggestion = (
                recovery_suggestion[: RECOVERY_SUGGESTION_MAX_LENGTH - 3] + "..."
            )
        details_lines.extend(["\nRecovery Suggestion:", recovery_suggestion])

    # Return comprehensive error details formatted for logging and debugging
    return "\n".join(details_lines)


def create_error_context(
    operation_name: Optional[str] = None,
    additional_context: Optional[dict] = None,
    include_caller_info: bool = False,
    include_system_info: bool = False,
) -> ErrorContext:
    """Create standardized error context dictionary with caller information, timestamp, and environment details for consistent error reporting.

    Args:
        operation_name (Optional[str]): Name of operation being performed
        additional_context (Optional[dict]): Additional context information
        include_caller_info (bool): Whether to include caller information
        include_system_info (bool): Whether to include system information

    Returns:
        ErrorContext: Standardized error context dictionary with timestamp and debugging information
    """
    # Create base context dictionary with current timestamp
    timestamp = time.time()

    # Determine component name from caller if not provided
    component_name = "unknown"
    if include_caller_info:
        try:
            frame_info = inspect.stack()[1]  # Get caller frame
            module_name = frame_info.filename.split("/")[-1].replace(".py", "")
            component_name = module_name
        except (IndexError, AttributeError):
            component_name = "unknown"

    # Add operation_name if provided for operation tracking
    operation = operation_name or "unknown_operation"

    # Create ErrorContext instance
    context = ErrorContext(
        component_name=component_name, operation_name=operation, timestamp=timestamp
    )

    # Include caller information (function name, line number) if include_caller_info is True
    if include_caller_info:
        context.add_caller_info(stack_depth=2)

    # Add system information (Python version, thread ID) if include_system_info is True
    if include_system_info:
        context.add_system_info()

    # Merge additional_context with validation and sanitization
    if additional_context:
        if isinstance(additional_context, dict):
            sanitized_additional = sanitize_error_context(additional_context)
            context.additional_data.update(sanitized_additional)

    # Add component identification and error tracking ID
    context.additional_data["error_tracking_id"] = str(uuid.uuid4())

    # Validate context size and truncate if necessary
    context_str = str(context.to_dict())
    if len(context_str) > ERROR_CONTEXT_MAX_LENGTH * 2:  # Allow some overhead
        # Truncate additional_data if too large
        context.additional_data = {
            "truncated": "Context too large, truncated for safety"
        }

    # Return standardized error context for consistent error handling
    return context


def log_exception_with_recovery(  # noqa: C901
    exception: Exception,
    logger: logging.Logger,
    context: Optional[dict] = None,
    recovery_action: Optional[str] = None,
    include_performance_impact: bool = False,
) -> None:
    """Log exception with detailed context, recovery suggestions, and performance impact analysis for development debugging and monitoring.

    Args:
        exception (Exception): Exception to log
        logger (logging.Logger): Logger instance to use
        context (Optional[dict]): Error context information
        recovery_action (Optional[str]): Recovery action taken or suggested
        include_performance_impact (bool): Whether to include performance impact analysis
    """
    # Validate exception and logger parameters
    if not isinstance(exception, Exception):
        raise TypeError("Exception parameter must be an Exception instance")
    # Accept both logging.Logger and ComponentLogger (which has .logger attribute)
    if not isinstance(logger, logging.Logger) and not hasattr(logger, "logger"):
        raise TypeError(
            "Logger parameter must be a logging.Logger instance or ComponentLogger"
        )

    # Create comprehensive error context using create_error_context
    error_context = create_error_context(
        operation_name="exception_logging",
        additional_context=context,
        include_caller_info=True,
        include_system_info=include_performance_impact,
    )

    # Sanitize context to prevent sensitive information disclosure
    if context:
        sanitized_context = sanitize_error_context(context)
        error_context.additional_data.update(sanitized_context)

    # Format detailed error message with exception details and context
    error_details = format_error_details(
        exception,
        context=error_context.to_dict(),
        recovery_suggestion=recovery_action,
        include_stack_trace=True,
    )

    # Include performance impact analysis if include_performance_impact is True
    if include_performance_impact:
        performance_info = {
            "timestamp": error_context.timestamp,
            "thread_id": error_context.thread_id,
            "memory_usage": "performance_analysis_placeholder",  # Would integrate with actual monitoring
        }
        error_details += f"\n\nPerformance Context: {performance_info}"

    # Add recovery_action suggestions and automated recovery status
    if recovery_action:
        error_details += f"\n\nRecovery Action: {recovery_action}"

    # Log with appropriate level based on exception severity and type
    if hasattr(exception, "severity"):
        severity = exception.severity
        if severity == ErrorSeverity.LOW:
            logger.info(error_details)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(error_details)
        elif severity == ErrorSeverity.HIGH:
            logger.error(error_details)
        else:  # CRITICAL
            logger.critical(error_details)
    else:
        # Default to error level for unknown exceptions
        logger.error(error_details)

    # Update error tracking statistics for component reliability analysis
    # This would integrate with actual monitoring systems in a full implementation
    if hasattr(exception, "component_name"):
        logger.debug(
            f"Error tracking updated for component: {exception.component_name}"
        )
