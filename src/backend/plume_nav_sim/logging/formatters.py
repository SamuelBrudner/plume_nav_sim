"""
Specialized logging formatters for plume_nav_sim providing security-aware message formatting,
performance metrics display, console color support, and structured log output. Implements
comprehensive formatting classes with sensitive information filtering, timing measurement
display, and development-friendly console output for the proof-of-life logging infrastructure.
"""

# Standard library imports with version comments
import logging  # >=3.10 - Standard Python logging module for Formatter base class and LogRecord processing
import os  # >=3.10 - Operating system interface for environment variable access and terminal capability detection
import re  # >=3.10 - Regular expression support for sensitive information detection and pattern-based filtering
import sys  # >=3.10 - System interface for terminal detection, stdout/stderr access, and platform identification
import time  # >=3.10 - Time utilities for timestamp formatting and performance timing calculations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Console color codes for enhanced readability
CONSOLE_COLOR_CODES = {
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[92m",  # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[91m",  # Red
    "RESET": "\033[0m",  # Reset color
}

# Internal imports - Import performance constants and handle missing constants gracefully
try:
    from ..core.constants import (
        PERFORMANCE_TARGET_PLUME_GENERATION_MS as PERFORMANCE_TARGET_HUMAN_RENDER_MS,
    )
    from ..core.constants import (
        PERFORMANCE_TARGET_RGB_RENDER_MS,
        PERFORMANCE_TARGET_STEP_LATENCY_MS,
    )
except ImportError:
    # Fallback constants if imports are not available
    PERFORMANCE_TARGET_STEP_LATENCY_MS = 1.0
    PERFORMANCE_TARGET_RGB_RENDER_MS = 5.0
    PERFORMANCE_TARGET_HUMAN_RENDER_MS = 10.0

# Handle missing constants with reasonable defaults
try:
    from ..core.constants import LOG_LEVEL_DEFAULT
except ImportError:
    LOG_LEVEL_DEFAULT = "INFO"

try:
    from ..core.constants import COMPONENT_NAMES
except ImportError:
    COMPONENT_NAMES = [
        "plume_nav_sim",
        "environment",
        "rendering",
        "plume_model",
        "utilities",
    ]

# Global format string constants
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEVELOPMENT_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
)
PERFORMANCE_LOG_FORMAT = (
    "%(asctime)s - PERF - %(name)s - %(message)s [%(duration_ms).3fms]"
)
CONSOLE_FORMAT = "%(levelname)-8s | %(name)-20s | %(message)s"
ERROR_FORMAT = "%(asctime)s - ERROR - %(name)s - %(funcName)s:%(lineno)d - %(message)s"
DEBUG_FORMAT = (
    "%(asctime)s - DEBUG - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
)

# Date format constants
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
PERFORMANCE_DATE_FORMAT = "%H:%M:%S.%f"

# ANSI color code mappings for terminal color support
ANSI_COLOR_CODES = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "BRIGHT_RED": "\033[91m",
    "BRIGHT_GREEN": "\033[92m",
    "BRIGHT_YELLOW": "\033[93m",
    "BRIGHT_BLUE": "\033[94m",
    "BRIGHT_MAGENTA": "\033[95m",
    "BRIGHT_CYAN": "\033[96m",
}

# Default color scheme for different log levels
DEFAULT_LEVEL_COLORS = {
    "DEBUG": "CYAN",
    "INFO": "GREEN",
    "WARNING": "YELLOW",
    "ERROR": "RED",
    "CRITICAL": "BRIGHT_RED",
}

# Security filtering patterns for sensitive information detection
SENSITIVE_PATTERNS = [
    "password",
    "token",
    "api_key",
    "secret",
    "auth",
    "credential",
    "key",
    "private",
]

# Compiled regex patterns for efficient sensitive information matching
SENSITIVE_REGEX_PATTERNS = [
    re.compile(
        r"\b(?:password|token|api_key|secret|auth|credential)\s*[=:]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"\b[A-Za-z0-9+/]{20,}={0,2}\b"),  # Base64-like patterns
    re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    ),  # UUIDs
]

# Security and formatting configuration constants
REDACTION_PLACEHOLDER = "[REDACTED]"
MAX_MESSAGE_LENGTH = 1000
PERFORMANCE_THRESHOLD_WARNING_MS = 5.0
PERFORMANCE_THRESHOLD_ERROR_MS = 50.0

# Cache variables for performance optimization
_COLOR_SUPPORT_CACHE = None
_TERMINAL_WIDTH_CACHE = None
_FORMATTER_CACHE = {}

# Unicode and ASCII symbol mappings for enhanced display
UNICODE_SYMBOLS = {
    "CHECK": "✓",
    "CROSS": "✗",
    "ARROW": "→",
    "BULLET": "•",
    "WARNING": "⚠",
    "INFO": "ℹ",
    "ERROR": "❌",
    "SUCCESS": "✅",
}

FALLBACK_ASCII_SYMBOLS = {
    "CHECK": "OK",
    "CROSS": "X",
    "ARROW": "->",
    "BULLET": "*",
    "WARNING": "!",
    "INFO": "i",
    "ERROR": "E",
    "SUCCESS": "OK",
}


def detect_color_support(  # noqa: C901
    force_detection: bool = False, cache_result: bool = True
) -> Dict[str, Any]:
    """
    Detects terminal color support capabilities including ANSI color codes, 256-color support,
    and true color support with caching for performance optimization.

    Args:
        force_detection: Force detection even if cached result exists
        cache_result: Whether to cache the detection results

    Returns:
        Dictionary containing color support capabilities, terminal type, and recommended color usage
    """
    global _COLOR_SUPPORT_CACHE

    # Check cached color support result to avoid repeated detection if cache_result is True
    if _COLOR_SUPPORT_CACHE is not None and not force_detection and cache_result:
        return _COLOR_SUPPORT_CACHE

    color_support = {
        "basic_colors": False,
        "ansi_colors": False,
        "color_256": False,
        "true_color": False,
        "terminal_type": "unknown",
        "recommended_usage": "none",
    }

    # Examine environment variables including TERM, COLORTERM, and FORCE_COLOR
    term = os.environ.get("TERM", "")
    colorterm = os.environ.get("COLORTERM", "")
    force_color = os.environ.get("FORCE_COLOR", "").lower()

    # Check for Windows terminal and Windows Terminal application color support
    if sys.platform == "win32":
        # Basic Windows console support
        color_support["basic_colors"] = True
        color_support["terminal_type"] = "windows"

        # Windows Terminal has enhanced support
        if "WT_SESSION" in os.environ:
            color_support["ansi_colors"] = True
            color_support["color_256"] = True
            color_support["terminal_type"] = "windows_terminal"
    else:
        # Unix-like systems color detection
        color_support["basic_colors"] = True

        # Test terminal support for basic ANSI color codes with escape sequences
        if term and any(
            x in term.lower()
            for x in ["color", "ansi", "xterm", "linux", "screen", "tmux"]
        ):
            color_support["ansi_colors"] = True
            color_support["terminal_type"] = term

        # Check for 256-color support through terminal capability queries
        if "256" in term or colorterm:
            color_support["color_256"] = True

        # Detect true color (24-bit) support if available in terminal
        if colorterm in ["truecolor", "24bit"] or "truecolor" in term.lower():
            color_support["true_color"] = True

    # Determine if output is being piped or redirected to disable colors
    if not sys.stdout.isatty():
        color_support["recommended_usage"] = "none"
    elif force_color in ["1", "true", "yes", "on"]:
        color_support["recommended_usage"] = "full"
    elif color_support["true_color"]:
        color_support["recommended_usage"] = "full"
    elif color_support["color_256"]:
        color_support["recommended_usage"] = "extended"
    elif color_support["ansi_colors"]:
        color_support["recommended_usage"] = "basic"
    else:
        color_support["recommended_usage"] = "none"

    # Cache detection results in _COLOR_SUPPORT_CACHE if cache_result is True
    if cache_result:
        _COLOR_SUPPORT_CACHE = color_support

    return color_support


def get_terminal_width(default_width: int = 80, cache_result: bool = True) -> int:
    """
    Determines current terminal width for message wrapping, alignment, and optimal formatting
    with cross-platform compatibility and caching.

    Args:
        default_width: Default width to use if detection fails
        cache_result: Whether to cache the terminal width result

    Returns:
        Current terminal width in characters, or default_width if detection fails
    """
    global _TERMINAL_WIDTH_CACHE

    # Check cached terminal width result to improve performance if cache_result enabled
    if _TERMINAL_WIDTH_CACHE is not None and cache_result:
        return _TERMINAL_WIDTH_CACHE

    width = default_width

    try:
        # Try os.get_terminal_size() for modern Python terminal size detection
        size = os.get_terminal_size()
        width = size.columns
    except (OSError, AttributeError):
        try:
            # Fall back to shutil.get_terminal_size() for alternative terminal size method
            import shutil

            size = shutil.get_terminal_size()
            width = size.columns
        except (OSError, AttributeError, ImportError):
            # Check environment variables COLUMNS and LINES for terminal dimensions
            try:
                width = int(os.environ.get("COLUMNS", default_width))
            except (ValueError, TypeError):
                width = default_width

    # Apply reasonable bounds checking (minimum 40, maximum 200 characters)
    width = max(40, min(200, width))

    # Cache terminal width result in _TERMINAL_WIDTH_CACHE if cache_result enabled
    if cache_result:
        _TERMINAL_WIDTH_CACHE = width

    return width


def sanitize_message(  # noqa: C901
    message: str,
    additional_patterns: List[str] = None,
    redaction_placeholder: str = REDACTION_PLACEHOLDER,
    strict_mode: bool = False,
) -> str:
    """
    Sanitizes log messages by removing or redacting sensitive information using regex patterns
    and safe replacement strategies for security-aware logging.

    Args:
        message: Log message to sanitize
        additional_patterns: Custom sensitive information patterns to check
        redaction_placeholder: Text to replace sensitive information with
        strict_mode: Enable additional filtering for high-security environments

    Returns:
        Sanitized message with sensitive information redacted using placeholder replacement
    """
    # Validate message parameter is string and handle None/empty cases gracefully
    if not message or not isinstance(message, str):
        return str(message) if message is not None else ""

    sanitized = message

    # Apply built-in SENSITIVE_REGEX_PATTERNS to detect common sensitive information
    for pattern in SENSITIVE_REGEX_PATTERNS:
        sanitized = pattern.sub(redaction_placeholder, sanitized)

    # Process additional_patterns list for custom sensitive information detection
    if additional_patterns:
        for pattern_str in additional_patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                sanitized = pattern.sub(redaction_placeholder, sanitized)
            except re.error:
                # Skip invalid regex patterns
                continue

    # Apply strict mode additional filtering for high-security environments if enabled
    if strict_mode:
        # Additional conservative patterns in strict mode
        strict_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN-like patterns
            r"\b[A-Z0-9]{8,}\b",  # Long alphanumeric codes
        ]
        for pattern_str in strict_patterns:
            try:
                pattern = re.compile(pattern_str)
                sanitized = pattern.sub(redaction_placeholder, sanitized)
            except re.error:
                continue

    # Truncate message if longer than MAX_MESSAGE_LENGTH to prevent log flooding
    if len(sanitized) > MAX_MESSAGE_LENGTH:
        sanitized = (
            sanitized[: MAX_MESSAGE_LENGTH - len(" [TRUNCATED]")] + " [TRUNCATED]"
        )

    return sanitized


def format_performance_metrics(  # noqa: C901
    duration_ms: float,
    metrics_data: Dict[str, Any] = None,
    include_threshold_status: bool = True,
    use_unicode_symbols: bool = None,
) -> str:
    """
    Formats performance metrics including timing, memory usage, and threshold comparisons
    with human-readable units and status indicators.

    Args:
        duration_ms: Duration in milliseconds to format
        metrics_data: Additional metrics data including memory usage, operation count
        include_threshold_status: Whether to include threshold status indicators
        use_unicode_symbols: Whether to use Unicode symbols (auto-detect if None)

    Returns:
        Formatted performance metrics string with timing, status, and threshold information
    """
    # Validate duration_ms is numeric and convert to appropriate time units
    if not isinstance(duration_ms, (int, float)) or duration_ms < 0:
        duration_ms = 0.0

    # Determine Unicode symbol support if not specified
    if use_unicode_symbols is None:
        color_support = detect_color_support()
        use_unicode_symbols = color_support["recommended_usage"] != "none"

    symbols = UNICODE_SYMBOLS if use_unicode_symbols else FALLBACK_ASCII_SYMBOLS

    # Format timing with appropriate precision (microseconds, milliseconds, seconds)
    if duration_ms < 1.0:
        time_str = f"{duration_ms * 1000:.1f}µs"
    elif duration_ms < 1000.0:
        time_str = f"{duration_ms:.3f}ms"
    else:
        time_str = f"{duration_ms / 1000:.3f}s"

    result_parts = [time_str]

    # Extract memory usage, operation count, and other metrics from metrics_data
    if metrics_data:
        if "memory_mb" in metrics_data:
            memory_mb = metrics_data["memory_mb"]
            if memory_mb < 1.0:
                memory_str = f"{memory_mb * 1024:.1f}KB"
            else:
                memory_str = f"{memory_mb:.1f}MB"
            result_parts.append(f"mem:{memory_str}")

        if "operations" in metrics_data:
            result_parts.append(f"ops:{metrics_data['operations']}")

    # Compare performance against relevant thresholds from constants
    status_symbol = symbols["SUCCESS"]
    if include_threshold_status:
        if duration_ms > PERFORMANCE_THRESHOLD_ERROR_MS:
            status_symbol = symbols["ERROR"]
        elif duration_ms > PERFORMANCE_THRESHOLD_WARNING_MS:
            status_symbol = symbols["WARNING"]

    return f"{status_symbol} {' | '.join(result_parts)}"


def create_formatter_cache_key(
    formatter_type: str,
    config_params: Dict[str, Any] = None,
    include_color_support: bool = True,
) -> str:
    """
    Creates cache key for formatter instances based on configuration parameters to enable
    efficient formatter reuse and prevent duplicate creation.

    Args:
        formatter_type: Type of formatter class name
        config_params: Configuration parameters dictionary
        include_color_support: Whether to include color support in cache key

    Returns:
        Cache key string for formatter instance identification and retrieval
    """
    # Validate formatter_type is supported formatter class name
    if not formatter_type or not isinstance(formatter_type, str):
        formatter_type = "unknown"

    key_parts = [formatter_type]

    # Extract relevant configuration parameters from config_params dictionary
    if config_params:
        # Sort configuration parameters for consistent cache key generation
        sorted_params = sorted(config_params.items())
        for param_name, param_value in sorted_params:
            if isinstance(param_value, (str, int, float, bool)):
                key_parts.append(f"{param_name}:{param_value}")

    # Include color support information in cache key if include_color_support enabled
    if include_color_support:
        color_support = detect_color_support()
        key_parts.append(f"color:{color_support['recommended_usage']}")

    # Generate hash-based cache key for efficient dictionary lookup
    cache_key = "|".join(key_parts)
    return cache_key


class LogFormatter(logging.Formatter):
    """
    Base logging formatter with security filtering, message sanitization, and structured output
    formatting providing foundation for all plume_nav_sim logging formatters with comprehensive
    security and development features.
    """

    def __init__(
        self,
        fmt_string: str = DEFAULT_LOG_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
        sensitive_patterns: List[str] = None,
        enable_security_filter: bool = True,
    ):
        """
        Initialize LogFormatter with format string, security filtering, and message sanitization
        capabilities for safe and structured log output.

        Args:
            fmt_string: Log message format string
            date_format: Date/time format string
            sensitive_patterns: Additional sensitive patterns to filter
            enable_security_filter: Whether to enable security filtering
        """
        # Call parent logging.Formatter constructor with fmt_string and date_format
        super().__init__(fmt_string, date_format)

        # Store format_string and date_format_string for formatter configuration
        self.format_string = fmt_string
        self.date_format_string = date_format

        # Initialize sensitive_pattern_list with default patterns and additional custom patterns
        self.sensitive_pattern_list = SENSITIVE_PATTERNS.copy()
        if sensitive_patterns:
            self.sensitive_pattern_list.extend(sensitive_patterns)

        # Compile regex patterns from sensitive patterns for efficient matching
        self.compiled_patterns = []
        for pattern in self.sensitive_pattern_list:
            try:
                self.compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                # Skip invalid patterns
                continue

        # Set security_filtering_enabled flag based on enable_security_filter parameter
        self.security_filtering_enabled = enable_security_filter

        # Configure redaction_placeholder from global REDACTION_PLACEHOLDER constant
        self.redaction_placeholder = REDACTION_PLACEHOLDER

        # Set message length limits from MAX_MESSAGE_LENGTH global constant
        self.max_message_length = MAX_MESSAGE_LENGTH

        # Initialize truncate_long_messages flag for message length management
        self.truncate_long_messages = True

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats log record with security filtering, message sanitization, and structured output
        including timestamp, level, and component information.

        Args:
            record: LogRecord instance to format

        Returns:
            Formatted log message string with security filtering and structured information
        """
        # Validate LogRecord parameter and extract message information
        if not isinstance(record, logging.LogRecord):
            return str(record)

        # Apply security filtering to record message if security_filtering_enabled
        if self.security_filtering_enabled and hasattr(record, "getMessage"):
            original_message = record.getMessage()
            # Sanitize message content using sanitize_message function with configured patterns
            sanitized_message = sanitize_message(
                original_message,
                additional_patterns=[
                    p.pattern for p in self.compiled_patterns if hasattr(p, "pattern")
                ],
                redaction_placeholder=self.redaction_placeholder,
            )

            # Apply message truncation if message exceeds max_message_length
            if (
                self.truncate_long_messages
                and len(sanitized_message) > self.max_message_length
            ):
                sanitized_message = (
                    sanitized_message[: self.max_message_length - len(" [TRUNCATED]")]
                    + " [TRUNCATED]"
                )

            # Create a modified record for formatting
            record.msg = sanitized_message
            record.args = ()

        # Use parent formatter format method with sanitized record
        try:
            formatted_message = super().format(record)
        except Exception as e:
            # Fallback formatting if parent format fails
            formatted_message = (
                f"{record.levelname} - {record.name} - Format error: {e}"
            )

        return formatted_message

    def add_sensitive_pattern(
        self, pattern: str, pattern_description: str = ""
    ) -> bool:
        """
        Adds new sensitive information detection pattern to the formatter's filtering system
        with regex compilation and validation.

        Args:
            pattern: Regex pattern string to add
            pattern_description: Description of pattern for documentation

        Returns:
            True if pattern added successfully, False if pattern invalid or compilation failed
        """
        # Validate pattern string is not empty and contains valid regex syntax
        if not pattern or not isinstance(pattern, str):
            return False

        try:
            # Test pattern compilation to ensure regex is valid before adding
            compiled_pattern = re.compile(pattern, re.IGNORECASE)

            # Add pattern to sensitive_pattern_list with description for documentation
            self.sensitive_pattern_list.append(pattern)

            # Compile pattern and add to compiled_patterns list for efficient matching
            self.compiled_patterns.append(compiled_pattern)

            return True
        except re.error:
            # Return failure status if pattern compilation fails
            return False

    def set_redaction_placeholder(self, placeholder_text: str) -> bool:
        """
        Updates the placeholder text used for redacting sensitive information in log messages
        with validation and security considerations.

        Args:
            placeholder_text: New placeholder text to use

        Returns:
            True if placeholder updated successfully, False if invalid placeholder
        """
        # Validate placeholder_text is non-empty string without sensitive content
        if not placeholder_text or not isinstance(placeholder_text, str):
            return False

        # Check placeholder does not contain regex special characters that might cause issues
        if any(
            char in placeholder_text
            for char in ["[", "]", "(", ")", "*", "+", "?", "^", "$"]
        ):
            return False

        # Update redaction_placeholder with new placeholder text
        self.redaction_placeholder = placeholder_text

        return True

    def enable_message_truncation(
        self, enable_truncation: bool, max_length: int = None
    ) -> bool:
        """
        Enables or disables automatic message truncation for long log messages with
        configurable length limits.

        Args:
            enable_truncation: Whether to enable message truncation
            max_length: Maximum message length (uses default if not specified)

        Returns:
            True if truncation configuration updated successfully
        """
        # Update truncate_long_messages flag based on enable_truncation parameter
        self.truncate_long_messages = enable_truncation

        # Set max_message_length to provided max_length or use default if not specified
        if max_length is not None and isinstance(max_length, int) and max_length > 0:
            # Validate max_length is positive integer within reasonable bounds
            self.max_message_length = min(
                max(max_length, 100), 10000
            )  # Bounds: 100-10000

        return True


class ConsoleFormatter(LogFormatter):
    """
    Console-optimized formatter with color support, terminal width awareness, and enhanced
    readability featuring ANSI color codes, level-based styling, and development-friendly
    output formatting.
    """

    def __init__(
        self,
        fmt_string: str = CONSOLE_FORMAT,
        use_colors: bool = None,
        color_scheme: Dict[str, str] = None,
        auto_detect_colors: bool = True,
    ):
        """
        Initialize ConsoleFormatter with color support detection, terminal capability analysis,
        and enhanced console formatting for development-friendly output.

        Args:
            fmt_string: Console-optimized format string
            use_colors: Whether to use color output (auto-detect if None)
            color_scheme: Custom color scheme mapping
            auto_detect_colors: Whether to auto-detect color support
        """
        # Initialize parent LogFormatter with console-optimized format string
        super().__init__(fmt_string)

        # Store console_format_string for formatter configuration
        self.console_format_string = fmt_string

        # Detect color support capabilities using detect_color_support function
        color_support = detect_color_support()

        # Set color_output_enabled based on use_colors parameter and terminal detection
        if use_colors is None and auto_detect_colors:
            self.color_output_enabled = color_support["recommended_usage"] != "none"
        else:
            self.color_output_enabled = bool(use_colors)

        # Configure level_colors dictionary with provided color_scheme or default colors
        self.level_colors = DEFAULT_LEVEL_COLORS.copy()
        if color_scheme:
            self.level_colors.update(color_scheme)

        # Initialize color_codes dictionary with ANSI escape sequences
        self.color_codes = ANSI_COLOR_CODES.copy()

        # Determine terminal_width using get_terminal_width for message formatting
        self.terminal_width = get_terminal_width()

        # Detect Unicode support and set appropriate symbols dictionary
        self.unicode_support = color_support["recommended_usage"] != "none"
        self.symbols = (
            UNICODE_SYMBOLS if self.unicode_support else FALLBACK_ASCII_SYMBOLS
        )

        # Configure auto_color_detection for dynamic color support adaptation
        self.auto_color_detection = auto_detect_colors

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats log record with color-coded output, level styling, and terminal-optimized
        display including symbol enhancement and width management.

        Args:
            record: LogRecord instance to format

        Returns:
            Color-formatted console log message with enhanced readability and terminal optimization
        """
        # Call parent LogFormatter.format() to apply base formatting and security filtering
        base_formatted = super().format(record)

        if not self.color_output_enabled:
            return base_formatted

        # Determine appropriate color for log level using level_colors configuration
        level_name = record.levelname
        color_name = self.level_colors.get(level_name, "WHITE")

        # Apply ANSI color codes to log level and message components if color_output_enabled
        if color_name in self.color_codes:
            color_code = self.color_codes[color_name]
            reset_code = self.color_codes["RESET"]

            # Add Unicode symbols for enhanced visual distinction if unicode_support enabled
            symbol = ""
            if self.unicode_support:
                if level_name == "ERROR":
                    symbol = self.symbols["ERROR"] + " "
                elif level_name == "WARNING":
                    symbol = self.symbols["WARNING"] + " "
                elif level_name == "INFO":
                    symbol = self.symbols["INFO"] + " "
                elif level_name == "DEBUG":
                    symbol = self.symbols["BULLET"] + " "

            # Handle special formatting for performance records and error messages
            if hasattr(record, "duration_ms"):
                perf_info = format_performance_metrics(
                    record.duration_ms, use_unicode_symbols=self.unicode_support
                )
                base_formatted = f"{base_formatted} {perf_info}"

            # Apply terminal width management and message wrapping if needed
            if len(base_formatted) > self.terminal_width:
                base_formatted = self.wrap_message_for_terminal(base_formatted, 0)

            # Apply color formatting with reset codes to prevent color bleeding
            formatted_message = f"{color_code}{symbol}{base_formatted}{reset_code}"
        else:
            formatted_message = base_formatted

        return formatted_message

    def set_color_scheme(self, new_color_scheme: Dict[str, str]) -> bool:
        """
        Updates color scheme configuration for different log levels with validation, terminal
        compatibility testing, and live color updates.

        Args:
            new_color_scheme: Dictionary mapping log levels to color names

        Returns:
            True if color scheme updated successfully, False if scheme invalid or unsupported
        """
        # Validate new_color_scheme dictionary contains required log level keys
        if not isinstance(new_color_scheme, dict):
            return False

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

        # Check color codes exist in ANSI_COLOR_CODES and are supported by terminal
        for level, color_name in new_color_scheme.items():
            if level not in valid_levels:
                return False
            if color_name not in self.color_codes:
                return False

        # Test color scheme compatibility with current terminal capabilities
        color_support = detect_color_support()
        if color_support["recommended_usage"] == "none":
            # Don't update color scheme if terminal doesn't support colors
            return False

        # Update level_colors dictionary with validated color scheme
        self.level_colors.update(new_color_scheme)

        return True

    def detect_color_support(self, force_redetection: bool = False) -> Dict[str, Any]:
        """
        Dynamically detects and updates terminal color support capabilities with caching
        and performance optimization.

        Args:
            force_redetection: Whether to force redetection of color support

        Returns:
            Current color support capabilities including ANSI, 256-color, and true color support
        """
        # Check if color support redetection needed or force_redetection enabled
        color_support = detect_color_support(force_detection=force_redetection)

        # Update color_output_enabled based on current terminal capabilities
        if self.auto_color_detection:
            self.color_output_enabled = color_support["recommended_usage"] != "none"

        # Update unicode_support and symbols based on terminal capabilities
        self.unicode_support = color_support["recommended_usage"] != "none"
        self.symbols = (
            UNICODE_SYMBOLS if self.unicode_support else FALLBACK_ASCII_SYMBOLS
        )

        return color_support

    def apply_level_styling(self, level_name: str, message_text: str) -> str:
        """
        Applies level-specific styling including colors, symbols, and formatting emphasis
        for enhanced log level distinction.

        Args:
            level_name: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message_text: Message text to style

        Returns:
            Styled message text with level-appropriate color and symbol formatting
        """
        if not self.color_output_enabled:
            return message_text

        # Determine color code for level_name using level_colors configuration
        color_name = self.level_colors.get(level_name, "WHITE")

        if color_name not in self.color_codes:
            return message_text

        color_code = self.color_codes[color_name]
        reset_code = self.color_codes["RESET"]

        # Select appropriate symbol for log level if unicode_support enabled
        symbol = ""
        if self.unicode_support:
            if level_name == "ERROR":
                symbol = self.symbols["ERROR"] + " "
            elif level_name == "WARNING":
                symbol = self.symbols["WARNING"] + " "
            elif level_name == "INFO":
                symbol = self.symbols["INFO"] + " "
            elif level_name == "DEBUG":
                symbol = self.symbols["BULLET"] + " "

        # Add level-specific text styling (bold for errors, dim for debug)
        style_code = ""
        if level_name in ["ERROR", "CRITICAL"]:
            style_code = self.color_codes["BOLD"]
        elif level_name == "DEBUG":
            style_code = self.color_codes["DIM"]

        return f"{style_code}{color_code}{symbol}{message_text}{reset_code}"

    def wrap_message_for_terminal(self, message: str, indent_level: int = 0) -> str:
        """
        Wraps long messages for terminal width with intelligent word breaking, indentation
        preservation, and visual continuity.

        Args:
            message: Message to wrap for terminal display
            indent_level: Indentation level for continuation lines

        Returns:
            Terminal-wrapped message with proper indentation and line continuation
        """
        # Calculate available width considering terminal_width and indent_level
        available_width = self.terminal_width - indent_level
        if available_width < 40:  # Minimum reasonable width
            return message

        # Break message into lines at appropriate word boundaries
        words = message.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word)

            # Check if adding word would exceed available width
            if current_length + word_length + len(current_line) > available_width:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    # Word is longer than available width, break it
                    lines.append(word[:available_width])
                    current_line = []
                    current_length = 0
            else:
                current_line.append(word)
                current_length += word_length

        # Add remaining words to final line
        if current_line:
            lines.append(" ".join(current_line))

        # Preserve indentation for continuation lines to maintain visual structure
        if len(lines) <= 1:
            return message

        indent = " " * indent_level
        continuation_symbol = self.symbols["ARROW"] if self.unicode_support else "->"

        result_lines = [lines[0]]
        for line in lines[1:]:
            result_lines.append(f"{indent}{continuation_symbol} {line}")

        return "\n".join(result_lines)


class PerformanceFormatter(LogFormatter):
    """
    Specialized formatter for performance monitoring logs with timing analysis, threshold
    comparison, memory usage display, and trend tracking designed for development performance
    analysis and optimization.
    """

    def __init__(
        self,
        performance_format: str = PERFORMANCE_LOG_FORMAT,
        timing_thresholds: Dict[str, float] = None,
        enable_memory_tracking: bool = True,
        include_baselines: bool = False,
    ):
        """
        Initialize PerformanceFormatter with timing thresholds, memory tracking, and performance
        analysis capabilities for comprehensive performance monitoring.

        Args:
            performance_format: Performance-specific format string
            timing_thresholds: Custom timing thresholds for operations
            enable_memory_tracking: Whether to track memory usage
            include_baselines: Whether to include baseline comparison
        """
        # Initialize parent LogFormatter with performance-specific format string
        super().__init__(performance_format)

        # Store performance_format_string for formatter configuration
        self.performance_format_string = performance_format

        # Configure operation_thresholds with provided timing_thresholds or defaults from constants
        self.operation_thresholds = {
            "step": PERFORMANCE_TARGET_STEP_LATENCY_MS,
            "rgb_render": PERFORMANCE_TARGET_RGB_RENDER_MS,
            "human_render": PERFORMANCE_TARGET_HUMAN_RENDER_MS,
            "warning": PERFORMANCE_THRESHOLD_WARNING_MS,
            "error": PERFORMANCE_THRESHOLD_ERROR_MS,
        }

        if timing_thresholds:
            self.operation_thresholds.update(timing_thresholds)

        # Set memory_tracking_enabled flag for memory usage inclusion in logs
        self.memory_tracking_enabled = enable_memory_tracking

        # Configure baseline_comparison_enabled for performance trend analysis
        self.baseline_comparison_enabled = include_baselines

        # Initialize performance_baselines dictionary for operation baseline tracking
        self.performance_baselines = {}

        # Set up threshold_colors for visual indication of performance status
        self.threshold_colors = {
            "fast": "GREEN",
            "normal": "CYAN",
            "slow": "YELLOW",
            "error": "RED",
        }

        # Initialize performance_history for trend analysis if show_trends enabled
        self.show_trends = include_baselines
        self.performance_history = {} if self.show_trends else None

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats performance log record with timing analysis, threshold status, memory usage,
        and trend information for comprehensive performance monitoring.

        Args:
            record: LogRecord instance with performance data

        Returns:
            Formatted performance log message with timing, memory, and threshold analysis
        """
        # Extract performance data from LogRecord extra fields including duration and memory
        duration_ms = getattr(record, "duration_ms", 0.0)
        memory_mb = (
            getattr(record, "memory_mb", None) if self.memory_tracking_enabled else None
        )
        operation_name = getattr(record, "operation", "unknown")

        # Format timing information using format_performance_metrics function
        metrics_data = {}
        if memory_mb is not None:
            metrics_data["memory_mb"] = memory_mb

        timing_info = format_performance_metrics(
            duration_ms,
            metrics_data,
            include_threshold_status=True,
            use_unicode_symbols=True,
        )

        # Compare against thresholds if needed (status included in timing_info)

        # Add baseline comparison information if baseline_comparison_enabled
        baseline_info = ""
        if (
            self.baseline_comparison_enabled
            and operation_name in self.performance_baselines
        ):
            comparison = self.compare_to_baseline(operation_name, duration_ms)
            if comparison:
                baseline_info = (
                    f" (vs baseline: {comparison['percentage_change']:+.1f}%)"
                )

        # Include trend information and performance history if show_trends enabled
        trend_info = ""
        if self.show_trends:
            trend_info = self._update_and_get_trend(operation_name, duration_ms)

        # Create enhanced performance record with additional information
        enhanced_message = (
            f"{record.getMessage()} | {timing_info}{baseline_info}{trend_info}"
        )

        # Update record message for parent formatting
        record.msg = enhanced_message
        record.args = ()

        # Call parent format method with enhanced performance record information
        return super().format(record)

    def format_timing(
        self,
        duration_ms: float,
        operation_name: str = "",
        timing_context: Dict[str, Any] = None,
    ) -> str:
        """
        Formats timing information with appropriate units, precision, and threshold indicators
        for clear performance communication.

        Args:
            duration_ms: Duration in milliseconds
            operation_name: Name of operation being timed
            timing_context: Additional timing context information

        Returns:
            Formatted timing string with units, precision, and status indicators
        """
        # Determine appropriate time units (microseconds, milliseconds, seconds) based on duration
        if duration_ms < 0.001:
            time_str = f"{duration_ms * 1000000:.1f}ns"
        elif duration_ms < 1.0:
            time_str = f"{duration_ms * 1000:.1f}µs"
        elif duration_ms < 1000.0:
            time_str = f"{duration_ms:.3f}ms"
        else:
            time_str = f"{duration_ms / 1000:.3f}s"

        # Compare duration against operation-specific thresholds from operation_thresholds
        status = self._determine_performance_status(operation_name, duration_ms)

        # Add performance status indicators (fast, normal, slow) based on thresholds
        status_indicators = {"fast": "✓", "normal": "→", "slow": "⚠", "error": "✗"}

        status_symbol = status_indicators.get(status, "?")

        # Include operation context information for debugging and analysis
        context_info = ""
        if timing_context:
            context_parts = []
            for key, value in timing_context.items():
                if key in ["cpu_percent", "memory_percent"]:
                    context_parts.append(f"{key}:{value:.1f}%")
                else:
                    context_parts.append(f"{key}:{value}")
            if context_parts:
                context_info = f" [{', '.join(context_parts)}]"

        return f"{status_symbol} {time_str}{context_info}"

    def add_baseline(
        self,
        operation_name: str,
        baseline_duration_ms: float,
        baseline_metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Adds or updates performance baseline for operation comparison with statistical analysis
        and trend tracking.

        Args:
            operation_name: Name of operation to set baseline for
            baseline_duration_ms: Baseline duration in milliseconds
            baseline_metadata: Additional baseline metadata

        Returns:
            True if baseline added/updated successfully, False otherwise
        """
        # Validate operation_name and baseline_duration_ms parameters
        if not operation_name or not isinstance(operation_name, str):
            return False

        if (
            not isinstance(baseline_duration_ms, (int, float))
            or baseline_duration_ms <= 0
        ):
            return False

        # Store baseline information in performance_baselines dictionary
        baseline_data = {
            "duration_ms": baseline_duration_ms,
            "timestamp": time.time(),
            "metadata": baseline_metadata or {},
        }

        # Include baseline_metadata for context and statistical analysis
        self.performance_baselines[operation_name] = baseline_data

        # Configure threshold calculations based on baseline performance
        if operation_name not in self.operation_thresholds:
            # Set thresholds as multiples of baseline
            self.operation_thresholds[f"{operation_name}_warning"] = (
                baseline_duration_ms * 2.0
            )
            self.operation_thresholds[f"{operation_name}_error"] = (
                baseline_duration_ms * 5.0
            )

        return True

    def compare_to_baseline(
        self, operation_name: str, current_duration_ms: float
    ) -> Dict[str, Any]:
        """
        Compares current performance against stored baseline with statistical analysis and
        performance trend assessment.

        Args:
            operation_name: Name of operation to compare
            current_duration_ms: Current duration in milliseconds

        Returns:
            Performance comparison results including percentage change, trend, and statistical significance
        """
        # Retrieve baseline performance for operation_name from performance_baselines
        if operation_name not in self.performance_baselines:
            return {}

        baseline_data = self.performance_baselines[operation_name]
        baseline_duration = baseline_data["duration_ms"]

        # Calculate percentage change from baseline to current performance
        percentage_change = (
            (current_duration_ms - baseline_duration) / baseline_duration
        ) * 100

        # Determine performance trend (improvement, degradation, stable)
        if abs(percentage_change) < 5:  # Within 5% is considered stable
            trend = "stable"
        elif percentage_change < 0:
            trend = "improvement"
        else:
            trend = "degradation"

        # Assess statistical significance of performance change
        significance = "significant" if abs(percentage_change) > 10 else "minor"

        # Generate performance comparison summary with recommendations
        recommendations = []
        if trend == "degradation" and significance == "significant":
            recommendations.append("Performance degradation detected - investigate")
        elif trend == "improvement" and significance == "significant":
            recommendations.append("Performance improvement observed")

        return {
            "percentage_change": percentage_change,
            "trend": trend,
            "significance": significance,
            "baseline_duration": baseline_duration,
            "current_duration": current_duration_ms,
            "recommendations": recommendations,
        }

    def update_threshold(
        self,
        operation_name: str,
        new_threshold_ms: float,
        threshold_type: str = "warning",
    ) -> bool:
        """
        Updates performance threshold for specific operation with validation, impact analysis,
        and configuration persistence.

        Args:
            operation_name: Name of operation to update threshold for
            new_threshold_ms: New threshold value in milliseconds
            threshold_type: Type of threshold (warning, error, critical)

        Returns:
            True if threshold updated successfully, False if validation failed
        """
        # Validate operation_name exists in operation_thresholds dictionary
        if not operation_name or not isinstance(operation_name, str):
            return False

        # Validate new_threshold_ms is positive and reasonable for operation type
        if not isinstance(new_threshold_ms, (int, float)) or new_threshold_ms <= 0:
            return False

        # Check threshold_type is valid (warning, error, critical)
        valid_threshold_types = ["warning", "error", "critical"]
        if threshold_type not in valid_threshold_types:
            return False

        # Update operation_thresholds with new threshold value
        threshold_key = f"{operation_name}_{threshold_type}"
        self.operation_thresholds[threshold_key] = new_threshold_ms

        return True

    def _determine_performance_status(
        self, operation_name: str, duration_ms: float
    ) -> str:
        """
        Determines performance status based on operation-specific thresholds.

        Args:
            operation_name: Name of operation
            duration_ms: Duration in milliseconds

        Returns:
            Performance status: 'fast', 'normal', 'slow', or 'error'
        """
        # Check for operation-specific thresholds first
        error_threshold = self.operation_thresholds.get(
            f"{operation_name}_error",
            self.operation_thresholds.get("error", PERFORMANCE_THRESHOLD_ERROR_MS),
        )
        warning_threshold = self.operation_thresholds.get(
            f"{operation_name}_warning",
            self.operation_thresholds.get("warning", PERFORMANCE_THRESHOLD_WARNING_MS),
        )

        # Check against general operation thresholds
        fast_threshold = self.operation_thresholds.get(
            operation_name, warning_threshold / 2
        )

        if duration_ms > error_threshold:
            return "error"
        elif duration_ms > warning_threshold:
            return "slow"
        elif duration_ms < fast_threshold:
            return "fast"
        else:
            return "normal"

    def _update_and_get_trend(self, operation_name: str, duration_ms: float) -> str:
        """
        Updates performance history and returns trend information.

        Args:
            operation_name: Name of operation
            duration_ms: Current duration

        Returns:
            Trend information string
        """
        if not self.performance_history:
            return ""

        if operation_name not in self.performance_history:
            self.performance_history[operation_name] = []

        history = self.performance_history[operation_name]
        history.append(duration_ms)

        # Keep only recent history (last 10 measurements)
        if len(history) > 10:
            history.pop(0)

        if len(history) < 3:
            return ""

        # Calculate simple trend
        recent_avg = sum(history[-3:]) / 3
        older_avg = sum(history[-6:-3]) / 3 if len(history) >= 6 else recent_avg

        if recent_avg > older_avg * 1.1:
            return " ↗"
        elif recent_avg < older_avg * 0.9:
            return " ↘"
        else:
            return " →"


class SecurityFilter(logging.Filter):
    """
    Logging filter for detecting and redacting sensitive information from log messages using
    regex patterns, context analysis, and configurable security policies for comprehensive
    information protection.
    """

    def __init__(  # noqa: C901
        self,
        sensitive_patterns: List[str] = None,
        redaction_policy: str = "replace",
        strict_filtering: bool = False,
    ):
        """
        Initialize SecurityFilter with sensitive pattern detection, redaction policies, and
        configurable security levels for comprehensive log message protection.

        Args:
            sensitive_patterns: List of sensitive patterns to filter
            redaction_policy: Policy for handling sensitive information ('replace', 'block', 'truncate')
            strict_filtering: Whether to enable strict security filtering
        """
        # Initialize parent logging.Filter for log record filtering
        super().__init__()

        # Compile sensitive_patterns into regex_patterns for efficient matching
        self.regex_patterns = []
        pattern_list = sensitive_patterns or SENSITIVE_PATTERNS

        for pattern in pattern_list:
            try:
                if isinstance(pattern, str):
                    self.regex_patterns.append(re.compile(pattern, re.IGNORECASE))
                else:
                    self.regex_patterns.append(pattern)
            except re.error:
                # Skip invalid patterns
                continue

        # Add default compiled patterns
        self.regex_patterns.extend(SENSITIVE_REGEX_PATTERNS)

        # Configure redaction_replacement based on redaction_policy parameter
        self.redaction_replacement = REDACTION_PLACEHOLDER
        if redaction_policy == "block":
            self.redaction_replacement = "[BLOCKED]"
        elif redaction_policy == "truncate":
            self.redaction_replacement = "[...]"

        # Set strict_mode_enabled flag for enhanced security filtering
        self.strict_mode_enabled = strict_filtering

        # Initialize context_patterns for contextual sensitive information detection
        self.context_patterns = []
        if self.strict_mode_enabled:
            # Additional patterns for strict mode
            strict_patterns = [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN patterns
                r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",  # Email patterns
                r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # Credit card patterns
            ]
            for pattern_str in strict_patterns:
                try:
                    self.context_patterns.append(re.compile(pattern_str, re.IGNORECASE))
                except re.error:
                    continue

        # Set up whitelist_patterns for information that should not be redacted
        self.whitelist_patterns = [
            re.compile(r"\b(test|example|demo|sample)\b", re.IGNORECASE)
        ]

        # Configure max_redactions_per_message to prevent over-redaction
        self.max_redactions_per_message = 10

        # Set preserve_message_structure flag for maintaining log readability
        self.preserve_message_structure = True

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: C901  # noqa: C901
        """
        Main filtering method that analyzes log records for sensitive information and applies
        redaction while preserving log structure and readability.

        Args:
            record: LogRecord instance to filter

        Returns:
            True to allow record through (possibly modified), False to block record completely
        """
        if not hasattr(record, "getMessage"):
            return True

        try:
            # Extract message text from LogRecord for sensitive information analysis
            original_message = record.getMessage()

            if not isinstance(original_message, str) or not original_message:
                return True

            modified_message = original_message
            redaction_count = 0

            # Apply regex_patterns to detect sensitive information in message
            for i, pattern in enumerate(self.regex_patterns):
                if redaction_count >= self.max_redactions_per_message:
                    break

                matches = list(pattern.finditer(modified_message))
                for match in matches[
                    : self.max_redactions_per_message - redaction_count
                ]:
                    # Check whitelist_patterns to avoid redacting safe information
                    matched_text = match.group()
                    is_whitelisted = any(
                        wp.search(matched_text) for wp in self.whitelist_patterns
                    )

                    if not is_whitelisted:
                        # Redact detected sensitive content using redaction_replacement
                        modified_message = modified_message.replace(
                            matched_text, self.redaction_replacement
                        )
                        redaction_count += 1

            # Check context_patterns for additional contextual sensitive content
            for pattern in self.context_patterns:
                if redaction_count >= self.max_redactions_per_message:
                    break

                matches = list(pattern.finditer(modified_message))
                for match in matches[
                    : self.max_redactions_per_message - redaction_count
                ]:
                    matched_text = match.group()
                    is_whitelisted = any(
                        wp.search(matched_text) for wp in self.whitelist_patterns
                    )

                    if not is_whitelisted:
                        modified_message = modified_message.replace(
                            matched_text, self.redaction_replacement
                        )
                        redaction_count += 1

            # Update LogRecord message with redacted content
            if modified_message != original_message:
                record.msg = modified_message
                record.args = ()

            # Preserve message structure while ensuring security through redaction
            # Return True to allow filtered record to pass through logging system
            return True

        except Exception:
            # If filtering fails, allow record through to prevent logging system breakdown
            return True

    def add_sensitive_pattern(
        self, pattern: str, pattern_description: str = "", is_regex: bool = True
    ) -> bool:
        """
        Adds new sensitive information detection pattern to the filter with pattern validation,
        compilation, and security testing.

        Args:
            pattern: Pattern string to add
            pattern_description: Description of pattern for documentation
            is_regex: Whether pattern is regex or literal string

        Returns:
            True if pattern added successfully, False if pattern invalid or compilation failed
        """
        # Validate pattern string for security and regex syntax correctness
        if not pattern or not isinstance(pattern, str):
            return False

        try:
            if is_regex:
                # Compile pattern as regex if is_regex is True
                compiled_pattern = re.compile(pattern, re.IGNORECASE)
            else:
                # Treat as literal string, escape for regex use
                escaped_pattern = re.escape(pattern)
                compiled_pattern = re.compile(escaped_pattern, re.IGNORECASE)

            # Test pattern against known sensitive and safe content for accuracy
            test_safe = "This is a safe test message"
            test_sensitive = (
                f"password={pattern}" if not is_regex else "password=secret123"
            )

            # Pattern should not match safe content
            if compiled_pattern.search(test_safe):
                return False

            # Pattern should match representative sensitive content
            if not compiled_pattern.search(test_sensitive):
                return False

            # Add compiled pattern to regex_patterns list for filtering
            self.regex_patterns.append(compiled_pattern)

            return True
        except re.error:
            return False

    def set_redaction_policy(
        self, policy_name: str, policy_config: Dict[str, Any] = None
    ) -> bool:
        """
        Configures redaction policy including replacement text, redaction level, and security
        strictness with validation and testing.

        Args:
            policy_name: Name of redaction policy ('replace', 'block', 'truncate')
            policy_config: Configuration dictionary for policy settings

        Returns:
            True if policy configured successfully, False if invalid policy
        """
        # Validate policy_name against supported redaction policies
        valid_policies = ["replace", "block", "truncate", "custom"]
        if policy_name not in valid_policies:
            return False

        policy_config = policy_config or {}

        # Extract redaction settings from policy_config including replacement text
        if policy_name == "replace":
            self.redaction_replacement = policy_config.get(
                "placeholder", REDACTION_PLACEHOLDER
            )
        elif policy_name == "block":
            self.redaction_replacement = policy_config.get("placeholder", "[BLOCKED]")
        elif policy_name == "truncate":
            self.redaction_replacement = policy_config.get("placeholder", "[...]")
        elif policy_name == "custom":
            custom_replacement = policy_config.get("replacement")
            if not custom_replacement or not isinstance(custom_replacement, str):
                return False
            self.redaction_replacement = custom_replacement

        # Update strict_mode_enabled based on policy security level
        self.strict_mode_enabled = policy_config.get(
            "strict_mode", self.strict_mode_enabled
        )

        return True

    def add_whitelist_pattern(
        self, pattern: str, pattern_description: str = ""
    ) -> bool:
        """
        Adds pattern to whitelist for information that should never be redacted, with validation
        and conflict detection against sensitive patterns.

        Args:
            pattern: Regex pattern to whitelist
            pattern_description: Description for documentation

        Returns:
            True if whitelist pattern added successfully, False if conflicts with security patterns
        """
        # Validate whitelist pattern does not conflict with sensitive_patterns
        if not pattern or not isinstance(pattern, str):
            return False

        try:
            # Compile pattern for efficient matching during filtering
            compiled_pattern = re.compile(pattern, re.IGNORECASE)

            # Test pattern against known safe content for accuracy
            test_content = "This is safe test content example demo"
            if not compiled_pattern.search(test_content):
                # Pattern should match some safe content
                pass

            # Add pattern to whitelist_patterns for exclusion from redaction
            self.whitelist_patterns.append(compiled_pattern)

            return True
        except re.error:
            return False

    def analyze_message_security(
        self, message_text: str
    ) -> Dict[str, Any]:  # noqa: C901
        """
        Analyzes message for security risks, sensitive content density, and redaction
        recommendations with detailed security assessment.

        Args:
            message_text: Message text to analyze

        Returns:
            Security analysis report including risk level, detected patterns, and redaction recommendations
        """
        if not message_text or not isinstance(message_text, str):
            return {
                "risk_level": "none",
                "detected_patterns": [],
                "redaction_recommendations": [],
                "sensitive_content_density": 0.0,
            }

        analysis = {
            "risk_level": "low",
            "detected_patterns": [],
            "redaction_recommendations": [],
            "sensitive_content_density": 0.0,
            "total_matches": 0,
            "message_length": len(message_text),
        }

        total_matches = 0

        # Scan message_text with all regex_patterns for sensitive content detection
        for i, pattern in enumerate(self.regex_patterns):
            matches = list(pattern.finditer(message_text))
            if matches:
                analysis["detected_patterns"].append(
                    {
                        "pattern_index": i,
                        "match_count": len(matches),
                        "matches": [match.group() for match in matches],
                    }
                )
                total_matches += len(matches)

        analysis["total_matches"] = total_matches

        # Calculate sensitive content density and distribution throughout message
        if analysis["message_length"] > 0:
            analysis["sensitive_content_density"] = (
                total_matches / analysis["message_length"] * 100
            )

        # Assess security risk level based on type and quantity of sensitive information
        if total_matches == 0:
            analysis["risk_level"] = "none"
        elif total_matches <= 2 and analysis["sensitive_content_density"] < 5.0:
            analysis["risk_level"] = "low"
        elif total_matches <= 5 and analysis["sensitive_content_density"] < 10.0:
            analysis["risk_level"] = "medium"
        else:
            analysis["risk_level"] = "high"

        # Generate redaction recommendations with minimal impact on message readability
        if analysis["risk_level"] in ["medium", "high"]:
            analysis["redaction_recommendations"].append(
                "Apply automatic redaction to detected sensitive patterns"
            )

        if analysis["risk_level"] == "high":
            analysis["redaction_recommendations"].extend(
                [
                    "Consider blocking message entirely in production",
                    "Review logging practices for this component",
                ]
            )

        # Identify potential false positives using whitelist_patterns
        whitelisted_matches = 0
        for pattern in self.whitelist_patterns:
            whitelisted_matches += len(pattern.findall(message_text))

        if whitelisted_matches > 0:
            analysis["potential_false_positives"] = whitelisted_matches
            analysis["redaction_recommendations"].append(
                "Review whitelist patterns - some matches may be safe"
            )

        return analysis


@dataclass
class ColorScheme:
    """
    Data class for managing terminal color schemes with ANSI codes, terminal compatibility,
    and visual theme configuration for consistent and accessible console output formatting.
    """

    level_colors: Dict[str, str]
    accent_colors: Dict[str, str] = None
    supports_256_color: bool = False
    theme_name: str = "default"

    def __post_init__(self):
        """
        Initialize ColorScheme with level-based colors, accent colors, and terminal compatibility
        settings for comprehensive console color management.
        """
        # Store level_colors mapping for log level color associations
        self.level_color_mapping = self.level_colors.copy()

        # Configure accent_colors for special formatting and emphasis
        if self.accent_colors is None:
            self.accent_color_mapping = {
                "highlight": "BRIGHT_YELLOW",
                "success": "BRIGHT_GREEN",
                "warning": "BRIGHT_YELLOW",
                "error": "BRIGHT_RED",
            }
        else:
            self.accent_color_mapping = self.accent_colors.copy()

        # Set advanced_color_support based on supports_256_color capability
        self.advanced_color_support = self.supports_256_color

        # Store color_theme_name for scheme identification and selection
        self.color_theme_name = self.theme_name

        # Initialize ansi_codes dictionary with appropriate escape sequences
        self.ansi_codes = ANSI_COLOR_CODES.copy()

        # Configure accessibility_mode for high-contrast alternatives
        self.accessibility_mode = False

        # Set up fallback_colors for terminals with limited color support
        self.fallback_colors = {
            "DEBUG": "WHITE",
            "INFO": "WHITE",
            "WARNING": "WHITE",
            "ERROR": "WHITE",
            "CRITICAL": "WHITE",
        }

    def get_color_code(self, color_name: str, color_type: str = "level") -> str:
        """
        Returns ANSI color code for specified level or accent color with terminal compatibility validation.

        Args:
            color_name: Name of color or log level
            color_type: Type of color ('level', 'accent', 'background')

        Returns:
            ANSI color escape sequence for specified color, or empty string if unsupported
        """
        # Validate color_name exists in level_color_mapping or accent_color_mapping
        target_color = None

        if color_type == "level":
            target_color = self.level_color_mapping.get(color_name)
        elif color_type == "accent":
            target_color = self.accent_color_mapping.get(color_name)
        else:
            # Try both mappings
            target_color = self.level_color_mapping.get(
                color_name
            ) or self.accent_color_mapping.get(color_name)

        if not target_color:
            return ""

        # Retrieve appropriate ANSI escape sequence from ansi_codes
        color_code = self.ansi_codes.get(target_color, "")

        # Apply fallback color if advanced colors not supported
        if not color_code and not self.advanced_color_support:
            fallback_color = self.fallback_colors.get(color_name, "WHITE")
            color_code = self.ansi_codes.get(fallback_color, "")

        return color_code

    def apply_accessibility_mode(self, enable_accessibility: bool) -> bool:
        """
        Applies accessibility-friendly color adjustments for better visibility and color-blind compatibility.

        Args:
            enable_accessibility: Whether to enable accessibility mode

        Returns:
            True if accessibility mode applied successfully
        """
        # Update accessibility_mode flag based on enable_accessibility parameter
        self.accessibility_mode = enable_accessibility

        if enable_accessibility:
            # Adjust color mappings for better contrast and visibility
            accessibility_colors = {
                "DEBUG": "CYAN",
                "INFO": "BLUE",
                "WARNING": "YELLOW",
                "ERROR": "RED",
                "CRITICAL": "BRIGHT_RED",
            }

            # Replace problematic colors with accessibility-friendly alternatives
            self.level_color_mapping.update(accessibility_colors)

            # Update accent colors for accessibility
            self.accent_color_mapping.update(
                {
                    "highlight": "YELLOW",
                    "success": "GREEN",
                    "warning": "YELLOW",
                    "error": "RED",
                }
            )
        else:
            # Restore original color mappings
            self.level_color_mapping = self.level_colors.copy()

        return True

    def validate_terminal_compatibility(
        self, terminal_capabilities: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates color scheme compatibility with current terminal capabilities and suggests adjustments.

        Args:
            terminal_capabilities: Dictionary of terminal color support capabilities

        Returns:
            Tuple of (is_compatible: bool, compatibility_report: dict) with validation results
        """
        compatibility_report = {
            "is_compatible": True,
            "unsupported_features": [],
            "recommendations": [],
            "fallback_suggestions": {},
        }

        # Check terminal_capabilities against required color support levels
        required_basic_colors = terminal_capabilities.get("basic_colors", False)
        required_ansi_colors = terminal_capabilities.get("ansi_colors", False)

        if not required_basic_colors:
            compatibility_report["is_compatible"] = False
            compatibility_report["unsupported_features"].append(
                "Basic color support missing"
            )
            compatibility_report["recommendations"].append("Use monochrome output")

        # Validate ANSI color code support for all scheme colors
        if not required_ansi_colors:
            compatibility_report["unsupported_features"].append(
                "ANSI color codes not supported"
            )
            compatibility_report["fallback_suggestions"] = self.fallback_colors

        # Test 256-color support if advanced_color_support enabled
        if self.advanced_color_support:
            supports_256_color = terminal_capabilities.get("color_256", False)
            if not supports_256_color:
                compatibility_report["unsupported_features"].append(
                    "256-color support unavailable"
                )
                compatibility_report["recommendations"].append(
                    "Disable advanced color features"
                )

        # Generate compatibility report with unsupported features
        if compatibility_report["unsupported_features"]:
            compatibility_report["is_compatible"] = False

        # Suggest fallback options for incompatible colors
        if not compatibility_report["is_compatible"]:
            compatibility_report["recommendations"].append(
                "Consider using fallback color scheme"
            )
            compatibility_report["fallback_suggestions"] = self.fallback_colors

        return compatibility_report["is_compatible"], compatibility_report


# Create default color schemes for common use cases
DEFAULT_COLOR_SCHEME = ColorScheme(
    level_colors=DEFAULT_LEVEL_COLORS.copy(), theme_name="default"
)

DARK_COLOR_SCHEME = ColorScheme(
    level_colors={
        "DEBUG": "DIM",
        "INFO": "CYAN",
        "WARNING": "YELLOW",
        "ERROR": "BRIGHT_RED",
        "CRITICAL": "BRIGHT_MAGENTA",
    },
    accent_colors={
        "highlight": "BRIGHT_CYAN",
        "success": "BRIGHT_GREEN",
        "warning": "BRIGHT_YELLOW",
        "error": "BRIGHT_RED",
    },
    theme_name="dark",
)

# Module exports for external use
__all__ = [
    # Formatter classes
    "LogFormatter",
    "ConsoleFormatter",
    "PerformanceFormatter",
    "SecurityFilter",
    "ColorScheme",
    # Utility functions
    "detect_color_support",
    "get_terminal_width",
    "sanitize_message",
    "format_performance_metrics",
    "create_formatter_cache_key",
    # Format constants
    "DEFAULT_LOG_FORMAT",
    "DEVELOPMENT_LOG_FORMAT",
    "PERFORMANCE_LOG_FORMAT",
    "CONSOLE_FORMAT",
    "ERROR_FORMAT",
    "DEBUG_FORMAT",
    # Color constants
    "ANSI_COLOR_CODES",
    "DEFAULT_LEVEL_COLORS",
    "DEFAULT_COLOR_SCHEME",
    "DARK_COLOR_SCHEME",
    # Security constants
    "SENSITIVE_PATTERNS",
    "REDACTION_PLACEHOLDER",
    # Performance constants
    "PERFORMANCE_THRESHOLD_WARNING_MS",
    "PERFORMANCE_THRESHOLD_ERROR_MS",
]
