"""
CLI Testing Package for {{cookiecutter.project_slug}}.

This package contains comprehensive test suites for command-line interface validation,
Click framework integration testing, and Hydra configuration composition validation.
The CLI testing infrastructure enables systematic validation of command-line interface
components with ≥80% coverage target per Section 6.6.3.1.

Key Testing Capabilities:
- pytest test discovery for CLI test modules
- click.testing.CliRunner integration for command-line interface validation
- Hydra configuration override testing through CLI parameters
- Parameter validation and error handling verification
- Interactive prompt and confirmation dialog testing
- CLI command initialization performance validation (<2s)
- Command execution testing with exit code verification
- Help text generation and usage documentation testing

Test Organization:
- test_cli_main.py: Comprehensive CLI interface validation
- CLI test utilities and fixtures for command validation
- Structured test organization for CLI components per Section 6.6.5.1

Dependencies:
- click.testing.CliRunner for command-line interface testing
- pytest-hydra plugin for configuration composition testing
- unittest.mock for CLI command mocking and isolation
- Hydra configuration override testing capabilities

Usage:
    pytest tests/cli/                    # Run all CLI tests
    pytest tests/cli/test_cli_main.py    # Run specific CLI test module
    pytest tests/cli/ -v                 # Verbose CLI test execution
    pytest tests/cli/ --cov=cli          # CLI coverage analysis

Testing Standards:
- All CLI commands must pass CliRunner validation
- Command initialization timing validation (<2 seconds)
- Parameter parsing security and validation testing
- Environment variable integration testing
- Hydra configuration override scenario validation
- Error handling and recovery strategy testing
"""

# Enable pytest test discovery
# This __init__.py file ensures pytest can discover and execute CLI tests
# following the pytest test discovery conventions and enabling
# comprehensive CLI testing infrastructure validation

# CLI testing package designation for pytest
__all__ = []

# Package metadata for CLI testing module
__version__ = "1.0.0"
__description__ = "CLI Testing Package for {{cookiecutter.project_slug}}"