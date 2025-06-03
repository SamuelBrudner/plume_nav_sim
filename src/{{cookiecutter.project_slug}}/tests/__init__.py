"""
Library-internal test package for {{cookiecutter.project_slug}}.

This package initializer establishes the test namespace for the odor plume navigation 
library and enables pytest test discovery for the library-internal test suite. This 
structure supports both unit and integration testing scenarios while maintaining 
separation from root-level integration tests.

Test Organization:
    - Unit tests: Isolated component testing with comprehensive mock coverage
    - Integration tests: Cross-component validation and API surface testing
    - Configuration tests: Hydra configuration composition and validation
    - CLI tests: Command-line interface validation through Click testing utilities
    - Database tests: SQLAlchemy session management with in-memory fixtures
    - Workflow tests: DVC/Snakemake pipeline validation and orchestration

Pytest Discovery:
    This module enables automatic test discovery by pytest across all test modules
    within the library package structure. Tests are organized by component domain
    following the enhanced cookiecutter-based architecture requirements.

Coverage Requirements:
    - Overall target: >80% line coverage across all library modules
    - Critical path coverage: >90% for core navigation components
    - Configuration system: >85% coverage for Hydra integration
    - CLI interfaces: >85% coverage for command validation
    - Database sessions: >80% coverage for persistence layer

Fixture Sharing:
    Test fixtures are shared across modules through conftest.py configuration,
    enabling consistent mock behavior and test data management across the
    library test suite. This includes enhanced fixtures for:
    - Hydra configuration composition testing
    - CLI command validation through CliRunner
    - In-memory database session management
    - Workflow orchestration mocking

Testing Best Practices:
    - Deterministic test execution through controlled randomization
    - Isolated test environments with comprehensive fixture teardown
    - Scientific accuracy validation using numpy.testing assertions
    - Configuration-driven test scenarios for research reproducibility
    - Performance validation against specified SLA requirements

Integration Points:
    - pytest-hydra plugin for advanced configuration testing
    - click.testing.CliRunner for CLI interface validation
    - SQLAlchemy in-memory sessions for database testing
    - pytest-mock for workflow orchestration testing
    - numpy.testing for numerical precision validation

Example Usage:
    Run all library tests:
        pytest src/{{cookiecutter.project_slug}}/tests/

    Run specific test categories:
        pytest src/{{cookiecutter.project_slug}}/tests/core/
        pytest src/{{cookiecutter.project_slug}}/tests/config/
        pytest src/{{cookiecutter.project_slug}}/tests/cli/

    Run with coverage analysis:
        pytest --cov={{cookiecutter.project_slug}} src/{{cookiecutter.project_slug}}/tests/

See Also:
    - tests/conftest.py: Shared fixture definitions and test configuration
    - Section 6.6 of Technical Specification: Comprehensive testing strategy
    - README.md: Project-specific testing guidelines and examples
"""

# Package version for test compatibility tracking
__version__ = "1.0.0"

# Test category markers for pytest organization
__test_categories__ = [
    "unit",
    "integration", 
    "config",
    "cli",
    "database",
    "workflow",
    "performance",
    "security"
]

# Pytest discovery support - ensure this package is recognized
__all__ = [
    "__version__",
    "__test_categories__"
]


def pytest_configure(config):
    """
    Pytest configuration hook for library-internal test customization.
    
    This function is automatically called by pytest during test discovery
    and allows for test-specific configuration without interfering with
    root-level test configuration.
    
    Args:
        config: pytest configuration object
        
    Note:
        This hook is optional and only executed when pytest discovers
        this package during test collection.
    """
    # Register custom markers for test categorization
    for category in __test_categories__:
        config.addinivalue_line(
            "markers", 
            f"{category}: mark test as {category} test category"
        )


def get_test_data_path():
    """
    Utility function to resolve test data directory path.
    
    Returns:
        pathlib.Path: Absolute path to test data directory
        
    Example:
        >>> from src.{{cookiecutter.project_slug}}.tests import get_test_data_path
        >>> test_data_dir = get_test_data_path()
        >>> config_file = test_data_dir / "sample_config.yaml"
    """
    import pathlib
    return pathlib.Path(__file__).parent / "data"


def create_temp_config_file(config_dict, suffix=".yaml"):
    """
    Utility function for creating temporary configuration files in tests.
    
    Args:
        config_dict (dict): Configuration data to write
        suffix (str): File extension for the temporary file
        
    Returns:
        pathlib.Path: Path to the created temporary file
        
    Example:
        >>> config = {"navigator": {"num_agents": 2}}
        >>> temp_file = create_temp_config_file(config)
        >>> # Use temp_file in test, automatic cleanup via pytest tmp_path
    """
    import tempfile
    import yaml
    import pathlib
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        if suffix.endswith('.yaml') or suffix.endswith('.yml'):
            yaml.safe_dump(config_dict, f)
        elif suffix.endswith('.json'):
            import json
            json.dump(config_dict, f, indent=2)
        else:
            f.write(str(config_dict))
    
    return pathlib.Path(f.name)


# Compatibility check for required test dependencies
def check_test_dependencies():
    """
    Verify that required testing dependencies are available.
    
    Returns:
        dict: Status of each required testing dependency
        
    Raises:
        ImportError: If critical testing dependencies are missing
    """
    dependencies = {}
    
    try:
        import pytest
        dependencies['pytest'] = pytest.__version__
    except ImportError:
        raise ImportError("pytest is required for test execution")
    
    try:
        import numpy.testing
        dependencies['numpy'] = numpy.__version__
    except ImportError:
        raise ImportError("numpy is required for numerical test assertions")
    
    # Optional enhanced testing dependencies
    optional_deps = {
        'pytest_hydra': 'Hydra configuration testing',
        'click.testing': 'CLI interface testing', 
        'sqlalchemy': 'Database session testing',
        'pytest_mock': 'Workflow orchestration mocking'
    }
    
    for dep_name, description in optional_deps.items():
        try:
            if '.' in dep_name:
                # Handle nested imports like click.testing
                module_parts = dep_name.split('.')
                module = __import__(module_parts[0])
                for part in module_parts[1:]:
                    module = getattr(module, part)
                dependencies[dep_name] = getattr(module, '__version__', 'available')
            else:
                module = __import__(dep_name)
                dependencies[dep_name] = getattr(module, '__version__', 'available')
        except ImportError:
            dependencies[dep_name] = f'not available - {description}'
    
    return dependencies


# Initialize test environment check on import
if __name__ != "__main__":
    # Only run dependency check during normal import, not direct execution
    try:
        _test_deps = check_test_dependencies()
    except ImportError as e:
        import warnings
        warnings.warn(f"Test dependency issue: {e}", UserWarning)