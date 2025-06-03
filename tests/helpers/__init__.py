"""Helper utilities for testing the refactored module structure.

This module provides comprehensive testing utilities for validating import compliance
and module organization in the cookiecutter-based {{cookiecutter.project_slug}} package.
The helpers support testing of the new hierarchical module structure while maintaining
compatibility with the existing test validation infrastructure.

Key Testing Capabilities:
- Import validation for new package structure  
- Module mapping verification for cookiecutter templates
- Legacy import path compatibility checking
- Test compliance enforcement across all modules

Updated for cookiecutter template compatibility and new module organization per
Section 0.2.1 mapping requirements.
"""

from tests.helpers.import_validator import (
    assert_imported_from,
    assert_all_imported_from,
    enforce_import_compliance,
    get_target_module
)

from tests.helpers.import_mapping import (
    TARGET_IMPORT_MAPPING,
    FILES_TO_MIGRATE,
    LEGACY_IMPORT_PATHS
)

__all__ = [
    # Import validator functions for new package structure
    'assert_imported_from',
    'assert_all_imported_from', 
    'enforce_import_compliance',
    'get_target_module',
    
    # Mapping dictionaries and lists for cookiecutter template validation
    'TARGET_IMPORT_MAPPING',
    'FILES_TO_MIGRATE',
    'LEGACY_IMPORT_PATHS'
]

# Package metadata for testing infrastructure
__version__ = "1.0.0"
__package_name__ = "{{cookiecutter.project_slug}}"

def validate_new_import_structure():
    """
    Convenience function to validate the complete new import structure.
    
    This function provides a single entry point for comprehensive validation
    of the updated package structure and import compliance per the cookiecutter
    template requirements.
    
    Returns:
        bool: True if all imports comply with new structure, False otherwise
    
    Raises:
        AssertionError: If import violations are detected
    """
    try:
        enforce_import_compliance()
        return True
    except AssertionError:
        return False

def get_cookiecutter_project_name():
    """
    Get the cookiecutter project name for template-aware testing.
    
    Returns:
        str: The cookiecutter project slug for import validation
    """
    return __package_name__