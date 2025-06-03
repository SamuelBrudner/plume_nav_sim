"""
Helper utilities for testing the refactored cookiecutter template structure.

This module provides comprehensive test utilities for validating the migration from
the legacy 'odor_plume_nav' package structure to the standardized 
'{{cookiecutter.project_slug}}' template organization. It supports the enhanced
cookiecutter-based architecture with systematic validation across configuration
management, CLI interfaces, database session stubs, and core navigation components.

The helper utilities ensure import compliance with the new package organization
including core/, data/, config/, api/, utils/, cli/, and db/ subdirectories,
providing systematic validation for the refactored system per Section 0.2.1
technical reorganization requirements.

Module Components:
    - Import validation utilities for enforcing new package structure
    - Mapping dictionaries for legacy-to-cookiecutter template migration
    - Compliance checking for {{cookiecutter.project_slug}} namespace validation
    - Enhanced support for Hydra configuration, CLI interfaces, and database components

Usage:
    from tests.helpers import assert_imported_from, TARGET_IMPORT_MAPPING
    
    # Validate imports from new cookiecutter template structure
    assert_imported_from(Navigator, "{{cookiecutter.project_slug}}.core.navigator")
    
    # Access target import mappings for template validation
    target_module = TARGET_IMPORT_MAPPING.get("NavigatorProtocol")
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
    # Import validator functions for cookiecutter template compliance
    'assert_imported_from',
    'assert_all_imported_from', 
    'enforce_import_compliance',
    'get_target_module',
    
    # Mapping dictionaries and lists for template migration validation
    'TARGET_IMPORT_MAPPING',
    'FILES_TO_MIGRATE',
    'LEGACY_IMPORT_PATHS'
]