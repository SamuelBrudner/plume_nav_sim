"""
Import validation helper for tests.

This module provides utilities to ensure that imports in tests are coming from 
the correct modules in the refactored cookiecutter template structure, helping 
enforce the target state for the {{cookiecutter.project_slug}} package organization.

The validator supports the new module organization including core/, data/, config/, 
api/, utils/, cli/, and db/ subdirectories, validating imports against the 
standardized project template structure.
"""

import sys
import inspect
from pathlib import Path
from types import ModuleType
from typing import Dict, Any, List, Optional, Set, Tuple

# Import the target mapping
from tests.helpers.import_mapping import TARGET_IMPORT_MAPPING


class ImportValidator:
    """Validates that objects are imported from the expected modules in the cookiecutter template structure."""
    
    # Valid subdirectories in the new package structure
    VALID_SUBDIRECTORIES = {
        'core', 'data', 'config', 'api', 'utils', 'cli', 'db'
    }
    
    @staticmethod
    def validate_import(obj: Any, expected_module_path: str) -> bool:
        """
        Check if an object was imported from the expected module.
        
        Args:
            obj: The object to check
            expected_module_path: The expected module path (e.g., '{{cookiecutter.project_slug}}.core.navigator')
            
        Returns:
            True if the object was imported from the expected module, False otherwise
        """
        # Get the actual module path
        if not hasattr(obj, '__module__'):
            return False
            
        actual_module_path = obj.__module__
        # Check if the actual module path matches the expected one
        return actual_module_path == expected_module_path
    
    @staticmethod
    def validate_module_structure(module_path: str) -> bool:
        """
        Validate that a module path follows the expected cookiecutter template structure.
        
        Args:
            module_path: The module path to validate
            
        Returns:
            True if the module path follows the expected structure, False otherwise
        """
        if not module_path.startswith('{{cookiecutter.project_slug}}.'):
            return False
            
        # Extract the subdirectory
        parts = module_path.split('.')
        if len(parts) < 3:  # Need at least project_slug.subdirectory.module
            return False
            
        subdirectory = parts[1]
        return subdirectory in ImportValidator.VALID_SUBDIRECTORIES
    
    @staticmethod
    def validate_imports(objects: Dict[str, Any], 
                        expected_module_path: str) -> Tuple[bool, List[str]]:
        """
        Check if multiple objects were imported from the expected module.
        
        Args:
            objects: Dictionary mapping names to objects
            expected_module_path: The expected module path
            
        Returns:
            Tuple of (success, list of failed imports)
        """
        failures = []
        
        # First validate the expected module path structure
        if not ImportValidator.validate_module_structure(expected_module_path):
            failures.append(f"Expected module path '{expected_module_path}' does not follow cookiecutter template structure")
        
        for name, obj in objects.items():
            if not ImportValidator.validate_import(obj, expected_module_path):
                actual_module = getattr(obj, '__module__', 'unknown')
                failures.append(f"{name} imported from {actual_module} not {expected_module_path}")
        
        return not failures, failures


def assert_imported_from(obj: Any, expected_module_path: str) -> None:
    """
    Assert that an object was imported from the expected cookiecutter template module.
    
    Args:
        obj: The object to check
        expected_module_path: The expected module path in {{cookiecutter.project_slug}} namespace
        
    Raises:
        AssertionError: If the object was not imported from the expected module
    """
    # First validate the expected module structure
    if not ImportValidator.validate_module_structure(expected_module_path):
        raise AssertionError(
            f"Expected module path '{expected_module_path}' does not follow "
            f"cookiecutter template structure. Expected subdirectories: "
            f"{', '.join(sorted(ImportValidator.VALID_SUBDIRECTORIES))}"
        )
    
    if not ImportValidator.validate_import(obj, expected_module_path):
        actual_module = getattr(obj, '__module__', 'unknown')
        obj_name = obj.__name__ if hasattr(obj, '__name__') else str(obj)
        
        # Provide specific guidance for legacy imports
        if actual_module.startswith('odor_plume_nav.'):
            suggestion = actual_module.replace('odor_plume_nav.', '{{cookiecutter.project_slug}}.', 1)
            # Update legacy structure mapping
            if 'domain' in actual_module:
                suggestion = suggestion.replace('.domain.', '.core.' if 'navigator' in actual_module else '.config.')
            elif 'services' in actual_module:
                suggestion = suggestion.replace('.services.', '.core.' if 'navigation' in actual_module else '.api.')
            elif 'adapters' in actual_module:
                suggestion = suggestion.replace('.adapters.', '.data.')
            elif 'interfaces' in actual_module:
                suggestion = suggestion.replace('.interfaces.', '.utils.')
            
            raise AssertionError(
                f"Object {obj_name} was imported from legacy module '{actual_module}', "
                f"but should be imported from '{expected_module_path}'. "
                f"Consider updating from: {actual_module} → {suggestion}"
            )
        else:
            raise AssertionError(
                f"Object {obj_name} was imported from '{actual_module}', "
                f"but should be imported from '{expected_module_path}' "
                f"per cookiecutter template structure."
            )


def assert_all_imported_from(objects: Dict[str, Any], 
                            expected_module_path: str) -> None:
    """
    Assert that all objects were imported from the expected cookiecutter template module.
    
    Args:
        objects: Dictionary mapping names to objects
        expected_module_path: The expected module path in {{cookiecutter.project_slug}} namespace
        
    Raises:
        AssertionError: If any object was not imported from the expected module
    """
    success, failures = ImportValidator.validate_imports(objects, expected_module_path)
    if not success:
        raise AssertionError(
            f"Import validation failed for cookiecutter template structure:\n" + 
            "\n".join(f"- {failure}" for failure in failures) +
            f"\n\nExpected module: {expected_module_path}\n" +
            f"Ensure imports follow the {{cookiecutter.project_slug}} package organization."
        )


# Enhanced validation utilities for cookiecutter template structure
def suggest_migration_path(legacy_module: str) -> str:
    """
    Suggest the appropriate migration path from legacy to cookiecutter template structure.
    
    Args:
        legacy_module: The legacy odor_plume_nav module path
        
    Returns:
        Suggested module path in the new structure
    """
    if not legacy_module.startswith('odor_plume_nav.'):
        return legacy_module
    
    # Basic namespace replacement
    suggested = legacy_module.replace('odor_plume_nav.', '{{cookiecutter.project_slug}}.', 1)
    
    # Structure-specific migrations based on the mapping table
    migrations = {
        '.domain.models': '.config.schemas',
        '.domain.navigator_protocol': '.core.navigator',
        '.domain.sensor_strategies': '.core.sensors',
        '.services.navigation.': '.core.controllers',
        '.services.config_loader': '.config.schemas',  # Now handled by Hydra
        '.services.simulation_runner': '.api.navigation',
        '.services.navigator_factory': '.api.navigation',
        '.services.video_plume_factory': '.data.video_plume',
        '.adapters.video_plume_opencv': '.data.video_plume',
        '.interfaces.visualization.': '.utils.visualization',
        '.interfaces.api': '.api.navigation',
        '.utils.logging_setup': '.utils.logging',
    }
    
    # Apply specific migrations
    for old_pattern, new_pattern in migrations.items():
        if old_pattern in suggested:
            suggested = suggested.replace(old_pattern, new_pattern)
            break
    
    return suggested


def validate_new_package_components() -> Tuple[bool, List[str]]:
    """
    Validate that new components introduced in the cookiecutter template are properly structured.
    
    Returns:
        Tuple of (success, list of validation messages)
    """
    validation_messages = []
    
    # Check for new components that should exist
    new_components = [
        '{{cookiecutter.project_slug}}.cli.main',
        '{{cookiecutter.project_slug}}.utils.seed_manager',
        '{{cookiecutter.project_slug}}.db.session',
        '{{cookiecutter.project_slug}}.config.schemas',
    ]
    
    for component in new_components:
        try:
            # Try to find the component in sys.modules
            if component in sys.modules:
                validation_messages.append(f"✓ New component {component} properly loaded")
            else:
                validation_messages.append(f"! New component {component} not found in loaded modules")
        except Exception as e:
            validation_messages.append(f"✗ Error validating component {component}: {e}")
    
    # All messages are informational, return success
    return True, validation_messages


# Create lookup helper functions for the mappings
def get_target_module(class_name: str) -> Optional[str]:
    """
    Get the target module path for a class name in the cookiecutter template structure.
    
    Args:
        class_name: Name of the class
        
    Returns:
        Target module path or None if not found
    """
    return TARGET_IMPORT_MAPPING.get(class_name)


def scan_cookiecutter_modules() -> Dict[str, List[str]]:
    """
    Scan and categorize all loaded modules in the cookiecutter template structure.
    
    Returns:
        Dictionary mapping subdirectories to lists of loaded modules
    """
    module_scan = {subdir: [] for subdir in ImportValidator.VALID_SUBDIRECTORIES}
    module_scan['legacy'] = []  # Track legacy modules
    module_scan['other'] = []   # Track other modules
    
    for module_name in sys.modules:
        if module_name.startswith('{{cookiecutter.project_slug}}.'):
            parts = module_name.split('.')
            if len(parts) >= 2:
                subdirectory = parts[1]
                if subdirectory in ImportValidator.VALID_SUBDIRECTORIES:
                    module_scan[subdirectory].append(module_name)
                else:
                    module_scan['other'].append(module_name)
        elif module_name.startswith('odor_plume_nav.'):
            module_scan['legacy'].append(module_name)
    
    return module_scan


def generate_migration_report() -> str:
    """
    Generate a comprehensive migration report showing current module state and suggestions.
    
    Returns:
        Formatted migration report string
    """
    module_scan = scan_cookiecutter_modules()
    _, new_component_messages = validate_new_package_components()
    
    report_lines = [
        "🔄 Cookiecutter Template Migration Report",
        "=" * 50,
        "",
        "📊 Module Distribution:",
    ]
    
    for subdir in sorted(ImportValidator.VALID_SUBDIRECTORIES):
        modules = module_scan[subdir]
        count = len(modules)
        status = "✓" if count > 0 else "⚠"
        report_lines.append(f"  {status} {subdir}/: {count} modules")
        if count > 0 and len(modules) <= 5:  # Show modules if reasonable number
            for module in modules:
                report_lines.append(f"    - {module}")
    
    # Legacy modules
    legacy_count = len(module_scan['legacy'])
    if legacy_count > 0:
        report_lines.extend([
            "",
            f"⚠ Legacy Modules Found: {legacy_count}",
            "  The following modules need migration:"
        ])
        for legacy_module in module_scan['legacy']:
            suggestion = suggest_migration_path(legacy_module)
            report_lines.append(f"    {legacy_module} → {suggestion}")
    
    # New component validation
    report_lines.extend([
        "",
        "🆕 New Component Validation:",
    ])
    for message in new_component_messages:
        report_lines.append(f"  {message}")
    
    # Other modules
    if module_scan['other']:
        report_lines.extend([
            "",
            "❓ Unexpected Module Structure:",
        ])
        for module in module_scan['other']:
            report_lines.append(f"  - {module}")
    
    return "\n".join(report_lines)


def check_import_compliance() -> Tuple[bool, List[str]]:
    """
    Check compliance with the expected cookiecutter template import structure.
    
    Scans the new module namespace ({{cookiecutter.project_slug}}) and validates 
    imports from the reorganized package structure including core/, data/, config/, 
    api/, utils/, cli/, and db/ subdirectories.
    
    Returns:
        Tuple of (success, list of violations)
    """
    violations = []
    
    # Get all loaded modules - scan both legacy and new namespaces
    for module_name, module in list(sys.modules.items()):
        # Check new cookiecutter template namespace
        if module_name.startswith('{{cookiecutter.project_slug}}.'):
            # Validate module structure compliance
            if not ImportValidator.validate_module_structure(module_name):
                violations.append(
                    f"Module {module_name} does not follow cookiecutter template structure. "
                    f"Expected subdirectories: {', '.join(sorted(ImportValidator.VALID_SUBDIRECTORIES))}"
                )
                continue
        # Check for legacy odor_plume_nav usage (should be migrated)
        elif module_name.startswith('odor_plume_nav.'):
            violations.append(
                f"Legacy module {module_name} found. Should be migrated to "
                f"{{cookiecutter.project_slug}} namespace per refactoring requirements."
            )
            continue
        else:
            # Skip non-package modules
            continue
            
        # Check imports in the module
        for name, obj in inspect.getmembers(module):
            # Using walrus operator to simplify assignment and conditional
            if expected_module := get_target_module(name):
                # Skip if this is the actual implementation module
                if module_name == expected_module:
                    continue
                    
                # Check if the object came from the expected module
                if hasattr(obj, '__module__') and obj.__module__ != expected_module:
                    violations.append(
                        f"In module {module_name}: {name} imported from "
                        f"{obj.__module__}, should be from {expected_module}. "
                        f"Update import to follow cookiecutter template structure."
                    )
    
    return not violations, violations


def enforce_import_compliance() -> None:
    """
    Enforce compliance with the expected cookiecutter template import structure.
    
    Validates imports against the new {{cookiecutter.project_slug}} package organization
    and provides clear guidance for updating legacy import paths.
    
    Raises:
        AssertionError: If any imports violate the expected structure
    """
    success, violations = check_import_compliance()
    if not success:
        error_message = (
            "Import structure violations detected in cookiecutter template migration:\n\n" +
            "\n".join(f"- {violation}" for violation in violations) +
            "\n\nPlease update imports to follow the new package structure:\n" +
            "  Core logic: {{cookiecutter.project_slug}}.core.*\n" +
            "  Data processing: {{cookiecutter.project_slug}}.data.*\n" +
            "  Configuration: {{cookiecutter.project_slug}}.config.*\n" +
            "  Public API: {{cookiecutter.project_slug}}.api.*\n" +
            "  Utilities: {{cookiecutter.project_slug}}.utils.*\n" +
            "  CLI interfaces: {{cookiecutter.project_slug}}.cli.*\n" +
            "  Database: {{cookiecutter.project_slug}}.db.*"
        )
        raise AssertionError(error_message)