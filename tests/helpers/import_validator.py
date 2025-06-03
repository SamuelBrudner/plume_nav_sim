"""
Import validation helper for tests.

This module provides utilities to ensure that imports in tests are coming from 
the correct modules in the refactored structure, helping enforce the target state
before the actual implementation has been fully migrated.
"""

import sys
import inspect
from pathlib import Path
from types import ModuleType
from typing import Dict, Any, List, Optional, Set, Tuple

# Import the target mapping
from tests.helpers.import_mapping import TARGET_IMPORT_MAPPING


class ImportValidator:
    """Validates that objects are imported from the expected modules."""
    
    @staticmethod
    def validate_import(obj: Any, expected_module_path: str) -> bool:
        """
        Check if an object was imported from the expected module.
        
        Args:
            obj: The object to check
            expected_module_path: The expected module path (e.g., 'odor_plume_nav.core.navigator')
            
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
        for name, obj in objects.items():
            if not ImportValidator.validate_import(obj, expected_module_path):
                actual_module = getattr(obj, '__module__', 'unknown')
                failures.append(f"{name} imported from {actual_module} not {expected_module_path}")
        
        return not failures, failures


def assert_imported_from(obj: Any, expected_module_path: str) -> None:
    """
    Assert that an object was imported from the expected module.
    
    Args:
        obj: The object to check
        expected_module_path: The expected module path
        
    Raises:
        AssertionError: If the object was not imported from the expected module
    """
    if not ImportValidator.validate_import(obj, expected_module_path):
        actual_module = getattr(obj, '__module__', 'unknown')
        raise AssertionError(
            f"Object {obj.__name__ if hasattr(obj, '__name__') else obj} "
            f"was imported from '{actual_module}', "
            f"but should be imported from '{expected_module_path}'"
        )


def assert_all_imported_from(objects: Dict[str, Any], 
                            expected_module_path: str) -> None:
    """
    Assert that all objects were imported from the expected module.
    
    Args:
        objects: Dictionary mapping names to objects
        expected_module_path: The expected module path
        
    Raises:
        AssertionError: If any object was not imported from the expected module
    """
    success, failures = ImportValidator.validate_imports(objects, expected_module_path)
    if not success:
        raise AssertionError(
            f"The following imports are incorrect:\n" + 
            "\n".join(f"- {failure}" for failure in failures)
        )


# Create lookup helper functions for the mappings
def get_target_module(class_name: str) -> Optional[str]:
    """
    Get the target module path for a class name.
    
    Args:
        class_name: Name of the class
        
    Returns:
        Target module path or None if not found
    """
    return TARGET_IMPORT_MAPPING.get(class_name)


def check_import_compliance() -> Tuple[bool, List[str]]:
    """
    Check compliance with the expected import structure.
    
    Returns:
        Tuple of (success, list of violations)
    """
    violations = []
    
    # Get all loaded modules
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith('odor_plume_nav.'):
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
                        f"{obj.__module__}, should be from {expected_module}"
                    )
    
    return not violations, violations


def enforce_import_compliance() -> None:
    """
    Enforce compliance with the expected {{cookiecutter.project_slug}} package structure.
    
    Validates the cookiecutter template organization including new subdirectories
    and provides migration guidance for legacy odor_plume_nav imports.
    
    Raises:
        AssertionError: If any imports violate the expected structure, with detailed
                       migration guidance for moving to cookiecutter template structure
    """
    success, violations = check_import_compliance()
    if not success:
        error_msg = (
            "{{cookiecutter.project_slug}} package structure violations detected:\n" + 
            "\n".join(f"- {violation}" for violation in violations) +
            "\n\nRefer to Section 0.2.1 of the technical specification for "
            "comprehensive mapping from legacy odor_plume_nav to cookiecutter template structure."
        )
        raise AssertionError(error_msg)


def scan_new_components() -> Dict[str, List[str]]:
    """
    Scan for new components introduced in the cookiecutter template structure.
    
    Returns:
        Dictionary mapping subdirectories to lists of available components
    """
    new_components = {
        'cli': ['main', 'run_simulation_cli', 'validate_config_cli'],
        'utils': ['set_global_seed', 'get_random_state', 'SeedManager'],
        'db': ['DatabaseSessionManager', 'get_session', 'create_engine'],
        'config': ['NavigatorConfig', 'ConfigValidationError'],
        'core': ['NavigatorProtocol', 'Navigator', 'SingleAgentController', 'MultiAgentController'],
        'data': ['VideoPlume'],
        'api': ['create_navigator', 'run_plume_simulation', 'create_navigator_from_config']
    }
    
    available_components = {}
    for subdirectory, components in new_components.items():
        available_components[subdirectory] = []
        for component in components:
            if get_target_module(component):
                available_components[subdirectory].append(component)
    
    return available_components


def validate_new_structure_integrity() -> Tuple[bool, List[str]]:
    """
    Validate the integrity of the new cookiecutter template structure.
    
    Checks that all expected new components (CLI, seed management, database sessions)
    are properly mapped and that the subdirectory organization is correct.
    
    Returns:
        Tuple of (success, list of integrity issues)
    """
    issues = []
    
    # Validate package structure mapping
    structure_valid, structure_violations = validate_package_structure()
    if not structure_valid:
        issues.extend(structure_violations)
    
    # Check for new component availability
    new_components = scan_new_components()
    required_components = {
        'cli': ['main'],
        'utils': ['set_global_seed', 'SeedManager'],
        'db': ['DatabaseSessionManager'],
    }
    
    for subdirectory, required in required_components.items():
        available = new_components.get(subdirectory, [])
        for component in required:
            if component not in available:
                issues.append(
                    f"Required new component '{component}' not available in "
                    f"{subdirectory}/ subdirectory"
                )
    
    return not issues, issues
