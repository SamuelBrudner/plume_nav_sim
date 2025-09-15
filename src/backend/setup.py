"""
Legacy setuptools compatibility setup script providing backwards-compatible package installation 
and build support for the plume_nav_sim reinforcement learning environment. This file serves as 
a bridge between modern pyproject.toml configuration and older build systems, tools, and workflows 
that require traditional setup.py based package installation and distribution management.

This setup.py file enables:
- Legacy build system compatibility for tools that do not yet support PEP 517/518 pyproject.toml
- Package metadata exposure through traditional setuptools interfaces
- Dependency management integration with runtime and development requirements
- Development workflow support including editable installs and optional dependencies
- Command-line utilities registration for package validation and benchmarking

The script dynamically reads package metadata from the primary package __init__.py file to ensure
version synchronization and leverages comprehensive dependency specifications for complete
development environment support.
"""

# External imports with version comments for dependency management and compatibility tracking
import setuptools  # >=61.0.0 - Legacy Python packaging library providing setup() function and build system compatibility
import pathlib  # >=3.10 - Modern path manipulation for reading README content and managing file system operations
import os  # >=3.10 - Operating system interface for environment variable access and cross-platform compatibility

# Internal imports for package metadata synchronization and configuration management
from plume_nav_sim import __version__, get_package_info

# Global path constants for consistent file location management across build operations
HERE = pathlib.Path(__file__).parent
PACKAGE_DIR = HERE / 'plume_nav_sim'
README_PATH = HERE / 'README.md'
REQUIREMENTS_PATH = HERE / 'requirements.txt'
DEV_REQUIREMENTS_PATH = HERE / 'requirements-dev.txt'

# Package metadata constants ensuring consistency with pyproject.toml and package configuration
PACKAGE_NAME = 'plume-nav-sim'
AUTHOR = 'plume_nav_sim Development Team'
AUTHOR_EMAIL = 'plume-nav-sim@example.com'
DESCRIPTION = 'Proof-of-life Gymnasium-compatible reinforcement learning environment for plume navigation research'
LICENSE = 'MIT'
URL = 'https://github.com/plume-nav-sim/plume_nav_sim'

# SEO and discovery keywords for PyPI categorization and package discoverability
KEYWORDS = [
    'reinforcement learning', 'gymnasium', 'plume navigation', 'chemical plume', 'simulation', 
    'robotics', 'autonomous navigation', 'olfactory navigation', 'scientific computing'
]

# PyPI classifiers for proper package categorization and compatibility specification
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Physics',
    'License :: OSI Approved :: MIT License',
    'Operating System :: POSIX :: Linux',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11', 
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
    'Framework :: Gymnasium'
]


def read_requirements(requirements_file: pathlib.Path) -> list:
    """
    Reads and parses requirements from specified requirements file, handling missing files 
    gracefully and extracting clean package specifications for setuptools dependency management.
    
    This function processes requirements.txt files to extract clean package specifications,
    filtering out comments and empty lines while preserving version constraints and optional
    dependency specifications for setuptools compatibility.
    
    Args:
        requirements_file (pathlib.Path): Path to requirements file for dependency parsing
        
    Returns:
        list: List of requirement strings suitable for setuptools install_requires specification
        
    Example:
        core_deps = read_requirements(HERE / 'requirements.txt')
        dev_deps = read_requirements(HERE / 'requirements-dev.txt')
    """
    # Check if requirements file exists using pathlib Path.exists() method
    if not requirements_file.exists():
        return []
    
    try:
        # Read requirements file content using UTF-8 encoding for cross-platform compatibility
        content = requirements_file.read_text(encoding='utf-8')
        
        # Split content into individual lines and strip whitespace from each line
        lines = [line.strip() for line in content.splitlines()]
        
        # Filter out empty lines and comment lines (starting with #) from requirements
        requirements = []
        for line in lines:
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Remove inline comments from requirement lines while preserving package specifications
            if '#' in line:
                line = line.split('#')[0].strip()
            
            # Add cleaned requirement to list if not empty after processing
            if line:
                requirements.append(line)
        
        # Return cleaned list of requirement strings for setuptools dependency specification
        return requirements
        
    except Exception as e:
        print(f"Warning: Failed to read requirements file {requirements_file}: {e}")
        return []


def read_long_description() -> str:
    """
    Reads README.md content for long_description in setup metadata, handling file existence 
    gracefully and providing fallback description for package distribution and PyPI display.
    
    This function attempts to read README.md content for use as the package long description
    in PyPI listings, with graceful fallback to the short description if the README file
    is not available or cannot be read.
    
    Returns:
        str: README content or fallback description for package long description display
        
    Example:
        long_desc = read_long_description()
        # Returns README.md content or fallback description string
    """
    try:
        # Check if README.md file exists in package directory using pathlib
        if README_PATH.exists():
            # Read README.md content with UTF-8 encoding for proper markdown handling
            return README_PATH.read_text(encoding='utf-8')
        else:
            # Return fallback short description if README file does not exist
            return DESCRIPTION
    
    except Exception as e:
        print(f"Warning: Failed to read README.md: {e}")
        # Handle encoding errors gracefully by returning fallback description
        return DESCRIPTION


def get_version_from_package() -> str:
    """
    Extracts version information from package __init__.py file dynamically, ensuring version 
    synchronization between setup.py and package metadata without importing the package during build.
    
    This function retrieves the version string from the already imported __version__ variable,
    ensuring consistency between setup.py metadata and the primary package configuration while
    avoiding circular import issues during the build process.
    
    Returns:
        str: Package version string extracted from __init__.py __version__ attribute
        
    Example:
        version = get_version_from_package()
        # Returns synchronized version string from package metadata
    """
    try:
        # Extract __version__ attribute from imported package module
        version_string = __version__
        
        # Validate version string follows semantic versioning format
        if not version_string or not isinstance(version_string, str):
            raise ValueError(f"Invalid version format: {version_string}")
        
        # Return version string for use in setuptools setup() configuration
        return version_string
        
    except Exception as e:
        print(f"Warning: Failed to get version from package: {e}")
        # Handle import errors by falling back to default version string
        return "0.0.1"


def get_package_data() -> dict:
    """
    Discovers and returns package data files including assets, configuration, and data files 
    that should be included in package distribution beyond Python source files.
    
    This function identifies non-Python files within the package directory structure that need
    to be included in the distribution, including rendering assets, configuration files, and
    example data for complete package functionality.
    
    Returns:
        dict: Dictionary mapping package names to lists of data file patterns for setuptools package_data
        
    Example:
        data_files = get_package_data()
        # Returns {'plume_nav_sim': ['assets/*.py', 'config/*.py', ...]}
    """
    try:
        # Define base package data patterns for assets, config, and data directories
        package_data_patterns = []
        
        # Include rendering assets from assets/ directory including colormaps and templates
        assets_dir = PACKAGE_DIR / 'assets'
        if assets_dir.exists():
            package_data_patterns.extend([
                'assets/*.py',
                'assets/*.json',
                'assets/*.txt'
            ])
        
        # Include configuration files from config/ directory for environment and rendering settings
        config_dir = PACKAGE_DIR / 'config'
        if config_dir.exists():
            package_data_patterns.extend([
                'config/*.py',
                'config/*.json',
                'config/*.yaml',
                'config/*.yml'
            ])
        
        # Include benchmark and example data from data/ directory for scientific reproducibility
        data_dir = PACKAGE_DIR / 'data'
        if data_dir.exists():
            package_data_patterns.extend([
                'data/*.py',
                'data/*.json',
                'data/*.csv',
                'data/*.txt'
            ])
        
        # Include logging configurations and formatters for development and debugging support
        logging_dir = PACKAGE_DIR / 'logging'
        if logging_dir.exists():
            package_data_patterns.extend([
                'logging/*.py',
                'logging/*.json',
                'logging/*.yaml',
                'logging/*.yml'
            ])
        
        # Return comprehensive package_data dictionary for setuptools inclusion
        return {
            'plume_nav_sim': package_data_patterns
        } if package_data_patterns else {}
        
    except Exception as e:
        print(f"Warning: Failed to discover package data: {e}")
        return {}


def setup_package():
    """
    Main setup function that configures and executes setuptools.setup() with comprehensive 
    package metadata, dependencies, and configuration for legacy build system compatibility.
    
    This function orchestrates the complete setuptools configuration by reading package metadata,
    processing dependencies, discovering data files, and executing the setup() call with all
    necessary parameters for proper package distribution and installation support.
    
    Returns:
        None: No return value - executes setuptools.setup() configuration
        
    Example:
        setup_package()  # Configures and executes complete setuptools setup
    """
    try:
        # Read package version using get_version_from_package() function
        version = get_version_from_package()
        print(f"Setting up {PACKAGE_NAME} version {version}")
        
        # Read long description from README.md using read_long_description() function
        long_description = read_long_description()
        
        # Read core dependencies from requirements.txt using read_requirements() function
        install_requires = read_requirements(REQUIREMENTS_PATH)
        if not install_requires:
            # Provide minimal core dependencies if requirements.txt is missing
            install_requires = [
                'gymnasium>=0.29.0',
                'numpy>=2.1.0', 
                'matplotlib>=3.9.0'
            ]
        
        # Read development dependencies from requirements-dev.txt for extras_require
        dev_requirements = read_requirements(DEV_REQUIREMENTS_PATH)
        if not dev_requirements:
            # Provide minimal development dependencies if file is missing
            dev_requirements = [
                'pytest>=8.0.0',
                'pytest-cov>=4.0.0',
                'black>=24.0.0',
                'flake8>=7.0.0'
            ]
        
        # Get package data files using get_package_data() function for complete distribution
        package_data = get_package_data()
        
        # Configure setuptools.setup() with comprehensive metadata including name, version, description
        setup_config = {
            'name': PACKAGE_NAME,
            'version': version,
            'description': DESCRIPTION,
            'long_description': long_description,
            'long_description_content_type': 'text/markdown',
            'author': AUTHOR,
            'author_email': AUTHOR_EMAIL,
            'url': URL,
            'license': LICENSE,
            'keywords': KEYWORDS,
            'classifiers': CLASSIFIERS,
            
            # Set package discovery using find_packages() with src/backend directory specification
            'packages': setuptools.find_packages(),
            'package_dir': {'': '.'},
            
            # Configure install_requires with core runtime dependencies
            'install_requires': install_requires,
            
            # Configure extras_require with optional dependency groups for dev, test, and other workflows
            'extras_require': {
                'dev': dev_requirements,
                'test': [
                    'pytest>=8.0.0',
                    'pytest-cov>=4.0.0',
                    'pytest-xdist>=3.0.0',
                    'pytest-benchmark>=4.0.0'
                ],
                'docs': [
                    'sphinx>=7.0.0',
                    'sphinx-rtd-theme>=2.0.0'
                ],
                'benchmark': [
                    'pytest-benchmark>=4.0.0',
                    'memory-profiler>=0.61.0',
                    'psutil>=5.9.0'
                ]
            },
            
            # Set entry_points for console scripts including validation and benchmarking utilities
            'entry_points': {
                'console_scripts': [
                    'plume-nav-validate=plume_nav_sim.scripts:validate_installation',
                    'plume-nav-benchmark=plume_nav_sim.scripts:run_benchmarks'
                ]
            },
            
            # Configure package_data for non-Python files inclusion in distribution
            'package_data': package_data,
            
            # Set python_requires for minimum Python version compatibility (>=3.10)
            'python_requires': '>=3.10',
            
            # Configure additional setuptools options
            'zip_safe': False,
            'include_package_data': True
        }
        
        # Execute setuptools.setup() with complete configuration dictionary
        setuptools.setup(**setup_config)
        
        print(f"Setup completed successfully for {PACKAGE_NAME} v{version}")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        raise


# Execute setup_package() when script is run directly for legacy setuptools compatibility
if __name__ == '__main__':
    setup_package()