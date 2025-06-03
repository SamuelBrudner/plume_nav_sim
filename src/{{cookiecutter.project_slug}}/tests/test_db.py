"""
Database Session Management Testing Module

This comprehensive testing module validates SQLAlchemy session lifecycle management,
connection handling, transaction management, and integration patterns for the Odor Plume
Navigation system's database infrastructure. The module uses in-memory database fixtures
to test database functionality without external dependencies while ensuring proper session
management, security validation, and performance compliance.

Key Testing Areas:
- SQLAlchemy session creation and configuration validation
- Connection lifecycle management and resource cleanup  
- Transaction handling and rollback scenarios
- Connection pooling and performance characteristics
- Database session security including SQL injection prevention
- Session isolation and state management for testing scenarios
- Database session performance meeting <100ms requirement
- Integration with Hydra configuration system and environment management
- Async support and future extensibility patterns
- Error handling and recovery mechanisms for database failures

Testing Strategy:
- In-memory SQLite databases for isolated, deterministic testing
- Comprehensive mock fixtures for external dependencies
- Performance benchmarking against specified SLA requirements
- Security validation through systematic SQL injection testing
- Configuration integration testing with Hydra composition
- Cross-platform compatibility verification

Authors: Cookiecutter Template Testing Framework
License: MIT
Version: 2.0.0
"""

import os
import time
import asyncio
import warnings
from contextlib import contextmanager
from typing import Dict, Any, Optional, Generator
from unittest.mock import Mock, patch, MagicMock
from urllib.parse import urlparse

import pytest
import numpy as np
from sqlalchemy import text, create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, TimeoutError as SQLTimeoutError
from sqlalchemy.pool import StaticPool, QueuePool

# Import the module under test
from src.{{cookiecutter.project_slug}}.db.session import (
    SessionManager,
    DatabaseBackend,
    DatabaseConfig,
    DatabaseConnectionInfo,
    get_session_manager,
    get_session,
    get_async_session,
    is_database_enabled,
    test_database_connection,
    test_async_database_connection,
    cleanup_database,
    PersistenceHooks,
    SQLALCHEMY_AVAILABLE,
    HYDRA_AVAILABLE,
    DOTENV_AVAILABLE
)


# Test database schema for testing
Base = declarative_base()


class TestTrajectoryRecord(Base):
    """Test table for trajectory data persistence testing."""
    __tablename__ = 'test_trajectories'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, nullable=False)
    x_position = Column(Float, nullable=False)
    y_position = Column(Float, nullable=False)
    timestamp = Column(Float, nullable=False)
    experiment_name = Column(String(255), nullable=True)


class TestExperimentMetadata(Base):
    """Test table for experiment metadata persistence testing."""
    __tablename__ = 'test_experiments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    parameters = Column(String(1000), nullable=True)
    start_time = Column(DateTime, nullable=False)
    status = Column(String(50), nullable=False)


@pytest.fixture
def in_memory_database_url():
    """Provides in-memory SQLite database URL for testing."""
    return "sqlite:///:memory:"


@pytest.fixture
def test_config_dict(in_memory_database_url):
    """Provides test configuration dictionary for database sessions."""
    return {
        'url': in_memory_database_url,
        'pool_size': 1,
        'max_overflow': 0,
        'pool_timeout': 5,
        'pool_recycle': -1,
        'echo': False,
        'enabled': True,
        'autocommit': False,
        'autoflush': True,
        'expire_on_commit': False
    }


@pytest.fixture
def mock_database_config(test_config_dict):
    """Provides mock DatabaseConfig protocol implementation."""
    class MockDatabaseConfig:
        def __init__(self, config_dict: Dict[str, Any]):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    return MockDatabaseConfig(test_config_dict)


@pytest.fixture
def mock_hydra_config(test_config_dict):
    """Provides mock Hydra DictConfig for configuration testing."""
    if not HYDRA_AVAILABLE:
        pytest.skip("Hydra not available for configuration testing")
    
    from omegaconf import DictConfig, OmegaConf
    return OmegaConf.create({
        'database': test_config_dict,
        'environment': 'testing'
    })


@pytest.fixture
def isolated_environment():
    """Provides isolated environment for environment variable testing."""
    original_env = os.environ.copy()
    
    # Clear database-related environment variables
    env_vars_to_clear = [
        'DATABASE_URL', 'DB_POOL_SIZE', 'DB_MAX_OVERFLOW', 
        'DB_ECHO', 'DB_ENABLED', 'DB_PASSWORD'
    ]
    
    for var in env_vars_to_clear:
        os.environ.pop(var, None)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def session_manager_cleanup():
    """Ensures session manager cleanup after tests."""
    yield
    cleanup_database()


@pytest.fixture
def temp_env_file(tmp_path):
    """Creates temporary .env file for environment variable testing."""
    env_file = tmp_path / ".env"
    env_content = """
DATABASE_URL=sqlite:///:memory:
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_ECHO=false
DB_ENABLED=true
"""
    env_file.write_text(env_content.strip())
    return str(env_file)


class TestDatabaseBackend:
    """Test suite for DatabaseBackend utility class."""
    
    def test_detect_sqlite_backend(self):
        """Test SQLite backend detection from various URL formats."""
        sqlite_urls = [
            "sqlite:///test.db",
            "sqlite:///:memory:",
            "sqlite:///path/to/database.sqlite"
        ]
        
        for url in sqlite_urls:
            backend = DatabaseBackend.detect_backend(url)
            assert backend == DatabaseBackend.SQLITE
    
    def test_detect_postgresql_backend(self):
        """Test PostgreSQL backend detection from connection URLs."""
        postgres_urls = [
            "postgresql://user:pass@localhost:5432/db",
            "postgresql+psycopg2://user:pass@host/db",
            "postgresql+asyncpg://user:pass@host/db"
        ]
        
        for url in postgres_urls:
            backend = DatabaseBackend.detect_backend(url)
            assert backend == DatabaseBackend.POSTGRESQL
    
    def test_detect_mysql_backend(self):
        """Test MySQL backend detection from connection URLs."""
        mysql_urls = [
            "mysql://user:pass@localhost:3306/db",
            "mysql+pymysql://user:pass@host/db"
        ]
        
        for url in mysql_urls:
            backend = DatabaseBackend.detect_backend(url)
            assert backend == DatabaseBackend.MYSQL
    
    def test_detect_memory_backend(self):
        """Test in-memory backend detection."""
        memory_urls = [
            "memory://test",
            "sqlite:///:memory:"
        ]
        
        for url in memory_urls:
            backend = DatabaseBackend.detect_backend(url)
            assert backend in [DatabaseBackend.MEMORY, DatabaseBackend.SQLITE]
    
    def test_detect_unknown_backend_defaults_to_sqlite(self):
        """Test that unknown backends default to SQLite with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            backend = DatabaseBackend.detect_backend("unknown://test")
            assert backend == DatabaseBackend.SQLITE
            assert len(w) == 1
            assert "Unknown database backend" in str(w[0].message)
    
    def test_get_sqlite_backend_defaults(self):
        """Test SQLite backend default configuration parameters."""
        defaults = DatabaseBackend.get_backend_defaults(DatabaseBackend.SQLITE)
        
        assert defaults['pool_size'] == 1
        assert defaults['max_overflow'] == 0
        assert defaults['poolclass'] == StaticPool
        assert 'check_same_thread' in defaults['connect_args']
        assert defaults['connect_args']['check_same_thread'] is False
    
    def test_get_postgresql_backend_defaults(self):
        """Test PostgreSQL backend default configuration parameters."""
        defaults = DatabaseBackend.get_backend_defaults(DatabaseBackend.POSTGRESQL)
        
        assert defaults['pool_size'] == 10
        assert defaults['max_overflow'] == 20
        assert defaults['poolclass'] == QueuePool
        assert 'connect_timeout' in defaults['connect_args']
        assert defaults['connect_args']['connect_timeout'] == 10
    
    def test_get_mysql_backend_defaults(self):
        """Test MySQL backend default configuration parameters."""
        defaults = DatabaseBackend.get_backend_defaults(DatabaseBackend.MYSQL)
        
        assert defaults['pool_size'] == 10
        assert defaults['max_overflow'] == 20
        assert defaults['poolclass'] == QueuePool
    
    def test_get_memory_backend_defaults(self):
        """Test in-memory backend default configuration parameters."""
        defaults = DatabaseBackend.get_backend_defaults(DatabaseBackend.MEMORY)
        
        assert defaults['pool_size'] == 1
        assert defaults['max_overflow'] == 0
        assert defaults['poolclass'] == StaticPool


class TestSessionManager:
    """Test suite for SessionManager class functionality."""
    
    def test_initialization_without_configuration(self, isolated_environment):
        """Test SessionManager initialization without database configuration."""
        session_manager = SessionManager()
        
        assert not session_manager.enabled
        assert session_manager.backend is None
        assert session_manager._engine is None
    
    def test_initialization_with_database_url(self, in_memory_database_url):
        """Test SessionManager initialization with database URL."""
        session_manager = SessionManager(database_url=in_memory_database_url)
        
        assert session_manager.enabled
        assert session_manager.backend == DatabaseBackend.SQLITE
        assert session_manager._engine is not None
    
    def test_initialization_with_config_dict(self, test_config_dict):
        """Test SessionManager initialization with configuration dictionary."""
        session_manager = SessionManager(config=test_config_dict)
        
        assert session_manager.enabled
        assert session_manager.backend == DatabaseBackend.SQLITE
        assert session_manager._engine is not None
    
    def test_initialization_with_config_object(self, mock_database_config):
        """Test SessionManager initialization with configuration object."""
        session_manager = SessionManager(config=mock_database_config)
        
        assert session_manager.enabled
        assert session_manager.backend == DatabaseBackend.SQLITE
        assert session_manager._engine is not None
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_initialization_with_hydra_config(self, mock_hydra_config):
        """Test SessionManager initialization with Hydra DictConfig."""
        session_manager = SessionManager.from_config(mock_hydra_config.database)
        
        assert session_manager.enabled
        assert session_manager.backend == DatabaseBackend.SQLITE
    
    def test_session_context_manager(self, test_config_dict):
        """Test context manager functionality for database sessions."""
        session_manager = SessionManager(config=test_config_dict)
        
        # Create test schema
        Base.metadata.create_all(session_manager._engine)
        
        with session_manager.session() as session:
            assert session is not None
            
            # Test basic database operations
            test_record = TestTrajectoryRecord(
                agent_id=1,
                x_position=10.5,
                y_position=20.3,
                timestamp=time.time(),
                experiment_name="test_experiment"
            )
            
            session.add(test_record)
            # Session should auto-commit on successful exit
        
        # Verify data was committed
        with session_manager.session() as session:
            records = session.query(TestTrajectoryRecord).all()
            assert len(records) == 1
            assert records[0].agent_id == 1
    
    def test_session_rollback_on_exception(self, test_config_dict):
        """Test automatic rollback on exception within session context."""
        session_manager = SessionManager(config=test_config_dict)
        Base.metadata.create_all(session_manager._engine)
        
        try:
            with session_manager.session() as session:
                test_record = TestTrajectoryRecord(
                    agent_id=2,
                    x_position=15.0,
                    y_position=25.0,
                    timestamp=time.time(),
                    experiment_name="rollback_test"
                )
                session.add(test_record)
                
                # Force an exception
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception
        
        # Verify data was rolled back
        with session_manager.session() as session:
            records = session.query(TestTrajectoryRecord).all()
            assert len(records) == 0
    
    @pytest.mark.asyncio
    async def test_async_session_context_manager(self, test_config_dict):
        """Test async context manager functionality for database sessions."""
        # Skip if async engine creation would fail
        if not test_config_dict['url'].startswith(('postgresql+asyncpg', 'mysql+aiomysql')):
            pytest.skip("Async support requires asyncpg or aiomysql drivers")
        
        session_manager = SessionManager(config=test_config_dict)
        
        async with session_manager.async_session() as session:
            if session is not None:
                # Test async database operations
                result = await session.execute(text("SELECT 1"))
                assert result is not None
    
    def test_connection_testing(self, test_config_dict):
        """Test database connection validation functionality."""
        session_manager = SessionManager(config=test_config_dict)
        
        # Test successful connection
        assert session_manager.test_connection() is True
    
    def test_connection_testing_failure(self):
        """Test connection testing with invalid database configuration."""
        invalid_config = {
            'url': 'postgresql://invalid:invalid@nonexistent:9999/invalid',
            'enabled': True
        }
        
        session_manager = SessionManager(config=invalid_config)
        
        # Should handle connection failure gracefully
        assert session_manager.test_connection() is False
    
    @pytest.mark.asyncio
    async def test_async_connection_testing(self, test_config_dict):
        """Test async database connection validation."""
        session_manager = SessionManager(config=test_config_dict)
        
        # Most in-memory SQLite won't support async, so expect False
        result = await session_manager.test_async_connection()
        assert isinstance(result, bool)
    
    def test_session_manager_cleanup(self, test_config_dict):
        """Test proper resource cleanup on session manager close."""
        session_manager = SessionManager(config=test_config_dict)
        
        assert session_manager.enabled
        assert session_manager._engine is not None
        
        session_manager.close()
        
        assert not session_manager.enabled
        assert session_manager._engine is None
        assert session_manager._sessionmaker is None
    
    def test_factory_method_from_config(self, test_config_dict):
        """Test SessionManager factory method with configuration."""
        session_manager = SessionManager.from_config(test_config_dict)
        
        assert session_manager.enabled
        assert session_manager.backend == DatabaseBackend.SQLITE
    
    def test_factory_method_from_url(self, in_memory_database_url):
        """Test SessionManager factory method with database URL."""
        session_manager = SessionManager.from_url(in_memory_database_url)
        
        assert session_manager.enabled
        assert session_manager.backend == DatabaseBackend.SQLITE
    
    def test_disabled_session_returns_none(self):
        """Test that disabled session manager returns None from context managers."""
        session_manager = SessionManager()  # No configuration
        
        with session_manager.session() as session:
            assert session is None
    
    @pytest.mark.asyncio
    async def test_disabled_async_session_returns_none(self):
        """Test that disabled session manager returns None from async context managers."""
        session_manager = SessionManager()  # No configuration
        
        async with session_manager.async_session() as session:
            assert session is None


class TestPerformanceRequirements:
    """Test suite for database performance SLA compliance."""
    
    def test_session_establishment_performance(self, test_config_dict):
        """Test database session establishment completes within 100ms per Section 6.6.3.3."""
        # Measure session manager initialization time
        start_time = time.perf_counter()
        session_manager = SessionManager(config=test_config_dict)
        initialization_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        
        # Session establishment should complete within 100ms
        assert initialization_time < 100.0, f"Session establishment took {initialization_time:.2f}ms, exceeds 100ms requirement"
        
        # Measure session creation time
        start_time = time.perf_counter()
        with session_manager.session() as session:
            session_creation_time = (time.perf_counter() - start_time) * 1000
            assert session is not None
        
        # Session creation should also be fast
        assert session_creation_time < 50.0, f"Session creation took {session_creation_time:.2f}ms"
    
    def test_connection_test_performance(self, test_config_dict):
        """Test connection testing performance within reasonable bounds."""
        session_manager = SessionManager(config=test_config_dict)
        
        start_time = time.perf_counter()
        result = session_manager.test_connection()
        test_time = (time.perf_counter() - start_time) * 1000
        
        assert result is True
        assert test_time < 50.0, f"Connection test took {test_time:.2f}ms"
    
    def test_multiple_session_performance(self, test_config_dict):
        """Test performance with multiple concurrent sessions."""
        session_manager = SessionManager(config=test_config_dict)
        Base.metadata.create_all(session_manager._engine)
        
        start_time = time.perf_counter()
        
        # Create multiple sessions sequentially
        for i in range(10):
            with session_manager.session() as session:
                test_record = TestTrajectoryRecord(
                    agent_id=i,
                    x_position=float(i),
                    y_position=float(i * 2),
                    timestamp=time.time(),
                    experiment_name=f"perf_test_{i}"
                )
                session.add(test_record)
        
        total_time = (time.perf_counter() - start_time) * 1000
        avg_time_per_session = total_time / 10
        
        # Average session time should be reasonable
        assert avg_time_per_session < 20.0, f"Average session time {avg_time_per_session:.2f}ms too high"


class TestSecurityValidation:
    """Test suite for database security including SQL injection prevention per Section 6.6.7.3."""
    
    def test_sql_injection_prevention_in_orm_queries(self, test_config_dict):
        """Test that ORM queries prevent SQL injection attacks."""
        session_manager = SessionManager(config=test_config_dict)
        Base.metadata.create_all(session_manager._engine)
        
        # Insert test data
        with session_manager.session() as session:
            test_record = TestTrajectoryRecord(
                agent_id=1,
                x_position=10.0,
                y_position=20.0,
                timestamp=time.time(),
                experiment_name="security_test"
            )
            session.add(test_record)
        
        # Attempt SQL injection through ORM query parameters
        malicious_inputs = [
            "'; DROP TABLE test_trajectories; --",
            "1 OR 1=1",
            "1; DELETE FROM test_trajectories; --",
            "1 UNION SELECT * FROM test_trajectories",
            "'; INSERT INTO test_trajectories VALUES (999, 0, 0, 0, 'hacked'); --"
        ]
        
        with session_manager.session() as session:
            for malicious_input in malicious_inputs:
                # ORM queries with bound parameters should be safe
                try:
                    # Test filtering with malicious input
                    results = session.query(TestTrajectoryRecord).filter(
                        TestTrajectoryRecord.experiment_name == malicious_input
                    ).all()
                    
                    # Should return empty results, not cause injection
                    assert len(results) == 0
                    
                except Exception as e:
                    # Any exception should be a legitimate database error, not injection
                    assert "syntax error" not in str(e).lower()
            
            # Verify original data still exists (wasn't deleted by injection)
            all_records = session.query(TestTrajectoryRecord).all()
            assert len(all_records) == 1
            assert all_records[0].experiment_name == "security_test"
    
    def test_parameterized_query_usage(self, test_config_dict):
        """Test that all database interactions use parameterized queries."""
        session_manager = SessionManager(config=test_config_dict)
        Base.metadata.create_all(session_manager._engine)
        
        with session_manager.session() as session:
            # Test that text() queries with parameters are safe
            malicious_param = "'; DROP TABLE test_trajectories; --"
            
            # This should be safe due to parameter binding
            result = session.execute(
                text("SELECT COUNT(*) as count FROM test_trajectories WHERE experiment_name = :name"),
                {"name": malicious_param}
            )
            
            count = result.scalar()
            assert count == 0  # No records with that exact name
            
            # Verify table still exists
            result = session.execute(text("SELECT COUNT(*) FROM test_trajectories"))
            assert result.scalar() == 0  # Table exists and is empty
    
    def test_connection_string_security(self):
        """Test secure handling of database connection strings."""
        # Test that sensitive information is not exposed in logs or errors
        sensitive_config = {
            'url': 'postgresql://user:secret_password@localhost:5432/testdb',
            'enabled': True
        }
        
        session_manager = SessionManager(config=sensitive_config)
        
        # Connection will fail, but password shouldn't be exposed
        try:
            session_manager.test_connection()
        except Exception as e:
            error_str = str(e)
            assert 'secret_password' not in error_str, "Password exposed in error message"
    
    def test_configuration_validation(self):
        """Test that configuration validation prevents unsafe parameters."""
        # Test invalid configuration parameters
        invalid_configs = [
            {'url': '', 'enabled': True},  # Empty URL
            {'url': 'invalid://not-a-database', 'enabled': True},  # Invalid scheme
            {'pool_size': -1, 'enabled': True},  # Negative pool size
            {'pool_timeout': -1, 'enabled': True},  # Negative timeout
        ]
        
        for invalid_config in invalid_configs:
            session_manager = SessionManager(config=invalid_config)
            # Should handle invalid configuration gracefully
            assert session_manager.test_connection() is False


class TestConfigurationIntegration:
    """Test suite for integration with Hydra configuration system."""
    
    def test_environment_variable_integration(self, isolated_environment):
        """Test integration with environment variables through python-dotenv."""
        # Set environment variables
        os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
        os.environ['DB_POOL_SIZE'] = '5'
        os.environ['DB_ECHO'] = 'true'
        
        session_manager = SessionManager(auto_configure=True)
        
        assert session_manager.enabled
        assert session_manager.backend == DatabaseBackend.SQLITE
    
    @pytest.mark.skipif(not DOTENV_AVAILABLE, reason="python-dotenv not available")
    def test_dotenv_file_loading(self, temp_env_file, isolated_environment, monkeypatch):
        """Test loading environment variables from .env file."""
        # Change to directory containing .env file
        env_dir = os.path.dirname(temp_env_file)
        monkeypatch.chdir(env_dir)
        
        session_manager = SessionManager(auto_configure=True)
        
        # Should load configuration from .env file
        assert session_manager.enabled
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_configuration_composition(self, mock_hydra_config):
        """Test Hydra configuration composition and override scenarios."""
        from omegaconf import OmegaConf
        
        # Test configuration override
        overrides = OmegaConf.create({
            'database': {
                'pool_size': 15,
                'echo': True
            }
        })
        
        merged_config = OmegaConf.merge(mock_hydra_config, overrides)
        session_manager = SessionManager.from_config(merged_config.database)
        
        assert session_manager.enabled
        # Verify overrides were applied (can't directly check SQLAlchemy config, but initialization should succeed)
    
    def test_configuration_precedence(self, isolated_environment):
        """Test configuration parameter precedence hierarchy."""
        # Set environment variable
        os.environ['DATABASE_URL'] = 'sqlite:///env.db'
        
        # Explicit URL should take precedence
        explicit_url = 'sqlite:///:memory:'
        session_manager = SessionManager(database_url=explicit_url)
        
        assert session_manager.enabled
        # Should use explicit URL, not environment variable
        assert ':memory:' in session_manager._get_database_url()
    
    def test_configuration_schema_validation(self):
        """Test configuration schema validation with invalid parameters."""
        # Test configuration with invalid types
        invalid_config = {
            'url': 123,  # Should be string
            'pool_size': 'invalid',  # Should be integer
            'enabled': 'yes'  # Should be boolean
        }
        
        # SessionManager should handle invalid configuration gracefully
        session_manager = SessionManager(config=invalid_config)
        assert not session_manager.enabled
    
    def test_multi_environment_configuration(self, isolated_environment):
        """Test environment-specific database configurations."""
        environments = {
            'development': 'sqlite:///dev.db',
            'testing': 'sqlite:///:memory:',
            'production': 'postgresql://prod:pass@prod-db:5432/prod'
        }
        
        for env_name, db_url in environments.items():
            os.environ.clear()
            os.environ['ENVIRONMENT'] = env_name
            os.environ['DATABASE_URL'] = db_url
            
            session_manager = SessionManager(auto_configure=True)
            
            if env_name == 'testing':
                assert session_manager.enabled
                assert session_manager.backend == DatabaseBackend.SQLITE
            elif env_name == 'production':
                # Would fail to connect but should detect PostgreSQL backend
                assert session_manager.backend == DatabaseBackend.POSTGRESQL


class TestGlobalSessionFunctions:
    """Test suite for global session management functions."""
    
    def test_get_session_manager_singleton(self, test_config_dict, session_manager_cleanup):
        """Test global session manager singleton behavior."""
        # First call creates instance
        manager1 = get_session_manager(config=test_config_dict)
        assert manager1.enabled
        
        # Second call returns same instance
        manager2 = get_session_manager()
        assert manager1 is manager2
        
        # Force recreation
        manager3 = get_session_manager(config=test_config_dict, force_recreate=True)
        assert manager3 is not manager1
        assert manager3.enabled
    
    def test_get_session_global_function(self, test_config_dict, session_manager_cleanup):
        """Test global get_session context manager function."""
        # Initialize global session manager
        get_session_manager(config=test_config_dict)
        
        with get_session() as session:
            assert session is not None
            
            # Test database operation
            result = session.execute(text("SELECT 1 as test"))
            assert result.scalar() == 1
    
    @pytest.mark.asyncio
    async def test_get_async_session_global_function(self, test_config_dict, session_manager_cleanup):
        """Test global get_async_session context manager function."""
        get_session_manager(config=test_config_dict)
        
        async with get_async_session() as session:
            # For SQLite, async session may be None
            if session is not None:
                result = await session.execute(text("SELECT 1 as test"))
                assert result.scalar() == 1
    
    def test_is_database_enabled_function(self, test_config_dict, session_manager_cleanup):
        """Test global database enabled status function."""
        # Initially disabled
        assert not is_database_enabled()
        
        # Enable through configuration
        get_session_manager(config=test_config_dict)
        assert is_database_enabled()
    
    def test_test_database_connection_function(self, test_config_dict, session_manager_cleanup):
        """Test global database connection testing function."""
        # Should fail when not configured
        assert not test_database_connection()
        
        # Should succeed when configured
        get_session_manager(config=test_config_dict)
        assert test_database_connection()
    
    @pytest.mark.asyncio
    async def test_test_async_database_connection_function(self, test_config_dict, session_manager_cleanup):
        """Test global async database connection testing function."""
        get_session_manager(config=test_config_dict)
        result = await test_async_database_connection()
        assert isinstance(result, bool)
    
    def test_cleanup_database_function(self, test_config_dict):
        """Test global database cleanup function."""
        # Initialize session manager
        manager = get_session_manager(config=test_config_dict)
        assert manager.enabled
        
        # Cleanup should reset global state
        cleanup_database()
        assert not is_database_enabled()


class TestPersistenceHooks:
    """Test suite for optional persistence hooks functionality."""
    
    def test_save_trajectory_data_when_enabled(self, test_config_dict, session_manager_cleanup):
        """Test trajectory data persistence when database enabled."""
        get_session_manager(config=test_config_dict)
        
        trajectory_data = {
            'agent_id': 1,
            'positions': np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            'timestamps': np.array([0.0, 0.1, 0.2]),
            'experiment_name': 'test_trajectory'
        }
        
        # Should return True when database enabled
        result = PersistenceHooks.save_trajectory_data(trajectory_data)
        assert result is True
    
    def test_save_trajectory_data_when_disabled(self, session_manager_cleanup):
        """Test trajectory data persistence when database disabled."""
        trajectory_data = {
            'agent_id': 1,
            'positions': np.array([[0.0, 0.0]]),
            'timestamps': np.array([0.0])
        }
        
        # Should return False when database disabled
        result = PersistenceHooks.save_trajectory_data(trajectory_data)
        assert result is False
    
    def test_save_experiment_metadata_when_enabled(self, test_config_dict, session_manager_cleanup):
        """Test experiment metadata persistence when database enabled."""
        get_session_manager(config=test_config_dict)
        
        metadata = {
            'experiment_name': 'test_experiment',
            'parameters': {
                'num_agents': 5,
                'duration': 100.0,
                'video_file': 'test_plume.mp4'
            },
            'start_time': time.time(),
            'status': 'completed'
        }
        
        # Should return True when database enabled
        result = PersistenceHooks.save_experiment_metadata(metadata)
        assert result is True
    
    def test_save_experiment_metadata_when_disabled(self, session_manager_cleanup):
        """Test experiment metadata persistence when database disabled."""
        metadata = {
            'experiment_name': 'test_experiment',
            'status': 'completed'
        }
        
        # Should return False when database disabled
        result = PersistenceHooks.save_experiment_metadata(metadata)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_async_save_trajectory_data(self, test_config_dict, session_manager_cleanup):
        """Test async trajectory data persistence."""
        get_session_manager(config=test_config_dict)
        
        trajectory_data = {
            'agent_id': 2,
            'positions': np.array([[5.0, 5.0]]),
            'timestamps': np.array([1.0])
        }
        
        # Should return True when database enabled (even if async not fully supported)
        result = await PersistenceHooks.async_save_trajectory_data(trajectory_data)
        assert result is True
    
    def test_persistence_hooks_with_existing_session(self, test_config_dict, session_manager_cleanup):
        """Test persistence hooks with existing database session."""
        session_manager = get_session_manager(config=test_config_dict)
        
        with session_manager.session() as session:
            trajectory_data = {'agent_id': 3, 'test': 'data'}
            
            # Should work with provided session
            result = PersistenceHooks.save_trajectory_data(trajectory_data, session)
            assert result is True
            
            metadata = {'experiment': 'test_with_session'}
            result = PersistenceHooks.save_experiment_metadata(metadata, session)
            assert result is True
    
    def test_persistence_hooks_error_handling(self, test_config_dict, session_manager_cleanup):
        """Test persistence hooks error handling with invalid data."""
        get_session_manager(config=test_config_dict)
        
        # Test with None data (should handle gracefully)
        result = PersistenceHooks.save_trajectory_data(None)
        assert result is True  # Should not fail, just log debug message
        
        result = PersistenceHooks.save_experiment_metadata(None)
        assert result is True  # Should not fail, just log debug message


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling scenarios."""
    
    def test_sqlalchemy_not_available_handling(self):
        """Test graceful handling when SQLAlchemy is not available."""
        # This test simulates the case where SQLAlchemy import fails
        with patch('src.{{cookiecutter.project_slug}}.db.session.SQLALCHEMY_AVAILABLE', False):
            session_manager = SessionManager()
            assert not session_manager.enabled
    
    def test_hydra_not_available_handling(self):
        """Test graceful handling when Hydra is not available."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.HYDRA_AVAILABLE', False):
            # Should still work with dict config
            config = {'url': 'sqlite:///:memory:', 'enabled': True}
            session_manager = SessionManager(config=config)
            assert session_manager.enabled
    
    def test_dotenv_not_available_handling(self):
        """Test graceful handling when python-dotenv is not available."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.DOTENV_AVAILABLE', False):
            session_manager = SessionManager(auto_configure=True)
            # Should work without dotenv
            assert isinstance(session_manager, SessionManager)
    
    def test_invalid_database_url_handling(self):
        """Test handling of invalid database URLs."""
        invalid_urls = [
            '',  # Empty string
            'not-a-url',  # Invalid format
            'unknown://test',  # Unknown scheme
            'postgresql://invalid-host:99999/db'  # Unreachable host
        ]
        
        for invalid_url in invalid_urls:
            config = {'url': invalid_url, 'enabled': True}
            session_manager = SessionManager(config=config)
            
            # Should handle gracefully
            if session_manager.enabled:
                # Connection test should fail gracefully
                assert session_manager.test_connection() is False
    
    def test_connection_pool_exhaustion_handling(self):
        """Test handling of connection pool exhaustion."""
        config = {
            'url': 'sqlite:///:memory:',
            'pool_size': 1,
            'max_overflow': 0,
            'pool_timeout': 1,  # Short timeout
            'enabled': True
        }
        
        session_manager = SessionManager(config=config)
        
        # This test is conceptual - SQLite in-memory doesn't truly exhaust pools
        # but verifies the configuration is accepted
        assert session_manager.enabled
        assert session_manager.test_connection()
    
    def test_transaction_deadlock_handling(self, test_config_dict):
        """Test transaction deadlock detection and recovery."""
        session_manager = SessionManager(config=test_config_dict)
        Base.metadata.create_all(session_manager._engine)
        
        # SQLite doesn't have true deadlocks, but test transaction rollback
        try:
            with session_manager.session() as session:
                test_record = TestTrajectoryRecord(
                    agent_id=999,
                    x_position=0.0,
                    y_position=0.0,
                    timestamp=time.time(),
                    experiment_name="deadlock_test"
                )
                session.add(test_record)
                
                # Force a rollback scenario
                raise SQLAlchemyError("Simulated database error")
        except SQLAlchemyError:
            pass  # Expected
        
        # Should recover gracefully
        assert session_manager.test_connection()
    
    def test_memory_usage_with_large_sessions(self, test_config_dict):
        """Test memory usage patterns with large session operations."""
        session_manager = SessionManager(config=test_config_dict)
        Base.metadata.create_all(session_manager._engine)
        
        # Test memory efficiency with bulk operations
        with session_manager.session() as session:
            # Create many records to test memory handling
            records = []
            for i in range(1000):
                record = TestTrajectoryRecord(
                    agent_id=i % 10,
                    x_position=float(i),
                    y_position=float(i * 2),
                    timestamp=time.time() + i,
                    experiment_name=f"bulk_test_{i}"
                )
                records.append(record)
            
            # Add all records
            session.add_all(records)
        
        # Verify all records were persisted
        with session_manager.session() as session:
            count = session.query(TestTrajectoryRecord).count()
            assert count == 1000
    
    def test_concurrent_session_access(self, test_config_dict):
        """Test concurrent access to session manager."""
        session_manager = SessionManager(config=test_config_dict)
        Base.metadata.create_all(session_manager._engine)
        
        # Simulate concurrent access (sequential due to SQLite limitations)
        results = []
        for i in range(5):
            with session_manager.session() as session:
                test_record = TestTrajectoryRecord(
                    agent_id=i,
                    x_position=float(i * 10),
                    y_position=float(i * 20),
                    timestamp=time.time() + i,
                    experiment_name=f"concurrent_test_{i}"
                )
                session.add(test_record)
                results.append(i)
        
        # All operations should succeed
        assert len(results) == 5
        
        # Verify all data was persisted
        with session_manager.session() as session:
            count = session.query(TestTrajectoryRecord).count()
            assert count == 5


class TestCrossModuleIntegration:
    """Test suite for integration with other system modules."""
    
    def test_integration_with_numpy_arrays(self, test_config_dict, session_manager_cleanup):
        """Test integration with NumPy arrays for trajectory storage."""
        get_session_manager(config=test_config_dict)
        
        # Simulate realistic trajectory data
        trajectory_positions = np.random.rand(100, 2) * 50.0  # 100 timesteps, 2D positions
        trajectory_timestamps = np.linspace(0, 10.0, 100)  # 10-second simulation
        
        trajectory_data = {
            'agent_id': 1,
            'positions': trajectory_positions,
            'timestamps': trajectory_timestamps,
            'experiment_name': 'numpy_integration_test'
        }
        
        # Should handle NumPy arrays correctly
        result = PersistenceHooks.save_trajectory_data(trajectory_data)
        assert result is True
    
    def test_integration_with_configuration_schemas(self, test_config_dict):
        """Test integration with Pydantic configuration schemas."""
        from typing import Optional
        from pydantic import BaseModel, validator
        
        class TestDatabaseConfig(BaseModel):
            """Test configuration schema for database parameters."""
            url: str
            pool_size: Optional[int] = 10
            enabled: bool = True
            
            @validator('pool_size')
            def validate_pool_size(cls, v):
                if v is not None and v <= 0:
                    raise ValueError('pool_size must be positive')
                return v
        
        # Test with valid configuration
        valid_config = TestDatabaseConfig(**test_config_dict)
        session_manager = SessionManager(config=valid_config)
        assert session_manager.enabled
        
        # Test with invalid configuration
        try:
            invalid_config_dict = test_config_dict.copy()
            invalid_config_dict['pool_size'] = -1
            TestDatabaseConfig(**invalid_config_dict)
            assert False, "Should have raised validation error"
        except ValueError as e:
            assert "pool_size must be positive" in str(e)
    
    def test_integration_with_environment_management(self, isolated_environment):
        """Test integration with environment variable management."""
        # Test multi-environment configuration
        test_environments = [
            ('development', 'sqlite:///dev.db'),
            ('testing', 'sqlite:///:memory:'),
            ('production', 'postgresql://prod:pass@localhost:5432/prod')
        ]
        
        for env_name, db_url in test_environments:
            os.environ.clear()
            os.environ['ENVIRONMENT'] = env_name
            os.environ['DATABASE_URL'] = db_url
            
            session_manager = SessionManager(auto_configure=True)
            
            if env_name in ['development', 'testing']:
                # These should initialize successfully
                assert session_manager.backend == DatabaseBackend.SQLITE
            else:
                # Production would fail to connect but should detect backend
                assert session_manager.backend == DatabaseBackend.POSTGRESQL
    
    def test_integration_with_logging_system(self, test_config_dict, caplog):
        """Test integration with logging system for debugging and monitoring."""
        import logging
        
        # Set logging level to capture debug messages
        caplog.set_level(logging.DEBUG)
        
        session_manager = SessionManager(config=test_config_dict)
        
        # Should log initialization
        assert session_manager.enabled
        
        # Test connection should log activity
        result = session_manager.test_connection()
        assert result is True
        
        # Cleanup should log activity
        session_manager.close()
        
        # Verify logging captured relevant events
        log_messages = [record.message for record in caplog.records]
        assert any('Database session manager' in msg for msg in log_messages)


# Performance benchmarking utilities
def measure_execution_time(func, *args, **kwargs):
    """Utility function to measure execution time in milliseconds."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    return result, execution_time


# Test data generation utilities
def generate_test_trajectory_data(num_points: int = 100, num_agents: int = 1) -> Dict[str, Any]:
    """Generate synthetic trajectory data for testing."""
    return {
        'agent_ids': np.arange(num_agents),
        'positions': np.random.rand(num_points, num_agents, 2) * 100.0,
        'orientations': np.random.rand(num_points, num_agents) * 2 * np.pi,
        'timestamps': np.linspace(0, 10.0, num_points),
        'experiment_name': f'test_experiment_{num_agents}_agents_{num_points}_points'
    }


def generate_test_experiment_metadata() -> Dict[str, Any]:
    """Generate synthetic experiment metadata for testing."""
    return {
        'experiment_name': f'test_experiment_{int(time.time())}',
        'parameters': {
            'num_agents': np.random.randint(1, 20),
            'duration': np.random.uniform(10.0, 100.0),
            'video_file': f'test_plume_{np.random.randint(1, 10)}.mp4',
            'algorithm': 'gradient_ascent',
            'learning_rate': np.random.uniform(0.01, 0.1)
        },
        'start_time': time.time(),
        'status': 'completed'
    }


# Module-level test configuration
pytestmark = pytest.mark.filterwarnings("ignore:.*deprecated.*:DeprecationWarning")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])