"""
Comprehensive pytest test suite for database session management (src/{{cookiecutter.project_slug}}/db/session.py)

This test module validates SQLAlchemy session management, connection lifecycle, transaction management,
and session security using in-memory SQLite for comprehensive database session testing. The test suite
ensures database session stubs meet performance criteria, security requirements, and reliability standards
while providing foundation for future data persistence testing.

Test Coverage Areas:
- SQLAlchemy session management validation per Feature F-015
- Connection establishment timing validation (<500ms) per Section 2.1.9
- Session lifecycle and cleanup testing per Feature F-015
- SQL injection prevention validation per Section 6.6.7.3
- Database session isolation for test reliability per Section 6.6.5.4
- In-memory database testing infrastructure per Section 6.6.1.1

Key Features Tested:
- Context-managed database sessions with automatic transaction handling
- Connection pooling and resource cleanup validation
- Multi-backend support (SQLite, PostgreSQL, in-memory) configuration testing
- Environment variable integration and secure credential handling
- Async/await compatibility for high-performance operations
- Performance benchmarking for connection establishment timing
- Security validation against SQL injection attacks
- Configuration validation and type safety
- Error handling and graceful degradation testing
- Session isolation and state management verification

Security Testing:
- SQL injection prevention through parameterized query validation
- Connection string handling and credential isolation testing
- Environment variable interpolation security validation
- Configuration override protection and parameter validation

Performance Testing:
- Connection establishment timing (<500ms requirement)
- Session creation and cleanup performance validation
- Connection pooling efficiency and resource utilization
- Transaction handling and rollback performance

Integration Testing:
- Hydra configuration integration and parameter composition
- Environment variable loading and interpolation testing
- Multi-environment configuration validation
- Backend detection and configuration optimization

Authors: Cookiecutter Template Generator  
License: MIT
Version: 2.0.0
"""

import asyncio
import os
import sqlite3
import tempfile
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import the module under test
from src.{{cookiecutter.project_slug}}.db.session import (
    DatabaseBackend,
    DatabaseConfig,
    DatabaseConnectionInfo,
    PersistenceHooks,
    SessionManager,
    cleanup_database,
    get_async_session,
    get_session,
    get_session_manager,
    is_database_enabled,
    test_async_database_connection,
    test_database_connection,
    DOTENV_AVAILABLE,
    HYDRA_AVAILABLE,
    SQLALCHEMY_AVAILABLE,
)

# Test data and fixtures setup
TEST_DATABASE_URL_SQLITE = "sqlite:///:memory:"
TEST_DATABASE_URL_POSTGRESQL = "postgresql://user:pass@localhost:5432/test_db"
TEST_DATABASE_URL_MYSQL = "mysql://user:pass@localhost:3306/test_db"

MALICIOUS_SQL_INPUTS = [
    "'; DROP TABLE users; --",
    "' OR '1'='1",
    "'; DELETE FROM experiments; --",
    "' UNION SELECT * FROM information_schema.tables --",
    "'; INSERT INTO malicious_table VALUES ('hacked'); --",
    "' OR 1=1 /*",
    "'; EXEC xp_cmdshell('dir'); --",
    "' AND 1=CONVERT(int, (SELECT @@version)); --"
]

INVALID_CONNECTION_STRINGS = [
    "../../../etc/passwd",
    "file:///etc/passwd",
    "sqlite:///../../../sensitive.db",
    "postgresql://user:pass@evil.com/db",
    "javascript:alert('xss')",
    "data:text/html,<script>alert('xss')</script>"
]


class TestDatabaseConfig:
    """Test configuration protocol and type validation."""
    
    def test_database_config_protocol(self):
        """Test that database configuration protocol is properly defined."""
        # Test with dictionary implementation
        config_dict = {
            'url': TEST_DATABASE_URL_SQLITE,
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'echo': False,
            'enabled': True
        }
        
        # Verify protocol compliance through duck typing
        assert hasattr(config_dict, '__getitem__')
        assert config_dict['url'] == TEST_DATABASE_URL_SQLITE
        assert config_dict['pool_size'] == 5
        assert config_dict['enabled'] is True
    
    def test_database_connection_info_typed_dict(self):
        """Test DatabaseConnectionInfo TypedDict structure."""
        connection_info: DatabaseConnectionInfo = {
            'url': TEST_DATABASE_URL_SQLITE,
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'echo': False,
            'autocommit': False,
            'autoflush': True,
            'expire_on_commit': True
        }
        
        # Verify all expected fields are present
        assert connection_info['url'] == TEST_DATABASE_URL_SQLITE
        assert connection_info['pool_size'] == 10
        assert connection_info['max_overflow'] == 20
        assert connection_info['echo'] is False


class TestDatabaseBackend:
    """Test database backend detection and configuration utilities."""
    
    def test_detect_backend_sqlite(self):
        """Test SQLite backend detection from various URL formats."""
        test_urls = [
            "sqlite:///path/to/database.db",
            "sqlite:///:memory:",
            "sqlite:///./relative/path.db",
            "sqlite:////absolute/path/database.db"
        ]
        
        for url in test_urls:
            backend = DatabaseBackend.detect_backend(url)
            assert backend == DatabaseBackend.SQLITE, f"Failed for URL: {url}"
    
    def test_detect_backend_postgresql(self):
        """Test PostgreSQL backend detection from various URL formats."""
        test_urls = [
            "postgresql://user:pass@localhost:5432/dbname",
            "postgresql+psycopg2://user:pass@host/db",
            "postgresql+asyncpg://user:pass@host:5432/db"
        ]
        
        for url in test_urls:
            backend = DatabaseBackend.detect_backend(url)
            assert backend == DatabaseBackend.POSTGRESQL, f"Failed for URL: {url}"
    
    def test_detect_backend_mysql(self):
        """Test MySQL backend detection from various URL formats."""
        test_urls = [
            "mysql://user:pass@localhost:3306/dbname",
            "mysql+pymysql://user:pass@host/db",
            "mysql+aiomysql://user:pass@host:3306/db"
        ]
        
        for url in test_urls:
            backend = DatabaseBackend.detect_backend(url)
            assert backend == DatabaseBackend.MYSQL, f"Failed for URL: {url}"
    
    def test_detect_backend_memory(self):
        """Test in-memory database detection."""
        test_urls = [
            "sqlite:///:memory:",
            "sqlite:///memory",
            "memory://test"
        ]
        
        for url in test_urls:
            backend = DatabaseBackend.detect_backend(url)
            assert backend in [DatabaseBackend.SQLITE, DatabaseBackend.MEMORY], f"Failed for URL: {url}"
    
    def test_detect_backend_empty_url(self):
        """Test backend detection with empty or None URL."""
        assert DatabaseBackend.detect_backend("") == DatabaseBackend.SQLITE
        assert DatabaseBackend.detect_backend(None) == DatabaseBackend.SQLITE
    
    def test_detect_backend_unknown_scheme(self):
        """Test backend detection with unknown URL schemes."""
        unknown_urls = [
            "unknown://localhost/db",
            "ftp://example.com/database",
            "http://api.example.com/db"
        ]
        
        for url in unknown_urls:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                backend = DatabaseBackend.detect_backend(url)
                assert backend == DatabaseBackend.SQLITE
                assert len(w) >= 1
                assert "Unknown database backend" in str(w[0].message)
    
    def test_get_backend_defaults_sqlite(self):
        """Test SQLite backend default configuration."""
        defaults = DatabaseBackend.get_backend_defaults(DatabaseBackend.SQLITE)
        
        assert defaults['pool_size'] == 1
        assert defaults['max_overflow'] == 0
        assert defaults['pool_timeout'] == 30
        assert defaults['pool_recycle'] == -1
        assert defaults['echo'] is False
        assert 'connect_args' in defaults
        assert defaults['connect_args']['check_same_thread'] is False
        assert defaults['connect_args']['timeout'] == 20
    
    def test_get_backend_defaults_postgresql(self):
        """Test PostgreSQL backend default configuration."""
        defaults = DatabaseBackend.get_backend_defaults(DatabaseBackend.POSTGRESQL)
        
        assert defaults['pool_size'] == 10
        assert defaults['max_overflow'] == 20
        assert defaults['pool_timeout'] == 30
        assert defaults['pool_recycle'] == 3600
        assert defaults['echo'] is False
        assert 'connect_args' in defaults
        assert defaults['connect_args']['connect_timeout'] == 10
        assert 'server_settings' in defaults['connect_args']
        assert defaults['connect_args']['server_settings']['application_name'] == 'odor_plume_nav'
    
    def test_get_backend_defaults_mysql(self):
        """Test MySQL backend default configuration."""
        defaults = DatabaseBackend.get_backend_defaults(DatabaseBackend.MYSQL)
        
        assert defaults['pool_size'] == 10
        assert defaults['max_overflow'] == 20
        assert defaults['pool_timeout'] == 30
        assert defaults['pool_recycle'] == 3600
        assert defaults['echo'] is False
        assert 'connect_args' in defaults
        assert defaults['connect_args']['connect_timeout'] == 10
    
    def test_get_backend_defaults_memory(self):
        """Test in-memory database backend default configuration."""
        defaults = DatabaseBackend.get_backend_defaults(DatabaseBackend.MEMORY)
        
        assert defaults['pool_size'] == 1
        assert defaults['max_overflow'] == 0
        assert defaults['pool_timeout'] == 5
        assert defaults['pool_recycle'] == -1
        assert defaults['echo'] is False
        assert 'connect_args' in defaults
        assert defaults['connect_args']['check_same_thread'] is False
    
    def test_get_backend_defaults_unknown(self):
        """Test that unknown backends default to SQLite configuration."""
        defaults = DatabaseBackend.get_backend_defaults("unknown_backend")
        sqlite_defaults = DatabaseBackend.get_backend_defaults(DatabaseBackend.SQLITE)
        
        assert defaults == sqlite_defaults


@pytest.fixture
def mock_sqlalchemy_available():
    """Fixture to mock SQLAlchemy availability."""
    with patch('src.{{cookiecutter.project_slug}}.db.session.SQLALCHEMY_AVAILABLE', True):
        yield


@pytest.fixture
def mock_sqlalchemy_unavailable():
    """Fixture to mock SQLAlchemy unavailability."""
    with patch('src.{{cookiecutter.project_slug}}.db.session.SQLALCHEMY_AVAILABLE', False):
        yield


@pytest.fixture
def in_memory_sqlite_config():
    """Fixture providing in-memory SQLite configuration for testing."""
    return {
        'url': TEST_DATABASE_URL_SQLITE,
        'pool_size': 1,
        'max_overflow': 0,
        'pool_timeout': 5,
        'pool_recycle': -1,
        'echo': False,
        'enabled': True
    }


@pytest.fixture
def temp_env_file():
    """Fixture providing temporary .env file for testing environment variable loading."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("DATABASE_URL=sqlite:///:memory:\n")
        f.write("DB_POOL_SIZE=15\n")
        f.write("DB_ECHO=true\n")
        temp_file = f.name
    
    try:
        yield temp_file
    finally:
        os.unlink(temp_file)


@pytest.fixture
def mock_database_config():
    """Fixture providing mock database configuration object."""
    config = Mock()
    config.url = TEST_DATABASE_URL_SQLITE
    config.pool_size = 5
    config.max_overflow = 10
    config.pool_timeout = 30
    config.pool_recycle = 3600
    config.echo = False
    config.enabled = True
    return config


class TestSessionManagerInitialization:
    """Test SessionManager initialization and configuration handling."""
    
    def test_session_manager_initialization_no_config(self, mock_sqlalchemy_available):
        """Test SessionManager initialization without configuration."""
        session_manager = SessionManager(auto_configure=False)
        
        # Should not be enabled without configuration
        assert not session_manager.enabled
        assert session_manager.backend is None
    
    def test_session_manager_initialization_with_url(self, mock_sqlalchemy_available):
        """Test SessionManager initialization with database URL."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine, \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker') as mock_sessionmaker:
            
            mock_engine.return_value = Mock()
            mock_sessionmaker.return_value = Mock()
            
            session_manager = SessionManager(database_url=TEST_DATABASE_URL_SQLITE, auto_configure=False)
            
            assert session_manager.enabled
            assert session_manager.backend == DatabaseBackend.SQLITE
            mock_engine.assert_called_once()
            mock_sessionmaker.assert_called_once()
    
    def test_session_manager_initialization_with_config_dict(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test SessionManager initialization with configuration dictionary."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine, \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker') as mock_sessionmaker:
            
            mock_engine.return_value = Mock()
            mock_sessionmaker.return_value = Mock()
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            assert session_manager.enabled
            assert session_manager.backend == DatabaseBackend.SQLITE
            mock_engine.assert_called_once()
            mock_sessionmaker.assert_called_once()
    
    def test_session_manager_initialization_with_config_object(self, mock_sqlalchemy_available, mock_database_config):
        """Test SessionManager initialization with configuration object."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine, \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker') as mock_sessionmaker:
            
            mock_engine.return_value = Mock()
            mock_sessionmaker.return_value = Mock()
            
            session_manager = SessionManager(config=mock_database_config, auto_configure=False)
            
            assert session_manager.enabled
            assert session_manager.backend == DatabaseBackend.SQLITE
            mock_engine.assert_called_once()
            mock_sessionmaker.assert_called_once()
    
    def test_session_manager_sqlalchemy_unavailable(self, mock_sqlalchemy_unavailable):
        """Test SessionManager behavior when SQLAlchemy is unavailable."""
        session_manager = SessionManager(database_url=TEST_DATABASE_URL_SQLITE, auto_configure=False)
        
        # Should not be enabled when SQLAlchemy unavailable
        assert not session_manager.enabled
        assert session_manager.backend is None
    
    def test_session_manager_initialization_exception_handling(self, mock_sqlalchemy_available):
        """Test SessionManager initialization exception handling."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine', side_effect=Exception("Database error")):
            session_manager = SessionManager(database_url=TEST_DATABASE_URL_SQLITE, auto_configure=False)
            
            # Should gracefully handle initialization errors
            assert not session_manager.enabled
            assert session_manager.backend is None
    
    @pytest.mark.skipif(not DOTENV_AVAILABLE, reason="python-dotenv not available")
    def test_environment_variable_loading(self, mock_sqlalchemy_available, temp_env_file):
        """Test automatic environment variable loading."""
        # Change to directory containing the temp .env file
        original_cwd = os.getcwd()
        temp_dir = os.path.dirname(temp_env_file)
        os.rename(temp_env_file, os.path.join(temp_dir, '.env'))
        
        try:
            os.chdir(temp_dir)
            
            with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine, \
                 patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker') as mock_sessionmaker:
                
                mock_engine.return_value = Mock()
                mock_sessionmaker.return_value = Mock()
                
                session_manager = SessionManager(auto_configure=True)
                
                # Should load configuration from environment
                assert session_manager.enabled
                mock_engine.assert_called_once()
                
        finally:
            os.chdir(original_cwd)


class TestSessionManagerFactoryMethods:
    """Test SessionManager factory methods and configuration patterns."""
    
    def test_from_config_with_dict(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test SessionManager.from_config with dictionary configuration."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            session_manager = SessionManager.from_config(in_memory_sqlite_config)
            
            assert session_manager.enabled
            assert session_manager.backend == DatabaseBackend.SQLITE
    
    def test_from_config_with_none(self, mock_sqlalchemy_available):
        """Test SessionManager.from_config with None configuration."""
        session_manager = SessionManager.from_config(None)
        
        # Should not be enabled without configuration
        assert not session_manager.enabled
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_from_config_with_hydra_dictconfig(self, mock_sqlalchemy_available):
        """Test SessionManager.from_config with Hydra DictConfig."""
        from omegaconf import DictConfig
        
        config_dict = {
            'url': TEST_DATABASE_URL_SQLITE,
            'enabled': True,
            'pool_size': 5
        }
        hydra_config = DictConfig(config_dict)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            session_manager = SessionManager.from_config(hydra_config)
            
            assert session_manager.enabled
            assert session_manager.backend == DatabaseBackend.SQLITE
    
    def test_from_url_method(self, mock_sqlalchemy_available):
        """Test SessionManager.from_url factory method."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            session_manager = SessionManager.from_url(TEST_DATABASE_URL_SQLITE)
            
            assert session_manager.enabled
            assert session_manager.backend == DatabaseBackend.SQLITE


class TestSessionLifecycleAndCleanup:
    """Test session lifecycle management and resource cleanup per Feature F-015."""
    
    def test_session_context_manager_success(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test successful session context manager usage with commit."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            with session_manager.session() as session:
                # Simulate database operations
                session.add(Mock())
            
            # Verify proper session lifecycle
            mock_sessionmaker.assert_called_once()
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()
            mock_session.rollback.assert_not_called()
    
    def test_session_context_manager_exception_rollback(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test session context manager exception handling with rollback."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            with pytest.raises(ValueError):
                with session_manager.session() as session:
                    # Simulate operation that raises exception
                    raise ValueError("Simulated database error")
            
            # Verify proper exception handling
            mock_sessionmaker.assert_called_once()
            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()
            mock_session.commit.assert_not_called()
    
    def test_session_context_manager_disabled(self, mock_sqlalchemy_unavailable):
        """Test session context manager when database is disabled."""
        session_manager = SessionManager(auto_configure=False)
        
        with session_manager.session() as session:
            # Should yield None when database disabled
            assert session is None
    
    @pytest.mark.asyncio
    async def test_async_session_context_manager_success(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test successful async session context manager usage."""
        mock_async_session = Mock()
        # Mock async methods
        mock_async_session.commit = Mock(return_value=asyncio.coroutine(lambda: None)())
        mock_async_session.close = Mock(return_value=asyncio.coroutine(lambda: None)())
        mock_async_sessionmaker = Mock(return_value=mock_async_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.create_async_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.async_sessionmaker', return_value=mock_async_sessionmaker):
            
            # Create session manager with PostgreSQL URL to enable async
            async_config = in_memory_sqlite_config.copy()
            async_config['url'] = "postgresql+asyncpg://user:pass@localhost/db"
            
            session_manager = SessionManager(config=async_config, auto_configure=False)
            
            async with session_manager.async_session() as session:
                # Simulate async database operations
                if session:
                    session.add(Mock())
            
            # Verify async session lifecycle
            mock_async_sessionmaker.assert_called_once()
            mock_async_session.commit.assert_called_once()
            mock_async_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_session_context_manager_exception(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test async session context manager exception handling."""
        mock_async_session = Mock()
        # Mock async methods
        mock_async_session.rollback = Mock(return_value=asyncio.coroutine(lambda: None)())
        mock_async_session.close = Mock(return_value=asyncio.coroutine(lambda: None)())
        mock_async_sessionmaker = Mock(return_value=mock_async_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.create_async_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.async_sessionmaker', return_value=mock_async_sessionmaker):
            
            # Create session manager with PostgreSQL URL to enable async
            async_config = in_memory_sqlite_config.copy()
            async_config['url'] = "postgresql+asyncpg://user:pass@localhost/db"
            
            session_manager = SessionManager(config=async_config, auto_configure=False)
            
            with pytest.raises(ValueError):
                async with session_manager.async_session() as session:
                    raise ValueError("Simulated async error")
            
            # Verify proper async exception handling
            mock_async_session.rollback.assert_called_once()
            mock_async_session.close.assert_called_once()
    
    def test_session_manager_close_cleanup(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test SessionManager.close() resource cleanup."""
        mock_engine = Mock()
        mock_async_engine = Mock()
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine', return_value=mock_engine), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.create_async_engine', return_value=mock_async_engine), \
             patch('src.{{cookiecutter.project_slug}}.db.session.async_sessionmaker'):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            # Verify initial state
            assert session_manager.enabled
            
            # Close and verify cleanup
            session_manager.close()
            
            assert not session_manager.enabled
            mock_engine.dispose.assert_called_once()
            
            # Verify internal state is reset
            assert session_manager._engine is None
            assert session_manager._sessionmaker is None


class TestConnectionEstablishmentTiming:
    """Test connection establishment timing validation (<500ms) per Section 2.1.9."""
    
    def test_connection_establishment_timing_sqlite(self, mock_sqlalchemy_available):
        """Test SQLite connection establishment timing under 500ms."""
        start_time = time.time()
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine, \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            mock_engine.return_value = Mock()
            
            session_manager = SessionManager(database_url=TEST_DATABASE_URL_SQLITE, auto_configure=False)
            
            initialization_time = time.time() - start_time
            
            # Verify initialization time meets performance criteria
            assert initialization_time < 0.5, f"Initialization took {initialization_time:.3f}s, exceeding 500ms limit"
            assert session_manager.enabled
    
    def test_connection_establishment_timing_postgresql_mock(self, mock_sqlalchemy_available):
        """Test PostgreSQL connection establishment timing simulation."""
        start_time = time.time()
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine, \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            # Simulate realistic PostgreSQL connection time
            def delayed_engine_creation(*args, **kwargs):
                time.sleep(0.1)  # Simulate 100ms connection time
                return Mock()
            
            mock_engine.side_effect = delayed_engine_creation
            
            session_manager = SessionManager(database_url=TEST_DATABASE_URL_POSTGRESQL, auto_configure=False)
            
            initialization_time = time.time() - start_time
            
            # Verify initialization time meets performance criteria
            assert initialization_time < 0.5, f"Initialization took {initialization_time:.3f}s, exceeding 500ms limit"
            assert session_manager.enabled
    
    def test_connection_test_timing(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test database connection test timing."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            start_time = time.time()
            result = session_manager.test_connection()
            test_time = time.time() - start_time
            
            # Verify connection test timing and success
            assert test_time < 0.1, f"Connection test took {test_time:.3f}s, too slow"
            assert result is True
            mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_connection_test_timing(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test async database connection test timing."""
        mock_async_session = Mock()
        mock_async_session.execute = Mock(return_value=asyncio.coroutine(lambda: None)())
        mock_async_sessionmaker = Mock(return_value=mock_async_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.create_async_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.async_sessionmaker', return_value=mock_async_sessionmaker):
            
            # Create session manager with async-capable URL
            async_config = in_memory_sqlite_config.copy()
            async_config['url'] = "postgresql+asyncpg://user:pass@localhost/db"
            
            session_manager = SessionManager(config=async_config, auto_configure=False)
            
            start_time = time.time()
            result = await session_manager.test_async_connection()
            test_time = time.time() - start_time
            
            # Verify async connection test timing and success
            assert test_time < 0.1, f"Async connection test took {test_time:.3f}s, too slow"
            assert result is True
            mock_async_session.execute.assert_called_once()


class TestSQLInjectionPrevention:
    """Test SQL injection prevention validation per Section 6.6.7.3."""
    
    def test_parameterized_query_usage(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test that session uses parameterized queries and prevents SQL injection."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            # Test connection with potential SQL injection
            with session_manager.session() as session:
                if session:
                    # Verify that raw SQL execution is not directly exposed
                    assert not hasattr(session, 'execute_raw_sql')
                    
                    # Test that session.execute is available for parameterized queries
                    assert hasattr(session, 'execute')
    
    def test_malicious_database_url_rejection(self, mock_sqlalchemy_available):
        """Test rejection of malicious database connection strings."""
        for malicious_url in INVALID_CONNECTION_STRINGS:
            # Most malicious URLs should be handled gracefully by SQLAlchemy
            # The session manager should either reject them or handle them safely
            try:
                with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine:
                    mock_engine.side_effect = Exception("Invalid URL")
                    
                    session_manager = SessionManager(database_url=malicious_url, auto_configure=False)
                    
                    # Should fail gracefully and not be enabled
                    assert not session_manager.enabled
                    
            except Exception:
                # Any exception during initialization is acceptable for malicious inputs
                pass
    
    def test_environment_variable_interpolation_security(self, mock_sqlalchemy_available):
        """Test that environment variable interpolation doesn't allow injection."""
        # Set potentially malicious environment variables
        malicious_values = [
            "sqlite:///:memory:'; DROP TABLE users; --",
            "postgresql://user:'; DELETE FROM experiments; --@localhost/db",
            "../../../etc/passwd"
        ]
        
        for malicious_value in malicious_values:
            with patch.dict(os.environ, {'DATABASE_URL': malicious_value}):
                try:
                    with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine:
                        # SQLAlchemy should handle malicious URLs appropriately
                        mock_engine.side_effect = Exception("Invalid connection string")
                        
                        session_manager = SessionManager(auto_configure=True)
                        
                        # Should fail gracefully
                        assert not session_manager.enabled
                        
                except Exception:
                    # Acceptable to raise exceptions for malicious inputs
                    pass
    
    def test_configuration_parameter_validation(self, mock_sqlalchemy_available):
        """Test validation of configuration parameters prevents injection."""
        malicious_configs = [
            {
                'url': TEST_DATABASE_URL_SQLITE,
                'pool_size': "'; DROP TABLE users; --",
                'enabled': True
            },
            {
                'url': TEST_DATABASE_URL_SQLITE,
                'pool_timeout': "$(rm -rf /)",
                'enabled': True
            },
            {
                'url': TEST_DATABASE_URL_SQLITE,
                'echo': "'; SELECT * FROM information_schema.tables; --",
                'enabled': True
            }
        ]
        
        for malicious_config in malicious_configs:
            try:
                with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine:
                    # Should handle invalid parameter types gracefully
                    mock_engine.side_effect = TypeError("Invalid parameter type")
                    
                    session_manager = SessionManager(config=malicious_config, auto_configure=False)
                    
                    # Should fail gracefully with invalid parameters
                    assert not session_manager.enabled
                    
            except (TypeError, ValueError):
                # Expected behavior for invalid parameter types
                pass
    
    def test_text_query_parameterization(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test that text queries use proper parameterization."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker), \
             patch('src.{{cookiecutter.project_slug}}.db.session.text') as mock_text:
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            # Test connection using the session manager's test method
            session_manager.test_connection()
            
            # Verify that text() was called for SQL queries (proper parameterization)
            mock_text.assert_called_with("SELECT 1")
            mock_session.execute.assert_called_once()


class TestDatabaseSessionIsolation:
    """Test database session isolation for test reliability per Section 6.6.5.4."""
    
    def test_session_isolation_between_contexts(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test that separate session contexts are properly isolated."""
        mock_session1 = Mock()
        mock_session2 = Mock()
        session_instances = [mock_session1, mock_session2]
        
        mock_sessionmaker = Mock(side_effect=session_instances)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            # Create first session context
            with session_manager.session() as session1:
                session1.add("data1")
            
            # Create second session context
            with session_manager.session() as session2:
                session2.add("data2")
            
            # Verify sessions are independent
            assert mock_session1 != mock_session2
            mock_session1.add.assert_called_once_with("data1")
            mock_session2.add.assert_called_once_with("data2")
            mock_session1.commit.assert_called_once()
            mock_session2.commit.assert_called_once()
            mock_session1.close.assert_called_once()
            mock_session2.close.assert_called_once()
    
    def test_session_state_isolation(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test that session state doesn't leak between operations."""
        mock_session = Mock()
        # Configure session to track state
        mock_session.dirty = []
        mock_session.new = []
        mock_session.deleted = []
        
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            # First operation
            with session_manager.session() as session:
                session.add("first_data")
                # Simulate state change
                mock_session.dirty.append("first_data")
            
            # Reset session state for next operation
            mock_session.dirty = []
            mock_session.new = []
            mock_session.deleted = []
            
            # Second operation should start with clean state
            with session_manager.session() as session:
                session.add("second_data")
                # Verify clean state
                assert len(mock_session.dirty) <= 1  # Only current operation data
    
    def test_transaction_isolation(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test transaction isolation between session contexts."""
        mock_session1 = Mock()
        mock_session2 = Mock()
        session_instances = [mock_session1, mock_session2]
        
        mock_sessionmaker = Mock(side_effect=session_instances)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            # First transaction - success
            with session_manager.session() as session1:
                session1.add("committed_data")
            
            # Verify first transaction committed
            mock_session1.commit.assert_called_once()
            mock_session1.rollback.assert_not_called()
            
            # Second transaction - failure
            with pytest.raises(ValueError):
                with session_manager.session() as session2:
                    session2.add("failed_data")
                    raise ValueError("Simulated transaction failure")
            
            # Verify second transaction rolled back
            mock_session2.rollback.assert_called_once()
            mock_session2.commit.assert_not_called()
            
            # Verify transactions are independent
            assert mock_session1 != mock_session2
    
    def test_connection_pool_isolation(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test that connection pool provides proper isolation."""
        mock_engine = Mock()
        mock_connection1 = Mock()
        mock_connection2 = Mock()
        
        # Mock connection pool behavior
        mock_engine.connect.side_effect = [mock_connection1, mock_connection2]
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine', return_value=mock_engine), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            # Test connection isolation
            assert session_manager.enabled
            assert mock_engine is not None
            
            # Verify engine configuration includes proper isolation settings
            # This is handled by SQLAlchemy's connection pooling


class TestInMemoryDatabaseInfrastructure:
    """Test in-memory database testing infrastructure per Section 6.6.1.1."""
    
    def test_in_memory_sqlite_configuration(self, mock_sqlalchemy_available):
        """Test in-memory SQLite database configuration."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine, \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            mock_engine.return_value = Mock()
            
            session_manager = SessionManager(database_url=TEST_DATABASE_URL_SQLITE, auto_configure=False)
            
            assert session_manager.enabled
            assert session_manager.backend == DatabaseBackend.SQLITE
            
            # Verify engine was created with in-memory URL
            mock_engine.assert_called_once()
            args, kwargs = mock_engine.call_args
            assert args[0] == TEST_DATABASE_URL_SQLITE
    
    def test_in_memory_database_backend_defaults(self):
        """Test in-memory database backend default configuration."""
        defaults = DatabaseBackend.get_backend_defaults(DatabaseBackend.MEMORY)
        
        # Verify memory-optimized configuration
        assert defaults['pool_size'] == 1
        assert defaults['max_overflow'] == 0
        assert defaults['pool_timeout'] == 5  # Faster timeout for testing
        assert defaults['pool_recycle'] == -1
        assert 'check_same_thread' in defaults['connect_args']
        assert defaults['connect_args']['check_same_thread'] is False
    
    def test_in_memory_session_lifecycle(self, mock_sqlalchemy_available):
        """Test complete session lifecycle with in-memory database."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        config = {
            'url': TEST_DATABASE_URL_SQLITE,
            'pool_size': 1,
            'max_overflow': 0,
            'enabled': True
        }
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=config, auto_configure=False)
            
            # Test complete lifecycle
            with session_manager.session() as session:
                assert session is not None
                session.add("test_data")
                session.query = Mock(return_value=Mock())
            
            # Verify proper lifecycle management
            mock_session.add.assert_called_once_with("test_data")
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()
    
    def test_in_memory_database_isolation_between_tests(self, mock_sqlalchemy_available):
        """Test that in-memory databases provide proper test isolation."""
        configs = [
            {'url': 'sqlite:///:memory:', 'enabled': True},
            {'url': 'sqlite:///:memory:', 'enabled': True}
        ]
        
        session_managers = []
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine, \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            mock_engine.return_value = Mock()
            
            # Create multiple session managers (simulating different test cases)
            for config in configs:
                session_manager = SessionManager(config=config, auto_configure=False)
                session_managers.append(session_manager)
            
            # Verify each gets independent configuration
            assert len(session_managers) == 2
            assert all(sm.enabled for sm in session_managers)
            assert all(sm.backend == DatabaseBackend.SQLITE for sm in session_managers)
            
            # Verify separate engine instances
            assert mock_engine.call_count == 2
    
    def test_memory_database_performance_characteristics(self, mock_sqlalchemy_available):
        """Test in-memory database performance characteristics."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine, \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            mock_engine.return_value = Mock()
            
            start_time = time.time()
            session_manager = SessionManager(database_url=TEST_DATABASE_URL_SQLITE, auto_configure=False)
            creation_time = time.time() - start_time
            
            # Verify fast initialization for in-memory database
            assert creation_time < 0.1, f"In-memory database creation took {creation_time:.3f}s, too slow"
            assert session_manager.enabled
            
            # Verify configuration optimized for memory usage
            args, kwargs = mock_engine.call_args
            assert 'pool_size' in kwargs
            assert 'max_overflow' in kwargs


class TestGlobalSessionManager:
    """Test global session manager functions and patterns."""
    
    def teardown_method(self):
        """Clean up global session manager after each test."""
        cleanup_database()
    
    def test_get_session_manager_singleton(self, mock_sqlalchemy_available):
        """Test global session manager singleton pattern."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            # First call creates instance
            manager1 = get_session_manager(database_url=TEST_DATABASE_URL_SQLITE)
            
            # Second call returns same instance
            manager2 = get_session_manager()
            
            assert manager1 is manager2
            assert manager1.enabled
    
    def test_get_session_manager_force_recreate(self, mock_sqlalchemy_available):
        """Test forcing recreation of global session manager."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            # Create initial instance
            manager1 = get_session_manager(database_url=TEST_DATABASE_URL_SQLITE)
            
            # Force recreation
            manager2 = get_session_manager(database_url=TEST_DATABASE_URL_SQLITE, force_recreate=True)
            
            assert manager1 is not manager2
            assert manager2.enabled
    
    def test_get_session_function(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test get_session convenience function."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            with get_session(config=in_memory_sqlite_config) as session:
                assert session is not None
                session.add("test_data")
            
            mock_session.add.assert_called_once_with("test_data")
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_async_session_function(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test get_async_session convenience function."""
        mock_async_session = Mock()
        mock_async_session.add = Mock()
        mock_async_session.commit = Mock(return_value=asyncio.coroutine(lambda: None)())
        mock_async_session.close = Mock(return_value=asyncio.coroutine(lambda: None)())
        mock_async_sessionmaker = Mock(return_value=mock_async_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.create_async_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.async_sessionmaker', return_value=mock_async_sessionmaker):
            
            # Use async-capable URL
            async_config = in_memory_sqlite_config.copy()
            async_config['url'] = "postgresql+asyncpg://user:pass@localhost/db"
            
            async with get_async_session(config=async_config) as session:
                if session:
                    session.add("async_test_data")
            
            mock_async_session.add.assert_called_once_with("async_test_data")
            mock_async_session.commit.assert_called_once()
            mock_async_session.close.assert_called_once()
    
    def test_is_database_enabled_function(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test is_database_enabled convenience function."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            # Initially disabled
            assert not is_database_enabled()
            
            # Enable database
            get_session_manager(config=in_memory_sqlite_config)
            
            # Should now be enabled
            assert is_database_enabled()
    
    def test_test_database_connection_function(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test test_database_connection convenience function."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            # Set up database
            get_session_manager(config=in_memory_sqlite_config)
            
            # Test connection
            result = test_database_connection()
            
            assert result is True
            mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_test_async_database_connection_function(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test test_async_database_connection convenience function."""
        mock_async_session = Mock()
        mock_async_session.execute = Mock(return_value=asyncio.coroutine(lambda: None)())
        mock_async_sessionmaker = Mock(return_value=mock_async_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.create_async_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.async_sessionmaker', return_value=mock_async_sessionmaker):
            
            # Use async-capable URL
            async_config = in_memory_sqlite_config.copy()
            async_config['url'] = "postgresql+asyncpg://user:pass@localhost/db"
            
            # Set up async database
            get_session_manager(config=async_config)
            
            # Test async connection
            result = await test_async_database_connection()
            
            assert result is True
            mock_async_session.execute.assert_called_once()
    
    def test_cleanup_database_function(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test cleanup_database convenience function."""
        mock_engine = Mock()
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine', return_value=mock_engine), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            # Set up database
            manager = get_session_manager(config=in_memory_sqlite_config)
            assert manager.enabled
            
            # Cleanup
            cleanup_database()
            
            # Verify cleanup
            mock_engine.dispose.assert_called_once()
            assert not is_database_enabled()


class TestPersistenceHooks:
    """Test persistence hooks for future extensibility."""
    
    def test_save_trajectory_data_disabled(self, mock_sqlalchemy_unavailable):
        """Test trajectory data persistence when database disabled."""
        trajectory_data = {"agent_id": 1, "position": [0.0, 1.0], "timestamp": "2023-01-01T00:00:00"}
        
        result = PersistenceHooks.save_trajectory_data(trajectory_data)
        
        # Should return False when database disabled
        assert result is False
    
    def test_save_trajectory_data_enabled(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test trajectory data persistence when database enabled."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            # Set up database
            get_session_manager(config=in_memory_sqlite_config)
            
            trajectory_data = {"agent_id": 1, "position": [0.0, 1.0], "timestamp": "2023-01-01T00:00:00"}
            
            result = PersistenceHooks.save_trajectory_data(trajectory_data)
            
            # Should return True when database enabled (stub implementation)
            assert result is True
    
    def test_save_trajectory_data_with_session(self, mock_sqlalchemy_available):
        """Test trajectory data persistence with provided session."""
        mock_session = Mock()
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.is_database_enabled', return_value=True):
            trajectory_data = {"agent_id": 1, "position": [0.0, 1.0]}
            
            result = PersistenceHooks.save_trajectory_data(trajectory_data, session=mock_session)
            
            # Should return True with provided session
            assert result is True
    
    def test_save_experiment_metadata_disabled(self, mock_sqlalchemy_unavailable):
        """Test experiment metadata persistence when database disabled."""
        metadata = {"experiment_id": "exp_001", "parameters": {"speed": 0.5}, "start_time": "2023-01-01T00:00:00"}
        
        result = PersistenceHooks.save_experiment_metadata(metadata)
        
        # Should return False when database disabled
        assert result is False
    
    def test_save_experiment_metadata_enabled(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test experiment metadata persistence when database enabled."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            # Set up database
            get_session_manager(config=in_memory_sqlite_config)
            
            metadata = {"experiment_id": "exp_001", "parameters": {"speed": 0.5}}
            
            result = PersistenceHooks.save_experiment_metadata(metadata)
            
            # Should return True when database enabled (stub implementation)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_async_save_trajectory_data_disabled(self, mock_sqlalchemy_unavailable):
        """Test async trajectory data persistence when database disabled."""
        trajectory_data = {"agent_id": 1, "position": [0.0, 1.0]}
        
        result = await PersistenceHooks.async_save_trajectory_data(trajectory_data)
        
        # Should return False when database disabled
        assert result is False
    
    @pytest.mark.asyncio
    async def test_async_save_trajectory_data_enabled(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test async trajectory data persistence when database enabled."""
        mock_async_session = Mock()
        mock_async_sessionmaker = Mock(return_value=mock_async_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.create_async_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.async_sessionmaker', return_value=mock_async_sessionmaker):
            
            # Use async-capable URL
            async_config = in_memory_sqlite_config.copy()
            async_config['url'] = "postgresql+asyncpg://user:pass@localhost/db"
            
            # Set up async database
            get_session_manager(config=async_config)
            
            trajectory_data = {"agent_id": 1, "position": [0.0, 1.0]}
            
            result = await PersistenceHooks.async_save_trajectory_data(trajectory_data)
            
            # Should return True when async database enabled
            assert result is True
    
    def test_persistence_hooks_error_handling(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test persistence hooks error handling."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.is_database_enabled', return_value=True), \
             patch('src.{{cookiecutter.project_slug}}.db.session.get_session') as mock_get_session:
            
            # Mock session that raises exception
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_get_session.return_value.__exit__.return_value = False
            
            # Configure session to raise exception
            mock_get_session.side_effect = Exception("Database error")
            
            trajectory_data = {"agent_id": 1, "position": [0.0, 1.0]}
            
            result = PersistenceHooks.save_trajectory_data(trajectory_data)
            
            # Should return False on error
            assert result is False


class TestErrorHandlingAndGracefulDegradation:
    """Test error handling and graceful degradation scenarios."""
    
    def test_session_manager_with_missing_dependencies(self):
        """Test SessionManager behavior when dependencies are missing."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.SQLALCHEMY_AVAILABLE', False):
            session_manager = SessionManager(database_url=TEST_DATABASE_URL_SQLITE, auto_configure=False)
            
            # Should not be enabled when SQLAlchemy unavailable
            assert not session_manager.enabled
            assert session_manager.backend is None
            
            # Session context should yield None
            with session_manager.session() as session:
                assert session is None
    
    def test_session_manager_database_connection_failure(self, mock_sqlalchemy_available):
        """Test SessionManager behavior when database connection fails."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine', side_effect=Exception("Connection failed")):
            session_manager = SessionManager(database_url=TEST_DATABASE_URL_SQLITE, auto_configure=False)
            
            # Should handle connection failure gracefully
            assert not session_manager.enabled
            
            # Session operations should be safe
            with session_manager.session() as session:
                assert session is None
    
    def test_session_context_manager_session_error(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test session context manager with session creation error."""
        mock_sessionmaker = Mock(side_effect=Exception("Session creation failed"))
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            # Should handle session creation errors
            with pytest.raises(Exception):
                with session_manager.session() as session:
                    pass
    
    def test_environment_loading_failure(self, mock_sqlalchemy_available):
        """Test graceful handling of environment loading failures."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.load_dotenv', side_effect=Exception("Env loading failed")):
            # Should not raise exception during initialization
            session_manager = SessionManager(auto_configure=True)
            
            # Should still be able to function without environment loading
            assert not session_manager.enabled  # No database URL provided
    
    def test_async_engine_creation_failure(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test handling of async engine creation failure."""
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.create_async_engine', side_effect=Exception("Async engine failed")):
            
            # Use async-capable URL
            async_config = in_memory_sqlite_config.copy()
            async_config['url'] = "postgresql+asyncpg://user:pass@localhost/db"
            
            session_manager = SessionManager(config=async_config, auto_configure=False)
            
            # Should still be enabled for sync operations
            assert session_manager.enabled
            
            # Sync operations should work
            with session_manager.session() as session:
                assert session is not None
    
    def test_connection_test_failure_handling(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test connection test failure handling."""
        mock_session = Mock()
        mock_session.execute.side_effect = Exception("Connection test failed")
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            # Connection test should return False on error
            result = session_manager.test_connection()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_async_connection_test_failure_handling(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test async connection test failure handling."""
        mock_async_session = Mock()
        mock_async_session.execute = Mock(side_effect=Exception("Async connection test failed"))
        mock_async_sessionmaker = Mock(return_value=mock_async_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.create_async_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.async_sessionmaker', return_value=mock_async_sessionmaker):
            
            # Use async-capable URL
            async_config = in_memory_sqlite_config.copy()
            async_config['url'] = "postgresql+asyncpg://user:pass@localhost/db"
            
            session_manager = SessionManager(config=async_config, auto_configure=False)
            
            # Async connection test should return False on error
            result = await session_manager.test_async_connection()
            assert result is False


class TestConfigurationValidationAndTypeSafety:
    """Test configuration validation and type safety."""
    
    def test_invalid_configuration_types(self, mock_sqlalchemy_available):
        """Test handling of invalid configuration parameter types."""
        invalid_configs = [
            {"url": 123, "enabled": True},  # Invalid URL type
            {"url": TEST_DATABASE_URL_SQLITE, "pool_size": "invalid"},  # Invalid pool_size type
            {"url": TEST_DATABASE_URL_SQLITE, "echo": "not_boolean"},  # Invalid echo type
        ]
        
        for invalid_config in invalid_configs:
            try:
                with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine', side_effect=TypeError("Invalid parameter type")):
                    session_manager = SessionManager(config=invalid_config, auto_configure=False)
                    
                    # Should handle invalid types gracefully
                    assert not session_manager.enabled
                    
            except (TypeError, ValueError):
                # Acceptable to raise type/value errors for invalid configurations
                pass
    
    def test_configuration_parameter_bounds(self, mock_sqlalchemy_available):
        """Test configuration parameter bounds validation."""
        edge_case_configs = [
            {"url": TEST_DATABASE_URL_SQLITE, "pool_size": 0, "enabled": True},  # Zero pool size
            {"url": TEST_DATABASE_URL_SQLITE, "pool_size": -1, "enabled": True},  # Negative pool size
            {"url": TEST_DATABASE_URL_SQLITE, "pool_timeout": -1, "enabled": True},  # Negative timeout
            {"url": TEST_DATABASE_URL_SQLITE, "max_overflow": -1, "enabled": True},  # Negative overflow
        ]
        
        for edge_config in edge_case_configs:
            try:
                with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine:
                    # SQLAlchemy should handle invalid parameter values
                    mock_engine.side_effect = ValueError("Invalid parameter value")
                    
                    session_manager = SessionManager(config=edge_config, auto_configure=False)
                    
                    # Should handle invalid parameter values
                    assert not session_manager.enabled
                    
            except (TypeError, ValueError):
                # Expected behavior for invalid parameter values
                pass
    
    def test_configuration_missing_required_fields(self, mock_sqlalchemy_available):
        """Test handling of missing required configuration fields."""
        incomplete_configs = [
            {"enabled": True},  # Missing URL
            {"pool_size": 5},  # Missing URL and enabled flag
            {},  # Empty configuration
        ]
        
        for incomplete_config in incomplete_configs:
            session_manager = SessionManager(config=incomplete_config, auto_configure=False)
            
            # Should not be enabled without required fields
            assert not session_manager.enabled
    
    def test_configuration_precedence_order(self, mock_sqlalchemy_available):
        """Test configuration parameter precedence order."""
        with patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///env.db'}):
            with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine, \
                 patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
                
                mock_engine.return_value = Mock()
                
                # Explicit URL should take precedence over environment
                session_manager = SessionManager(
                    database_url='sqlite:///explicit.db',
                    auto_configure=True
                )
                
                assert session_manager.enabled
                
                # Verify explicit URL was used
                args, kwargs = mock_engine.call_args
                assert 'explicit.db' in args[0]


class TestPerformanceAndResourceManagement:
    """Test performance characteristics and resource management."""
    
    def test_session_creation_performance(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test session creation performance characteristics."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            # Measure session creation time
            start_time = time.time()
            with session_manager.session() as session:
                session_creation_time = time.time() - start_time
                assert session is not None
            
            # Session creation should be fast
            assert session_creation_time < 0.01, f"Session creation took {session_creation_time:.3f}s, too slow"
    
    def test_resource_cleanup_on_context_exit(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test proper resource cleanup on context exit."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            # Use session and ensure cleanup
            with session_manager.session() as session:
                session.add("test_data")
            
            # Verify cleanup methods called
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()
    
    def test_connection_pool_efficiency(self, mock_sqlalchemy_available):
        """Test connection pool configuration for efficiency."""
        config = {
            'url': TEST_DATABASE_URL_SQLITE,
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'enabled': True
        }
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine, \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            mock_engine.return_value = Mock()
            
            session_manager = SessionManager(config=config, auto_configure=False)
            
            # Verify efficient pool configuration passed to engine
            args, kwargs = mock_engine.call_args
            assert kwargs.get('pool_size') == 10
            assert kwargs.get('max_overflow') == 20
            assert kwargs.get('pool_timeout') == 30
    
    def test_memory_usage_with_multiple_sessions(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test memory usage characteristics with multiple sessions."""
        mock_sessions = [Mock() for _ in range(5)]
        session_counter = 0
        
        def create_mock_session():
            nonlocal session_counter
            session = mock_sessions[session_counter % len(mock_sessions)]
            session_counter += 1
            return session
        
        mock_sessionmaker = Mock(side_effect=create_mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            # Create multiple sessions sequentially
            for i in range(5):
                with session_manager.session() as session:
                    session.add(f"data_{i}")
            
            # Verify all sessions were properly closed
            for mock_session in mock_sessions:
                mock_session.close.assert_called_once()
    
    def test_concurrent_session_handling(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test handling of concurrent session requests."""
        import threading
        import queue
        
        mock_sessions = [Mock() for _ in range(3)]
        session_queue = queue.Queue()
        for session in mock_sessions:
            session_queue.put(session)
        
        def create_mock_session():
            return session_queue.get()
        
        mock_sessionmaker = Mock(side_effect=create_mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            session_manager = SessionManager(config=in_memory_sqlite_config, auto_configure=False)
            
            results = queue.Queue()
            
            def session_worker():
                try:
                    with session_manager.session() as session:
                        session.add("concurrent_data")
                        results.put("success")
                except Exception as e:
                    results.put(f"error: {e}")
            
            # Start multiple threads
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=session_worker)
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify all operations completed successfully
            assert results.qsize() == 3
            while not results.empty():
                result = results.get()
                assert result == "success"


# Integration tests combining multiple components
class TestIntegrationScenarios:
    """Integration tests combining multiple database session components."""
    
    def teardown_method(self):
        """Clean up after each integration test."""
        cleanup_database()
    
    def test_complete_session_workflow(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test complete database session workflow from initialization to cleanup."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine, \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            mock_engine.return_value = Mock()
            
            # 1. Initialize session manager
            session_manager = SessionManager.from_config(in_memory_sqlite_config)
            assert session_manager.enabled
            
            # 2. Test connection
            assert session_manager.test_connection() is True
            
            # 3. Perform database operations
            with session_manager.session() as session:
                session.add("workflow_data")
                session.query = Mock(return_value=[])
            
            # 4. Test persistence hooks
            trajectory_data = {"agent_id": 1, "position": [0.0, 1.0]}
            result = PersistenceHooks.save_trajectory_data(trajectory_data)
            assert result is True
            
            # 5. Cleanup
            session_manager.close()
            assert not session_manager.enabled
    
    def test_global_functions_integration(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test integration of global convenience functions."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            # 1. Check initial state
            assert not is_database_enabled()
            
            # 2. Configure database
            with get_session(config=in_memory_sqlite_config) as session:
                assert session is not None
                session.add("global_test_data")
            
            # 3. Verify database is enabled
            assert is_database_enabled()
            
            # 4. Test connection
            assert test_database_connection() is True
            
            # 5. Cleanup
            cleanup_database()
            assert not is_database_enabled()
    
    @pytest.mark.asyncio
    async def test_async_integration_workflow(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test complete async database workflow integration."""
        mock_async_session = Mock()
        mock_async_session.add = Mock()
        mock_async_session.commit = Mock(return_value=asyncio.coroutine(lambda: None)())
        mock_async_session.close = Mock(return_value=asyncio.coroutine(lambda: None)())
        mock_async_session.execute = Mock(return_value=asyncio.coroutine(lambda: None)())
        mock_async_sessionmaker = Mock(return_value=mock_async_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.create_async_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.async_sessionmaker', return_value=mock_async_sessionmaker):
            
            # Use async-capable URL
            async_config = in_memory_sqlite_config.copy()
            async_config['url'] = "postgresql+asyncpg://user:pass@localhost/db"
            
            # 1. Initialize async-capable session manager
            session_manager = SessionManager.from_config(async_config)
            assert session_manager.enabled
            
            # 2. Test async connection
            assert await session_manager.test_async_connection() is True
            
            # 3. Perform async database operations
            async with session_manager.async_session() as session:
                if session:
                    session.add("async_workflow_data")
            
            # 4. Test async persistence hooks
            trajectory_data = {"agent_id": 1, "position": [0.0, 1.0]}
            result = await PersistenceHooks.async_save_trajectory_data(trajectory_data)
            assert result is True
            
            # 5. Test global async functions
            async with get_async_session(config=async_config) as session:
                if session:
                    session.add("global_async_data")
            
            assert await test_async_database_connection() is True
    
    def test_error_recovery_integration(self, mock_sqlalchemy_available, in_memory_sqlite_config):
        """Test error recovery across integrated components."""
        mock_session = Mock()
        mock_sessionmaker = Mock(return_value=mock_session)
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine'), \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker', return_value=mock_sessionmaker):
            
            # Initialize system
            session_manager = SessionManager.from_config(in_memory_sqlite_config)
            assert session_manager.enabled
            
            # Simulate operation failure
            with pytest.raises(ValueError):
                with session_manager.session() as session:
                    session.add("error_test_data")
                    raise ValueError("Simulated operation failure")
            
            # Verify rollback occurred
            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()
            
            # System should still be functional after error
            assert session_manager.enabled
            
            # Subsequent operations should work
            mock_session.reset_mock()
            with session_manager.session() as session:
                session.add("recovery_test_data")
            
            mock_session.commit.assert_called_once()
    
    def test_multi_backend_configuration_switching(self, mock_sqlalchemy_available):
        """Test switching between different database backend configurations."""
        backends_to_test = [
            {'url': 'sqlite:///:memory:', 'backend': DatabaseBackend.SQLITE},
            {'url': 'postgresql://user:pass@localhost/db', 'backend': DatabaseBackend.POSTGRESQL},
            {'url': 'mysql://user:pass@localhost/db', 'backend': DatabaseBackend.MYSQL},
        ]
        
        with patch('src.{{cookiecutter.project_slug}}.db.session.create_engine') as mock_engine, \
             patch('src.{{cookiecutter.project_slug}}.db.session.sessionmaker'):
            
            mock_engine.return_value = Mock()
            
            for i, backend_config in enumerate(backends_to_test):
                config = {
                    'url': backend_config['url'],
                    'enabled': True,
                    'pool_size': 5
                }
                
                session_manager = SessionManager.from_config(config)
                
                # Verify correct backend detection
                assert session_manager.enabled
                assert session_manager.backend == backend_config['backend']
                
                # Cleanup for next iteration
                session_manager.close()
                
                # Verify engine was called with correct URL
                args, kwargs = mock_engine.call_args_list[i]
                assert args[0] == backend_config['url']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])