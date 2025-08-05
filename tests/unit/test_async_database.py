"""
Test suite for async_database.py following TDD principles.

This test suite is designed to:
1. Test async database adapter interface compatibility
2. Test connection pooling functionality
3. Test async context manager behavior
4. Test proper exception handling with custom exceptions
5. Test all database operations (execute_query, get_schema, insert operations)

Following TDD: These tests will initially fail and pass after implementation.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.agent.exceptions import (
    DatabaseConnectionException,
    DatabaseQueryException,
    DatabaseTransactionException,
)


def create_mock_async_context_manager(return_value):
    """Helper function to create a properly mocked async context manager."""

    # Create a mock that supports the async context manager protocol
    async_cm = MagicMock()

    # Mock the __aenter__ and __aexit__ methods as async coroutines
    async def mock_aenter(self):
        return return_value

    async def mock_aexit(self, exc_type, exc_val, exc_tb):
        return False

    async_cm.__aenter__ = mock_aenter
    async_cm.__aexit__ = mock_aexit

    return async_cm


@pytest.fixture
def database_config():
    """Standard async database configuration for tests."""
    return {
        "connection_string": "postgresql://test:test@localhost:5432/test",
        "db_type": "postgres",
        "min_connections": 1,
        "max_connections": 10,
        "connection_timeout": 30,
    }


@pytest.fixture
async def async_database_adapter(database_config):
    """Create an async database adapter instance for testing."""
    # This will fail initially since we haven't implemented the class yet
    from src.agent.adapters.async_database import AsyncDatabaseAdapter

    adapter = AsyncDatabaseAdapter(database_config)
    yield adapter
    # Cleanup: ensure adapter is properly closed
    if hasattr(adapter, "pool") and adapter.pool:
        await adapter.disconnect()


class TestAsyncDatabaseAdapterInitialization:
    """Test async database adapter initialization and configuration."""

    def test_init_with_valid_config(self, database_config):
        """Test that adapter initializes correctly with valid configuration."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)

        assert adapter.connection_string == database_config["connection_string"]
        assert adapter.db_type == database_config["db_type"]
        assert adapter.min_connections == database_config["min_connections"]
        assert adapter.max_connections == database_config["max_connections"]
        assert adapter.connection_timeout == database_config["connection_timeout"]
        assert adapter.pool is None  # Not connected yet

    def test_init_with_minimal_config(self):
        """Test adapter initialization with minimal configuration (defaults)."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        minimal_config = {
            "connection_string": "postgresql://test:test@localhost:5432/test"
        }
        adapter = AsyncDatabaseAdapter(minimal_config)

        assert adapter.db_type == "postgres"  # default
        assert adapter.min_connections == 1  # default
        assert adapter.max_connections == 10  # default
        assert adapter.connection_timeout == 30  # default

    def test_init_inherits_from_abstract_database(self, database_config):
        """Test that AsyncDatabaseAdapter inherits from AsyncAbstractDatabase."""
        from src.agent.adapters.async_database import (
            AsyncDatabaseAdapter,
            AsyncAbstractDatabase,
        )

        adapter = AsyncDatabaseAdapter(database_config)
        assert isinstance(adapter, AsyncAbstractDatabase)


class TestAsyncContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager_connect_disconnect(self, database_config):
        """Test that async context manager properly connects and disconnects."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)

        with patch.object(adapter, "connect", new_callable=AsyncMock) as mock_connect:
            with patch.object(
                adapter, "disconnect", new_callable=AsyncMock
            ) as mock_disconnect:
                async with adapter as db:
                    assert db is adapter
                    mock_connect.assert_called_once()

                mock_disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_exception_handling(self, database_config):
        """Test that async context manager properly disconnects even on exceptions."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)

        with patch.object(adapter, "connect", new_callable=AsyncMock):
            with patch.object(
                adapter, "disconnect", new_callable=AsyncMock
            ) as mock_disconnect:
                with pytest.raises(ValueError):
                    async with adapter:
                        raise ValueError("Test exception")

                mock_disconnect.assert_called_once()


class TestConnectionPooling:
    """Test connection pooling functionality."""

    @pytest.mark.asyncio
    @patch(
        "src.agent.adapters.async_database.asyncpg.create_pool", new_callable=AsyncMock
    )
    async def test_connect_creates_connection_pool(
        self, mock_create_pool, database_config
    ):
        """Test that connect() creates an asyncpg connection pool."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool

        adapter = AsyncDatabaseAdapter(database_config)
        await adapter.connect()

        mock_create_pool.assert_called_once_with(
            database_config["connection_string"],
            min_size=database_config["min_connections"],
            max_size=database_config["max_connections"],
            timeout=database_config["connection_timeout"],
        )
        assert adapter.pool is mock_pool

    @pytest.mark.asyncio
    @patch("src.agent.adapters.async_database.asyncpg.create_pool")
    async def test_connect_handles_connection_failure(
        self, mock_create_pool, database_config
    ):
        """Test that connect() raises DatabaseConnectionException on failure."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        original_error = Exception("Connection refused")
        mock_create_pool.side_effect = original_error

        adapter = AsyncDatabaseAdapter(database_config)

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await adapter.connect()

        # Test exception chaining
        assert exc_info.value.original_exception is original_error
        assert exc_info.value.__cause__ is original_error

        # Test context preservation
        context = exc_info.value.context
        assert "connection_string" in context
        assert "db_type" in context
        assert context["db_type"] == "postgres"
        assert "operation" in context
        assert context["operation"] == "create_pool"

    @pytest.mark.asyncio
    async def test_disconnect_closes_pool(self, database_config):
        """Test that disconnect() properly closes the connection pool."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        mock_pool = AsyncMock()
        adapter.pool = mock_pool

        await adapter.disconnect()

        mock_pool.close.assert_called_once()
        mock_pool.wait_closed.assert_called_once()
        assert adapter.pool is None

    @pytest.mark.asyncio
    async def test_disconnect_handles_none_pool(self, database_config):
        """Test that disconnect() handles None pool gracefully."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        assert adapter.pool is None

        # Should not raise exception
        await adapter.disconnect()


class TestExecuteQuery:
    """Test async execute_query functionality."""

    @pytest.mark.asyncio
    async def test_execute_query_success(self, database_config):
        """Test successful query execution returns DataFrame."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()

        # Create an async context manager mock - pool.acquire() should NOT be a coroutine
        # It should return the context manager directly
        mock_pool.acquire = Mock(
            return_value=create_mock_async_context_manager(mock_connection)
        )

        # Mock query result
        mock_records = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]
        mock_connection.fetch.return_value = mock_records
        adapter.pool = mock_pool

        sql_statement = "SELECT * FROM test_table"
        params = {"limit": 10}

        result = await adapter.execute_query(sql_statement, params)

        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) == 2
        assert list(result["data"].columns) == ["id", "name"]

        mock_connection.fetch.assert_called_once_with(sql_statement, *params.values())

    @pytest.mark.asyncio
    async def test_execute_query_no_pool_raises_exception(self, database_config):
        """Test that execute_query raises exception when pool is not available."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        assert adapter.pool is None

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await adapter.execute_query("SELECT * FROM test")

        context = exc_info.value.context
        assert "pool_available" in context
        assert context["pool_available"] is False
        assert "operation" in context
        assert context["operation"] == "execute_query"

    @pytest.mark.asyncio
    async def test_execute_query_database_error_raises_query_exception(
        self, database_config
    ):
        """Test that database errors during query execution raise DatabaseQueryException."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_pool.acquire = Mock(
            return_value=create_mock_async_context_manager(mock_connection)
        )

        original_error = Exception("Syntax error in SQL")
        mock_connection.fetch.side_effect = original_error
        adapter.pool = mock_pool

        sql_statement = "SELECT * FROM nonexistent_table"
        params = {"limit": 10}

        with pytest.raises(DatabaseQueryException) as exc_info:
            await adapter.execute_query(sql_statement, params)

        # Test exception chaining
        assert exc_info.value.original_exception is original_error
        assert exc_info.value.__cause__ is original_error

        # Test context preservation
        context = exc_info.value.context
        assert "query" in context
        assert context["query"] == sql_statement
        assert "parameters" in context
        assert context["parameters"] == params
        assert "operation" in context
        assert context["operation"] == "execute_query"

    @pytest.mark.asyncio
    async def test_execute_query_with_no_params(self, database_config):
        """Test query execution without parameters."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_pool.acquire = Mock(
            return_value=create_mock_async_context_manager(mock_connection)
        )

        mock_records = [{"count": 5}]
        mock_connection.fetch.return_value = mock_records
        adapter.pool = mock_pool

        sql_statement = "SELECT COUNT(*) as count FROM test_table"

        result = await adapter.execute_query(sql_statement)

        assert isinstance(result["data"], pd.DataFrame)
        assert result["data"].iloc[0]["count"] == 5

        mock_connection.fetch.assert_called_once_with(sql_statement)


class TestGetSchema:
    """Test async get_schema functionality."""

    @pytest.mark.asyncio
    async def test_get_schema_success(self, database_config):
        """Test successful schema retrieval."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_pool.acquire = Mock(
            return_value=create_mock_async_context_manager(mock_connection)
        )

        # Mock schema query results
        mock_tables = [
            {"table_name": "users", "table_schema": "public"},
            {"table_name": "orders", "table_schema": "public"},
        ]
        mock_columns = [
            {"table_name": "users", "column_name": "id", "data_type": "integer"},
            {"table_name": "users", "column_name": "name", "data_type": "varchar"},
        ]

        mock_connection.fetch.side_effect = [mock_tables, mock_columns]
        adapter.pool = mock_pool

        result = await adapter.get_schema()

        assert isinstance(result, dict)
        assert "tables" in result
        assert "columns" in result
        assert len(result["tables"]) == 2
        assert len(result["columns"]) == 2

    @pytest.mark.asyncio
    async def test_get_schema_no_pool_raises_exception(self, database_config):
        """Test that get_schema raises exception when pool is not available."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        assert adapter.pool is None

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await adapter.get_schema()

        context = exc_info.value.context
        assert "pool_available" in context
        assert context["pool_available"] is False
        assert "operation" in context
        assert context["operation"] == "get_schema"

    @pytest.mark.asyncio
    async def test_get_schema_database_error_raises_query_exception(
        self, database_config
    ):
        """Test that database errors during schema retrieval raise DatabaseQueryException."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_pool.acquire = Mock(
            return_value=create_mock_async_context_manager(mock_connection)
        )

        original_error = Exception("Permission denied")
        mock_connection.fetch.side_effect = original_error
        adapter.pool = mock_pool

        with pytest.raises(DatabaseQueryException) as exc_info:
            await adapter.get_schema()

        # Test exception chaining
        assert exc_info.value.original_exception is original_error
        assert exc_info.value.__cause__ is original_error

        # Test context preservation
        context = exc_info.value.context
        assert "operation" in context
        assert context["operation"] == "schema_reflection"


class TestInsertOperations:
    """Test async insert operations (insert_data and insert_batch)."""

    @pytest.mark.asyncio
    async def test_insert_data_success(self, database_config):
        """Test successful single row insertion."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_pool.acquire = Mock(
            return_value=create_mock_async_context_manager(mock_connection)
        )
        mock_connection.transaction = Mock(
            return_value=create_mock_async_context_manager(None)
        )

        adapter.pool = mock_pool

        table_name = "test_table"
        data = {"id": 1, "name": "test"}

        result = await adapter.insert_data(table_name, data)

        assert result is True
        mock_connection.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_batch_success(self, database_config):
        """Test successful batch insertion."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_pool.acquire = Mock(
            return_value=create_mock_async_context_manager(mock_connection)
        )
        mock_connection.transaction = Mock(
            return_value=create_mock_async_context_manager(None)
        )

        adapter.pool = mock_pool

        table_name = "test_table"
        data_list = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]

        result = await adapter.insert_batch(table_name, data_list)

        assert result is True
        # Should call execute for each row
        assert mock_connection.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_insert_batch_no_pool_raises_exception(self, database_config):
        """Test that insert_batch raises exception when pool is not available."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        assert adapter.pool is None

        data = [{"id": 1, "name": "test"}]

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await adapter.insert_batch("test_table", data)

        context = exc_info.value.context
        assert "pool_available" in context
        assert context["pool_available"] is False
        assert "operation" in context
        assert context["operation"] == "insert_batch"

    @pytest.mark.asyncio
    async def test_insert_batch_empty_data_returns_true(self, database_config):
        """Test that insert_batch returns True for empty data list."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        mock_pool = AsyncMock()
        adapter.pool = mock_pool

        result = await adapter.insert_batch("test_table", [])

        assert result is True
        # Should not try to acquire connection for empty data
        mock_pool.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_insert_batch_database_error_raises_transaction_exception(
        self, database_config
    ):
        """Test that database errors during insertion raise DatabaseTransactionException."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_pool.acquire = Mock(
            return_value=create_mock_async_context_manager(mock_connection)
        )
        mock_connection.transaction = Mock(
            return_value=create_mock_async_context_manager(None)
        )

        original_error = Exception("Duplicate key constraint")
        mock_connection.execute.side_effect = original_error
        adapter.pool = mock_pool

        table_name = "test_table"
        data_list = [{"id": 1, "name": "test"}]

        with pytest.raises(DatabaseTransactionException) as exc_info:
            await adapter.insert_batch(table_name, data_list)

        # Test exception chaining
        assert exc_info.value.original_exception is original_error
        assert exc_info.value.__cause__ is original_error

        # Test context preservation
        context = exc_info.value.context
        assert "table_name" in context
        assert context["table_name"] == table_name
        assert "row_count" in context
        assert context["row_count"] == 1
        assert "operation" in context
        assert context["operation"] == "batch_insert"


class TestCompatibilityWithSyncAdapter:
    """Test that async adapter maintains compatibility with sync adapter interface."""

    def test_same_method_signatures_as_sync_adapter(self, database_config):
        """Test that async adapter has same methods as sync adapter (async versions)."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter
        from src.agent.adapters.database import BaseDatabaseAdapter

        async_adapter = AsyncDatabaseAdapter(database_config)
        sync_adapter = BaseDatabaseAdapter(database_config)

        # Both should have the same core methods
        # Filter to core database methods
        core_methods = [
            "connect",
            "disconnect",
            "execute_query",
            "get_schema",
            "insert_data",
            "insert_batch",
        ]

        for method in core_methods:
            assert hasattr(async_adapter, method), (
                f"AsyncDatabaseAdapter missing {method}"
            )
            assert hasattr(sync_adapter, method), (
                f"BaseDatabaseAdapter missing {method}"
            )

    def test_initialization_parameters_compatible(self, database_config):
        """Test that async adapter accepts same initialization parameters as sync adapter."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter
        from src.agent.adapters.database import BaseDatabaseAdapter

        # Both should initialize with same config
        async_adapter = AsyncDatabaseAdapter(database_config)
        sync_adapter = BaseDatabaseAdapter(database_config)

        assert async_adapter.connection_string == sync_adapter.connection_string
        assert async_adapter.db_type == sync_adapter.db_type

    @pytest.mark.asyncio
    async def test_execute_query_returns_same_format_as_sync(self, database_config):
        """Test that async execute_query returns same format as sync version."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        adapter = AsyncDatabaseAdapter(database_config)
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_pool.acquire = Mock(
            return_value=create_mock_async_context_manager(mock_connection)
        )

        mock_records = [{"id": 1, "name": "test"}]
        mock_connection.fetch.return_value = mock_records
        adapter.pool = mock_pool

        result = await adapter.execute_query("SELECT * FROM test")

        # Should return same format as sync adapter: {"data": DataFrame}
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], pd.DataFrame)


class TestSensitiveDataFiltering:
    """Test that async adapter properly filters sensitive data in exceptions."""

    @pytest.mark.asyncio
    @patch("src.agent.adapters.async_database.asyncpg.create_pool")
    async def test_connection_exception_filters_sensitive_data(
        self, mock_create_pool, database_config
    ):
        """Test that connection exceptions filter sensitive data in context."""
        from src.agent.adapters.async_database import AsyncDatabaseAdapter

        # Use connection string with credentials
        database_config["connection_string"] = (
            "postgresql://user:secret123@localhost:5432/db"
        )

        original_error = Exception("Connection failed")
        mock_create_pool.side_effect = original_error

        adapter = AsyncDatabaseAdapter(database_config)

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await adapter.connect()

        # Test that sensitive data is filtered in sanitized context
        sanitized_context = exc_info.value.get_sanitized_context()
        assert sanitized_context["connection_string"] == "[FILTERED]"

        # But original context still contains real data for debugging
        assert "secret123" in exc_info.value.context["connection_string"]
