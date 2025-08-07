"""
Test suite for database.py exception handling with async implementation.

This test suite validates:
1. Proper exception types are raised for different error conditions
2. Context preservation for debugging
3. Exception chaining from original errors
4. Error propagation (no silent failures)
5. Async operation exception handling
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError

from src.agent.adapters.database import BaseDatabaseAdapter
from src.agent.exceptions import (
    DatabaseConnectionException,
    DatabaseQueryException,
    DatabaseTransactionException,
)


@pytest.fixture
def database_config():
    """Standard database configuration for tests."""
    return {
        "connection_string": "postgresql://test:test@localhost:5432/test",
        "db_type": "postgres",
    }


@pytest.fixture
def database_adapter(database_config):
    """Create a database adapter instance for testing."""
    return BaseDatabaseAdapter(database_config)


class TestAsyncExceptionBehavior:
    """Test exception handling in async database operations."""

    @pytest.mark.asyncio
    @patch("src.agent.adapters.database.create_async_engine")
    async def test_create_async_engine_raises_database_connection_exception(
        self, mock_create_async_engine, database_adapter
    ):
        """Test that _create_async_engine raises DatabaseConnectionException with proper context."""
        original_error = OperationalError("Connection timeout", None, None)
        mock_create_async_engine.side_effect = original_error

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await database_adapter._create_async_engine()

        # Test exception chaining
        assert exc_info.value.original_exception is original_error
        assert exc_info.value.__cause__ is original_error

        # Test context preservation
        context = exc_info.value.context
        assert "connection_string" in context
        assert "db_type" in context
        assert context["db_type"] == "postgres"

        # Test message quality
        assert "connection" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_query_async_raises_database_query_exception(
        self, database_adapter
    ):
        """Test that execute_query_async raises DatabaseQueryException with proper context."""
        # Mock the engine and session
        mock_engine = AsyncMock()
        mock_session = AsyncMock()

        database_adapter.engine = mock_engine
        database_adapter.session_maker = Mock(return_value=mock_session)

        # Setup the async context manager
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None

        original_error = SQLAlchemyError("Invalid SQL syntax")
        mock_session.execute.side_effect = original_error

        sql_statement = "SELECT * FROM nonexistent_table"
        params = {"limit": 10}

        with pytest.raises(DatabaseQueryException) as exc_info:
            await database_adapter.execute_query_async(sql_statement, params)

        # Test exception chaining
        assert exc_info.value.original_exception is original_error
        assert exc_info.value.__cause__ is original_error

        # Test context preservation
        context = exc_info.value.context
        assert "query" in context
        assert context["query"] == sql_statement
        assert "parameters" in context
        assert context["parameters"] == params

        # Test message quality
        assert "query" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_get_schema_async_raises_database_query_exception(
        self, database_adapter
    ):
        """Test that get_schema_async raises DatabaseQueryException with proper context."""
        mock_engine = AsyncMock()
        mock_session = AsyncMock()

        database_adapter.engine = mock_engine
        database_adapter.session_maker = Mock(return_value=mock_session)

        # Setup the async context manager
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None

        # Mock the run_sync method to raise an error
        original_error = OperationalError("Permission denied", None, None)
        mock_session.run_sync.side_effect = original_error

        with pytest.raises(DatabaseQueryException) as exc_info:
            await database_adapter.get_schema_async()

        # Test exception chaining
        assert exc_info.value.original_exception is original_error
        assert exc_info.value.__cause__ is original_error

        # Test context preservation
        context = exc_info.value.context
        assert "operation" in context
        assert context["operation"] == "schema_reflection"

        # Test message quality
        assert "schema" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_insert_batch_raises_database_transaction_exception(
        self, database_adapter
    ):
        """Test that insert_batch raises DatabaseTransactionException with proper context."""
        mock_engine = AsyncMock()
        mock_session = AsyncMock()
        mock_transaction = AsyncMock()

        database_adapter.engine = mock_engine
        database_adapter.session_maker = Mock(return_value=mock_session)

        # Setup the async context managers
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None

        # Setup the begin() method to return an async context manager
        mock_session.begin = Mock(return_value=mock_transaction)
        mock_transaction.__aenter__.return_value = mock_transaction
        mock_transaction.__aexit__.return_value = None

        original_error = IntegrityError("Duplicate key", None, None)
        mock_session.execute.side_effect = original_error

        data = [{"id": 1, "name": "test"}, {"id": 2, "name": "test2"}]
        table_name = "test_table"

        with pytest.raises(DatabaseTransactionException) as exc_info:
            await database_adapter.insert_batch(table_name, data)

        # Test exception chaining
        assert exc_info.value.original_exception is original_error
        assert exc_info.value.__cause__ is original_error

        # Test context preservation
        context = exc_info.value.context
        assert "table_name" in context
        assert context["table_name"] == table_name
        assert "row_count" in context
        assert context["row_count"] == 2
        assert "operation" in context
        assert context["operation"] == "batch_insert"

        # Test message quality
        assert "insert" in str(exc_info.value).lower()


class TestContextPreservation:
    """Test that exceptions preserve important context information."""

    @pytest.mark.asyncio
    @patch("src.agent.adapters.database.create_async_engine")
    async def test_connection_exception_preserves_sensitive_data_filtering(
        self, mock_create_async_engine, database_config
    ):
        """Test that connection exceptions filter sensitive data in context."""
        # Use a connection string with credentials
        database_config["connection_string"] = (
            "postgresql://user:secret123@localhost:5432/db"
        )
        database_adapter = BaseDatabaseAdapter(database_config)

        mock_create_async_engine.side_effect = Exception("Connection failed")

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await database_adapter._create_async_engine()

        # Test that sensitive data is filtered
        sanitized_context = exc_info.value.get_sanitized_context()
        assert sanitized_context["connection_string"] == "[FILTERED]"

        # But original context still contains real data for debugging
        assert "secret123" in exc_info.value.context["connection_string"]

    @pytest.mark.asyncio
    async def test_query_exception_preserves_execution_context(self, database_adapter):
        """Test that query exceptions preserve execution timing and query details."""
        mock_engine = AsyncMock()
        mock_session = AsyncMock()

        database_adapter.engine = mock_engine
        database_adapter.session_maker = Mock(return_value=mock_session)

        # Setup the async context manager
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.execute.side_effect = Exception("Query timeout")

        sql_statement = "SELECT * FROM large_table WHERE complex_condition = ?"
        params = {"complex_condition": "value"}

        with pytest.raises(DatabaseQueryException) as exc_info:
            await database_adapter.execute_query_async(sql_statement, params)

        context = exc_info.value.context
        assert context["query"] == sql_statement
        assert context["parameters"] == params
        assert "db_type" in context


class TestExceptionChaining:
    """Test proper exception chaining from original exceptions."""

    @pytest.mark.asyncio
    @patch("src.agent.adapters.database.create_async_engine")
    async def test_connection_exception_chains_original_sqlalchemy_error(
        self, mock_create_async_engine, database_adapter
    ):
        """Test that DatabaseConnectionException properly chains SQLAlchemy errors."""
        original_error = OperationalError("Connection refused", None, None)
        mock_create_async_engine.side_effect = original_error

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await database_adapter._create_async_engine()

        # Test __cause__ is set for exception chaining
        assert exc_info.value.__cause__ is original_error
        assert exc_info.value.original_exception is original_error

        # Test that the full exception chain is preserved
        chain = []
        current = exc_info.value
        while current:
            chain.append(type(current).__name__)
            current = current.__cause__

        assert "DatabaseConnectionException" in chain
        assert "OperationalError" in chain

    @pytest.mark.asyncio
    async def test_query_exception_chains_original_error(self, database_adapter):
        """Test that DatabaseQueryException properly chains original errors."""
        mock_engine = AsyncMock()
        mock_session = AsyncMock()

        database_adapter.engine = mock_engine
        database_adapter.session_maker = Mock(return_value=mock_session)

        # Setup the async context manager
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None

        original_error = ValueError("Invalid SQL syntax")
        mock_session.execute.side_effect = original_error

        with pytest.raises(DatabaseQueryException) as exc_info:
            await database_adapter.execute_query_async("SELECT invalid syntax")

        assert exc_info.value.__cause__ is original_error
        assert exc_info.value.original_exception is original_error


class TestErrorPropagation:
    """Test that errors are properly propagated instead of silently failing."""

    @pytest.mark.asyncio
    async def test_execute_query_async_with_no_engine_raises_exception(
        self, database_adapter
    ):
        """Test that execute_query_async raises exception when engine is None instead of returning None."""
        # Engine is None by default
        assert database_adapter.engine is None

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await database_adapter.execute_query_async("SELECT * FROM test")

        context = exc_info.value.context
        assert "engine_available" in context
        assert context["engine_available"] is False
        assert "operation" in context
        assert context["operation"] == "execute_query"

    @pytest.mark.asyncio
    async def test_get_schema_async_with_no_engine_raises_exception(
        self, database_adapter
    ):
        """Test that get_schema_async raises exception when engine is None instead of returning None."""
        assert database_adapter.engine is None

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await database_adapter.get_schema_async()

        context = exc_info.value.context
        assert "engine_available" in context
        assert context["engine_available"] is False
        assert "operation" in context
        assert context["operation"] == "get_schema"

    @pytest.mark.asyncio
    async def test_insert_batch_with_no_engine_raises_exception(self, database_adapter):
        """Test that insert_batch raises exception when engine is None instead of returning False."""
        assert database_adapter.engine is None

        data = [{"id": 1, "name": "test"}]

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await database_adapter.insert_batch("test_table", data)

        context = exc_info.value.context
        assert "engine_available" in context
        assert context["engine_available"] is False
        assert "operation" in context
        assert context["operation"] == "insert_batch"


class TestSpecificExceptionTypes:
    """Test that specific exception types are raised for different error conditions."""

    @pytest.mark.asyncio
    @patch("src.agent.adapters.database.create_async_engine")
    async def test_connection_timeout_raises_connection_exception(
        self, mock_create_async_engine, database_adapter
    ):
        """Test that connection timeouts raise DatabaseConnectionException."""
        mock_create_async_engine.side_effect = OperationalError(
            "Connection timeout", None, None
        )

        with pytest.raises(DatabaseConnectionException):
            await database_adapter._create_async_engine()

    @pytest.mark.asyncio
    async def test_sql_syntax_error_raises_query_exception(self, database_adapter):
        """Test that SQL syntax errors raise DatabaseQueryException."""
        mock_engine = AsyncMock()
        mock_session = AsyncMock()

        database_adapter.engine = mock_engine
        database_adapter.session_maker = Mock(return_value=mock_session)

        # Setup the async context manager
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session.execute.side_effect = SQLAlchemyError("Syntax error")

        with pytest.raises(DatabaseQueryException):
            await database_adapter.execute_query_async("SELECT * FROM")

    @pytest.mark.asyncio
    async def test_integrity_constraint_raises_transaction_exception(
        self, database_adapter
    ):
        """Test that integrity constraint violations raise DatabaseTransactionException."""
        mock_engine = AsyncMock()
        mock_session = AsyncMock()
        mock_transaction = AsyncMock()

        database_adapter.engine = mock_engine
        database_adapter.session_maker = Mock(return_value=mock_session)

        # Setup the async context managers
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None

        # Setup the begin() method to return an async context manager
        mock_session.begin = Mock(return_value=mock_transaction)
        mock_transaction.__aenter__.return_value = mock_transaction
        mock_transaction.__aexit__.return_value = None

        mock_session.execute.side_effect = IntegrityError("Duplicate key", None, None)

        data = [{"id": 1, "name": "test"}]

        with pytest.raises(DatabaseTransactionException):
            await database_adapter.insert_batch("test_table", data)


class TestSyncMethodsBackwardCompatibility:
    """Test that sync wrapper methods work correctly."""

    def test_execute_query_sync_wrapper_calls_async(self, database_adapter):
        """Test that the sync execute_query method properly wraps async version."""
        with patch.object(
            database_adapter, "execute_query_async", new_callable=AsyncMock
        ) as mock_async:
            mock_async.return_value = {"result": "test"}

            result = database_adapter.execute_query("SELECT * FROM test")

            mock_async.assert_called_once_with("SELECT * FROM test", None, None, None)
            assert result == {"result": "test"}

    def test_get_schema_sync_wrapper_calls_async(self, database_adapter):
        """Test that the sync get_schema method properly wraps async version."""
        with patch.object(
            database_adapter, "get_schema_async", new_callable=AsyncMock
        ) as mock_async:
            mock_async.return_value = {"tables": ["test"]}

            result = database_adapter.get_schema()

            mock_async.assert_called_once()
            assert result == {"tables": ["test"]}

    @pytest.mark.asyncio
    async def test_connect_initializes_engine(self, database_adapter):
        """Test that connect() method properly initializes the engine."""
        with patch.object(
            database_adapter, "_create_async_engine", new_callable=AsyncMock
        ) as mock_create:
            with patch.object(
                database_adapter, "_test_connection", new_callable=AsyncMock
            ) as mock_test:
                mock_engine = AsyncMock()
                mock_create.return_value = mock_engine
                mock_test.return_value = None

                await database_adapter.connect()

                assert database_adapter.engine == mock_engine
                mock_create.assert_called_once()
                mock_test.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_disposes_engine(self, database_adapter):
        """Test that disconnect() method properly disposes the engine."""
        mock_engine = AsyncMock()
        database_adapter.engine = mock_engine

        await database_adapter.disconnect()

        mock_engine.dispose.assert_called_once()
        assert database_adapter.engine is None
