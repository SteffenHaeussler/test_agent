"""
Test suite for database adapter with retry functionality.

This test suite follows TDD principles and tests the integration of retry logic
with the database adapters. It ensures that:

1. Retry logic is properly applied to database operations
2. Retryable exceptions trigger retries with exponential backoff
3. Non-retryable exceptions are not retried
4. Existing functionality is preserved
5. Retry attempts are properly logged
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError

from src.agent.adapters.database import BaseDatabaseAdapter
from src.agent.adapters.async_database import AsyncDatabaseAdapter
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
def sync_adapter(database_config):
    """Create a sync database adapter instance for testing."""
    return BaseDatabaseAdapter(database_config)


@pytest.fixture
def async_adapter(database_config):
    """Create an async database adapter instance for testing."""
    return AsyncDatabaseAdapter(database_config)


class TestSyncDatabaseAdapterRetry:
    """Test retry functionality in synchronous database adapter."""

    @patch("src.agent.adapters.database.create_engine")
    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_get_connection_retries_on_connection_failure(
        self, mock_sleep, mock_create_engine, sync_adapter
    ):
        """Test that _get_connection retries on connection failures."""
        # First two attempts fail with retryable exception, third succeeds
        mock_engine = Mock()
        mock_create_engine.side_effect = [
            OperationalError("Connection timeout", None, None),
            OperationalError("Connection refused", None, None),
            mock_engine,
        ]

        result = sync_adapter._get_connection()

        assert result == mock_engine
        assert mock_create_engine.call_count == 3
        # Should have slept twice (between retries)
        assert mock_sleep.call_count == 2

    @patch("src.agent.adapters.database.create_engine")
    def test_get_connection_does_not_retry_permission_errors(
        self, mock_create_engine, sync_adapter
    ):
        """Test that _get_connection does not retry permission errors."""
        # Simulate a permission error (non-retryable)
        permission_error = Exception("Permission denied for database")
        mock_create_engine.side_effect = permission_error

        with pytest.raises(DatabaseConnectionException):
            sync_adapter._get_connection()

        # Should only try once (no retries for non-retryable errors)
        assert mock_create_engine.call_count == 1

    @patch("src.agent.adapters.database.pd.read_sql_query")
    @patch("time.sleep")
    def test_execute_query_retries_on_timeout(
        self, mock_sleep, mock_read_sql_query, sync_adapter
    ):
        """Test that execute_query retries on timeout errors."""
        sync_adapter.engine = Mock()

        # First attempt fails with timeout, second succeeds
        import pandas as pd

        success_df = pd.DataFrame({"id": [1], "name": ["test"]})
        mock_read_sql_query.side_effect = [
            Exception("Query timeout after 30 seconds"),
            success_df,
        ]

        result = sync_adapter.execute_query("SELECT * FROM users")

        assert result == {"data": success_df}
        assert mock_read_sql_query.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("src.agent.adapters.database.pd.read_sql_query")
    def test_execute_query_does_not_retry_syntax_errors(
        self, mock_read_sql_query, sync_adapter
    ):
        """Test that execute_query does not retry SQL syntax errors."""
        sync_adapter.engine = Mock()

        syntax_error = SQLAlchemyError("SQL syntax error near 'FROM'")
        mock_read_sql_query.side_effect = syntax_error

        with pytest.raises(DatabaseQueryException):
            sync_adapter.execute_query("SELECT * FROM")

        # Should only try once (no retries for syntax errors)
        assert mock_read_sql_query.call_count == 1

    @patch("src.agent.adapters.database.text")
    @patch("time.sleep")
    def test_insert_batch_retries_on_connection_issues(
        self, mock_sleep, mock_text, sync_adapter
    ):
        """Test that insert_batch retries on connection-related issues."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        sync_adapter.engine = mock_engine

        # First attempt fails with connection issue, second succeeds
        mock_engine.begin.return_value.__enter__.side_effect = [
            Exception("Connection lost during transaction"),
            mock_conn,
        ]

        data = [{"id": 1, "name": "test"}]
        result = sync_adapter.insert_batch("test_table", data)

        assert result is True
        assert mock_engine.begin.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("src.agent.adapters.database.text")
    def test_insert_batch_does_not_retry_integrity_violations(
        self, mock_text, sync_adapter
    ):
        """Test that insert_batch does not retry integrity constraint violations."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn
        sync_adapter.engine = mock_engine

        integrity_error = IntegrityError("UNIQUE constraint failed", None, None)
        mock_conn.execute.side_effect = integrity_error

        data = [{"id": 1, "name": "test"}]

        with pytest.raises(DatabaseTransactionException):
            sync_adapter.insert_batch("test_table", data)

        # Should only try once (no retries for integrity violations)
        assert mock_engine.begin.call_count == 1

    @patch("src.agent.adapters.database.MetaData")
    @patch("time.sleep")
    def test_get_schema_retries_on_temporary_failures(
        self, mock_sleep, mock_metadata_class, sync_adapter
    ):
        """Test that get_schema retries on temporary failures."""
        sync_adapter.engine = Mock()
        mock_metadata = Mock()
        mock_metadata_class.return_value = mock_metadata

        # First attempt fails with temporary issue, second succeeds
        mock_metadata.reflect.side_effect = [
            Exception("Database temporarily unavailable"),
            None,  # Success
        ]

        # This should succeed after retry
        result = sync_adapter.get_schema()

        assert result == mock_metadata
        assert mock_metadata.reflect.call_count == 2
        assert mock_sleep.call_count == 1


class TestAsyncDatabaseAdapterRetry:
    """Test retry functionality in asynchronous database adapter."""

    @pytest.mark.asyncio
    @patch(
        "src.agent.adapters.async_database.asyncpg.create_pool", new_callable=AsyncMock
    )
    @patch("asyncio.sleep")
    async def test_connect_retries_on_connection_failure(
        self, mock_sleep, mock_create_pool, async_adapter
    ):
        """Test that connect retries on connection failures."""
        # First two attempts fail, third succeeds
        mock_pool = Mock()
        mock_create_pool.side_effect = [
            Exception("Connection refused"),
            Exception("Connection timeout"),
            mock_pool,
        ]

        await async_adapter.connect()

        assert async_adapter.pool == mock_pool
        assert mock_create_pool.call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    @patch(
        "src.agent.adapters.async_database.asyncpg.create_pool", new_callable=AsyncMock
    )
    async def test_connect_does_not_retry_authentication_errors(
        self, mock_create_pool, async_adapter
    ):
        """Test that connect does not retry authentication errors."""
        auth_error = Exception("Authentication failed for user")
        mock_create_pool.side_effect = auth_error

        with pytest.raises(DatabaseConnectionException):
            await async_adapter.connect()

        # Should only try once
        assert mock_create_pool.call_count == 1

    @pytest.mark.asyncio
    @patch("asyncio.sleep")
    async def test_execute_query_retries_on_timeout(self, mock_sleep, async_adapter):
        """Test that execute_query retries on timeout errors."""
        # Mock the pool and connection
        mock_pool = MagicMock()
        mock_connection = AsyncMock()
        async_adapter.pool = mock_pool

        # Mock the async context manager for pool.acquire()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_connection
        )
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        # First attempt times out, second succeeds
        mock_records = [{"id": 1, "name": "test"}]
        mock_connection.fetch.side_effect = [
            Exception("Query timeout after 30 seconds"),
            mock_records,
        ]

        result = await async_adapter.execute_query("SELECT * FROM users")

        assert "data" in result
        assert len(result["data"]) == 1
        assert mock_connection.fetch.call_count == 2
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_query_does_not_retry_syntax_errors(self, async_adapter):
        """Test that execute_query does not retry SQL syntax errors."""
        # Mock the pool and connection
        mock_pool = MagicMock()
        mock_connection = AsyncMock()
        async_adapter.pool = mock_pool

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_connection
        )
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        syntax_error = Exception("SQL syntax error near 'FROM'")
        mock_connection.fetch.side_effect = syntax_error

        with pytest.raises(DatabaseQueryException):
            await async_adapter.execute_query("SELECT * FROM")

        # Should only try once
        assert mock_connection.fetch.call_count == 1

    @pytest.mark.asyncio
    @patch("asyncio.sleep")
    async def test_insert_batch_retries_on_connection_issues(
        self, mock_sleep, async_adapter
    ):
        """Test that insert_batch retries on connection-related issues."""
        # Mock the pool and connection
        mock_pool = MagicMock()
        mock_connection = MagicMock()
        mock_transaction = MagicMock()
        async_adapter.pool = mock_pool

        # First attempt fails with connection issue
        mock_pool.acquire.return_value.__aenter__.side_effect = [
            Exception("Connection lost"),
            mock_connection,
        ]
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        # Create proper async context manager mock for transaction
        mock_transaction_cm = MagicMock()
        mock_transaction_cm.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction_cm.__aexit__ = AsyncMock(return_value=None)
        mock_connection.transaction = MagicMock(return_value=mock_transaction_cm)
        mock_connection.execute = AsyncMock(return_value=None)

        data = [{"id": 1, "name": "test"}]
        result = await async_adapter.insert_batch("test_table", data)

        assert result is True
        assert mock_pool.acquire.call_count == 2
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_insert_batch_does_not_retry_integrity_violations(
        self, async_adapter
    ):
        """Test that insert_batch does not retry integrity constraint violations."""
        # Mock the pool and connection
        mock_pool = MagicMock()
        mock_connection = AsyncMock()
        async_adapter.pool = mock_pool

        # Create proper async context manager mock
        mock_transaction_cm = MagicMock()
        mock_transaction_cm.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_transaction_cm.__aexit__ = AsyncMock(return_value=None)
        mock_connection.transaction = MagicMock(return_value=mock_transaction_cm)

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_connection
        )
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        integrity_error = Exception("UNIQUE constraint failed")
        mock_connection.execute.side_effect = integrity_error

        data = [{"id": 1, "name": "test"}]

        with pytest.raises(DatabaseTransactionException):
            await async_adapter.insert_batch("test_table", data)

        # Should only try once
        assert mock_pool.acquire.call_count == 1


class TestRetryLogging:
    """Test that retry attempts are properly logged."""

    @patch("src.agent.adapters.database.create_engine")
    @patch("time.sleep")
    @patch("src.agent.utils.retry.logger")
    def test_sync_adapter_logs_retry_attempts(
        self, mock_logger, mock_sleep, mock_create_engine, sync_adapter
    ):
        """Test that sync adapter logs retry attempts."""
        mock_engine = Mock()
        mock_create_engine.side_effect = [
            OperationalError("Connection failed", None, None),
            mock_engine,
        ]

        sync_adapter._get_connection()

        # Should log the retry attempt
        mock_logger.warning.assert_called_once()
        log_message = mock_logger.warning.call_args[0][0]
        assert "failed on attempt 1" in log_message.lower()
        assert "retrying" in log_message.lower()

    @pytest.mark.asyncio
    @patch(
        "src.agent.adapters.async_database.asyncpg.create_pool", new_callable=AsyncMock
    )
    @patch("asyncio.sleep")
    @patch("src.agent.utils.retry.logger")
    async def test_async_adapter_logs_retry_attempts(
        self, mock_logger, mock_sleep, mock_create_pool, async_adapter
    ):
        """Test that async adapter logs retry attempts."""
        mock_pool = Mock()
        mock_create_pool.side_effect = [Exception("Connection failed"), mock_pool]

        await async_adapter.connect()

        # Should log the retry attempt
        mock_logger.warning.assert_called_once()
        log_message = mock_logger.warning.call_args[0][0]
        assert "failed on attempt 1" in log_message.lower()
        assert "retrying" in log_message.lower()


class TestBackwardCompatibility:
    """Test that adding retry logic doesn't break existing functionality."""

    def test_sync_adapter_preserves_existing_behavior(self, sync_adapter):
        """Test that sync adapter still works as expected with successful operations."""
        with patch.object(sync_adapter, "_get_connection") as mock_get_conn:
            mock_engine = Mock()
            mock_get_conn.return_value = mock_engine

            sync_adapter.connect()

            assert sync_adapter.engine == mock_engine
            mock_get_conn.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_adapter_preserves_existing_behavior(self, async_adapter):
        """Test that async adapter still works as expected with successful operations."""
        with patch(
            "src.agent.adapters.async_database.asyncpg.create_pool",
            new_callable=AsyncMock,
        ) as mock_create_pool:
            mock_pool = Mock()
            mock_create_pool.return_value = mock_pool

            await async_adapter.connect()

            assert async_adapter.pool == mock_pool
            mock_create_pool.assert_called_once()
