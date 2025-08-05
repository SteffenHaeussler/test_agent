"""
Integration tests for database adapter retry functionality.

These tests demonstrate the retry logic working in realistic scenarios
without complex mocking. They focus on the integration between the retry
decorators and the database adapters.
"""

import pytest
from unittest.mock import patch, Mock, AsyncMock
import time

from src.agent.adapters.database import BaseDatabaseAdapter
from src.agent.adapters.async_database import AsyncDatabaseAdapter
from src.agent.exceptions import (
    DatabaseConnectionException,
)


class TestSyncDatabaseRetryIntegration:
    """Integration tests for sync database adapter with retry logic."""

    def test_retry_with_exponential_backoff_timing(self):
        """Test that retry actually waits with exponential backoff."""
        config = {
            "connection_string": "postgresql://test:test@localhost:5432/test",
            "db_type": "postgres",
        }
        adapter = BaseDatabaseAdapter(config)

        # Mock create_engine to always fail
        with patch("src.agent.adapters.database.create_engine") as mock_create_engine:
            mock_create_engine.side_effect = Exception("Connection refused")

            start_time = time.time()

            with pytest.raises(DatabaseConnectionException):
                adapter._get_connection()

            end_time = time.time()
            elapsed = end_time - start_time

            # Should have tried 4 times (initial + 3 retries) with delays
            # Total expected delay: ~1 + 2 + 4 = 7 seconds (with jitter, could be 3.5-7 seconds)
            assert elapsed >= 3.0  # At least 3 seconds with jitter
            assert mock_create_engine.call_count == 4

    def test_retry_stops_on_non_retryable_exception(self):
        """Test that retry stops immediately on non-retryable exceptions."""
        config = {
            "connection_string": "postgresql://test:test@localhost:5432/test",
            "db_type": "postgres",
        }
        adapter = BaseDatabaseAdapter(config)

        # Mock create_engine to fail with syntax error
        with patch("src.agent.adapters.database.create_engine") as mock_create_engine:
            mock_create_engine.side_effect = Exception("SQL syntax error near 'FROM'")

            start_time = time.time()

            with pytest.raises(DatabaseConnectionException):
                adapter._get_connection()

            end_time = time.time()
            elapsed = end_time - start_time

            # Should fail immediately without retries
            assert elapsed < 0.5  # Very quick failure
            assert mock_create_engine.call_count == 1

    def test_retry_success_after_failures(self):
        """Test successful operation after some failures."""
        config = {
            "connection_string": "postgresql://test:test@localhost:5432/test",
            "db_type": "postgres",
        }
        adapter = BaseDatabaseAdapter(config)

        # Mock create_engine to fail twice then succeed
        mock_engine = Mock()
        with patch("src.agent.adapters.database.create_engine") as mock_create_engine:
            mock_create_engine.side_effect = [
                Exception("Connection timeout"),  # First attempt fails
                Exception("Connection refused"),  # Second attempt fails
                mock_engine,  # Third attempt succeeds
            ]

            result = adapter._get_connection()

            assert result == mock_engine
            assert mock_create_engine.call_count == 3

    def test_execute_query_retry_preserves_parameters(self):
        """Test that query parameters are preserved across retry attempts."""
        config = {
            "connection_string": "postgresql://test:test@localhost:5432/test",
            "db_type": "postgres",
        }
        adapter = BaseDatabaseAdapter(config)
        adapter.engine = Mock()  # Set engine to avoid connection check

        # Mock pd.read_sql_query to fail once then succeed
        import pandas as pd

        success_df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})

        with patch("src.agent.adapters.database.pd.read_sql_query") as mock_read_sql:
            mock_read_sql.side_effect = [
                Exception("Query timeout after 30 seconds"),
                success_df,
            ]

            sql_statement = "SELECT * FROM users WHERE active = :active"
            params = {"active": True}

            result = adapter.execute_query(sql_statement, params)

            assert result == {"data": success_df}
            assert mock_read_sql.call_count == 2

            # Verify parameters were passed correctly in both attempts
            for call in mock_read_sql.call_args_list:
                call_args, call_kwargs = call
                assert call_kwargs.get("params") == params


class TestAsyncDatabaseRetryIntegration:
    """Integration tests for async database adapter with retry logic."""

    @pytest.mark.asyncio
    async def test_async_retry_with_timing(self):
        """Test that async retry waits with exponential backoff."""
        config = {
            "connection_string": "postgresql://test:test@localhost:5432/test",
            "db_type": "postgres",
        }
        adapter = AsyncDatabaseAdapter(config)

        # Mock create_pool to always fail
        with patch(
            "src.agent.adapters.async_database.asyncpg.create_pool",
            new_callable=AsyncMock,
        ) as mock_create_pool:
            mock_create_pool.side_effect = Exception("Connection refused")

            start_time = time.time()

            with pytest.raises(DatabaseConnectionException):
                await adapter.connect()

            end_time = time.time()
            elapsed = end_time - start_time

            # Should have tried 4 times with delays
            assert elapsed >= 3.0  # At least 3 seconds with jitter
            assert mock_create_pool.call_count == 4

    @pytest.mark.asyncio
    async def test_async_retry_success_after_failures(self):
        """Test async successful operation after some failures."""
        config = {
            "connection_string": "postgresql://test:test@localhost:5432/test",
            "db_type": "postgres",
        }
        adapter = AsyncDatabaseAdapter(config)

        # Mock create_pool to fail once then succeed
        mock_pool = Mock()
        with patch(
            "src.agent.adapters.async_database.asyncpg.create_pool",
            new_callable=AsyncMock,
        ) as mock_create_pool:
            mock_create_pool.side_effect = [
                Exception("Connection timeout"),  # First attempt fails
                mock_pool,  # Second attempt succeeds
            ]

            await adapter.connect()

            assert adapter.pool == mock_pool
            assert mock_create_pool.call_count == 2


class TestRetryLoggingIntegration:
    """Integration tests for retry logging functionality."""

    def test_retry_logging_contains_proper_context(self):
        """Test that retry logging includes proper context information."""
        config = {
            "connection_string": "postgresql://test:test@localhost:5432/test",
            "db_type": "postgres",
        }
        adapter = BaseDatabaseAdapter(config)

        # Mock create_engine to fail then succeed
        mock_engine = Mock()
        with patch("src.agent.adapters.database.create_engine") as mock_create_engine:
            with patch("src.agent.utils.retry.logger") as mock_logger:
                mock_create_engine.side_effect = [
                    Exception("Connection timeout"),
                    mock_engine,
                ]

                result = adapter._get_connection()

                assert result == mock_engine

                # Should have logged the retry attempt
                mock_logger.warning.assert_called_once()
                log_message = mock_logger.warning.call_args[0][0]

                # Verify log message content
                assert "_get_connection" in log_message
                assert "failed on attempt 1" in log_message
                assert "retrying" in log_message
                assert "Connection timeout" in log_message

    def test_no_logging_on_successful_first_attempt(self):
        """Test that no retry logging occurs on successful first attempt."""
        config = {
            "connection_string": "postgresql://test:test@localhost:5432/test",
            "db_type": "postgres",
        }
        adapter = BaseDatabaseAdapter(config)

        # Mock create_engine to succeed immediately
        mock_engine = Mock()
        with patch("src.agent.adapters.database.create_engine") as mock_create_engine:
            with patch("src.agent.utils.retry.logger") as mock_logger:
                mock_create_engine.return_value = mock_engine

                result = adapter._get_connection()

                assert result == mock_engine

                # Should not have logged any retry attempts
                mock_logger.warning.assert_not_called()


class TestRetryExceptionPreservation:
    """Test that retry logic preserves exception context and chaining."""

    def test_final_exception_preserves_context(self):
        """Test that the final exception after all retries preserves original context."""
        config = {
            "connection_string": "postgresql://test:test@localhost:5432/test",
            "db_type": "postgres",
        }
        adapter = BaseDatabaseAdapter(config)

        # Mock create_engine to always fail with the same exception
        original_exception = Exception("Connection refused")
        with patch("src.agent.adapters.database.create_engine") as mock_create_engine:
            mock_create_engine.side_effect = original_exception

            with pytest.raises(DatabaseConnectionException) as exc_info:
                adapter._get_connection()

            # The final exception should be a DatabaseConnectionException that wraps the original
            assert exc_info.value.original_exception == original_exception
            assert "Connection refused" in str(exc_info.value)

            # Should have tried the maximum number of times
            assert mock_create_engine.call_count == 4  # initial + 3 retries

    def test_exception_chaining_preserved_through_retries(self):
        """Test that exception chaining is preserved even with retries."""
        config = {
            "connection_string": "postgresql://test:test@localhost:5432/test",
            "db_type": "postgres",
        }
        adapter = BaseDatabaseAdapter(config)

        # Create a chain of exceptions
        root_cause = ValueError("Root cause error")
        intermediate_exception = Exception("Intermediate error")
        intermediate_exception.__cause__ = root_cause

        with patch("src.agent.adapters.database.create_engine") as mock_create_engine:
            mock_create_engine.side_effect = intermediate_exception

            with pytest.raises(DatabaseConnectionException) as exc_info:
                adapter._get_connection()

            # Verify the exception chain is preserved
            database_exception = exc_info.value
            assert database_exception.original_exception == intermediate_exception
            assert database_exception.__cause__ == intermediate_exception
            assert intermediate_exception.__cause__ == root_cause
