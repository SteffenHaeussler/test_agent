"""
Integration tests for async database adapter.

These tests demonstrate how to use the AsyncDatabaseAdapter in real scenarios
and verify integration with the async/await ecosystem.
"""

import pytest
from unittest.mock import patch, AsyncMock

from src.agent.adapters.async_database import AsyncDatabaseAdapter
from src.agent.exceptions import DatabaseConnectionException


@pytest.fixture
def async_database_config():
    """Configuration for async database adapter integration tests."""
    return {
        "connection_string": "postgresql://test:test@localhost:5432/test",
        "db_type": "postgres",
        "min_connections": 1,
        "max_connections": 5,
        "connection_timeout": 30,
    }


class TestAsyncDatabaseIntegration:
    """Integration tests for AsyncDatabaseAdapter."""

    def test_async_database_init(self, async_database_config):
        """Test that async database adapter initializes correctly."""
        adapter = AsyncDatabaseAdapter(async_database_config)

        assert adapter.connection_string == "postgresql://test:test@localhost:5432/test"
        assert adapter.db_type == "postgres"
        assert adapter.min_connections == 1
        assert adapter.max_connections == 5
        assert adapter.connection_timeout == 30

    @pytest.mark.asyncio
    @patch(
        "src.agent.adapters.async_database.asyncpg.create_pool", new_callable=AsyncMock
    )
    async def test_async_context_manager_usage(
        self, mock_create_pool, async_database_config
    ):
        """Test using async database adapter as async context manager."""
        # Setup mock
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool

        # Test context manager usage
        async with AsyncDatabaseAdapter(async_database_config) as db:
            assert db.pool is mock_pool
            assert isinstance(db, AsyncDatabaseAdapter)

        # Verify pool was closed
        mock_pool.close.assert_called_once()
        mock_pool.wait_closed.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "src.agent.adapters.async_database.asyncpg.create_pool", new_callable=AsyncMock
    )
    async def test_query_execution_workflow(
        self, mock_create_pool, async_database_config
    ):
        """Test complete workflow of connecting, querying, and disconnecting."""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_connection = AsyncMock()
        mock_pool.acquire = lambda: create_mock_async_context_manager(mock_connection)
        mock_create_pool.return_value = mock_pool

        # Mock query results
        mock_records = [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ]
        mock_connection.fetch.return_value = mock_records

        adapter = AsyncDatabaseAdapter(async_database_config)

        try:
            # Connect
            await adapter.connect()
            assert adapter.pool is mock_pool

            # Execute query
            result = await adapter.execute_query(
                "SELECT id, name, email FROM users WHERE active = $1", {"active": True}
            )

            # Verify results
            assert "data" in result
            assert len(result["data"]) == 2
            assert list(result["data"].columns) == ["id", "name", "email"]
            assert result["data"].iloc[0]["name"] == "Alice"
            assert result["data"].iloc[1]["name"] == "Bob"

        finally:
            # Disconnect
            await adapter.disconnect()
            assert adapter.pool is None

    @pytest.mark.asyncio
    @patch("src.agent.adapters.async_database.asyncpg.create_pool")
    async def test_connection_failure_handling(
        self, mock_create_pool, async_database_config
    ):
        """Test proper handling of connection failures."""
        # Simulate connection failure
        original_error = ConnectionError("Database server unavailable")
        mock_create_pool.side_effect = original_error

        adapter = AsyncDatabaseAdapter(async_database_config)

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await adapter.connect()

        # Verify exception details
        assert exc_info.value.original_exception is original_error
        assert "connection pool" in str(exc_info.value).lower()

        # Verify sensitive data filtering
        sanitized_context = exc_info.value.get_sanitized_context()
        assert sanitized_context["connection_string"] == "[FILTERED]"


def create_mock_async_context_manager(return_value):
    """Helper function to create a properly mocked async context manager."""
    from unittest.mock import MagicMock

    async_cm = MagicMock()

    async def mock_aenter(self):
        return return_value

    async def mock_aexit(self, exc_type, exc_val, exc_tb):
        return False

    async_cm.__aenter__ = mock_aenter
    async_cm.__aexit__ = mock_aexit

    return async_cm
