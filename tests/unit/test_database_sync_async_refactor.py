"""
Test suite for database adapter sync/async refactor to eliminate code duplication.

Following TDD principles:
1. Red: Write failing tests that demonstrate the desired behavior
2. Green: Implement minimal code to make tests pass
3. Refactor: Clean up implementation while keeping tests passing

This test ensures:
- Sync methods are simple wrappers around async methods
- No duplicate implementation logic
- Proper handling of event loops
- Backward compatibility maintained
"""

import asyncio
import pytest
from unittest.mock import Mock, patch
import pandas as pd

from src.agent.adapters.database import BaseDatabaseAdapter
from src.agent.exceptions import DatabaseConnectionException


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


class TestSyncAsyncRefactor:
    """Test that sync methods are simple wrappers around async implementations."""

    @patch("src.agent.adapters.database.BaseDatabaseAdapter.execute_query_async")
    def test_execute_query_sync_calls_async_version(
        self, mock_async_execute, database_adapter
    ):
        """Test that execute_query_sync calls execute_query_async via helper."""
        expected_result = {"data": pd.DataFrame([{"id": 1, "name": "test"}])}
        mock_async_execute.return_value = expected_result

        query = "SELECT * FROM test_table"
        params = {"limit": 10}
        limit = 5
        offset = 0

        # This should work and call the async version internally
        result = database_adapter.execute_query_sync(query, params, limit, offset)

        # Verify the async method was called with correct parameters
        mock_async_execute.assert_called_once_with(query, params, limit, offset)
        assert result == expected_result

    @patch("src.agent.adapters.database.BaseDatabaseAdapter.get_schema_async")
    def test_get_schema_sync_calls_async_version(
        self, mock_async_get_schema, database_adapter
    ):
        """Test that get_schema_sync calls get_schema_async via helper."""
        expected_schema = {"tables": ["test_table"]}
        mock_async_get_schema.return_value = expected_schema

        result = database_adapter.get_schema_sync()

        mock_async_get_schema.assert_called_once()
        assert result == expected_schema

    def test_sync_methods_use_run_async_helper(self, database_adapter):
        """Test that sync methods use a common _run_async helper."""
        # This test ensures we have a helper method to avoid duplicating event loop logic
        assert hasattr(database_adapter, "_run_async")
        assert callable(database_adapter._run_async)

    @pytest.mark.asyncio
    async def test_async_methods_remain_primary_implementation(self, database_adapter):
        """Test that async methods contain the actual implementation logic."""
        # Mock the engine to avoid connection issues
        database_adapter.engine = None

        # These should raise the proper exceptions showing they contain real logic
        with pytest.raises(DatabaseConnectionException):
            await database_adapter.execute_query_async("SELECT 1")

        with pytest.raises(DatabaseConnectionException):
            await database_adapter.get_schema_async()

    def test_no_duplicate_logic_in_sync_methods(self, database_adapter):
        """Test that sync methods don't contain duplicate business logic."""
        import inspect

        # Check that sync methods have minimal lines (just wrapper code)
        sync_execute_source = inspect.getsource(database_adapter.execute_query_sync)
        sync_schema_source = inspect.getsource(database_adapter.get_schema_sync)

        # These methods should be short - exclude method signature and docstring lines
        def get_logic_lines(source):
            lines = source.split("\n")
            logic_lines = []
            in_docstring = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('"""'):
                    in_docstring = not in_docstring
                    continue
                if in_docstring or not stripped:
                    continue
                if stripped.startswith(
                    ("def ", "self,", "query:", "params:", "limit:", "offset:", ") ->")
                ):
                    continue
                logic_lines.append(stripped)
            return logic_lines

        sync_execute_logic = get_logic_lines(sync_execute_source)
        sync_schema_logic = get_logic_lines(sync_schema_source)

        # Should be just calling _run_async with the async method
        assert len(sync_execute_logic) <= 1  # Just the return statement
        assert len(sync_schema_logic) <= 1  # Just the return statement

    def test_run_async_helper_handles_event_loops_correctly(self, database_adapter):
        """Test that _run_async helper handles different event loop scenarios."""

        # Mock an async function to test with
        async def mock_async_func():
            return "test_result"

        # This should work in a sync context (no running loop)
        result = database_adapter._run_async(mock_async_func())
        assert result == "test_result"

    @pytest.mark.asyncio
    async def test_run_async_helper_handles_running_event_loop(self, database_adapter):
        """Test _run_async helper when called from within an async context."""

        async def mock_async_func():
            return "async_test_result"

        # When called from async context, should handle running loop properly
        result = database_adapter._run_async(mock_async_func())
        assert result == "async_test_result"


class TestBackwardCompatibilityAfterRefactor:
    """Test that the public API remains unchanged after refactor."""

    @patch("src.agent.adapters.database.BaseDatabaseAdapter.execute_query_async")
    def test_execute_query_still_works_as_sync_method(
        self, mock_async_execute, database_adapter
    ):
        """Test that execute_query (the main public method) still works synchronously."""
        expected_result = {"data": pd.DataFrame([{"id": 1, "name": "test"}])}
        mock_async_execute.return_value = expected_result

        result = database_adapter.execute_query("SELECT 1")
        assert result == expected_result

    @patch("src.agent.adapters.database.BaseDatabaseAdapter.get_schema_async")
    def test_get_schema_still_works_as_sync_method(
        self, mock_async_get_schema, database_adapter
    ):
        """Test that get_schema (the main public method) still works synchronously."""
        expected_schema = {"tables": ["test"]}
        mock_async_get_schema.return_value = expected_schema

        result = database_adapter.get_schema()
        assert result == expected_schema

    @pytest.mark.asyncio
    async def test_async_methods_still_work(self, database_adapter):
        """Test that async methods continue to work properly."""
        database_adapter.engine = None  # Will cause connection error

        with pytest.raises(DatabaseConnectionException):
            await database_adapter.execute_query_async("SELECT 1")


class TestEventLoopHandling:
    """Test proper handling of various event loop scenarios."""

    def test_sync_method_creates_new_loop_when_none_exists(self, database_adapter):
        """Test sync method behavior when no event loop exists."""
        # Ensure no event loop is running
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                pytest.skip("Event loop is running, can't test no-loop scenario")
        except RuntimeError:
            pass  # No loop exists, which is what we want

        async def mock_coro():
            return "no_loop_result"

        result = database_adapter._run_async(mock_coro())
        assert result == "no_loop_result"

    def test_sync_method_uses_thread_when_loop_running(self, database_adapter):
        """Test that sync method uses thread pool when event loop is running."""

        async def mock_coro():
            return "thread_result"

        # Simulate running in an event loop by mocking the event loop detection
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = Mock()
            mock_loop.is_running.return_value = True
            mock_get_loop.return_value = mock_loop

            with patch("threading.Thread") as mock_thread_class:
                mock_thread = Mock()
                mock_thread_class.return_value = mock_thread

                # Mock the thread to set the result when started
                def set_result():
                    # Simulate the thread running the coroutine
                    pass

                mock_thread.start = Mock(side_effect=set_result)
                mock_thread.join = Mock()

                database_adapter._run_async(mock_coro())
                # The result should be "thread_result" from the async coroutine
                # Since we're mocking, we need to check that Thread was used
                mock_thread_class.assert_called_once()
                mock_thread.start.assert_called_once()
                mock_thread.join.assert_called_once()


class TestCodeDuplicationElimination:
    """Test that code duplication has been eliminated."""

    def test_no_duplicate_connection_handling_code(self, database_adapter):
        """Test that connection handling logic exists only in async methods."""
        import inspect

        # Get source code
        sync_execute_source = inspect.getsource(database_adapter.execute_query_sync)
        async_execute_source = inspect.getsource(database_adapter.execute_query_async)

        # Sync method should not contain connection handling logic
        assert (
            "engine" not in sync_execute_source or "self.engine" in sync_execute_source
        )  # Only property access
        assert (
            "DatabaseConnectionException" not in sync_execute_source
        )  # No exception handling

        # Async method should contain the real logic
        assert "engine" in async_execute_source
        assert "DatabaseConnectionException" in async_execute_source

    def test_no_duplicate_query_execution_logic(self, database_adapter):
        """Test that query execution logic exists only in async methods."""
        import inspect

        sync_execute_source = inspect.getsource(database_adapter.execute_query_sync)
        async_execute_source = inspect.getsource(database_adapter.execute_query_async)

        # Sync should not contain SQL execution logic
        assert "session" not in sync_execute_source
        assert "text(" not in sync_execute_source
        assert "execute(" not in sync_execute_source

        # Async should contain the real SQL logic
        assert "session" in async_execute_source
        assert "text(" in async_execute_source

    def test_no_duplicate_error_handling_patterns(self, database_adapter):
        """Test that error handling patterns exist only in async methods."""
        import inspect

        sync_schema_source = inspect.getsource(database_adapter.get_schema_sync)
        async_schema_source = inspect.getsource(database_adapter.get_schema_async)

        # Count exception handling patterns
        sync_try_count = sync_schema_source.count("try:")
        sync_except_count = sync_schema_source.count("except")

        async_try_count = async_schema_source.count("try:")
        async_except_count = async_schema_source.count("except")

        # Sync methods should have minimal or no exception handling
        assert sync_try_count <= 1  # At most one try block for the helper
        assert sync_except_count <= 1

        # Async methods should have the real exception handling
        assert async_try_count >= 1
        assert async_except_count >= 1
