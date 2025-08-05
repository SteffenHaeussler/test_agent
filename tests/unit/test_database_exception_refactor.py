"""
Test suite for database.py exception handling refactor.

This test suite follows TDD principles:
1. Tests the current behavior (baseline) - will pass initially
2. Tests the new behavior with specific exceptions - will fail initially
3. Tests proper context preservation
4. Tests exception chaining
5. Tests error propagation (no more silent failures)

These tests are designed to fail with the current implementation
and pass after refactoring to use the new exception hierarchy.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
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


# Baseline tests removed - database.py has been refactored to use new exceptions


class TestNewExceptionBehavior:
    """Test the new behavior with specific database exceptions - these will initially fail."""

    @patch("src.agent.adapters.database.create_engine")
    def test__get_connection_raises_database_connection_exception(
        self, mock_create_engine, database_adapter
    ):
        """Test that _get_connection raises DatabaseConnectionException with proper context."""
        original_error = OperationalError("Connection timeout", None, None)
        mock_create_engine.side_effect = original_error

        with pytest.raises(DatabaseConnectionException) as exc_info:
            database_adapter._get_connection()

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

    @patch("src.agent.adapters.database.pd.read_sql_query")
    def test_execute_query_raises_database_query_exception(
        self, mock_read_sql_query, database_adapter
    ):
        """Test that execute_query raises DatabaseQueryException with proper context."""
        database_adapter.engine = Mock()
        original_error = SQLAlchemyError("Invalid SQL syntax")
        mock_read_sql_query.side_effect = original_error

        sql_statement = "SELECT * FROM nonexistent_table"
        params = {"limit": 10}

        with pytest.raises(DatabaseQueryException) as exc_info:
            database_adapter.execute_query(sql_statement, params)

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

    @patch("src.agent.adapters.database.MetaData")
    def test_get_schema_raises_database_query_exception(
        self, mock_metadata_class, database_adapter
    ):
        """Test that get_schema raises DatabaseQueryException with proper context."""
        database_adapter.engine = Mock()
        mock_metadata = Mock()
        mock_metadata_class.return_value = mock_metadata
        original_error = OperationalError("Permission denied", None, None)
        mock_metadata.reflect.side_effect = original_error

        with pytest.raises(DatabaseQueryException) as exc_info:
            database_adapter.get_schema()

        # Test exception chaining
        assert exc_info.value.original_exception is original_error
        assert exc_info.value.__cause__ is original_error

        # Test context preservation
        context = exc_info.value.context
        assert "operation" in context
        assert context["operation"] == "schema_reflection"

        # Test message quality
        assert "schema" in str(exc_info.value).lower()

    @patch("src.agent.adapters.database.text")
    def test_insert_batch_raises_database_transaction_exception(
        self, mock_text, database_adapter
    ):
        """Test that insert_batch raises DatabaseTransactionException with proper context."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn
        original_error = IntegrityError("Duplicate key", None, None)
        mock_conn.execute.side_effect = original_error
        database_adapter.engine = mock_engine

        data = [{"id": 1, "name": "test"}, {"id": 2, "name": "test2"}]
        table_name = "test_table"

        with pytest.raises(DatabaseTransactionException) as exc_info:
            database_adapter.insert_batch(table_name, data)

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

    @patch("src.agent.adapters.database.create_engine")
    def test_connection_exception_preserves_sensitive_data_filtering(
        self, mock_create_engine, database_config
    ):
        """Test that connection exceptions filter sensitive data in context."""
        # Use a connection string with credentials
        database_config["connection_string"] = (
            "postgresql://user:secret123@localhost:5432/db"
        )
        database_adapter = BaseDatabaseAdapter(database_config)

        mock_create_engine.side_effect = Exception("Connection failed")

        with pytest.raises(DatabaseConnectionException) as exc_info:
            database_adapter._get_connection()

        # Test that sensitive data is filtered
        sanitized_context = exc_info.value.get_sanitized_context()
        assert sanitized_context["connection_string"] == "[FILTERED]"

        # But original context still contains real data for debugging
        assert "secret123" in exc_info.value.context["connection_string"]

    @patch("src.agent.adapters.database.pd.read_sql_query")
    def test_query_exception_preserves_execution_context(
        self, mock_read_sql_query, database_adapter
    ):
        """Test that query exceptions preserve execution timing and query details."""
        database_adapter.engine = Mock()
        mock_read_sql_query.side_effect = Exception("Query timeout")

        sql_statement = "SELECT * FROM large_table WHERE complex_condition = ?"
        params = {"complex_condition": "value"}

        with pytest.raises(DatabaseQueryException) as exc_info:
            database_adapter.execute_query(sql_statement, params)

        context = exc_info.value.context
        assert context["query"] == sql_statement
        assert context["parameters"] == params
        # These would be added in the actual implementation
        assert "db_type" in context


class TestExceptionChaining:
    """Test proper exception chaining from original exceptions."""

    @patch("src.agent.adapters.database.create_engine")
    def test_connection_exception_chains_original_sqlalchemy_error(
        self, mock_create_engine, database_adapter
    ):
        """Test that DatabaseConnectionException properly chains SQLAlchemy errors."""
        original_error = OperationalError("Connection refused", None, None)
        mock_create_engine.side_effect = original_error

        with pytest.raises(DatabaseConnectionException) as exc_info:
            database_adapter._get_connection()

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

    @patch("src.agent.adapters.database.pd.read_sql_query")
    def test_query_exception_chains_pandas_error(
        self, mock_read_sql_query, database_adapter
    ):
        """Test that DatabaseQueryException properly chains pandas/SQLAlchemy errors."""
        database_adapter.engine = Mock()
        original_error = ValueError("Invalid SQL syntax")
        mock_read_sql_query.side_effect = original_error

        with pytest.raises(DatabaseQueryException) as exc_info:
            database_adapter.execute_query("SELECT invalid syntax")

        assert exc_info.value.__cause__ is original_error
        assert exc_info.value.original_exception is original_error


class TestErrorPropagation:
    """Test that errors are properly propagated instead of silently failing."""

    def test_execute_query_with_no_engine_raises_exception(self, database_adapter):
        """Test that execute_query raises exception when engine is None instead of returning None."""
        # Engine is None by default
        assert database_adapter.engine is None

        with pytest.raises(DatabaseConnectionException) as exc_info:
            database_adapter.execute_query("SELECT * FROM test")

        context = exc_info.value.context
        assert "engine_available" in context
        assert context["engine_available"] is False
        assert "operation" in context
        assert context["operation"] == "execute_query"

    def test_get_schema_with_no_engine_raises_exception(self, database_adapter):
        """Test that get_schema raises exception when engine is None instead of returning None."""
        assert database_adapter.engine is None

        with pytest.raises(DatabaseConnectionException) as exc_info:
            database_adapter.get_schema()

        context = exc_info.value.context
        assert "engine_available" in context
        assert context["engine_available"] is False
        assert "operation" in context
        assert context["operation"] == "get_schema"

    def test_insert_batch_with_no_engine_raises_exception(self, database_adapter):
        """Test that insert_batch raises exception when engine is None instead of returning False."""
        assert database_adapter.engine is None

        data = [{"id": 1, "name": "test"}]

        with pytest.raises(DatabaseConnectionException) as exc_info:
            database_adapter.insert_batch("test_table", data)

        context = exc_info.value.context
        assert "engine_available" in context
        assert context["engine_available"] is False
        assert "operation" in context
        assert context["operation"] == "insert_batch"


class TestSpecificExceptionTypes:
    """Test that specific exception types are raised for different error conditions."""

    @patch("src.agent.adapters.database.create_engine")
    def test_connection_timeout_raises_connection_exception(
        self, mock_create_engine, database_adapter
    ):
        """Test that connection timeouts raise DatabaseConnectionException."""
        mock_create_engine.side_effect = OperationalError(
            "Connection timeout", None, None
        )

        with pytest.raises(DatabaseConnectionException):
            database_adapter._get_connection()

    @patch("src.agent.adapters.database.pd.read_sql_query")
    def test_sql_syntax_error_raises_query_exception(
        self, mock_read_sql_query, database_adapter
    ):
        """Test that SQL syntax errors raise DatabaseQueryException."""
        database_adapter.engine = Mock()
        mock_read_sql_query.side_effect = SQLAlchemyError("Syntax error")

        with pytest.raises(DatabaseQueryException):
            database_adapter.execute_query("SELECT * FROM")

    @patch("src.agent.adapters.database.text")
    def test_integrity_constraint_raises_transaction_exception(
        self, mock_text, database_adapter
    ):
        """Test that integrity constraint violations raise DatabaseTransactionException."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.side_effect = IntegrityError("Duplicate key", None, None)
        database_adapter.engine = mock_engine

        data = [{"id": 1, "name": "test"}]

        with pytest.raises(DatabaseTransactionException):
            database_adapter.insert_batch("test_table", data)


class TestBackwardCompatibilityDuringTransition:
    """Test behavior during the transition period - these might be temporary tests."""

    def test_connect_method_behavior_unchanged(self, database_adapter):
        """Test that connect() method behavior is preserved during refactor."""
        # This test ensures connect() still works as expected
        with patch.object(database_adapter, "_get_connection") as mock_get_conn:
            mock_engine = Mock()
            mock_get_conn.return_value = mock_engine

            database_adapter.connect()

            assert database_adapter.engine == mock_engine
            mock_get_conn.assert_called_once()

    def test_disconnect_method_behavior_unchanged(self, database_adapter):
        """Test that disconnect() method behavior is preserved during refactor."""
        mock_engine = Mock()
        database_adapter.engine = mock_engine

        database_adapter.disconnect()

        mock_engine.dispose.assert_called_once()
        assert database_adapter.engine is None
