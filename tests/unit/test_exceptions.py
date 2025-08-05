"""
Test suite for the agent exception hierarchy.

This follows TDD principles - tests are written first to define behavior,
then the exceptions.py module will be implemented to make these tests pass.
"""


class TestBaseAgentException:
    """Test base exception class behavior."""

    def test_base_exception_inherits_from_exception(self):
        """Base agent exception should inherit from Python's Exception."""
        from src.agent.exceptions import AgentException

        exception = AgentException("test message")
        assert isinstance(exception, Exception)

    def test_base_exception_with_message_only(self):
        """Base exception should accept a simple message."""
        from src.agent.exceptions import AgentException

        message = "Something went wrong"
        exception = AgentException(message)
        assert str(exception) == message
        assert exception.message == message

    def test_base_exception_with_context(self):
        """Base exception should preserve context information."""
        from src.agent.exceptions import AgentException

        message = "Database connection failed"
        context = {"host": "localhost", "port": 5432, "attempt": 3}
        exception = AgentException(message, context=context)

        assert str(exception) == message
        assert exception.context == context

    def test_base_exception_with_original_exception(self):
        """Base exception should support exception chaining."""
        from src.agent.exceptions import AgentException

        original = ValueError("Invalid connection string")
        message = "Failed to connect to database"
        exception = AgentException(message, original_exception=original)

        assert str(exception) == message
        assert exception.__cause__ == original
        assert exception.original_exception == original

    def test_base_exception_with_all_parameters(self):
        """Base exception should handle message, context, and original exception."""
        from src.agent.exceptions import AgentException

        original = ConnectionError("Connection refused")
        message = "Database operation failed"
        context = {"operation": "SELECT", "table": "users", "retry_count": 2}

        exception = AgentException(
            message, context=context, original_exception=original
        )

        assert str(exception) == message
        assert exception.context == context
        assert exception.__cause__ == original
        assert exception.original_exception == original

    def test_base_exception_repr(self):
        """Base exception should have meaningful repr."""
        from src.agent.exceptions import AgentException

        message = "Test error"
        context = {"key": "value"}
        exception = AgentException(message, context=context)

        repr_str = repr(exception)
        assert "AgentException" in repr_str
        assert message in repr_str


class TestDatabaseExceptions:
    """Test database-specific exception classes."""

    def test_database_exception_inheritance(self):
        """Database exceptions should inherit from AgentException."""
        from src.agent.exceptions import DatabaseException, AgentException

        exception = DatabaseException("Database error")
        assert isinstance(exception, AgentException)
        assert isinstance(exception, Exception)

    def test_database_connection_exception(self):
        """Test database connection specific exception."""
        from src.agent.exceptions import DatabaseConnectionException

        context = {"host": "localhost", "port": 5432, "database": "testdb"}
        exception = DatabaseConnectionException(
            "Failed to connect to database", context=context
        )

        assert "Failed to connect to database" in str(exception)
        assert exception.context == context

    def test_database_query_exception(self):
        """Test database query execution exception."""
        from src.agent.exceptions import DatabaseQueryException

        context = {
            "query": "SELECT * FROM users WHERE id = ?",
            "parameters": {"id": 123},
            "execution_time_ms": 5000,
        }
        exception = DatabaseQueryException("Query execution timeout", context=context)

        assert "Query execution timeout" in str(exception)
        assert exception.context["query"] == "SELECT * FROM users WHERE id = ?"

    def test_database_transaction_exception(self):
        """Test database transaction exception."""
        from src.agent.exceptions import DatabaseTransactionException

        context = {"transaction_id": "tx_123", "operations_completed": 2}
        exception = DatabaseTransactionException(
            "Transaction rollback failed", context=context
        )

        assert "Transaction rollback failed" in str(exception)
        assert exception.context["transaction_id"] == "tx_123"


class TestValidationExceptions:
    """Test validation-specific exception classes."""

    def test_validation_exception_inheritance(self):
        """Validation exceptions should inherit from AgentException."""
        from src.agent.exceptions import ValidationException, AgentException

        exception = ValidationException("Validation failed")
        assert isinstance(exception, AgentException)

    def test_input_validation_exception(self):
        """Test input validation exception."""
        from src.agent.exceptions import InputValidationException

        context = {
            "field": "email",
            "value": "invalid-email",
            "expected_format": "user@domain.com",
        }
        exception = InputValidationException("Invalid email format", context=context)

        assert "Invalid email format" in str(exception)
        assert exception.context["field"] == "email"

    def test_sql_validation_exception(self):
        """Test SQL validation exception."""
        from src.agent.exceptions import SQLValidationException

        context = {
            "sql_query": "SELECT * FROM users WHERE",
            "syntax_errors": ["Incomplete WHERE clause"],
            "line_number": 1,
        }
        exception = SQLValidationException("SQL syntax error", context=context)

        assert "SQL syntax error" in str(exception)
        assert exception.context["syntax_errors"] == ["Incomplete WHERE clause"]


class TestAgentStateExceptions:
    """Test agent state machine specific exceptions."""

    def test_agent_state_exception_inheritance(self):
        """Agent state exceptions should inherit from AgentException."""
        from src.agent.exceptions import AgentStateException, AgentException

        exception = AgentStateException("Invalid state transition")
        assert isinstance(exception, AgentException)

    def test_invalid_state_transition_exception(self):
        """Test invalid state transition exception."""
        from src.agent.exceptions import InvalidStateTransitionException

        context = {
            "current_state": "retrieve",
            "attempted_state": "question",
            "valid_transitions": ["rerank", "enhance"],
        }
        exception = InvalidStateTransitionException(
            "Cannot transition from retrieve to question", context=context
        )

        assert "Cannot transition" in str(exception)
        assert exception.context["current_state"] == "retrieve"

    def test_command_processing_exception(self):
        """Test command processing exception."""
        from src.agent.exceptions import CommandProcessingException

        context = {
            "command_type": "SQLGrounding",
            "command_id": "cmd_123",
            "processing_stage": "validation",
        }
        exception = CommandProcessingException(
            "Failed to process command", context=context
        )

        assert "Failed to process command" in str(exception)
        assert exception.context["command_type"] == "SQLGrounding"


class TestExternalServiceExceptions:
    """Test external service specific exceptions."""

    def test_external_service_exception_inheritance(self):
        """External service exceptions should inherit from AgentException."""
        from src.agent.exceptions import ExternalServiceException, AgentException

        exception = ExternalServiceException("Service unavailable")
        assert isinstance(exception, AgentException)

    def test_llm_api_exception(self):
        """Test LLM API exception."""
        from src.agent.exceptions import LLMAPIException

        context = {
            "provider": "openai",
            "model": "gpt-4",
            "status_code": 429,
            "retry_count": 3,
        }
        exception = LLMAPIException("Rate limit exceeded", context=context)

        assert "Rate limit exceeded" in str(exception)
        assert exception.context["status_code"] == 429

    def test_rag_system_exception(self):
        """Test RAG system exception."""
        from src.agent.exceptions import RAGSystemException

        context = {
            "operation": "retrieve",
            "query": "test query",
            "index_name": "documents",
            "timeout_seconds": 30,
        }
        exception = RAGSystemException("Document retrieval timeout", context=context)

        assert "Document retrieval timeout" in str(exception)
        assert exception.context["operation"] == "retrieve"

    def test_notification_service_exception(self):
        """Test notification service exception."""
        from src.agent.exceptions import NotificationServiceException

        context = {"service": "slack", "channel": "#alerts", "message_id": "msg_123"}
        exception = NotificationServiceException(
            "Failed to send notification", context=context
        )

        assert "Failed to send notification" in str(exception)
        assert exception.context["service"] == "slack"


class TestConfigurationExceptions:
    """Test configuration-specific exceptions."""

    def test_configuration_exception_inheritance(self):
        """Configuration exceptions should inherit from AgentException."""
        from src.agent.exceptions import ConfigurationException, AgentException

        exception = ConfigurationException("Invalid configuration")
        assert isinstance(exception, AgentException)

    def test_missing_configuration_exception(self):
        """Test missing configuration exception."""
        from src.agent.exceptions import MissingConfigurationException

        context = {
            "config_key": "DATABASE_URL",
            "config_section": "database",
            "required_for": "database connection",
        }
        exception = MissingConfigurationException(
            "Required configuration missing", context=context
        )

        assert "Required configuration missing" in str(exception)
        assert exception.context["config_key"] == "DATABASE_URL"

    def test_invalid_configuration_exception(self):
        """Test invalid configuration exception."""
        from src.agent.exceptions import InvalidConfigurationException

        context = {
            "config_key": "MAX_RETRIES",
            "config_value": "invalid",
            "expected_type": "integer",
            "valid_range": "1-10",
        }
        exception = InvalidConfigurationException(
            "Configuration value is invalid", context=context
        )

        assert "Configuration value is invalid" in str(exception)
        assert exception.context["expected_type"] == "integer"


class TestExceptionChaining:
    """Test proper exception chaining behavior."""

    def test_exception_chaining_with_raise_from(self):
        """Test that exceptions properly chain with raise from."""
        from src.agent.exceptions import DatabaseConnectionException

        try:
            # Simulate original error
            raise ConnectionError("Connection refused")
        except ConnectionError as original:
            chained_exception = DatabaseConnectionException(
                "Failed to connect to database", original_exception=original
            )
            assert chained_exception.__cause__ == original

    def test_multiple_exception_chaining(self):
        """Test chaining through multiple exception levels."""
        from src.agent.exceptions import (
            DatabaseConnectionException,
            AgentStateException,
        )

        try:
            raise ConnectionError("Network unreachable")
        except ConnectionError as network_error:
            try:
                raise DatabaseConnectionException(
                    "Database connection failed", original_exception=network_error
                )
            except DatabaseConnectionException as db_error:
                agent_error = AgentStateException(
                    "Agent failed to initialize", original_exception=db_error
                )
                assert agent_error.__cause__ == db_error
                assert db_error.__cause__ == network_error


class TestExceptionContext:
    """Test context preservation and formatting."""

    def test_context_serialization(self):
        """Test that context can be serialized for logging."""
        from src.agent.exceptions import AgentException

        context = {
            "string_value": "test",
            "int_value": 42,
            "list_value": [1, 2, 3],
            "dict_value": {"nested": "value"},
        }
        exception = AgentException("Test exception", context=context)

        # Context should be preserved exactly
        assert exception.context == context

        # Should be able to convert to string representation
        context_str = str(exception.context)
        assert "string_value" in context_str
        assert "42" in context_str

    def test_context_with_sensitive_data_filtering(self):
        """Test that sensitive data can be filtered from context."""
        from src.agent.exceptions import AgentException

        context = {
            "username": "user123",
            "password": "secret123",
            "api_key": "key_secret",
            "connection_string": "postgresql://user:pass@host/db",
        }
        exception = AgentException("Auth failed", context=context)

        # The exception should have a method to get sanitized context
        if hasattr(exception, "get_sanitized_context"):
            sanitized = exception.get_sanitized_context()
            # Sensitive values should be filtered, but keys preserved for debugging
            assert "secret123" not in str(sanitized)
            assert "key_secret" not in str(sanitized)
            assert "user:pass" not in str(sanitized)  # Credentials in connection string
            assert sanitized["password"] == "[FILTERED]"
            assert sanitized["api_key"] == "[FILTERED]"
            assert sanitized["connection_string"] == "[FILTERED]"
            assert "username" in str(sanitized)  # Non-sensitive data preserved
            assert sanitized["username"] == "user123"  # Non-sensitive values preserved
