"""
Examples demonstrating how to use the new exception hierarchy
to replace problematic patterns in the codebase.

This shows the transformation from:
- Generic Exception catching with silent failures
- Return None on errors
- Lost context information

To:
- Specific exception types
- Proper error chaining
- Context preservation
- Meaningful error messages
"""

import pytest
from unittest.mock import Mock
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from src.agent.exceptions import (
    DatabaseConnectionException,
    DatabaseQueryException,
    DatabaseTransactionException,
)


class TestDatabaseAdapterRefactoring:
    """
    Demonstrate how to refactor database.py patterns using new exceptions.

    These examples show the before/after of replacing problematic patterns:
    1. Generic Exception catching -> Specific exception types
    2. return None -> raise meaningful exceptions with context
    3. Lost error context -> preserved context and chaining
    """

    def test_connection_error_handling_before_and_after(self):
        """
        Show how connection error handling improves with specific exceptions.

        BEFORE: Generic exception catching, return None, lost context
        AFTER: Specific exception, context preservation, proper chaining
        """

        # BEFORE - problematic pattern (from database.py line 78)
        def old_get_connection(connection_string):
            try:
                # Simulated SQLAlchemy engine creation
                if "invalid" in connection_string:
                    raise OperationalError("Connection failed", None, None)
                return Mock()  # Success case
            except Exception:
                # PROBLEMS:
                # 1. Generic Exception catch loses specific error type
                # 2. return None makes it hard to distinguish errors
                # 3. Original exception context is lost
                # 4. Calling code doesn't know what went wrong
                return None

        # AFTER - improved pattern with specific exceptions
        def new_get_connection(connection_string):
            try:
                # Simulated SQLAlchemy engine creation
                if "invalid" in connection_string:
                    raise OperationalError("Connection failed", None, None)
                return Mock()  # Success case
            except OperationalError as e:
                # IMPROVEMENTS:
                # 1. Specific exception type for database connection issues
                # 2. Context preserved for debugging
                # 3. Original exception chained for full error trace
                # 4. Clear error message with actionable information
                context = {
                    "connection_string": connection_string,
                    "database_type": "postgresql",
                    "operation": "create_engine",
                }
                raise DatabaseConnectionException(
                    f"Failed to create database engine: {str(e)}",
                    context=context,
                    original_exception=e,
                ) from e

        # Test old pattern - silent failure
        result = old_get_connection("invalid://connection")
        assert result is None  # Lost information about what went wrong

        # Test new pattern - specific exception with context
        with pytest.raises(DatabaseConnectionException) as exc_info:
            new_get_connection("invalid://connection")

        exception = exc_info.value
        assert "Failed to create database engine" in str(exception)
        assert exception.context["operation"] == "create_engine"
        assert exception.context["database_type"] == "postgresql"
        assert isinstance(exception.__cause__, OperationalError)

    def test_query_execution_error_handling_refactoring(self):
        """
        Show how query execution error handling improves.

        Demonstrates transformation from generic exception handling
        to specific, context-aware exception raising.
        """

        # BEFORE - problematic pattern (from database.py line 122)
        def old_execute_query(engine, sql_statement, params=None):
            try:
                # Simulated pd.read_sql_query that fails
                if "invalid_table" in sql_statement:
                    raise SQLAlchemyError("table does not exist")
                return {"data": Mock()}
            except Exception:
                # PROBLEMS: Same as above - generic catch, return None, lost context
                return None

        # AFTER - improved pattern
        def new_execute_query(engine, sql_statement, params=None):
            try:
                # Simulated pd.read_sql_query that fails
                if "invalid_table" in sql_statement:
                    raise SQLAlchemyError("table does not exist")
                return {"data": Mock()}
            except SQLAlchemyError as e:
                context = {
                    "sql_statement": sql_statement,
                    "parameters": params,
                    "engine_type": str(type(engine).__name__),
                    "operation": "execute_query",
                }
                raise DatabaseQueryException(
                    f"Query execution failed: {str(e)}",
                    context=context,
                    original_exception=e,
                ) from e

        # Test old pattern
        result = old_execute_query(Mock(), "SELECT * FROM invalid_table")
        assert result is None  # No information about the failure

        # Test new pattern
        with pytest.raises(DatabaseQueryException) as exc_info:
            new_execute_query(Mock(), "SELECT * FROM invalid_table", {"id": 1})

        exception = exc_info.value
        assert "Query execution failed" in str(exception)
        assert exception.context["sql_statement"] == "SELECT * FROM invalid_table"
        assert exception.context["parameters"] == {"id": 1}
        assert exception.context["operation"] == "execute_query"
        assert isinstance(exception.__cause__, SQLAlchemyError)

    def test_transaction_error_handling_refactoring(self):
        """
        Show how transaction error handling can be improved.

        This demonstrates handling of transaction-related errors
        with proper context and specific exception types.
        """

        # AFTER - improved transaction handling (new pattern)
        def execute_transaction(engine, operations):
            transaction_id = "tx_123"
            operations_completed = 0

            try:
                # Simulate transaction operations
                for i, operation in enumerate(operations):
                    if operation == "fail":
                        raise SQLAlchemyError("Operation failed")
                    operations_completed = i + 1
                return True
            except SQLAlchemyError as e:
                context = {
                    "transaction_id": transaction_id,
                    "operations_completed": operations_completed,
                    "total_operations": len(operations),
                    "failed_operation": operations[operations_completed]
                    if operations_completed < len(operations)
                    else None,
                }
                raise DatabaseTransactionException(
                    f"Transaction {transaction_id} failed after {operations_completed} operations",
                    context=context,
                    original_exception=e,
                ) from e

        # Test successful transaction
        result = execute_transaction(Mock(), ["op1", "op2", "op3"])
        assert result is True

        # Test failed transaction with context preservation
        with pytest.raises(DatabaseTransactionException) as exc_info:
            execute_transaction(Mock(), ["op1", "fail", "op3"])

        exception = exc_info.value
        assert "Transaction tx_123 failed after 1 operations" in str(exception)
        assert exception.context["operations_completed"] == 1
        assert exception.context["failed_operation"] == "fail"
        assert exception.context["total_operations"] == 3
        assert isinstance(exception.__cause__, SQLAlchemyError)

    def test_error_context_sanitization_in_database_operations(self):
        """
        Demonstrate how sensitive database information is properly filtered.

        Shows that connection strings and other sensitive data
        are filtered when logging exceptions.
        """

        def connect_with_credentials(connection_string):
            try:
                # Simulate connection failure
                raise OperationalError("Authentication failed", None, None)
            except OperationalError as e:
                context = {
                    "connection_string": connection_string,
                    "retry_count": 3,
                    "timeout_seconds": 30,
                    "database_name": "production_db",
                }
                raise DatabaseConnectionException(
                    "Failed to authenticate with database",
                    context=context,
                    original_exception=e,
                ) from e

        # Test with sensitive connection string
        sensitive_connection = (
            "postgresql://user:secret_password@db.example.com:5432/prod"
        )

        with pytest.raises(DatabaseConnectionException) as exc_info:
            connect_with_credentials(sensitive_connection)

        exception = exc_info.value

        # Regular context contains sensitive data
        assert "secret_password" in exception.context["connection_string"]

        # Sanitized context filters sensitive data
        sanitized = exception.get_sanitized_context()
        assert sanitized["connection_string"] == "[FILTERED]"
        assert sanitized["database_name"] == "production_db"  # Non-sensitive preserved
        assert sanitized["retry_count"] == 3  # Non-sensitive preserved


class TestErrorHandlingPatterns:
    """
    Demonstrate recommended patterns for error handling with the new exceptions.
    """

    def test_proper_exception_handling_in_service_layer(self):
        """
        Show how service layer should handle and potentially transform exceptions.
        """

        def database_operation():
            """Simulate a database operation that fails."""
            context = {"query": "SELECT * FROM users", "timeout": 30}
            raise DatabaseQueryException(
                "Query timeout exceeded",
                context=context,
                original_exception=TimeoutError("30 second timeout"),
            )

        def service_operation():
            """Service layer that handles database exceptions appropriately."""
            try:
                return database_operation()
            except DatabaseQueryException as e:
                # Service layer can:
                # 1. Log the error with sanitized context
                # 2. Add service-level context
                # 3. Re-raise or transform to service-level exception
                # 4. Return appropriate response to client

                # For this example, we'll add service context and re-raise
                service_context = {
                    "service": "user_service",
                    "operation": "get_users",
                    "user_id": "current_user",
                }

                # Combine contexts (service + original database context)
                combined_context = {**e.context, **service_context}

                raise DatabaseQueryException(
                    f"User service operation failed: {str(e)}",
                    context=combined_context,
                    original_exception=e,
                ) from e

        with pytest.raises(DatabaseQueryException) as exc_info:
            service_operation()

        exception = exc_info.value
        assert "User service operation failed" in str(exception)
        assert exception.context["service"] == "user_service"
        assert (
            exception.context["query"] == "SELECT * FROM users"
        )  # Original context preserved
        assert exception.context["timeout"] == 30  # Original context preserved

    def test_exception_handling_best_practices(self):
        """
        Demonstrate best practices for exception handling.
        """

        def operation_with_multiple_failure_points():
            """
            Operation that can fail at multiple points with different exception types.
            Shows how to handle each failure type appropriately.
            """

            # Validation phase
            data = {"email": "invalid-email"}
            if "@" not in data["email"]:
                context = {
                    "field": "email",
                    "value": data["email"],
                    "validation_rule": "must_contain_at_symbol",
                }
                from src.agent.exceptions import InputValidationException

                raise InputValidationException(
                    "Email validation failed", context=context
                )

            # Database phase
            try:
                # Simulate database operation
                raise SQLAlchemyError("Connection lost")
            except SQLAlchemyError as e:
                context = {
                    "operation": "user_creation",
                    "data": data,
                    "connection_pool_size": 10,
                }
                raise DatabaseQueryException(
                    "Failed to create user record",
                    context=context,
                    original_exception=e,
                ) from e

        # Test validation failure
        with pytest.raises(Exception) as exc_info:
            operation_with_multiple_failure_points()

        # Should be InputValidationException due to email validation
        from src.agent.exceptions import InputValidationException

        assert isinstance(exc_info.value, InputValidationException)
        assert exc_info.value.context["field"] == "email"
