# Exception Handling Guide

This guide demonstrates how to use the comprehensive exception hierarchy in the agentic AI framework and how to migrate from problematic patterns.

## Overview

The exception hierarchy provides:
- **Specific exception types** for different error conditions
- **Context preservation** for debugging and monitoring
- **Exception chaining** to maintain error history
- **Sensitive data filtering** for safe logging
- **Clear inheritance structure** for easy categorization

## Exception Hierarchy

```
AgentException (base)
├── DatabaseException
│   ├── DatabaseConnectionException
│   ├── DatabaseQueryException
│   └── DatabaseTransactionException
├── ValidationException
│   ├── InputValidationException
│   └── SQLValidationException
├── AgentStateException
│   ├── InvalidStateTransitionException
│   └── CommandProcessingException
├── ExternalServiceException
│   ├── LLMAPIException
│   ├── RAGSystemException
│   └── NotificationServiceException
└── ConfigurationException
    ├── MissingConfigurationException
    └── InvalidConfigurationException
```

## Basic Usage

### Creating Exceptions with Context

```python
from src.agent.exceptions import DatabaseConnectionException

# Simple exception
raise DatabaseConnectionException("Failed to connect to database")

# Exception with context
context = {
    "host": "localhost",
    "port": 5432,
    "database": "production",
    "retry_count": 3
}
raise DatabaseConnectionException(
    "Database connection timeout after 3 attempts",
    context=context
)

# Exception with chaining
try:
    # Some operation that fails
    raise ConnectionError("Network unreachable")
except ConnectionError as original:
    context = {"operation": "database_connect", "timeout": 30}
    raise DatabaseConnectionException(
        "Failed to establish database connection",
        context=context,
        original_exception=original
    ) from original
```

### Handling Exceptions

```python
from src.agent.exceptions import DatabaseException, DatabaseConnectionException

try:
    # Database operation
    perform_database_operation()
except DatabaseConnectionException as e:
    # Handle specific connection issues
    logger.error(f"Connection failed: {e}")
    logger.debug(f"Context: {e.get_sanitized_context()}")
    # Maybe retry connection
except DatabaseException as e:
    # Handle any database-related issue
    logger.error(f"Database error: {e}")
    # Maybe switch to fallback data source
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}")
    raise
```

## Migration from Problematic Patterns

### BEFORE: Generic Exception Handling

```python
# ❌ Problematic pattern from database.py
def execute_query(self, sql_statement: str) -> Optional[Dict[str, pd.DataFrame]]:
    try:
        df = pd.read_sql_query(sql=text(sql_statement), con=self.engine)
        return {"data": df}
    except Exception as e:  # Generic catch-all
        logger.error(f"Error executing query: {e}")  # Lost context
        return None  # Silent failure, hard to debug
```

### AFTER: Specific Exception Handling

```python
# ✅ Improved pattern with specific exceptions
from src.agent.exceptions import DatabaseQueryException

def execute_query(self, sql_statement: str) -> Dict[str, pd.DataFrame]:
    if not self.engine:
        context = {"sql_statement": sql_statement, "engine_status": "not_connected"}
        raise DatabaseConnectionException(
            "Cannot execute query: database engine not connected",
            context=context
        )

    try:
        df = pd.read_sql_query(sql=text(sql_statement), con=self.engine)
        return {"data": df}
    except (SQLAlchemyError, DatabaseError) as e:
        context = {
            "sql_statement": sql_statement,
            "engine_type": str(type(self.engine).__name__),
            "database_url": self.connection_string,  # Will be filtered in logs
            "operation": "read_sql_query"
        }
        raise DatabaseQueryException(
            f"Query execution failed: {str(e)}",
            context=context,
            original_exception=e
        ) from e
```

### BEFORE: Silent Failures with None Returns

```python
# ❌ Problematic pattern
def get_schema(self) -> Dict[str, Any]:
    try:
        metadata.reflect(bind=self.engine)
        return metadata
    except Exception as e:
        logger.error(f"Error reflecting metadata: {e}")
        return None  # Caller can't distinguish between "no schema" and "error"
```

### AFTER: Explicit Exception Raising

```python
# ✅ Improved pattern
from src.agent.exceptions import DatabaseException

def get_schema(self) -> Dict[str, Any]:
    if not self.engine:
        raise DatabaseConnectionException(
            "Cannot retrieve schema: database engine not connected",
            context={"operation": "get_schema"}
        )

    try:
        metadata = MetaData()
        metadata.reflect(bind=self.engine)
        return metadata
    except SQLAlchemyError as e:
        context = {
            "operation": "reflect_metadata",
            "engine_type": str(type(self.engine).__name__),
            "database_url": self.connection_string
        }
        raise DatabaseException(
            f"Failed to retrieve database schema: {str(e)}",
            context=context,
            original_exception=e
        ) from e
```

## Exception Types and When to Use Them

### Database Exceptions

```python
from src.agent.exceptions import (
    DatabaseConnectionException,
    DatabaseQueryException,
    DatabaseTransactionException
)

# Connection issues
raise DatabaseConnectionException(
    "Failed to connect to database",
    context={"host": "db.example.com", "port": 5432, "timeout": 30}
)

# Query execution problems
raise DatabaseQueryException(
    "Query execution timeout",
    context={"query": "SELECT * FROM large_table", "timeout_seconds": 300}
)

# Transaction management issues
raise DatabaseTransactionException(
    "Transaction rollback failed",
    context={"transaction_id": "tx_123", "operations_completed": 5}
)
```

### Validation Exceptions

```python
from src.agent.exceptions import InputValidationException, SQLValidationException

# Input validation
raise InputValidationException(
    "Invalid email format",
    context={"field": "email", "value": "invalid-email", "pattern": r".*@.*\..*"}
)

# SQL validation
raise SQLValidationException(
    "SQL syntax error",
    context={"query": "SELECT * FROM", "error": "Incomplete SELECT statement"}
)
```

### Agent State Exceptions

```python
from src.agent.exceptions import InvalidStateTransitionException, CommandProcessingException

# Invalid state transitions
raise InvalidStateTransitionException(
    "Cannot transition from 'completed' to 'processing'",
    context={
        "current_state": "completed",
        "attempted_state": "processing",
        "valid_transitions": ["reset", "archive"]
    }
)

# Command processing failures
raise CommandProcessingException(
    "Failed to process SQLGrounding command",
    context={
        "command_type": "SQLGrounding",
        "command_id": "cmd_123",
        "stage": "table_mapping"
    }
)
```

## Sensitive Data Filtering

The exception system automatically filters sensitive information:

```python
from src.agent.exceptions import DatabaseConnectionException

# Context with sensitive data
context = {
    "connection_string": "postgresql://user:secret123@db.example.com/prod",
    "api_key": "sk-1234567890abcdef",
    "username": "admin_user",  # Not sensitive
    "retry_count": 3  # Not sensitive
}

exception = DatabaseConnectionException("Connection failed", context=context)

# Raw context (for internal use)
print(exception.context)
# {'connection_string': 'postgresql://user:secret123@db.example.com/prod',
#  'api_key': 'sk-1234567890abcdef', 'username': 'admin_user', 'retry_count': 3}

# Sanitized context (safe for logging)
print(exception.get_sanitized_context())
# {'connection_string': '[FILTERED]', 'api_key': '[FILTERED]',
#  'username': 'admin_user', 'retry_count': 3}
```

## Best Practices

### 1. Use Specific Exception Types

```python
# ✅ Good: Specific exception type
raise DatabaseConnectionException("Connection timeout")

# ❌ Bad: Generic exception
raise Exception("Database error")
```

### 2. Always Provide Context

```python
# ✅ Good: Rich context for debugging
context = {
    "operation": "user_creation",
    "user_id": user_id,
    "retry_count": 2,
    "timeout_seconds": 30
}
raise DatabaseQueryException("User creation failed", context=context)

# ❌ Bad: No context
raise DatabaseQueryException("User creation failed")
```

### 3. Chain Exceptions Properly

```python
# ✅ Good: Proper exception chaining
try:
    risky_operation()
except ValueError as e:
    raise InputValidationException(
        "Invalid input data",
        context={"field": "age", "value": -5},
        original_exception=e
    ) from e

# ❌ Bad: Lost original exception
try:
    risky_operation()
except ValueError:
    raise InputValidationException("Invalid input data")
```

### 4. Handle Exceptions at Appropriate Levels

```python
# Service layer - transform and add context
def service_operation():
    try:
        return database_operation()
    except DatabaseException as e:
        # Add service-level context
        service_context = {"service": "user_service", "operation": "get_user"}
        combined_context = {**e.context, **service_context}

        raise DatabaseException(
            f"Service operation failed: {str(e)}",
            context=combined_context,
            original_exception=e
        ) from e

# API layer - log and return appropriate HTTP response
def api_endpoint():
    try:
        return service_operation()
    except DatabaseException as e:
        logger.error(f"Database error: {e}", extra={"context": e.get_sanitized_context()})
        return {"error": "Internal server error"}, 500
    except ValidationException as e:
        logger.warning(f"Validation error: {e}", extra={"context": e.get_sanitized_context()})
        return {"error": str(e)}, 400
```

### 5. Log with Sanitized Context

```python
import logging
from src.agent.exceptions import AgentException

logger = logging.getLogger(__name__)

try:
    risky_operation()
except AgentException as e:
    # Always use sanitized context for logging
    logger.error(
        f"Operation failed: {e}",
        extra={
            "exception_type": type(e).__name__,
            "context": e.get_sanitized_context(),
            "original_exception": str(e.original_exception) if e.original_exception else None
        }
    )
    raise
```

## Integration with Existing Code

To integrate the exception hierarchy into existing code:

1. **Import the appropriate exceptions** where needed
2. **Replace generic `except Exception` blocks** with specific exception types
3. **Replace `return None` patterns** with explicit exception raising
4. **Add context information** to exception instantiation
5. **Use exception chaining** to preserve original error information
6. **Update error handling code** to catch specific exception types

Example integration in database.py:

```python
# Add import at the top
from src.agent.exceptions import (
    DatabaseConnectionException,
    DatabaseQueryException,
    DatabaseTransactionException
)

# Update methods one by one
def _get_connection(self) -> Any:
    try:
        engine = create_engine(self.connection_string)
        logger.info("SQLAlchemy engine created successfully.")
        return engine
    except Exception as e:
        context = {
            "connection_string": self.connection_string,
            "db_type": self.db_type,
            "operation": "create_engine"
        }
        raise DatabaseConnectionException(
            f"Failed to create database engine: {str(e)}",
            context=context,
            original_exception=e
        ) from e
```

This gradual approach allows for incremental migration while maintaining system stability.
