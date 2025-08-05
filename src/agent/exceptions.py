"""
Comprehensive exception hierarchy for the agentic AI framework.

This module provides a structured approach to error handling with:
- Clear inheritance hierarchy
- Context preservation for debugging
- Proper exception chaining
- Sensitive data filtering for logging

The hierarchy follows the principle of specific exceptions for specific
error conditions, making debugging and error handling more effective.
"""

import re
from typing import Any, Dict, List, Optional, Set


class AgentException(Exception):
    """
    Base exception class for all agent-related errors.

    This class provides:
    - Context preservation for debugging
    - Exception chaining support
    - Sensitive data filtering for safe logging
    - Structured error information

    Args:
        message: Human-readable error description
        context: Additional context information for debugging
        original_exception: The original exception that caused this error (for chaining)
    """

    # Sensitive keys that should be filtered from context when logging
    _SENSITIVE_KEYS: Set[str] = {
        "password",
        "passwd",
        "pwd",
        "secret",
        "api_key",
        "token",
        "auth",
        "authorization",
        "credential",
        "key",
        "private_key",
        "connection_string",
        "database_url",
        "dsn",
    }

    # Patterns for sensitive data in string values
    _SENSITIVE_PATTERNS: List[str] = [
        r"password=[\w\-_]+",
        r"://[^:]+:[^@]+@",  # URLs with credentials
        r"Bearer\s+[\w\-_\.]+",  # Bearer tokens
        r"key_[\w\-_]+",  # API keys
    ]

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.original_exception = original_exception

        # Set up exception chaining if original exception provided
        if original_exception:
            self.__cause__ = original_exception

    def __str__(self) -> str:
        """Return the error message."""
        return self.message

    def __repr__(self) -> str:
        """Return a detailed representation of the exception."""
        class_name = self.__class__.__name__
        context_info = f", context={self.context}" if self.context else ""
        return f"{class_name}('{self.message}'{context_info})"

    def get_sanitized_context(self) -> Dict[str, Any]:
        """
        Get context with sensitive information filtered out.

        This method creates a safe version of the context that can be
        logged without exposing sensitive information like passwords,
        API keys, or connection strings.

        Returns:
            Dictionary with sensitive values replaced with '[FILTERED]'
        """
        if not self.context:
            return {}

        sanitized = {}
        for key, value in self.context.items():
            if self._is_sensitive_key(key):
                sanitized[key] = "[FILTERED]"
            elif isinstance(value, str) and self._contains_sensitive_data(value):
                sanitized[key] = self._filter_sensitive_patterns(value)
            else:
                sanitized[key] = value

        return sanitized

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key is considered sensitive."""
        return key.lower() in self._SENSITIVE_KEYS

    def _contains_sensitive_data(self, value: str) -> bool:
        """Check if a string value contains sensitive patterns."""
        for pattern in self._SENSITIVE_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False

    def _filter_sensitive_patterns(self, value: str) -> str:
        """Filter sensitive patterns from a string value."""
        filtered_value = value
        for pattern in self._SENSITIVE_PATTERNS:
            filtered_value = re.sub(
                pattern, "[FILTERED]", filtered_value, flags=re.IGNORECASE
            )
        return filtered_value


# Database Exceptions
class DatabaseException(AgentException):
    """Base class for database-related exceptions."""

    pass


class DatabaseConnectionException(DatabaseException):
    """
    Exception raised when database connection fails.

    Common contexts:
    - host, port, database: Connection parameters
    - timeout_seconds: Connection timeout
    - retry_count: Number of connection attempts
    """

    pass


class DatabaseQueryException(DatabaseException):
    """
    Exception raised when database query execution fails.

    Common contexts:
    - query: The SQL query that failed
    - parameters: Query parameters
    - execution_time_ms: Time taken before failure
    - affected_rows: Number of rows affected (for DML)
    """

    pass


class DatabaseTransactionException(DatabaseException):
    """
    Exception raised when database transaction operations fail.

    Common contexts:
    - transaction_id: Unique transaction identifier
    - operations_completed: Number of operations before failure
    - rollback_attempted: Whether rollback was attempted
    """

    pass


# Validation Exceptions
class ValidationException(AgentException):
    """Base class for validation-related exceptions."""

    pass


class InputValidationException(ValidationException):
    """
    Exception raised when input validation fails.

    Common contexts:
    - field: Field name that failed validation
    - value: The invalid value (will be filtered if sensitive)
    - expected_format: Description of expected format
    - validation_rule: The validation rule that failed
    """

    pass


class SQLValidationException(ValidationException):
    """
    Exception raised when SQL validation fails.

    Common contexts:
    - sql_query: The SQL query that failed validation
    - syntax_errors: List of syntax error messages
    - line_number: Line number where error occurred
    - column_number: Column number where error occurred
    """

    pass


# Agent State Machine Exceptions
class AgentStateException(AgentException):
    """Base class for agent state machine related exceptions."""

    pass


class InvalidStateTransitionException(AgentStateException):
    """
    Exception raised when an invalid state transition is attempted.

    Common contexts:
    - current_state: The current agent state
    - attempted_state: The state transition that was attempted
    - valid_transitions: List of valid transitions from current state
    - agent_id: Unique identifier for the agent instance
    """

    pass


class CommandProcessingException(AgentStateException):
    """
    Exception raised when command processing fails.

    Common contexts:
    - command_type: The type of command being processed
    - command_id: Unique identifier for the command
    - processing_stage: Stage where processing failed
    - agent_state: Current agent state
    """

    pass


# External Service Exceptions
class ExternalServiceException(AgentException):
    """Base class for external service related exceptions."""

    pass


class LLMAPIException(ExternalServiceException):
    """
    Exception raised when LLM API calls fail.

    Common contexts:
    - provider: LLM provider (openai, anthropic, etc.)
    - model: Model name
    - status_code: HTTP status code
    - retry_count: Number of retry attempts
    - rate_limit_reset: When rate limit resets (if applicable)
    """

    pass


class RAGSystemException(ExternalServiceException):
    """
    Exception raised when RAG system operations fail.

    Common contexts:
    - operation: The RAG operation that failed (retrieve, embed, etc.)
    - query: The query that was being processed
    - index_name: Name of the search index
    - timeout_seconds: Operation timeout
    - documents_found: Number of documents found before failure
    """

    pass


class NotificationServiceException(ExternalServiceException):
    """
    Exception raised when notification service operations fail.

    Common contexts:
    - service: Notification service (slack, email, webhook, etc.)
    - channel: Target channel or recipient
    - message_id: Unique message identifier
    - retry_count: Number of retry attempts
    """

    pass


# Configuration Exceptions
class ConfigurationException(AgentException):
    """Base class for configuration-related exceptions."""

    pass


class MissingConfigurationException(ConfigurationException):
    """
    Exception raised when required configuration is missing.

    Common contexts:
    - config_key: The missing configuration key
    - config_section: Configuration section where key should be
    - required_for: What functionality requires this configuration
    - config_file: Configuration file being read
    """

    pass


class InvalidConfigurationException(ConfigurationException):
    """
    Exception raised when configuration values are invalid.

    Common contexts:
    - config_key: The configuration key with invalid value
    - config_value: The invalid value (will be filtered if sensitive)
    - expected_type: Expected data type
    - valid_range: Valid range or options for the value
    - validation_error: Specific validation error message
    """

    pass


# Export all exception classes for easy importing
__all__ = [
    # Base exception
    "AgentException",
    # Database exceptions
    "DatabaseException",
    "DatabaseConnectionException",
    "DatabaseQueryException",
    "DatabaseTransactionException",
    # Validation exceptions
    "ValidationException",
    "InputValidationException",
    "SQLValidationException",
    # Agent state exceptions
    "AgentStateException",
    "InvalidStateTransitionException",
    "CommandProcessingException",
    # External service exceptions
    "ExternalServiceException",
    "LLMAPIException",
    "RAGSystemException",
    "NotificationServiceException",
    # Configuration exceptions
    "ConfigurationException",
    "MissingConfigurationException",
    "InvalidConfigurationException",
]
