"""
Validation middleware for integrating validators with handlers.

This module provides middleware that can be integrated into the message bus
and handler pipeline to automatically validate commands before processing.
"""

from typing import Any, Callable, Optional

from loguru import logger

from src.agent.domain import commands
from src.agent.exceptions import InputValidationException
from src.agent.validators.base import ValidationResult
from src.agent.validators.config import ValidatorConfig
from src.agent.validators.factory import ValidatorFactory


class ValidationMiddleware:
    """
    Middleware for validating commands before handler execution.

    This middleware automatically validates different types of commands
    using the appropriate validators and prevents invalid commands from
    reaching handlers.
    """

    def __init__(self, config: ValidatorConfig):
        """
        Initialize validation middleware.

        Args:
            config: Validator configuration
        """
        self.config = config
        self.factory = ValidatorFactory(config)

        # Pre-create validators for efficiency
        self.question_validator = self.factory.create_question_validator()
        self.sql_validator = self.factory.create_sql_validator()
        self.composite_validator = self.factory.create_composite_validator(
            ["question", "sql"]
        )

    def validate_command(self, command: Any) -> ValidationResult:
        """
        Validate a command based on its type.

        Args:
            command: Command to validate

        Returns:
            ValidationResult with validation outcome

        Raises:
            InputValidationException: If validation fails
        """
        if isinstance(command, commands.Question):
            return self._validate_question_command(command)
        elif isinstance(command, commands.SQLQuestion):
            return self._validate_sql_question_command(command)
        else:
            # For unknown command types, return valid by default
            logger.debug(f"No validation configured for command type: {type(command)}")
            return ValidationResult(
                is_valid=True, errors=[], warnings=[], sanitized_input=None
            )

    def _validate_question_command(
        self, command: commands.Question
    ) -> ValidationResult:
        """
        Validate a Question command.

        Args:
            command: Question command to validate

        Returns:
            ValidationResult with validation outcome
        """
        logger.debug(f"Validating Question command: {command.q_id}")

        result = self.question_validator.validate_input(command.question)

        if result.is_valid:
            logger.debug(f"Question validation passed for {command.q_id}")
        else:
            logger.warning(
                f"Question validation failed for {command.q_id}: {result.errors}",
                extra={
                    "q_id": command.q_id,
                    "errors": result.errors,
                    "warnings": result.warnings,
                },
            )

        return result

    def _validate_sql_question_command(
        self, command: commands.SQLQuestion
    ) -> ValidationResult:
        """
        Validate an SQLQuestion command.

        Args:
            command: SQLQuestion command to validate

        Returns:
            ValidationResult with validation outcome
        """
        logger.debug(f"Validating SQLQuestion command: {command.q_id}")

        # First validate the question text
        question_result = self.question_validator.validate_input(command.question)

        # If SQL validation is enabled and we have SQL content, validate it too
        if self.sql_validator and hasattr(command, "sql_query") and command.sql_query:
            sql_result = self.sql_validator.validate_input(command.sql_query)

            # Combine results
            combined_errors = question_result.errors + sql_result.errors
            combined_warnings = question_result.warnings + sql_result.warnings

            result = ValidationResult(
                is_valid=question_result.is_valid and sql_result.is_valid,
                errors=combined_errors,
                warnings=combined_warnings,
                sanitized_input=question_result.sanitized_input,
            )
        else:
            result = question_result

        if result.is_valid:
            logger.debug(f"SQLQuestion validation passed for {command.q_id}")
        else:
            logger.warning(
                f"SQLQuestion validation failed for {command.q_id}: {result.errors}",
                extra={
                    "q_id": command.q_id,
                    "errors": result.errors,
                    "warnings": result.warnings,
                },
            )

        return result

    def validate_and_call(
        self, handler: Callable, command: Any, *args, **kwargs
    ) -> Any:
        """
        Validate command and call handler if validation passes.

        Args:
            handler: Handler function to call
            command: Command to validate and pass to handler
            *args: Additional arguments to pass to handler
            **kwargs: Additional keyword arguments to pass to handler

        Returns:
            Result from handler execution

        Raises:
            InputValidationException: If validation fails
        """
        # Validate the command
        result = self.validate_command(command)

        if not result.is_valid:
            error_message = "; ".join(result.errors)
            context = {
                "command_type": type(command).__name__,
                "command_id": getattr(command, "q_id", "unknown"),
                "errors": result.errors,
                "warnings": result.warnings,
            }
            raise InputValidationException(
                message=f"Command validation failed: {error_message}", context=context
            )

        # Log warnings if present
        if result.warnings:
            logger.warning(
                f"Command validation passed with warnings: {result.warnings}",
                extra={
                    "command_type": type(command).__name__,
                    "warnings": result.warnings,
                },
            )

        # Call the handler with the original command
        return handler(command, *args, **kwargs)

    @classmethod
    def from_env(cls) -> "ValidationMiddleware":
        """
        Create validation middleware from environment configuration.

        Returns:
            ValidationMiddleware with environment-based configuration
        """
        config = ValidatorConfig.from_env()
        return cls(config)


class ValidationDecorator:
    """
    Decorator for adding validation to handler functions.

    This provides a convenient way to add validation to existing handlers
    without modifying their implementation.
    """

    def __init__(self, config: Optional[ValidatorConfig] = None):
        """
        Initialize validation decorator.

        Args:
            config: Validator configuration, defaults to environment config
        """
        if config is None:
            config = ValidatorConfig.from_env()
        self.middleware = ValidationMiddleware(config)

    def __call__(self, handler: Callable) -> Callable:
        """
        Decorate a handler function with validation.

        Args:
            handler: Handler function to decorate

        Returns:
            Decorated handler function
        """

        def wrapper(command: Any, *args, **kwargs) -> Any:
            return self.middleware.validate_and_call(handler, command, *args, **kwargs)

        wrapper.__name__ = handler.__name__
        wrapper.__doc__ = handler.__doc__
        return wrapper


# Convenience function for decorating handlers
def validate_input(config: Optional[ValidatorConfig] = None):
    """
    Decorator function for adding input validation to handlers.

    Usage:
        @validate_input()
        def my_handler(command, adapter, notifications=None):
            # Handler implementation
            pass

    Args:
        config: Optional validator configuration

    Returns:
        Decorator function
    """
    return ValidationDecorator(config)
