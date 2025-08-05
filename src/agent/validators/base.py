"""
Base validator infrastructure for input validation framework.

This module provides:
- Abstract base classes for validators and sanitizers
- ValidationResult dataclass for structured validation results
- Composable validation rules
- Integration with custom exception hierarchy
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

from src.agent.exceptions import InputValidationException


@dataclass
class ValidationResult:
    """
    Result of a validation operation.

    This dataclass encapsulates the outcome of validation, providing:
    - Validation status (valid/invalid)
    - List of error messages for failed validations
    - List of warning messages for suspicious but not invalid inputs
    - Sanitized version of the input for safe processing

    Args:
        is_valid: Whether the input passed validation
        errors: List of error messages if validation failed
        warnings: List of warning messages for suspicious patterns
        sanitized_input: Cleaned/sanitized version of the input
    """

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_input: Optional[str] = None


class ValidationRule(ABC):
    """
    Abstract base class for individual validation rules.

    ValidationRule represents a single, composable validation check
    that can be applied to input data. Rules can be combined to create
    comprehensive validators.

    Each rule should:
    - Check a specific aspect of the input
    - Return a ValidationResult with clear error messages
    - Be reusable across different validators
    """

    @abstractmethod
    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a value according to this rule.

        Args:
            value: The value to validate

        Returns:
            ValidationResult with validation outcome
        """
        pass


class Sanitizer(ABC):
    """
    Abstract base class for input sanitization.

    Sanitizers clean and normalize input data without changing
    its fundamental meaning. They handle:
    - HTML/script tag removal
    - Unicode normalization
    - Whitespace normalization
    - Special character escaping
    """

    @abstractmethod
    def sanitize(self, input_data: str) -> str:
        """
        Sanitize input data.

        Args:
            input_data: The raw input to sanitize

        Returns:
            Sanitized version of the input
        """
        pass


class BaseValidator(ABC):
    """
    Abstract base class for all input validators.

    BaseValidator provides the core interface for validation in the
    agentic AI framework. It supports:
    - Composable validation rules
    - Clear validation result structure
    - Integration with exception hierarchy
    - Consistent error handling

    Concrete validators should:
    - Implement validate_input method
    - Use ValidationResult for structured responses
    - Provide clear, actionable error messages
    - Support sanitization of valid inputs
    """

    @abstractmethod
    def validate_input(self, input_data: Any) -> ValidationResult:
        """
        Validate input data.

        Args:
            input_data: The input to validate

        Returns:
            ValidationResult with validation outcome and sanitized input
        """
        pass

    def validate_and_raise(self, input_data: Any) -> str:
        """
        Validate input and raise InputValidationException if invalid.

        This is a convenience method for cases where you want validation
        to fail fast with an exception rather than handling ValidationResult.

        Args:
            input_data: The input to validate

        Returns:
            Sanitized input if validation passes

        Raises:
            InputValidationException: If validation fails
        """
        result = self.validate_input(input_data)

        if not result.is_valid:
            error_message = "; ".join(result.errors)
            context = {
                "input_data": str(input_data)[:100],  # Truncate for safety
                "errors": result.errors,
                "warnings": result.warnings,
            }
            raise InputValidationException(
                message=f"Input validation failed: {error_message}", context=context
            )

        return result.sanitized_input or str(input_data)
