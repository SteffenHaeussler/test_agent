"""
Enhanced SQL validator that fully integrates with the validator framework.

This validator extends the existing SQL validator functionality to work
seamlessly with the new validation framework while maintaining backward
compatibility.
"""

import re
from typing import Any, Optional

from loguru import logger

from src.agent.validators.base import BaseValidator, ValidationRule, ValidationResult
from src.agent.validators.sanitizers import SqlInjectionSanitizer
from src.agent.validators.sql_validator import SQLValidator


class SqlEmptyRule(ValidationRule):
    """Validation rule for detecting empty SQL queries."""

    def validate(self, value: Any) -> ValidationResult:
        """
        Check if the SQL query is empty or contains only whitespace.

        Args:
            value: The SQL query to validate

        Returns:
            ValidationResult indicating if query is non-empty
        """
        if value is None:
            return ValidationResult(
                is_valid=False,
                errors=["SQL query cannot be None"],
                warnings=[],
                sanitized_input="",
            )

        str_value = str(value).strip()
        if not str_value:
            return ValidationResult(
                is_valid=False,
                errors=["SQL query cannot be empty"],
                warnings=[],
                sanitized_input="",
            )

        return ValidationResult(
            is_valid=True, errors=[], warnings=[], sanitized_input=str_value
        )


class SqlSyntaxRule(ValidationRule):
    """Validation rule that wraps the existing SQLValidator logic."""

    def __init__(self):
        """Initialize SQL syntax rule with existing validator."""
        self.sql_validator = SQLValidator()

    def validate(self, value: Any) -> ValidationResult:
        """
        Validate SQL syntax using existing SQLValidator.

        Args:
            value: The SQL query to validate

        Returns:
            ValidationResult with validation outcome
        """
        str_value = str(value) if value is not None else ""

        try:
            # Use existing validator logic
            self.sql_validator.validate(str_value)
            return ValidationResult(
                is_valid=True, errors=[], warnings=[], sanitized_input=str_value
            )
        except ValueError as e:
            return ValidationResult(
                is_valid=False, errors=[str(e)], warnings=[], sanitized_input=str_value
            )


class SqlComplexityRule(ValidationRule):
    """Validation rule for checking SQL query complexity."""

    def __init__(self, max_joins: int = 10, max_subqueries: int = 5):
        """
        Initialize complexity rule with limits.

        Args:
            max_joins: Maximum number of joins allowed
            max_subqueries: Maximum number of subqueries allowed
        """
        self.max_joins = max_joins
        self.max_subqueries = max_subqueries

    def validate(self, value: Any) -> ValidationResult:
        """
        Check SQL query complexity.

        Args:
            value: The SQL query to validate

        Returns:
            ValidationResult with complexity analysis
        """
        str_value = str(value).upper() if value is not None else ""
        warnings = []
        errors = []

        # Count JOINs
        join_count = len(re.findall(r"\bJOIN\b", str_value))
        if join_count > self.max_joins:
            errors.append(
                f"Too many JOINs ({join_count}). Maximum allowed: {self.max_joins}"
            )
        elif join_count > self.max_joins // 2:
            warnings.append(
                f"High number of JOINs ({join_count}) may impact performance"
            )

        # Count subqueries (simplified - looks for nested SELECT)
        subquery_count = len(re.findall(r"\(\s*SELECT\b", str_value))
        if subquery_count > self.max_subqueries:
            errors.append(
                f"Too many subqueries ({subquery_count}). Maximum allowed: {self.max_subqueries}"
            )
        elif subquery_count > self.max_subqueries // 2:
            warnings.append(
                f"High number of subqueries ({subquery_count}) may impact performance"
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input=str(value) if value is not None else "",
        )


class EnhancedSQLValidator(BaseValidator):
    """
    Enhanced SQL validator that fully integrates with the validator framework.

    This validator provides comprehensive SQL validation including:
    - Empty query detection
    - SQL injection prevention (using existing SQLValidator)
    - Query complexity analysis
    - Input sanitization
    - Configuration support
    """

    def __init__(
        self,
        enable_complexity_check: bool = True,
        max_joins: int = 10,
        max_subqueries: int = 5,
        enable_sanitization: bool = True,
    ):
        """
        Initialize enhanced SQL validator.

        Args:
            enable_complexity_check: Whether to enable complexity checking
            max_joins: Maximum number of joins allowed
            max_subqueries: Maximum number of subqueries allowed
            enable_sanitization: Whether to enable SQL sanitization
        """
        self.enable_complexity_check = enable_complexity_check
        self.enable_sanitization = enable_sanitization

        # Initialize validation rules
        self.rules = [
            SqlEmptyRule(),
            SqlSyntaxRule(),  # Wraps existing SQLValidator
        ]

        if enable_complexity_check:
            self.rules.append(SqlComplexityRule(max_joins, max_subqueries))

        # Initialize sanitizer
        self.sanitizer = SqlInjectionSanitizer() if enable_sanitization else None

    def validate_input(self, input_data: Any) -> ValidationResult:
        """
        Validate and optionally sanitize SQL input.

        Args:
            input_data: The SQL query to validate

        Returns:
            ValidationResult with validation outcome and sanitized SQL
        """
        # Convert input to string for processing
        if input_data is None:
            str_input = ""
        else:
            str_input = str(input_data)

        # Apply sanitization if enabled
        sanitized = str_input
        if self.sanitizer:
            sanitized = self.sanitizer.sanitize(str_input)

        # Collect all validation results
        all_errors = []
        all_warnings = []
        is_valid = True

        # Apply all validation rules
        for rule in self.rules:
            result = rule.validate(str_input)  # Validate original, not sanitized
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            if not result.is_valid:
                is_valid = False

        logger.debug(
            f"SQL validation result: valid={is_valid}, errors={len(all_errors)}, warnings={len(all_warnings)}"
        )

        return ValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            sanitized_input=sanitized,
        )

    @classmethod
    def from_config(cls, config) -> "EnhancedSQLValidator":
        """
        Create enhanced SQL validator from configuration.

        Args:
            config: ValidatorConfig instance

        Returns:
            Configured EnhancedSQLValidator instance
        """
        return cls(
            enable_complexity_check=getattr(
                config, "enable_sql_complexity_check", True
            ),
            max_joins=getattr(config, "sql_max_joins", 10),
            max_subqueries=getattr(config, "sql_max_subqueries", 5),
            enable_sanitization=getattr(
                config, "enable_sql_injection_sanitization", True
            ),
        )


# Maintain backward compatibility by providing the same interface as SQLValidator
class BackwardCompatibleSQLValidator(EnhancedSQLValidator):
    """
    Backward compatible wrapper that provides the same interface as the original SQLValidator.

    This allows existing code to continue working while benefiting from the enhanced validation.
    """

    def validate(self, sql_query: Optional[str]) -> bool:
        """
        Validate SQL query for safety (backward compatible interface).

        Args:
            sql_query: The SQL query to validate

        Returns:
            True if query is safe

        Raises:
            ValueError: If query contains forbidden operations
        """
        result = self.validate_input(sql_query)

        if not result.is_valid:
            # Raise the first error to maintain compatibility
            error_message = (
                result.errors[0] if result.errors else "SQL validation failed"
            )
            raise ValueError(error_message)

        return True
