"""
Factory for creating configured validators.

This module provides a factory pattern for creating validators
with proper configuration, making it easy to create consistent
validator instances throughout the application.
"""

from typing import List, Optional

from src.agent.validators.base import BaseValidator, ValidationResult
from src.agent.validators.config import ValidatorConfig
from src.agent.validators.question_validator import QuestionValidator
from src.agent.validators.enhanced_sql_validator import EnhancedSQLValidator


class NoOpValidator(BaseValidator):
    """
    No-operation validator that always passes validation.

    Used when a validator is disabled via configuration.
    """

    def __init__(self):
        """Initialize no-op validator."""
        self._disabled = True

    def validate_input(self, input_data) -> ValidationResult:
        """
        Always return valid result.

        Args:
            input_data: Input to validate (ignored)

        Returns:
            ValidationResult indicating success
        """
        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            sanitized_input=str(input_data) if input_data is not None else "",
        )


class CompositeValidator(BaseValidator):
    """
    Composite validator that combines multiple validators.

    This validator runs multiple validators in sequence and
    combines their results.
    """

    def __init__(self, validators: List[BaseValidator]):
        """
        Initialize composite validator.

        Args:
            validators: List of validators to combine
        """
        self.validators = validators or []

    def validate_input(self, input_data) -> ValidationResult:
        """
        Run all validators and combine results.

        Args:
            input_data: Input to validate

        Returns:
            Combined ValidationResult from all validators
        """
        all_errors = []
        all_warnings = []
        is_valid = True
        sanitized_input = str(input_data) if input_data is not None else ""

        for validator in self.validators:
            result = validator.validate_input(input_data)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

            if not result.is_valid:
                is_valid = False

            # Use the last sanitized input
            if result.sanitized_input:
                sanitized_input = result.sanitized_input

        return ValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            sanitized_input=sanitized_input,
        )


class ValidatorFactory:
    """
    Factory for creating configured validators.

    This factory creates validator instances with proper configuration,
    ensuring consistency across the application.
    """

    def __init__(self, config: ValidatorConfig):
        """
        Initialize factory with configuration.

        Args:
            config: Validator configuration
        """
        self.config = config

    def create_question_validator(self) -> BaseValidator:
        """
        Create a question validator with current configuration.

        Returns:
            Configured QuestionValidator instance
        """
        return QuestionValidator.from_config(self.config)

    def create_sql_validator(self) -> Optional[BaseValidator]:
        """
        Create a SQL validator with current configuration.

        Returns:
            Configured EnhancedSQLValidator instance or None if disabled
        """
        if not self.config.sql_validation_enabled:
            return None

        return EnhancedSQLValidator.from_config(self.config)

    def create_composite_validator(self, validator_types: List[str]) -> BaseValidator:
        """
        Create a composite validator with specified validator types.

        Args:
            validator_types: List of validator types ('question', 'sql')

        Returns:
            CompositeValidator combining specified validators
        """
        validators = []

        for validator_type in validator_types:
            if validator_type == "question":
                validators.append(self.create_question_validator())
            elif validator_type == "sql" and self.config.sql_validation_enabled:
                sql_validator = self.create_sql_validator()
                if sql_validator:
                    validators.append(sql_validator)

        return CompositeValidator(validators)

    @classmethod
    def from_env(cls) -> "ValidatorFactory":
        """
        Create factory with configuration from environment variables.

        Returns:
            ValidatorFactory with environment-based configuration
        """
        config = ValidatorConfig.from_env()
        return cls(config)
