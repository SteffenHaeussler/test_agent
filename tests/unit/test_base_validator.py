"""Tests for the base validator infrastructure."""

import pytest
from typing import Any

from src.agent.exceptions import InputValidationException


class TestValidationResult:
    """Test suite for ValidationResult dataclass."""

    def test_should_create_valid_result(self):
        """Should create a valid ValidationResult."""
        # This test will fail until we implement ValidationResult
        from src.agent.validators.base import ValidationResult

        # Arrange & Act
        result = ValidationResult(
            is_valid=True, errors=[], warnings=[], sanitized_input="test input"
        )

        # Assert
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.sanitized_input == "test input"

    def test_should_create_invalid_result_with_errors(self):
        """Should create an invalid ValidationResult with errors."""
        from src.agent.validators.base import ValidationResult

        # Arrange & Act
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            sanitized_input="sanitized input",
        )

        # Assert
        assert result.is_valid is False
        assert result.errors == ["Error 1", "Error 2"]
        assert result.warnings == ["Warning 1"]
        assert result.sanitized_input == "sanitized input"


class TestValidationRule:
    """Test suite for ValidationRule abstract base class."""

    def test_validation_rule_should_be_abstract(self):
        """ValidationRule should be an abstract base class."""
        from src.agent.validators.base import ValidationRule

        # Act & Assert - should not be able to instantiate directly
        with pytest.raises(TypeError):
            ValidationRule()

    def test_concrete_validation_rule_should_implement_validate(self):
        """Concrete ValidationRule implementations should implement validate method."""
        from src.agent.validators.base import ValidationRule, ValidationResult

        # Arrange - Create a concrete implementation
        class TestRule(ValidationRule):
            def validate(self, value: Any) -> ValidationResult:
                return ValidationResult(True, [], [], str(value))

        # Act
        rule = TestRule()
        result = rule.validate("test")

        # Assert
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True


class TestBaseValidator:
    """Test suite for BaseValidator abstract base class."""

    def test_base_validator_should_be_abstract(self):
        """BaseValidator should be an abstract base class."""
        from src.agent.validators.base import BaseValidator

        # Act & Assert - should not be able to instantiate directly
        with pytest.raises(TypeError):
            BaseValidator()

    def test_concrete_validator_should_implement_validate_input(self):
        """Concrete BaseValidator implementations should implement validate_input method."""
        from src.agent.validators.base import BaseValidator, ValidationResult

        # Arrange - Create a concrete implementation
        class TestValidator(BaseValidator):
            def validate_input(self, input_data: Any) -> ValidationResult:
                return ValidationResult(True, [], [], str(input_data))

        # Act
        validator = TestValidator()
        result = validator.validate_input("test")

        # Assert
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validator_should_support_rule_composition(self):
        """BaseValidator should support composing multiple validation rules."""
        from src.agent.validators.base import (
            BaseValidator,
            ValidationRule,
            ValidationResult,
        )

        # Arrange - Create test rules
        class TestRule1(ValidationRule):
            def validate(self, value: Any) -> ValidationResult:
                if not value:
                    return ValidationResult(False, ["Value is empty"], [], str(value))
                return ValidationResult(True, [], [], str(value))

        class TestRule2(ValidationRule):
            def validate(self, value: Any) -> ValidationResult:
                if len(str(value)) > 10:
                    return ValidationResult(False, ["Value too long"], [], str(value))
                return ValidationResult(True, [], [], str(value))

        class CompositeValidator(BaseValidator):
            def __init__(self):
                self.rules = [TestRule1(), TestRule2()]

            def validate_input(self, input_data: Any) -> ValidationResult:
                errors = []
                warnings = []
                sanitized = str(input_data)

                for rule in self.rules:
                    result = rule.validate(input_data)
                    errors.extend(result.errors)
                    warnings.extend(result.warnings)
                    if result.sanitized_input:
                        sanitized = result.sanitized_input

                return ValidationResult(len(errors) == 0, errors, warnings, sanitized)

        # Act
        validator = CompositeValidator()

        # Test valid input
        result1 = validator.validate_input("valid")
        assert result1.is_valid is True

        # Test empty input
        result2 = validator.validate_input("")
        assert result2.is_valid is False
        assert "Value is empty" in result2.errors

        # Test too long input
        result3 = validator.validate_input("this is too long")
        assert result3.is_valid is False
        assert "Value too long" in result3.errors

    def test_validator_should_raise_validation_exception_on_validate_and_raise(self):
        """BaseValidator should provide a validate_and_raise method that throws ValidationException."""
        from src.agent.validators.base import BaseValidator, ValidationResult

        # Arrange
        class TestValidator(BaseValidator):
            def validate_input(self, input_data: Any) -> ValidationResult:
                if input_data == "invalid":
                    return ValidationResult(
                        False, ["Invalid input"], [], str(input_data)
                    )
                return ValidationResult(True, [], [], str(input_data))

        validator = TestValidator()

        # Act & Assert - valid input should not raise
        validator.validate_and_raise("valid input")

        # Act & Assert - invalid input should raise ValidationException
        with pytest.raises(InputValidationException) as exc_info:
            validator.validate_and_raise("invalid")

        assert "Invalid input" in str(exc_info.value)


class TestSanitizer:
    """Test suite for input sanitization utilities."""

    def test_should_provide_sanitizer_interface(self):
        """Should provide a Sanitizer abstract base class."""
        from src.agent.validators.base import Sanitizer

        # Act & Assert - should be abstract
        with pytest.raises(TypeError):
            Sanitizer()

    def test_concrete_sanitizer_should_implement_sanitize(self):
        """Concrete Sanitizer implementations should implement sanitize method."""
        from src.agent.validators.base import Sanitizer

        # Arrange
        class TestSanitizer(Sanitizer):
            def sanitize(self, input_data: str) -> str:
                return input_data.strip()

        # Act
        sanitizer = TestSanitizer()
        result = sanitizer.sanitize("  test  ")

        # Assert
        assert result == "test"
