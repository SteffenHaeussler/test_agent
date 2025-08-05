"""Integration tests for validator framework with handlers."""

import pytest
from unittest.mock import Mock, patch

from src.agent.domain import commands
from src.agent.exceptions import InputValidationException
from src.agent.validators.config import ValidatorConfig
from src.agent.validators.factory import ValidatorFactory


class TestValidatorHandlerIntegration:
    """Integration tests for validators with handlers."""

    def test_should_validate_question_before_processing(self):
        """Should validate questions before passing to handlers."""
        # Test validation logic that would be used before handler processing
        # (avoiding direct handler import due to dependency issues)

        # Arrange
        config = ValidatorConfig(question_max_length=50)
        factory = ValidatorFactory(config)
        validator = factory.create_question_validator()

        # Test with valid question
        valid_command = commands.Question(question="What is Python?", q_id="test-1")

        # Validate before calling handler
        result = validator.validate_input(valid_command.question)
        assert result.is_valid is True

        # Test with invalid question (too long)
        long_question = "This is a very long question that exceeds the configured maximum length limit"
        invalid_command = commands.Question(question=long_question, q_id="test-2")

        result = validator.validate_input(invalid_command.question)
        assert result.is_valid is False
        assert "too long" in " ".join(result.errors).lower()

    def test_should_validate_sql_questions_before_processing(self):
        """Should validate SQL questions before processing."""
        # Test SQL validation logic that would be used before handler processing
        # (avoiding direct handler import due to dependency issues)

        # Arrange
        config = ValidatorConfig(sql_validation_enabled=True)
        factory = ValidatorFactory(config)
        sql_validator = factory.create_sql_validator()

        # Test with valid SQL question
        # Validate SQL content
        result = sql_validator.validate_input("SELECT * FROM users")
        assert result.is_valid is True

        # Test with dangerous SQL
        dangerous_sql = "DROP TABLE users; --"
        result = sql_validator.validate_input(dangerous_sql)
        assert result.is_valid is False
        assert "DROP operations are not allowed" in " ".join(result.errors)

    def test_should_integrate_validation_with_composite_validator(self):
        """Should integrate multiple validators using composite validator."""
        # Arrange
        config = ValidatorConfig(question_max_length=100, sql_validation_enabled=True)
        factory = ValidatorFactory(config)
        composite_validator = factory.create_composite_validator(["question", "sql"])

        # Test valid input
        result = composite_validator.validate_input("What is the weather?")
        assert result.is_valid is True

        # Test invalid input (malicious pattern)
        result = composite_validator.validate_input("'; DROP TABLE users; --")
        assert result.is_valid is False

    def test_should_handle_validation_errors_gracefully(self):
        """Should handle validation errors gracefully in handler context."""
        # Arrange
        config = ValidatorConfig(question_max_length=10)
        factory = ValidatorFactory(config)
        validator = factory.create_question_validator()

        long_question = "This question is way too long for the configured limit"

        # Act & Assert
        with pytest.raises(InputValidationException) as exc_info:
            validator.validate_and_raise(long_question)

        assert "too long" in str(exc_info.value).lower()
        assert exc_info.value.context["input_data"] == long_question[:100]  # Truncated

    def test_should_sanitize_input_for_safe_processing(self):
        """Should sanitize input for safe processing by handlers."""
        # Arrange
        config = ValidatorConfig(enable_html_sanitization=True)
        factory = ValidatorFactory(config)
        validator = factory.create_question_validator()

        # Test HTML sanitization
        html_question = (
            "What is the <b>capital</b> of <script>alert('xss')</script> France?"
        )

        # Act
        result = validator.validate_input(html_question)

        # Assert
        assert result.is_valid is False  # Script tags should fail
        assert "script" not in result.sanitized_input
        assert "capital" in result.sanitized_input

    def test_should_work_with_disabled_validators(self):
        """Should work correctly when validators are disabled."""
        # Arrange
        config = ValidatorConfig(
            sql_validation_enabled=False, enable_html_sanitization=False
        )
        factory = ValidatorFactory(config)

        # SQL validator should be None when disabled
        sql_validator = factory.create_sql_validator()
        assert sql_validator is None

        # Question validator should preserve HTML when sanitization disabled
        question_validator = factory.create_question_validator()
        result = question_validator.validate_input("What is the <b>capital</b>?")
        assert result.is_valid is True
        assert "<b>" in result.sanitized_input

    def test_should_support_environment_configuration(self):
        """Should support configuration from environment variables."""
        import os

        # Arrange
        original_env = os.environ.copy()
        try:
            os.environ["VALIDATOR_QUESTION_MAX_LENGTH"] = "25"
            os.environ["VALIDATOR_SQL_VALIDATION_ENABLED"] = "false"

            # Act
            factory = ValidatorFactory.from_env()

            # Assert
            assert factory.config.question_max_length == 25
            assert factory.config.sql_validation_enabled is False

            # Test validator behavior
            validator = factory.create_question_validator()
            long_question = "This is a long question that should fail"
            result = validator.validate_input(long_question)
            assert result.is_valid is False  # Should fail due to short limit

        finally:
            # Cleanup
            os.environ.clear()
            os.environ.update(original_env)


class TestValidatorMiddleware:
    """Tests for validator middleware integration."""

    def test_should_create_validation_middleware(self):
        """Should create validation middleware for handlers."""
        from src.agent.validators.middleware import ValidationMiddleware

        # Arrange
        config = ValidatorConfig()
        middleware = ValidationMiddleware(config)

        # Mock handler
        def mock_handler(command, adapter, notifications=None):
            return "processed"

        # Test valid command
        valid_command = commands.Question(question="What is Python?", q_id="test")
        result = middleware.validate_and_call(mock_handler, valid_command, Mock())
        assert result == "processed"

        # Test invalid command
        empty_command = commands.Question(question="", q_id="test")
        with pytest.raises(InputValidationException):
            middleware.validate_and_call(mock_handler, empty_command, Mock())

    def test_should_integrate_with_message_bus(self):
        """Should integrate validation with message bus."""
        # Test validation middleware integration concept
        # (avoiding direct messagebus import due to dependency issues)
        from src.agent.validators.middleware import ValidationMiddleware

        # Arrange
        config = ValidatorConfig()
        middleware = ValidationMiddleware(config)

        # Mock handler function
        def mock_handler(command, adapter, notifications=None):
            return f"processed {command.q_id}"

        # Test that validation occurs before command processing
        with pytest.raises(InputValidationException):
            invalid_command = commands.Question(question="", q_id="test")
            middleware.validate_and_call(mock_handler, invalid_command, Mock())

    def test_should_log_validation_results(self):
        """Should log validation results for monitoring."""
        from src.agent.validators.middleware import ValidationMiddleware

        # Arrange
        config = ValidatorConfig()
        middleware = ValidationMiddleware(config)

        with patch("src.agent.validators.middleware.logger") as mock_logger:
            # Test valid input logging
            valid_command = commands.Question(question="Valid question", q_id="test")
            result = middleware.validate_command(valid_command)
            assert result.is_valid is True

            mock_logger.debug.assert_called()

            # Test invalid input logging
            invalid_command = commands.Question(question="", q_id="test")
            result = middleware.validate_command(invalid_command)
            assert result.is_valid is False

            mock_logger.warning.assert_called()

    def test_should_handle_different_command_types(self):
        """Should handle validation for different command types."""
        from src.agent.validators.middleware import ValidationMiddleware

        # Arrange
        config = ValidatorConfig()
        middleware = ValidationMiddleware(config)

        # Test Question command
        question_cmd = commands.Question(question="What is Python?", q_id="q1")
        result = middleware.validate_command(question_cmd)
        assert result.is_valid is True

        # Test SQLQuestion command
        sql_cmd = commands.SQLQuestion(question="Show users", q_id="sql1")
        result = middleware.validate_command(sql_cmd)
        assert result.is_valid is True

        # Test with malicious SQL in SQLQuestion
        malicious_sql_cmd = commands.SQLQuestion(
            question="'; DROP TABLE users; --", q_id="sql2"
        )
        result = middleware.validate_command(malicious_sql_cmd)
        assert result.is_valid is False
