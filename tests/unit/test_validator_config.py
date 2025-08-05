"""Tests for validator configuration support."""

import pytest


class TestValidatorConfig:
    """Test suite for validator configuration."""

    def test_should_provide_default_config(self):
        """Should provide default configuration values."""
        from src.agent.validators.config import ValidatorConfig

        # Act
        config = ValidatorConfig()

        # Assert
        assert config.question_max_length > 0
        assert config.question_max_length == 5000  # Default value
        assert config.sql_validation_enabled is True
        assert isinstance(config.malicious_patterns, list)
        # Default patterns list is empty, patterns are built-in to the rules

    def test_should_allow_custom_config_values(self):
        """Should allow customization of configuration values."""
        from src.agent.validators.config import ValidatorConfig

        # Arrange
        custom_settings = {
            "question_max_length": 1000,
            "sql_validation_enabled": False,
            "enable_html_sanitization": False,
        }

        # Act
        config = ValidatorConfig(**custom_settings)

        # Assert
        assert config.question_max_length == 1000
        assert config.sql_validation_enabled is False
        assert config.enable_html_sanitization is False

    def test_should_load_from_environment_variables(self):
        """Should load configuration from environment variables."""
        import os
        from src.agent.validators.config import ValidatorConfig

        # Arrange
        os.environ["VALIDATOR_QUESTION_MAX_LENGTH"] = "2000"
        os.environ["VALIDATOR_SQL_VALIDATION_ENABLED"] = "false"

        try:
            # Act
            config = ValidatorConfig.from_env()

            # Assert
            assert config.question_max_length == 2000
            assert config.sql_validation_enabled is False
        finally:
            # Cleanup
            os.environ.pop("VALIDATOR_QUESTION_MAX_LENGTH", None)
            os.environ.pop("VALIDATOR_SQL_VALIDATION_ENABLED", None)

    def test_should_validate_config_values(self):
        """Should validate configuration values."""
        from src.agent.validators.config import ValidatorConfig
        from src.agent.exceptions import InvalidConfigurationException

        # Act & Assert - negative max length should fail
        with pytest.raises(InvalidConfigurationException):
            ValidatorConfig(question_max_length=-1)

        # Act & Assert - zero max length should fail
        with pytest.raises(InvalidConfigurationException):
            ValidatorConfig(question_max_length=0)

        # Act & Assert - extremely large max length should fail
        with pytest.raises(InvalidConfigurationException):
            ValidatorConfig(question_max_length=10_000_000)

    def test_should_provide_config_dict(self):
        """Should provide configuration as dictionary."""
        from src.agent.validators.config import ValidatorConfig

        # Arrange
        config = ValidatorConfig(question_max_length=1500)

        # Act
        config_dict = config.to_dict()

        # Assert
        assert isinstance(config_dict, dict)
        assert config_dict["question_max_length"] == 1500
        assert "sql_validation_enabled" in config_dict
        assert "enable_html_sanitization" in config_dict


class TestValidatorFactory:
    """Test suite for validator factory with configuration."""

    def test_should_create_question_validator_with_config(self):
        """Should create question validator with configuration."""
        from src.agent.validators.config import ValidatorConfig
        from src.agent.validators.factory import ValidatorFactory

        # Arrange
        config = ValidatorConfig(question_max_length=1000)
        factory = ValidatorFactory(config)

        # Act
        validator = factory.create_question_validator()

        # Assert
        assert validator is not None
        assert validator.max_length == 1000

    def test_should_create_sql_validator_with_config(self):
        """Should create SQL validator with configuration."""
        from src.agent.validators.config import ValidatorConfig
        from src.agent.validators.factory import ValidatorFactory

        # Arrange
        config = ValidatorConfig(sql_validation_enabled=True)
        factory = ValidatorFactory(config)

        # Act
        validator = factory.create_sql_validator()

        # Assert
        assert validator is not None

    def test_should_respect_disabled_validators(self):
        """Should respect disabled validator configuration."""
        from src.agent.validators.config import ValidatorConfig
        from src.agent.validators.factory import ValidatorFactory

        # Arrange
        config = ValidatorConfig(sql_validation_enabled=False)
        factory = ValidatorFactory(config)

        # Act
        validator = factory.create_sql_validator()

        # Assert - should return None or a no-op validator
        assert validator is None or hasattr(validator, "_disabled")

    def test_should_create_composite_validator(self):
        """Should create composite validator with multiple validators."""
        from src.agent.validators.config import ValidatorConfig
        from src.agent.validators.factory import ValidatorFactory

        # Arrange
        config = ValidatorConfig()
        factory = ValidatorFactory(config)

        # Act
        validator = factory.create_composite_validator(["question", "sql"])

        # Assert
        assert validator is not None
        # Should combine multiple validators
        result = validator.validate_input("What is Python?")
        assert result.is_valid is True


class TestConfigurableValidators:
    """Test suite for validators using configuration."""

    def test_question_validator_should_use_config_max_length(self):
        """Question validator should use configured max length."""
        from src.agent.validators.config import ValidatorConfig
        from src.agent.validators.question_validator import QuestionValidator

        # Arrange
        config = ValidatorConfig(question_max_length=50)
        validator = QuestionValidator.from_config(config)
        long_question = "This is a very long question that exceeds the configured limit of 50 characters"

        # Act
        result = validator.validate_input(long_question)

        # Assert
        assert result.is_valid is False
        assert "too long" in " ".join(result.errors).lower()

    def test_question_validator_should_use_config_sanitization_settings(self):
        """Question validator should use configured sanitization settings."""
        from src.agent.validators.config import ValidatorConfig
        from src.agent.validators.question_validator import QuestionValidator

        # Arrange
        config = ValidatorConfig(enable_html_sanitization=False)
        validator = QuestionValidator.from_config(config)
        html_question = "What is the <b>capital</b> of France?"

        # Act
        result = validator.validate_input(html_question)

        # Assert
        # With HTML sanitization disabled, tags should remain
        assert "<b>" in result.sanitized_input

    def test_sql_validator_should_use_config_settings(self):
        """SQL validator should use configured settings."""
        from src.agent.validators.config import ValidatorConfig
        from src.agent.validators.sql_validator import SQLValidator

        # Arrange
        config = ValidatorConfig(sql_validation_enabled=True)
        validator = SQLValidator.from_config(config)

        # Act
        result = validator.validate("SELECT * FROM users")

        # Assert
        assert result is True  # Should pass validation
