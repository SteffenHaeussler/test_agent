"""
Configuration management for validators.

This module provides centralized configuration for all validators,
supporting:
- Default configuration values
- Environment variable overrides
- Custom configuration
- Configuration validation
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.agent.exceptions import InvalidConfigurationException


@dataclass
class ValidatorConfig:
    """
    Configuration for validator framework.

    This class centralizes all validator configuration options with
    sensible defaults and environment variable support.
    """

    # Question validator settings
    question_max_length: int = 5000
    enable_html_sanitization: bool = True
    enable_unicode_normalization: bool = True
    enable_whitespace_normalization: bool = True

    # SQL validator settings
    sql_validation_enabled: bool = True
    enable_sql_complexity_check: bool = True
    sql_max_joins: int = 10
    sql_max_subqueries: int = 5

    # Malicious pattern detection
    enable_malicious_pattern_detection: bool = True
    enable_prompt_injection_detection: bool = True

    # Custom malicious patterns (in addition to built-in ones)
    malicious_patterns: List[str] = field(default_factory=list)

    # Sanitization settings
    enable_sql_injection_sanitization: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate configuration values.

        Raises:
            InvalidConfigurationException: If configuration is invalid
        """
        # Validate question max length
        if self.question_max_length <= 0:
            raise InvalidConfigurationException(
                "question_max_length must be positive",
                context={"question_max_length": self.question_max_length},
            )

        if self.question_max_length > 1_000_000:  # 1MB limit
            raise InvalidConfigurationException(
                "question_max_length is too large (max 1,000,000 characters)",
                context={"question_max_length": self.question_max_length},
            )

        # Validate malicious patterns
        if not isinstance(self.malicious_patterns, list):
            raise InvalidConfigurationException(
                "malicious_patterns must be a list",
                context={"malicious_patterns": type(self.malicious_patterns)},
            )

    @classmethod
    def from_env(cls) -> "ValidatorConfig":
        """
        Create configuration from environment variables.

        Environment variables:
        - VALIDATOR_QUESTION_MAX_LENGTH: Maximum question length
        - VALIDATOR_SQL_VALIDATION_ENABLED: Enable SQL validation (true/false)
        - VALIDATOR_ENABLE_HTML_SANITIZATION: Enable HTML sanitization (true/false)
        - VALIDATOR_ENABLE_MALICIOUS_DETECTION: Enable malicious pattern detection (true/false)

        Returns:
            ValidatorConfig with values from environment
        """
        config_dict = {}

        # Question settings
        if "VALIDATOR_QUESTION_MAX_LENGTH" in os.environ:
            try:
                config_dict["question_max_length"] = int(
                    os.environ["VALIDATOR_QUESTION_MAX_LENGTH"]
                )
            except ValueError:
                raise InvalidConfigurationException(
                    "VALIDATOR_QUESTION_MAX_LENGTH must be an integer",
                    context={"value": os.environ["VALIDATOR_QUESTION_MAX_LENGTH"]},
                )

        # Boolean settings
        bool_env_vars = {
            "VALIDATOR_SQL_VALIDATION_ENABLED": "sql_validation_enabled",
            "VALIDATOR_ENABLE_HTML_SANITIZATION": "enable_html_sanitization",
            "VALIDATOR_ENABLE_UNICODE_NORMALIZATION": "enable_unicode_normalization",
            "VALIDATOR_ENABLE_WHITESPACE_NORMALIZATION": "enable_whitespace_normalization",
            "VALIDATOR_ENABLE_MALICIOUS_DETECTION": "enable_malicious_pattern_detection",
            "VALIDATOR_ENABLE_PROMPT_INJECTION_DETECTION": "enable_prompt_injection_detection",
            "VALIDATOR_ENABLE_SQL_INJECTION_SANITIZATION": "enable_sql_injection_sanitization",
        }

        for env_var, config_key in bool_env_vars.items():
            if env_var in os.environ:
                value = os.environ[env_var].lower()
                if value in ("true", "1", "yes", "on"):
                    config_dict[config_key] = True
                elif value in ("false", "0", "no", "off"):
                    config_dict[config_key] = False
                else:
                    raise InvalidConfigurationException(
                        f"{env_var} must be a boolean value (true/false)",
                        context={"value": os.environ[env_var]},
                    )

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "question_max_length": self.question_max_length,
            "enable_html_sanitization": self.enable_html_sanitization,
            "enable_unicode_normalization": self.enable_unicode_normalization,
            "enable_whitespace_normalization": self.enable_whitespace_normalization,
            "sql_validation_enabled": self.sql_validation_enabled,
            "enable_malicious_pattern_detection": self.enable_malicious_pattern_detection,
            "enable_prompt_injection_detection": self.enable_prompt_injection_detection,
            "malicious_patterns": self.malicious_patterns.copy(),
            "enable_sql_injection_sanitization": self.enable_sql_injection_sanitization,
        }

    def merge(self, other: "ValidatorConfig") -> "ValidatorConfig":
        """
        Merge this configuration with another, other takes precedence.

        Args:
            other: Configuration to merge with

        Returns:
            New ValidatorConfig with merged values
        """
        merged_dict = self.to_dict()
        merged_dict.update(other.to_dict())
        return ValidatorConfig(**merged_dict)
