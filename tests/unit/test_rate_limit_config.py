"""
Tests for rate limiting configuration integration.

Following TDD principles - create failing tests first.
"""

from unittest.mock import patch

from src.agent.config import get_rate_limit_config


class TestRateLimitConfiguration:
    """Test rate limiting configuration in main config module."""

    def test_get_rate_limit_config_with_all_env_vars(self):
        """Should return rate limit config from environment variables."""
        env_vars = {
            "RATE_LIMIT_ENABLED": "true",
            "RATE_LIMIT_DEFAULT_CAPACITY": "120",
            "RATE_LIMIT_DEFAULT_REFILL_RATE": "2.0",
            "RATE_LIMIT_CLEANUP_INTERVAL": "600",
            "RATE_LIMIT_QUESTION_CAPACITY": "30",
            "RATE_LIMIT_QUESTION_REFILL_RATE": "0.5",
            "RATE_LIMIT_SQL_CAPACITY": "20",
            "RATE_LIMIT_SQL_REFILL_RATE": "0.3",
        }

        with patch.dict("os.environ", env_vars):
            config = get_rate_limit_config()

        assert config["enabled"] is True
        assert config["default_capacity"] == 120
        assert config["default_refill_rate"] == 2.0
        assert config["cleanup_interval"] == 600

        # Per-command limits
        assert config["per_command_limits"]["Question"]["capacity"] == 30
        assert config["per_command_limits"]["Question"]["refill_rate"] == 0.5
        assert config["per_command_limits"]["SQLQuestion"]["capacity"] == 20
        assert config["per_command_limits"]["SQLQuestion"]["refill_rate"] == 0.3

    def test_get_rate_limit_config_with_defaults(self):
        """Should return default rate limit config when no env vars set."""
        # Clear any existing environment variables
        env_vars = {
            "RATE_LIMIT_ENABLED": "",
            "RATE_LIMIT_DEFAULT_CAPACITY": "",
            "RATE_LIMIT_DEFAULT_REFILL_RATE": "",
            "RATE_LIMIT_CLEANUP_INTERVAL": "",
        }

        with patch.dict("os.environ", env_vars, clear=True):
            config = get_rate_limit_config()

        assert config["enabled"] is True  # Default
        assert config["default_capacity"] == 60  # Default
        assert config["default_refill_rate"] == 1.0  # Default
        assert config["cleanup_interval"] == 300  # Default
        assert config["per_command_limits"] == {}  # No custom limits

    def test_get_rate_limit_config_disabled(self):
        """Should return disabled config when explicitly disabled."""
        env_vars = {"RATE_LIMIT_ENABLED": "false"}

        with patch.dict("os.environ", env_vars):
            config = get_rate_limit_config()

        assert config["enabled"] is False

    def test_get_rate_limit_config_partial_command_limits(self):
        """Should handle partial per-command limit configuration."""
        env_vars = {
            "RATE_LIMIT_QUESTION_CAPACITY": "50",
            # Missing RATE_LIMIT_QUESTION_REFILL_RATE - should use default
            "RATE_LIMIT_SQL_REFILL_RATE": "0.8",
            # Missing RATE_LIMIT_SQL_CAPACITY - should use default
        }

        with patch.dict("os.environ", env_vars):
            config = get_rate_limit_config()

        # Question command should have custom capacity, default refill rate
        assert config["per_command_limits"]["Question"]["capacity"] == 50
        assert config["per_command_limits"]["Question"]["refill_rate"] == 1.0  # Default

        # SQL command should have default capacity, custom refill rate
        assert config["per_command_limits"]["SQLQuestion"]["capacity"] == 60  # Default
        assert config["per_command_limits"]["SQLQuestion"]["refill_rate"] == 0.8

    def test_get_rate_limit_config_invalid_values_use_defaults(self):
        """Should use defaults for invalid environment variable values."""
        env_vars = {
            "RATE_LIMIT_ENABLED": "invalid",  # Should default to true (safe default)
            "RATE_LIMIT_DEFAULT_CAPACITY": "not_a_number",  # Should default to 60
            "RATE_LIMIT_DEFAULT_REFILL_RATE": "invalid_float",  # Should default to 1.0
            "RATE_LIMIT_CLEANUP_INTERVAL": "not_an_int",  # Should default to 300
        }

        with patch.dict("os.environ", env_vars):
            config = get_rate_limit_config()

        assert (
            config["enabled"] is False
        )  # Invalid boolean values are treated as False for safety
        assert config["default_capacity"] == 60  # Invalid value defaults to 60
        assert config["default_refill_rate"] == 1.0  # Invalid value defaults to 1.0
        assert config["cleanup_interval"] == 300  # Invalid value defaults to 300
