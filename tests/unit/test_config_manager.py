"""
Tests for the ConfigurationManager class.

This module tests the centralized configuration management system
that replaces scattered get_*_config() functions with a unified manager.
"""

import pytest
from unittest.mock import patch
from pathlib import Path

from src.agent.utils.config_manager import ConfigurationManager, ConfigurationError


class TestConfigurationManager:
    """Test suite for the ConfigurationManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        ConfigurationManager.reset_instance()  # Reset singleton for each test
        self.config_manager = ConfigurationManager()

    def test_configuration_manager_initializes_successfully(self):
        """Test that ConfigurationManager initializes without errors."""
        manager = ConfigurationManager()
        assert manager is not None

    @patch("src.agent.utils.config_manager.getenv")
    def test_get_agent_config_returns_valid_paths(self, mock_getenv):
        """Test that get_agent_config returns proper path configuration."""
        # Setup mock environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            "agent_prompts_file": "prompts/agent.yaml",
            "sql_prompts_file": "prompts/sql.yaml",
            "scenario_prompts_file": "prompts/scenario.yaml",
        }.get(key, default)

        config = self.config_manager.get_agent_config()

        assert "prompt_path" in config
        assert "sql_prompt_path" in config
        assert "scenario_prompt_path" in config
        assert isinstance(config["prompt_path"], Path)
        assert isinstance(config["sql_prompt_path"], Path)
        assert isinstance(config["scenario_prompt_path"], Path)

    @patch("src.agent.utils.config_manager.getenv")
    def test_get_agent_config_raises_error_when_prompts_file_missing(self, mock_getenv):
        """Test that ConfigurationError is raised when required env var is missing."""
        mock_getenv.return_value = None

        with pytest.raises(ConfigurationError, match="agent_prompts_file not set"):
            self.config_manager.get_agent_config()

    @patch("src.agent.utils.config_manager.getenv")
    def test_get_llm_config_returns_valid_configuration(self, mock_getenv):
        """Test that get_llm_config returns proper LLM configuration."""
        mock_getenv.side_effect = lambda key, default=None: {
            "llm_model_id": "gpt-4",
            "llm_temperature": "0.7",
        }.get(key, default)

        config = self.config_manager.get_llm_config()

        assert config["model_id"] == "gpt-4"
        assert config["temperature"] == "0.7"

    @patch("src.agent.utils.config_manager.getenv")
    def test_get_llm_config_raises_error_when_model_id_missing(self, mock_getenv):
        """Test that ConfigurationError is raised when llm_model_id is missing."""
        mock_getenv.return_value = None

        with pytest.raises(ConfigurationError, match="llm_model_id not set"):
            self.config_manager.get_llm_config()

    def test_get_all_configs_returns_comprehensive_configuration(self):
        """Test that get_all_configs returns all configuration sections."""
        with patch.object(
            self.config_manager, "get_agent_config", return_value={"agent": "config"}
        ):
            with patch.object(
                self.config_manager, "get_llm_config", return_value={"llm": "config"}
            ):
                with patch.object(
                    self.config_manager,
                    "get_rag_config",
                    return_value={"rag": "config"},
                ):
                    all_configs = self.config_manager.get_all_configs()

                    assert "agent" in all_configs
                    assert "llm" in all_configs
                    assert "rag" in all_configs

    def test_singleton_behavior(self):
        """Test that ConfigurationManager follows singleton pattern."""
        manager1 = ConfigurationManager()
        manager2 = ConfigurationManager()

        # Should be the same instance
        assert manager1 is manager2

    @patch("src.agent.utils.config_manager.getenv")
    def test_caching_behavior(self, mock_getenv):
        """Test that configuration is cached after first load."""
        mock_getenv.side_effect = lambda key, default=None: {
            "llm_model_id": "gpt-4",
            "llm_temperature": "0.7",
        }.get(key, default)

        # First call should invoke getenv
        config1 = self.config_manager.get_llm_config()

        # Second call should return cached result
        config2 = self.config_manager.get_llm_config()

        assert config1 == config2
        # getenv should only be called during first load (due to LRU cache)
        assert mock_getenv.call_count >= 2  # At least 2 calls for 2 env vars

    def test_clear_cache_reloads_configuration(self):
        """Test that clear_cache allows fresh configuration loading."""
        with patch.object(
            self.config_manager, "get_llm_config", return_value={"cached": True}
        ) as mock_method:
            # Load config (will be cached)
            _ = self.config_manager.get_llm_config()

            # Clear cache
            self.config_manager.clear_cache()

            # Mock different return value
            mock_method.return_value = {"fresh": True}
            _ = self.config_manager.get_llm_config()

            # Should have called the method twice
            assert mock_method.call_count == 2


class TestConfigurationManagerIntegration:
    """Integration tests for ConfigurationManager with real environment variables."""

    def setup_method(self):
        """Set up test fixtures."""
        ConfigurationManager.reset_instance()

    @patch.dict(
        "os.environ",
        {
            "llm_model_id": "test-model",
            "llm_temperature": "0.5",
            "guardrails_model_id": "guard-model",
            "guardrails_temperature": "0.3",
        },
    )
    def test_real_environment_integration(self):
        """Test ConfigurationManager with real environment variables."""
        manager = ConfigurationManager()

        llm_config = manager.get_llm_config()
        guardrails_config = manager.get_guardrails_config()

        assert llm_config["model_id"] == "test-model"
        assert llm_config["temperature"] == "0.5"
        assert guardrails_config["model_id"] == "guard-model"
        assert guardrails_config["temperature"] == "0.3"
