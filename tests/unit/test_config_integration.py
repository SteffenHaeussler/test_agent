"""
Tests for ConfigurationManager integration with BaseAgent.

This module tests the integration of ConfigurationManager with BaseAgent,
ensuring that configuration is properly centralized and accessible.
"""

import pytest
from unittest.mock import patch

from src.agent.domain import commands
from src.agent.domain.model import BaseAgent
from src.agent.utils.config_manager import ConfigurationManager


class TestConfigurationManagerIntegration:
    """Test suite for ConfigurationManager integration with BaseAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        ConfigurationManager.reset_instance()  # Reset singleton for each test

    @patch("src.agent.domain.model.yaml.safe_load")
    @patch("builtins.open")
    @patch("src.agent.utils.config_manager.getenv")
    def test_base_agent_uses_configuration_manager_for_prompt_path(
        self, mock_getenv, mock_open, mock_yaml
    ):
        """Test that BaseAgent gets prompt path from ConfigurationManager when kwargs don't contain it."""
        # Setup ConfigurationManager mock
        mock_getenv.side_effect = lambda key, default=None: {
            "agent_prompts_file": "prompts/agent.yaml",
            "sql_prompts_file": "prompts/sql.yaml",
            "scenario_prompts_file": "prompts/scenario.yaml",
        }.get(key, default)

        # Setup yaml mock
        mock_yaml.return_value = {"test": "prompts"}

        # Create BaseAgent without prompt_path in kwargs
        question = commands.Question(question="Test question", q_id="test")
        agent = BaseAgent(question, kwargs={})

        # Verify that ConfigurationManager was used to get the prompt path
        assert agent.config_manager is not None
        assert isinstance(agent.config_manager, ConfigurationManager)

    @patch("src.agent.domain.model.yaml.safe_load")
    @patch("builtins.open")
    def test_base_agent_backward_compatibility_with_kwargs(self, mock_open, mock_yaml):
        """Test that BaseAgent maintains backward compatibility when prompt_path is in kwargs."""
        mock_yaml.return_value = {"test": "prompts"}

        question = commands.Question(question="Test question", q_id="test")
        kwargs = {"prompt_path": "/custom/path/prompts.yaml"}
        agent = BaseAgent(question, kwargs=kwargs)

        # Should still work with existing pattern
        assert agent.base_prompts == {"test": "prompts"}

        # Should have opened the custom path
        mock_open.assert_called_with("/custom/path/prompts.yaml", "r")

    @patch("src.agent.domain.model.yaml.safe_load")
    @patch("builtins.open")
    @patch("src.agent.utils.config_manager.getenv")
    def test_base_agent_can_access_all_configurations(
        self, mock_getenv, mock_open, mock_yaml
    ):
        """Test that BaseAgent can access all configuration sections through ConfigurationManager."""
        # Setup mocks for all required environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            "agent_prompts_file": "prompts/agent.yaml",
            "sql_prompts_file": "prompts/sql.yaml",
            "scenario_prompts_file": "prompts/scenario.yaml",
            "llm_model_id": "gpt-4",
            "llm_temperature": "0.7",
            "guardrails_model_id": "guard-model",
        }.get(key, default)

        mock_yaml.return_value = {"test": "prompts"}

        question = commands.Question(question="Test question", q_id="test")
        agent = BaseAgent(question, kwargs={})

        # Should be able to access different configuration sections
        llm_config = agent.config_manager.get_llm_config()
        assert llm_config["model_id"] == "gpt-4"
        assert llm_config["temperature"] == "0.7"

        guardrails_config = agent.config_manager.get_guardrails_config()
        assert guardrails_config["model_id"] == "guard-model"

    def test_configuration_manager_singleton_behavior_in_multiple_agents(self):
        """Test that multiple BaseAgent instances share the same ConfigurationManager instance."""
        with patch("src.agent.domain.model.yaml.safe_load"):
            with patch("builtins.open"):
                question1 = commands.Question(question="Test 1", q_id="test-1")
                question2 = commands.Question(question="Test 2", q_id="test-2")

                agent1 = BaseAgent(question1, kwargs={"prompt_path": "test.yaml"})
                agent2 = BaseAgent(question2, kwargs={"prompt_path": "test.yaml"})

                # Both agents should have the same ConfigurationManager instance
                assert agent1.config_manager is agent2.config_manager

    @patch("src.agent.domain.model.yaml.safe_load")
    @patch("builtins.open")
    @patch("src.agent.utils.config_manager.getenv")
    def test_configuration_manager_handles_missing_environment_variables_gracefully(
        self, mock_getenv, mock_open, mock_yaml
    ):
        """Test that ConfigurationManager handles missing environment variables gracefully."""
        # Setup minimal required environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            "agent_prompts_file": "prompts/agent.yaml",
            "sql_prompts_file": "prompts/sql.yaml",
            "scenario_prompts_file": "prompts/scenario.yaml",
        }.get(key, default)

        mock_yaml.return_value = {"test": "prompts"}

        question = commands.Question(question="Test question", q_id="test")

        # Should not fail even with minimal environment variables
        agent = BaseAgent(question, kwargs={})
        assert agent.config_manager is not None

        # Should be able to get agent config
        agent_config = agent.config_manager.get_agent_config()
        assert "prompt_path" in agent_config


class TestConfigurationManagerErrorHandling:
    """Test suite for ConfigurationManager error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        ConfigurationManager.reset_instance()

    @patch("src.agent.utils.config_manager.getenv")
    def test_configuration_manager_raises_error_for_missing_required_vars(
        self, mock_getenv
    ):
        """Test that ConfigurationManager raises appropriate errors for missing required variables."""
        mock_getenv.return_value = None  # Simulate missing environment variables

        config_manager = ConfigurationManager()

        # Should raise ConfigurationError for missing required variables
        from src.agent.utils.config_manager import ConfigurationError

        with pytest.raises(ConfigurationError, match="agent_prompts_file not set"):
            config_manager.get_agent_config()

    @patch("src.agent.domain.model.yaml.safe_load")
    @patch("builtins.open")
    @patch("src.agent.utils.config_manager.getenv")
    def test_base_agent_handles_configuration_errors_gracefully(
        self, mock_getenv, mock_open, mock_yaml
    ):
        """Test that BaseAgent handles configuration errors gracefully."""
        # Simulate ConfigurationManager failing
        mock_getenv.return_value = None
        mock_yaml.return_value = {"test": "prompts"}

        question = commands.Question(question="Test question", q_id="test")

        # When ConfigurationManager fails, should fall back to kwargs or raise appropriate error
        with pytest.raises(
            (ValueError, Exception)
        ):  # Should handle configuration errors appropriately
            BaseAgent(question, kwargs={})
