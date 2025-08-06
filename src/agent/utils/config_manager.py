"""
Configuration Manager for centralized configuration handling.

This module provides a ConfigurationManager class that centralizes all configuration
loading and management, replacing the scattered get_*_config() functions with a
unified, testable, and maintainable solution.
"""

from functools import lru_cache
from os import getenv
from pathlib import Path
from typing import Dict, Any, Optional

from src.agent.utils.constants import EnvVars, Database, URLs


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


class ConfigurationManager:
    """
    Centralized configuration manager implementing the Singleton pattern.

    This class manages all application configuration through environment variables,
    providing caching and validation capabilities. It replaces multiple scattered
    get_*_config() functions with a unified interface.
    """

    _instance: Optional["ConfigurationManager"] = None

    def __new__(cls) -> "ConfigurationManager":
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def root_dir(self) -> str:
        """Get the root directory of the project."""
        return str(Path(__file__).resolve().parents[3])

    def _get_env_var(
        self, key: str, required: bool = True, default: Optional[str] = None
    ) -> Optional[str]:
        """
        Get environment variable with optional validation.

        Args:
            key: Environment variable key
            required: Whether the variable is required
            default: Default value if not set and not required

        Returns:
            The environment variable value or default

        Raises:
            ConfigurationError: If required variable is not set
        """
        value = getenv(key, default)
        if required and value is None:
            raise ConfigurationError(f"{key} not set in environment variables")
        return value

    def _build_path(self, relative_path: str) -> Path:
        """Build absolute path from relative path."""
        return Path(self.root_dir, relative_path)

    def _build_url(self, base: str, endpoint: str) -> str:
        """Build URL from base and endpoint."""
        return f"{base}/{endpoint}"

    @lru_cache(maxsize=1)
    def get_agent_config(self) -> Dict[str, Any]:
        """
        Get agent configuration including prompt file paths.

        Returns:
            Dictionary with agent configuration

        Raises:
            ConfigurationError: If required environment variables are missing
        """
        prompts_file = self._get_env_var(EnvVars.AGENT_PROMPTS_FILE)
        sql_prompts_file = self._get_env_var(EnvVars.SQL_PROMPTS_FILE)
        scenario_prompts_file = self._get_env_var(EnvVars.SCENARIO_PROMPTS_FILE)

        return {
            "prompt_path": self._build_path(prompts_file),
            "sql_prompt_path": self._build_path(sql_prompts_file),
            "scenario_prompt_path": self._build_path(scenario_prompts_file),
        }

    @lru_cache(maxsize=1)
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration.

        Returns:
            Dictionary with LLM configuration

        Raises:
            ConfigurationError: If required environment variables are missing
        """
        model_id = self._get_env_var(EnvVars.LLM_MODEL_ID)
        temperature = self._get_env_var(EnvVars.LLM_TEMPERATURE, required=False)

        return {
            "model_id": model_id,
            "temperature": temperature,
        }

    @lru_cache(maxsize=1)
    def get_guardrails_config(self) -> Dict[str, Any]:
        """
        Get guardrails configuration.

        Returns:
            Dictionary with guardrails configuration

        Raises:
            ConfigurationError: If required environment variables are missing
        """
        model_id = self._get_env_var(EnvVars.GUARDRAILS_MODEL_ID)
        temperature = self._get_env_var(EnvVars.GUARDRAILS_TEMPERATURE, required=False)

        return {
            "model_id": model_id,
            "temperature": temperature,
        }

    @lru_cache(maxsize=1)
    def get_rag_config(self) -> Dict[str, Any]:
        """
        Get RAG (Retrieval Augmented Generation) configuration.

        Returns:
            Dictionary with RAG configuration

        Raises:
            ConfigurationError: If required environment variables are missing
        """
        # Get base URLs and endpoints
        embedding_api_base = self._get_env_var(EnvVars.EMBEDDING_API_BASE)
        retrieval_api_base = self._get_env_var(EnvVars.RETRIEVAL_API_BASE)
        ranking_api_base = self._get_env_var(EnvVars.RANKING_API_BASE)

        embedding_endpoint = self._get_env_var(EnvVars.EMBEDDING_ENDPOINT)
        ranking_endpoint = self._get_env_var(EnvVars.RANKING_ENDPOINT)
        retrieval_endpoint = self._get_env_var(EnvVars.RETRIEVAL_ENDPOINT)

        # Get other configuration
        n_ranking_candidates = self._get_env_var(
            EnvVars.N_RANKING_CANDIDATES, required=False
        )
        n_retrieval_candidates = self._get_env_var(
            EnvVars.N_RETRIEVAL_CANDIDATES, required=False
        )
        retrieval_table = self._get_env_var(EnvVars.RETRIEVAL_TABLE)

        return {
            "embedding_url": self._build_url(embedding_api_base, embedding_endpoint),
            "ranking_url": self._build_url(ranking_api_base, ranking_endpoint),
            "retrieval_url": self._build_url(retrieval_api_base, retrieval_endpoint),
            "n_ranking_candidates": n_ranking_candidates,
            "n_retrieval_candidates": n_retrieval_candidates,
            "retrieval_table": retrieval_table,
        }

    @lru_cache(maxsize=1)
    def get_tools_config(self) -> Dict[str, Any]:
        """
        Get tools configuration.

        Returns:
            Dictionary with tools configuration

        Raises:
            ConfigurationError: If required environment variables are missing
        """
        llm_model_id = self._get_env_var(EnvVars.TOOLS_MODEL_ID)
        llm_api_base = self._get_env_var(EnvVars.TOOLS_MODEL_API_BASE, required=False)
        max_steps = self._get_env_var(EnvVars.TOOLS_MAX_STEPS, required=False)
        prompts_file = self._get_env_var(EnvVars.TOOLS_PROMPTS_FILE)
        tools_api_base = self._get_env_var(EnvVars.TOOLS_API_BASE)
        tools_api_limit = self._get_env_var(EnvVars.TOOLS_API_LIMIT, required=False)

        return {
            "llm_model_id": llm_model_id,
            "llm_api_base": llm_api_base,
            "max_steps": max_steps,
            "prompt_path": self._build_path(prompts_file),
            "tools_api_base": tools_api_base,
            "tools_api_limit": tools_api_limit,
        }

    @lru_cache(maxsize=1)
    def get_tracing_config(self) -> Dict[str, Any]:
        """
        Get tracing configuration.

        Returns:
            Dictionary with tracing configuration

        Raises:
            ConfigurationError: If required environment variables are missing
        """
        langfuse_public_key = self._get_env_var(EnvVars.LANGFUSE_PUBLIC_KEY)
        langfuse_secret_key = self._get_env_var(EnvVars.LANGFUSE_SECRET_KEY)
        langfuse_project_id = self._get_env_var(EnvVars.LANGFUSE_PROJECT_ID)
        langfuse_host = self._get_env_var(EnvVars.LANGFUSE_HOST)
        telemetry_enabled = self._get_env_var(
            EnvVars.TELEMETRY_ENABLED,
            required=False,
            default=Database.DEFAULT_TELEMETRY_ENABLED,
        )

        return {
            "langfuse_public_key": langfuse_public_key,
            "langfuse_project_id": langfuse_project_id,
            "langfuse_host": langfuse_host,
            "langfuse_secret_key": langfuse_secret_key,
            "otel_exporter_otlp_endpoint": URLs.LANGFUSE_OTEL_ENDPOINT,
            "telemetry_enabled": telemetry_enabled,
        }

    @lru_cache(maxsize=1)
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.

        Returns:
            Dictionary with logging configuration
        """
        logging_level = self._get_env_var(EnvVars.LOGGING_LEVEL, required=False)
        logging_format = self._get_env_var(EnvVars.LOGGING_FORMAT, required=False)

        return {
            "logging_level": logging_level,
            "logging_format": logging_format,
        }

    @lru_cache(maxsize=1)
    def get_email_config(self) -> Dict[str, Any]:
        """
        Get email configuration.

        Returns:
            Dictionary with email configuration
        """
        smtp_host = self._get_env_var(EnvVars.SMTP_HOST, required=False)
        smtp_port = self._get_env_var(EnvVars.SMTP_PORT, required=False)
        receiver_email = self._get_env_var(EnvVars.RECEIVER_EMAIL, required=False)
        sender_email = self._get_env_var(EnvVars.SENDER_EMAIL, required=False)
        app_password = self._get_env_var(EnvVars.APP_PASSWORD, required=False)

        return {
            "smtp_host": smtp_host,
            "smtp_port": smtp_port,
            "sender_email": sender_email,
            "receiver_email": receiver_email,
            "app_password": app_password,
        }

    @lru_cache(maxsize=1)
    def get_slack_config(self) -> Dict[str, Any]:
        """
        Get Slack configuration.

        Returns:
            Dictionary with Slack configuration
        """
        slack_webhook_url = self._get_env_var(EnvVars.SLACK_WEBHOOK_URL, required=False)

        return {
            "slack_webhook_url": slack_webhook_url,
        }

    @lru_cache(maxsize=1)
    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration.

        Returns:
            Dictionary with database configuration

        Raises:
            ConfigurationError: If required environment variables are missing
        """
        db_user = self._get_env_var(EnvVars.PG_USER)
        db_password = self._get_env_var(EnvVars.PG_PASSWORD)
        db_host = self._get_env_var(EnvVars.PG_HOST)
        db_port = self._get_env_var(EnvVars.PG_PORT)
        db_name = self._get_env_var(EnvVars.PG_NAME)
        database_type = self._get_env_var(
            EnvVars.DATABASE_TYPE,
            required=False,
            default=Database.DEFAULT_DATABASE_TYPE,
        )

        connection_string = Database.CONNECTION_STRING_TEMPLATE.format(
            user=db_user, password=db_password, host=db_host, port=db_port, name=db_name
        )

        return {
            "connection_string": connection_string,
            "db_type": database_type,
        }

    @lru_cache(maxsize=1)
    def get_evaluation_database_config(self) -> Dict[str, Any]:
        """
        Get evaluation database configuration.

        Returns:
            Dictionary with evaluation database configuration

        Raises:
            ConfigurationError: If required environment variables are missing
        """
        db_user = self._get_env_var(EnvVars.PG_USER)
        db_password = self._get_env_var(EnvVars.PG_PASSWORD)
        db_host = self._get_env_var(EnvVars.PG_HOST)
        db_port = self._get_env_var(EnvVars.PG_PORT)
        db_name = self._get_env_var(
            EnvVars.PG_EVAL_DB, required=False, default=Database.DEFAULT_EVAL_DB
        )

        connection_string = Database.CONNECTION_STRING_TEMPLATE.format(
            user=db_user, password=db_password, host=db_host, port=db_port, name=db_name
        )

        return {
            "connection_string": connection_string,
            "db_type": Database.TYPE_POSTGRES,
            "db_name": db_name,
        }

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all configuration sections.

        Returns:
            Dictionary with all configuration sections
        """
        return {
            "agent": self.get_agent_config(),
            "llm": self.get_llm_config(),
            "guardrails": self.get_guardrails_config(),
            "rag": self.get_rag_config(),
            "tools": self.get_tools_config(),
            "tracing": self.get_tracing_config(),
            "logging": self.get_logging_config(),
            "email": self.get_email_config(),
            "slack": self.get_slack_config(),
            "database": self.get_database_config(),
            "evaluation_database": self.get_evaluation_database_config(),
        }

    def clear_cache(self) -> None:
        """Clear all cached configurations to force reload."""
        # Clear LRU cache for all methods
        self.get_agent_config.cache_clear()
        self.get_llm_config.cache_clear()
        self.get_guardrails_config.cache_clear()
        self.get_rag_config.cache_clear()
        self.get_tools_config.cache_clear()
        self.get_tracing_config.cache_clear()
        self.get_logging_config.cache_clear()
        self.get_email_config.cache_clear()
        self.get_slack_config.cache_clear()
        self.get_database_config.cache_clear()
        self.get_evaluation_database_config.cache_clear()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance for testing purposes."""
        cls._instance = None


# Backward compatibility functions that delegate to ConfigurationManager
def get_agent_config():
    """Backward compatibility function."""
    return ConfigurationManager().get_agent_config()


def get_llm_config():
    """Backward compatibility function."""
    return ConfigurationManager().get_llm_config()


def get_guardrails_config():
    """Backward compatibility function."""
    return ConfigurationManager().get_guardrails_config()


def get_rag_config():
    """Backward compatibility function."""
    return ConfigurationManager().get_rag_config()


def get_tools_config():
    """Backward compatibility function."""
    return ConfigurationManager().get_tools_config()


def get_tracing_config():
    """Backward compatibility function."""
    return ConfigurationManager().get_tracing_config()


def get_logging_config():
    """Backward compatibility function."""
    return ConfigurationManager().get_logging_config()


def get_email_config():
    """Backward compatibility function."""
    return ConfigurationManager().get_email_config()


def get_slack_config():
    """Backward compatibility function."""
    return ConfigurationManager().get_slack_config()


def get_database_config():
    """Backward compatibility function."""
    return ConfigurationManager().get_database_config()


def get_evaluation_database_config():
    """Backward compatibility function."""
    return ConfigurationManager().get_evaluation_database_config()
