"""
Constants for the agentic AI framework.

This module centralizes all magic strings, configuration keys, and repeated
literals used throughout the codebase to improve maintainability and reduce
the risk of typos.

The constants are organized into logical groups:
- PROMPT_KEYS: Keys used in prompt templates
- ENV_VARS: Environment variable names
- STATUS_MESSAGES: Step status messages for UI updates
- ERROR_MESSAGES: Error messages and exception strings
- DATABASE: Database-related constants
- SECURITY: Security and sensitive data patterns
- URLS: URL patterns and endpoints
"""

from typing import List, Set


# =============================================================================
# PROMPT KEYS
# =============================================================================
class PromptKeys:
    """Keys used in prompt template lookups."""

    FINALIZE = "finalize"
    ENHANCE = "enhance"
    GUARDRAILS = "guardrails"
    PRE_CHECK = "pre_check"
    POST_CHECK = "post_check"


# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
class EnvVars:
    """Environment variable names used throughout the application."""

    # Agent configuration
    AGENT_PROMPTS_FILE = "agent_prompts_file"
    SQL_PROMPTS_FILE = "sql_prompts_file"
    SCENARIO_PROMPTS_FILE = "scenario_prompts_file"

    # LLM configuration
    LLM_MODEL_ID = "llm_model_id"
    LLM_TEMPERATURE = "llm_temperature"

    # Guardrails configuration
    GUARDRAILS_MODEL_ID = "guardrails_model_id"
    GUARDRAILS_TEMPERATURE = "guardrails_temperature"

    # RAG configuration
    EMBEDDING_API_BASE = "embedding_api_base"
    RETRIEVAL_API_BASE = "retrieval_api_base"
    RANKING_API_BASE = "ranking_api_base"
    EMBEDDING_ENDPOINT = "embedding_endpoint"
    RANKING_ENDPOINT = "ranking_endpoint"
    RETRIEVAL_ENDPOINT = "retrieval_endpoint"
    N_RANKING_CANDIDATES = "n_ranking_candidates"
    N_RETRIEVAL_CANDIDATES = "n_retrieval_candidates"
    RETRIEVAL_TABLE = "retrieval_table"

    # Tools configuration
    TOOLS_MODEL_ID = "tools_model_id"
    TOOLS_MODEL_API_BASE = "tools_model_api_base"
    TOOLS_MAX_STEPS = "tools_max_steps"
    TOOLS_PROMPTS_FILE = "tools_prompts_file"
    TOOLS_API_BASE = "tools_api_base"
    TOOLS_API_LIMIT = "tools_api_limit"

    # Tracing configuration
    LANGFUSE_PUBLIC_KEY = "langfuse_public_key"
    LANGFUSE_SECRET_KEY = "langfuse_secret_key"
    LANGFUSE_PROJECT_ID = "langfuse_project_id"
    LANGFUSE_HOST = "langfuse_host"
    TELEMETRY_ENABLED = "telemetry_enabled"

    # Logging configuration
    LOGGING_LEVEL = "logging_level"
    LOGGING_FORMAT = "logging_format"

    # Email configuration
    SMTP_HOST = "smtp_host"
    SMTP_PORT = "smtp_port"
    RECEIVER_EMAIL = "receiver_email"
    SENDER_EMAIL = "sender_email"
    APP_PASSWORD = "app_password"

    # Slack configuration
    SLACK_WEBHOOK_URL = "slack_webhook_url"

    # Database configuration
    PG_USER = "PG_USER"
    PG_PASSWORD = "PG_PASSWORD"
    PG_HOST = "PG_HOST"
    PG_PORT = "PG_PORT"
    PG_NAME = "PG_NAME"
    PG_EVAL_DB = "PG_EVAL_DB"
    DATABASE_TYPE = "database_type"


# =============================================================================
# STATUS MESSAGES
# =============================================================================
class StatusMessages:
    """Status messages displayed during different processing steps."""

    PROCESSING = "Processing..."
    CHECKING = "Checking..."
    RETRIEVING = "Retrieving..."
    ENHANCING = "Enhancing..."
    FINETUNING = "Finetuning..."
    ANSWERING = "Answering..."
    FINALIZING = "Finalizing..."
    EVALUATING = "Evaluating..."
    GROUNDING = "Grounding..."
    FILTERING = "Filtering..."
    JOINING = "Joining..."
    AGGREGATING = "Aggregating..."
    CONSTRUCTING = "Constructing..."
    VALIDATING = "Validating..."
    EXECUTING = "Executing..."
    THINKING = "Thinking..."


# =============================================================================
# ERROR MESSAGES
# =============================================================================
class ErrorMessages:
    """Standard error messages used throughout the application."""

    DUPLICATE_COMMAND = "Internal error: Duplicate command"
    NO_QUESTION_ASKED = "No question asked"
    QUESTION_REQUIRED_ENHANCE = "Question is required to enhance"
    TOOL_ANSWER_REQUIRED = "Tool answer is required for LLM response"
    INVALID_COMMAND_TYPE = "Invalid command type"
    PROMPT_NOT_FOUND = "Prompt not found"

    # Configuration errors
    PROMPTS_FILE_NOT_SET = "prompts_file not set in environment variables"
    SQL_PROMPTS_FILE_NOT_SET = "sql_prompts_file not set in environment variables"
    SCENARIO_PROMPTS_FILE_NOT_SET = (
        "scenario_prompts_file not set in environment variables"
    )
    LLM_MODEL_ID_NOT_SET = "llm_model_id not set in environment variables"
    GUARDRAILS_MODEL_ID_NOT_SET = "guardrails_model_id not set in environment variables"
    TOOLS_MODEL_ID_NOT_SET = "tools_model_id not set in environment variables"
    TOOLS_PROMPTS_FILE_NOT_SET = "tools_prompts_file not set in environment variables"
    TOOLS_API_BASE_NOT_SET = "tools_api_base not set in environment variables"
    LANGFUSE_PUBLIC_KEY_NOT_SET = "langfuse_public_key not set in environment variables"
    LANGFUSE_SECRET_KEY_NOT_SET = "langfuse_secret_key not set in environment variables"
    LANGFUSE_PROJECT_ID_NOT_SET = "langfuse_project_id not set in environment variables"
    LANGFUSE_HOST_NOT_SET = "langfuse_host not set in environment variables"
    EMBEDDING_API_BASE_OR_ENDPOINT_NOT_SET = (
        "embedding_api_base or embedding_endpoint not set in environment variables"
    )
    RETRIEVAL_API_BASE_OR_ENDPOINT_NOT_SET = (
        "retrieval_api_base or retrieval_endpoint not set in environment variables"
    )
    RANKING_API_BASE_OR_ENDPOINT_NOT_SET = (
        "ranking_api_base or ranking_endpoint not set in environment variables"
    )
    RETRIEVAL_TABLE_NOT_SET = "retrieval_table not set in environment variables"
    DATABASE_CONNECTION_STRING_NOT_SET = (
        "database_connection_string not set in environment variables"
    )
    POSTGRESQL_CONNECTION_PARAMETERS_NOT_SET = (
        "PostgreSQL connection parameters not set in environment variables"
    )

    # File and path errors
    PROMPT_PATH_NOT_FOUND = "Prompt path not found: {path}"
    ERROR_LOADING_PROMPTS = "Error loading prompts: {error}"
    CONFIG_PATH_ERROR = "Could not get prompt path from configuration: {error}"


# =============================================================================
# DATABASE CONSTANTS
# =============================================================================
class Database:
    """Database-related constants."""

    TYPE_POSTGRES = "postgres"
    DEFAULT_EVAL_DB = "evaluation"
    CONNECTION_STRING_TEMPLATE = (
        "postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    )

    # Default values
    DEFAULT_DATABASE_TYPE = "postgres"
    DEFAULT_TELEMETRY_ENABLED = "false"


# =============================================================================
# SECURITY CONSTANTS
# =============================================================================
class Security:
    """Security-related constants for sensitive data filtering."""

    FILTERED_PLACEHOLDER = "[FILTERED]"

    # Sensitive keys that should be filtered from logs
    SENSITIVE_KEYS: Set[str] = {
        "password",
        "passwd",
        "pwd",
        "secret",
        "api_key",
        "token",
        "auth",
        "authorization",
        "credential",
        "key",
        "private_key",
        "connection_string",
        "database_url",
        "dsn",
    }

    # Patterns for sensitive data in string values
    SENSITIVE_PATTERNS: List[str] = [
        r"password=[\w\-_]+",
        r"://[^:]+:[^@]+@",  # URLs with credentials
        r"Bearer\s+[\w\-_\.]+",  # Bearer tokens
        r"key_[\w\-_]+",  # API keys
    ]


# =============================================================================
# URL CONSTANTS
# =============================================================================
class URLs:
    """URL patterns and endpoints."""

    LANGFUSE_OTEL_ENDPOINT = "https://cloud.langfuse.com/api/public/otel"


# =============================================================================
# TRACE AND HANDLER NAMES
# =============================================================================
class TraceNames:
    """Names used for tracing and observability."""

    ANSWER_HANDLER = "answer handler"
    QUERY_HANDLER = "query handler"
    SEND_RESPONSE_HANDLER = "send_response handler"
    SEND_REJECTED_HANDLER = "send_rejected handler"
    SEND_STATUS_UPDATE_HANDLER = "send_status_update handler"
    FINALIZE = "finalize"
    ENHANCE = "enhance"


# =============================================================================
# EVENT MESSAGES
# =============================================================================
class EventMessages:
    """Messages used in events and responses."""

    END_RESPONSE = "end"


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================
# For backward compatibility, expose constants at module level
# This allows existing code to import constants directly

# Prompt keys
PROMPT_FINALIZE = PromptKeys.FINALIZE
PROMPT_ENHANCE = PromptKeys.ENHANCE
PROMPT_GUARDRAILS = PromptKeys.GUARDRAILS
PROMPT_PRE_CHECK = PromptKeys.PRE_CHECK
PROMPT_POST_CHECK = PromptKeys.POST_CHECK

# Common error messages
DUPLICATE_COMMAND_ERROR = ErrorMessages.DUPLICATE_COMMAND
QUESTION_REQUIRED_ERROR = ErrorMessages.QUESTION_REQUIRED_ENHANCE
NO_QUESTION_ERROR = ErrorMessages.NO_QUESTION_ASKED

# Database constants
POSTGRES_TYPE = Database.TYPE_POSTGRES
DEFAULT_DB_TYPE = Database.DEFAULT_DATABASE_TYPE

# Security
FILTERED = Security.FILTERED_PLACEHOLDER
SENSITIVE_KEYS = Security.SENSITIVE_KEYS
SENSITIVE_PATTERNS = Security.SENSITIVE_PATTERNS
