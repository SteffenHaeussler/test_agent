from os import getenv
from pathlib import Path

from src.agent.utils.constants import EnvVars, ErrorMessages, Database, URLs

ROOTDIR: str = str(Path(__file__).resolve().parents[2])


def get_agent_config():
    prompts_file = getenv(EnvVars.AGENT_PROMPTS_FILE)

    if prompts_file is None:
        raise ValueError(ErrorMessages.PROMPTS_FILE_NOT_SET)

    sql_prompts_file = getenv(EnvVars.SQL_PROMPTS_FILE)

    if sql_prompts_file is None:
        raise ValueError(ErrorMessages.SQL_PROMPTS_FILE_NOT_SET)

    scenario_prompts_file = getenv(EnvVars.SCENARIO_PROMPTS_FILE)

    if scenario_prompts_file is None:
        raise ValueError(ErrorMessages.SCENARIO_PROMPTS_FILE_NOT_SET)

    prompt_path = Path(ROOTDIR, prompts_file)
    sql_prompt_path = Path(ROOTDIR, sql_prompts_file)
    scenario_prompt_path = Path(ROOTDIR, scenario_prompts_file)

    return dict(
        prompt_path=prompt_path,
        sql_prompt_path=sql_prompt_path,
        scenario_prompt_path=scenario_prompt_path,
    )


def get_llm_config():
    model_id = getenv(EnvVars.LLM_MODEL_ID)
    temperature = getenv(EnvVars.LLM_TEMPERATURE)

    if model_id is None:
        raise ValueError(ErrorMessages.LLM_MODEL_ID_NOT_SET)

    return dict(model_id=model_id, temperature=temperature)


def get_guardrails_config():
    model_id = getenv(EnvVars.GUARDRAILS_MODEL_ID)
    temperature = getenv(EnvVars.GUARDRAILS_TEMPERATURE)
    if model_id is None:
        raise ValueError(ErrorMessages.GUARDRAILS_MODEL_ID_NOT_SET)

    return dict(model_id=model_id, temperature=temperature)


def get_rag_config():
    embedding_api_base = getenv(EnvVars.EMBEDDING_API_BASE)
    retrieval_api_base = getenv(EnvVars.RETRIEVAL_API_BASE)
    ranking_api_base = getenv(EnvVars.RANKING_API_BASE)

    embedding_endpoint = getenv(EnvVars.EMBEDDING_ENDPOINT)
    ranking_endpoint = getenv(EnvVars.RANKING_ENDPOINT)
    retrieval_endpoint = getenv(EnvVars.RETRIEVAL_ENDPOINT)

    n_ranking_candidates = getenv(EnvVars.N_RANKING_CANDIDATES)
    n_retrieval_candidates = getenv(EnvVars.N_RETRIEVAL_CANDIDATES)
    retrieval_table = getenv(EnvVars.RETRIEVAL_TABLE)
    if embedding_api_base is None or embedding_endpoint is None:
        raise ValueError(ErrorMessages.EMBEDDING_API_BASE_OR_ENDPOINT_NOT_SET)

    if retrieval_api_base is None or retrieval_endpoint is None:
        raise ValueError(ErrorMessages.RETRIEVAL_API_BASE_OR_ENDPOINT_NOT_SET)

    if ranking_api_base is None or ranking_endpoint is None:
        raise ValueError(ErrorMessages.RANKING_API_BASE_OR_ENDPOINT_NOT_SET)

    if retrieval_table is None:
        raise ValueError(ErrorMessages.RETRIEVAL_TABLE_NOT_SET)

    embedding_url = f"{embedding_api_base}/{embedding_endpoint}"
    ranking_url = f"{ranking_api_base}/{ranking_endpoint}"
    retrieval_url = f"{retrieval_api_base}/{retrieval_endpoint}"

    return dict(
        embedding_url=embedding_url,
        ranking_url=ranking_url,
        retrieval_url=retrieval_url,
        n_ranking_candidates=n_ranking_candidates,
        n_retrieval_candidates=n_retrieval_candidates,
        retrieval_table=retrieval_table,
    )


def get_tools_config():
    llm_model_id = getenv(EnvVars.TOOLS_MODEL_ID)
    llm_api_base = getenv(EnvVars.TOOLS_MODEL_API_BASE)
    max_steps = getenv(EnvVars.TOOLS_MAX_STEPS)
    prompts_file = getenv(EnvVars.TOOLS_PROMPTS_FILE)
    tools_api_base = getenv(EnvVars.TOOLS_API_BASE)
    tools_api_limit = getenv(EnvVars.TOOLS_API_LIMIT)

    if llm_model_id is None:
        raise ValueError(ErrorMessages.TOOLS_MODEL_ID_NOT_SET)

    if prompts_file is None:
        raise ValueError(ErrorMessages.TOOLS_PROMPTS_FILE_NOT_SET)

    if tools_api_base is None:
        raise ValueError(ErrorMessages.TOOLS_API_BASE_NOT_SET)

    prompt_path = Path(ROOTDIR, prompts_file)

    return dict(
        llm_model_id=llm_model_id,
        llm_api_base=llm_api_base,
        max_steps=max_steps,
        prompt_path=prompt_path,
        tools_api_base=tools_api_base,
        tools_api_limit=tools_api_limit,
    )


def get_tracing_config():
    langfuse_public_key = getenv(EnvVars.LANGFUSE_PUBLIC_KEY)
    langfuse_secret_key = getenv(EnvVars.LANGFUSE_SECRET_KEY)
    langfuse_project_id = getenv(EnvVars.LANGFUSE_PROJECT_ID)
    langfuse_host = getenv(EnvVars.LANGFUSE_HOST)
    otel_exporter_otlp_endpoint = URLs.LANGFUSE_OTEL_ENDPOINT
    telemetry_enabled = getenv(
        EnvVars.TELEMETRY_ENABLED, Database.DEFAULT_TELEMETRY_ENABLED
    )

    if langfuse_public_key is None:
        raise ValueError(ErrorMessages.LANGFUSE_PUBLIC_KEY_NOT_SET)

    if langfuse_project_id is None:
        raise ValueError(ErrorMessages.LANGFUSE_PROJECT_ID_NOT_SET)

    if langfuse_host is None:
        raise ValueError(ErrorMessages.LANGFUSE_HOST_NOT_SET)

    if langfuse_secret_key is None:
        raise ValueError(ErrorMessages.LANGFUSE_SECRET_KEY_NOT_SET)

    return dict(
        langfuse_public_key=langfuse_public_key,
        langfuse_project_id=langfuse_project_id,
        langfuse_host=langfuse_host,
        langfuse_secret_key=langfuse_secret_key,
        otel_exporter_otlp_endpoint=otel_exporter_otlp_endpoint,
        telemetry_enabled=telemetry_enabled,
    )


def get_logging_config():
    logging_level = getenv(EnvVars.LOGGING_LEVEL)
    logging_format = getenv(EnvVars.LOGGING_FORMAT)

    return dict(logging_level=logging_level, logging_format=logging_format)


def get_email_config():
    smtp_host = getenv(EnvVars.SMTP_HOST)
    smtp_port = getenv(EnvVars.SMTP_PORT)
    receiver_email = getenv(EnvVars.RECEIVER_EMAIL)
    sender_email = getenv(EnvVars.SENDER_EMAIL)
    app_password = getenv(EnvVars.APP_PASSWORD)

    return dict(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        sender_email=sender_email,
        receiver_email=receiver_email,
        app_password=app_password,
    )


def get_slack_config():
    slack_webhook_url = getenv(EnvVars.SLACK_WEBHOOK_URL)

    return dict(slack_webhook_url=slack_webhook_url)


def get_database_config():
    db_user = getenv(EnvVars.PG_USER)
    db_password = getenv(EnvVars.PG_PASSWORD)
    db_host = getenv(EnvVars.PG_HOST)
    db_port = getenv(EnvVars.PG_PORT)
    db_name = getenv(EnvVars.PG_NAME)

    database_connection_string = Database.CONNECTION_STRING_TEMPLATE.format(
        user=db_user, password=db_password, host=db_host, port=db_port, name=db_name
    )
    database_type = getenv(EnvVars.DATABASE_TYPE, Database.DEFAULT_DATABASE_TYPE)

    if database_connection_string is None:
        raise ValueError(ErrorMessages.DATABASE_CONNECTION_STRING_NOT_SET)

    return dict(
        connection_string=database_connection_string,
        db_type=database_type,
    )


def get_evaluation_database_config():
    """Get configuration for the evaluation database."""
    db_user = getenv(EnvVars.PG_USER)
    db_password = getenv(EnvVars.PG_PASSWORD)
    db_host = getenv(EnvVars.PG_HOST)
    db_port = getenv(EnvVars.PG_PORT)
    db_name = getenv(EnvVars.PG_EVAL_DB, Database.DEFAULT_EVAL_DB)

    evaluation_connection_string = Database.CONNECTION_STRING_TEMPLATE.format(
        user=db_user, password=db_password, host=db_host, port=db_port, name=db_name
    )

    if not all([db_user, db_password, db_host, db_port]):
        raise ValueError(ErrorMessages.POSTGRESQL_CONNECTION_PARAMETERS_NOT_SET)

    return dict(
        connection_string=evaluation_connection_string,
        db_type=Database.TYPE_POSTGRES,
        db_name=db_name,
    )
