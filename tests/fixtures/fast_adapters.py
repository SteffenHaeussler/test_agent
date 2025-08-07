"""
Fast adapter fixtures for testing that override retry delays.

These fixtures provide adapter instances with minimal retry delays
to speed up test execution.
"""

import pytest


@pytest.fixture
def fast_database_adapter():
    """Database adapter with fast retry settings for tests."""
    from src.agent.adapters.database import BaseDatabaseAdapter

    config = {
        "connection_string": "postgresql://test:test@localhost:5432/test",
        "db_type": "postgres",
        "max_retries": 1,  # Reduce from 3
        "base_delay": 0.001,  # 1ms instead of 1s
        "max_delay": 0.002,  # 2ms instead of 60s
    }
    return BaseDatabaseAdapter(config)


@pytest.fixture
def fast_llm_adapter():
    """LLM adapter with fast retry settings for tests."""
    from src.agent.adapters.llm import LLM

    config = {
        "llm_model": "test-model",
        "max_retries": 1,
        "base_delay": 0.001,
        "max_delay": 0.002,
    }
    return LLM(config)


@pytest.fixture
def fast_rag_adapter():
    """RAG adapter with fast retry settings for tests."""
    from src.agent.adapters.rag import BaseRAG

    config = {
        "embedding_url": "http://test-embedding",
        "ranking_url": "http://test-ranking",
        "retrieval_url": "http://test-retrieval",
        "max_retries": 1,
        "base_delay": 0.001,
        "max_delay": 0.002,
    }
    return BaseRAG(config)


@pytest.fixture
def instant_async_sleep(monkeypatch):
    """Replace asyncio.sleep with instant return for specific tests."""
    import asyncio

    original_sleep = asyncio.sleep
    call_count = {"count": 0}

    async def counting_instant_sleep(delay):
        """Track sleep calls but return instantly."""
        call_count["count"] += 1
        # For the first few calls, actually sleep a tiny amount to allow context switches
        if call_count["count"] <= 2:
            await original_sleep(0.0001)
        return

    monkeypatch.setattr("asyncio.sleep", counting_instant_sleep)
    return call_count


@pytest.fixture
def skip_health_checks(monkeypatch):
    """Skip health checks in adapters to speed up initialization."""

    async def mock_health_check(self):
        return True

    monkeypatch.setattr(
        "src.agent.adapters.database.BaseDatabaseAdapter.health_check",
        mock_health_check,
    )
    monkeypatch.setattr("src.agent.adapters.llm.LLM.health_check", mock_health_check)
    monkeypatch.setattr(
        "src.agent.adapters.rag.BaseRAG.health_check", mock_health_check
    )
