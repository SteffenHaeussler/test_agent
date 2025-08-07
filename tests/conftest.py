# tests/conftest.py
import os
import asyncio
from unittest.mock import patch

import pytest
from dotenv import load_dotenv


def pytest_addoption(parser):
    parser.addoption(
        "--envfile",
        action="store",
        default=".env.tests",
        help="Specify the .env file to load for tests",
    )


def pytest_configure(config):
    os.environ["IS_TESTING"] = "true"  # Set early
    env_file = config.getoption("--envfile")
    load_dotenv(env_file, override=True)

    # Set fast retry settings for tests
    os.environ["TEST_MAX_RETRIES"] = "1"  # Reduce retries
    os.environ["TEST_BASE_DELAY"] = "0.01"  # 10ms instead of 1s
    os.environ["TEST_MAX_DELAY"] = "0.02"  # 20ms instead of 60s


@pytest.fixture(scope="session", autouse=True)
def load_test_environment_fixture():
    assert os.getenv("IS_TESTING") == "true", "IS_TESTING not true in fixture!"


@pytest.fixture(autouse=True)
def fast_retry_delays():
    """
    Automatically patch retry delays for all tests to speed up execution.
    Tests that need real delays can override this fixture.
    """
    import asyncio

    # Store original sleep function
    original_sleep = asyncio.sleep

    async def fast_sleep(seconds):
        """Sleep for a much shorter duration in tests."""
        if seconds > 0.1:  # If sleep is more than 100ms
            # Cap it at 10ms for tests
            await original_sleep(0.01)
        else:
            # Keep short sleeps as-is for proper async behavior
            await original_sleep(seconds)

    # Patch asyncio.sleep globally for all tests
    with patch("asyncio.sleep", fast_sleep):
        yield


@pytest.fixture
def mock_async_sleep(monkeypatch):
    """Mock asyncio.sleep to make async tests run instantly."""

    async def instant_sleep(seconds):
        """Sleep for 0 seconds instead of the requested time."""
        await asyncio.sleep(0)  # Yield control but don't actually wait
        return

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    return instant_sleep


@pytest.fixture
def mock_time_sleep(monkeypatch):
    """Mock time.sleep to make sync tests run instantly."""
    import time

    def instant_sleep(seconds):
        """Sleep for 0 seconds instead of the requested time."""
        return

    monkeypatch.setattr(time, "sleep", instant_sleep)
    return instant_sleep
