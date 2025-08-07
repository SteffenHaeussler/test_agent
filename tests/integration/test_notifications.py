from unittest.mock import patch
import pytest

from src.agent.adapters.adapter import AbstractAdapter
from src.agent.adapters.notifications import (
    CliNotifications,
    EmailNotifications,
    SlackNotifications,
    SSENotifications,
    WSNotifications,
)
from src.agent.bootstrap import bootstrap
from src.agent.domain import events


class MockAdapter(AbstractAdapter):
    """Simple adapter for notification testing that doesn't require agents."""

    def __init__(self):
        # Don't call super().__init__() to avoid complex adapter setup
        self.events = []

    def collect_new_events(self):
        """Return empty list since these tests don't create new events."""
        return []

    def add(self, agent):
        """Mock add method for interface compliance."""
        pass


class TestNotification:
    @pytest.mark.asyncio
    async def test_send_cli_notification_called(self):
        bus = bootstrap(
            adapter=MockAdapter(),
            notifications=[CliNotifications()],
        )
        event = events.Response(
            question="test_query",
            response="test_response",
            q_id="test_session_id",
        )
        with patch.object(CliNotifications, "send", return_value=None) as mock_send:
            await bus.handle(event)
            mock_send.assert_called_once_with("test_session_id", event)

    @pytest.mark.asyncio
    async def test_send_email_notification_called(self):
        bus = bootstrap(
            adapter=MockAdapter(),
            notifications=[EmailNotifications()],
        )
        event = events.Response(
            question="test_query",
            response="test_response",
            q_id="test_session_id",
        )

        with patch.object(EmailNotifications, "send", return_value=None) as mock_send:
            await bus.handle(event)
            mock_send.assert_called_once_with("test_session_id", event)

    @pytest.mark.asyncio
    async def test_send_slack_notification_called(self):
        event = events.Response(
            question="test_query",
            response="test_response",
            q_id="test_session_id",
        )

        bus = bootstrap(
            adapter=MockAdapter(),
            notifications=[SlackNotifications()],
        )
        with patch.object(SlackNotifications, "send", return_value=None) as mock_send:
            await bus.handle(event)
            mock_send.assert_called_once_with("test_session_id", event)

    @pytest.mark.asyncio
    async def test_send_ws_notification_called(self):
        event = events.Response(
            question="test_query",
            response="test_response",
            q_id="test_session_id",
        )

        bus = bootstrap(
            adapter=MockAdapter(),
            notifications=[WSNotifications()],
        )
        with patch.object(WSNotifications, "send", return_value=None) as mock_send:
            await bus.handle(event)
            mock_send.assert_called_once_with("test_session_id", event)

    @pytest.mark.asyncio
    async def test_send_sse_notification_called(self):
        event = events.Response(
            question="test_query",
            response="test_response",
            q_id="test_session_id",
        )
        bus = bootstrap(
            adapter=MockAdapter(),
            notifications=[SSENotifications()],
        )
        with patch.object(SSENotifications, "send", return_value=None) as mock_send:
            await bus.handle(event)
            mock_send.assert_called_once_with("test_session_id", event)
