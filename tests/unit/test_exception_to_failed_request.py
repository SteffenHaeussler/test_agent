"""
Test exception handling and automatic FailedRequest event creation.

This test demonstrates the NEW behavior where exceptions automatically
create FailedRequest events to notify users via WebSocket.
"""

from unittest.mock import Mock

from src.agent.adapters import adapter as AbstractAdapter
from src.agent.domain import commands, events
from src.agent.exceptions import DatabaseConnectionException
from src.agent.service_layer import messagebus


class TestExceptionHandling:
    """Test automatic exception to FailedRequest conversion."""

    def test_new_behavior_exception_creates_failed_request(self):
        """
        Test that exceptions now automatically create FailedRequest events
        instead of being re-raised.
        """
        # Arrange
        adapter = Mock(spec=AbstractAdapter)
        adapter.collect_new_events = Mock(return_value=[])

        notifications = Mock()

        # Create a handler that raises DatabaseConnectionException
        def failing_handler(command):
            raise DatabaseConnectionException(
                "Failed to connect to database",
                context={"host": "localhost", "port": 5432},
            )

        command_handlers = {commands.Question: failing_handler}

        failed_request_handler = Mock()
        event_handlers = {
            events.FailedRequest: [failed_request_handler]  # This WILL be called now
        }

        bus = messagebus.MessageBus(
            adapter=adapter,
            event_handlers=event_handlers,
            command_handlers=command_handlers,
            notifications=notifications,
        )

        question = commands.Question(
            question="What is the sales data?", q_id="test-123"
        )

        # Act
        bus.handle(question)  # No longer raises exception

        # Assert
        # Verify that FailedRequest event WAS created (new behavior)
        failed_request_handler.assert_called_once()

        # Check the failed request event details
        failed_event = failed_request_handler.call_args[0][0]
        assert isinstance(failed_event, events.FailedRequest)
        assert failed_event.question == "What is the sales data?"
        assert failed_event.q_id == "test-123"
        assert "Failed to connect to database" in failed_event.exception

        # Verify adapter.collect_new_events was still called
        assert adapter.collect_new_events.called

    def test_exception_context_formatting(self):
        """
        Test how exception context is formatted for user messages.
        """
        # Create an exception with context
        exception = DatabaseConnectionException(
            "Connection timeout",
            context={
                "host": "db.example.com",
                "port": 5432,
                "connection_string": "postgresql://user:password@db.example.com/prod",
                "timeout_seconds": 30,
                "retry_count": 3,
            },
        )

        # When we create a FailedRequest from this exception,
        # the message should be just the exception string (no sensitive context)
        assert str(exception) == "Connection timeout"

        # The sanitized context can be logged separately but isn't sent to users
        sanitized_context = exception.get_sanitized_context()
        assert sanitized_context["connection_string"] == "[FILTERED]"
        assert sanitized_context["host"] == "db.example.com"
        assert sanitized_context["port"] == 5432
