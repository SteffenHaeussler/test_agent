"""
Test suite for MessageBus error handling with automatic FailedRequest generation.

This tests that exceptions are automatically converted to FailedRequest events
so users are notified via WebSocket when errors occur.
"""

from unittest.mock import Mock

from src.agent.service_layer.messagebus import MessageBus
from src.agent.domain import commands, events
from src.agent.exceptions import (
    DatabaseConnectionException,
    ValidationException,
)


class TestMessageBusErrorHandling:
    """Test that MessageBus converts exceptions to FailedRequest events."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = Mock()
        self.adapter.collect_new_events = Mock(return_value=[])
        self.notifications = Mock()

        # Mock handlers
        self.command_handlers = {}
        self.event_handlers = {
            events.FailedRequest: [Mock()],
            events.Response: [Mock()],  # Add Response handler for test
        }

        self.bus = MessageBus(
            adapter=self.adapter,
            command_handlers=self.command_handlers,
            event_handlers=self.event_handlers,
            notifications=self.notifications,
        )

    def test_database_exception_creates_failed_request_event(self):
        """Test that DatabaseException creates a FailedRequest event."""
        # Arrange
        command = commands.Question(
            question="What is the total revenue?", q_id="test123"
        )

        def failing_handler(cmd):
            raise DatabaseConnectionException(
                "Failed to connect to database",
                context={"host": "localhost", "port": 5432},
            )

        self.command_handlers[commands.Question] = failing_handler

        # Act
        self.bus.handle(command)

        # Assert
        # Check that FailedRequest event handler was called
        failed_handler = self.event_handlers[events.FailedRequest][0]
        failed_handler.assert_called_once()

        # Verify the FailedRequest event content
        failed_event = failed_handler.call_args[0][0]
        assert isinstance(failed_event, events.FailedRequest)
        assert failed_event.question == "What is the total revenue?"
        assert failed_event.q_id == "test123"
        assert "Failed to connect to database" in failed_event.exception

    def test_generic_exception_creates_failed_request_event(self):
        """Test that generic exceptions also create FailedRequest events."""
        # Arrange
        command = commands.Question(question="Calculate metrics", q_id="test456")

        def failing_handler(cmd):
            raise RuntimeError("Unexpected error occurred")

        self.command_handlers[commands.Question] = failing_handler

        # Act
        self.bus.handle(command)

        # Assert
        failed_handler = self.event_handlers[events.FailedRequest][0]
        failed_handler.assert_called_once()

        failed_event = failed_handler.call_args[0][0]
        assert isinstance(failed_event, events.FailedRequest)
        assert failed_event.question == "Calculate metrics"
        assert "RuntimeError" in failed_event.exception

    def test_agent_exception_preserves_context_in_message(self):
        """Test that AgentException context is included in user message."""
        # Arrange
        command = commands.Question(question="Get user data", q_id="test789")

        def failing_handler(cmd):
            raise ValidationException(
                "Invalid input format",
                context={"field": "email", "value": "not-an-email"},
            )

        self.command_handlers[commands.Question] = failing_handler

        # Act
        self.bus.handle(command)

        # Assert
        failed_handler = self.event_handlers[events.FailedRequest][0]
        failed_event = failed_handler.call_args[0][0]

        # User-friendly message should include main error
        assert "Invalid input format" in failed_event.exception
        # But not include sensitive context details
        assert "not-an-email" not in failed_event.exception

    def test_exception_in_event_handler_continues_processing(self):
        """Test that exceptions in event handlers don't stop processing."""
        # Arrange
        command = commands.Question(question="Test query", q_id="test111")

        # First handler fails with exception
        def failing_command_handler(cmd):
            raise DatabaseConnectionException("Connection failed")

        # Event handler that also fails
        def failing_event_handler(event):
            raise RuntimeError("Event handler error")

        # Second event handler that should still run
        successful_handler = Mock()

        # Create new bus with custom handlers to avoid infinite loop
        bus = MessageBus(
            adapter=self.adapter,
            command_handlers={commands.Question: failing_command_handler},
            event_handlers={
                events.FailedRequest: [failing_event_handler, successful_handler]
            },
            notifications=self.notifications,
        )

        # Act
        bus.handle(command)

        # Assert
        # Second handler should still be called despite first handler exception
        successful_handler.assert_called_once()
        failed_event = successful_handler.call_args[0][0]
        assert isinstance(failed_event, events.FailedRequest)

    def test_command_without_question_field_handles_gracefully(self):
        """Test handling commands that don't have question/q_id fields."""
        # Arrange
        # Use a different command type that might not have question field
        command = Mock(spec=commands.Command)
        command.question = None  # No question field
        command.q_id = None

        def failing_handler(cmd):
            raise ValueError("Processing failed")

        # Create a separate failed request handler
        failed_request_handler = Mock()

        # Create new bus to avoid infinite loop
        bus = MessageBus(
            adapter=self.adapter,
            command_handlers={type(command): failing_handler},
            event_handlers={events.FailedRequest: [failed_request_handler]},
            notifications=self.notifications,
        )

        # Act
        bus.handle(command)

        # Assert
        failed_request_handler.assert_called_once()
        failed_event = failed_request_handler.call_args[0][0]

        # Should use defaults for missing fields
        assert failed_event.question == "Unknown"
        assert failed_event.q_id == "unknown"
        assert "ValueError" in failed_event.exception

    def test_original_behavior_preserved_for_successful_commands(self):
        """Test that successful commands work as before."""
        # Arrange
        command = commands.Question(question="Successful query", q_id="success123")

        successful_handler = Mock()
        response_handler = Mock()
        failed_handler = Mock()

        # Simulate agent creating a Response event
        response_event = events.Response(
            question="Successful query",
            response="42",  # Changed from 'answer' to 'response'
            q_id="success123",
        )
        adapter = Mock()
        # First call returns the response event, subsequent calls return empty
        adapter.collect_new_events = Mock(side_effect=[[response_event], []])

        # Create clean bus for this test
        bus = MessageBus(
            adapter=adapter,
            command_handlers={commands.Question: successful_handler},
            event_handlers={
                events.Response: [response_handler],
                events.FailedRequest: [failed_handler],
            },
            notifications=self.notifications,
        )

        # Act
        bus.handle(command)

        # Assert
        # Command handler should be called normally
        successful_handler.assert_called_once_with(command)

        # Response handler should be called for the response event
        response_handler.assert_called_once()

        # No FailedRequest event should be created
        failed_handler.assert_not_called()

    def test_sensitive_data_filtered_from_error_messages(self):
        """Test that sensitive data is filtered from error messages sent to users."""
        # Arrange
        command = commands.Question(question="Connect to database", q_id="secure123")

        def failing_handler(cmd):
            raise DatabaseConnectionException(
                "Failed to connect to database",
                context={
                    "connection_string": "postgresql://user:secret123@localhost/db",
                    "error_code": 1045,
                },
            )

        self.command_handlers[commands.Question] = failing_handler

        # Act
        self.bus.handle(command)

        # Assert
        failed_handler = self.event_handlers[events.FailedRequest][0]
        failed_event = failed_handler.call_args[0][0]

        # Error message should not contain password
        assert "secret123" not in failed_event.exception
        assert "Failed to connect to database" in failed_event.exception


class TestMessageBusEventHandlingWithErrors:
    """Test error handling for event processing."""

    def test_exception_in_event_handler_logs_but_continues(self):
        """Test that exceptions in event handlers are logged but don't stop processing."""
        # This tests the existing behavior in handle_event which already
        # catches exceptions and continues
        adapter = Mock()
        adapter.collect_new_events = Mock(return_value=[])

        failing_handler = Mock(side_effect=RuntimeError("Handler failed"))
        successful_handler = Mock()

        event_handlers = {events.Response: [failing_handler, successful_handler]}

        bus = MessageBus(
            adapter=adapter,
            command_handlers={},
            event_handlers=event_handlers,
        )

        # Act
        event = events.Response(
            question="Test",
            response="Answer",  # Changed from 'answer' to 'response'
            q_id="test123",
        )
        bus.handle(event)

        # Assert
        # Both handlers should be called
        failing_handler.assert_called_once()
        successful_handler.assert_called_once()
