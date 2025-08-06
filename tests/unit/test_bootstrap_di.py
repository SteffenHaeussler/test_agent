"""Test cases for bootstrap.py with DI container integration."""

from unittest.mock import Mock

from src.agent.adapters.adapter import AbstractAdapter
from src.agent.adapters.notifications import AbstractNotifications, CliNotifications
from src.agent.bootstrap import bootstrap
from src.agent.service_layer.messagebus import MessageBus


class TestBootstrapDIIntegration:
    """Test bootstrap integration with DI container."""

    def test_should_bootstrap_with_default_dependencies(self):
        """Test that bootstrap creates MessageBus with default dependencies."""
        bus = bootstrap()

        assert isinstance(bus, MessageBus)
        assert bus.adapter is not None
        assert bus.notifications is not None
        assert isinstance(bus.notifications, CliNotifications)

    def test_should_bootstrap_with_custom_adapter(self):
        """Test that bootstrap accepts custom adapter."""
        custom_adapter = Mock(spec=AbstractAdapter)

        bus = bootstrap(adapter=custom_adapter)

        assert isinstance(bus, MessageBus)
        assert bus.adapter is custom_adapter

    def test_should_bootstrap_with_custom_notifications(self):
        """Test that bootstrap accepts custom notifications."""
        custom_notifications = Mock(spec=AbstractNotifications)

        bus = bootstrap(notifications=custom_notifications)

        assert isinstance(bus, MessageBus)
        assert bus.notifications is custom_notifications

    def test_should_maintain_backward_compatibility(self):
        """Test that existing bootstrap interface works as expected."""
        # This ensures we don't break existing code
        custom_adapter = Mock(spec=AbstractAdapter)
        custom_notifications = Mock(spec=AbstractNotifications)

        bus = bootstrap(adapter=custom_adapter, notifications=custom_notifications)

        assert isinstance(bus, MessageBus)
        assert bus.adapter is custom_adapter
        assert bus.notifications is custom_notifications

    def test_should_handle_multiple_notifications(self):
        """Test that bootstrap correctly handles notification collections."""
        # For now, the current bootstrap only accepts single notifications
        # This test is for when we enhance it with DI container
        notification = Mock(spec=AbstractNotifications)

        bus = bootstrap(notifications=notification)

        assert isinstance(bus, MessageBus)
        assert bus.notifications is notification
