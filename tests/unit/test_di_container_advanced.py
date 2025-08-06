"""Advanced test cases for DI container integration with the application."""

from unittest.mock import Mock

from src.agent.adapters.adapter import AbstractAdapter
from src.agent.adapters.notifications import AbstractNotifications
from src.agent.bootstrap import bootstrap, get_container
from src.agent.utils.di_container import DIContainer, Lifetime


class TestDIContainerAdvanced:
    """Test advanced DI container scenarios."""

    def test_should_get_global_container_with_services_registered(self):
        """Test that global container has all expected services registered."""
        container = get_container()

        # Should have all main services registered
        assert container.is_registered(AbstractAdapter)
        assert container.is_registered(AbstractNotifications)

        # Should resolve concrete implementations
        adapter = container.resolve(AbstractAdapter)
        notifications = container.resolve(AbstractNotifications)

        assert adapter is not None
        assert notifications is not None

    def test_should_use_custom_container_in_bootstrap(self):
        """Test that bootstrap can use a custom container."""
        # Create custom container with mock services
        custom_container = DIContainer()
        mock_adapter = Mock(spec=AbstractAdapter)
        mock_notifications = Mock(spec=AbstractNotifications)

        custom_container.register_instance(AbstractAdapter, mock_adapter)
        custom_container.register_instance(AbstractNotifications, mock_notifications)

        # Bootstrap with custom container
        bus = bootstrap(container=custom_container)

        # Should use services from custom container
        assert bus.adapter is mock_adapter
        assert bus.notifications is mock_notifications

    def test_should_maintain_singleton_behavior_across_scopes(self):
        """Test that singletons are shared across different scopes."""
        container = get_container()

        # Create two different scopes
        scope1 = container.create_scope()
        scope2 = container.create_scope()

        # Resolve adapter from both scopes
        adapter1 = scope1.resolve(AbstractAdapter)
        adapter2 = scope2.resolve(AbstractAdapter)

        # Should be the same instance (singleton)
        assert adapter1 is adapter2

    def test_should_dispose_resources_properly(self):
        """Test that scoped resources are disposed when scope ends."""
        container = get_container()

        class MockDisposableService:
            def __init__(self):
                self.disposed = False

            def dispose(self):
                self.disposed = True

        # Register disposable service with scoped lifetime
        container.register(
            MockDisposableService, MockDisposableService, Lifetime.SCOPED
        )

        service = None
        with container.create_scope() as scope:
            service = scope.resolve(MockDisposableService)
            assert not service.disposed

        # Service should be disposed after scope exits
        assert service.disposed

    def test_should_handle_dependency_chains(self):
        """Test that complex dependency chains are resolved correctly."""
        container = DIContainer()

        class ServiceA:
            def get_name(self):
                return "ServiceA"

        class ServiceB:
            def __init__(self, service_a: ServiceA):
                self.service_a = service_a

            def get_name(self):
                return f"ServiceB({self.service_a.get_name()})"

        class ServiceC:
            def __init__(self, service_b: ServiceB):
                self.service_b = service_b

            def get_name(self):
                return f"ServiceC({self.service_b.get_name()})"

        # Register services
        container.register(ServiceA, ServiceA, Lifetime.SINGLETON)
        container.register(ServiceB, ServiceB, Lifetime.SINGLETON)
        container.register(ServiceC, ServiceC, Lifetime.TRANSIENT)

        # Resolve top-level service
        service_c = container.resolve(ServiceC)

        # Should have resolved entire dependency chain
        assert service_c.get_name() == "ServiceC(ServiceB(ServiceA))"
        assert isinstance(service_c.service_b.service_a, ServiceA)

    def test_should_override_services_in_child_container(self):
        """Test that child containers can override parent services."""
        parent = DIContainer()

        class OriginalService:
            def get_message(self):
                return "Original"

        class OverrideService:
            def get_message(self):
                return "Override"

        parent.register(OriginalService, OriginalService, Lifetime.SINGLETON)

        child = parent.create_scope()
        child.register(OriginalService, OverrideService, Lifetime.TRANSIENT)

        # Parent should use original service
        parent_service = parent.resolve(OriginalService)
        assert parent_service.get_message() == "Original"

        # Child should use override service
        child_service = child.resolve(OriginalService)
        assert child_service.get_message() == "Override"
