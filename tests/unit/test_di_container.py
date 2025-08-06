"""Test cases for Dependency Injection container."""

import pytest
from abc import ABC, abstractmethod

from src.agent.utils.di_container import DIContainer, Lifetime


class AbstractService(ABC):
    @abstractmethod
    def execute(self) -> str:
        pass


class ConcreteService(AbstractService):
    def execute(self) -> str:
        return "ConcreteService executed"


class DependentService:
    def __init__(self, service: AbstractService):
        self.service = service

    def process(self) -> str:
        return f"DependentService using: {self.service.execute()}"


class DisposableService:
    """Service that tracks disposal for lifecycle management tests."""

    def __init__(self):
        self.disposed = False

    def dispose(self):
        self.disposed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()


class TestDIContainerBasics:
    """Test basic DI container functionality."""

    def test_should_register_and_resolve_singleton_type(self):
        """Test that singleton types can be registered and resolved."""
        container = DIContainer()
        container.register(AbstractService, ConcreteService, Lifetime.SINGLETON)

        service1 = container.resolve(AbstractService)
        service2 = container.resolve(AbstractService)

        assert isinstance(service1, ConcreteService)
        assert service1 is service2  # Same instance for singleton
        assert service1.execute() == "ConcreteService executed"

    def test_should_register_and_resolve_transient_type(self):
        """Test that transient types create new instances each time."""
        container = DIContainer()
        container.register(AbstractService, ConcreteService, Lifetime.TRANSIENT)

        service1 = container.resolve(AbstractService)
        service2 = container.resolve(AbstractService)

        assert isinstance(service1, ConcreteService)
        assert isinstance(service2, ConcreteService)
        assert service1 is not service2  # Different instances for transient
        assert service1.execute() == "ConcreteService executed"
        assert service2.execute() == "ConcreteService executed"

    def test_should_register_and_resolve_instance(self):
        """Test that instances can be registered and resolved."""
        container = DIContainer()
        instance = ConcreteService()
        container.register_instance(AbstractService, instance)

        service = container.resolve(AbstractService)

        assert service is instance
        assert service.execute() == "ConcreteService executed"

    def test_should_register_and_resolve_factory(self):
        """Test that factories can be registered and resolved."""
        container = DIContainer()

        def factory():
            return ConcreteService()

        container.register_factory(AbstractService, factory, Lifetime.SINGLETON)

        service1 = container.resolve(AbstractService)
        service2 = container.resolve(AbstractService)

        assert isinstance(service1, ConcreteService)
        assert service1 is service2  # Singleton lifetime
        assert service1.execute() == "ConcreteService executed"

    def test_should_auto_resolve_dependencies(self):
        """Test that constructor dependencies are automatically resolved."""
        container = DIContainer()
        container.register(AbstractService, ConcreteService, Lifetime.SINGLETON)
        container.register(DependentService, DependentService, Lifetime.TRANSIENT)

        dependent = container.resolve(DependentService)

        assert isinstance(dependent, DependentService)
        assert isinstance(dependent.service, ConcreteService)
        assert dependent.process() == "DependentService using: ConcreteService executed"

    def test_should_raise_error_for_unregistered_type(self):
        """Test that resolving unregistered type raises error."""
        container = DIContainer()

        with pytest.raises(ValueError, match="Type .* is not registered"):
            container.resolve(AbstractService)

    def test_should_check_if_type_is_registered(self):
        """Test that container can check if type is registered."""
        container = DIContainer()

        assert not container.is_registered(AbstractService)

        container.register(AbstractService, ConcreteService, Lifetime.SINGLETON)

        assert container.is_registered(AbstractService)


class TestDIContainerScoped:
    """Test scoped container functionality."""

    def test_should_create_scoped_container(self):
        """Test that scoped containers can be created."""
        parent = DIContainer()
        parent.register(AbstractService, ConcreteService, Lifetime.SINGLETON)

        scoped = parent.create_scope()

        assert scoped is not parent
        assert scoped.is_registered(
            AbstractService
        )  # Should inherit parent registrations

    def test_should_resolve_scoped_services_as_singleton_within_scope(self):
        """Test that scoped services behave as singletons within their scope."""
        container = DIContainer()
        container.register(AbstractService, ConcreteService, Lifetime.SCOPED)

        scope1 = container.create_scope()
        service1a = scope1.resolve(AbstractService)
        service1b = scope1.resolve(AbstractService)

        scope2 = container.create_scope()
        service2a = scope2.resolve(AbstractService)

        assert service1a is service1b  # Same instance within scope
        assert service1a is not service2a  # Different instances across scopes

    def test_should_inherit_parent_registrations(self):
        """Test that child containers inherit parent registrations."""
        parent = DIContainer()
        parent.register(AbstractService, ConcreteService, Lifetime.SINGLETON)

        child = parent.create_scope()

        # Parent singleton should be shared with child
        parent_service = parent.resolve(AbstractService)
        child_service = child.resolve(AbstractService)

        assert parent_service is child_service

    def test_should_override_parent_registrations(self):
        """Test that child containers can override parent registrations."""
        parent = DIContainer()
        parent.register(AbstractService, ConcreteService, Lifetime.SINGLETON)

        child = parent.create_scope()

        class ChildService(AbstractService):
            def execute(self) -> str:
                return "ChildService executed"

        child.register(AbstractService, ChildService, Lifetime.TRANSIENT)

        parent_service = parent.resolve(AbstractService)
        child_service = child.resolve(AbstractService)

        assert isinstance(parent_service, ConcreteService)
        assert isinstance(child_service, ChildService)
        assert parent_service.execute() == "ConcreteService executed"
        assert child_service.execute() == "ChildService executed"


class TestDIContainerLifecycle:
    """Test lifecycle management functionality."""

    def test_should_dispose_resources_with_context_manager(self):
        """Test that disposable resources are properly cleaned up."""
        container = DIContainer()
        container.register(DisposableService, DisposableService, Lifetime.SINGLETON)

        with container.create_scope() as scope:
            service = scope.resolve(DisposableService)
            assert not service.disposed

        # Resource should be disposed after context exits
        assert service.disposed
