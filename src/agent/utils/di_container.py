"""Dependency Injection container implementation."""

import inspect
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, TypeVar


T = TypeVar("T")


class Lifetime(Enum):
    """Service lifetime enumeration."""

    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceRegistration:
    """Represents a service registration."""

    def __init__(
        self,
        service_type: Type,
        implementation: Optional[Type] = None,
        factory: Optional[Callable] = None,
        instance: Optional[Any] = None,
        lifetime: Lifetime = Lifetime.TRANSIENT,
    ):
        self.service_type = service_type
        self.implementation = implementation
        self.factory = factory
        self.instance = instance
        self.lifetime = lifetime
        self.singleton_instance: Optional[Any] = None


class DIContainer:
    """Dependency Injection container."""

    def __init__(self, parent: Optional["DIContainer"] = None):
        self._services: Dict[Type, ServiceRegistration] = {}
        self._parent = parent
        self._scoped_instances: Dict[Type, Any] = {}
        self._disposables: list = []

    def register(
        self,
        service_type: Type[T],
        implementation: Type[T],
        lifetime: Lifetime = Lifetime.TRANSIENT,
    ) -> "DIContainer":
        """Register a service type with its implementation."""
        registration = ServiceRegistration(
            service_type=service_type, implementation=implementation, lifetime=lifetime
        )
        self._services[service_type] = registration
        return self

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[[], T],
        lifetime: Lifetime = Lifetime.TRANSIENT,
    ) -> "DIContainer":
        """Register a service type with a factory function."""
        registration = ServiceRegistration(
            service_type=service_type, factory=factory, lifetime=lifetime
        )
        self._services[service_type] = registration
        return self

    def register_instance(self, service_type: Type[T], instance: T) -> "DIContainer":
        """Register a service type with a specific instance (singleton)."""
        registration = ServiceRegistration(
            service_type=service_type, instance=instance, lifetime=Lifetime.SINGLETON
        )
        self._services[service_type] = registration
        return self

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service instance."""
        registration = self._get_registration(service_type)

        if registration is None:
            raise ValueError(f"Type {service_type} is not registered")

        return self._create_instance(registration)

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return self._get_registration(service_type) is not None

    def create_scope(self) -> "DIContainer":
        """Create a scoped container."""
        return DIContainer(parent=self)

    def __enter__(self) -> "DIContainer":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and dispose resources."""
        self.dispose()

    def dispose(self):
        """Dispose all managed resources."""
        # Dispose scoped instances first
        for instance in self._scoped_instances.values():
            self._dispose_instance(instance)

        # Dispose tracked disposables
        for disposable in self._disposables:
            self._dispose_instance(disposable)

        self._scoped_instances.clear()
        self._disposables.clear()

    def _dispose_instance(self, instance):
        """Dispose an individual instance if it supports disposal."""
        if hasattr(instance, "dispose") and callable(getattr(instance, "dispose")):
            try:
                instance.dispose()
            except Exception:
                pass  # Silently ignore disposal errors

        if hasattr(instance, "__exit__") and callable(getattr(instance, "__exit__")):
            try:
                instance.__exit__(None, None, None)
            except Exception:
                pass  # Silently ignore disposal errors

    def _get_registration(self, service_type: Type) -> Optional[ServiceRegistration]:
        """Get registration for a service type, checking parent containers."""
        if service_type in self._services:
            return self._services[service_type]

        if self._parent is not None:
            return self._parent._get_registration(service_type)

        return None

    def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create an instance based on registration configuration."""
        # Handle pre-existing instance
        if registration.instance is not None:
            return registration.instance

        # Handle singleton lifetime
        if registration.lifetime == Lifetime.SINGLETON:
            if registration.singleton_instance is not None:
                instance = registration.singleton_instance
                # Track for disposal if it's disposable and we're in a scoped container
                if (
                    hasattr(instance, "dispose") or hasattr(instance, "__exit__")
                ) and instance not in self._disposables:
                    self._disposables.append(instance)
                return instance

            instance = self._instantiate(registration)
            registration.singleton_instance = instance

            # Track disposable instances
            if hasattr(instance, "dispose") or hasattr(instance, "__exit__"):
                self._disposables.append(instance)

            return instance

        # Handle scoped lifetime
        if registration.lifetime == Lifetime.SCOPED:
            if registration.service_type in self._scoped_instances:
                return self._scoped_instances[registration.service_type]

            instance = self._instantiate(registration)
            self._scoped_instances[registration.service_type] = instance

            # Track disposable instances
            if hasattr(instance, "dispose") or hasattr(instance, "__exit__"):
                self._disposables.append(instance)

            return instance

        # Handle transient lifetime (default)
        instance = self._instantiate(registration)

        # Track disposable transient instances if we have a parent (scoped container)
        if self._parent and (
            hasattr(instance, "dispose") or hasattr(instance, "__exit__")
        ):
            self._disposables.append(instance)

        return instance

    def _instantiate(self, registration: ServiceRegistration) -> Any:
        """Create a new instance using factory or constructor."""
        if registration.factory is not None:
            return registration.factory()

        if registration.implementation is not None:
            return self._create_with_dependencies(registration.implementation)

        raise ValueError(
            f"No implementation or factory registered for {registration.service_type}"
        )

    def _create_with_dependencies(self, implementation_type: Type) -> Any:
        """Create an instance and resolve its constructor dependencies."""
        try:
            # Get constructor signature
            signature = inspect.signature(implementation_type.__init__)
            parameters = list(signature.parameters.values())[1:]  # Skip 'self'

            # Resolve constructor dependencies
            args = {}
            for param in parameters:
                if param.annotation != param.empty:
                    dependency = self.resolve(param.annotation)
                    args[param.name] = dependency

            return implementation_type(**args)

        except Exception as e:
            # Fallback to parameterless constructor
            try:
                return implementation_type()
            except Exception:
                raise ValueError(
                    f"Failed to create instance of {implementation_type}: {str(e)}"
                )
