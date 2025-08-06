import inspect
from typing import Dict, Optional

from fastapi.websockets import WebSocket
from langfuse import get_client, observe

from src.agent import config
from src.agent.adapters import (
    adapter as adapters_module,
    agent_tools,
    database,
    llm,
    rag,
)
from src.agent.adapters.notifications import AbstractNotifications, CliNotifications
from src.agent.observability.context import ctx_query_id
from src.agent.service_layer import handlers, messagebus
from src.agent.utils.di_container import DIContainer, Lifetime

connected_clients: Dict[str, WebSocket] = {}

# Global DI container for the application
_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Get or create the global DI container with all registrations."""
    global _container
    if _container is None:
        _container = _configure_container()
    return _container


def _configure_container() -> DIContainer:
    """Configure the DI container with all service registrations."""
    container = DIContainer()

    # Register adapters as singletons
    container.register_factory(
        adapters_module.AbstractAdapter,
        lambda: adapters_module.RouterAdapter(),
        Lifetime.SINGLETON,
    )

    # Register database adapters
    container.register_factory(
        database.AbstractDatabase,
        lambda: database.BaseDatabaseAdapter(kwargs=config.get_database_config()),
        Lifetime.SINGLETON,
    )

    # Register LLM services
    container.register_factory(
        llm.AbstractLLM,
        lambda: llm.LLM(kwargs=config.get_llm_config()),
        Lifetime.SINGLETON,
    )

    # Register RAG services
    container.register_factory(
        rag.AbstractModel,
        lambda: rag.BaseRAG(config.get_rag_config()),
        Lifetime.SINGLETON,
    )

    # Register tools
    container.register_factory(
        agent_tools.AbstractTools,
        lambda: agent_tools.Tools(kwargs=config.get_tools_config()),
        Lifetime.SINGLETON,
    )

    # Register notifications as singleton by default
    container.register(AbstractNotifications, CliNotifications, Lifetime.SINGLETON)

    # Register handlers as transient (new instance per resolution)
    for command_type, handler in handlers.COMMAND_HANDLERS.items():
        container.register_factory(handler, lambda h=handler: h, Lifetime.TRANSIENT)

    for event_handlers_list in handlers.EVENT_HANDLERS.values():
        for handler in event_handlers_list:
            container.register_factory(handler, lambda h=handler: h, Lifetime.TRANSIENT)

    return container


@observe()
def bootstrap(
    adapter: adapters_module.AbstractAdapter = None,
    notifications: AbstractNotifications = None,
    container: DIContainer = None,
) -> messagebus.MessageBus:
    """
    Bootstraps the agent with dependency injection support.

    Args:
        adapter: adapters_module.AbstractAdapter: The adapter to use (optional).
        notifications: AbstractNotifications: The notifications to use (optional).
        container: DIContainer: Custom DI container (optional).

    Returns:
        messagebus.MessageBus: The message bus.
    """
    langfuse = get_client()

    langfuse.update_current_trace(
        name="bootstrap",
        session_id=ctx_query_id.get(),
    )

    # Use provided container or get global one
    if container is None:
        container = get_container()

    # Create a request-scoped container
    scoped_container = container.create_scope()

    # Override with provided dependencies if any
    if adapter is not None:
        scoped_container.register_instance(adapters_module.AbstractAdapter, adapter)

    if notifications is not None:
        scoped_container.register_instance(AbstractNotifications, notifications)

    # Resolve dependencies from container
    resolved_adapter = (
        scoped_container.resolve(adapters_module.AbstractAdapter)
        if adapter is None
        else adapter
    )
    resolved_notifications = (
        scoped_container.resolve(AbstractNotifications)
        if notifications is None
        else notifications
    )

    # Use existing dependency injection for handlers (maintaining backward compatibility)
    dependencies = {
        "adapter": resolved_adapter,
        "notifications": resolved_notifications,
    }

    injected_event_handlers = {
        event_type: [
            inject_dependencies(handler, dependencies) for handler in event_handlers
        ]
        for event_type, event_handlers in handlers.EVENT_HANDLERS.items()
    }
    injected_command_handlers = {
        command_type: inject_dependencies(handler, dependencies)
        for command_type, handler in handlers.COMMAND_HANDLERS.items()
    }

    return messagebus.MessageBus(
        adapter=resolved_adapter,
        event_handlers=injected_event_handlers,
        command_handlers=injected_command_handlers,
        notifications=resolved_notifications,
    )


def inject_dependencies(handler, dependencies):
    """
    Injects the dependencies into the handler.

    Args:
        handler: The handler to inject the dependencies into.
        dependencies: The dependencies to inject.
    """

    params = inspect.signature(handler).parameters
    deps = {
        name: dependency for name, dependency in dependencies.items() if name in params
    }
    return lambda message: handler(message, **deps)
