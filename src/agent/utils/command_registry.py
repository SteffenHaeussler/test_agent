"""
Command Handler Registry implementing the Strategy pattern.

This module provides a registry system for command handlers,
replacing switch statements with a more maintainable and extensible design.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

from src.agent.domain import commands


class CommandHandler(ABC):
    """Abstract base class for command handlers."""

    @abstractmethod
    def can_handle(self, command: commands.Command) -> bool:
        """Check if this handler can process the given command."""
        pass

    @abstractmethod
    def handle(self, command: commands.Command, agent) -> Optional[commands.Command]:
        """Process the command and return the next command in the chain."""
        pass


class CommandHandlerRegistry:
    """
    Registry for command handlers implementing the Strategy pattern.

    This replaces switch statements with a more maintainable approach,
    allowing for easy extension and modification of command processing logic.
    """

    def __init__(self):
        self._handlers: Dict[Type[commands.Command], CommandHandler] = {}
        self._fallback_handler: Optional[CommandHandler] = None

    def register(
        self, command_type: Type[commands.Command], handler: CommandHandler
    ) -> None:
        """
        Register a handler for a specific command type.

        Args:
            command_type: The type of command to handle
            handler: The handler instance for this command type
        """
        self._handlers[command_type] = handler

    def register_fallback(self, handler: CommandHandler) -> None:
        """
        Register a fallback handler for unregistered command types.

        Args:
            handler: The fallback handler instance
        """
        self._fallback_handler = handler

    def get_handler(self, command: commands.Command) -> Optional[CommandHandler]:
        """
        Get the appropriate handler for a command.

        Args:
            command: The command to get a handler for

        Returns:
            The handler for this command type, or the fallback handler if no specific handler exists
        """
        command_type = type(command)
        handler = self._handlers.get(command_type)

        if handler and handler.can_handle(command):
            return handler

        # Try fallback handler
        if self._fallback_handler and self._fallback_handler.can_handle(command):
            return self._fallback_handler

        return None

    def process(self, command: commands.Command, agent) -> Optional[commands.Command]:
        """
        Process a command using the registered handler.

        Args:
            command: The command to process
            agent: The agent instance processing the command

        Returns:
            The next command in the chain, or None if processing is complete

        Raises:
            NotImplementedError: If no handler is registered for the command type
        """
        handler = self.get_handler(command)

        if handler is None:
            raise NotImplementedError(
                f"No handler registered for command type: {type(command).__name__}"
            )

        return handler.handle(command, agent)

    def unregister(self, command_type: Type[commands.Command]) -> None:
        """
        Remove a handler for a specific command type.

        Args:
            command_type: The command type to unregister
        """
        if command_type in self._handlers:
            del self._handlers[command_type]

    def clear(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._fallback_handler = None

    def get_registered_types(self) -> list[Type[commands.Command]]:
        """Get a list of all registered command types."""
        return list(self._handlers.keys())
