import asyncio
from typing import Callable, Dict, List, Type, Union

from loguru import logger

from src.agent.adapters import adapter
from src.agent.domain import commands, events
from src.agent.exceptions import AgentException

Message = Union[commands.Command, events.Event]


class MessageBus:
    """
    MessageBus is a class that handles command and events.

    Commands are send to the agent and events are sent to the notifications.

    Args:
        adapter: adapter.AbstractAdapter: The adapter to use.
        event_handlers: Dict[Type[events.Event], List[Callable]]: The event handlers.
        command_handlers: Dict[Type[commands.Command], Callable]: The command handlers.

    Returns:
        None
    """

    def __init__(
        self,
        adapter: adapter.AbstractAdapter,
        event_handlers: Dict[Type[events.Event], List[Callable]],
        command_handlers: Dict[Type[commands.Command], Callable],
        notifications=None,
    ) -> None:
        self.adapter = adapter
        self.event_handlers = event_handlers
        self.command_handlers = command_handlers
        self.notifications = notifications

    async def handle(
        self,
        message: Message,
    ) -> None:
        """
        Handles incoming messages or gets them from the internal queue.

        Args:
            message: Message: The message to handle.

        Returns:
            None
        """
        self.queue = [message]
        while self.queue:
            message = self.queue.pop(0)
            if isinstance(message, events.Event):
                await self.handle_event(message)
            elif isinstance(message, commands.Command):
                await self.handle_command(message)
            else:
                raise Exception(f"{message} was not an Event or Command")

    async def handle_command(
        self,
        command: commands.Command,
    ) -> None:
        """
        Handles incoming commands and collects new commands/events from the agent.events list.

        If an exception occurs, it automatically creates a FailedRequest event
        to notify the user via WebSocket/notifications.

        Args:
            command: commands.Command: The command to handle.
        """
        logger.debug("handling command %s", command)
        try:
            handler = self.command_handlers[type(command)]
            await handler(command)
            self.queue.extend(self.adapter.collect_new_events())
        except Exception as e:
            logger.exception("Exception handling command %s", command)

            # Create FailedRequest event to notify user
            # Extract question and q_id from command if available
            question = getattr(command, "question", None) or "Unknown"
            q_id = getattr(command, "q_id", None) or "unknown"

            # Create user-friendly error message
            if isinstance(e, AgentException):
                # For our custom exceptions, use the message directly
                # Context is logged but not sent to user for security
                error_message = str(e)
            else:
                # For unexpected exceptions, provide generic message
                error_message = f"An error occurred while processing your request: {type(e).__name__}"

            # Create and queue the FailedRequest event
            failed_event = events.FailedRequest(
                question=question, exception=error_message, q_id=q_id
            )

            # Add to queue so it gets processed
            self.queue.append(failed_event)

            # Don't re-raise - let the FailedRequest event notify the user
            # This ensures the message bus continues processing

    async def handle_event(
        self,
        event: events.Event,
    ) -> None:
        """
        Handles incoming events and collects new commands/events from the agent.events list.

        Args:
            event: events.Event: The event to handle.

        Returns:
            None
        """
        # Process event handlers concurrently for better performance
        handler_tasks = []
        for handler in self.event_handlers[type(event)]:
            handler_tasks.append(self._handle_single_event(handler, event))

        if handler_tasks:
            await asyncio.gather(*handler_tasks, return_exceptions=True)
            self.queue.extend(self.adapter.collect_new_events())

    async def _handle_single_event(
        self,
        handler: Callable,
        event: events.Event,
    ) -> None:
        """
        Handle a single event with a specific handler.

        Args:
            handler: Callable: The event handler function.
            event: events.Event: The event to handle.
        """
        try:
            logger.debug(f"handling event {str(event)} with handler {str(handler)}")
            await handler(event)
        except Exception:
            logger.exception(f"Exception handling event {event}")
