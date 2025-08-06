"""
Concrete command handler implementations.

This module provides concrete implementations of the CommandHandler interface
for each command type in the agent system, replacing the match/case logic
with the Strategy pattern.
"""

from typing import Optional

from src.agent.domain import commands
from src.agent.utils.command_registry import CommandHandler


class QuestionHandler(CommandHandler):
    """Handler for Question commands."""

    def can_handle(self, command: commands.Command) -> bool:
        """Check if this handler can process Question commands."""
        return isinstance(command, commands.Question)

    def handle(self, command: commands.Command, agent) -> Optional[commands.Command]:
        """Process a Question command and return a Check command."""
        if not isinstance(command, commands.Question):
            return None

        return agent.prepare_guardrails_check(command)


class CheckHandler(CommandHandler):
    """Handler for Check commands."""

    def can_handle(self, command: commands.Command) -> bool:
        """Check if this handler can process Check commands."""
        return isinstance(command, commands.Check)

    def handle(self, command: commands.Command, agent) -> Optional[commands.Command]:
        """Process a Check command and return a Retrieve command or RejectedRequest."""
        if not isinstance(command, commands.Check):
            return None

        return agent.prepare_retrieval(command)


class RetrieveHandler(CommandHandler):
    """Handler for Retrieve commands."""

    def can_handle(self, command: commands.Command) -> bool:
        """Check if this handler can process Retrieve commands."""
        return isinstance(command, commands.Retrieve)

    def handle(self, command: commands.Command, agent) -> Optional[commands.Command]:
        """Process a Retrieve command and return a Rerank command."""
        if not isinstance(command, commands.Retrieve):
            return None

        return agent.prepare_rerank(command)


class RerankHandler(CommandHandler):
    """Handler for Rerank commands."""

    def can_handle(self, command: commands.Command) -> bool:
        """Check if this handler can process Rerank commands."""
        return isinstance(command, commands.Rerank)

    def handle(self, command: commands.Command, agent) -> Optional[commands.Command]:
        """Process a Rerank command and return an Enhance command."""
        if not isinstance(command, commands.Rerank):
            return None

        return agent.prepare_enhancement(command)


class EnhanceHandler(CommandHandler):
    """Handler for Enhance commands."""

    def can_handle(self, command: commands.Command) -> bool:
        """Check if this handler can process Enhance commands."""
        return isinstance(command, commands.Enhance)

    def handle(self, command: commands.Command, agent) -> Optional[commands.Command]:
        """Process an Enhance command and return a UseTools command."""
        if not isinstance(command, commands.Enhance):
            return None

        return agent.prepare_agent_call(command)


class UseToolsHandler(CommandHandler):
    """Handler for UseTools commands."""

    def can_handle(self, command: commands.Command) -> bool:
        """Check if this handler can process UseTools commands."""
        return isinstance(command, commands.UseTools)

    def handle(self, command: commands.Command, agent) -> Optional[commands.Command]:
        """Process a UseTools command and return an LLMResponse command."""
        if not isinstance(command, commands.UseTools):
            return None

        return agent.prepare_finalization(command)


class LLMResponseHandler(CommandHandler):
    """Handler for LLMResponse commands."""

    def can_handle(self, command: commands.Command) -> bool:
        """Check if this handler can process LLMResponse commands."""
        return isinstance(command, commands.LLMResponse)

    def handle(self, command: commands.Command, agent) -> Optional[commands.Command]:
        """Process an LLMResponse command and return a FinalCheck command."""
        if not isinstance(command, commands.LLMResponse):
            return None

        return agent.prepare_response(command)


class FinalCheckHandler(CommandHandler):
    """Handler for FinalCheck commands."""

    def can_handle(self, command: commands.Command) -> bool:
        """Check if this handler can process FinalCheck commands."""
        return isinstance(command, commands.FinalCheck)

    def handle(self, command: commands.Command, agent) -> Optional[commands.Command]:
        """Process a FinalCheck command and return None (end of chain)."""
        if not isinstance(command, commands.FinalCheck):
            return None

        agent.prepare_evaluation(command)
        return None
