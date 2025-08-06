import json
from typing import Dict, List, Optional, Union

import yaml

from src.agent.domain import commands, events
from src.agent.utils import populate_template
from src.agent.utils.command_registry import CommandHandlerRegistry
from src.agent.utils.constants import PromptKeys, ErrorMessages
from src.agent.utils.command_handlers import (
    QuestionHandler,
    CheckHandler,
    RetrieveHandler,
    RerankHandler,
    EnhanceHandler,
    UseToolsHandler,
    LLMResponseHandler,
    FinalCheckHandler,
)
from src.agent.utils.config_manager import ConfigurationManager


class BaseAgent:
    """
    BaseAgent is the model logic for the agent. It's uses a state machine to process and propagate different commands.
    The update method is the main method that decides the next command based on the current state and the incoming command.

    The events list is used to store outgoing events and will be picked up for notifications.
    Is_answered is used to check if the agent has answered the question and stops the state machine.
    Previous_command is used to check if the command is a duplicate and stops the state machine.

    Following commands are supported:
    - Question: The initial command to start the agent.
    - Check: Check the incoming question via guardrails.
    - Retrieve: Retrieve the most relevant documents from the knowledge base.
    - Rerank: Rerank the documents from the knowledge base.
    - Enhance: Enhance the question via LLM based on the reranked document.
    - UseTools: Use the agent tools to process the question.
    - LLMResponse: Use the LLM to process the question.
    - FinalCheck: Check the final answer via guardrails.

    Methods:
    - init_prompts: Initialize the prompts for the agent.
    - change_llm_response: Change the LLM response.
    - final_check: Check the final answer.
    - check_question: Check the question.
    - change_check: Change the check.
    - change_retrieve: Change the retrieve.
    - change_rerank: Change the rerank.
    - change_question: Change the question.
    - change_use_tools: Change the use tools.
    - create_prompt: Create the prompt for the command.
    - update: Update the state of the agent.
    """

    def __init__(self, question: commands.Question, kwargs: Dict = None):
        if not question or not question.question:
            raise ValueError(ErrorMessages.QUESTION_REQUIRED_ENHANCE)

        self.kwargs = kwargs or {}
        self.events = []
        self.is_answered = False

        self.agent_memory = None
        self.enhancement = None
        self.evaluation = None
        self.q_id = question.q_id
        self.question = question.question
        self.previous_command = None
        self.response = None
        self.send_response = None
        self.tool_answer = None

        # Initialize ConfigurationManager for centralized configuration
        self.config_manager = ConfigurationManager()

        self.base_prompts = self.init_prompts()

        # Initialize command registry with handlers
        self.command_registry = CommandHandlerRegistry()
        self._register_command_handlers()

    def _register_command_handlers(self) -> None:
        """
        Register all command handlers with the registry.

        This method sets up the Strategy pattern for command processing,
        replacing the match/case logic with a more maintainable registry.
        """
        self.command_registry.register(commands.Question, QuestionHandler())
        self.command_registry.register(commands.Check, CheckHandler())
        self.command_registry.register(commands.Retrieve, RetrieveHandler())
        self.command_registry.register(commands.Rerank, RerankHandler())
        self.command_registry.register(commands.Enhance, EnhanceHandler())
        self.command_registry.register(commands.UseTools, UseToolsHandler())
        self.command_registry.register(commands.LLMResponse, LLMResponseHandler())
        self.command_registry.register(commands.FinalCheck, FinalCheckHandler())

    def _get_prompt_template(self, command: commands.Command) -> str:
        """
        Get the appropriate prompt template for the given command type.

        Args:
            command: The command to get the template for

        Returns:
            The prompt template string

        Raises:
            ValueError: If command type is invalid or template is not found
        """
        if type(command) is commands.UseTools:
            template = self.base_prompts.get(PromptKeys.FINALIZE, None)
        elif type(command) is commands.Rerank:
            template = self.base_prompts.get(PromptKeys.ENHANCE, None)
        elif type(command) is commands.Question:
            template = self.base_prompts.get(PromptKeys.GUARDRAILS, {}).get(
                PromptKeys.PRE_CHECK, None
            )
        elif type(command) is commands.LLMResponse:
            template = self.base_prompts.get(PromptKeys.GUARDRAILS, {}).get(
                PromptKeys.POST_CHECK, None
            )
        else:
            raise ValueError(ErrorMessages.INVALID_COMMAND_TYPE)

        if template is None:
            raise ValueError(ErrorMessages.PROMPT_NOT_FOUND)

        return template

    def _get_prompt_variables(
        self, command: commands.Command, memory: List[str] = None
    ) -> dict:
        """
        Get the variables dictionary for template population based on command type.

        Args:
            command: The command to extract variables from
            memory: Optional memory context for LLMResponse commands

        Returns:
            Dictionary of variables for template population

        Raises:
            ValueError: If command type is invalid
        """
        if type(command) is commands.UseTools:
            return {
                "question": command.question,
                "response": command.response,
            }
        elif type(command) is commands.Rerank:
            candidates = [i.model_dump() for i in command.candidates]
            candidates_json = json.dumps(candidates)
            return {
                "question": command.question,
                "information": candidates_json,
            }
        elif type(command) is commands.Question:
            return {
                "question": command.question,
            }
        elif type(command) is commands.LLMResponse:
            memory_str = "\n".join(memory) if memory else ""
            return {
                "question": command.question,
                "response": command.response,
                "memory": memory_str,
            }
        else:
            raise ValueError("Invalid command type")

    def create_prompt(
        self,
        command: commands.Command,
        memory: List[str] = None,
    ) -> str:
        """
        Gets and preprocesses the prompt by the incoming command.

        This method has been refactored to use smaller, focused helper methods
        for better maintainability and testability.

        Args:
            command: The command to create the prompt for
            memory: Optional memory context for certain commands

        Returns:
            The prepared prompt for the command

        Raises:
            ValueError: If command type is invalid or template is not found
        """
        template = self._get_prompt_template(command)
        variables = self._get_prompt_variables(command, memory)

        return populate_template(template, variables)

    def init_prompts(self) -> Dict:
        """
        Initialize the prompts for the agent.

        Uses ConfigurationManager to get prompt path if not provided in kwargs,
        maintaining backward compatibility.

        Returns:
            base_prompts: Dict: The base prompts for the agent.

        Raises:
            ValueError: If prompt path is not found or file cannot be loaded
        """
        # Get prompt path from kwargs (backward compatibility) or ConfigurationManager
        if "prompt_path" in self.kwargs:
            prompt_path = self.kwargs["prompt_path"]
        else:
            try:
                agent_config = self.config_manager.get_agent_config()
                prompt_path = agent_config["prompt_path"]
            except Exception as e:
                raise ValueError(ErrorMessages.CONFIG_PATH_ERROR.format(error=e))

        try:
            with open(prompt_path, "r") as file:
                base_prompts = yaml.safe_load(file)
        except FileNotFoundError:
            raise ValueError(
                ErrorMessages.PROMPT_PATH_NOT_FOUND.format(path=prompt_path)
            )
        except Exception as e:
            raise ValueError(ErrorMessages.ERROR_LOADING_PROMPTS.format(error=e))

        return base_prompts

    def prepare_agent_call(self, command: commands.Enhance) -> commands.UseTools:
        """
        Prepares the tool agent call after the question enhancement.

        Args:
            command: commands.Enhance: The command to change the question.

        Returns:
            new_command: commands.UseTools: The new command.
        """
        if command.response is None:
            self.enhancement = self.question
        else:
            self.enhancement = command.response

        new_command = commands.UseTools(
            question=self.enhancement,
            q_id=command.q_id,
        )

        return new_command

    def prepare_enhancement(self, command: commands.Rerank) -> commands.Enhance:
        """
        Prepares the question enhancement after the reranking.

        Args:
            command: commands.Rerank: The command to change the rerank.

        Returns:
            new_command: commands.Enhance: The new command.
        """
        prompt = self.create_prompt(command)

        new_command = commands.Enhance(
            question=prompt,
            q_id=command.q_id,
        )

        return new_command

    def prepare_evaluation(self, command: commands.FinalCheck) -> None:
        """
        Prepares the evaluation event to be sent.

        Args:
            command: commands.FinalCheck: The command to final check the answer.

        Returns:
            None
        """
        self.is_answered = True

        self.evaluation = events.Evaluation(
            response=self.response.response,
            question=self.question,
            q_id=self.q_id,
            approved=command.approved,
            summary=command.summary,
            issues=command.issues,
            plausibility=command.plausibility,
            factual_consistency=command.factual_consistency,
            clarity=command.clarity,
            completeness=command.completeness,
        )

        return None

    def prepare_finalization(self, command: commands.UseTools) -> commands.LLMResponse:
        """
        Prepares the answer generation after the tool call.

        Args:
            command: commands.UseTools: The command to change the use tools.

        Returns:
            new_command: commands.LLMResponse: The new command.
        """
        self.tool_answer = command
        self.agent_memory = command.memory
        prompt = self.create_prompt(command)
        new_command = commands.LLMResponse(
            question=prompt,
            q_id=command.q_id,
            data=command.data,
        )

        return new_command

    def prepare_guardrails_check(self, command: commands.Question) -> commands.Check:
        """
        Prepares the guardrails check for the question.

        Args:
            command: commands.Question: The command to change the question.

        Returns:
            new_command: commands.Check: The new command.
        """
        prompt = self.create_prompt(command)

        new_command = commands.Check(
            question=prompt,
            q_id=command.q_id,
        )

        return new_command

    def prepare_response(self, command: commands.LLMResponse) -> commands.FinalCheck:
        """
        Prepares the final guardrailscheck and agent response after the final LLM call.

        Args:
            command: commands.LLMResponse: The command to change the LLM response.

        Returns:
            new_command: commands.FinalCheck: The new command.
        """
        if self.tool_answer is None:
            raise ValueError(ErrorMessages.TOOL_ANSWER_REQUIRED)

        response = events.Response(
            question=self.question,
            response=command.response,
            q_id=self.q_id,
            data=command.data,
        )

        # duplication as a fix - i want to keep self.response for testing adn validation
        self.send_response = response
        self.response = response

        prompt = self.create_prompt(command, self.agent_memory)

        new_command = commands.FinalCheck(
            question=prompt,
            q_id=command.q_id,
        )
        return new_command

    def prepare_retrieval(
        self, command: commands.Check
    ) -> Union[commands.Retrieve, events.FailedRequest]:
        """
        Prepares the retrieval after the check. If the check is not approved, the agent will stop and return a FailedRequest event.

        Args:
            command: commands.Check: The command to change the check.

        Returns:
            new_command: Union[commands.Retrieve, events.FailedRequest]: The new command.
        """
        if command.approved:
            new_command = commands.Retrieve(
                question=self.question,
                q_id=command.q_id,
            )
        else:
            self.is_answered = True
            new_command = events.RejectedRequest(
                question=self.question,
                response=command.response,
                q_id=command.q_id,
            )
            self.send_response = new_command
            self.response = new_command

        return new_command

    def prepare_rerank(self, command: commands.Retrieve) -> commands.Rerank:
        """
        Prepares the retrieved data for the reranking.

        Args:
            command: commands.Retrieve: The command to change the retrieve.

        Returns:
            new_command: commands.Rerank: The new command.
        """
        # if not command.question:
        #     raise ValueError("Question is required to enhance")
        # retrieve = self.cls_rag.retrieve(question)
        new_command = commands.Rerank(
            question=command.question,
            q_id=command.q_id,
            candidates=command.candidates,
        )

        return new_command

    def _update_state(self, response: commands.Command) -> None:
        """
        Update the internal state of the agent and check for repetition.

        Args:
            response: commands.Command: The command to update the state.

        Returns:
            None
        """
        if self.previous_command is type(response):
            self.is_answered = True
            self.response = events.FailedRequest(
                question=self.question,
                exception=ErrorMessages.DUPLICATE_COMMAND,
                q_id=self.q_id,
            )

        else:
            self.previous_command = type(response)

        return None

    def update(self, command: commands.Command) -> Optional[commands.Command]:
        """
        Update the state of the agent.

        Args:
            command: commands.Command: The command to update the state.

        Returns:
            Optional[commands.Command]: The next command.
        """
        self._update_state(command)

        if self.is_answered:
            return None

        # Process command using the registry (Strategy pattern)
        new_command = self.command_registry.process(command, self)
        return new_command
