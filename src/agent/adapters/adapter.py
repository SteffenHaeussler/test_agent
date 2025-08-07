import asyncio
from abc import ABC

from langfuse import get_client, observe
from loguru import logger
from sqlalchemy import MetaData


from src.agent import config
from src.agent.adapters import agent_tools, database, llm, rag
from src.agent.domain import commands, model
from src.agent.utils.constants import TraceNames


class AbstractAdapter(ABC):
    """
    AbstractAdapter is an abstract base class for all adapters.
    It defines the interface for external services.

    It defines the flow of commands from the model agent to the external service.

    Question -> Check -> UseTools -> Retrieve -> Rerank -> Enhance -> LLMResponse -> FinalCheck

    Methods:
        - add(agent: model.BaseAgent): Add an agent to the adapter.
        - collect_new_events(): Collect new events from the model agent.
        - answer(command: commands.Command) -> str: General entrypoint for a command.
    """

    def __init__(self):
        self.agent = None
        self.database = database.AbstractDatabase()
        self.llm = llm.AbstractLLM()
        self.tools = agent_tools.AbstractTools()
        self.rag = rag.AbstractModel()
        self.guardrails = llm.AbstractLLM()

    def add(self, agent: model.BaseAgent):
        """
        Add an agent to the adapter.

        Args:
            agent: model.BaseAgent: The agent to add.

        Returns:
            None
        """
        self.agent = agent

    def answer(self, command: commands.Command) -> str:
        """
        Answer a command.

        Args:
            command: commands.Command: The command to answer.

        Returns:
            str: The answer to the command.
        """
        raise NotImplementedError("Not implemented yet")

    def collect_new_events(self):
        """
        Collect new events from the model agent.

        Returns:
            An iterator of events.
        """
        while self.agent.events:
            event = self.agent.events.pop(0)
            yield event


class RouterAdapter(AbstractAdapter):
    """Router adapter that selects the appropriate adapter based on command type."""

    def __init__(self):
        self.agent_adapter = AgentAdapter()
        self.sql_adapter = SQLAgentAdapter()
        self.scenario_adapter = ScenarioAdapter()

    def answer(self, command):
        """Route to appropriate adapter based on command type."""
        return self.agent_adapter.answer(command)

    async def answer_async(self, command):
        """Route to appropriate adapter based on command type (async)."""
        return await self.agent_adapter.answer_async(command)

    def query(self, command):
        """Route to appropriate adapter based on command type."""
        return self.sql_adapter.query(command)

    async def query_async(self, command):
        """Route to appropriate adapter based on command type (async)."""
        return await self.sql_adapter.query_async(command)

    def scenario(self, command):
        """Route to appropriate adapter based on command type."""
        return self.scenario_adapter.query(command)

    async def scenario_async(self, command):
        """Route to appropriate adapter based on command type (async)."""
        return await self.scenario_adapter.query_async(command)

    def add(self, agent):
        """Add agent to both adapters."""
        self.agent_adapter.add(agent)
        self.sql_adapter.add(agent)
        self.scenario_adapter.add(agent)

    def collect_new_events(self):
        """Collect events from both adapters."""
        events = []
        events.extend(self.agent_adapter.collect_new_events())
        events.extend(self.sql_adapter.collect_new_events())
        events.extend(self.scenario_adapter.collect_new_events())
        return events


class AgentAdapter(AbstractAdapter):
    """
    AgentAdapter is an adapter for the model agent.
    It defines the flow of commands from the model agent to the external service.

    Question -> Check -> UseTools -> Retrieve -> Rerank -> Enhance -> LLMResponse -> FinalCheck

    Methods:
        - answer(command: commands.Command) -> commands.Command: General entrypoint for a command.
        - check(command: commands.Check) -> commands.Check: Check the incoming question via guardrails.
        - evaluation(command: commands.FinalCheck) -> commands.FinalCheck: Evaluate the response via guardrails.
        - finalize(command: commands.LLMResponse) -> commands.LLMResponse: Finalize the response via LLM.
        - question(command: commands.Question) -> commands.Question: only for tracing.
        - use(command: commands.UseTools) -> commands.UseTools: Use the agent tools to process the question.
        - retrieve(command: commands.Retrieve) -> commands.Retrieve: Retrieve the most relevant documents from the knowledge base.
        - rerank(command: commands.Rerank) -> commands.Rerank: Rerank the documents from the knowledge base.
        - enhance(command: commands.Enhance) -> commands.Enhance: Enhance the question via LLM based on the reranked document.

    Adapters:
        - database: Database adapter.
        - guardrails: Performs checks via guardrails.
        - llm: Calls a LLM.
        - rag: RAG model to enhance questions and retrieve documents.
        - tools: Use the agent tools to process the question.
    """

    def __init__(self):
        super().__init__()

        self.database = database.BaseDatabaseAdapter(
            kwargs=config.get_database_config(),
        )

        self.guardrails = llm.LLM(
            kwargs=config.get_guardrails_config(),
        )
        self.llm = llm.LLM(
            kwargs=config.get_llm_config(),
        )
        self.rag = rag.BaseRAG(config.get_rag_config())
        self.tools = agent_tools.Tools(
            kwargs=config.get_tools_config(),
        )

    def answer(self, command: commands.Command) -> commands.Command:
        """
        Answer a command. Processes each request by the command type

        Args:
            command: commands.Command: The command to answer.

        Returns:
            commands.Command: The command to answer.
        """
        match command:
            case commands.Question():
                response = self.question(command)
            case commands.Check():
                response = self.check(command)
            case commands.Retrieve():
                response = self.retrieve(command)
            case commands.Rerank():
                response = self.rerank(command)
            case commands.Enhance():
                response = self.enhance(command)
            case commands.UseTools():
                response = self.use(command)
            case commands.LLMResponse():
                response = self.finalize(command)
            case commands.FinalCheck():
                response = self.evaluate(command)
            case _:
                raise NotImplementedError(
                    f"Not implemented in AgentAdapter: {type(command)}"
                )
        return response

    async def answer_async(self, command: commands.Command) -> commands.Command:
        """
        Answer a command asynchronously. Processes each request by the command type

        Args:
            command: commands.Command: The command to answer.

        Returns:
            commands.Command: The command to answer.
        """
        match command:
            case commands.Question():
                response = self.question(command)
            case commands.Check():
                response = await self.check_async(command)
            case commands.Retrieve():
                response = await self.retrieve_async(command)
            case commands.Rerank():
                response = await self.rerank_async(command)
            case commands.Enhance():
                response = await self.enhance_async(command)
            case commands.UseTools():
                response = await self.use_async(command)
            case commands.LLMResponse():
                response = await self.finalize_async(command)
            case commands.FinalCheck():
                response = await self.evaluate_async(command)
            case _:
                raise NotImplementedError(
                    f"Not implemented in AgentAdapter: {type(command)}"
                )
        return response

    @observe()
    def check(self, command: commands.Check) -> commands.Check:
        """
        Check the incoming question via guardrails.

        Args:
            command: commands.Check: The command to check.

        Returns:
            commands.Check: The command to check.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="check",
            session_id=command.q_id,
        )
        response = self.guardrails.use(
            command.question, commands.GuardrailPreCheckModel
        )

        command.response = response.response
        command.chain_of_thought = response.chain_of_thought
        command.approved = response.approved

        return command

    @observe()
    def enhance(self, command: commands.Enhance):
        """
        Enhance the question via LLM based on the reranked document.

        Args:
            command: commands.Enhance: The command to enhance the question.

        Returns:
            commands.Enhance: The command to enhance the question.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name=TraceNames.ENHANCE,
            session_id=command.q_id,
        )

        response = self.llm.use(command.question, commands.LLMResponseModel)

        command.response = response.response
        command.chain_of_thought = response.chain_of_thought

        return command

    @observe()
    def evaluate(self, command: commands.FinalCheck) -> commands.FinalCheck:
        """
        Evaluate the response via guardrails.

        Args:
            command: commands.FinalCheck: The command to evaluate.

        Returns:
            commands.FinalCheck: The command to evaluate.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="evaluation",
            session_id=command.q_id,
        )
        response = self.guardrails.use(
            command.question, commands.GuardrailPostCheckModel
        )

        command.chain_of_thought = response.chain_of_thought
        command.approved = response.approved
        command.summary = response.summary
        command.issues = response.issues
        command.plausibility = response.plausibility
        command.factual_consistency = response.factual_consistency
        command.clarity = response.clarity
        command.completeness = response.completeness

        return command

    @observe()
    def finalize(self, command: commands.LLMResponse) -> commands.LLMResponse:
        """
        Finalize the response via LLM.

        Args:
            command: commands.LLMResponse: The command to finalize the response.

        Returns:
            commands.LLMResponse: The command to finalize the response.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name=TraceNames.FINALIZE,
            session_id=command.q_id,
        )

        response = self.llm.use(command.question, commands.LLMResponseModel)

        command.response = response.response
        command.chain_of_thought = response.chain_of_thought

        return command

    @observe()
    def question(self, command: commands.Question) -> commands.Question:
        """
        Only for tracing.

        Args:
            command: commands.Question: The command to handle a question.

        Returns:
            commands.Question: The command to handle a question.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="question",
            session_id=command.q_id,
        )

        return command

    @observe()
    def rerank(self, command: commands.Rerank):
        """
        Rerank the documents from the knowledge base.

        Args:
            command: commands.Rerank: The command to rerank the documents.

        Returns:
            commands.Rerank: The command to rerank the documents.
        """
        candidates = []

        for candidate in command.candidates:
            response = self.rag.rerank(command.question, candidate.description)

            temp = candidate.model_dump()
            temp.pop("score", None)
            candidates.append(commands.RerankResponse(**response, **temp))

        candidates = sorted(candidates, key=lambda x: -x.score)

        command.candidates = candidates[: self.rag.n_ranking_candidates]
        return command

    @observe()
    def retrieve(self, command: commands.Retrieve):
        """
        Retrieve the most relevant documents from the knowledge base.

        Args:
            command: commands.Retrieve: The command to retrieve the most relevant documents.

        Returns:
            commands.Retrieve: The command to retrieve the most relevant documents.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="retrieve",
            session_id=command.q_id,
        )
        candidates = []
        response = self.rag.embed(command.question)

        if response is not None:
            response = self.rag.retrieve(response["embedding"])

            for candidate in response["results"]:
                candidates.append(commands.KBResponse(**candidate))

        command.candidates = candidates
        return command

    @observe()
    def use(self, command: commands.UseTools) -> commands.UseTools:
        """
        Use the agent tools to process the question.

        Args:
            command: commands.UseTools: The command to use the agent tools.

        Returns:
            commands.UseTools: The command to use the agent tools.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="use",
            session_id=command.q_id,
        )
        response, memory = self.tools.use(command.question)

        command.memory = memory

        if isinstance(response, dict) and "data" in response:
            command.data = response
            command.response = (
                "Response is a data extraction. FileStorage is not implemented yet."
            )

        elif isinstance(response, dict) and "plot" in response:
            command.response = "Response is a plot."
            command.data = response
        else:
            command.response = response

        return command

    # Async versions of all methods
    @observe()
    async def check_async(self, command: commands.Check) -> commands.Check:
        """
        Check the incoming question via guardrails (async).

        Args:
            command: commands.Check: The command to check.

        Returns:
            commands.Check: The command to check.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="check_async",
            session_id=command.q_id,
        )
        response = await self.guardrails.use_async(
            command.question, commands.GuardrailPreCheckModel
        )

        command.response = response.response
        command.chain_of_thought = response.chain_of_thought
        command.approved = response.approved

        return command

    @observe()
    async def enhance_async(self, command: commands.Enhance):
        """
        Enhance the question via LLM based on the reranked document (async).

        Args:
            command: commands.Enhance: The command to enhance the question.

        Returns:
            commands.Enhance: The command to enhance the question.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name=TraceNames.ENHANCE + "_async",
            session_id=command.q_id,
        )

        response = await self.llm.use_async(command.question, commands.LLMResponseModel)

        command.response = response.response
        command.chain_of_thought = response.chain_of_thought

        return command

    @observe()
    async def evaluate_async(self, command: commands.FinalCheck) -> commands.FinalCheck:
        """
        Evaluate the response via guardrails (async).

        Args:
            command: commands.FinalCheck: The command to evaluate.

        Returns:
            commands.FinalCheck: The command to evaluate.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="evaluation_async",
            session_id=command.q_id,
        )
        response = await self.guardrails.use_async(
            command.question, commands.GuardrailPostCheckModel
        )

        command.chain_of_thought = response.chain_of_thought
        command.approved = response.approved
        command.summary = response.summary
        command.issues = response.issues
        command.plausibility = response.plausibility
        command.factual_consistency = response.factual_consistency
        command.clarity = response.clarity
        command.completeness = response.completeness

        return command

    @observe()
    async def finalize_async(
        self, command: commands.LLMResponse
    ) -> commands.LLMResponse:
        """
        Finalize the response via LLM (async).

        Args:
            command: commands.LLMResponse: The command to finalize the response.

        Returns:
            commands.LLMResponse: The command to finalize the response.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name=TraceNames.FINALIZE + "_async",
            session_id=command.q_id,
        )

        response = await self.llm.use_async(command.question, commands.LLMResponseModel)

        command.response = response.response
        command.chain_of_thought = response.chain_of_thought

        return command

    @observe()
    async def rerank_async(self, command: commands.Rerank):
        """
        Rerank the documents from the knowledge base (async).

        Args:
            command: commands.Rerank: The command to rerank the documents.

        Returns:
            commands.Rerank: The command to rerank the documents.
        """
        candidates = []

        # Process reranking concurrently for better performance
        rerank_tasks = []
        for candidate in command.candidates:
            rerank_tasks.append(
                self.rag.rerank_async(command.question, candidate.description)
            )

        rerank_responses = await asyncio.gather(*rerank_tasks)

        for candidate, response in zip(command.candidates, rerank_responses):
            temp = candidate.model_dump()
            temp.pop("score", None)
            candidates.append(commands.RerankResponse(**response, **temp))

        candidates = sorted(candidates, key=lambda x: -x.score)

        command.candidates = candidates[: self.rag.n_ranking_candidates]
        return command

    @observe()
    async def retrieve_async(self, command: commands.Retrieve):
        """
        Retrieve the most relevant documents from the knowledge base (async).

        Args:
            command: commands.Retrieve: The command to retrieve the most relevant documents.

        Returns:
            commands.Retrieve: The command to retrieve the most relevant documents.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="retrieve_async",
            session_id=command.q_id,
        )
        candidates = []
        response = await self.rag.embed_async(command.question)

        if response is not None:
            response = await self.rag.retrieve_async(response["embedding"])

            for candidate in response["results"]:
                candidates.append(commands.KBResponse(**candidate))

        command.candidates = candidates
        return command

    @observe()
    async def use_async(self, command: commands.UseTools) -> commands.UseTools:
        """
        Use the agent tools to process the question (async).

        Args:
            command: commands.UseTools: The command to use the agent tools.

        Returns:
            commands.UseTools: The command to use the agent tools.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="use_async",
            session_id=command.q_id,
        )
        # Use thread wrapper since tools don't have async method yet
        response, memory = await asyncio.to_thread(self.tools.use, command.question)

        command.memory = memory

        if isinstance(response, dict) and "data" in response:
            command.data = response
            command.response = (
                "Response is a data extraction. FileStorage is not implemented yet."
            )

        elif isinstance(response, dict) and "plot" in response:
            command.response = "Response is a plot."
            command.data = response
        else:
            command.response = response

        return command

    # Placeholder async methods that wrap sync methods for compatibility
    async def validation_async(self, command):
        return await asyncio.to_thread(self.validation, command)


class SQLAgentAdapter(AbstractAdapter):
    """
    SQLAgentAdapter is an adapter for the SQL agent.
    It defines the flow of commands from the model agent to the external service.

    Question -> Check -> UseTools -> Retrieve -> Rerank -> Enhance -> LLMResponse -> FinalCheck

    Methods:
        - aggregation(command: commands.SQLAggregation) -> commands.SQLAggregation: Aggregate the question to schema elements.
        - check(command: commands.SQLCheck) -> commands.SQLCheck: Check the incoming question via guardrails.
        - construction(command: commands.SQLConstruction) -> commands.SQLConstruction: Construct the question to schema elements.
        - filter(command: commands.SQLFilter) -> commands.SQLFilter: Validate the question to schema elements.
        - finalize(command: commands.LLMResponse) -> commands.LLMResponse: Finalize the response via LLM.
        - grounding(command: commands.SQLGrounding) -> commands.SQLGrounding: Ground the question to schema elements.
        - join_inference(command: commands.SQLJoinInference) -> commands.SQLJoinInference: Filter the question to schema elements.
        - question(command: commands.SQLQuestion) -> commands.SQLQuestion: only for tracing.
        - validation(command: commands.SQLValidation) -> commands.SQLValidation: Validate the question to schema elements.

    Adapters:
        - database: Database adapter.
        - guardrails: Performs checks via guardrails.
        - llm: Calls a LLM.
    """

    def __init__(self):
        super().__init__()

        self.database = database.BaseDatabaseAdapter(
            kwargs=config.get_database_config(),
        )

        self.guardrails = llm.LLM(
            kwargs=config.get_guardrails_config(),
        )
        self.llm = llm.LLM(
            kwargs=config.get_llm_config(),
        )
        self.rag = rag.BaseRAG(config.get_rag_config())

    @observe()
    def aggregation(self, command: commands.SQLAggregation) -> commands.SQLAggregation:
        """
        Aggregate the question to schema elements.

        Args:
            command: commands.SQLAggregation: The command to aggregation.

        Returns:
            commands.SQLAggregation: The command to aggregation.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="aggregation",
            session_id=command.q_id,
        )
        response = self.llm.use(command.question, commands.AggregationResponse)

        command.aggregations = response.aggregations
        command.group_by_columns = response.group_by_columns
        command.is_aggregation_query = response.is_aggregation_query
        command.chain_of_thought = response.chain_of_thought

        return command

    @observe()
    def check(self, command: commands.Check) -> commands.Check:
        """
        Check the incoming question via guardrails.

        Args:
            command: commands.Check: The command to check.

        Returns:
            commands.Check: The command to check.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="check",
            session_id=command.q_id,
        )
        response = self.guardrails.use(
            command.question, commands.GuardrailPreCheckModel
        )

        command.response = response.response
        command.chain_of_thought = response.chain_of_thought
        command.approved = response.approved

        return command

    @observe()
    def construction(
        self, command: commands.SQLConstruction
    ) -> commands.SQLConstruction:
        """
        Construct the question to schema elements.

        Args:
            command: commands.SQLConstruction: The command to construction.

        Returns:
            commands.SQLConstruction: The command to construction.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="construction",
            session_id=command.q_id,
        )
        response = self.llm.use(command.question, commands.ConstructionResponse)

        command.sql_query = response.sql_query
        command.chain_of_thought = response.chain_of_thought

        return command

    def convert_schema(self, schema: MetaData) -> commands.DatabaseSchema:
        """
        Convert the schema to a more readable format.
        """

        tables = []
        for table_name, table in schema.tables.items():
            # Create Column objects for each column
            columns = []
            for column in table.columns:
                if column.name in ["created_at", "updated_at"]:
                    continue

                columns.append(
                    commands.Column(
                        name=column.name,
                        type=str(column.type),
                        description=column.description,
                    )
                )

            # Create Table object
            tables.append(
                commands.Table(
                    name=table_name, columns=columns, description=table.description
                )
            )

        # Build relationships list
        relationships = []
        for table_name, table in schema.tables.items():
            for fk in table.foreign_keys:
                relationship = commands.Relationship(
                    table_name=table_name,
                    column_name=fk.parent.name,
                    foreign_table_name=fk.column.table.name,
                    foreign_column_name=fk.column.name,
                )
                relationships.append(relationship)

        new_schema = commands.DatabaseSchema(
            tables=tables,
            relationships=relationships,
        )

        return new_schema

    @observe()
    def filter(self, command: commands.SQLFilter) -> commands.SQLFilter:
        """
        Validate the question to schema elements.

        Args:
            command: commands.SQLValidation: The command to validation.

        Returns:
            commands.SQLValidation: The command to validation.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="validation",
            session_id=command.q_id,
        )
        response = self.llm.use(command.question, commands.FilterResponse)

        command.chain_of_thought = response.chain_of_thought
        command.conditions = response.conditions

        return command

    @observe()
    def grounding(self, command: commands.SQLGrounding) -> commands.SQLGrounding:
        """
        Ground the question to schema elements.

        Args:
            command: commands.SQLGrounding: The command to ground.

        Returns:
            commands.SQLGrounding: The command to ground.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="grounding",
            session_id=command.q_id,
        )
        response = self.llm.use(command.question, commands.GroundingResponse)

        command.table_mapping = response.table_mapping
        command.column_mapping = response.column_mapping
        command.chain_of_thought = response.chain_of_thought

        return command

    @observe()
    def join_inference(
        self, command: commands.SQLJoinInference
    ) -> commands.SQLJoinInference:
        """
        Filter the question to schema elements.

        Args:
            command: commands.SQLJoinInference: The command to join inference.

        Returns:
            commands.SQLJoinInference: The command to join inference.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="grounding",
            session_id=command.q_id,
        )
        response = self.llm.use(command.question, commands.JoinInferenceResponse)

        command.joins = response.joins
        command.chain_of_thought = response.chain_of_thought

        return command

    def query(self, command: commands.Command) -> commands.Command:
        """
        Answer a command. Processes each request by the command type

        Args:
            command: commands.Command: The command to answer.

        Returns:
            commands.Command: The command to answer.
        """
        match command:
            case commands.SQLQuestion():
                response = self.question(command)
            case commands.SQLCheck():
                response = self.check(command)
            case commands.SQLGrounding():
                response = self.grounding(command)
            case commands.SQLFilter():
                response = self.filter(command)
            case commands.SQLJoinInference():
                response = self.join_inference(command)
            case commands.SQLAggregation():
                response = self.aggregation(command)
            case commands.SQLConstruction():
                response = self.construction(command)
            case commands.SQLExecution():
                response = self.sql_execution(command)
            case commands.SQLValidation():
                response = self.validation(command)
            case _:
                raise NotImplementedError(
                    f"Not implemented in AgentAdapter: {type(command)}"
                )
        return response

    async def query_async(self, command: commands.Command) -> commands.Command:
        """
        Answer a command asynchronously. Processes each request by the command type

        Args:
            command: commands.Command: The command to answer.

        Returns:
            commands.Command: The command to answer.
        """
        match command:
            case commands.SQLQuestion():
                response = self.question(command)
            case commands.SQLCheck():
                response = await self.check_async(command)
            case commands.SQLGrounding():
                response = await self.grounding_async(command)
            case commands.SQLFilter():
                response = await self.filter_async(command)
            case commands.SQLJoinInference():
                response = await self.join_inference_async(command)
            case commands.SQLAggregation():
                response = await self.aggregation_async(command)
            case commands.SQLConstruction():
                response = await self.construction_async(command)
            case commands.SQLExecution():
                response = await self.sql_execution_async(command)
            case commands.SQLValidation():
                response = await self.validation_async(command)
            case _:
                raise NotImplementedError(
                    f"Not implemented in SQLAgentAdapter: {type(command)}"
                )
        return response

    @observe()
    def question(self, command: commands.Question) -> commands.Question:
        """
        Gets the schema info from the database.

        Args:
            command: commands.Question: The command to handle a question.

        Returns:
            commands.Question: The command to handle a question.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="question",
            session_id=command.q_id,
        )

        with self.database as db:
            schema = db.get_schema()

        schema = self.convert_schema(schema)

        command.schema_info = schema

        logger.info(
            f"Schema created with {len(schema.tables)} tables and {len(schema.relationships)} relationships"
        )

        return command

    @observe()
    def sql_execution(self, command: commands.SQLExecution) -> commands.SQLExecution:
        """
        Execute the SQL query.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="sql_execution",
            session_id=command.q_id,
        )

        with self.database as db:
            data = db.execute_query(command.sql_query)

        command.data = data

        return command

    @observe()
    def validation(self, command: commands.SQLValidation) -> commands.SQLValidation:
        """
        Ground the question to schema elements.

        Args:
            command: commands.SQLFilter: The command to filter.

        Returns:
            commands.SQLFilter: The command to filter.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="validation",
            session_id=command.q_id,
        )

        response = self.llm.use(command.question, commands.ValidationResponse)

        command.approved = response.approved
        command.summary = response.summary
        command.issues = response.issues
        command.confidence = response.confidence
        command.chain_of_thought = response.chain_of_thought

        return command

    # Placeholder async methods that wrap sync methods for compatibility
    async def check_async(self, command):
        return await asyncio.to_thread(self.check, command)

    async def grounding_async(self, command):
        return await asyncio.to_thread(self.grounding, command)

    async def filter_async(self, command):
        return await asyncio.to_thread(self.filter, command)

    async def join_inference_async(self, command):
        return await asyncio.to_thread(self.join_inference, command)

    async def aggregation_async(self, command):
        return await asyncio.to_thread(self.aggregation, command)

    async def construction_async(self, command):
        return await asyncio.to_thread(self.construction, command)

    async def sql_execution_async(self, command):
        return await asyncio.to_thread(self.sql_execution, command)

    async def validation_async(self, command):
        return await asyncio.to_thread(self.validation, command)


class ScenarioAdapter(AbstractAdapter):
    """
    ScenarioAdapter is an adapter for the scenario agent.
    It defines the flow of commands from the model agent to the external service.

    Question -> Check -> LLMResponse -> FinalCheck

    Methods:
        - check(command: commands.Scenario) -> commands.Scenario: Check the incoming question via guardrails.
        - finalize(command: commands.LLMResponse) -> commands.LLMResponse: Finalize the response via LLM.
        - question(command: commands.Scenario) -> commands.Scenario: only for tracing.
        - validation(command: commands.Scenario) -> commands.Scenario: Validate the question to schema elements.

    Adapters:
        - guardrails: Performs checks via guardrails.
        - llm: Calls a LLM.
    """

    def __init__(self):
        super().__init__()

        self.database = database.BaseDatabaseAdapter(
            kwargs=config.get_database_config(),
        )

        self.guardrails = llm.LLM(
            kwargs=config.get_guardrails_config(),
        )
        self.llm = llm.LLM(
            kwargs=config.get_llm_config(),
        )

    @observe()
    def check(self, command: commands.Scenario) -> commands.Scenario:
        """
        Check the incoming question via guardrails.

        Args:
            command: commands.Scenario: The command to check.

        Returns:
            commands.Scenario: The command to check.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="check",
            session_id=command.q_id,
        )
        response = self.guardrails.use(
            command.question, commands.GuardrailPreCheckModel
        )

        command.response = response.response
        command.chain_of_thought = response.chain_of_thought

        return command

    @observe()
    def finalize(
        self, command: commands.ScenarioLLMResponse
    ) -> commands.ScenarioLLMResponse:
        """
        Finalize the response via LLM.
        This is the final step in the scenario agent.

        Args:
            command: commands.ScenarioLLMResponse: The command to finalize.

        Returns:
            commands.ScenarioLLMResponse: The command to finalize.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name=TraceNames.FINALIZE,
            session_id=command.q_id,
        )
        response = self.llm.use(command.question, commands.ScenarioResponse)

        command.chain_of_thought = response.chain_of_thought
        command.candidates = response.candidates

        return command

    def convert_schema(self, schema: MetaData) -> commands.DatabaseSchema:
        """
        Convert the schema to a more readable format.
        """

        tables = []
        for table_name, table in schema.tables.items():
            # Create Column objects for each column
            columns = []
            for column in table.columns:
                if column.name in ["created_at", "updated_at"]:
                    continue

                columns.append(
                    commands.Column(
                        name=column.name,
                        type=str(column.type),
                        description=column.description,
                    )
                )

            # Create Table object
            tables.append(
                commands.Table(
                    name=table_name, columns=columns, description=table.description
                )
            )

        # Build relationships list
        relationships = []
        for table_name, table in schema.tables.items():
            for fk in table.foreign_keys:
                relationship = commands.Relationship(
                    table_name=table_name,
                    column_name=fk.parent.name,
                    foreign_table_name=fk.column.table.name,
                    foreign_column_name=fk.column.name,
                )
                relationships.append(relationship)

        new_schema = commands.DatabaseSchema(
            tables=tables,
            relationships=relationships,
        )

        return new_schema

    def query(self, command: commands.Command) -> commands.Command:
        """
        Answer a command. Processes each request by the command type

        Args:
            command: commands.Command: The command to answer.

        Returns:
            commands.Command: The command to answer.
        """
        match command:
            case commands.Scenario():
                response = self.question(command)
            case commands.Check():
                response = self.check(command)
            case commands.ScenarioLLMResponse():
                response = self.finalize(command)
            case commands.ScenarioFinalCheck():
                response = self.validation(command)
            case _:
                raise NotImplementedError(
                    f"Not implemented in ScenarioAdapter: {type(command)}"
                )
        return response

    async def query_async(self, command: commands.Command) -> commands.Command:
        """
        Answer a command asynchronously. Processes each request by the command type

        Args:
            command: commands.Command: The command to answer.

        Returns:
            commands.Command: The command to answer.
        """
        match command:
            case commands.Scenario():
                response = self.question(command)
            case commands.Check():
                response = await self.check_async(command)
            case commands.ScenarioLLMResponse():
                response = await self.finalize_async(command)
            case commands.ScenarioFinalCheck():
                response = await self.validation_async(command)
            case _:
                raise NotImplementedError(
                    f"Not implemented in ScenarioAdapter: {type(command)}"
                )
        return response

    @observe()
    def question(self, command: commands.Question) -> commands.Question:
        """
        Gets the schema info from the database.

        Args:
            command: commands.Question: The command to handle a question.

        Returns:
            commands.Question: The command to handle a question.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="question",
            session_id=command.q_id,
        )

        with self.database as db:
            schema = db.get_schema()

        schema = self.convert_schema(schema)

        command.schema_info = schema

        logger.info(
            f"Schema created with {len(schema.tables)} tables and {len(schema.relationships)} relationships"
        )

        return command

    @observe()
    def validation(
        self, command: commands.ScenarioFinalCheck
    ) -> commands.ScenarioFinalCheck:
        """
        Validate the question to schema elements.

        Args:
            command: commands.ScenarioFinalCheck: The command to validate.

        Returns:
            commands.ScenarioFinalCheck: The command to validate.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="validation",
            session_id=command.q_id,
        )

        response = self.llm.use(command.question, commands.ScenarioValidationResponse)

        command.approved = response.approved
        command.summary = response.summary
        command.issues = response.issues
        command.plausibility = response.plausibility
        command.usefulness = response.usefulness
        command.clarity = response.clarity
        command.chain_of_thought = response.chain_of_thought

        return command

    # Placeholder async methods that wrap sync methods for compatibility
    async def check_async(self, command):
        return await asyncio.to_thread(self.check, command)

    async def finalize_async(self, command):
        return await asyncio.to_thread(self.finalize, command)

    async def validation_async(self, command):
        return await asyncio.to_thread(self.validation, command)
