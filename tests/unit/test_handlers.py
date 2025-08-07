from collections import defaultdict
from unittest.mock import Mock

import pandas as pd
import pytest

from src.agent.adapters.adapter import (
    AbstractAdapter,
    RouterAdapter,
    ScenarioAdapter,
    SQLAgentAdapter,
)
from src.agent.adapters.notifications import AbstractNotifications
from src.agent.bootstrap import bootstrap
from src.agent.domain import commands, events


class MockMetadata:
    """Mock SQLAlchemy MetaData object."""

    def __init__(self):
        self.tables = {}  # MetaData.tables is a dict


class MockContextManager:
    """A mock context manager for database operations."""

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def fetch_schema_info(self):
        return commands.DatabaseSchema(tables=[], relationships=[])

    def get_schema(self):
        # Return a mock MetaData object for ScenarioAdapter
        return MockMetadata()


class FakeAgentAdapter(AbstractAdapter):
    def __init__(self):
        # Initialize minimal attributes needed for testing
        self.agent = None
        self.database = None
        self.llm = None
        self.tools = None
        self.rag = None
        self.guardrails = None

    def collect_new_events(self):
        """Override to handle None agent case."""
        if self.agent is None:
            return []
        return super().collect_new_events()

    async def answer_async(self, command):
        """Async version of answer method."""
        return self.answer(command)

    def answer(self, command):
        if type(command) is commands.Question:
            response = command
        elif isinstance(command, commands.UseTools):
            response = command
            response.response = "test tools response"
            response.memory = ["test memory"]
        elif isinstance(command, commands.LLMResponse):
            response = command
            response.response = "test second llm response"
            response.chain_of_thought = "test chain of thought"
        elif isinstance(command, commands.Enhance):
            response = command
            response.response = "test first llm response"
            response.chain_of_thought = "test chain of thought"
        elif isinstance(command, commands.Rerank):
            response = command
            response.candidates = [
                commands.RerankResponse(
                    question="test question",
                    text="test text",
                    score=0.5,
                    id="test id",
                    tag="test tag",
                    name="test name",
                )
            ]
        elif isinstance(command, commands.Retrieve):
            response = command
        elif isinstance(command, commands.Check):
            if command.q_id == "test_session_id":
                response = command
                response.response = "test first llm response"
                response.approved = True
            else:
                response = command
                response.response = "test second llm response"
        elif isinstance(command, commands.FinalCheck):
            response = command
            response.chain_of_thought = "test chain of thought"
            response.approved = True
            response.summary = "test summary"
            response.issues = []
            response.plausibility = "test plausibility"
            response.factual_consistency = "test factual consistency"
        return response


class FakeSQLAgentAdapter(SQLAgentAdapter):
    def __init__(self):
        # Initialize minimal attributes needed for testing
        self.agent = None
        self.database = None
        self.llm = None
        self.tools = None
        self.rag = None
        self.guardrails = None

    def collect_new_events(self):
        """Override to handle None agent case."""
        if self.agent is None:
            return []
        return super().collect_new_events()

    async def query_async(self, command):
        """Async version of query method."""
        return self.query(command)

    def query(self, command):
        if type(command) is commands.SQLQuestion:
            response = command
            response.schema_info = commands.DatabaseSchema(tables=[], relationships=[])
        elif isinstance(command, commands.SQLCheck):
            if command.q_id != "test_rejected_id":
                response = command
                response.response = "SQL check response"
                response.approved = True
            else:
                response = command
                response.response = "SQL check response"
                response.approved = False
        elif isinstance(command, commands.SQLGrounding):
            response = command
            response.table_mapping = []
            response.column_mapping = []
        elif isinstance(command, commands.SQLFilter):
            response = command
            response.conditions = []
        elif isinstance(command, commands.SQLJoinInference):
            response = command
            response.joins = []
        elif isinstance(command, commands.SQLAggregation):
            response = command
            response.aggregations = []
            response.group_by_columns = []
            response.is_aggregation_query = False
        elif isinstance(command, commands.SQLConstruction):
            response = command
            response.sql_query = "SELECT * FROM test_table"
        elif isinstance(command, commands.SQLExecution):
            response = command
            response.data = {"data": pd.DataFrame({"test": [1, 2, 3]})}
        elif isinstance(command, commands.SQLValidation):
            if command.q_id == "test_rejected_id":
                response = command
                response.approved = False
                response.summary = "SQL validation summary"
                response.issues = []
            else:
                response = command
                response.approved = True
                response.summary = "SQL validation summary"
                response.issues = []
        else:
            response = command
        return response


class FakeScenarioAdapter(ScenarioAdapter):
    def __init__(self):
        # Initialize minimal attributes needed for testing
        self.agent = None
        self.database = MockContextManager()  # Use mock context manager
        self.llm = Mock()  # Mock LLM
        self.tools = None
        self.rag = None
        self.guardrails = Mock()  # Mock guardrails

        # Set up mock responses for guardrails and llm
        # For check method
        def mock_guardrails_use(question, model):
            # Create a proper mock with all required attributes
            mock_response = Mock()
            # Set all possible attributes to avoid Mock sub-objects
            mock_response.response = "Scenario check response"
            mock_response.approved = True
            mock_response.summary = "Scenario validation summary"
            mock_response.issues = []
            return mock_response

        self.guardrails.use = Mock(side_effect=mock_guardrails_use)

        # Mock LLM for both finalize and validation methods
        def mock_llm_use(question, model):
            mock_response = Mock()
            if model.__name__ == "ScenarioResponse":
                # For finalize method
                mock_response.chain_of_thought = "test chain of thought"
                mock_response.candidates = [
                    commands.ScenarioCandidate(
                        question="test scenario question", endpoint="test endpoint"
                    )
                ]
            elif model.__name__ == "ScenarioValidationResponse":
                # For validation method
                mock_response.approved = True
                mock_response.summary = "Scenario validation summary"
                mock_response.issues = []
                mock_response.plausibility = "test plausibility"
                mock_response.usefulness = "test usefulness"
                mock_response.clarity = "test clarity"
                mock_response.chain_of_thought = "test chain of thought"
            return mock_response

        self.llm.use = Mock(side_effect=mock_llm_use)

    def collect_new_events(self):
        """Override to handle None agent case."""
        if self.agent is None:
            return []
        return super().collect_new_events()

    async def scenario_async(self, command):
        """Async version of query method."""
        return self.query(command)

    def query(self, command):
        if type(command) is commands.Scenario:
            response = command
            response.schema_info = commands.DatabaseSchema(tables=[], relationships=[])
        elif isinstance(command, commands.Check):
            response = command
            response.response = "Scenario check response"
            response.approved = True
        elif isinstance(command, commands.ScenarioLLMResponse):
            response = command
            response.candidates = [
                commands.ScenarioCandidate(
                    question="test scenario question", endpoint="test endpoint"
                )
            ]
        elif isinstance(command, commands.ScenarioFinalCheck):
            response = command
            response.approved = True
            response.summary = "Scenario validation summary"
            response.issues = []
        else:
            response = command
        return response


class FakeRouterAdapter(RouterAdapter):
    def __init__(self):
        # Initialize with our fake adapters
        self.agent_adapter = FakeAgentAdapter()
        self.sql_adapter = FakeSQLAgentAdapter()
        self.scenario_adapter = FakeScenarioAdapter()


class FakeNotifications(AbstractNotifications):
    def __init__(self):
        self.sent = defaultdict(list)

    def send(self, destination, event: events.Event):
        self.sent[destination].append(event)


def bootstrap_test_app():
    return bootstrap(adapter=FakeRouterAdapter(), notifications=[FakeNotifications()])


class TestAnswer:
    @pytest.mark.asyncio
    async def test_answers(self):
        bus = bootstrap_test_app()
        await bus.handle(
            commands.Question(question="test query", q_id="test_session_id")
        )

        # get the agent from the adapter
        agent = bus.adapter.agent_adapter.agent

        assert agent.q_id == "test_session_id"
        assert agent.question == "test query"

    @pytest.mark.asyncio
    async def test_answer_invalid_question(self):
        # With the new error handling, InvalidQuestion creates a FailedRequest event
        # We need to mock the event handler to verify this
        failed_request_handler = []

        def capture_failed_request(event, notifications=None):
            failed_request_handler.append(event)

        # Create custom event handlers that capture FailedRequest
        event_handlers = defaultdict(list)
        event_handlers[events.FailedRequest].append(capture_failed_request)

        # Create bus with custom handlers
        adapter = FakeRouterAdapter()
        bus = bootstrap(adapter=adapter)
        bus.event_handlers = event_handlers

        # Handle invalid question
        await bus.handle(commands.Question(question="", q_id="test_session_id"))

        # Verify FailedRequest was created
        assert len(failed_request_handler) == 1
        assert isinstance(failed_request_handler[0], events.FailedRequest)
        assert "InvalidQuestion" in failed_request_handler[0].exception

    @pytest.mark.asyncio
    async def test_for_new_agent(self):
        bus = bootstrap_test_app()

        assert bus.adapter.agent_adapter.agent is None

        await bus.handle(
            commands.Question(question="test query", q_id="test_session_id")
        )
        assert bus.adapter.agent_adapter.agent is not None

    @pytest.mark.asyncio
    async def test_return_response(self):
        bus = bootstrap_test_app()
        await bus.handle(
            commands.Question(question="test query", q_id="test_session_id")
        )

        agent = bus.adapter.agent_adapter.agent

        assert agent.response.response == "test second llm response"

    @pytest.mark.asyncio
    async def test_sends_notification(self):
        fake_notifs = FakeNotifications()
        bus = bootstrap(adapter=FakeRouterAdapter(), notifications=[fake_notifs])
        await bus.handle(
            commands.Question(question="test query", q_id="test_session_id")
        )

        test_request = events.Response(
            question="test query",
            response="test second llm response",
            q_id="test_session_id",
        )
        test_evaluation = events.Evaluation(
            question="test query",
            response="test second llm response",
            q_id="test_session_id",
            chain_of_thought="test chain of thought",
            approved=True,
            summary="test summary",
            issues=[],
            plausibility="test plausibility",
            factual_consistency="test factual consistency",
            clarity=None,
            completeness=None,
        )

        status_event = events.StatusUpdate(
            step_name="Processing...",
            q_id="test_session_id",
        )

        end_of_event = events.EndOfEvent(q_id="test_session_id")
        assert fake_notifs.sent["test_session_id"][0] == status_event
        assert fake_notifs.sent["test_session_id"][7] == test_request
        assert fake_notifs.sent["test_session_id"][9] == test_evaluation
        assert fake_notifs.sent["test_session_id"][10] == end_of_event
        assert len(fake_notifs.sent["test_session_id"]) == 11

    @pytest.mark.asyncio
    async def test_sends_rejected_notification(self):
        fake_notifs = FakeNotifications()
        bus = bootstrap(adapter=FakeRouterAdapter(), notifications=[fake_notifs])
        await bus.handle(
            commands.Question(question="test query", q_id="test_rejected_id")
        )

        rejected_request = events.RejectedRequest(
            question="test query",
            response="test second llm response",
            q_id="test_rejected_id",
        )
        end_of_event = events.EndOfEvent(q_id="test_rejected_id")

        status_event = events.StatusUpdate(
            step_name="Processing...",
            q_id="test_rejected_id",
        )

        assert fake_notifs.sent["test_rejected_id"][0] == status_event
        assert fake_notifs.sent["test_rejected_id"][2] == rejected_request
        assert fake_notifs.sent["test_rejected_id"][3] == end_of_event
        assert len(fake_notifs.sent["test_rejected_id"]) == 4


class TestQuery:
    @pytest.mark.asyncio
    async def test_sql_answers(self):
        bus = bootstrap_test_app()
        await bus.handle(
            commands.SQLQuestion(question="test query", q_id="test_sql_id")
        )

        # get the agent from the SQL adapter
        agent = bus.adapter.sql_adapter.agent

        assert agent.q_id == "test_sql_id"
        assert agent.question == "test query"

    @pytest.mark.asyncio
    async def test_sql_invalid_question(self):
        # With the new error handling, InvalidQuestion creates a FailedRequest event
        failed_request_handler = []

        def capture_failed_request(event, notifications=None):
            failed_request_handler.append(event)

        # Create custom event handlers that capture FailedRequest
        event_handlers = defaultdict(list)
        event_handlers[events.FailedRequest].append(capture_failed_request)

        # Create bus with custom handlers
        adapter = FakeRouterAdapter()
        bus = bootstrap(adapter=adapter)
        bus.event_handlers = event_handlers

        # Handle invalid question
        await bus.handle(commands.SQLQuestion(question="", q_id="test_sql_id"))

        # Verify FailedRequest was created
        assert len(failed_request_handler) == 1
        assert isinstance(failed_request_handler[0], events.FailedRequest)
        assert "InvalidQuestion" in failed_request_handler[0].exception

    @pytest.mark.asyncio
    async def test_sql_for_new_agent(self):
        bus = bootstrap_test_app()

        assert bus.adapter.sql_adapter.agent is None

        await bus.handle(
            commands.SQLQuestion(question="test query", q_id="test_sql_id")
        )
        assert bus.adapter.sql_adapter.agent is not None

    @pytest.mark.asyncio
    async def test_sql_return_response(self):
        bus = bootstrap_test_app()
        await bus.handle(
            commands.SQLQuestion(question="test query", q_id="test_sql_id")
        )

        agent = bus.adapter.sql_adapter.agent

        # SQL agent should have executed the query
        assert agent.sql_query == "SELECT * FROM test_table"
        assert (
            agent.response.response
            == "|   test |\n|-------:|\n|      1 |\n|      2 |\n|      3 |"
        )

    @pytest.mark.asyncio
    async def test_sql_sends_notification(self):
        fake_notifs = FakeNotifications()
        bus = bootstrap(adapter=FakeRouterAdapter(), notifications=[fake_notifs])
        await bus.handle(
            commands.SQLQuestion(question="test query", q_id="test_sql_id")
        )

        # Check that SQL-specific status updates are sent
        status_event = events.StatusUpdate(
            step_name="Processing...",
            q_id="test_sql_id",
        )

        # SQL has different steps than regular answer
        sql_response = events.Response(
            question="test query",
            response="|   test |\n|-------:|\n|      1 |\n|      2 |\n|      3 |",
            q_id="test_sql_id",
        )

        sql_validation = events.Evaluation(
            question="test query",
            response="|   test |\n|-------:|\n|      1 |\n|      2 |\n|      3 |",
            q_id="test_sql_id",
            approved=True,
            summary="SQL validation summary\n\nHere is the SQL query:\n\nSELECT * FROM test_table",
            issues=[],
            confidence=None,
        )

        end_of_event = events.EndOfEvent(q_id="test_sql_id", response="end")

        assert fake_notifs.sent["test_sql_id"][0] == status_event
        assert fake_notifs.sent["test_sql_id"][8] == sql_response
        assert fake_notifs.sent["test_sql_id"][10] == sql_validation
        assert fake_notifs.sent["test_sql_id"][11] == end_of_event
        assert len(fake_notifs.sent["test_sql_id"]) == 12

    @pytest.mark.asyncio
    async def test_sends_rejected_notification(self):
        fake_notifs = FakeNotifications()
        bus = bootstrap(adapter=FakeRouterAdapter(), notifications=[fake_notifs])
        await bus.handle(
            commands.SQLQuestion(question="test query", q_id="test_rejected_id")
        )

        rejected_request = events.RejectedRequest(
            question="test query",
            response="SQL check response",
            q_id="test_rejected_id",
        )
        end_of_event = events.EndOfEvent(q_id="test_rejected_id", response="end")

        status_event = events.StatusUpdate(
            step_name="Processing...",
            q_id="test_rejected_id",
        )

        assert fake_notifs.sent["test_rejected_id"][0] == status_event
        assert fake_notifs.sent["test_rejected_id"][2] == rejected_request
        assert fake_notifs.sent["test_rejected_id"][3] == end_of_event
        assert len(fake_notifs.sent["test_rejected_id"]) == 4


class TestScenario:
    @pytest.mark.asyncio
    async def test_scenario_answers(self):
        bus = bootstrap_test_app()
        await bus.handle(
            commands.Scenario(question="test scenario", q_id="test_scenario_id")
        )

        # get the agent from the scenario adapter
        agent = bus.adapter.scenario_adapter.agent

        assert agent.q_id == "test_scenario_id"
        assert agent.question == "test scenario"

    @pytest.mark.asyncio
    async def test_scenario_invalid_question(self):
        # With the new error handling, InvalidQuestion creates a FailedRequest event
        failed_request_handler = []

        def capture_failed_request(event, notifications=None):
            failed_request_handler.append(event)

        # Create custom event handlers that capture FailedRequest
        event_handlers = defaultdict(list)
        event_handlers[events.FailedRequest].append(capture_failed_request)

        # Create bus with custom handlers
        adapter = FakeRouterAdapter()
        bus = bootstrap(adapter=adapter)
        bus.event_handlers = event_handlers

        # Handle invalid question
        await bus.handle(commands.Scenario(question="", q_id="test_scenario_id"))

        # Verify FailedRequest was created
        assert len(failed_request_handler) == 1
        assert isinstance(failed_request_handler[0], events.FailedRequest)
        assert "InvalidQuestion" in failed_request_handler[0].exception

    @pytest.mark.asyncio
    async def test_scenario_for_new_agent(self):
        bus = bootstrap_test_app()

        assert bus.adapter.scenario_adapter.agent is None

        await bus.handle(
            commands.Scenario(question="test scenario", q_id="test_scenario_id")
        )
        assert bus.adapter.scenario_adapter.agent is not None

    @pytest.mark.asyncio
    async def test_scenario_return_response(self):
        bus = bootstrap_test_app()
        await bus.handle(
            commands.Scenario(question="test scenario", q_id="test_scenario_id")
        )

        agent = bus.adapter.scenario_adapter.agent

        # Scenario agent should have candidates
        assert (
            agent.response.response
            == '[{"sub_id": "sub-1", "question": "test scenario question", "endpoint": "test endpoint"}]'
        )

    @pytest.mark.asyncio
    async def test_scenario_sends_notification(self):
        fake_notifs = FakeNotifications()
        bus = bootstrap(adapter=FakeRouterAdapter(), notifications=[fake_notifs])
        await bus.handle(
            commands.Scenario(question="test scenario", q_id="test_scenario_id")
        )

        # Check that scenario-specific status updates are sent
        status_event = events.StatusUpdate(
            step_name="Processing...",
            q_id="test_scenario_id",
        )

        end_of_event = events.EndOfEvent(q_id="test_scenario_id", response="end")

        response = events.Response(
            question="test scenario",
            response='[{"sub_id": "sub-1", "question": "test scenario question", "endpoint": "test endpoint"}]',
            q_id="test_scenario_id",
        )

        scenario_validation = events.Evaluation(
            question="test scenario",
            response='[{"sub_id": "sub-1", "question": "test scenario question", "endpoint": "test endpoint"}]',
            q_id="test_scenario_id",
            approved=True,
            summary="Scenario validation summary",
            issues=[],
            confidence=None,
        )
        # breakpoint()
        assert fake_notifs.sent["test_scenario_id"][0] == status_event
        assert fake_notifs.sent["test_scenario_id"][3] == response
        assert fake_notifs.sent["test_scenario_id"][5] == scenario_validation
        assert fake_notifs.sent["test_scenario_id"][6] == end_of_event
        assert len(fake_notifs.sent["test_scenario_id"]) == 7
