import time
import uuid
from pathlib import Path

import pytest

from evals.utils import get_model_info_for_test, load_yaml_fixtures, save_test_report
from src.agent.adapters.llm import LLM
from src.agent.domain import commands, sql_model

current_path = Path(__file__).parent
# Load fixtures from YAML file
fixtures = load_yaml_fixtures(current_path, "pre_check")


class TestEvalPreCheck:
    """Pre-check guardrails evaluation tests."""

    def setup_class(self):
        """Setup report file."""
        self.results = []

    def teardown_class(self):
        """Save results to report file."""
        model_info = get_model_info_for_test("sql_pre_check")
        save_test_report(self.results, "sql_pre_check", model_info)

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval(self, fixture_name, fixture, agent_config, llm_config):
        question_text = fixture["question"]
        expected_response = fixture["approved"]

        q_id = "eval_pre_check_" + str(uuid.uuid4())
        question = commands.SQLQuestion(question=question_text, q_id=q_id)

        llm = LLM(llm_config)
        agent = sql_model.SQLBaseAgent(
            question=question,
            kwargs=agent_config,
        )

        start_time = time.time()

        # Prepare guardrails check
        check = agent.update(question)
        response = llm.use(
            check.question, response_model=commands.GuardrailPreCheckModel
        )

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Add delay to avoid rate limiting
        time.sleep(1)
        # Extract response
        actual_response = response.approved

        # Record result
        result = {
            "test_name": fixture_name,
            "question": question_text,
            "expected": expected_response,
            "actual": actual_response,
            "passed": actual_response == expected_response,
            "execution_time_ms": execution_time_ms,
        }

        self.__class__.results.append(result)

        # Simple assert for exact match
        assert actual_response == expected_response
