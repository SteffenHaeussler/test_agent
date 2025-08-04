"""Test SQLAgent with SQL validator integration."""

import pytest
from unittest.mock import patch, MagicMock

from src.agent.domain import commands
from src.agent.domain.sql_model import SQLBaseAgent


class TestSQLAgentWithValidator:
    """Test suite for SQLAgent with integrated SQL validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.question = commands.Question(question="test query", q_id="test_id")
        self.config = MagicMock()
        self.config.get.return_value = "mock_prompt"

    def test_sql_agent_validates_query_before_execution(self):
        """Should validate SQL query before execution."""
        # Arrange
        agent = SQLBaseAgent(self.question, self.config)

        # Create a command with malicious SQL
        command = commands.SQLConstruction(
            question="What users exist?",
            q_id="test_id",
            sql_query="SELECT * FROM users; DROP TABLE users;--",
        )

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            agent.prepare_execution(command)

        assert "DROP operations are not allowed" in str(exc_info.value)

    def test_sql_agent_accepts_valid_query(self):
        """Should accept valid SELECT queries."""
        # Arrange
        agent = SQLBaseAgent(self.question, self.config)

        # Create a command with valid SQL
        command = commands.SQLConstruction(
            question="What users exist?",
            q_id="test_id",
            sql_query="SELECT id, name FROM users WHERE active = true;",
        )

        # Act
        result = agent.prepare_execution(command)

        # Assert - should return an Execution command
        assert isinstance(result, commands.SQLExecution)
        assert result.sql_query == command.sql_query

    def test_sql_agent_rejects_delete_in_construction(self):
        """Should reject DELETE operations in construction phase."""
        # Arrange
        agent = SQLBaseAgent(self.question, self.config)

        command = commands.SQLConstruction(
            question="Remove inactive users",
            q_id="test_id",
            sql_query="DELETE FROM users WHERE active = false;",
        )

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            agent.prepare_execution(command)

        assert "DELETE operations are not allowed" in str(exc_info.value)

    def test_sql_agent_rejects_drop_in_construction(self):
        """Should reject DROP operations in construction phase."""
        # Arrange
        agent = SQLBaseAgent(self.question, self.config)

        command = commands.SQLConstruction(
            question="Clean up old tables",
            q_id="test_id",
            sql_query="DROP TABLE obsolete_data;",
        )

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            agent.prepare_execution(command)

        assert "DROP operations are not allowed" in str(exc_info.value)

    def test_sql_agent_handles_empty_query(self):
        """Should handle empty SQL queries gracefully."""
        # Arrange
        agent = SQLBaseAgent(self.question, self.config)

        command = commands.SQLConstruction(
            question="What users exist?", q_id="test_id", sql_query=""
        )

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            agent.prepare_execution(command)

        assert "Empty SQL query" in str(exc_info.value)

    def test_sql_agent_validates_cte_queries(self):
        """Should accept valid CTE queries."""
        # Arrange
        agent = SQLBaseAgent(self.question, self.config)

        command = commands.SQLConstruction(
            question="Get user statistics",
            q_id="test_id",
            sql_query="""
            WITH user_stats AS (
                SELECT user_id, COUNT(*) as order_count
                FROM orders
                GROUP BY user_id
            )
            SELECT u.name, s.order_count
            FROM users u
            JOIN user_stats s ON u.id = s.user_id;
            """,
        )

        # Act
        result = agent.prepare_execution(command)

        # Assert
        assert isinstance(result, commands.SQLExecution)
        assert result.sql_query == command.sql_query

    @patch("src.agent.domain.sql_model.logger")
    def test_sql_agent_logs_validation_failures(self, mock_logger):
        """Should log SQL validation failures."""
        # Arrange
        agent = SQLBaseAgent(self.question, self.config)

        command = commands.SQLConstruction(
            question="Malicious query",
            q_id="test_id",
            sql_query="SELECT * FROM users UNION SELECT * FROM passwords",
        )

        # Act
        try:
            agent.prepare_execution(command)
        except ValueError:
            pass

        # Assert - should have logged the validation failure
        assert mock_logger.error.called or mock_logger.warning.called
