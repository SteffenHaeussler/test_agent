import pytest

from src.agent.validators.sql_validator import SQLValidator


class TestSQLValidator:
    """Test suite for SQL injection protection validator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SQLValidator()

    def test_rejects_drop_statements(self):
        """Should reject DROP statements."""
        # Arrange
        sql_query = "DROP TABLE users;"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            self.validator.validate(sql_query)

        assert "DROP operations are not allowed" in str(exc_info.value)

    def test_rejects_delete_statements(self):
        """Should reject DELETE statements."""
        # Arrange
        sql_query = "DELETE FROM users WHERE id = 1;"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            self.validator.validate(sql_query)

        assert "DELETE operations are not allowed" in str(exc_info.value)

    def test_rejects_insert_statements(self):
        """Should reject INSERT statements."""
        # Arrange
        sql_query = "INSERT INTO users (name) VALUES ('test');"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            self.validator.validate(sql_query)

        assert "INSERT operations are not allowed" in str(exc_info.value)

    def test_rejects_update_statements(self):
        """Should reject UPDATE statements."""
        # Arrange
        sql_query = "UPDATE users SET name = 'test' WHERE id = 1;"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            self.validator.validate(sql_query)

        assert "UPDATE operations are not allowed" in str(exc_info.value)

    def test_accepts_valid_select_statements(self):
        """Should accept valid SELECT statements."""
        # Arrange
        sql_queries = [
            "SELECT * FROM users;",
            "SELECT id, name FROM users WHERE active = true;",
            "SELECT COUNT(*) FROM orders;",
            "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id;",
        ]

        # Act & Assert
        for sql_query in sql_queries:
            # Should not raise any exception
            result = self.validator.validate(sql_query)
            assert result is True

    def test_accepts_cte_with_select(self):
        """Should accept CTEs (WITH clause) when used with SELECT."""
        # Arrange
        sql_query = """
        WITH active_users AS (
            SELECT * FROM users WHERE active = true
        )
        SELECT * FROM active_users;
        """

        # Act & Assert
        result = self.validator.validate(sql_query)
        assert result is True

    def test_rejects_nested_forbidden_operations(self):
        """Should reject SELECT with subquery containing forbidden operations."""
        # Arrange
        sql_queries = [
            "SELECT * FROM (DELETE FROM users RETURNING *);",
            "SELECT * FROM users WHERE id IN (UPDATE orders SET status = 'done' RETURNING user_id);",
            "WITH deleted AS (DROP TABLE temp) SELECT * FROM users;",
        ]

        # Act & Assert
        for sql_query in sql_queries:
            with pytest.raises(ValueError):
                self.validator.validate(sql_query)

    def test_rejects_sql_injection_attempts(self):
        """Should reject common SQL injection patterns."""
        # Arrange
        sql_queries = [
            "SELECT * FROM users; DROP TABLE users;--",
            "SELECT * FROM users WHERE name = ''; DELETE FROM users;--'",
            "SELECT * FROM users UNION SELECT * FROM passwords",  # Changed to UNION without ALL
        ]

        # Act & Assert
        for sql_query in sql_queries:
            with pytest.raises(ValueError):
                self.validator.validate(sql_query)

    def test_case_insensitive_validation(self):
        """Should validate SQL keywords regardless of case."""
        # Arrange
        sql_queries = [
            "drop table users;",
            "DeLeTe FROM users;",
            "InSeRt INTO users VALUES (1);",
            "UpDaTe users SET name = 'test';",
        ]

        # Act & Assert
        for sql_query in sql_queries:
            with pytest.raises(ValueError):
                self.validator.validate(sql_query)

    def test_handles_multiline_queries(self):
        """Should properly handle multiline SQL queries."""
        # Arrange
        valid_query = """
        SELECT
            u.id,
            u.name,
            COUNT(o.id) as order_count
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        GROUP BY u.id, u.name
        HAVING COUNT(o.id) > 5;
        """

        invalid_query = """
        SELECT * FROM users;

        DROP TABLE sensitive_data;
        """

        # Act & Assert
        # Valid query should pass
        result = self.validator.validate(valid_query)
        assert result is True

        # Invalid query should fail
        with pytest.raises(ValueError):
            self.validator.validate(invalid_query)

    def test_empty_or_none_query(self):
        """Should handle empty or None queries gracefully."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            self.validator.validate("")
        assert "Empty SQL query" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            self.validator.validate(None)
        assert "Empty SQL query" in str(exc_info.value)

    def test_whitespace_only_query(self):
        """Should reject whitespace-only queries."""
        # Arrange
        sql_queries = ["   ", "\n\n", "\t\t", "   \n\t   "]

        # Act & Assert
        for sql_query in sql_queries:
            with pytest.raises(ValueError) as exc_info:
                self.validator.validate(sql_query)
            assert "Empty SQL query" in str(exc_info.value)
