"""Tests for enhanced SQL validator."""

import pytest

from src.agent.exceptions import InputValidationException
from src.agent.validators.config import ValidatorConfig


class TestEnhancedSQLValidator:
    """Test suite for enhanced SQL validator."""

    def test_should_accept_valid_sql_queries(self):
        """Should accept valid SQL queries."""
        from src.agent.validators.enhanced_sql_validator import EnhancedSQLValidator

        # Arrange
        validator = EnhancedSQLValidator()
        valid_queries = [
            "SELECT * FROM users",
            "SELECT id, name FROM users WHERE active = true",
            "SELECT COUNT(*) FROM orders",
            "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id",
            "WITH active_users AS (SELECT * FROM users WHERE active = true) SELECT * FROM active_users",
        ]

        # Act & Assert
        for query in valid_queries:
            result = validator.validate_input(query)
            assert result.is_valid is True, f"Query should be valid: {query}"
            assert len(result.errors) == 0

    def test_should_reject_empty_sql_queries(self):
        """Should reject empty or None SQL queries."""
        from src.agent.validators.enhanced_sql_validator import EnhancedSQLValidator

        # Arrange
        validator = EnhancedSQLValidator()
        empty_queries = [
            "",
            "   ",
            "\n\n",
            "\t\t",
            None,
        ]

        # Act & Assert
        for query in empty_queries:
            result = validator.validate_input(query)
            assert result.is_valid is False
            assert "empty" in " ".join(result.errors).lower()

    def test_should_reject_dangerous_sql_operations(self):
        """Should reject dangerous SQL operations."""
        from src.agent.validators.enhanced_sql_validator import EnhancedSQLValidator

        # Arrange
        validator = EnhancedSQLValidator()
        dangerous_queries = [
            "DROP TABLE users",
            "DELETE FROM users",
            "INSERT INTO users VALUES (1, 'test')",
            "UPDATE users SET name = 'test'",
            "CREATE TABLE test (id INT)",
            "ALTER TABLE users ADD COLUMN test VARCHAR(255)",
        ]

        # Act & Assert
        for query in dangerous_queries:
            result = validator.validate_input(query)
            assert result.is_valid is False, f"Query should be invalid: {query}"
            assert len(result.errors) > 0

    def test_should_detect_sql_injection_attempts(self):
        """Should detect SQL injection attempts."""
        from src.agent.validators.enhanced_sql_validator import EnhancedSQLValidator

        # Arrange
        validator = EnhancedSQLValidator()
        injection_queries = [
            "SELECT * FROM users; DROP TABLE users; --",
            "SELECT * FROM users WHERE name = ''; DELETE FROM users; --'",
            "SELECT * FROM users UNION SELECT * FROM passwords",
        ]

        # Act & Assert
        for query in injection_queries:
            result = validator.validate_input(query)
            assert result.is_valid is False, f"Injection should be detected: {query}"

    def test_should_check_query_complexity(self):
        """Should check query complexity and provide warnings."""
        from src.agent.validators.enhanced_sql_validator import EnhancedSQLValidator

        # Arrange
        validator = EnhancedSQLValidator(max_joins=2, max_subqueries=1)

        # Test query with too many joins
        complex_join_query = """
        SELECT * FROM users u1
        JOIN orders o1 ON u1.id = o1.user_id
        JOIN orders o2 ON u1.id = o2.user_id
        JOIN orders o3 ON u1.id = o3.user_id
        """

        result = validator.validate_input(complex_join_query)
        assert result.is_valid is False
        assert any("join" in error.lower() for error in result.errors)

        # Test query with too many subqueries
        complex_subquery = """
        SELECT * FROM users WHERE id IN (
            SELECT user_id FROM orders WHERE total > (
                SELECT AVG(total) FROM orders WHERE date > (
                    SELECT MAX(date) FROM orders
                )
            )
        )
        """

        result = validator.validate_input(complex_subquery)
        assert result.is_valid is False
        assert any("subquer" in error.lower() for error in result.errors)

    def test_should_provide_performance_warnings(self):
        """Should provide performance warnings for moderately complex queries."""
        from src.agent.validators.enhanced_sql_validator import EnhancedSQLValidator

        # Arrange - set lower thresholds to trigger warnings
        validator = EnhancedSQLValidator(max_joins=8, max_subqueries=4)

        # Query with moderate complexity should generate warnings (6 JOINs > max_joins//2 = 4)
        moderate_query = """
        SELECT * FROM users u1
        JOIN orders o1 ON u1.id = o1.user_id
        JOIN orders o2 ON u1.id = o2.user_id
        JOIN orders o3 ON u1.id = o3.user_id
        JOIN orders o4 ON u1.id = o4.user_id
        JOIN orders o5 ON u1.id = o5.user_id
        JOIN orders o6 ON u1.id = o6.user_id
        """

        result = validator.validate_input(moderate_query)
        assert result.is_valid is True  # Should still be valid
        assert len(result.warnings) > 0  # But should have warnings
        assert any("performance" in warning.lower() for warning in result.warnings)

    def test_should_sanitize_sql_input(self):
        """Should sanitize SQL input when enabled."""
        from src.agent.validators.enhanced_sql_validator import EnhancedSQLValidator

        # Arrange
        validator = EnhancedSQLValidator(enable_sanitization=True)
        query_with_quotes = "SELECT * FROM users WHERE name = 'O'Brien'"

        # Act
        result = validator.validate_input(query_with_quotes)

        # Assert
        assert result.is_valid is True
        assert "''" in result.sanitized_input  # Single quotes should be escaped

    def test_should_work_without_sanitization(self):
        """Should work without sanitization when disabled."""
        from src.agent.validators.enhanced_sql_validator import EnhancedSQLValidator

        # Arrange
        validator = EnhancedSQLValidator(enable_sanitization=False)
        query = "SELECT * FROM users WHERE name = 'test'"

        # Act
        result = validator.validate_input(query)

        # Assert
        assert result.is_valid is True
        assert result.sanitized_input == query  # Should be unchanged

    def test_should_work_without_complexity_check(self):
        """Should work without complexity checking when disabled."""
        from src.agent.validators.enhanced_sql_validator import EnhancedSQLValidator

        # Arrange
        validator = EnhancedSQLValidator(enable_complexity_check=False)

        # Complex query should pass without warnings
        complex_query = """
        SELECT * FROM users u1
        JOIN orders o1 ON u1.id = o1.user_id
        JOIN orders o2 ON u1.id = o2.user_id
        JOIN orders o3 ON u1.id = o3.user_id
        JOIN orders o4 ON u1.id = o4.user_id
        JOIN orders o5 ON u1.id = o5.user_id
        JOIN orders o6 ON u1.id = o6.user_id
        """

        result = validator.validate_input(complex_query)
        assert result.is_valid is True
        # Should not have complexity warnings
        assert not any("join" in warning.lower() for warning in result.warnings)

    def test_should_create_from_config(self):
        """Should create validator from configuration."""
        from src.agent.validators.enhanced_sql_validator import EnhancedSQLValidator

        # Arrange
        config = ValidatorConfig(
            enable_sql_complexity_check=True,
            sql_max_joins=3,
            sql_max_subqueries=2,
            enable_sql_injection_sanitization=True,
        )

        # Act
        validator = EnhancedSQLValidator.from_config(config)

        # Assert
        assert validator.enable_complexity_check is True
        assert validator.enable_sanitization is True

        # Test that configuration is applied
        complex_query = """
        SELECT * FROM users u1
        JOIN orders o1 ON u1.id = o1.user_id
        JOIN orders o2 ON u1.id = o2.user_id
        JOIN orders o3 ON u1.id = o3.user_id
        JOIN orders o4 ON u1.id = o4.user_id
        """

        result = validator.validate_input(complex_query)
        assert result.is_valid is False  # Should fail due to max_joins=3

    def test_should_provide_validate_and_raise_method(self):
        """Should provide validate_and_raise method."""
        from src.agent.validators.enhanced_sql_validator import EnhancedSQLValidator

        # Arrange
        validator = EnhancedSQLValidator()

        # Act & Assert - valid query should not raise
        sanitized = validator.validate_and_raise("SELECT * FROM users")
        assert sanitized == "SELECT * FROM users"

        # Act & Assert - invalid query should raise
        with pytest.raises(InputValidationException):
            validator.validate_and_raise("DROP TABLE users")


class TestBackwardCompatibleSQLValidator:
    """Test suite for backward compatible SQL validator."""

    def test_should_maintain_original_interface(self):
        """Should maintain the original SQLValidator interface."""
        from src.agent.validators.enhanced_sql_validator import (
            BackwardCompatibleSQLValidator,
        )

        # Arrange
        validator = BackwardCompatibleSQLValidator()

        # Act & Assert - valid query should return True
        result = validator.validate("SELECT * FROM users")
        assert result is True

        # Act & Assert - invalid query should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            validator.validate("DROP TABLE users")

        assert "DROP operations are not allowed" in str(exc_info.value)

    def test_should_work_with_existing_code(self):
        """Should work as a drop-in replacement for existing SQLValidator."""
        from src.agent.validators.enhanced_sql_validator import (
            BackwardCompatibleSQLValidator,
        )

        # Arrange
        validator = BackwardCompatibleSQLValidator()

        # Test all the same cases as the original validator
        valid_queries = [
            "SELECT * FROM users;",
            "SELECT id, name FROM users WHERE active = true;",
            "SELECT COUNT(*) FROM orders;",
        ]

        for query in valid_queries:
            result = validator.validate(query)
            assert result is True

        # Test invalid queries
        invalid_queries = [
            "DROP TABLE users;",
            "DELETE FROM users;",
            "INSERT INTO users VALUES (1, 'test');",
        ]

        for query in invalid_queries:
            with pytest.raises(ValueError):
                validator.validate(query)

    def test_should_handle_none_and_empty_queries(self):
        """Should handle None and empty queries like original validator."""
        from src.agent.validators.enhanced_sql_validator import (
            BackwardCompatibleSQLValidator,
        )

        # Arrange
        validator = BackwardCompatibleSQLValidator()

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            validator.validate(None)
        assert "cannot be None" in str(exc_info.value) or "cannot be empty" in str(
            exc_info.value
        )

        with pytest.raises(ValueError) as exc_info:
            validator.validate("")
        assert "cannot be empty" in str(exc_info.value)
