"""SQL validator to prevent SQL injection attacks."""

import re
from typing import Optional

from loguru import logger


class SQLValidator:
    """Validates SQL queries to prevent injection attacks."""

    # DDL (Data Definition Language) operations that modify schema
    FORBIDDEN_DDL_OPERATIONS = frozenset(
        ["CREATE", "ALTER", "DROP", "TRUNCATE", "RENAME"]
    )

    # DML (Data Manipulation Language) operations that modify data
    FORBIDDEN_DML_OPERATIONS = frozenset(
        ["INSERT", "UPDATE", "DELETE", "MERGE", "REPLACE"]
    )

    # DCL (Data Control Language) operations that modify permissions
    FORBIDDEN_DCL_OPERATIONS = frozenset(["GRANT", "REVOKE"])

    # Other potentially dangerous operations
    FORBIDDEN_OTHER_OPERATIONS = frozenset(["EXEC", "EXECUTE", "CALL"])

    # Combine all forbidden operations
    FORBIDDEN_OPERATIONS = (
        FORBIDDEN_DDL_OPERATIONS
        | FORBIDDEN_DML_OPERATIONS
        | FORBIDDEN_DCL_OPERATIONS
        | FORBIDDEN_OTHER_OPERATIONS
    )

    # Allowed operations (whitelist approach)
    ALLOWED_OPERATIONS = frozenset(["SELECT", "WITH"])

    def validate(self, sql_query: Optional[str]) -> bool:
        """
        Validate SQL query for safety.

        Args:
            sql_query: The SQL query to validate

        Returns:
            True if query is safe

        Raises:
            ValueError: If query contains forbidden operations
        """
        # Check for empty or None queries
        if not sql_query or sql_query.strip() == "":
            raise ValueError("Empty SQL query")

        # Normalize query for checking
        normalized_query = sql_query.upper().strip()

        # List of forbidden operations
        forbidden_operations = self.FORBIDDEN_OPERATIONS

        # Check for forbidden operations
        for operation in forbidden_operations:
            # Use word boundaries to match whole words
            pattern = rf"\b{operation}\b"
            if re.search(pattern, normalized_query):
                logger.warning(
                    "SQL validation failed: {operation} operation detected in query",
                    operation=operation,
                    query_snippet=sql_query[:100],
                )
                raise ValueError(f"{operation} operations are not allowed")

        # Check for multiple statements (semicolon not in string)
        # Simple check - a more robust solution would parse the SQL
        if self._contains_multiple_statements(sql_query):
            logger.warning(
                "SQL validation failed: Multiple statements detected",
                query_snippet=sql_query[:100],
            )
            raise ValueError("Multiple SQL statements are not allowed")

        # Check for UNION (often used in SQL injection)
        # Allow UNION ALL but be suspicious of plain UNION
        if re.search(r"\bUNION\b(?!\s+ALL)", normalized_query):
            logger.warning(
                "SQL validation failed: UNION operation detected",
                query_snippet=sql_query[:100],
            )
            raise ValueError("UNION operations are not allowed")

        logger.debug("SQL query validated successfully")
        return True

    def _contains_multiple_statements(self, sql_query: str) -> bool:
        """
        Check if SQL contains multiple statements.

        This is a simple check that looks for semicolons outside of strings.
        A more robust implementation would use a proper SQL parser.
        """
        # Remove string literals to avoid false positives
        # This is a simplified approach
        cleaned = sql_query

        # Remove single-quoted strings
        cleaned = re.sub(r"'[^']*'", "", cleaned)

        # Remove double-quoted strings
        cleaned = re.sub(r'"[^"]*"', "", cleaned)

        # Remove comments
        cleaned = re.sub(r"--.*$", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)

        # Check for multiple semicolons or semicolon not at the end
        # Count semicolons
        semicolon_count = cleaned.count(";")

        if semicolon_count == 0:
            return False
        elif semicolon_count == 1:
            # One semicolon is OK if it's at the end
            return not cleaned.strip().endswith(";")
        else:
            # Multiple semicolons indicate multiple statements
            return True
