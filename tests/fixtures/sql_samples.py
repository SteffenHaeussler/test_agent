"""SQL samples for testing SQL validator."""

# Valid SELECT queries
VALID_SELECT_QUERIES = [
    "SELECT * FROM users;",
    "SELECT id, name, email FROM customers WHERE active = 1;",
    "SELECT COUNT(*) as total FROM orders WHERE created_at > '2024-01-01';",
    "SELECT u.name, p.title FROM users u JOIN posts p ON u.id = p.user_id;",
    "SELECT * FROM products ORDER BY price DESC LIMIT 10;",
    "SELECT category, SUM(price) as total FROM products GROUP BY category;",
    "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE total > 100);",
]

# Valid CTE queries
VALID_CTE_QUERIES = [
    """
    WITH active_users AS (
        SELECT * FROM users WHERE status = 'active'
    )
    SELECT * FROM active_users;
    """,
    """
    WITH RECURSIVE subordinates AS (
        SELECT id, name, manager_id FROM employees WHERE id = 1
        UNION ALL
        SELECT e.id, e.name, e.manager_id
        FROM employees e
        JOIN subordinates s ON e.manager_id = s.id
    )
    SELECT * FROM subordinates;
    """,
]

# Forbidden DDL operations
FORBIDDEN_DDL_QUERIES = [
    "DROP TABLE users;",
    "DROP DATABASE production;",
    "CREATE TABLE new_table (id INT);",
    "ALTER TABLE users ADD COLUMN password VARCHAR(255);",
    "TRUNCATE TABLE logs;",
    "CREATE INDEX idx_users_email ON users(email);",
]

# Forbidden DML operations
FORBIDDEN_DML_QUERIES = [
    "DELETE FROM users WHERE id = 1;",
    "INSERT INTO users (name, email) VALUES ('test', 'test@example.com');",
    "UPDATE users SET admin = true WHERE email = 'hacker@evil.com';",
    "INSERT INTO passwords SELECT * FROM user_credentials;",
]

# SQL injection attempts
SQL_INJECTION_ATTEMPTS = [
    "SELECT * FROM users; DROP TABLE users;--",
    "SELECT * FROM users WHERE id = 1 OR 1=1;",
    "SELECT * FROM users WHERE name = ''; DELETE FROM users;--'",
    "SELECT * FROM users WHERE id = 1 UNION SELECT * FROM passwords;",
    "SELECT * FROM users WHERE name = 'admin'--' AND password = 'anything'",
]

# Complex queries that should be rejected
COMPLEX_FORBIDDEN_QUERIES = [
    """
    SELECT * FROM users
    WHERE id IN (
        DELETE FROM orders WHERE status = 'pending' RETURNING user_id
    );
    """,
    """
    WITH deleted_users AS (
        DROP TABLE user_sessions
    )
    SELECT * FROM users;
    """,
    """
    SELECT
        CASE
            WHEN 1=1 THEN (UPDATE users SET admin = true)
            ELSE id
        END as result
    FROM users;
    """,
]
