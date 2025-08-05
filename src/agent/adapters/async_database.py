"""
Async database adapter for the agentic AI framework.

This module provides async versions of database operations using asyncpg for PostgreSQL
with connection pooling support. It maintains compatibility with the existing synchronous
database adapter interface while providing async/await functionality.

Features:
- Connection pooling with configurable min/max connections
- Async context manager support
- Same interface as synchronous adapter for drop-in replacement
- Proper exception handling with custom exception classes
- Sensitive data filtering in exception context
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import asyncpg
import pandas as pd
from loguru import logger

from src.agent.exceptions import (
    DatabaseConnectionException,
    DatabaseQueryException,
    DatabaseTransactionException,
)
from src.agent.utils.retry import with_async_database_retry


class AsyncAbstractDatabase(ABC):
    """
    AsyncAbstractDatabase is an abstract base class for all async database adapters.

    This class defines the async interface that all database adapters must implement,
    including async context manager support for automatic connection management.

    Methods:
        - async execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]
        - async connect(self) -> None: Connect to the database and setup connection pool
        - async disconnect(self) -> None: Disconnect from the database and cleanup pool
        - async get_schema(self) -> Dict[str, Any]: Get database schema information
        - async insert_data(self, table_name: str, data: Dict[str, Any]) -> bool
        - async insert_batch(self, table_name: str, data_list: List[Dict[str, Any]]) -> bool
    """

    def __init__(self):
        pass

    async def __aenter__(self):
        """Enter the async context manager, connect to the database."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager, disconnect from the database."""
        await self.disconnect()

    @abstractmethod
    async def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Execute a SQL query and return results as DataFrame."""
        raise NotImplementedError

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the database and setup connection pool."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the database and cleanup connection pool."""
        pass

    @abstractmethod
    async def get_schema(self) -> Dict[str, Any]:
        """Get database schema information."""
        raise NotImplementedError

    @abstractmethod
    async def insert_data(self, table_name: str, data: Dict[str, Any]) -> bool:
        """Insert a single row of data into a table."""
        raise NotImplementedError

    @abstractmethod
    async def insert_batch(
        self, table_name: str, data_list: List[Dict[str, Any]]
    ) -> bool:
        """Insert multiple rows of data into a table in a single transaction."""
        raise NotImplementedError


class AsyncDatabaseAdapter(AsyncAbstractDatabase):
    """
    AsyncDatabaseAdapter implements the AsyncAbstractDatabase interface using asyncpg.

    This adapter provides async database operations with connection pooling for PostgreSQL.
    It maintains compatibility with the synchronous BaseDatabaseAdapter interface while
    providing async/await functionality.

    Features:
    - Connection pooling with configurable parameters
    - Transaction support for batch operations
    - Proper exception handling and context preservation
    - Sensitive data filtering for secure logging
    - DataFrame results compatible with pandas analysis

    Args:
        kwargs: Configuration dictionary containing:
            - connection_string: PostgreSQL connection string
            - db_type: Database type (defaults to "postgres")
            - min_connections: Minimum connections in pool (defaults to 1)
            - max_connections: Maximum connections in pool (defaults to 10)
            - connection_timeout: Connection timeout in seconds (defaults to 30)
    """

    def __init__(self, kwargs: Dict[str, Any]):
        """
        Initialize the AsyncDatabaseAdapter.

        Args:
            kwargs: Dict[str, Any]: The configuration parameters.
        """
        super().__init__()
        self.kwargs = kwargs
        self.connection_string = kwargs.get("connection_string")
        self.db_type = kwargs.get("db_type", "postgres")
        self.min_connections = kwargs.get("min_connections", 1)
        self.max_connections = kwargs.get("max_connections", 10)
        self.connection_timeout = kwargs.get("connection_timeout", 30)
        self.pool: Optional[asyncpg.Pool] = None

    @with_async_database_retry(max_retries=3, initial_delay=1.0)
    async def connect(self) -> None:
        """
        Connect to the database and create connection pool.

        Creates an asyncpg connection pool with the configured parameters.

        Raises:
            DatabaseConnectionException: If connection pool creation fails.
        """
        if self.pool is None:
            try:
                self.pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=self.min_connections,
                    max_size=self.max_connections,
                    timeout=self.connection_timeout,
                )
                logger.info(
                    f"Connected to {self.db_type} database with pool "
                    f"(min: {self.min_connections}, max: {self.max_connections})"
                )
            except Exception as e:
                logger.error(f"Error creating connection pool: {e}")
                context = {
                    "connection_string": self.connection_string,
                    "db_type": self.db_type,
                    "min_connections": self.min_connections,
                    "max_connections": self.max_connections,
                    "connection_timeout": self.connection_timeout,
                    "operation": "create_pool",
                }
                raise DatabaseConnectionException(
                    f"Failed to create database connection pool: {e}",
                    context=context,
                    original_exception=e,
                )

    async def disconnect(self) -> None:
        """
        Disconnect from the database and cleanup connection pool.

        Properly closes the connection pool and waits for all connections to close.
        """
        if self.pool:
            await self.pool.close()
            await self.pool.wait_closed()
            self.pool = None
            logger.info("Disconnected from database and closed connection pool")

    @with_async_database_retry(max_retries=3, initial_delay=1.0)
    async def execute_query(
        self,
        sql_statement: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute a SELECT SQL query and return the result as a Pandas DataFrame.

        This method acquires a connection from the pool, executes the query,
        and converts the results to a DataFrame for compatibility with the
        synchronous adapter interface.

        Args:
            sql_statement: SQL query to execute
            params: Optional parameters for the query

        Returns:
            Dict containing "data" key with pandas DataFrame of results

        Raises:
            DatabaseConnectionException: If connection pool is not available.
            DatabaseQueryException: If query execution fails.
        """
        if not self.pool:
            logger.error("Connection pool not available. Cannot execute query.")
            context = {
                "pool_available": False,
                "operation": "execute_query",
                "query": sql_statement,
                "parameters": params,
            }
            raise DatabaseConnectionException(
                "Database connection pool not available for query execution",
                context=context,
            )

        try:
            async with self.pool.acquire() as connection:
                if params:
                    # Convert dict params to positional args for asyncpg
                    param_values = list(params.values())
                    records = await connection.fetch(sql_statement, *param_values)
                else:
                    records = await connection.fetch(sql_statement)

                # Convert asyncpg records to DataFrame
                if records:
                    # Convert Record objects to list of dicts
                    data = [dict(record) for record in records]
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame()

                return {"data": df}

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            context = {
                "query": sql_statement,
                "parameters": params,
                "db_type": self.db_type,
                "operation": "execute_query",
            }
            raise DatabaseQueryException(
                f"Failed to execute database query: {e}",
                context=context,
                original_exception=e,
            )

    @with_async_database_retry(max_retries=3, initial_delay=1.0)
    async def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema of the database.

        Retrieves table and column information from the PostgreSQL information schema.

        Returns:
            Dict containing schema information with tables and columns

        Raises:
            DatabaseConnectionException: If connection pool is not available.
            DatabaseQueryException: If schema reflection fails.
        """
        if not self.pool:
            logger.error("Connection pool not available. Cannot get schema.")
            context = {
                "pool_available": False,
                "operation": "get_schema",
            }
            raise DatabaseConnectionException(
                "Database connection pool not available for schema reflection",
                context=context,
            )

        try:
            async with self.pool.acquire() as connection:
                # Get table information
                tables_query = """
                    SELECT table_name, table_schema
                    FROM information_schema.tables
                    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                    ORDER BY table_schema, table_name
                """
                table_records = await connection.fetch(tables_query)
                tables = [dict(record) for record in table_records]

                # Get column information
                columns_query = """
                    SELECT table_name, column_name, data_type, is_nullable,
                           column_default, ordinal_position, table_schema
                    FROM information_schema.columns
                    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                    ORDER BY table_schema, table_name, ordinal_position
                """
                column_records = await connection.fetch(columns_query)
                columns = [dict(record) for record in column_records]

                return {"tables": tables, "columns": columns}

        except Exception as e:
            logger.error(f"Error reflecting schema: {e}")
            context = {
                "operation": "schema_reflection",
                "db_type": self.db_type,
            }
            raise DatabaseQueryException(
                f"Failed to reflect database schema: {e}",
                context=context,
                original_exception=e,
            )

    async def insert_data(self, table_name: str, data: Dict[str, Any]) -> bool:
        """
        Insert a single row of data into a table.

        Args:
            table_name: Name of the table to insert into
            data: Dictionary of column names and values

        Returns:
            bool: True if successful

        Raises:
            DatabaseConnectionException: If connection pool is not available.
            DatabaseTransactionException: If insertion fails.
        """
        return await self.insert_batch(table_name, [data])

    @with_async_database_retry(max_retries=3, initial_delay=1.0)
    async def insert_batch(
        self, table_name: str, data_list: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert multiple rows of data into a table in a single transaction.

        Args:
            table_name: Name of the table to insert into
            data_list: List of dictionaries, each containing column names and values

        Returns:
            bool: True if successful

        Raises:
            DatabaseConnectionException: If connection pool is not available.
            DatabaseTransactionException: If transaction fails.
        """
        if not self.pool:
            logger.error("Connection pool not available. Cannot insert data.")
            context = {
                "pool_available": False,
                "operation": "insert_batch",
                "table_name": table_name,
                "row_count": len(data_list),
            }
            raise DatabaseConnectionException(
                "Database connection pool not available for data insertion",
                context=context,
            )

        if not data_list:
            logger.warning("No data to insert.")
            return True

        try:
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    for data in data_list:
                        # Build INSERT query for each row
                        columns = ", ".join([f'"{col}"' for col in data.keys()])
                        placeholders = ", ".join(
                            [f"${i + 1}" for i in range(len(data))]
                        )
                        query = f'INSERT INTO "{table_name}" ({columns}) VALUES ({placeholders})'

                        # Execute with values as positional arguments
                        await connection.execute(query, *data.values())

            count = len(data_list)
            if count == 1:
                logger.info(f"Successfully inserted 1 row into {table_name}")
            else:
                logger.info(f"Successfully inserted {count} rows into {table_name}")
            return True

        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            context = {
                "table_name": table_name,
                "row_count": len(data_list),
                "operation": "batch_insert",
                "db_type": self.db_type,
            }
            raise DatabaseTransactionException(
                f"Failed to insert data into {table_name}: {e}",
                context=context,
                original_exception=e,
            )
