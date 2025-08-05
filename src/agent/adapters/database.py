from abc import ABC
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import MetaData, create_engine, text

from src.agent.exceptions import (
    DatabaseConnectionException,
    DatabaseQueryException,
    DatabaseTransactionException,
)
from src.agent.utils.retry import with_database_retry


class AbstractDatabase(ABC):
    """
    AbstractDatabase is an abstract base class for all database adapters.

    Methods:
        - execute_query(self, query: str) -> Any: Execute a SQL query.
        - connect(self) -> None: Connect to the database.
        - disconnect(self) -> None: Disconnect from the database.
    """

    def __init__(self):
        pass

    def __enter__(self):
        """Enter the context manager, connect to the database."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, disconnect from the database."""
        self.disconnect()

    def execute_query(self, query: str) -> Dict[str, pd.DataFrame]:
        """Execute a SQL query."""
        raise NotImplementedError

    def connect(self) -> None:
        """Connect to the database."""
        pass

    def disconnect(self) -> None:
        """Disconnect from the database."""
        pass


class BaseDatabaseAdapter(AbstractDatabase):
    """
    BaseDatabaseAdapter is a class that implements the AbstractDatabase interface.

    Methods:
        - execute_query(self, query: str) -> Any: Execute a SQL query.
        - connect(self) -> None: Connect to the database.
        - disconnect(self) -> None: Disconnect from the database.
    """

    def __init__(self, kwargs: Dict[str, Any]):
        """
        Initialize the BaseDatabaseAdapter.

        Args:
            kwargs: Dict[str, Any]: The configuration parameters.
        """
        super().__init__()
        self.kwargs = kwargs
        self.schema_info = None
        self.connection_string = kwargs.get("connection_string")
        self.db_type = kwargs.get("db_type", "postgres")
        self.engine = None

    @with_database_retry(max_retries=3, initial_delay=1.0)
    def _get_connection(self) -> Any:
        """
        Get database connection based on database type.

        Returns:
            connection: Any: The database connection object.

        Raises:
            DatabaseConnectionException: If connection creation fails.
        """
        try:
            engine = create_engine(self.connection_string)
            logger.info("SQLAlchemy engine created successfully.")
            return engine
        except Exception as e:
            logger.error(f"Error creating SQLAlchemy engine: {e}")
            context = {
                "connection_string": self.connection_string,
                "db_type": self.db_type,
                "operation": "create_engine",
            }
            raise DatabaseConnectionException(
                f"Failed to create database connection: {e}",
                context=context,
                original_exception=e,
            )

    def connect(self) -> None:
        """
        Connect to the database.
        """
        if self.engine is None:
            self.engine = self._get_connection()
            if self.engine:
                logger.info(f"Connected to {self.db_type} database")
            else:
                logger.error("Failed to connect to database")

    def disconnect(self) -> None:
        """
        Disconnect from the database.
        """
        if self.engine:
            self.engine.dispose()
            self.engine = None
            logger.info("Disconnected from database")

    @with_database_retry(max_retries=3, initial_delay=1.0)
    def execute_query(
        self,
        sql_statement: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Executes a SELECT SQL query and returns the result as a Pandas DataFrame.
        This is the officially supported way.

        Raises:
            DatabaseConnectionException: If engine is not available.
            DatabaseQueryException: If query execution fails.
        """
        if not self.engine:
            logger.error("Engine not available. Cannot execute query.")
            context = {
                "engine_available": False,
                "operation": "execute_query",
                "query": sql_statement,
                "parameters": params,
            }
            raise DatabaseConnectionException(
                "Database engine not available for query execution",
                context=context,
            )

        try:
            # Pandas reads directly from the SQLAlchemy engine
            df = pd.read_sql_query(
                sql=text(sql_statement), con=self.engine, params=params
            )
            return {"data": df}
        except Exception as e:
            logger.error(f"Error executing query to DataFrame: {e}")
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

    @with_database_retry(max_retries=3, initial_delay=1.0)
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema of the database.

        Raises:
            DatabaseConnectionException: If engine is not available.
            DatabaseQueryException: If schema reflection fails.
        """
        if not self.engine:
            logger.error("Engine not available. Cannot execute query.")
            context = {
                "engine_available": False,
                "operation": "get_schema",
            }
            raise DatabaseConnectionException(
                "Database engine not available for schema reflection",
                context=context,
            )

        metadata = MetaData()
        try:
            metadata.reflect(bind=self.engine)
            return metadata
        except Exception as e:
            logger.error(f"Error reflecting metadata: {e}")
            context = {
                "operation": "schema_reflection",
                "db_type": self.db_type,
            }
            raise DatabaseQueryException(
                f"Failed to reflect database schema: {e}",
                context=context,
                original_exception=e,
            )

    def insert_data(self, table_name: str, data: Dict[str, Any]) -> bool:
        """
        Insert a single row of data into a table.

        Args:
            table_name: Name of the table to insert into
            data: Dictionary of column names and values

        Returns:
            bool: True if successful, False otherwise
        """
        return self.insert_batch(table_name, [data])

    @with_database_retry(max_retries=3, initial_delay=1.0)
    def insert_batch(self, table_name: str, data_list: List[Dict[str, Any]]) -> bool:
        """
        Insert multiple rows of data into a table in a single transaction.

        Args:
            table_name: Name of the table to insert into
            data_list: List of dictionaries, each containing column names and values

        Returns:
            bool: True if successful

        Raises:
            DatabaseConnectionException: If engine is not available.
            DatabaseTransactionException: If transaction fails.
        """
        if not self.engine:
            logger.error("Engine not available. Cannot insert data.")
            context = {
                "engine_available": False,
                "operation": "insert_batch",
                "table_name": table_name,
                "row_count": len(data_list),
            }
            raise DatabaseConnectionException(
                "Database engine not available for data insertion",
                context=context,
            )

        if not data_list:
            logger.warning("No data to insert.")
            return True

        try:
            with self.engine.begin() as conn:
                for data in data_list:
                    # Build INSERT query for each row to handle different columns
                    columns = ", ".join([f'"{col}"' for col in data.keys()])
                    placeholders = ", ".join([f":{key}" for key in data.keys()])
                    query = f'INSERT INTO "{table_name}" ({columns}) VALUES ({placeholders})'
                    conn.execute(text(query), data)

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
