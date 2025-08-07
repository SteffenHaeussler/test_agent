import asyncio
import re
from abc import ABC
from typing import Any, AsyncIterator, Dict, List, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import MetaData, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.exc import SQLAlchemyError

from src.agent.exceptions import (
    DatabaseConnectionException,
    DatabaseQueryException,
    DatabaseTransactionException,
)
from src.agent.utils.constants import Database
from src.agent.adapters.cache import CacheManager, CacheStrategy, get_ttl_for_strategy


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
        """Enter the synchronous context manager, connect to the database."""
        # For backward compatibility, provide a synchronous interface
        import asyncio

        if asyncio.iscoroutinefunction(self.connect):
            # If connect is async, run it in a new event loop for sync compatibility
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If there's already a running loop, we need to use a thread
                    import concurrent.futures

                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            new_loop.run_until_complete(self.connect())
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        executor.submit(run_in_thread).result()
                else:
                    loop.run_until_complete(self.connect())
            except RuntimeError as e:
                # Check if it's "no current event loop" error
                if "There is no current event loop" in str(e):
                    # Create new event loop if none exists
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.connect())
                    finally:
                        # Don't close the loop, keep it for later use
                        pass
                else:
                    raise
        else:
            # If connect is not async, call it directly
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the synchronous context manager, disconnect from the database."""
        # For backward compatibility, provide synchronous disconnect
        import asyncio

        if asyncio.iscoroutinefunction(self.disconnect):
            # If disconnect is async, run it in the event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If there's already a running loop, we need to use a thread
                    import concurrent.futures

                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            new_loop.run_until_complete(self.disconnect())
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        executor.submit(run_in_thread).result()
                else:
                    loop.run_until_complete(self.disconnect())
            except RuntimeError as e:
                # Check if it's "no current event loop" error
                if "There is no current event loop" in str(e):
                    # Create new event loop if none exists
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.disconnect())
                    finally:
                        loop.close()
                else:
                    raise
        else:
            # If disconnect is not async, call it directly
            self.disconnect()

    async def __aenter__(self):
        """Enter the async context manager, connect to the database."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager, disconnect from the database."""
        await self.disconnect()

    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Execute a SQL query with optional pagination."""
        raise NotImplementedError

    async def execute_query_streaming(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
    ) -> AsyncIterator[pd.DataFrame]:
        """Execute a SQL query with streaming results."""
        raise NotImplementedError

    async def connect(self) -> None:
        """Connect to the database."""
        pass

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        pass

    async def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        raise NotImplementedError


class BaseDatabaseAdapter(AbstractDatabase):
    """
    BaseDatabaseAdapter is a class that implements the AbstractDatabase interface with async support.

    Features:
    - Async SQLAlchemy with connection pooling
    - Query pagination and streaming support
    - Connection health checks and retry logic
    - Exponential backoff for failed connections
    - Query timeout handling
    - Proper async context managers

    Methods:
        - execute_query(self, query: str) -> Any: Execute a SQL query with pagination.
        - execute_query_streaming(self, query: str) -> AsyncIterator: Execute streaming query.
        - connect(self) -> None: Connect to the database.
        - disconnect(self) -> None: Disconnect from the database.
        - health_check(self) -> bool: Check connection health.
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
        self.db_type = kwargs.get("db_type", Database.TYPE_POSTGRES)
        self.engine: Optional[AsyncEngine] = None
        self.session_maker = None

        # Connection pool settings
        self.pool_size = kwargs.get("pool_size", 20)
        self.max_overflow = kwargs.get("max_overflow", 30)
        self.pool_timeout = kwargs.get("pool_timeout", 30)

        # Query settings
        self.query_timeout = kwargs.get("query_timeout", 300)  # 5 minutes
        self.default_chunk_size = kwargs.get("chunk_size", 1000)

        # Retry settings
        self.max_retries = kwargs.get("max_retries", 3)
        self.base_delay = kwargs.get("base_delay", 1.0)
        self.max_delay = kwargs.get("max_delay", 60.0)

        # Cache configuration
        self.cache_manager: Optional[CacheManager] = kwargs.get("cache_manager")
        self.cache_enabled = kwargs.get("cache_enabled", True)

    async def _create_async_engine(self) -> AsyncEngine:
        """
        Create async database engine with connection pooling.

        Returns:
            AsyncEngine: The async SQLAlchemy engine.

        Raises:
            DatabaseConnectionException: If engine creation fails.
        """
        try:
            # Convert PostgreSQL connection string to async format
            async_connection_string = self.connection_string
            if self.connection_string.startswith("postgresql://"):
                async_connection_string = self.connection_string.replace(
                    "postgresql://", "postgresql+asyncpg://", 1
                )
            elif self.connection_string.startswith("postgresql+psycopg2://"):
                async_connection_string = self.connection_string.replace(
                    "postgresql+psycopg2://", "postgresql+asyncpg://", 1
                )

            engine = create_async_engine(
                async_connection_string,
                # Don't specify poolclass for async engine - it will use NullPool or AsyncAdaptedQueuePool
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_pre_ping=True,  # Enable connection health checks
                pool_recycle=3600,  # Recycle connections after 1 hour
                echo=False,  # Set to True for SQL debugging
            )

            self.session_maker = async_sessionmaker(engine, expire_on_commit=False)

            logger.info(
                f"Async SQLAlchemy engine created successfully with pool_size={self.pool_size}, "
                f"max_overflow={self.max_overflow}, pool_timeout={self.pool_timeout}"
            )
            return engine
        except Exception as e:
            logger.error(f"Error creating async SQLAlchemy engine: {e}")
            context = {
                "connection_string": self.connection_string,
                "db_type": self.db_type,
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow,
                "pool_timeout": self.pool_timeout,
                "operation": "create_async_engine",
            }
            raise DatabaseConnectionException(
                f"Failed to create async database connection: {e}",
                context=context,
                original_exception=e,
            )

    def _run_async(self, coro):
        """
        Helper method to run async code in sync context with proper event loop handling.

        Args:
            coro: Coroutine to run

        Returns:
            Result of the coroutine execution
        """
        import asyncio
        import concurrent.futures

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If there's already a running loop, we need to use a thread
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(coro)
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(run_in_thread).result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    # Synchronous wrapper methods for backward compatibility
    def execute_query_sync(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Synchronous wrapper for execute_query_async."""
        return self._run_async(self.execute_query_async(query, params, limit, offset))

    def get_schema_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper for get_schema_async."""
        return self._run_async(self.get_schema_async())

    def connect_sync(self) -> None:
        """Synchronous wrapper for async connect."""
        return self._run_async(self.connect())

    def disconnect_sync(self) -> None:
        """Synchronous wrapper for async disconnect."""
        return self._run_async(self.disconnect())

    def insert_data_sync(self, table_name: str, data: Dict[str, Any]) -> bool:
        """Synchronous wrapper for async insert_data."""
        return self._run_async(self.insert_data(table_name, data))

    def insert_batch_sync(
        self, table_name: str, data_list: List[Dict[str, Any]]
    ) -> bool:
        """Synchronous wrapper for async insert_batch."""
        return self._run_async(self.insert_batch(table_name, data_list))

    def _get_connection(self) -> AsyncEngine:
        """
        Legacy synchronous method for backward compatibility.
        This should not be used in the async implementation.

        Raises:
            NotImplementedError: Always, as this method is deprecated.
        """
        raise NotImplementedError(
            "_get_connection() is deprecated. Use async connect() method instead."
        )

    # Provide backward-compatible synchronous methods with original names
    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Synchronous execute_query method for backward compatibility.
        For async usage, use execute_query_async.
        """
        return self.execute_query_sync(query, params, limit, offset)

    def get_schema(self) -> Dict[str, Any]:
        """
        Synchronous get_schema method for backward compatibility.
        For async usage, use get_schema_async.
        """
        return self.get_schema_sync()

    async def connect(self) -> None:
        """
        Connect to the database with retry logic and exponential backoff.

        Raises:
            DatabaseConnectionException: If connection fails after all retries.
        """
        if self.engine is None:
            retry_count = 0
            last_exception = None

            while retry_count <= self.max_retries:
                try:
                    self.engine = await self._create_async_engine()
                    await self._test_connection()
                    logger.info(f"Connected to {self.db_type} database")
                    return
                except Exception as e:
                    last_exception = e
                    retry_count += 1

                    if retry_count <= self.max_retries:
                        # Calculate exponential backoff delay
                        delay = min(
                            self.base_delay * (2 ** (retry_count - 1)), self.max_delay
                        )
                        logger.warning(
                            f"Database connection failed (attempt {retry_count}/{self.max_retries + 1}): {e}. "
                            f"Retrying in {delay:.1f} seconds..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Failed to connect to database after {self.max_retries + 1} attempts"
                        )

                        context = {
                            "connection_string": self.connection_string,
                            "db_type": self.db_type,
                            "retry_count": retry_count,
                            "max_retries": self.max_retries,
                            "operation": "connect",
                        }
                        raise DatabaseConnectionException(
                            f"Failed to connect to database after {retry_count} attempts: {last_exception}",
                            context=context,
                            original_exception=last_exception,
                        )

    async def disconnect(self) -> None:
        """
        Disconnect from the database and dispose of the engine.
        """
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.session_maker = None
            logger.info("Disconnected from database")

    async def execute_query_async(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute a SQL query with optional pagination and timeout handling (async version).

        Args:
            query: SQL query string
            params: Query parameters
            limit: Maximum number of rows to return
            offset: Number of rows to skip

        Returns:
            Dict containing the query result as a pandas DataFrame

        Raises:
            DatabaseConnectionException: If engine is not available.
            DatabaseQueryException: If query execution fails.
        """
        if not self.engine:
            logger.error("Engine not available. Cannot execute query.")
            context = {
                "engine_available": False,
                "operation": "execute_query",
                "query": query,
                "parameters": params,
                "limit": limit,
                "offset": offset,
            }
            raise DatabaseConnectionException(
                "Database engine not available for query execution",
                context=context,
            )

        try:
            # Add LIMIT and OFFSET to query if specified
            modified_query = query
            if limit is not None:
                modified_query += f" LIMIT {limit}"
            if offset is not None:
                modified_query += f" OFFSET {offset}"

            # Execute query with timeout using asyncio.wait_for
            async def query_task():
                async with self.session_maker() as session:
                    result = await session.execute(text(modified_query), params or {})
                    rows = result.fetchall()
                    columns = result.keys()
                    return rows, columns

            rows, columns = await asyncio.wait_for(
                query_task(), timeout=self.query_timeout
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                [dict(row._mapping) for row in rows], columns=list(columns)
            )

            logger.debug(f"Query executed successfully, returned {len(df)} rows")
            return {"data": df}

        except asyncio.TimeoutError as e:
            logger.error(
                f"Query timeout after {self.query_timeout} seconds: {query[:100]}..."
            )
            context = {
                "query": query,
                "parameters": params,
                "timeout_seconds": self.query_timeout,
                "db_type": self.db_type,
                "operation": "execute_query_timeout",
            }
            raise DatabaseQueryException(
                f"Query timed out after {self.query_timeout} seconds",
                context=context,
                original_exception=e,
            )
        except (SQLAlchemyError, Exception) as e:
            logger.error(f"Error executing query: {e}")
            context = {
                "query": query,
                "parameters": params,
                "limit": limit,
                "offset": offset,
                "db_type": self.db_type,
                "operation": "execute_query",
            }
            raise DatabaseQueryException(
                f"Failed to execute database query: {e}",
                context=context,
                original_exception=e,
            )

    async def execute_query_streaming(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
    ) -> AsyncIterator[pd.DataFrame]:
        """
        Execute a SQL query with streaming results for memory efficiency.

        Args:
            query: SQL query string
            params: Query parameters
            chunk_size: Number of rows per chunk

        Yields:
            pandas DataFrames containing chunks of the query result

        Raises:
            DatabaseConnectionException: If engine is not available.
            DatabaseQueryException: If query execution fails.
        """
        if not self.engine:
            logger.error("Engine not available. Cannot execute streaming query.")
            context = {
                "engine_available": False,
                "operation": "execute_query_streaming",
                "query": query,
                "parameters": params,
                "chunk_size": chunk_size,
            }
            raise DatabaseConnectionException(
                "Database engine not available for streaming query execution",
                context=context,
            )

        try:
            # Use asyncio.wait_for for timeout handling
            async def streaming_task():
                async with self.session_maker() as session:
                    result = await session.execute(text(query), params or {})
                    columns = list(result.keys())

                    chunk_rows = []
                    total_rows = 0
                    chunks = []

                    for row in result:
                        chunk_rows.append(dict(row._mapping))

                        if len(chunk_rows) >= chunk_size:
                            df_chunk = pd.DataFrame(chunk_rows, columns=columns)
                            total_rows += len(df_chunk)
                            chunks.append(df_chunk)
                            chunk_rows = []

                    # Add remaining rows
                    if chunk_rows:
                        df_chunk = pd.DataFrame(chunk_rows, columns=columns)
                        total_rows += len(df_chunk)
                        chunks.append(df_chunk)

                    return chunks, total_rows

            chunks, total_rows = await asyncio.wait_for(
                streaming_task(), timeout=self.query_timeout
            )

            # Yield all chunks
            for chunk in chunks:
                yield chunk

            logger.debug(
                f"Streaming query completed, processed {total_rows} rows in chunks of {chunk_size}"
            )

        except asyncio.TimeoutError as e:
            logger.error(
                f"Streaming query timeout after {self.query_timeout} seconds: {query[:100]}..."
            )
            context = {
                "query": query,
                "parameters": params,
                "timeout_seconds": self.query_timeout,
                "chunk_size": chunk_size,
                "db_type": self.db_type,
                "operation": "execute_query_streaming_timeout",
            }
            raise DatabaseQueryException(
                f"Streaming query timed out after {self.query_timeout} seconds",
                context=context,
                original_exception=e,
            )
        except (SQLAlchemyError, Exception) as e:
            logger.error(f"Error executing streaming query: {e}")
            context = {
                "query": query,
                "parameters": params,
                "chunk_size": chunk_size,
                "db_type": self.db_type,
                "operation": "execute_query_streaming",
            }
            raise DatabaseQueryException(
                f"Failed to execute streaming database query: {e}",
                context=context,
                original_exception=e,
            )

    async def health_check(self) -> bool:
        """
        Check if the database connection is healthy.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        if not self.engine:
            logger.warning("No database engine available for health check")
            return False

        try:
            # Use asyncio.wait_for with 10-second timeout for health check
            async def health_check_task():
                async with self.session_maker() as session:
                    await session.execute(text("SELECT 1"))

            await asyncio.wait_for(health_check_task(), timeout=10)

            logger.debug("Database health check passed")
            return True

        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False

    async def get_schema_async(self) -> Dict[str, Any]:
        """
        Get the schema of the database (async version).

        Raises:
            DatabaseConnectionException: If engine is not available.
            DatabaseQueryException: If schema reflection fails.
        """
        if not self.engine:
            logger.error("Engine not available. Cannot get schema.")
            context = {
                "engine_available": False,
                "operation": "get_schema",
            }
            raise DatabaseConnectionException(
                "Database engine not available for schema reflection",
                context=context,
            )

        try:
            metadata = MetaData()
            async with self.session_maker() as session:
                # Run metadata reflection in an async context
                await session.run_sync(
                    lambda sync_conn: metadata.reflect(bind=sync_conn)
                )
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

    async def insert_data(self, table_name: str, data: Dict[str, Any]) -> bool:
        """
        Insert a single row of data into a table.

        Args:
            table_name: Name of the table to insert into
            data: Dictionary of column names and values

        Returns:
            bool: True if successful
        """
        return await self.insert_batch(table_name, [data])

    async def insert_batch(
        self, table_name: str, data_list: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert multiple rows of data into a table in a single async transaction.

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
            async with self.session_maker() as session:
                async with session.begin():
                    for data in data_list:
                        # Build INSERT query for each row to handle different columns
                        columns = ", ".join([f'"{col}"' for col in data.keys()])
                        placeholders = ", ".join([f":{key}" for key in data.keys()])
                        query = f'INSERT INTO "{table_name}" ({columns}) VALUES ({placeholders})'
                        await session.execute(text(query), data)

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

    async def _test_connection(self) -> None:
        """
        Test the database connection by executing a simple query.

        Raises:
            Exception: If the connection test fails.
        """
        async with self.session_maker() as session:
            await session.execute(text("SELECT 1"))
            logger.debug("Database connection test successful")

    # Cache-enabled methods

    async def execute_cached_query_async(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute a database query with caching support.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Dict containing the query result as a pandas DataFrame
        """
        # Check if caching should be used
        if not self._should_use_cache():
            return await self.execute_query_async(query, params)

        # Check for write operations that should not be cached
        if self._is_write_query(query):
            result = await self.execute_query_async(query, params)
            # Invalidate related cache entries for write operations
            await self._invalidate_affected_cache(query)
            return result

        # Generate cache key
        cache_key = self._generate_cache_key(query, params)

        # Try to get from cache
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Database cache hit for key: {cache_key}")
            # Reconstruct DataFrame from cached data
            if "data" in cached_result:
                df = pd.DataFrame(cached_result["data"])
                return {"data": df}
            return cached_result

        # Cache miss - execute query and cache result
        logger.debug(f"Database cache miss for key: {cache_key}")
        result = await self.execute_query_async(query, params)

        # Cache the result
        await self._cache_query_result(cache_key, result)

        return result

    async def execute_write_query_async(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute a write query and handle cache invalidation.

        Args:
            query: SQL write query (INSERT, UPDATE, DELETE)
            params: Query parameters

        Returns:
            Dict containing the query result
        """
        result = await self.execute_query_async(query, params)

        # Invalidate related cache entries
        if self.cache_manager and self.cache_manager.enabled:
            await self._invalidate_affected_cache(query)

        return result

    def _should_use_cache(self) -> bool:
        """
        Determine if caching should be used for this request.

        Returns:
            bool: True if caching should be used
        """
        return (
            self.cache_enabled
            and self.cache_manager is not None
            and self.cache_manager.enabled
        )

    def _is_write_query(self, query: str) -> bool:
        """
        Check if a query is a write operation.

        Args:
            query: SQL query string

        Returns:
            bool: True if query is a write operation
        """
        query_upper = query.strip().upper()
        write_keywords = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "DROP",
            "ALTER",
            "TRUNCATE",
        ]
        return any(query_upper.startswith(keyword) for keyword in write_keywords)

    def _generate_cache_key(self, query: str, params: Optional[Dict[str, Any]]) -> str:
        """
        Generate a cache key for the database query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            str: The generated cache key
        """
        key_params = {
            "query": query.strip(),
            "params": params or {},
            "db_type": self.db_type,
        }

        return self.cache_manager.generate_cache_key("database_query", **key_params)

    async def _cache_query_result(
        self, cache_key: str, result: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Cache the database query result.

        Args:
            cache_key: The cache key
            result: The result to cache
        """
        try:
            ttl = get_ttl_for_strategy(CacheStrategy.DATABASE_QUERY)

            # Convert DataFrame to JSON-serializable format
            cache_data = {}
            for key, df in result.items():
                if isinstance(df, pd.DataFrame):
                    cache_data[key] = df.to_dict("records")
                else:
                    cache_data[key] = df

            await self.cache_manager.set(cache_key, cache_data, ttl)
            logger.debug(f"Cached database result with TTL {ttl}s")

        except Exception as e:
            logger.error(f"Failed to cache database result: {e}")

    async def _invalidate_affected_cache(self, query: str) -> None:
        """
        Invalidate cache entries affected by a write query.

        Args:
            query: SQL query string
        """
        try:
            # Extract table names from query
            table_names = self._extract_table_names(query)

            # Invalidate cache entries for each affected table
            total_invalidated = 0
            for table_name in table_names:
                pattern = f"database_query:*{table_name}*"
                invalidated = await self.cache_manager.delete_pattern(pattern)
                total_invalidated += invalidated
                logger.debug(
                    f"Invalidated {invalidated} cache entries for table: {table_name}"
                )

            if total_invalidated > 0:
                logger.info(
                    f"Invalidated {total_invalidated} database cache entries due to write operation"
                )

        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")

    def _extract_table_names(self, query: str) -> List[str]:
        """
        Extract table names from SQL query.

        Args:
            query: SQL query string

        Returns:
            List of table names found in the query
        """
        table_names = set()
        query_upper = query.upper()

        # Common SQL patterns to extract table names
        patterns = [
            r"\bFROM\s+([\w.]+)",
            r"\bINTO\s+([\w.]+)",
            r"\bUPDATE\s+([\w.]+)",
            r"\bJOIN\s+([\w.]+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query_upper)
            for match in matches:
                # Remove schema prefixes and quotes
                table_name = match.split(".")[-1].strip("\"'")
                table_names.add(table_name.lower())

        return list(table_names)

    async def invalidate_cache_pattern(self, pattern: str) -> int:
        """
        Invalidate database cache entries matching a pattern.

        Args:
            pattern: Redis pattern to match

        Returns:
            int: Number of keys deleted
        """
        if not self._should_use_cache():
            return 0

        return await self.cache_manager.delete_pattern(pattern)
