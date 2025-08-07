import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import pytest

from src.agent.adapters.database import BaseDatabaseAdapter
from src.agent.exceptions import (
    DatabaseConnectionException,
    DatabaseQueryException,
)


@pytest.fixture
def database_instance():
    kwargs = {
        "connection_string": "postgresql://test:test@localhost:5432/test",
        "db_type": "postgres",
        "query_timeout": 30,
        "max_retries": 2,
        "base_delay": 0.1,
        "max_delay": 1.0,
    }
    return BaseDatabaseAdapter(kwargs)


@pytest.fixture
def mock_engine():
    """Mock async engine for testing."""
    engine = AsyncMock()
    engine.dispose = AsyncMock()
    return engine


@pytest.fixture
def mock_session():
    """Mock async session for testing."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.execute = AsyncMock()

    # Mock the begin context manager
    mock_begin = AsyncMock()
    mock_begin.__aenter__ = AsyncMock(return_value=session)
    mock_begin.__aexit__ = AsyncMock(return_value=None)
    session.begin = MagicMock(return_value=mock_begin)

    return session


class TestAsyncDatabaseAdapter:
    """Test the async database adapter functionality."""

    def test_database_init(self, database_instance):
        """Test database adapter initialization."""
        assert (
            database_instance.connection_string
            == "postgresql://test:test@localhost:5432/test"
        )
        assert database_instance.db_type == "postgres"
        assert database_instance.query_timeout == 30
        assert database_instance.max_retries == 2

    @pytest.mark.asyncio
    async def test_connect_success(self, database_instance):
        """Test successful async connection."""
        with (
            patch.object(
                database_instance, "_create_async_engine", new_callable=AsyncMock
            ) as mock_create,
            patch.object(
                database_instance, "_test_connection", new_callable=AsyncMock
            ) as mock_test,
        ):
            mock_engine = AsyncMock()
            mock_create.return_value = mock_engine

            await database_instance.connect()

            mock_create.assert_called_once()
            mock_test.assert_called_once()
            assert database_instance.engine == mock_engine

    @pytest.mark.asyncio
    async def test_connect_with_retries(self, database_instance):
        """Test connection with retry logic."""
        with (
            patch.object(
                database_instance, "_create_async_engine", new_callable=AsyncMock
            ) as mock_create,
            patch.object(
                database_instance, "_test_connection", new_callable=AsyncMock
            ) as mock_test,
        ):
            # Mock successful engine creation but failed connection test first, then success
            mock_engine = AsyncMock()
            mock_create.return_value = mock_engine
            mock_test.side_effect = [Exception("Connection test failed"), None]

            await database_instance.connect()

            assert mock_create.call_count == 2  # Called twice due to retry
            assert mock_test.call_count == 2  # First fails, second succeeds
            assert database_instance.engine == mock_engine

    @pytest.mark.asyncio
    async def test_connect_max_retries_exceeded(self, database_instance):
        """Test connection failure after max retries."""
        with patch.object(
            database_instance, "_create_async_engine", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = Exception("Persistent connection error")

            with pytest.raises(DatabaseConnectionException):
                await database_instance.connect()

            # Should try max_retries + 1 times (initial + retries)
            assert mock_create.call_count == database_instance.max_retries + 1

    @pytest.mark.asyncio
    async def test_disconnect(self, database_instance, mock_engine):
        """Test async disconnect."""
        database_instance.engine = mock_engine

        await database_instance.disconnect()

        mock_engine.dispose.assert_called_once()
        assert database_instance.engine is None
        assert database_instance.session_maker is None

    @pytest.mark.asyncio
    async def test_health_check_success(self, database_instance, mock_session):
        """Test successful health check."""

        def mock_session_maker():
            return mock_session

        database_instance.session_maker = mock_session_maker
        database_instance.engine = AsyncMock()

        result = await database_instance.health_check()

        assert result is True
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, database_instance, mock_session):
        """Test health check failure."""

        def mock_session_maker():
            return mock_session

        mock_session.execute.side_effect = Exception("Health check failed")
        database_instance.session_maker = mock_session_maker
        database_instance.engine = AsyncMock()

        result = await database_instance.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_no_engine(self, database_instance):
        """Test health check with no engine."""
        database_instance.engine = None

        result = await database_instance.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_query_success(self, database_instance, mock_session):
        """Test successful query execution."""
        # Mock result data
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(_mapping={"id": 1, "name": "test"}),
            MagicMock(_mapping={"id": 2, "name": "test2"}),
        ]
        mock_result.keys.return_value = ["id", "name"]

        mock_session.execute.return_value = mock_result

        # Create a callable that returns the mock session
        def mock_session_maker():
            return mock_session

        database_instance.engine = AsyncMock()
        database_instance.session_maker = mock_session_maker

        result = await database_instance.execute_query_async("SELECT * FROM test_table")

        assert "data" in result
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) == 2
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_with_pagination(self, database_instance, mock_session):
        """Test query execution with pagination."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(_mapping={"id": 1, "name": "test"})
        ]
        mock_result.keys.return_value = ["id", "name"]

        mock_session.execute.return_value = mock_result

        # Create a callable that returns the mock session
        def mock_session_maker():
            return mock_session

        database_instance.engine = AsyncMock()
        database_instance.session_maker = mock_session_maker

        result = await database_instance.execute_query_async(
            "SELECT * FROM test_table", limit=10, offset=20
        )

        # Check that the query was modified with LIMIT and OFFSET
        call_args = mock_session.execute.call_args
        query_text = str(call_args[0][0])
        assert "LIMIT 10" in query_text
        assert "OFFSET 20" in query_text

        assert "data" in result
        assert isinstance(result["data"], pd.DataFrame)

    @pytest.mark.asyncio
    async def test_execute_query_no_engine(self, database_instance):
        """Test query execution with no engine."""
        database_instance.engine = None

        with pytest.raises(DatabaseConnectionException):
            await database_instance.execute_query_async("SELECT * FROM test_table")

    @pytest.mark.asyncio
    async def test_execute_query_timeout(self, database_instance, mock_session):
        """Test query execution timeout."""

        def mock_session_maker():
            return mock_session

        # Mock a slow query that will timeout
        mock_session.execute.side_effect = asyncio.sleep(60)  # Simulate long query

        database_instance.engine = AsyncMock()
        database_instance.session_maker = mock_session_maker
        database_instance.query_timeout = 0.1  # Very short timeout

        with pytest.raises(DatabaseQueryException):
            await database_instance.execute_query_async("SELECT * FROM test_table")

    @pytest.mark.asyncio
    async def test_execute_query_streaming_success(
        self, database_instance, mock_session
    ):
        """Test successful streaming query execution."""
        # Mock result data
        mock_result = MagicMock()
        mock_result.__iter__.return_value = [
            MagicMock(_mapping={"id": 1, "name": "test1"}),
            MagicMock(_mapping={"id": 2, "name": "test2"}),
            MagicMock(_mapping={"id": 3, "name": "test3"}),
        ]
        mock_result.keys.return_value = ["id", "name"]

        mock_session.execute.return_value = mock_result

        def mock_session_maker():
            return mock_session

        database_instance.engine = AsyncMock()
        database_instance.session_maker = mock_session_maker

        chunks = []
        async for chunk in database_instance.execute_query_streaming(
            "SELECT * FROM test_table", chunk_size=2
        ):
            chunks.append(chunk)
            assert isinstance(chunk, pd.DataFrame)

        # Should have 2 chunks (2 rows + 1 row)
        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 1

    @pytest.mark.asyncio
    async def test_insert_batch_success(self, database_instance, mock_session):
        """Test successful batch insert."""

        def mock_session_maker():
            return mock_session

        database_instance.engine = AsyncMock()
        database_instance.session_maker = mock_session_maker

        data_list = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]

        result = await database_instance.insert_batch("test_table", data_list)

        assert result is True
        # Should have called execute twice (once per row)
        assert mock_session.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_insert_batch_no_engine(self, database_instance):
        """Test batch insert with no engine."""
        database_instance.engine = None

        with pytest.raises(DatabaseConnectionException):
            await database_instance.insert_batch(
                "test_table", [{"id": 1, "name": "test"}]
            )

    @pytest.mark.asyncio
    async def test_insert_batch_empty_data(self, database_instance):
        """Test batch insert with empty data."""
        database_instance.engine = AsyncMock()

        result = await database_instance.insert_batch("test_table", [])

        assert result is True

    @pytest.mark.asyncio
    async def test_insert_data_single_row(self, database_instance, mock_session):
        """Test inserting a single row of data."""

        def mock_session_maker():
            return mock_session

        database_instance.engine = AsyncMock()
        database_instance.session_maker = mock_session_maker

        result = await database_instance.insert_data(
            "test_table", {"id": 1, "name": "test"}
        )

        assert result is True
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_schema_success(self, database_instance, mock_session):
        """Test successful schema retrieval."""
        mock_session.run_sync = AsyncMock()

        def mock_session_maker():
            return mock_session

        database_instance.engine = AsyncMock()
        database_instance.session_maker = mock_session_maker

        with patch("src.agent.adapters.database.MetaData") as mock_metadata:
            mock_metadata_instance = MagicMock()
            mock_metadata.return_value = mock_metadata_instance

            result = await database_instance.get_schema_async()

            assert result == mock_metadata_instance
            mock_session.run_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_schema_no_engine(self, database_instance):
        """Test schema retrieval with no engine."""
        database_instance.engine = None

        with pytest.raises(DatabaseConnectionException):
            await database_instance.get_schema_async()

    @pytest.mark.asyncio
    async def test_context_manager(self, database_instance):
        """Test async context manager functionality."""
        with (
            patch.object(
                database_instance, "connect", new_callable=AsyncMock
            ) as mock_connect,
            patch.object(
                database_instance, "disconnect", new_callable=AsyncMock
            ) as mock_disconnect,
        ):
            async with database_instance as db:
                assert db == database_instance
                mock_connect.assert_called_once()

            mock_disconnect.assert_called_once()

    def test_legacy_get_connection_raises(self, database_instance):
        """Test that legacy _get_connection method raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            database_instance._get_connection()
