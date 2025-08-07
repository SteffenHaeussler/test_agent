from unittest.mock import patch

import pandas as pd
import pytest

from src.agent.adapters.database import BaseDatabaseAdapter
from src.agent.exceptions import DatabaseConnectionException


@pytest.fixture
def database_instance():
    kwargs = {
        "connection_string": "postgresql://test:test@localhost:5432/test",
        "db_type": "postgres",
        "max_retries": 1,  # Reduce retries for tests
        "base_delay": 0.001,  # 1ms instead of 1s
        "max_delay": 0.002,  # 2ms instead of 60s
    }
    return BaseDatabaseAdapter(kwargs)


class TestDatabase:
    def test_database_init(self, database_instance):
        assert (
            database_instance.connection_string
            == "postgresql://test:test@localhost:5432/test"
        )
        assert database_instance.db_type == "postgres"

    def test_execute_query(self, database_instance):
        from unittest.mock import AsyncMock, MagicMock

        # Mock the async engine creation and connection
        mock_engine = MagicMock()
        database_instance.engine = mock_engine

        # Mock execute_query_sync to return empty DataFrame
        with patch.object(
            database_instance,
            "execute_query_sync",
            return_value={"data": pd.DataFrame()},
        ):
            # Mock connect and disconnect for context manager
            with patch.object(database_instance, "connect", new_callable=AsyncMock):
                with patch.object(
                    database_instance, "disconnect", new_callable=AsyncMock
                ):
                    with database_instance as db:
                        result = db.execute_query("SELECT * FROM your_table")

        pd.testing.assert_frame_equal(result["data"], pd.DataFrame())

    @patch(
        "src.agent.adapters.database.create_async_engine",
        side_effect=Exception("Database connection failed"),
    )
    def test_connection_error(self, mock_create_engine, database_instance):
        with pytest.raises(DatabaseConnectionException):
            with database_instance as db:
                db.execute_query("SELECT * FROM your_table")

    def test_execute_query_no_engine(self, database_instance):
        with pytest.raises(DatabaseConnectionException):
            database_instance.execute_query("SELECT * FROM your_table")
