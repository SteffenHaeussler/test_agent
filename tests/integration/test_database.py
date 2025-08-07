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
        from unittest.mock import MagicMock

        # Mock the sync engine to avoid actual connection
        mock_engine = MagicMock()
        mock_session_maker = MagicMock()

        # Mock _get_sync_engine to return mocked engine
        with patch.object(
            database_instance,
            "_get_sync_engine",
            return_value=(mock_engine, mock_session_maker),
        ):
            # Mock _connect_sync_impl to avoid actual connection
            with patch.object(database_instance, "_connect_sync_impl"):
                # Mock execute_query_sync to return empty DataFrame
                with patch.object(
                    database_instance,
                    "execute_query_sync",
                    return_value={"data": pd.DataFrame()},
                ):
                    with database_instance as db:
                        result = db.execute_query("SELECT * FROM your_table")

        pd.testing.assert_frame_equal(result["data"], pd.DataFrame())

    def test_connection_error(self, database_instance):
        # Mock _connect_sync_impl to raise an exception
        with patch.object(
            database_instance,
            "_connect_sync_impl",
            side_effect=DatabaseConnectionException("Database connection failed"),
        ):
            with pytest.raises(DatabaseConnectionException):
                with database_instance as db:
                    db.execute_query("SELECT * FROM your_table")

    def test_execute_query_no_engine(self, database_instance):
        with pytest.raises(DatabaseConnectionException):
            database_instance.execute_query("SELECT * FROM your_table")
