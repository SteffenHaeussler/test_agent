import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from src.agent.entrypoints.app import app, cache_manager


class TestCacheMetricsEndpoints:
    """Test suite for cache metrics endpoints."""

    def test_cache_info_when_cache_disabled(self):
        """Test cache info endpoint when cache is disabled."""
        # Temporarily set cache_manager to None to simulate disabled cache
        original_cache_manager = cache_manager
        app.state.cache_manager = None

        with TestClient(app) as client:
            response = client.get("/cache/info")

            assert response.status_code == 200
            data = response.json()
            assert data["cache_enabled"] is False
            assert data["status"] == "Cache not configured"
            assert "timestamp" in data

        # Restore original cache manager
        app.state.cache_manager = original_cache_manager

    def test_cache_metrics_when_cache_disabled(self):
        """Test cache metrics endpoint when cache is disabled."""
        # Mock cache_manager as None to simulate disabled cache
        original_cache_manager = cache_manager
        app.state.cache_manager = None

        with TestClient(app) as client:
            response = client.get("/metrics/cache")

            assert response.status_code == 503
            data = response.json()
            assert data["detail"] == "Cache is not available"

        # Restore original cache manager
        app.state.cache_manager = original_cache_manager

    def test_cache_invalidate_when_cache_disabled(self):
        """Test cache invalidate endpoint when cache is disabled."""
        # Mock cache_manager as None to simulate disabled cache
        original_cache_manager = cache_manager
        app.state.cache_manager = None

        with TestClient(app) as client:
            response = client.post("/cache/invalidate?pattern=test:*")

            assert response.status_code == 503
            data = response.json()
            assert data["detail"] == "Cache is not available"

        # Restore original cache manager
        app.state.cache_manager = original_cache_manager

    def test_health_endpoint_still_works(self):
        """Test that health endpoint is not affected by cache changes."""
        with TestClient(app) as client:
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["version"] == "0.0.1"
            assert "timestamp" in data


class TestCacheMetricsWithMockedManager:
    """Test cache metrics endpoints with mocked cache manager."""

    @pytest.fixture
    def mock_cache_manager(self, monkeypatch):
        """Create a mocked cache manager for testing."""
        mock_manager = MagicMock()
        mock_manager.enabled = True
        mock_manager.redis = AsyncMock()

        # Mock metrics
        mock_metrics = MagicMock()
        mock_metrics.hits = 100
        mock_metrics.misses = 50
        mock_metrics.sets = 75
        mock_metrics.deletes = 10
        mock_metrics.errors = 2
        mock_metrics.hit_ratio = 0.67
        mock_manager.metrics = mock_metrics

        # Mock config
        mock_manager.config = {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "max_connections": 10,
        }

        # Mock Redis operations
        mock_manager.redis.ping = AsyncMock(return_value=True)
        mock_manager.redis.info = AsyncMock(
            return_value={
                "used_memory_human": "1.5MB",
                "connected_clients": 3,
                "total_commands_processed": 1000,
                "db0": {"keys": 50, "expires": 20},
            }
        )
        mock_manager.delete_pattern = AsyncMock(return_value=25)

        # Patch the cache_manager in the app module
        monkeypatch.setattr("src.agent.entrypoints.app.cache_manager", mock_manager)
        return mock_manager

    def test_cache_info_with_enabled_cache(self, mock_cache_manager):
        """Test cache info endpoint with enabled cache."""
        with TestClient(app) as client:
            response = client.get("/cache/info")

            assert response.status_code == 200
            data = response.json()
            assert data["cache_enabled"] is True
            assert data["status"] == "connected"
            assert data["config"]["host"] == "localhost"
            assert data["config"]["port"] == 6379

    def test_cache_metrics_with_enabled_cache(self, mock_cache_manager):
        """Test cache metrics endpoint with enabled cache."""
        with TestClient(app) as client:
            response = client.get("/metrics/cache")

            assert response.status_code == 200
            data = response.json()
            assert data["cache_enabled"] is True
            assert data["metrics"]["hits"] == 100
            assert data["metrics"]["misses"] == 50
            assert data["metrics"]["hit_ratio"] == 0.67
            assert data["redis_info"]["memory_usage"] == "1.5MB"

    def test_cache_invalidate_with_enabled_cache(self, mock_cache_manager):
        """Test cache invalidate endpoint with enabled cache."""
        with TestClient(app) as client:
            response = client.post("/cache/invalidate?pattern=test:*")

            assert response.status_code == 200
            data = response.json()
            assert data["pattern"] == "test:*"
            assert data["deleted_keys"] == 25
            assert "timestamp" in data

            # Verify the delete_pattern method was called
            mock_cache_manager.delete_pattern.assert_called_once_with("test:*")
