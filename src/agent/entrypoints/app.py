import asyncio
import os
from time import time

import src.agent.service_layer.handlers as handlers
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from loguru import logger
from src.agent.adapters.adapter import RouterAdapter
from src.agent.adapters.notifications import SlackNotifications, WSNotifications
from src.agent.bootstrap import bootstrap
from src.agent.config import get_logging_config, get_tracing_config, get_cache_config
from src.agent.domain.commands import Question, Scenario, SQLQuestion
from src.agent.observability.context import connected_clients, ctx_query_id

if os.getenv("IS_TESTING") != "true":
    load_dotenv(".env")
from src.agent.observability.logging import setup_logging
from src.agent.observability.tracing import setup_tracing
from starlette.responses import StreamingResponse
from starlette.websockets import WebSocketState

setup_tracing(get_tracing_config())
setup_logging(get_logging_config())

app = FastAPI()

# Initialize cache manager
cache_manager = None
try:
    from src.agent.adapters.cache import CacheManager

    cache_config = get_cache_config()
    if cache_config["enabled"]:
        cache_manager = CacheManager(cache_config)

        # Initialize cache connection on startup
        @app.on_event("startup")
        async def startup_event():
            if cache_manager:
                await cache_manager.initialize()

        @app.on_event("shutdown")
        async def shutdown_event():
            if cache_manager:
                await cache_manager.close()

            # Close database connections for all adapters
            if hasattr(bus.adapter, "sql_adapter"):
                if hasattr(bus.adapter.sql_adapter, "database"):
                    await bus.adapter.sql_adapter.database.disconnect()
            if hasattr(bus.adapter, "scenario_adapter"):
                if hasattr(bus.adapter.scenario_adapter, "database"):
                    await bus.adapter.scenario_adapter.database.disconnect()
            if hasattr(bus.adapter, "agent_adapter"):
                if hasattr(bus.adapter.agent_adapter, "database"):
                    await bus.adapter.agent_adapter.database.disconnect()
except Exception as e:
    logger.warning(f"Failed to initialize cache manager: {e}")

bus = bootstrap(
    adapter=RouterAdapter(),
    notifications=[
        SlackNotifications(),
        WSNotifications(),
    ],
)


@app.get("/answer")
async def answer(
    question: str,
    x_session_id: str = Header(alias="X-Session-ID"),
):
    """
    Entrypoint for the agent.

    Args:
        question: str: The question to answer.
        x_session_id: str: Session ID from X-Session-ID header.

    Returns:
        response: str: The response to the question.

    Raises:
        HTTPException: If the question is invalid.
        ValueError: If the question is invalid.
    """
    session_id = x_session_id
    logger.info(f"session_id: {session_id}")

    if not question:
        raise HTTPException(status_code=400, detail="No question asked")

    ctx_query_id.set(session_id)

    try:
        command = Question(question=question, q_id=session_id)
        # Run the command handling in the background (now natively async)
        asyncio.create_task(bus.handle(command))
        return {"status": "processing", "message": "Event triggered successfully"}

    except (handlers.InvalidQuestion, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/query")
async def query(
    question: str,
    x_session_id: str = Header(alias="X-Session-ID"),
):
    """
    Entrypoint for the SQL agent.

    Args:
        question: str: The question to answer via SQL.
        x_session_id: str: Session ID from X-Session-ID header.

    Returns:
        response: str: The response to the question.

    Raises:
        HTTPException: If the question is invalid.
        ValueError: If the question is invalid.
    """
    session_id = x_session_id
    logger.info(f"session_id: {session_id}")

    if not question:
        raise HTTPException(status_code=400, detail="No question asked")

    ctx_query_id.set(session_id)

    try:
        command = SQLQuestion(question=question, q_id=session_id)
        # Run the command handling in the background (now natively async)
        asyncio.create_task(bus.handle(command))
        return {"status": "processing", "message": "Event triggered successfully"}

    except (handlers.InvalidQuestion, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/scenario")
async def scenario(
    question: str,
    x_session_id: str = Header(alias="X-Session-ID"),
):
    """
    Entrypoint for the Scenario agent.

    Args:
        question: str: The question to answer via SQL.
        x_session_id: str: Session ID from X-Session-ID header.

    Returns:
        response: str: The response to the question.

    Raises:
        HTTPException: If the question is invalid.
        ValueError: If the question is invalid.
    """
    session_id = x_session_id
    logger.info(f"session_id: {session_id}")

    if not question:
        raise HTTPException(status_code=400, detail="No question asked")

    ctx_query_id.set(session_id)

    try:
        command = Scenario(question=question, q_id=session_id)
        # Run the command handling in the background (now natively async)
        asyncio.create_task(bus.handle(command))
        return {"status": "processing", "message": "Event triggered successfully"}

    except (handlers.InvalidQuestion, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health(request: Request):
    logger.debug(f"Methode: {request.method} on {request.url.path}")
    return {"version": "0.0.1", "timestamp": time()}


@app.get("/metrics/cache")
async def cache_metrics():
    """
    Get cache metrics including hit/miss ratios and performance statistics.

    Returns:
        dict: Cache metrics and statistics
    """
    if not cache_manager or not cache_manager.enabled:
        raise HTTPException(status_code=503, detail="Cache is not available")

    try:
        metrics = cache_manager.metrics

        # Get additional cache info if possible
        cache_info = {}
        if cache_manager.redis:
            try:
                info = await cache_manager.redis.info()
                cache_info = {
                    "memory_usage": info.get("used_memory_human", "N/A"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace": {
                        db: info.get(db, {})
                        for db in info.keys()
                        if db.startswith("db")
                    },
                }
            except Exception as e:
                logger.warning(f"Could not get Redis info: {e}")

        return {
            "cache_enabled": cache_manager.enabled,
            "metrics": {
                "hits": metrics.hits,
                "misses": metrics.misses,
                "sets": metrics.sets,
                "deletes": metrics.deletes,
                "errors": metrics.errors,
                "hit_ratio": metrics.hit_ratio,
                "total_operations": metrics.hits + metrics.misses,
            },
            "redis_info": cache_info,
            "timestamp": time(),
        }
    except Exception as e:
        logger.error(f"Error getting cache metrics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving cache metrics")


@app.post("/cache/invalidate")
async def invalidate_cache(
    pattern: str = Query(
        ..., description="Redis pattern to match for cache invalidation"
    ),
):
    """
    Manually invalidate cache entries matching a pattern.

    Args:
        pattern: Redis pattern to match (e.g., 'llm_response:*', 'database_query:*users*')

    Returns:
        dict: Number of keys deleted
    """
    if not cache_manager or not cache_manager.enabled:
        raise HTTPException(status_code=503, detail="Cache is not available")

    try:
        deleted_count = await cache_manager.delete_pattern(pattern)
        logger.info(
            f"Manually invalidated {deleted_count} cache entries matching pattern: {pattern}"
        )

        return {"pattern": pattern, "deleted_keys": deleted_count, "timestamp": time()}
    except Exception as e:
        logger.error(f"Error invalidating cache pattern {pattern}: {e}")
        raise HTTPException(status_code=500, detail="Error invalidating cache")


@app.get("/cache/info")
async def cache_info():
    """
    Get general cache configuration and status information.

    Returns:
        dict: Cache configuration and status
    """
    if not cache_manager:
        return {
            "cache_enabled": False,
            "status": "Cache not configured",
            "timestamp": time(),
        }

    try:
        config = cache_manager.config

        # Test connectivity
        is_connected = False
        connection_error = None
        if cache_manager.redis:
            try:
                await cache_manager.redis.ping()
                is_connected = True
            except Exception as e:
                connection_error = str(e)

        return {
            "cache_enabled": cache_manager.enabled,
            "status": "connected" if is_connected else "disconnected",
            "connection_error": connection_error,
            "config": {
                "host": config.get("host", "N/A"),
                "port": config.get("port", "N/A"),
                "db": config.get("db", "N/A"),
                "max_connections": config.get("max_connections", "N/A"),
            },
            "timestamp": time(),
        }
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving cache info")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = Query(...)):
    await websocket.accept()

    current_loop = asyncio.get_running_loop()
    connected_time = time()
    connected_clients[session_id] = {
        "ws": websocket,
        "loop": current_loop,
        "last_event_time": connected_time,  # Initialize last_event_time
    }

    logger.info(f"Client connected: {session_id}, loop: {id(current_loop)}")

    timeout = 90

    try:
        while True:
            await asyncio.sleep(1)

            client_info = connected_clients.get(session_id)
            if (
                not client_info
                or client_info["ws"].client_state == WebSocketState.DISCONNECTED
            ):
                logger.info(
                    f"Client {session_id} no longer in registry or disconnected, breaking loop."
                )
                break

            if time() - client_info["last_event_time"] > timeout:
                logger.info(f"Session timeout: {session_id}")
                await websocket.close(code=1000, reason="Idle timeout")
                break

    except WebSocketDisconnect:
        logger.info(f"Disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: session_id={session_id}: {e}")
    finally:
        connected_clients.pop(session_id, None)
        logger.info(f"WebSocket removed from registry: session_id={session_id}")

        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close(
                    code=1001, reason="Server shutting down connection"
                )
            except Exception:
                pass  # May already be closed or in a bad state
        logger.info(f"Connection closed for {session_id}")


@app.get("/sse/{session_id}")
async def sse(request: Request, session_id: str):
    loop = asyncio.get_event_loop()

    if session_id not in connected_clients:
        connected_clients[session_id] = {
            "queue": asyncio.Queue(),
            "loop": loop,
            "last_event_time": time(),
        }

    queue = connected_clients[session_id]["queue"]

    async def event_stream():
        while True:
            if await request.is_disconnected():
                logger.info(f"Client {session_id} disconnected from SSE.")
                connected_clients.pop(session_id, None)
                break

            try:
                message = await asyncio.wait_for(queue.get(), timeout=20.0)
                yield message
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
