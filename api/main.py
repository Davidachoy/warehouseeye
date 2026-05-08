"""FastAPI app exposing the WarehouseEye processing pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from api.query_resolver import resolve_query
from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BenchmarkResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from warehouseeye import __version__
from warehouseeye.pipeline.db import get_video, init_db, set_video_status, upsert_video_start

try:
    import structlog
except ImportError:  # pragma: no cover - optional dependency fallback
    structlog = None


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if structlog is not None:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


_configure_logging()
logger = structlog.get_logger(__name__) if structlog is not None else logging.getLogger(__name__)

try:
    from api.pipeline_runner import run_pipeline_job as _default_run_pipeline_job
except Exception as exc:  # pragma: no cover - allows tests without heavy runtime deps
    logger.warning(
        "pipeline_runner_unavailable: import failed (%s: %s). Install deps (e.g. ffmpeg-python) or fix imports.",
        type(exc).__name__,
        exc,
    )
    _default_run_pipeline_job = None

run_pipeline_job = _default_run_pipeline_job


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize app state for task tracking and idempotency DB; release on shutdown."""
    data_root = Path(os.getenv("WAREHOUSEEYE_DATA_ROOT", "data"))
    data_root.mkdir(parents=True, exist_ok=True)
    status_db_path = data_root / "warehouseeye.sqlite3"
    init_db(status_db_path).close()

    conn = init_db(status_db_path)
    stale = int(conn.execute("SELECT COUNT(*) FROM videos WHERE status = 'running'").fetchone()[0])
    if stale:
        conn.execute(
            "UPDATE videos SET status = 'failed', error = ? WHERE status = 'running'",
            ("Server restarted; in-memory progress was lost.",),
        )
        conn.commit()
        if structlog is not None:
            logger.info("lifespan_cleared_stale_running_videos", count=stale)
        else:
            logger.info("lifespan_cleared_stale_running_videos count=%s", stale)
    conn.close()

    app.state.data_root = data_root
    app.state.status_db_path = status_db_path
    app.state.tasks = {}
    app.state.task_queues = {}
    app.state.last_benchmark = {
        "frames_analyzed": 0,
        "tokens_per_second_avg": 0.0,
        "latency_per_crop_ms": 0.0,
        "wall_time_sec": 0.0,
        "total_cost_usd": 0.0,
        "vs_gpt4v_estimated_savings_pct": 35.0,
    }
    if structlog is not None:
        logger.info(
            "lifespan_startup_complete",
            data_root=str(data_root),
            status_db=str(status_db_path),
            pipeline_runner_loaded=run_pipeline_job is not None,
        )
    else:
        logger.info(
            "lifespan_startup_complete data_root=%s status_db=%s pipeline_runner_loaded=%s",
            data_root,
            status_db_path,
            run_pipeline_job is not None,
        )
    yield


app = FastAPI(title="WarehouseEye API", version=__version__, lifespan=app_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Attach and log a correlated request_id for each HTTP request."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    if structlog is not None:
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        logger.info("request_started", method=request.method, path=request.url.path)
    else:
        logger.info("request_started request_id=%s method=%s path=%s", request_id, request.method, request.url.path)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    if structlog is not None:
        logger.info("request_finished", method=request.method, path=request.url.path, status=response.status_code)
    else:
        logger.info(
            "request_finished request_id=%s method=%s path=%s status=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
        )
    return response


def _push_progress(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[dict[str, Any]], payload: dict[str, Any]) -> None:
    loop.call_soon_threadsafe(queue.put_nowait, payload)


def _launch_pipeline_task(
    task_id: str,
    video_id: str,
    video_url: str,
    loop: asyncio.AbstractEventLoop,
) -> None:
    queue = app.state.task_queues[task_id]

    def emit(payload: dict[str, Any]) -> None:
        _push_progress(loop, queue, payload)

    def worker() -> None:
        if structlog is not None:
            logger.info("pipeline_worker_started", task_id=task_id, video_id=video_id)
        else:
            logger.info("pipeline_worker_started task_id=%s video_id=%s", task_id, video_id)
        try:
            if run_pipeline_job is None:
                raise RuntimeError("Pipeline runner is unavailable in this environment.")
            result = run_pipeline_job(
                video_id=video_id,
                video_url=video_url,
                task_id=task_id,
                data_root=app.state.data_root,
                status_db_path=app.state.status_db_path,
                emit=emit,
            )
            app.state.tasks[task_id]["status"] = "completed"
            app.state.tasks[task_id]["result"] = result
            app.state.last_benchmark = result.get("benchmark", app.state.last_benchmark)
            if structlog is not None:
                logger.info("pipeline_worker_completed", task_id=task_id, video_id=video_id)
            else:
                logger.info("pipeline_worker_completed task_id=%s video_id=%s", task_id, video_id)
        except Exception as exc:  # pragma: no cover - runtime guard
            app.state.tasks[task_id]["status"] = "failed"
            app.state.tasks[task_id]["error"] = str(exc)
            logging.getLogger(__name__).exception(
                "pipeline_worker_failed task_id=%s video_id=%s",
                task_id,
                video_id,
            )
            emit({"stage": "failed", "percent": 100, "message": str(exc)})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    if structlog is not None:
        logger.info("pipeline_thread_spawned", task_id=task_id, video_id=video_id, thread_name=thread.name)
    else:
        logger.info("pipeline_thread_spawned task_id=%s video_id=%s thread=%s", task_id, video_id, thread.name)


def _reset_video_state(video_id: str, conn) -> None:  # type: ignore[no-untyped-def]
    """Delete artifacts + status rows so one video_id can be reprocessed from zero."""
    artifact_dir = app.state.data_root / video_id
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
    conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
    conn.commit()
    stale_task_ids = [
        task_id
        for task_id, task in app.state.tasks.items()
        if task.get("video_id") == video_id
    ]
    for task_id in stale_task_ids:
        app.state.tasks.pop(task_id, None)
        app.state.task_queues.pop(task_id, None)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Launch the WarehouseEye pipeline asynchronously and return immediately."""
    conn = init_db(app.state.status_db_path)
    existing = get_video(conn, request.video_id)

    if request.force:
        if existing and existing.get("status") == "running":
            tid = str(existing.get("task_id") or "")
            if tid and tid in app.state.task_queues:
                conn.close()
                if structlog is not None:
                    logger.info("analyze_force_ignored_active_run", video_id=request.video_id, task_id=tid)
                else:
                    logger.info("analyze_force_ignored_active_run video_id=%s task_id=%s", request.video_id, tid)
                return AnalyzeResponse(status="running", task_id=tid)
        _reset_video_state(request.video_id, conn)
        existing = None

    if existing and existing.get("status") == "completed":
        tid = str(existing.get("task_id") or "unknown")
        conn.close()
        if structlog is not None:
            logger.info("analyze_skip_completed", video_id=request.video_id, task_id=tid)
        else:
            logger.info("analyze_skip_completed video_id=%s task_id=%s", request.video_id, tid)
        return AnalyzeResponse(status="completed", task_id=tid)

    if existing and existing.get("status") == "running":
        tid = str(existing.get("task_id") or "")
        if tid and tid in app.state.task_queues:
            conn.close()
            if structlog is not None:
                logger.info("analyze_skip_active_run", video_id=request.video_id, task_id=tid)
            else:
                logger.info("analyze_skip_active_run video_id=%s task_id=%s", request.video_id, tid)
            return AnalyzeResponse(status="running", task_id=tid)
        if structlog is not None:
            logger.warning(
                "analyze_stale_running_row",
                video_id=request.video_id,
                stale_task_id=tid,
                detail="No in-memory SSE queue; resetting DB row so a new run can start.",
            )
        else:
            logger.warning(
                "analyze_stale_running_row video_id=%s stale_task_id=%s (resetting)",
                request.video_id,
                tid,
            )
        set_video_status(
            conn,
            video_id=request.video_id,
            status="failed",
            completed_at=None,
            error="Stale running state (server restarted or lost queue).",
        )

    task_id = str(uuid.uuid4())
    upsert_video_start(conn, video_id=request.video_id, url=request.video_url, task_id=task_id)
    conn.close()
    app.state.tasks[task_id] = {"video_id": request.video_id, "status": "running"}
    app.state.task_queues[task_id] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    if structlog is not None:
        logger.info(
            "analyze_started",
            video_id=request.video_id,
            task_id=task_id,
            video_url=request.video_url,
            runner_available=run_pipeline_job is not None,
        )
    else:
        logger.info(
            "analyze_started video_id=%s task_id=%s runner_available=%s url=%s",
            request.video_id,
            task_id,
            run_pipeline_job is not None,
            request.video_url,
        )
    _launch_pipeline_task(task_id=task_id, video_id=request.video_id, video_url=request.video_url, loop=loop)
    return AnalyzeResponse(status="started", task_id=task_id)


@app.get("/stream/{task_id}")
async def stream(task_id: str) -> StreamingResponse:
    """Stream task progress updates as Server-Sent Events."""
    queue = app.state.task_queues.get(task_id)
    if queue is None:
        known = list(app.state.task_queues.keys())
        if structlog is not None:
            logger.warning("stream_unknown_task_id", task_id=task_id, known_task_ids=known[:20])
        else:
            logger.warning("stream_unknown_task_id task_id=%s known_count=%s", task_id, len(known))
        raise HTTPException(
            status_code=404,
            detail="Unknown task_id (server may have restarted, or analyze never started this task).",
        )
    if structlog is not None:
        logger.info("stream_client_connected", task_id=task_id)
    else:
        logger.info("stream_client_connected task_id=%s", task_id)

    async def event_generator():  # type: ignore[no-untyped-def]
        while True:
            event = await queue.get()
            if structlog is not None:
                logger.info("stream_event", task_id=task_id, stage=event.get("stage"), percent=event.get("percent"))
            else:
                logger.info("stream_event task_id=%s stage=%s percent=%s", task_id, event.get("stage"), event.get("percent"))
            yield f"event: progress\ndata: {json.dumps(event)}\n\n"
            if event.get("stage") in {"done", "failed"}:
                break

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.get("/timeline/{video_id}")
async def get_timeline(video_id: str) -> JSONResponse:
    """Return the full timeline JSON for a processed video."""
    timeline_path = app.state.data_root / video_id / "timeline.json"
    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="Timeline not found")
    payload = json.loads(timeline_path.read_text(encoding="utf-8"))
    return JSONResponse(payload)


@app.get("/timeline/{video_id}/track/{track_id}")
async def get_timeline_track(video_id: str, track_id: int) -> JSONResponse:
    """Return timeline entries filtered to one track ID."""
    timeline_path = app.state.data_root / video_id / "timeline.json"
    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="Timeline not found")
    payload = json.loads(timeline_path.read_text(encoding="utf-8"))
    track_timeline = [entry for entry in payload.get("timeline", []) if int(entry.get("track_id", -1)) == track_id]
    payload["timeline"] = track_timeline
    return JSONResponse(payload)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Resolve a natural-language query over one video's timeline data."""
    db_path = app.state.data_root / request.video_id / "warehouseeye.sqlite3"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Video not processed")
    payload = await asyncio.to_thread(resolve_query, db_path=db_path, question=request.question)
    return QueryResponse(**payload)


@app.get("/benchmark", response_model=BenchmarkResponse)
async def benchmark() -> BenchmarkResponse:
    """Return benchmarking metrics from the latest successful run."""
    return BenchmarkResponse(**app.state.last_benchmark)


async def _is_vllm_reachable() -> bool:
    base_url = os.getenv("AMD_URL", "").rstrip("/")
    if not base_url:
        return False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{base_url}/health")
            return response.status_code < 400
    except Exception:
        return False


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health and dependency reachability."""
    return HealthResponse(
        status="ok",
        vllm_reachable=await _is_vllm_reachable(),
        db_path=str(app.state.status_db_path),
        version=__version__,
        reid_enabled=os.getenv("ENABLE_REID", "0") == "1",
        embedding_url=os.getenv("EMBEDDING_URL"),
    )
