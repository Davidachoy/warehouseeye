"""Pydantic schemas for WarehouseEye API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request body for launching analysis."""

    video_url: str = Field(..., description="Video URL or local path accepted by the pipeline.")
    video_id: str = Field(..., description="Stable identifier used for idempotent processing.")
    force: bool = Field(
        default=False,
        description="If true, reset existing artifacts/status for this video_id and launch a fresh run.",
    )


class AnalyzeResponse(BaseModel):
    """Response body for analysis launch endpoint."""

    status: str = Field(..., description="Task state: started, running, or completed.")
    task_id: str = Field(..., description="Task UUID used by SSE progress stream.")


class QueryRequest(BaseModel):
    """Request body for natural language timeline query."""

    video_id: str
    question: str


class QueryAlternative(BaseModel):
    """Ambiguous track alternative presented to the caller."""

    track_id: int
    color_tag: str | None = None
    crop_path: str | None = None


class QueryResponse(BaseModel):
    """Response payload for timeline NL query."""

    matched_track_id: int | None = None
    ambiguous: bool
    alternatives: list[QueryAlternative]
    narrative: str
    timeline: list[dict[str, Any]]


class ActivityAnalysisSchema(BaseModel):
    """Structured activity contract returned by VLM analysis."""

    activity: Literal[
        "walking",
        "standing",
        "handling_object",
        "lifting",
        "interacting",
        "inspecting",
        "idle",
        "other",
    ]
    activity_description: str = Field(..., min_length=1)
    objects_involved: list[str]
    zone_inference: str = Field(..., min_length=1)
    interaction_with_others: str | None = None
    anomaly_flag: bool
    anomaly_reason: str | None = None
    supervisor_attention_recommended: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1)


class BenchmarkResponse(BaseModel):
    """Benchmark metrics from the most recent finished run."""

    frames_analyzed: int
    tokens_per_second_avg: float
    latency_per_crop_ms: float
    wall_time_sec: float
    total_cost_usd: float
    vs_gpt4v_estimated_savings_pct: float


class HealthResponse(BaseModel):
    """Service health status."""

    status: str
    vllm_reachable: bool
    db_path: str
    version: str
    reid_enabled: bool = False
    embedding_url: str | None = None
