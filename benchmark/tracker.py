"""Benchmark tracking utilities for WarehouseEye demo runs."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

INSTANCE_COST_PER_HOUR_USD = 1.99
GPT4V_INPUT_COST_PER_M_TOKEN_USD = 10.0
GPT4V_OUTPUT_COST_PER_M_TOKEN_USD = 30.0


@dataclass
class StageTiming:
    stage: str
    elapsed_sec: float


@dataclass
class RequestMetric:
    name: str
    started_at: float
    elapsed_sec: float
    input_tokens: int
    output_tokens: int
    success: bool
    stage: str | None = None


@dataclass
class BenchmarkTracker:
    """Collects lightweight runtime and cost metrics for demo telemetry."""

    started_at: float = field(default_factory=time.perf_counter)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    request_count: int = 0
    failure_count: int = 0
    stage_timings: list[StageTiming] = field(default_factory=list)
    request_metrics: list[RequestMetric] = field(default_factory=list)

    def record_stage(self, stage: str, elapsed_sec: float) -> None:
        self.stage_timings.append(StageTiming(stage=stage, elapsed_sec=max(0.0, elapsed_sec)))

    def record_request(
        self,
        *,
        name: str,
        elapsed_sec: float,
        input_tokens: int,
        output_tokens: int,
        success: bool = True,
        stage: str | None = None,
    ) -> None:
        safe_in = max(0, int(input_tokens))
        safe_out = max(0, int(output_tokens))
        self.request_count += 1
        if not success:
            self.failure_count += 1
        self.total_input_tokens += safe_in
        self.total_output_tokens += safe_out
        self.request_metrics.append(
            RequestMetric(
                name=name,
                started_at=time.perf_counter(),
                elapsed_sec=max(0.0, elapsed_sec),
                input_tokens=safe_in,
                output_tokens=safe_out,
                success=success,
                stage=stage,
            )
        )

    def record_failure(self, *, name: str, elapsed_sec: float, stage: str | None = None) -> None:
        self.record_request(
            name=name,
            elapsed_sec=elapsed_sec,
            input_tokens=0,
            output_tokens=0,
            success=False,
            stage=stage,
        )

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def wall_time_sec(self) -> float:
        return max(0.0, time.perf_counter() - self.started_at)

    @property
    def cumulative_cost_usd(self) -> float:
        return round((self.wall_time_sec / 3600.0) * INSTANCE_COST_PER_HOUR_USD, 6)

    def throughput_series(self) -> list[dict[str, float]]:
        points: list[dict[str, float]] = []
        elapsed_cursor = 0.0
        for metric in self.request_metrics:
            elapsed_cursor += metric.elapsed_sec
            produced_tokens = metric.input_tokens + metric.output_tokens
            tps = produced_tokens / metric.elapsed_sec if metric.elapsed_sec > 0 else 0.0
            points.append(
                {
                    "elapsed_sec": round(elapsed_cursor, 3),
                    "tokens_per_sec": round(tps, 3),
                }
            )
        return points

    def compare_vs_gpt4v(
        self,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> dict[str, float]:
        in_tokens = self.total_input_tokens if input_tokens is None else max(0, int(input_tokens))
        out_tokens = self.total_output_tokens if output_tokens is None else max(0, int(output_tokens))
        gpt4v_cost = (
            (in_tokens / 1_000_000.0) * GPT4V_INPUT_COST_PER_M_TOKEN_USD
            + (out_tokens / 1_000_000.0) * GPT4V_OUTPUT_COST_PER_M_TOKEN_USD
        )
        warehouseeye_cost = self.cumulative_cost_usd
        savings = gpt4v_cost - warehouseeye_cost
        savings_pct = (savings / gpt4v_cost * 100.0) if gpt4v_cost > 0 else 0.0
        return {
            "warehouseeye_cost_usd": round(warehouseeye_cost, 6),
            "gpt4v_estimated_cost_usd": round(gpt4v_cost, 6),
            "savings_usd": round(savings, 6),
            "savings_pct": round(savings_pct, 2),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "totals": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "request_count": self.request_count,
                "failure_count": self.failure_count,
                "wall_time_sec": round(self.wall_time_sec, 3),
                "cumulative_cost_usd": self.cumulative_cost_usd,
            },
            "stages": [asdict(item) for item in self.stage_timings],
            "requests": [asdict(item) for item in self.request_metrics],
            "throughput": self.throughput_series(),
            "gpt4v_comparison": self.compare_vs_gpt4v(),
        }

    def export_to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        return target
