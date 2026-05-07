"""Semantic activity analyzer over tracked person crops."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from warehouseeye.gpu.vllm_client import VLLMClient
from warehouseeye.pipeline.db import update_track_activity

logger = logging.getLogger(__name__)

REQUIRED_ACTIVITY_KEYS = {
    "activity",
    "relative_location",
    "visible_tools",
    "object_interaction",
    "posture",
    "anomaly",
    "severity",
}
VALID_SEVERITY_VALUES = {None, "low", "medium", "high"}


class VisionAnalyzer:
    """Analyze person tracks by querying a VLM on representative crops."""

    def __init__(
        self,
        vllm_client: VLLMClient,
        prompt_path: str | Path | None = None,
        strict_prompt_path: str | Path | None = None,
        base_dir: str | Path | None = None,
        concurrency: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.vllm_client = vllm_client
        self.base_dir = Path(base_dir).resolve() if base_dir else None
        self.concurrency = concurrency or int(os.getenv("AMD_CONCURRENCY", "8"))
        self.max_tokens = max_tokens or int(os.getenv("AMD_MAX_TOKENS", "300"))

        prompt_path_value = Path(
            prompt_path or os.getenv("ACTIVITY_PROMPT_PATH", "prompts/activity_extraction.txt")
        )
        strict_prompt_path_value = Path(
            strict_prompt_path
            or os.getenv("ACTIVITY_PROMPT_STRICT_PATH", "prompts/activity_extraction_strict.txt")
        )
        self.prompt = prompt_path_value.read_text(encoding="utf-8").strip()
        self.strict_prompt = strict_prompt_path_value.read_text(encoding="utf-8").strip()

        self.total_tracks = 0
        self.total_crops = 0
        self.parse_successes = 0
        self.parse_failures = 0

    @staticmethod
    def select_representative_rows(
        rows: list[tuple[Any, ...]], min_count: int = 3, max_count: int = 5
    ) -> list[tuple[Any, ...]]:
        """Pick temporally spread rows while capping request volume."""
        total = len(rows)
        if total <= max_count:
            return rows
        sample_count = min(max_count, total)
        sample_count = max(sample_count, min(min_count, total))
        if sample_count >= total:
            return rows
        indices = sorted(
            {
                round(step * (total - 1) / (sample_count - 1))
                for step in range(sample_count)
            }
        )
        return [rows[index] for index in indices]

    @staticmethod
    def _extract_json_block(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            parts = stripped.split("```")
            for part in parts:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[len("json") :].strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    return candidate

        start = stripped.find("{")
        if start == -1:
            raise ValueError("No JSON object found in model response.")

        depth = 0
        end = -1
        for idx, char in enumerate(stripped[start:], start=start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        if end == -1:
            raise ValueError("Unbalanced JSON object in model response.")
        return stripped[start : end + 1]

    @staticmethod
    def _validate_activity_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload.keys()) != REQUIRED_ACTIVITY_KEYS:
            raise ValueError("Activity JSON keys do not match required contract.")
        if not isinstance(payload["activity"], str):
            raise ValueError("activity must be a string.")
        if not isinstance(payload["relative_location"], str):
            raise ValueError("relative_location must be a string.")
        if not isinstance(payload["visible_tools"], list) or not all(
            isinstance(item, str) for item in payload["visible_tools"]
        ):
            raise ValueError("visible_tools must be a list of strings.")
        if not isinstance(payload["object_interaction"], str):
            raise ValueError("object_interaction must be a string.")
        if not isinstance(payload["posture"], str):
            raise ValueError("posture must be a string.")
        if not isinstance(payload["anomaly"], bool):
            raise ValueError("anomaly must be a boolean.")
        if payload["severity"] not in VALID_SEVERITY_VALUES:
            raise ValueError("severity must be null or low|medium|high.")
        return payload

    def _resolve_crop_path(self, crop_path: str, db_parent: Path) -> Path:
        path = Path(crop_path)
        if path.is_absolute():
            return path

        candidates = [Path.cwd() / path, db_parent / path]
        if self.base_dir:
            candidates.append(self.base_dir / path)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return (Path.cwd() / path).resolve()

    def _load_track_crops(
        self,
        conn: sqlite3.Connection,
        *,
        track_id: int,
        db_parent: Path,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT id, timestamp_sec, frame_idx, crop_path
            FROM tracks
            WHERE track_id = ? AND crop_path IS NOT NULL
            ORDER BY timestamp_sec
            """,
            (track_id,),
        ).fetchall()
        sampled_rows = self.select_representative_rows(rows=rows)
        crops: list[dict[str, Any]] = []
        for row_id, timestamp_sec, frame_idx, crop_path in sampled_rows:
            if not crop_path:
                continue
            resolved_path = self._resolve_crop_path(str(crop_path), db_parent=db_parent)
            crops.append(
                {
                    "row_id": int(row_id),
                    "timestamp_sec": float(timestamp_sec),
                    "frame_idx": int(frame_idx),
                    "crop_path": str(crop_path),
                    "resolved_path": resolved_path,
                }
            )
        return crops

    async def _describe_and_parse(
        self, crop: dict[str, Any], prompt: str
    ) -> tuple[dict[str, Any], str]:
        response_text = await self.vllm_client.describe_image_async(
            image_path=crop["resolved_path"], prompt=prompt, max_tokens=self.max_tokens
        )
        json_block = self._extract_json_block(response_text)
        parsed = json.loads(json_block)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed payload is not an object.")
        validated = self._validate_activity_payload(parsed)
        return validated, response_text

    async def _analyze_crop(self, track_id: int, crop: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed, response_text = await self._describe_and_parse(crop=crop, prompt=self.prompt)
            self.parse_successes += 1
            return {
                "track_id": track_id,
                "row_id": crop["row_id"],
                "frame_idx": crop["frame_idx"],
                "timestamp_sec": crop["timestamp_sec"],
                "crop_path": crop["crop_path"],
                "analysis": parsed,
                "raw_response": response_text,
                "used_strict_prompt": False,
                "error": None,
            }
        except Exception as first_error:
            try:
                parsed, response_text = await self._describe_and_parse(
                    crop=crop, prompt=self.strict_prompt
                )
                self.parse_successes += 1
                return {
                    "track_id": track_id,
                    "row_id": crop["row_id"],
                    "frame_idx": crop["frame_idx"],
                    "timestamp_sec": crop["timestamp_sec"],
                    "crop_path": crop["crop_path"],
                    "analysis": parsed,
                    "raw_response": response_text,
                    "used_strict_prompt": True,
                    "error": None,
                }
            except Exception as strict_error:
                self.parse_failures += 1
                return {
                    "track_id": track_id,
                    "row_id": crop["row_id"],
                    "frame_idx": crop["frame_idx"],
                    "timestamp_sec": crop["timestamp_sec"],
                    "crop_path": crop["crop_path"],
                    "analysis": {
                        "parse_error": True,
                        "message": str(strict_error),
                        "raw_snippet": str(first_error)[:200],
                    },
                    "raw_response": "",
                    "used_strict_prompt": True,
                    "error": str(strict_error),
                }

    async def analyze_track(self, track_id: int, crops_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Analyze a single track from a list of crops."""
        semaphore = asyncio.Semaphore(self.concurrency)

        async def analyze_with_limit(crop: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self._analyze_crop(track_id=track_id, crop=crop)

        tasks = [asyncio.create_task(analyze_with_limit(crop)) for crop in crops_list]
        if not tasks:
            return []
        results = await asyncio.gather(*tasks, return_exceptions=True)

        normalized: list[dict[str, Any]] = []
        for crop, result in zip(crops_list, results):
            if isinstance(result, Exception):
                self.parse_failures += 1
                normalized.append(
                    {
                        "track_id": track_id,
                        "row_id": crop["row_id"],
                        "frame_idx": crop["frame_idx"],
                        "timestamp_sec": crop["timestamp_sec"],
                        "crop_path": crop["crop_path"],
                        "analysis": {
                            "parse_error": True,
                            "message": str(result),
                            "raw_snippet": "",
                        },
                        "raw_response": "",
                        "used_strict_prompt": False,
                        "error": str(result),
                    }
                )
                continue
            normalized.append(result)
        return normalized

    async def analyze_all_tracks_async(
        self,
        db_path: str | Path,
        dry_run: bool = False,
        on_track_complete: Callable[[dict[str, Any]], None] | None = None,
        on_crop_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Analyze representative crops for all tracks and persist to SQLite."""
        db_path_value = Path(db_path)
        db_parent = db_path_value.resolve().parent
        conn = sqlite3.connect(str(db_path_value))
        try:
            track_rows = conn.execute(
                "SELECT DISTINCT track_id FROM tracks ORDER BY track_id"
            ).fetchall()
            track_ids = [int(row[0]) for row in track_rows]
            self.total_tracks = len(track_ids)
            crops_by_track: dict[int, list[dict[str, Any]]] = {}
            for track_id in track_ids:
                crops_by_track[track_id] = self._load_track_crops(
                    conn=conn,
                    track_id=track_id,
                    db_parent=db_parent,
                )
            self.total_crops = sum(len(crops) for crops in crops_by_track.values())

            analyzed_tracks = 0
            completed_crops = 0
            semantic_started_at = time.perf_counter()
            for index, track_id in enumerate(track_ids, start=1):
                track_started_at = time.perf_counter()
                try:
                    crops = crops_by_track.get(track_id, [])
                    results = await self.analyze_track(track_id=track_id, crops_list=crops)
                    for result in results:
                        completed_crops += 1
                        if on_crop_progress is not None:
                            elapsed_sec = time.perf_counter() - semantic_started_at
                            avg_sec_per_crop = elapsed_sec / completed_crops if completed_crops > 0 else 0.0
                            remaining_crops = max(self.total_crops - completed_crops, 0)
                            on_crop_progress(
                                {
                                    "track_id": track_id,
                                    "index": index,
                                    "total_tracks": self.total_tracks,
                                    "completed_crops": completed_crops,
                                    "total_crops": self.total_crops,
                                    "remaining_crops": remaining_crops,
                                    "elapsed_sec": elapsed_sec,
                                    "avg_sec_per_crop": avg_sec_per_crop,
                                    "eta_sec": avg_sec_per_crop * remaining_crops,
                                    "row_id": int(result["row_id"]),
                                    "status": "ok" if not result.get("error") else "failed",
                                }
                            )
                        if dry_run:
                            continue
                        update_track_activity(
                            conn=conn,
                            row_id=int(result["row_id"]),
                            activity_json=json.dumps(result["analysis"], ensure_ascii=False),
                        )
                    analyzed_tracks += 1
                    if on_track_complete is not None:
                        on_track_complete(
                            {
                                "track_id": track_id,
                                "index": index,
                                "total_tracks": self.total_tracks,
                                "processed_crops": len(crops),
                                "elapsed_sec": time.perf_counter() - track_started_at,
                                "status": "ok",
                            }
                        )
                except Exception:
                    logger.exception("track_analysis_failed", extra={"track_id": track_id})
                    if on_track_complete is not None:
                        on_track_complete(
                            {
                                "track_id": track_id,
                                "index": index,
                                "total_tracks": self.total_tracks,
                                "processed_crops": 0,
                                "elapsed_sec": time.perf_counter() - track_started_at,
                                "status": "failed",
                            }
                        )
                    continue

            return {
                "total_tracks": self.total_tracks,
                "analyzed_tracks": analyzed_tracks,
                "total_crops": self.total_crops,
                "parse_successes": self.parse_successes,
                "parse_failures": self.parse_failures,
                "request_count": self.vllm_client.request_count,
                "average_latency_ms": self.vllm_client.average_latency_ms,
                "total_tokens": self.vllm_client.total_tokens,
                "max_tokens": self.max_tokens,
                "completed_crops": completed_crops,
            }
        finally:
            conn.close()

    def analyze_all_tracks(
        self,
        db_path: str | Path,
        dry_run: bool = False,
        on_track_complete: Callable[[dict[str, Any]], None] | None = None,
        on_crop_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Synchronous wrapper for all-track analysis."""
        return asyncio.run(
            self.analyze_all_tracks_async(
                db_path=db_path,
                dry_run=dry_run,
                on_track_complete=on_track_complete,
                on_crop_progress=on_crop_progress,
            )
        )
