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

from PIL import Image, ImageDraw

from api.schemas import ActivityAnalysisSchema
from warehouseeye.gpu.vllm_client import VLLMClient
from warehouseeye.pipeline.db import update_track_activity

logger = logging.getLogger(__name__)

VALID_ACTIVITIES = {
    "walking",
    "standing",
    "handling_object",
    "lifting",
    "interacting",
    "inspecting",
    "idle",
    "other",
}


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
        self.min_gate_width = float(os.getenv("WAREHOUSEEYE_VLM_MIN_WIDTH", "30"))
        self.min_gate_height = float(os.getenv("WAREHOUSEEYE_VLM_MIN_HEIGHT", "60"))

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
        rows: list[tuple[Any, ...]], min_count: int = 2, max_count: int = 3
    ) -> list[tuple[Any, ...]]:
        """Select 2-3 rows with the largest bbox area."""
        if not rows:
            return []
        total = len(rows)
        sample_count = min(max_count, total)
        sample_count = max(sample_count, min(min_count, total))
        ranked = sorted(
            rows,
            key=lambda row: (
                max(float(row[7]) - float(row[5]), 0.0) * max(float(row[8]) - float(row[6]), 0.0),
                -float(row[1]),
            ),
            reverse=True,
        )
        selected = ranked[:sample_count]
        return sorted(selected, key=lambda row: float(row[1]))

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
        candidate = dict(payload)
        if "relative_location" in candidate:
            legacy_activity = str(candidate.get("activity") or "").strip().lower()
            if legacy_activity not in VALID_ACTIVITIES:
                legacy_activity = "other"
            legacy_anomaly = bool(candidate.get("anomaly"))
            legacy_severity = candidate.get("severity")
            candidate = {
                "activity": legacy_activity,
                "activity_description": str(candidate.get("object_interaction") or "Observed person activity."),
                "objects_involved": [
                    str(item)
                    for item in candidate.get("visible_tools", [])
                    if isinstance(item, str) and item.strip()
                ],
                "zone_inference": str(candidate.get("relative_location") or "unknown"),
                "interaction_with_others": None,
                "anomaly_flag": legacy_anomaly,
                "anomaly_reason": str(legacy_severity) if legacy_anomaly and legacy_severity else None,
                "supervisor_attention_recommended": legacy_anomaly,
                "confidence": 0.45,
                "reasoning": "Converted from legacy activity schema.",
            }
        if "activity" in candidate:
            normalized_activity = str(candidate["activity"]).strip().lower()
            if normalized_activity not in VALID_ACTIVITIES:
                candidate["activity"] = "other"
            else:
                candidate["activity"] = normalized_activity
        try:
            candidate["confidence"] = float(candidate.get("confidence", 0.5))
        except (TypeError, ValueError):
            candidate["confidence"] = 0.5
        candidate["confidence"] = min(max(candidate["confidence"], 0.0), 1.0)
        model = ActivityAnalysisSchema.model_validate(candidate)
        return model.model_dump()

    def _resolve_media_path(self, maybe_path: str | None, db_parent: Path) -> Path | None:
        if not maybe_path:
            return None
        path = Path(maybe_path)
        candidates: list[Path]
        if path.is_absolute():
            candidates = [path]
        else:
            candidates = [Path.cwd() / path, db_parent / path]
            if self.base_dir:
                candidates.append(self.base_dir / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _load_track_crops(
        self,
        conn: sqlite3.Connection,
        *,
        track_id: int,
        db_parent: Path,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT id, timestamp_sec, frame_idx, crop_path, frame_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM tracks
            WHERE track_id = ? AND crop_path IS NOT NULL
            ORDER BY timestamp_sec
            """,
            (track_id,),
        ).fetchall()
        sampled_rows = self.select_representative_rows(rows=rows)
        crops: list[dict[str, Any]] = []
        for row in sampled_rows:
            row_id, timestamp_sec, frame_idx, crop_path, frame_path, x1, y1, x2, y2 = row
            resolved_crop_path = self._resolve_media_path(str(crop_path), db_parent=db_parent)
            if resolved_crop_path is None:
                continue
            bbox = (float(x1), float(y1), float(x2), float(y2))
            width = max(bbox[2] - bbox[0], 1.0)
            height = max(bbox[3] - bbox[1], 1.0)
            crops.append(
                {
                    "row_id": int(row_id),
                    "track_id": int(track_id),
                    "timestamp_sec": float(timestamp_sec),
                    "frame_idx": int(frame_idx),
                    "crop_path": str(crop_path),
                    "frame_path": str(frame_path) if frame_path else None,
                    "resolved_crop_path": resolved_crop_path,
                    "resolved_frame_path": self._resolve_media_path(
                        str(frame_path) if frame_path else None, db_parent=db_parent
                    ),
                    "bbox": bbox,
                    "bbox_width": width,
                    "bbox_height": height,
                }
            )
        return crops

    def _insufficient_resolution_payload(self, *, width: float, height: float) -> dict[str, Any]:
        return {
            "activity": "other",
            "activity_description": "insufficient_resolution",
            "objects_involved": [],
            "zone_inference": "unknown",
            "interaction_with_others": None,
            "anomaly_flag": False,
            "anomaly_reason": f"bbox_too_small:{int(width)}x{int(height)}",
            "supervisor_attention_recommended": False,
            "confidence": 0.0,
            "reasoning": "Target bbox is below minimum size threshold; VLM call skipped.",
            "_status": "insufficient_resolution",
        }

    def _artifact_dir(self, track_id: int) -> Path:
        root = self.base_dir if self.base_dir is not None else Path.cwd()
        packet_dir = root / "vlm_packets" / f"track_{track_id:04d}"
        packet_dir.mkdir(parents=True, exist_ok=True)
        return packet_dir

    @staticmethod
    def _expanded_bbox(
        bbox: tuple[float, float, float, float], width: int, height: int, expand_ratio: float = 0.5
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)
        pad_x = box_w * expand_ratio
        pad_y = box_h * expand_ratio
        ex1 = max(0, int(round(x1 - pad_x)))
        ey1 = max(0, int(round(y1 - pad_y)))
        ex2 = min(width, int(round(x2 + pad_x)))
        ey2 = min(height, int(round(y2 + pad_y)))
        ex2 = max(ex1 + 1, ex2)
        ey2 = max(ey1 + 1, ey2)
        return ex1, ey1, ex2, ey2

    def _collect_frame_boxes(
        self, conn: sqlite3.Connection, *, frame_path: str | None, frame_idx: int
    ) -> list[tuple[int, tuple[float, float, float, float]]]:
        if frame_path:
            rows = conn.execute(
                """
                SELECT track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2
                FROM tracks
                WHERE frame_path = ?
                """,
                (frame_path,),
            ).fetchall()
            if rows:
                return [
                    (int(track_id), (float(x1), float(y1), float(x2), float(y2)))
                    for track_id, x1, y1, x2, y2 in rows
                ]
        rows = conn.execute(
            """
            SELECT track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM tracks
            WHERE frame_idx = ?
            """,
            (frame_idx,),
        ).fetchall()
        return [
            (int(track_id), (float(x1), float(y1), float(x2), float(y2)))
            for track_id, x1, y1, x2, y2 in rows
        ]

    @staticmethod
    def _draw_overlay(
        image: Image.Image,
        *,
        boxes: list[tuple[int, tuple[float, float, float, float]]],
        target_track: int,
    ) -> Image.Image:
        copy = image.copy()
        draw = ImageDraw.Draw(copy)
        for box_track_id, box in boxes:
            color = (235, 235, 235) if box_track_id != target_track else (255, 0, 0)
            thickness = 1 if box_track_id != target_track else 3
            x1, y1, x2, y2 = box
            for delta in range(thickness):
                draw.rectangle((x1 - delta, y1 - delta, x2 + delta, y2 + delta), outline=color)
        return copy

    def _find_previous_track_frame(
        self, conn: sqlite3.Connection, crop: dict[str, Any], db_parent: Path
    ) -> tuple[Path, tuple[float, float, float, float]] | None:
        current_ts = float(crop["timestamp_sec"])
        if current_ts < 0.9:
            return None
        target_ts = current_ts - 1.0
        row = conn.execute(
            """
            SELECT frame_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM tracks
            WHERE track_id = ? AND timestamp_sec <= ?
            ORDER BY ABS(timestamp_sec - ?) ASC
            LIMIT 1
            """,
            (int(crop["track_id"]), current_ts - 0.2, target_ts),
        ).fetchone()
        if row is None:
            return None
        frame_path, x1, y1, x2, y2 = row
        resolved = self._resolve_media_path(str(frame_path) if frame_path else None, db_parent=db_parent)
        if resolved is None:
            return None
        return resolved, (float(x1), float(y1), float(x2), float(y2))

    def _build_visual_packet(
        self,
        conn: sqlite3.Connection,
        *,
        crop: dict[str, Any],
        db_parent: Path,
    ) -> list[Path]:
        frame_path = crop.get("resolved_frame_path")
        if frame_path is None:
            return [Path(crop["resolved_crop_path"])]

        packet_dir = self._artifact_dir(int(crop["track_id"]))
        target_row = int(crop["row_id"])
        frame_image = Image.open(frame_path).convert("RGB")
        frame_boxes = self._collect_frame_boxes(
            conn,
            frame_path=str(crop.get("frame_path")) if crop.get("frame_path") else None,
            frame_idx=int(crop["frame_idx"]),
        )
        target_box = tuple(crop["bbox"])

        full_frame_overlay = self._draw_overlay(
            frame_image,
            boxes=frame_boxes,
            target_track=int(crop["track_id"]),
        )
        full_frame_path = packet_dir / f"row_{target_row:06d}_full_frame.jpg"
        full_frame_overlay.save(full_frame_path)

        ex1, ey1, ex2, ey2 = self._expanded_bbox(target_box, frame_image.width, frame_image.height, expand_ratio=0.5)
        expanded_crop = frame_image.crop((ex1, ey1, ex2, ey2))
        expanded_crop_path = packet_dir / f"row_{target_row:06d}_expanded_crop.jpg"
        expanded_crop.save(expanded_crop_path)

        packet_paths = [full_frame_path, expanded_crop_path]
        previous = self._find_previous_track_frame(conn, crop=crop, db_parent=db_parent)
        if previous is not None:
            prev_frame_path, prev_bbox = previous
            prev_image = Image.open(prev_frame_path).convert("RGB")
            prev_overlay = self._draw_overlay(
                prev_image,
                boxes=[(int(crop["track_id"]), prev_bbox)],
                target_track=int(crop["track_id"]),
            )
            prev_path = packet_dir / f"row_{target_row:06d}_previous_frame.jpg"
            prev_overlay.save(prev_path)
            packet_paths.append(prev_path)
        return packet_paths

    async def _describe_and_parse(
        self, crop: dict[str, Any], prompt: str, packet_paths: list[Path]
    ) -> tuple[dict[str, Any], str]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{prompt}\n"
                    f"Track ID: {crop['track_id']}\n"
                    f"Timestamp sec: {crop['timestamp_sec']:.2f}\n"
                    "Use visual evidence from all images."
                ),
            }
        ]
        for path in packet_paths:
            content.append(
                {"type": "image_url", "image_url": {"url": VLLMClient._image_data_url(path)}}
            )
        response_text = await self.vllm_client.chat_completion_async(
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_tokens,
        )
        json_block = self._extract_json_block(response_text)
        parsed = json.loads(json_block)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed payload is not an object.")
        validated = self._validate_activity_payload(parsed)
        return validated, response_text

    async def _analyze_crop(
        self,
        *,
        conn: sqlite3.Connection,
        track_id: int,
        crop: dict[str, Any],
        db_parent: Path,
    ) -> dict[str, Any]:
        if crop["bbox_width"] < self.min_gate_width or crop["bbox_height"] < self.min_gate_height:
            payload = self._insufficient_resolution_payload(
                width=float(crop["bbox_width"]),
                height=float(crop["bbox_height"]),
            )
            return {
                "track_id": track_id,
                "row_id": crop["row_id"],
                "frame_idx": crop["frame_idx"],
                "timestamp_sec": crop["timestamp_sec"],
                "crop_path": crop["crop_path"],
                "frame_path": crop["frame_path"],
                "analysis": payload,
                "raw_response": "",
                "used_strict_prompt": False,
                "error": None,
                "vlm_packet_paths": [],
            }

        packet_paths = self._build_visual_packet(conn, crop=crop, db_parent=db_parent)
        packet_strings = [str(path) for path in packet_paths]
        try:
            parsed, response_text = await self._describe_and_parse(
                crop=crop,
                prompt=self.prompt,
                packet_paths=packet_paths,
            )
            self.parse_successes += 1
            parsed["_status"] = "ok"
            parsed["_vlm_packet_paths"] = packet_strings
            return {
                "track_id": track_id,
                "row_id": crop["row_id"],
                "frame_idx": crop["frame_idx"],
                "timestamp_sec": crop["timestamp_sec"],
                "crop_path": crop["crop_path"],
                "frame_path": crop["frame_path"],
                "analysis": parsed,
                "raw_response": response_text,
                "used_strict_prompt": False,
                "error": None,
                "vlm_packet_paths": packet_strings,
            }
        except Exception as first_error:
            try:
                parsed, response_text = await self._describe_and_parse(
                    crop=crop,
                    prompt=self.strict_prompt,
                    packet_paths=packet_paths,
                )
                self.parse_successes += 1
                parsed["_status"] = "ok"
                parsed["_vlm_packet_paths"] = packet_strings
                return {
                    "track_id": track_id,
                    "row_id": crop["row_id"],
                    "frame_idx": crop["frame_idx"],
                    "timestamp_sec": crop["timestamp_sec"],
                    "crop_path": crop["crop_path"],
                    "frame_path": crop["frame_path"],
                    "analysis": parsed,
                    "raw_response": response_text,
                    "used_strict_prompt": True,
                    "error": None,
                    "vlm_packet_paths": packet_strings,
                }
            except Exception as strict_error:
                self.parse_failures += 1
                return {
                    "track_id": track_id,
                    "row_id": crop["row_id"],
                    "frame_idx": crop["frame_idx"],
                    "timestamp_sec": crop["timestamp_sec"],
                    "crop_path": crop["crop_path"],
                    "frame_path": crop["frame_path"],
                    "analysis": {
                        "activity": "other",
                        "activity_description": "parse_error",
                        "objects_involved": [],
                        "zone_inference": "unknown",
                        "interaction_with_others": None,
                        "anomaly_flag": False,
                        "anomaly_reason": str(strict_error),
                        "supervisor_attention_recommended": False,
                        "confidence": 0.0,
                        "reasoning": f"Parser failed and fallback used: {strict_error}",
                        "_status": "parse_error",
                        "_vlm_packet_paths": packet_strings,
                        "raw_snippet": str(first_error)[:200],
                    },
                    "raw_response": "",
                    "used_strict_prompt": True,
                    "error": str(strict_error),
                    "vlm_packet_paths": packet_strings,
                }

    async def analyze_track(
        self,
        *,
        conn: sqlite3.Connection,
        track_id: int,
        crops_list: list[dict[str, Any]],
        db_parent: Path,
    ) -> list[dict[str, Any]]:
        """Analyze a single track from a list of crops."""
        semaphore = asyncio.Semaphore(self.concurrency)

        async def analyze_with_limit(crop: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self._analyze_crop(
                    conn=conn,
                    track_id=track_id,
                    crop=crop,
                    db_parent=db_parent,
                )

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
                        "frame_path": crop["frame_path"],
                        "analysis": {
                            "activity": "other",
                            "activity_description": "parse_error",
                            "objects_involved": [],
                            "zone_inference": "unknown",
                            "interaction_with_others": None,
                            "anomaly_flag": False,
                            "anomaly_reason": str(result),
                            "supervisor_attention_recommended": False,
                            "confidence": 0.0,
                            "reasoning": "Unexpected exception while analyzing track.",
                            "_status": "parse_error",
                            "_vlm_packet_paths": [],
                        },
                        "raw_response": "",
                        "used_strict_prompt": False,
                        "error": str(result),
                        "vlm_packet_paths": [],
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
                    results = await self.analyze_track(
                        conn=conn,
                        track_id=track_id,
                        crops_list=crops,
                        db_parent=db_parent,
                    )
                    for result in results:
                        completed_crops += 1
                        status_value = str(result.get("analysis", {}).get("_status", "ok"))
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
                                    "status": status_value,
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
