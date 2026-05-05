"""Build unified visual + audio timeline artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class TimelineBuilder:
    """Create a searchable timeline from tracks and Whisper words."""

    def __init__(self, window_sec: float = 2.0) -> None:
        self.window_sec = window_sec
        self._timeline: list[dict[str, Any]] = []

    @staticmethod
    def _extract_activity(activity_json: str | None) -> str:
        if not activity_json:
            return "unknown"
        try:
            payload = json.loads(activity_json)
        except Exception:
            return "unknown"
        if isinstance(payload, dict):
            activity = payload.get("activity")
            if isinstance(activity, str) and activity.strip():
                return activity.strip()
        return "unknown"

    def _audio_context_for_time(self, timestamp: float, audio_transcript: list[dict[str, Any]]) -> str:
        lower = timestamp - self.window_sec
        upper = timestamp + self.window_sec
        words = [
            str(entry.get("word", "")).strip()
            for entry in audio_transcript
            if float(entry.get("end", -1.0)) >= lower and float(entry.get("start", -1.0)) <= upper
        ]
        clean_words = [word for word in words if word]
        return " ".join(clean_words)

    def build(self, db_path: str | Path, audio_transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build timeline rows grouped by timestamp from tracked detections."""
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                """
                SELECT timestamp_sec, track_id, color_tag, activity_json
                FROM tracks
                ORDER BY timestamp_sec, track_id
                """
            ).fetchall()
        finally:
            conn.close()

        grouped: dict[float, list[dict[str, Any]]] = {}
        for timestamp_sec, track_id, color_tag, activity_json in rows:
            timestamp = float(timestamp_sec)
            grouped.setdefault(timestamp, []).append(
                {
                    "track_id": int(track_id),
                    "activity": self._extract_activity(activity_json),
                    "color": color_tag or "",
                }
            )

        timeline: list[dict[str, Any]] = []
        for timestamp in sorted(grouped.keys()):
            timeline.append(
                {
                    "timestamp": timestamp,
                    "tracks": grouped[timestamp],
                    "audio_context": self._audio_context_for_time(timestamp, audio_transcript),
                }
            )

        self._timeline = timeline
        return timeline

    def build_per_track(self, track_id: int) -> list[dict[str, Any]]:
        """Return only timeline entries associated with one tracked identity."""
        target = int(track_id)
        filtered: list[dict[str, Any]] = []
        for event in self._timeline:
            tracks = [item for item in event.get("tracks", []) if int(item.get("track_id", -1)) == target]
            if tracks:
                filtered.append(
                    {
                        "timestamp": event["timestamp"],
                        "tracks": tracks,
                        "audio_context": event.get("audio_context", ""),
                    }
                )
        return filtered

    def export_json(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._timeline, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def export_summary_text(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for event in self._timeline:
            track_descriptions = [
                f"track={track['track_id']} activity={track['activity']} color={track['color']}"
                for track in event.get("tracks", [])
            ]
            line = (
                f"[{event['timestamp']:.2f}s] "
                + " | ".join(track_descriptions)
                + f" | audio={event.get('audio_context', '')}"
            )
            lines.append(line)

        path.write_text("\n".join(lines), encoding="utf-8")
        return path


def build_rich_timeline_from_db(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return timeline entries plus identity summaries."""
    rows = conn.execute(
        """
        SELECT
            t.track_id,
            t.timestamp_sec,
            t.frame_idx,
            t.bbox_x1,
            t.bbox_y1,
            t.bbox_x2,
            t.bbox_y2,
            t.confidence,
            t.color_tag,
            t.crop_path,
            t.activity_json,
            i.narrative_summary
        FROM tracks t
        LEFT JOIN identities i ON i.track_id = t.track_id
        ORDER BY t.timestamp_sec, t.track_id
        """
    ).fetchall()

    timeline: list[dict[str, Any]] = []
    for row in rows:
        activity_json = row[10] or "{}"
        try:
            activity = json.loads(activity_json)
            if not isinstance(activity, dict):
                activity = {"raw": activity_json}
        except json.JSONDecodeError:
            activity = {"raw": activity_json}
        timeline.append(
            {
                "track_id": row[0],
                "timestamp_sec": row[1],
                "frame_idx": row[2],
                "bbox": [row[3], row[4], row[5], row[6]],
                "confidence": row[7],
                "color_tag": row[8],
                "crop_path": row[9],
                "activity": activity,
                "narrative_summary": row[11],
            }
        )

    identities_rows = conn.execute(
        """
        SELECT track_id, color_tag, first_seen_sec, last_seen_sec, total_frames, narrative_summary
        FROM identities
        ORDER BY track_id
        """
    ).fetchall()
    identities = [
        {
            "track_id": row[0],
            "color_tag": row[1],
            "first_seen_sec": row[2],
            "last_seen_sec": row[3],
            "total_frames": row[4],
            "narrative_summary": row[5],
        }
        for row in identities_rows
    ]
    return {"timeline": timeline, "identities": identities}


def write_timeline_json(payload: dict[str, Any], output_path: str | Path) -> None:
    """Persist timeline payload to JSON."""
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

