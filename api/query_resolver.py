"""Natural language query resolution for WarehouseEye timelines."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from warehouseeye.gpu.vllm_client import VLLMClient

COLOR_KEYWORDS = ("orange", "yellow", "green", "blue", "red", "black", "white")
GARMENT_KEYWORDS = ("vest", "hoodie", "shirt", "top", "jacket", "bandana")
POSITION_KEYWORDS = ("left", "right", "center", "box", "area", "zone")
ACTIVITY_KEYWORDS = ("walk", "carry", "load", "pack", "idle", "stand", "move", "lift")


def parse_descriptors(question: str) -> dict[str, Any]:
    """Extract basic identity descriptors from natural-language question."""
    lowered = question.lower().strip()
    color = next((word for word in COLOR_KEYWORDS if word in lowered), None)
    garment = next((word for word in GARMENT_KEYWORDS if word in lowered), None)
    position = next((word for word in POSITION_KEYWORDS if word in lowered), None)
    activity = next((word for word in ACTIVITY_KEYWORDS if word in lowered), None)
    return {
        "color": color,
        "garment": garment,
        "position": position,
        "activity_keyword": activity,
        "question": lowered,
    }


def find_matching_tracks(descriptors: dict[str, Any], conn: sqlite3.Connection) -> list[int]:
    """Find candidate tracks by descriptors against identities and activity JSON."""
    rows = conn.execute("SELECT track_id, color_tag FROM identities ORDER BY track_id").fetchall()
    matches: list[int] = []
    for track_id, color_tag in rows:
        color_tag_value = (color_tag or "").lower()
        color_ok = not descriptors.get("color") or descriptors["color"] in color_tag_value
        garment_ok = not descriptors.get("garment") or descriptors["garment"] in color_tag_value
        if color_ok and garment_ok:
            matches.append(int(track_id))

    if matches or not descriptors.get("activity_keyword"):
        return matches

    scored: Counter[int] = Counter()
    activity_rows = conn.execute("SELECT track_id, activity_json FROM tracks ORDER BY timestamp_sec").fetchall()
    for track_id, activity_json in activity_rows:
        if not activity_json:
            continue
        try:
            payload = json.loads(activity_json)
        except json.JSONDecodeError:
            continue
        activity = str(payload.get("activity", "")).lower()
        location = str(payload.get("relative_location", "")).lower()
        if descriptors["activity_keyword"] in activity:
            scored[int(track_id)] += 2
        if descriptors.get("position") and descriptors["position"] in location:
            scored[int(track_id)] += 1
    return [track_id for track_id, _ in scored.most_common()]


def _timeline_entries(conn: sqlite3.Connection, track_id: int | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT track_id, timestamp_sec, frame_idx, color_tag, crop_path, activity_json
        FROM tracks
    """
    args: tuple[Any, ...] = ()
    if track_id is not None:
        sql += " WHERE track_id = ?"
        args = (track_id,)
    sql += " ORDER BY timestamp_sec, track_id"
    rows = conn.execute(sql, args).fetchall()
    timeline: list[dict[str, Any]] = []
    for row in rows:
        try:
            activity = json.loads(row[5] or "{}")
        except json.JSONDecodeError:
            activity = {"raw": row[5]}
        timeline.append(
            {
                "track_id": int(row[0]),
                "timestamp_sec": float(row[1]),
                "frame_idx": int(row[2]),
                "color_tag": row[3],
                "crop_path": row[4],
                "activity": activity,
            }
        )
    return timeline


def _build_narrative_prompt(track_id: int, timeline: list[dict[str, Any]], base_summary: str) -> str:
    activity_tokens = [
        str(entry.get("activity", {}).get("activity", "")).strip()
        for entry in timeline
        if isinstance(entry.get("activity"), dict)
    ]
    top_activities = [token for token, _ in Counter(activity_tokens).most_common(5) if token]
    return (
        "Write a concise warehouse-worker timeline narrative.\n"
        f"Track ID: {track_id}\n"
        f"Identity summary: {base_summary}\n"
        f"Top activities: {', '.join(top_activities) if top_activities else 'unknown'}\n"
        "Keep it factual and short (3-5 sentences)."
    )


def generate_narrative(
    track_id: int,
    conn: sqlite3.Connection,
    vllm_client: VLLMClient | None = None,
) -> str:
    """Generate short narrative for one track from identities and activity rows."""
    identity = conn.execute(
        "SELECT narrative_summary FROM identities WHERE track_id = ?",
        (track_id,),
    ).fetchone()
    base_summary = (identity[0] if identity else "") or f"Track {track_id}"
    timeline = _timeline_entries(conn, track_id=track_id)

    if vllm_client is None:
        return f"{base_summary}. Observed {len(timeline)} timeline events."

    prompt = _build_narrative_prompt(track_id=track_id, timeline=timeline, base_summary=base_summary)
    messages = [{"role": "user", "content": prompt}]
    try:
        text = asyncio.run(vllm_client.chat_completion_async(messages=messages, max_tokens=200))
        asyncio.run(vllm_client.aclose())
        return text.strip()
    except Exception:
        return f"{base_summary}. Observed {len(timeline)} timeline events."


def _load_candidate_crops(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT t.track_id, i.color_tag, MIN(t.crop_path)
        FROM tracks t
        LEFT JOIN identities i ON i.track_id = t.track_id
        WHERE t.crop_path IS NOT NULL
        GROUP BY t.track_id, i.color_tag
        ORDER BY t.track_id
        """
    ).fetchall()
    return [{"track_id": int(row[0]), "color_tag": row[1], "crop_path": row[2]} for row in rows if row[2]]


def _choose_track_with_vlm(
    conn: sqlite3.Connection,
    question: str,
    candidates: list[dict[str, Any]],
) -> int | None:
    if not candidates:
        return None
    try:
        client = VLLMClient()
    except Exception:
        return None

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Choose the best matching track for this user question.\n"
                f"Question: {question}\n"
                "Return JSON only: {\"track_id\": <integer>}."
            ),
        }
    ]
    for candidate in candidates:
        crop_path = Path(candidate["crop_path"])
        if not crop_path.exists():
            continue
        content.append({"type": "text", "text": f"Candidate track_id={candidate['track_id']}"})
        content.append({"type": "image_url", "image_url": {"url": VLLMClient._image_data_url(crop_path)}})

    if len(content) == 1:
        return None

    messages = [{"role": "user", "content": content}]
    try:
        response = asyncio.run(client.chat_completion_async(messages=messages, max_tokens=120))
        asyncio.run(client.aclose())
    except Exception:
        return None

    match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
        return int(payload.get("track_id"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _resolve_aggregate_query(question: str, conn: sqlite3.Connection) -> dict[str, Any] | None:
    lowered = question.lower()
    full_timeline = _timeline_entries(conn)

    if "how many people" in lowered:
        count = conn.execute("SELECT COUNT(*) FROM identities").fetchone()[0]
        return {
            "matched_track_id": None,
            "ambiguous": False,
            "alternatives": [],
            "narrative": f"There are {count} tracked people in this video.",
            "timeline": full_timeline,
        }

    if "most time" in lowered and "box" in lowered:
        seconds_by_track: Counter[int] = Counter()
        for entry in full_timeline:
            activity = entry.get("activity", {})
            location = str(activity.get("relative_location", "")).lower() if isinstance(activity, dict) else ""
            if "box" in location:
                seconds_by_track[int(entry["track_id"])] += 1
        if not seconds_by_track:
            return {
                "matched_track_id": None,
                "ambiguous": False,
                "alternatives": [],
                "narrative": "No track had a clear box-area location signal.",
                "timeline": full_timeline,
            }
        best_track, best_count = seconds_by_track.most_common(1)[0]
        return {
            "matched_track_id": best_track,
            "ambiguous": False,
            "alternatives": [],
            "narrative": f"Track {best_track} spent the most detected time near the box area ({best_count} observations).",
            "timeline": _timeline_entries(conn, track_id=best_track),
        }

    if "anomal" in lowered:
        anomalies = [
            row
            for row in full_timeline
            if isinstance(row.get("activity"), dict) and bool(row["activity"].get("anomaly"))
        ]
        return {
            "matched_track_id": None,
            "ambiguous": False,
            "alternatives": [],
            "narrative": f"Detected {len(anomalies)} anomalous timeline events.",
            "timeline": anomalies,
        }

    if "full timeline" in lowered:
        return {
            "matched_track_id": None,
            "ambiguous": False,
            "alternatives": [],
            "narrative": "Returning the full timeline for this video.",
            "timeline": full_timeline,
        }
    return None


def resolve_query(db_path: str | Path, question: str) -> dict[str, Any]:
    """Resolve a user query against one video's SQLite timeline data."""
    conn = sqlite3.connect(str(db_path))
    try:
        aggregate = _resolve_aggregate_query(question, conn)
        if aggregate is not None:
            return aggregate

        descriptors = parse_descriptors(question)
        matched_tracks = find_matching_tracks(descriptors, conn)
        alternatives = _load_candidate_crops(conn)

        if len(matched_tracks) == 1:
            track_id = matched_tracks[0]
            return {
                "matched_track_id": track_id,
                "ambiguous": False,
                "alternatives": [],
                "narrative": generate_narrative(track_id=track_id, conn=conn),
                "timeline": _timeline_entries(conn, track_id=track_id),
            }

        if len(matched_tracks) > 1:
            return {
                "matched_track_id": None,
                "ambiguous": True,
                "alternatives": [item for item in alternatives if item["track_id"] in set(matched_tracks)],
                "narrative": "Multiple matching tracks found. Please refine your description.",
                "timeline": [],
            }

        selected_track = _choose_track_with_vlm(conn=conn, question=question, candidates=alternatives)
        if selected_track is not None:
            return {
                "matched_track_id": selected_track,
                "ambiguous": False,
                "alternatives": [],
                "narrative": generate_narrative(track_id=selected_track, conn=conn),
                "timeline": _timeline_entries(conn, track_id=selected_track),
            }

        return {
            "matched_track_id": None,
            "ambiguous": True,
            "alternatives": alternatives,
            "narrative": "No confident match found for that description.",
            "timeline": [],
        }
    finally:
        conn.close()
