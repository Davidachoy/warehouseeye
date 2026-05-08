"""Natural language query resolution for WarehouseEye timelines."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from warehouseeye.gpu.vllm_client import VLLMClient
from warehouseeye.pipeline.activity_adapter import normalize_activity_payload

COLOR_KEYWORDS = ("orange", "yellow", "green", "blue", "red", "black", "white")
GARMENT_KEYWORDS = ("vest", "hoodie", "shirt", "top", "jacket", "bandana")
POSITION_KEYWORDS = ("left", "right", "center", "box", "area", "zone")
ACTIVITY_KEYWORDS = ("walk", "carry", "load", "pack", "idle", "stand", "move", "lift")
MAX_VLLM_IMAGES_PER_PROMPT = 4
PICTURE_KEYWORDS = ("picture", "photo", "image", "keyframe", "frame", "snapshot")
END_OF_VIDEO_KEYWORDS = ("al final", "final del video", "ultimo", "último", "end", "at the end", "last")


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
            payload = normalize_activity_payload(json.loads(activity_json))
        except json.JSONDecodeError:
            continue
        activity = str(payload.get("activity", "")).lower()
        location = str(payload.get("zone_inference", "")).lower()
        if descriptors["activity_keyword"] in activity:
            scored[int(track_id)] += 2
        if descriptors.get("position") and descriptors["position"] in location:
            scored[int(track_id)] += 1
    return [track_id for track_id, _ in scored.most_common()]


def _timeline_entries(conn: sqlite3.Connection, track_id: int | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT track_id, timestamp_sec, frame_idx, color_tag, crop_path, frame_path, activity_json
        FROM tracks
    """
    args: tuple[Any, ...] = ()
    if track_id is not None:
        sql += " WHERE track_id = ?"
        args = (track_id,)
    sql += " ORDER BY timestamp_sec, track_id"
    try:
        rows = conn.execute(sql, args).fetchall()
        has_frame_path = True
    except sqlite3.OperationalError:
        rows = conn.execute(
            sql.replace(", frame_path", ""),
            args,
        ).fetchall()
        has_frame_path = False
    timeline: list[dict[str, Any]] = []
    for row in rows:
        try:
            activity_raw = row[6] if has_frame_path else row[5]
            activity = normalize_activity_payload(json.loads(activity_raw or "{}"))
        except json.JSONDecodeError:
            activity = normalize_activity_payload({"raw": row[6] if has_frame_path else row[5]})
        timeline.append(
            {
                "track_id": int(row[0]),
                "timestamp_sec": float(row[1]),
                "frame_idx": int(row[2]),
                "color_tag": row[3],
                "crop_path": row[4],
                "frame_path": row[5] if has_frame_path else None,
                "vlm_packet_paths": activity.get("_vlm_packet_paths", []),
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


def _resolve_media_path(maybe_path: str | None, db_parent: Path) -> Path | None:
    if not maybe_path:
        return None
    path = Path(maybe_path)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, db_parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _select_keyframes_for_track(conn: sqlite3.Connection, track_id: int, limit: int = 3) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT track_id, timestamp_sec, frame_idx, crop_path, frame_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM tracks
            WHERE track_id = ?
            ORDER BY ((bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)) DESC, timestamp_sec ASC
            LIMIT ?
            """,
            (track_id, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = conn.execute(
            """
            SELECT track_id, timestamp_sec, frame_idx, crop_path, NULL, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM tracks
            WHERE track_id = ?
            ORDER BY ((bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)) DESC, timestamp_sec ASC
            LIMIT ?
            """,
            (track_id, limit),
        ).fetchall()
    return [
        {
            "track_id": int(row[0]),
            "timestamp_sec": float(row[1]),
            "frame_idx": int(row[2]),
            "crop_path": row[3],
            "frame_path": row[4],
            "bbox": (float(row[5]), float(row[6]), float(row[7]), float(row[8])),
        }
        for row in rows
    ]


def _select_end_keyframes_for_track(conn: sqlite3.Connection, track_id: int, limit: int = 3) -> list[dict[str, Any]]:
    """Select latest keyframes for a track, then return chronologically."""
    try:
        rows = conn.execute(
            """
            SELECT track_id, timestamp_sec, frame_idx, crop_path, frame_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM tracks
            WHERE track_id = ?
            ORDER BY timestamp_sec DESC
            LIMIT ?
            """,
            (track_id, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = conn.execute(
            """
            SELECT track_id, timestamp_sec, frame_idx, crop_path, NULL, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM tracks
            WHERE track_id = ?
            ORDER BY timestamp_sec DESC
            LIMIT ?
            """,
            (track_id, limit),
        ).fetchall()
    rows = sorted(rows, key=lambda row: float(row[1]))
    return [
        {
            "track_id": int(row[0]),
            "timestamp_sec": float(row[1]),
            "frame_idx": int(row[2]),
            "crop_path": row[3],
            "frame_path": row[4],
            "bbox": (float(row[5]), float(row[6]), float(row[7]), float(row[8])),
        }
        for row in rows
    ]


def _is_end_of_video_query(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in END_OF_VIDEO_KEYWORDS)


def _annotated_frame_path(
    conn: sqlite3.Connection,
    *,
    keyframe: dict[str, Any],
    db_parent: Path,
) -> Path | None:
    frame_path = _resolve_media_path(keyframe.get("frame_path"), db_parent=db_parent)
    if frame_path is None:
        return _resolve_media_path(keyframe.get("crop_path"), db_parent=db_parent)

    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if keyframe.get("frame_path"):
        others = conn.execute(
            """
            SELECT track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM tracks
            WHERE frame_path = ?
            """,
            (keyframe["frame_path"],),
        ).fetchall()
    else:
        others = []
    target_track = int(keyframe["track_id"])
    for row in others:
        box_track = int(row[0])
        box = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
        color = (235, 235, 235) if box_track != target_track else (255, 0, 0)
        thickness = 1 if box_track != target_track else 3
        for delta in range(thickness):
            draw.rectangle(
                (box[0] - delta, box[1] - delta, box[2] + delta, box[3] + delta),
                outline=color,
            )
    if not others:
        box = keyframe["bbox"]
        for delta in range(3):
            draw.rectangle(
                (box[0] - delta, box[1] - delta, box[2] + delta, box[3] + delta),
                outline=(255, 0, 0),
            )

    out_dir = db_parent / "query_packets" / f"track_{target_track:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frame_{int(keyframe['frame_idx']):05d}.jpg"
    image.save(out_path)
    return out_path


def _choose_track_with_vlm(
    conn: sqlite3.Connection,
    question: str,
    candidates: list[dict[str, Any]],
    db_parent: Path,
    vllm_client: VLLMClient | None = None,
) -> int | None:
    if not candidates:
        return None
    if vllm_client is None:
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
    image_count = 0
    for candidate in candidates:
        if image_count >= MAX_VLLM_IMAGES_PER_PROMPT:
            break
        crop_path = _resolve_media_path(candidate["crop_path"], db_parent=db_parent)
        if crop_path is None:
            continue
        content.append({"type": "text", "text": f"Candidate track_id={candidate['track_id']}"})
        content.append({"type": "image_url", "image_url": {"url": VLLMClient._image_data_url(crop_path)}})
        image_count += 1

    if len(content) == 1:
        return None

    messages = [{"role": "user", "content": content}]
    try:
        response = asyncio.run(vllm_client.chat_completion_async(messages=messages, max_tokens=120))
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


def _build_video_summary(conn: sqlite3.Connection) -> str:
    count_tracks = int(conn.execute("SELECT COUNT(*) FROM identities").fetchone()[0])
    rows = conn.execute(
        "SELECT activity_json FROM tracks WHERE activity_json IS NOT NULL ORDER BY timestamp_sec"
    ).fetchall()
    activity_counts: Counter[str] = Counter()
    anomaly_count = 0
    for (activity_json,) in rows:
        try:
            payload = normalize_activity_payload(json.loads(activity_json or "{}"))
        except json.JSONDecodeError:
            continue
        activity_counts[str(payload.get("activity") or "unknown")] += 1
        if bool(payload.get("anomaly_flag")):
            anomaly_count += 1
    top_activities = ", ".join(name for name, _ in activity_counts.most_common(5)) or "unknown"
    return (
        f"Tracked people: {count_tracks}. "
        f"Top activities: {top_activities}. "
        f"Anomaly-tagged events: {anomaly_count}."
    )


def _extract_target_second(question: str) -> float | None:
    lowered = question.lower()
    minute_match = re.search(r"\b(?:minute|min)\s+(\d{1,3})\b", lowered)
    if minute_match:
        return float(int(minute_match.group(1)) * 60)

    mm_ss_match = re.search(r"\b(\d{1,2})\s*:\s*([0-5]\d)\b", lowered)
    if mm_ss_match:
        minutes = int(mm_ss_match.group(1))
        seconds = int(mm_ss_match.group(2))
        return float((minutes * 60) + seconds)
    return None


def _is_picture_request(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in PICTURE_KEYWORDS) or "show me" in lowered


def _resolve_keyframe_minute_query(
    *,
    conn: sqlite3.Connection,
    question: str,
) -> dict[str, Any] | None:
    if not _is_picture_request(question):
        return None

    full_timeline = _timeline_entries(conn)
    if not full_timeline:
        return {
            "matched_track_id": None,
            "ambiguous": False,
            "alternatives": [],
            "narrative": "No timeline rows are available for this video yet.",
            "timeline": [],
            "intent": "keyframe_lookup",
        }

    descriptors = parse_descriptors(question)
    matched_tracks = find_matching_tracks(descriptors, conn)
    if not matched_tracks:
        top_rows = conn.execute(
            """
            SELECT track_id, COUNT(*) AS c
            FROM tracks
            GROUP BY track_id
            ORDER BY c DESC
            LIMIT 3
            """
        ).fetchall()
        matched_tracks = [int(row[0]) for row in top_rows]
    allowed_tracks = set(matched_tracks)

    target_second = _extract_target_second(question)
    selected_rows: list[dict[str, Any]] = []
    if target_second is not None:
        window_start = max(0.0, target_second - 10.0)
        window_end = target_second + 10.0
        nearby = [
            row
            for row in full_timeline
            if window_start <= float(row.get("timestamp_sec", 0.0)) <= window_end
            and (not allowed_tracks or int(row.get("track_id", -1)) in allowed_tracks)
        ]
        if not nearby:
            wider_start = max(0.0, target_second - 30.0)
            wider_end = target_second + 30.0
            nearby = [
                row
                for row in full_timeline
                if wider_start <= float(row.get("timestamp_sec", 0.0)) <= wider_end
                and (not allowed_tracks or int(row.get("track_id", -1)) in allowed_tracks)
            ]
        nearby.sort(key=lambda row: abs(float(row.get("timestamp_sec", 0.0)) - target_second))
        seen_keys: set[tuple[int, int]] = set()
        for row in nearby:
            crop_path = row.get("crop_path")
            if not crop_path:
                continue
            key = (int(row.get("track_id", -1)), int(row.get("frame_idx", -1)))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected_rows.append(row)
            if len(selected_rows) >= 8:
                break
    else:
        preferred_tracks = matched_tracks[:2]
        if not preferred_tracks:
            preferred_tracks = sorted({int(row.get("track_id", -1)) for row in full_timeline if row.get("track_id") is not None})[:2]
        use_end_keyframes = _is_end_of_video_query(question)
        by_key: dict[tuple[int, int], dict[str, Any]] = {
            (int(row.get("track_id", -1)), int(row.get("frame_idx", -1))): row for row in full_timeline
        }
        for track_id in preferred_tracks:
            keyframes = (
                _select_end_keyframes_for_track(conn, track_id=track_id, limit=4)
                if use_end_keyframes
                else _select_keyframes_for_track(conn, track_id=track_id, limit=4)
            )
            for keyframe in keyframes:
                key = (int(keyframe["track_id"]), int(keyframe["frame_idx"]))
                timeline_row = by_key.get(key)
                if timeline_row is None:
                    continue
                if not timeline_row.get("crop_path"):
                    continue
                selected_rows.append(timeline_row)
            if len(selected_rows) >= 8:
                break

    if not selected_rows:
        if target_second is not None:
            mm = int(target_second // 60)
            ss = int(target_second % 60)
            failure = f"No keyframes found near {mm:02d}:{ss:02d} for the requested person."
        elif _is_end_of_video_query(question):
            failure = "No end-of-video keyframes were found for the requested person."
        else:
            failure = "No keyframes were found for that request."
        return {
            "matched_track_id": None,
            "ambiguous": False,
            "alternatives": _load_candidate_crops(conn),
            "narrative": failure,
            "timeline": [],
            "intent": "keyframe_lookup",
        }

    alternatives = [
        {
            "track_id": int(row.get("track_id", -1)),
            "color_tag": row.get("color_tag"),
            "crop_path": row.get("crop_path"),
        }
        for row in selected_rows
    ]
    if target_second is not None:
        mm = int(target_second // 60)
        ss = int(target_second % 60)
        focus_phrase = f"near {mm:02d}:{ss:02d}"
    elif _is_end_of_video_query(question):
        focus_phrase = "from the end of the video"
    else:
        focus_phrase = "for the requested person"
    tracks_in_result = sorted({int(row.get("track_id", -1)) for row in selected_rows if row.get("track_id") is not None})
    track_label = ", ".join(f"#{track_id}" for track_id in tracks_in_result if track_id >= 0) or "unknown tracks"
    narrative = (
        f"Showing {len(alternatives)} keyframe(s) {focus_phrase} "
        f"for track(s) {track_label}."
    )
    return {
        "matched_track_id": tracks_in_result[0] if len(tracks_in_result) == 1 else None,
        "ambiguous": False,
        "alternatives": alternatives,
        "narrative": narrative,
        "timeline": selected_rows[:30],
        "intent": "keyframe_lookup",
    }


def resolve_conversational_query(
    *,
    conn: sqlite3.Connection,
    question: str,
    db_parent: Path,
    vllm_client: VLLMClient | None = None,
) -> dict[str, Any]:
    full_timeline = _timeline_entries(conn)
    focus_on_end = _is_end_of_video_query(question)
    if vllm_client is None:
        return {
            "matched_track_id": None,
            "ambiguous": False,
            "alternatives": _load_candidate_crops(conn),
            "narrative": "VLM is unavailable; returning rule-based timeline context only.",
            "timeline": full_timeline,
        }

    descriptors = parse_descriptors(question)
    matched_tracks = find_matching_tracks(descriptors, conn)
    if not matched_tracks:
        top_rows = conn.execute(
            """
            SELECT track_id, COUNT(*) AS c
            FROM tracks
            GROUP BY track_id
            ORDER BY c DESC
            LIMIT 3
            """
        ).fetchall()
        matched_tracks = [int(row[0]) for row in top_rows]
    relevant_tracks = matched_tracks[:3]

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are answering a user question about a warehouse video.\n"
                f"Video summary: {_build_video_summary(conn)}\n"
                f"User question: {question}\n"
                "Use the provided keyframes (tagged with track_id) as visual evidence and answer in 2-5 sentences."
            ),
        }
    ]
    image_count = 0
    for track_id in relevant_tracks:
        if image_count >= MAX_VLLM_IMAGES_PER_PROMPT:
            break
        keyframes = (
            _select_end_keyframes_for_track(conn, track_id=track_id, limit=3)
            if focus_on_end
            else _select_keyframes_for_track(conn, track_id=track_id, limit=3)
        )
        for keyframe in keyframes:
            if image_count >= MAX_VLLM_IMAGES_PER_PROMPT:
                break
            annotated = _annotated_frame_path(conn, keyframe=keyframe, db_parent=db_parent)
            if annotated is None:
                continue
            content.append(
                {
                    "type": "text",
                    "text": f"Keyframe track_id={track_id}, timestamp={keyframe['timestamp_sec']:.2f}s",
                }
            )
            content.append(
                {"type": "image_url", "image_url": {"url": VLLMClient._image_data_url(annotated)}}
            )
            image_count += 1

    if len(content) == 1:
        return {
            "matched_track_id": None,
            "ambiguous": True,
            "alternatives": _load_candidate_crops(conn),
            "narrative": "No visual keyframes were available for conversational answering.",
            "timeline": [],
        }

    try:
        response = asyncio.run(
            vllm_client.chat_completion_async(
                messages=[{"role": "user", "content": content}],
                max_tokens=260,
            )
        )
    except Exception:
        response = "Unable to query the VLM for this question right now."
    selected_timeline = [
        row for row in full_timeline if int(row.get("track_id", -1)) in set(relevant_tracks)
    ]
    return {
        "matched_track_id": relevant_tracks[0] if len(relevant_tracks) == 1 else None,
        "ambiguous": False,
        "alternatives": _load_candidate_crops(conn),
        "narrative": response.strip(),
        "timeline": (
            sorted(selected_timeline, key=lambda row: float(row.get("timestamp_sec", 0.0)), reverse=True)[:40]
            if focus_on_end
            else selected_timeline
        ),
    }


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
            location = str(activity.get("zone_inference", "")).lower() if isinstance(activity, dict) else ""
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
            if isinstance(row.get("activity"), dict)
            and (bool(row["activity"].get("anomaly_flag")) or bool(row["activity"].get("anomaly")))
        ]
        return {
            "matched_track_id": None,
            "ambiguous": False,
            "alternatives": [],
            "narrative": f"Detected {len(anomalies)} anomalous timeline events.",
            "timeline": anomalies,
        }

    if "timeline" in lowered:
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
        keyframe_lookup = _resolve_keyframe_minute_query(conn=conn, question=question)
        if keyframe_lookup is not None:
            return keyframe_lookup

        db_parent = Path(db_path).resolve().parent
        try:
            client = VLLMClient()
        except Exception:
            client = None
        try:
            conversational = resolve_conversational_query(
                conn=conn,
                question=question,
                db_parent=db_parent,
                vllm_client=client,
            )
            return conversational
        finally:
            # Avoid closing this async client via a different event loop;
            # this sync resolver may call asyncio.run multiple times.
            # Process lifetime cleanup is acceptable for this short-lived path.
            pass
    finally:
        conn.close()
