"""Build cached annotated demo videos with track ID overlays."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency on Streamlit Cloud
    cv2 = None  # type: ignore[assignment]

try:
    import ffmpeg
except Exception:  # pragma: no cover - optional dep at runtime
    ffmpeg = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

MAX_SEGMENT_HOLD_SEC = 0.6
MAX_TERMINAL_HOLD_SEC = 0.25


@dataclass(frozen=True)
class TrackSegment:
    track_id: int
    start_sec: float
    end_sec: float
    bbox: tuple[int, int, int, int]
    anomaly: bool


def _clamp_bbox(
    bbox_raw: list[float] | tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1_f, y1_f, x2_f, y2_f = bbox_raw
    x1 = max(0, min(frame_width - 1, int(round(x1_f))))
    y1 = max(0, min(frame_height - 1, int(round(y1_f))))
    x2 = max(x1 + 1, min(frame_width - 1, int(round(x2_f))))
    y2 = max(y1 + 1, min(frame_height - 1, int(round(y2_f))))
    return (x1, y1, x2, y2)


def _track_color(track_id: int, anomaly: bool) -> tuple[int, int, int]:
    if anomaly:
        return (40, 40, 235)  # Red in BGR.
    palette = [
        (255, 170, 0),
        (0, 200, 255),
        (80, 220, 100),
        (255, 120, 210),
        (180, 120, 255),
        (80, 255, 220),
    ]
    return palette[track_id % len(palette)]


def _derive_output_path(video_path: Path, video_id: str) -> Path:
    if video_path.parent.name == video_id:
        artifact_root = video_path.parent
    else:
        artifact_root = video_path.parent / video_id
    return artifact_root / "videos" / "annotated_ids.mp4"


def _timeline_signature(timeline_rows: list[dict[str, Any]]) -> str:
    compact_rows: list[dict[str, Any]] = []
    for row in timeline_rows:
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        compact_rows.append(
            {
                "track_id": int(row.get("track_id", -1)),
                "timestamp_sec": float(row.get("timestamp_sec", 0.0)),
                "bbox": [round(float(value), 3) for value in bbox],
                "anomaly": bool(isinstance(row.get("activity"), dict) and row.get("activity", {}).get("anomaly")),
            }
        )
    payload = json.dumps(
        {
            "rows": compact_rows,
            "max_segment_hold_sec": MAX_SEGMENT_HOLD_SEC,
            "max_terminal_hold_sec": MAX_TERMINAL_HOLD_SEC,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _build_segments(
    timeline_rows: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> list[TrackSegment]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in timeline_rows:
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        track_id = int(row.get("track_id", -1))
        grouped.setdefault(track_id, []).append(row)

    timestamps = sorted(
        {
            float(row.get("timestamp_sec", 0.0))
            for row in timeline_rows
            if isinstance(row.get("bbox"), list) and len(row.get("bbox")) == 4
        }
    )
    if len(timestamps) > 1:
        deltas = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
        default_hold_sec = (sum(deltas) / len(deltas)) if deltas else 0.5
    else:
        default_hold_sec = 0.5
    default_hold_sec = min(default_hold_sec, MAX_SEGMENT_HOLD_SEC)

    segments: list[TrackSegment] = []
    for track_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: float(item.get("timestamp_sec", 0.0)))
        for index, row in enumerate(ordered):
            start_sec = float(row.get("timestamp_sec", 0.0))
            is_terminal_row = index + 1 >= len(ordered)
            if is_terminal_row:
                next_sec = start_sec + min(default_hold_sec, MAX_TERMINAL_HOLD_SEC)
            else:
                next_sec = float(ordered[index + 1].get("timestamp_sec", start_sec + default_hold_sec))
            next_sec = min(next_sec, start_sec + MAX_SEGMENT_HOLD_SEC)
            clamped = _clamp_bbox(
                bbox_raw=row["bbox"],
                frame_width=frame_width,
                frame_height=frame_height,
            )
            anomaly = bool(isinstance(row.get("activity"), dict) and row.get("activity", {}).get("anomaly"))
            segments.append(
                TrackSegment(
                    track_id=track_id,
                    start_sec=start_sec,
                    end_sec=max(start_sec + 0.05, next_sec),
                    bbox=clamped,
                    anomaly=anomaly,
                )
            )
    return sorted(segments, key=lambda item: item.start_sec)


def _draw_segment(frame: Any, segment: TrackSegment) -> None:
    color = _track_color(segment.track_id, segment.anomaly)
    x1, y1, x2, y2 = segment.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID {segment.track_id}"
    if segment.anomaly:
        label = f"{label} !"
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    plate_top = max(0, y1 - label_h - 10)
    cv2.rectangle(frame, (x1, plate_top), (x1 + label_w + 10, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 5, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )


def ensure_annotated_video(video_path: Path, video_id: str, timeline_rows: list[dict[str, Any]]) -> Path | None:
    """Create/reuse an annotated video with ID boxes and return its path."""
    # Streamlit Cloud runtime can miss system libs required by OpenCV (e.g. libGL).
    # Fall back to original video in that case instead of crashing at import/runtime.
    if cv2 is None:
        logger.warning("annotated_video_unavailable: OpenCV import failed; using original video")
        return None

    if not video_path.exists() or not timeline_rows:
        return None

    output_path = _derive_output_path(video_path=video_path, video_id=video_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".meta.json")

    signature = _timeline_signature(timeline_rows)
    source_mtime = video_path.stat().st_mtime

    if output_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        if (
            metadata.get("timeline_signature") == signature
            and float(metadata.get("source_mtime", -1.0)) == source_mtime
        ):
            return output_path

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    intermediate_path = output_path.with_suffix(".mp4v.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(intermediate_path), fourcc, fps, (frame_width, frame_height))
    segments = _build_segments(
        timeline_rows=timeline_rows,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        current_sec = frame_index / fps
        for segment in segments:
            if segment.start_sec <= current_sec < segment.end_sec:
                _draw_segment(frame, segment)
        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()

    if not intermediate_path.exists():
        return None

    final_ok = _transcode_to_h264(intermediate_path, output_path)
    if not final_ok:
        # Fall back to the mp4v file (some browsers may still play it on macOS).
        try:
            if output_path.exists():
                output_path.unlink()
            shutil.move(str(intermediate_path), str(output_path))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("annotated_video_move_failed: %s", exc)
            return None
    else:
        try:
            intermediate_path.unlink(missing_ok=True)
        except Exception:  # pragma: no cover - cleanup best-effort
            pass

    metadata_path.write_text(
        json.dumps(
            {
                "source_path": str(video_path),
                "source_mtime": source_mtime,
                "timeline_signature": signature,
                "max_segment_hold_sec": MAX_SEGMENT_HOLD_SEC,
                "max_terminal_hold_sec": MAX_TERMINAL_HOLD_SEC,
                "h264_transcoded": bool(final_ok),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path if output_path.exists() else None


def _transcode_to_h264(source: Path, target: Path) -> bool:
    """Re-encode an mp4v MP4 into H.264 so HTML5 <video> can play it reliably."""
    if ffmpeg is None:
        logger.warning("annotated_video_transcode_skipped: ffmpeg-python not installed")
        return False
    try:
        if target.exists():
            target.unlink()
        (
            ffmpeg.input(str(source))
            .output(
                str(target),
                vcodec="libx264",
                pix_fmt="yuv420p",
                preset="veryfast",
                movflags="+faststart",
                an=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )
        return target.exists()
    except Exception as exc:
        logger.warning("annotated_video_transcode_failed: %s", exc)
        return False
