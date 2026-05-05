"""SQLite helpers for track-level and identity-level records."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Initialize schema and return an open SQLite connection."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            timestamp_sec REAL NOT NULL,
            frame_idx INTEGER NOT NULL,
            bbox_x1 REAL NOT NULL,
            bbox_y1 REAL NOT NULL,
            bbox_x2 REAL NOT NULL,
            bbox_y2 REAL NOT NULL,
            confidence REAL NOT NULL,
            color_tag TEXT,
            crop_path TEXT,
            activity_json TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS identities (
            track_id INTEGER PRIMARY KEY,
            color_tag TEXT,
            first_seen_sec REAL,
            last_seen_sec REAL,
            total_frames INTEGER,
            narrative_summary TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            status TEXT NOT NULL,
            task_id TEXT,
            created_at REAL NOT NULL,
            completed_at REAL,
            error TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_track_id ON tracks(track_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_timestamp ON tracks(timestamp_sec)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status)")
    conn.commit()
    return conn


def insert_track(
    conn: sqlite3.Connection,
    track_id: int,
    timestamp_sec: float,
    frame_idx: int,
    bbox: tuple[float, float, float, float],
    confidence: float,
    color_tag: str | None,
    crop_path: str | None,
    activity_json: str = "{}",
) -> None:
    """Insert one tracked detection row."""
    conn.execute(
        """
        INSERT INTO tracks (
            track_id, timestamp_sec, frame_idx, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            confidence, color_tag, crop_path, activity_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            track_id,
            timestamp_sec,
            frame_idx,
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
            confidence,
            color_tag,
            crop_path,
            activity_json,
        ),
    )
    conn.commit()


def upsert_identity(
    conn: sqlite3.Connection,
    track_id: int,
    color_tag: str,
    first_seen_sec: float,
    last_seen_sec: float,
    total_frames: int,
    narrative_summary: str,
) -> None:
    """Insert or update identity aggregates for a tracked person."""
    conn.execute(
        """
        INSERT INTO identities (
            track_id, color_tag, first_seen_sec, last_seen_sec, total_frames, narrative_summary
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(track_id) DO UPDATE SET
            color_tag=excluded.color_tag,
            first_seen_sec=excluded.first_seen_sec,
            last_seen_sec=excluded.last_seen_sec,
            total_frames=excluded.total_frames,
            narrative_summary=excluded.narrative_summary
        """,
        (track_id, color_tag, first_seen_sec, last_seen_sec, total_frames, narrative_summary),
    )
    conn.commit()


def get_tracks_by_id(conn: sqlite3.Connection, track_id: int) -> list[tuple[Any, ...]]:
    """Return all track rows for one identity."""
    cur = conn.execute("SELECT * FROM tracks WHERE track_id = ? ORDER BY timestamp_sec", (track_id,))
    return cur.fetchall()


def get_all_identities(conn: sqlite3.Connection) -> list[tuple[Any, ...]]:
    """Return all identities ordered by track_id."""
    cur = conn.execute("SELECT * FROM identities ORDER BY track_id")
    return cur.fetchall()


def update_track_activity(conn: sqlite3.Connection, row_id: int, activity_json: str) -> None:
    """Update semantic activity JSON for one track row."""
    conn.execute("UPDATE tracks SET activity_json = ? WHERE id = ?", (activity_json, row_id))
    conn.commit()


def upsert_video_start(
    conn: sqlite3.Connection,
    video_id: str,
    url: str,
    task_id: str,
) -> None:
    """Insert or reset a video run to running state."""
    conn.execute(
        """
        INSERT INTO videos (video_id, url, status, task_id, created_at, completed_at, error)
        VALUES (?, ?, 'running', ?, ?, NULL, NULL)
        ON CONFLICT(video_id) DO UPDATE SET
            url=excluded.url,
            status='running',
            task_id=excluded.task_id,
            completed_at=NULL,
            error=NULL
        """,
        (video_id, url, task_id, time.time()),
    )
    conn.commit()


def get_video(conn: sqlite3.Connection, video_id: str) -> dict[str, Any] | None:
    """Return one video row by id, if present."""
    row = conn.execute(
        """
        SELECT video_id, url, status, task_id, created_at, completed_at, error
        FROM videos
        WHERE video_id = ?
        """,
        (video_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "video_id": row[0],
        "url": row[1],
        "status": row[2],
        "task_id": row[3],
        "created_at": row[4],
        "completed_at": row[5],
        "error": row[6],
    }


def set_video_status(
    conn: sqlite3.Connection,
    video_id: str,
    status: str,
    completed_at: float | None = None,
    error: str | None = None,
) -> None:
    """Set terminal or intermediate status fields for a video row."""
    conn.execute(
        """
        UPDATE videos
        SET status = ?, completed_at = ?, error = ?
        WHERE video_id = ?
        """,
        (status, completed_at, error, video_id),
    )
    conn.commit()


def is_video_completed(conn: sqlite3.Connection, video_id: str) -> bool:
    """Return True when video status is completed."""
    row = conn.execute("SELECT status FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    return bool(row and row[0] == "completed")

