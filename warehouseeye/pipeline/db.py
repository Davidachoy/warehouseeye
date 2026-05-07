"""SQLite helpers for track-level and identity-level records."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np


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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            track_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            dimension INTEGER NOT NULL,
            first_seen_sec REAL NOT NULL,
            last_seen_sec REAL NOT NULL,
            state TEXT NOT NULL,
            match_count INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_anchors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            dimension INTEGER NOT NULL,
            timestamp_sec REAL NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_track_id ON tracks(track_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_timestamp ON tracks(timestamp_sec)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_state ON embeddings(state)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_last_seen ON embeddings(last_seen_sec)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_embedding_anchors_track ON embedding_anchors(track_id)")
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


def save_embedding(
    conn: sqlite3.Connection,
    track_id: int,
    vector: np.ndarray,
    timestamp: float,
) -> None:
    """Insert or update an embedding vector for one tracked identity."""
    vector_f32 = np.asarray(vector, dtype=np.float32).reshape(-1)
    conn.execute(
        """
        INSERT INTO embeddings (
            track_id, embedding, dimension, first_seen_sec, last_seen_sec, state, match_count
        ) VALUES (?, ?, ?, ?, ?, 'active', 0)
        ON CONFLICT(track_id) DO UPDATE SET
            embedding=excluded.embedding,
            dimension=excluded.dimension,
            last_seen_sec=excluded.last_seen_sec,
            state='active'
        """,
        (
            track_id,
            vector_f32.tobytes(),
            int(vector_f32.shape[0]),
            timestamp,
            timestamp,
        ),
    )
    conn.commit()


def get_lost_embeddings(
    conn: sqlite3.Connection,
    max_age_sec: float,
    current_timestamp: float | None = None,
) -> list[tuple[int, np.ndarray]]:
    """Return lost-track embeddings newer than current_timestamp-max_age_sec."""
    if current_timestamp is None:
        current_timestamp = time.time()
    cutoff = current_timestamp - max_age_sec
    rows = conn.execute(
        """
        SELECT track_id, embedding, dimension
        FROM embeddings
        WHERE state = 'lost' AND last_seen_sec >= ?
        ORDER BY last_seen_sec DESC
        """,
        (cutoff,),
    ).fetchall()
    results: list[tuple[int, np.ndarray]] = []
    for track_id, blob, dimension in rows:
        vector = np.frombuffer(blob, dtype=np.float32).reshape(int(dimension))
        results.append((int(track_id), vector))
    return results


def mark_track_active(conn: sqlite3.Connection, track_id: int, current_timestamp: float) -> None:
    """Mark one embedding entry as active and refresh last-seen timestamp."""
    conn.execute(
        """
        UPDATE embeddings
        SET state = 'active', last_seen_sec = ?
        WHERE track_id = ?
        """,
        (current_timestamp, track_id),
    )
    conn.commit()


def mark_track_lost(conn: sqlite3.Connection, track_id: int, last_seen_timestamp: float) -> None:
    """Mark one embedding entry as lost at a given timestamp."""
    conn.execute(
        """
        UPDATE embeddings
        SET state = 'lost', last_seen_sec = ?
        WHERE track_id = ?
        """,
        (last_seen_timestamp, track_id),
    )
    conn.commit()


def increment_match_count(conn: sqlite3.Connection, track_id: int) -> None:
    """Increment Re-ID recovery count for one track."""
    conn.execute(
        """
        UPDATE embeddings
        SET match_count = COALESCE(match_count, 0) + 1
        WHERE track_id = ?
        """,
        (track_id,),
    )
    conn.commit()


def add_anchor(
    conn: sqlite3.Connection,
    track_id: int,
    vector: np.ndarray,
    timestamp: float,
    *,
    max_anchors: int = 5,
    min_distance: float = 0.15,
) -> bool:
    """Append a new embedding anchor for one track if it adds enough novelty.

    Drops the request when there is already a near-duplicate anchor for the
    same track (cosine similarity above 1-min_distance), and trims the oldest
    anchor when max_anchors is exceeded so the gallery stays bounded.
    """
    vector_f32 = np.asarray(vector, dtype=np.float32).reshape(-1)
    rows = conn.execute(
        """
        SELECT id, embedding, dimension, timestamp_sec
        FROM embedding_anchors
        WHERE track_id = ?
        ORDER BY timestamp_sec ASC
        """,
        (track_id,),
    ).fetchall()
    for _row_id, blob, dim, _ts in rows:
        existing = np.frombuffer(blob, dtype=np.float32).reshape(int(dim))
        similarity = float(
            np.dot(existing, vector_f32)
            / max(np.linalg.norm(existing) * np.linalg.norm(vector_f32), 1e-9)
        )
        if similarity >= (1.0 - min_distance):
            return False
    conn.execute(
        """
        INSERT INTO embedding_anchors (track_id, embedding, dimension, timestamp_sec)
        VALUES (?, ?, ?, ?)
        """,
        (track_id, vector_f32.tobytes(), int(vector_f32.shape[0]), timestamp),
    )
    if len(rows) + 1 > max_anchors:
        oldest_row_id = rows[0][0]
        conn.execute("DELETE FROM embedding_anchors WHERE id = ?", (oldest_row_id,))
    conn.commit()
    return True


def get_anchors_for_lost_tracks(
    conn: sqlite3.Connection,
    max_age_sec: float,
    current_timestamp: float | None = None,
) -> list[tuple[int, np.ndarray]]:
    """Return (track_id, anchor_vector) pairs for all anchors of lost tracks."""
    if current_timestamp is None:
        current_timestamp = time.time()
    cutoff = current_timestamp - max_age_sec
    rows = conn.execute(
        """
        SELECT a.track_id, a.embedding, a.dimension
        FROM embedding_anchors AS a
        INNER JOIN embeddings AS e ON e.track_id = a.track_id
        WHERE e.state = 'lost' AND e.last_seen_sec >= ?
        ORDER BY a.timestamp_sec DESC
        """,
        (cutoff,),
    ).fetchall()
    results: list[tuple[int, np.ndarray]] = []
    for track_id, blob, dimension in rows:
        vector = np.frombuffer(blob, dtype=np.float32).reshape(int(dimension))
        results.append((int(track_id), vector))
    return results

