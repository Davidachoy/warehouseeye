"""Timeline builder placeholder for future narrative enrichment."""

from __future__ import annotations

import sqlite3
from typing import Any


def build_timeline_from_db(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return a minimal ordered timeline from track rows."""
    rows = conn.execute(
        "SELECT track_id, timestamp_sec, frame_idx FROM tracks ORDER BY timestamp_sec, track_id"
    ).fetchall()
    return [{"track_id": row[0], "timestamp_sec": row[1], "frame_idx": row[2]} for row in rows]

