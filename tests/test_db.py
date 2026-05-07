"""Tests for SQLite helpers."""

import numpy as np

from warehouseeye.pipeline.db import (
    get_all_identities,
    get_lost_embeddings,
    get_tracks_by_id,
    increment_match_count,
    init_db,
    insert_track,
    mark_track_lost,
    save_embedding,
    update_track_activity,
    upsert_identity,
)


def test_init_and_insert_track(tmp_path) -> None:
    db_path = tmp_path / "test.sqlite3"
    conn = init_db(db_path)
    insert_track(
        conn=conn,
        track_id=1,
        timestamp_sec=1.5,
        frame_idx=2,
        bbox=(10.0, 20.0, 30.0, 40.0),
        confidence=0.9,
        color_tag="orange_vest",
        crop_path="crop.jpg",
        activity_json="{}",
    )
    rows = get_tracks_by_id(conn, 1)
    assert len(rows) == 1
    assert rows[0][1] == 1
    conn.close()


def test_upsert_identity(tmp_path) -> None:
    db_path = tmp_path / "test.sqlite3"
    conn = init_db(db_path)
    upsert_identity(conn, 7, "blue_top", 1.0, 5.0, 4, "summary")
    upsert_identity(conn, 7, "blue_top", 1.0, 7.0, 6, "summary2")
    rows = get_all_identities(conn)
    assert len(rows) == 1
    assert rows[0][0] == 7
    assert rows[0][4] == 6
    conn.close()


def test_update_track_activity(tmp_path) -> None:
    db_path = tmp_path / "test.sqlite3"
    conn = init_db(db_path)
    insert_track(
        conn=conn,
        track_id=2,
        timestamp_sec=2.0,
        frame_idx=5,
        bbox=(1.0, 2.0, 3.0, 4.0),
        confidence=0.8,
        color_tag="yellow_vest",
        crop_path="crop2.jpg",
        activity_json="{}",
    )
    rows = get_tracks_by_id(conn, 2)
    row_id = rows[0][0]
    update_track_activity(conn=conn, row_id=row_id, activity_json='{"activity":"packing"}')
    updated = get_tracks_by_id(conn, 2)
    assert updated[0][11] == '{"activity":"packing"}'
    conn.close()


def test_embedding_lifecycle_helpers(tmp_path) -> None:
    db_path = tmp_path / "test_embeddings.sqlite3"
    conn = init_db(db_path)
    vector = np.ones(1536, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    save_embedding(conn, track_id=99, vector=vector, timestamp=10.0)
    mark_track_lost(conn, track_id=99, last_seen_timestamp=11.0)

    lost_rows = get_lost_embeddings(conn, max_age_sec=30, current_timestamp=20.0)
    assert len(lost_rows) == 1
    assert lost_rows[0][0] == 99
    assert lost_rows[0][1].shape == (1536,)

    increment_match_count(conn, track_id=99)
    row = conn.execute("SELECT match_count FROM embeddings WHERE track_id = 99").fetchone()
    assert row is not None
    assert row[0] == 1
    conn.close()

