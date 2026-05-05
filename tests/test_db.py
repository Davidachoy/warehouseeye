"""Tests for SQLite helpers."""

from warehouseeye.pipeline.db import (
    get_all_identities,
    get_tracks_by_id,
    init_db,
    insert_track,
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

