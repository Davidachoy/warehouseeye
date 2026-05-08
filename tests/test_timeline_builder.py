"""Tests for unified timeline building."""

from __future__ import annotations

import json

from warehouseeye.pipeline.db import init_db, insert_track
from warehouseeye.pipeline.timeline_builder import TimelineBuilder


def _seed_tracks(db_path) -> None:
    conn = init_db(db_path)
    insert_track(
        conn=conn,
        track_id=7,
        timestamp_sec=12.0,
        frame_idx=1,
        bbox=(1.0, 2.0, 3.0, 4.0),
        confidence=0.91,
        color_tag="orange_vest",
        crop_path="track7.jpg",
        activity_json='{"activity":"packing"}',
    )
    insert_track(
        conn=conn,
        track_id=3,
        timestamp_sec=12.0,
        frame_idx=1,
        bbox=(1.0, 2.0, 3.0, 4.0),
        confidence=0.88,
        color_tag="black_hoodie",
        crop_path="track3.jpg",
        activity_json='{"parse_error":true}',
    )
    insert_track(
        conn=conn,
        track_id=7,
        timestamp_sec=25.0,
        frame_idx=2,
        bbox=(1.0, 2.0, 3.0, 4.0),
        confidence=0.92,
        color_tag="orange_vest",
        crop_path="track7_2.jpg",
        activity_json='{"activity":"loading pallet"}',
    )
    conn.close()


def test_build_timeline_merges_visual_and_audio(tmp_path) -> None:
    db_path = tmp_path / "test.sqlite3"
    _seed_tracks(db_path)
    words = [
        {"word": "dispatch", "start": 11.3, "end": 11.7, "confidence": 0.8},
        {"word": "zone", "start": 11.7, "end": 11.9, "confidence": 0.7},
        {"word": "alert", "start": 12.1, "end": 12.3, "confidence": 0.9},
        {"word": "forklift", "start": 24.4, "end": 24.9, "confidence": 0.6},
    ]

    builder = TimelineBuilder(window_sec=2.0)
    timeline = builder.build(db_path=db_path, audio_transcript=words)

    assert len(timeline) == 2
    assert timeline[0]["timestamp"] == 12.0
    assert timeline[0]["audio_context"] == "dispatch zone alert"
    assert timeline[0]["tracks"][0]["track_id"] == 3
    assert timeline[0]["tracks"][0]["activity"] == "other"
    assert timeline[0]["tracks"][1]["track_id"] == 7
    assert timeline[0]["tracks"][1]["activity"] == "handling_object"
    assert timeline[1]["audio_context"] == "forklift"


def test_build_per_track_and_exports(tmp_path) -> None:
    db_path = tmp_path / "test.sqlite3"
    _seed_tracks(db_path)
    builder = TimelineBuilder()
    builder.build(db_path=db_path, audio_transcript=[])

    per_track = builder.build_per_track(track_id=7)
    assert len(per_track) == 2
    assert all(len(event["tracks"]) == 1 for event in per_track)
    assert all(event["tracks"][0]["track_id"] == 7 for event in per_track)

    json_path = builder.export_json(tmp_path / "timeline.json")
    summary_path = builder.export_summary_text(tmp_path / "timeline.txt")
    assert json_path.exists()
    assert summary_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert "track=7" in summary_path.read_text(encoding="utf-8")
