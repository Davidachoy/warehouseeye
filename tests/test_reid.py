"""Tests for Re-ID matching logic."""

from __future__ import annotations

import numpy as np

from warehouseeye.pipeline import db as pipeline_db
from warehouseeye.tracking.reid import ReIDEngine


class _FakeEmbeddingClient:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector

    def compute_embedding(self, _image_path):
        return self.vector


class _FlipAwareEmbeddingClient:
    """Returns different vectors for original vs horizontally flipped crops."""

    def __init__(self, original: np.ndarray, flipped: np.ndarray) -> None:
        self.original = original
        self.flipped = flipped

    def compute_embedding(self, image: np.ndarray) -> np.ndarray:
        # Detect horizontal flip by checking the marker in column 0 of the crop.
        # The np.ndarray annotation is required so ReIDEngine routes the call
        # through _supports_ndarray_input == True (no tempfile roundtrip).
        if int(image[0, 0, 0]) == 1:
            return self.original
        return self.flipped


def test_find_match_returns_none_when_no_candidates(tmp_path) -> None:
    conn = pipeline_db.init_db(tmp_path / "reid_none.sqlite3")
    fake_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    engine = ReIDEngine(_FakeEmbeddingClient(fake_vector), similarity_threshold=0.85)

    result = engine.find_match(new_embedding=fake_vector, db_conn=conn, current_timestamp=10.0)
    assert result is None
    # find_match should still record the attempt for diagnostics.
    rows = conn.execute("SELECT reason, matched FROM reid_attempts").fetchall()
    assert rows == [("no_candidates", 0)]
    conn.close()


def test_find_match_returns_track_id_for_obvious_match(tmp_path) -> None:
    conn = pipeline_db.init_db(tmp_path / "reid_match.sqlite3")
    existing = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    existing /= np.linalg.norm(existing)
    pipeline_db.save_embedding(conn, track_id=42, vector=existing, timestamp=5.0)
    pipeline_db.mark_track_lost(conn, track_id=42, last_seen_timestamp=5.0)

    engine = ReIDEngine(_FakeEmbeddingClient(existing), similarity_threshold=0.8)
    query = np.array([0.99, 0.01, 0.0], dtype=np.float32)
    query /= np.linalg.norm(query)
    matched = engine.find_match(
        new_embedding=query,
        db_conn=conn,
        current_timestamp=9.0,
        frame_idx=7,
        query_tracker_id=99,
        video_id="vidA",
    )

    assert matched == 42
    rows = conn.execute("SELECT state, match_count FROM embeddings WHERE track_id = 42").fetchone()
    assert rows == ("active", 1)
    attempt = conn.execute(
        "SELECT reason, matched, query_tracker_id, candidate_track_id, video_id, frame_idx FROM reid_attempts"
    ).fetchone()
    assert attempt == ("matched", 1, 99, 42, "vidA", 7)
    conn.close()


def test_find_match_logs_below_threshold(tmp_path) -> None:
    conn = pipeline_db.init_db(tmp_path / "reid_below.sqlite3")
    existing = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pipeline_db.save_embedding(conn, track_id=7, vector=existing, timestamp=1.0)
    pipeline_db.mark_track_lost(conn, track_id=7, last_seen_timestamp=1.0)

    engine = ReIDEngine(_FakeEmbeddingClient(existing), similarity_threshold=0.99)
    query = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    matched = engine.find_match(new_embedding=query, db_conn=conn, current_timestamp=2.0)

    assert matched is None
    reason, matched_flag = conn.execute("SELECT reason, matched FROM reid_attempts").fetchone()
    assert reason == "below_threshold"
    assert matched_flag == 0
    conn.close()


def test_aggregate_mean_topk_falls_back_when_few_anchors(tmp_path) -> None:
    conn = pipeline_db.init_db(tmp_path / "reid_topk.sqlite3")
    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pipeline_db.save_embedding(conn, track_id=11, vector=base, timestamp=1.0)
    pipeline_db.mark_track_lost(conn, track_id=11, last_seen_timestamp=1.0)
    pipeline_db.add_anchor(conn, track_id=11, vector=base, timestamp=1.0)
    other = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    other /= np.linalg.norm(other)
    pipeline_db.add_anchor(conn, track_id=11, vector=other, timestamp=2.0)

    engine = ReIDEngine(
        _FakeEmbeddingClient(base),
        similarity_threshold=0.5,
        aggregation="mean_topk",
        topk=10,  # k larger than anchor count -> mean of all
    )
    matched = engine.find_match(new_embedding=base, db_conn=conn, current_timestamp=5.0)
    assert matched == 11
    sim = conn.execute(
        "SELECT best_similarity FROM reid_attempts WHERE matched = 1 LIMIT 1"
    ).fetchone()[0]
    # Mean of [1.0, 0.6] = 0.8 (within float tolerance).
    assert abs(sim - 0.8) < 1e-3
    conn.close()


def test_tta_hflip_returns_unit_norm_vector() -> None:
    original = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    flipped = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    crop = np.zeros((4, 4, 3), dtype=np.uint8)
    crop[0, 0, 0] = 1  # marker so the fake client can distinguish original from flipped

    engine = ReIDEngine(
        _FlipAwareEmbeddingClient(original=original, flipped=flipped),
        tta_hflip=True,
    )
    merged = engine.compute_embedding(crop)
    assert abs(np.linalg.norm(merged) - 1.0) < 1e-5
