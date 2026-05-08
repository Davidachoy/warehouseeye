"""API tests for WarehouseEye FastAPI endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))
import api.main as api_main
from warehouseeye.pipeline.db import init_db, insert_track, update_track_activity, upsert_identity


def _build_benchmark() -> dict[str, Any]:
    return {
        "frames_analyzed": 4,
        "tokens_per_second_avg": 30.1,
        "latency_per_crop_ms": 1000.0,
        "wall_time_sec": 2.5,
        "total_cost_usd": 0.0014,
        "vs_gpt4v_estimated_savings_pct": 35.0,
    }


def _seed_video_db(db_path: Path) -> None:
    conn = init_db(db_path)
    upsert_identity(conn, 1, "orange_vest", 0.0, 6.0, 3, "Orange vest worker moved boxes.")
    upsert_identity(conn, 2, "blue_top", 1.0, 5.0, 2, "Blue top worker stood near aisle.")
    insert_track(
        conn,
        1,
        1.0,
        0,
        (0.0, 0.0, 1.0, 1.0),
        0.9,
        "orange_vest",
        "crop1.jpg",
        activity_json="{}",
    )
    insert_track(
        conn,
        1,
        2.0,
        1,
        (0.0, 0.0, 1.0, 1.0),
        0.9,
        "orange_vest",
        "crop2.jpg",
        activity_json='{"activity":"packing","relative_location":"box area","anomaly":false}',
    )
    insert_track(
        conn,
        1,
        3.0,
        2,
        (0.0, 0.0, 1.0, 1.0),
        0.9,
        "orange_vest",
        "crop3.jpg",
        activity_json='{"activity":"packing","relative_location":"box area","anomaly":true}',
    )
    insert_track(
        conn,
        2,
        4.0,
        3,
        (0.0, 0.0, 1.0, 1.0),
        0.8,
        "blue_top",
        "crop4.jpg",
        activity_json='{"activity":"idle","relative_location":"aisle","anomaly":false}',
    )
    rows = conn.execute("SELECT id, track_id FROM tracks ORDER BY id").fetchall()
    for row_id, track_id in rows:
        if track_id == 1:
            update_track_activity(
                conn,
                row_id=row_id,
                activity_json='{"activity":"packing","relative_location":"box area","anomaly":false}',
            )
    conn.close()


def test_analyze_stream_and_timeline(monkeypatch, tmp_path) -> None:
    data_root = tmp_path / "api_data"

    def fake_run_pipeline_job(
        *,
        video_id: str,
        video_url: str,
        task_id: str,
        data_root: str | Path,
        status_db_path: str | Path,
        emit,
    ) -> dict[str, Any]:
        del video_url, status_db_path
        video_dir = Path(data_root) / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        db_path = video_dir / "warehouseeye.sqlite3"
        _seed_video_db(db_path)
        timeline_path = video_dir / "timeline.json"
        timeline_payload = {
            "video_id": video_id,
            "task_id": task_id,
            "timeline": [{"track_id": 1, "timestamp_sec": 1.0, "frame_idx": 0}],
            "identities": [{"track_id": 1, "color_tag": "orange_vest"}],
        }
        timeline_path.write_text(json.dumps(timeline_payload), encoding="utf-8")
        emit({"stage": "extracting_frames", "percent": 25})
        emit({"stage": "done", "percent": 100})
        return {"timeline_path": str(timeline_path), "db_path": str(db_path), "benchmark": _build_benchmark()}

    monkeypatch.setattr(api_main, "run_pipeline_job", fake_run_pipeline_job)

    api_main.os.environ["WAREHOUSEEYE_DATA_ROOT"] = str(data_root)
    with TestClient(api_main.app) as client:
        response = client.post(
            "/analyze",
            json={"video_url": "https://example.com/video.mp4", "video_id": "warehouse_1"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "started"
        task_id = body["task_id"]

        progress_events: list[dict[str, Any]] = []
        with client.stream("GET", f"/stream/{task_id}") as stream:
            assert stream.status_code == 200
            for raw in stream.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if line.startswith("data: "):
                    progress_events.append(json.loads(line[len("data: ") :]))
                if progress_events and progress_events[-1].get("stage") == "done":
                    break

        assert progress_events
        assert progress_events[-1]["stage"] == "done"

        timeline_response = client.get("/timeline/warehouse_1")
        assert timeline_response.status_code == 200
        assert timeline_response.json()["video_id"] == "warehouse_1"


def test_query_endpoint_variants(tmp_path) -> None:
    data_root = tmp_path / "query_data"
    video_id = "warehouse_1"
    video_dir = data_root / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    _seed_video_db(video_dir / "warehouseeye.sqlite3")

    api_main.os.environ["WAREHOUSEEYE_DATA_ROOT"] = str(data_root)
    with TestClient(api_main.app) as client:
        test_cases = [
            "How many people are there?",
            "What did the person in the orange vest do?",
            "Who spent the most time in the box area?",
            "Are there any anomalies?",
            "Give me the full timeline",
        ]
        for question in test_cases:
            response = client.post("/query", json={"video_id": video_id, "question": question})
            assert response.status_code == 200
            payload = response.json()
            assert "ambiguous" in payload
            assert "alternatives" in payload
            assert "narrative" in payload
            assert "timeline" in payload
