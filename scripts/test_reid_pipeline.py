"""Compare tracking metrics with and without Re-ID enabled."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from warehouseeye.pipeline.orchestrator import Orchestrator


def _load_tracks(db_path: Path) -> list[dict[str, float | int]]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """
        SELECT frame_idx, track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2
        FROM tracks
        ORDER BY frame_idx, id
        """
    ).fetchall()
    conn.close()
    return [
        {
            "frame_idx": int(row[0]),
            "track_id": int(row[1]),
            "x1": float(row[2]),
            "y1": float(row[3]),
            "x2": float(row[4]),
            "y2": float(row[5]),
        }
        for row in rows
    ]


def _iou(box_a: dict[str, float | int], box_b: dict[str, float | int]) -> float:
    x_left = max(float(box_a["x1"]), float(box_b["x1"]))
    y_top = max(float(box_a["y1"]), float(box_b["y1"]))
    x_right = min(float(box_a["x2"]), float(box_b["x2"]))
    y_bottom = min(float(box_a["y2"]), float(box_b["y2"]))
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = (float(box_a["x2"]) - float(box_a["x1"])) * (float(box_a["y2"]) - float(box_a["y1"]))
    area_b = (float(box_b["x2"]) - float(box_b["x1"])) * (float(box_b["y2"]) - float(box_b["y1"]))
    union = max(area_a + area_b - inter, 1e-6)
    return inter / union


def _estimate_id_switches(tracks: list[dict[str, float | int]], min_iou: float = 0.3) -> int:
    per_frame: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for item in tracks:
        per_frame[int(item["frame_idx"])].append(item)

    frame_ids = sorted(per_frame.keys())
    switches = 0
    for prev_frame, cur_frame in zip(frame_ids, frame_ids[1:], strict=False):
        prev_items = per_frame[prev_frame]
        cur_items = per_frame[cur_frame]
        for current in cur_items:
            best_iou = 0.0
            best_prev = None
            for previous in prev_items:
                score = _iou(previous, current)
                if score > best_iou:
                    best_iou = score
                    best_prev = previous
            if best_prev is not None and best_iou >= min_iou:
                if int(best_prev["track_id"]) != int(current["track_id"]):
                    switches += 1
    return switches


def _run_once(video_path: str, base_dir: Path, enable_reid: bool) -> dict[str, float | int | str]:
    orchestrator = Orchestrator(base_dir=base_dir)
    db_path = orchestrator.run(video_path, enable_reid=enable_reid)
    tracks = _load_tracks(db_path)
    unique_tracks = len({int(row["track_id"]) for row in tracks})
    id_switches = _estimate_id_switches(tracks)
    return {
        "db_path": str(db_path),
        "unique_tracks": unique_tracks,
        "id_switches": id_switches,
        "reid_matches": int(orchestrator.last_reid_stats.get("match_count", 0.0)),
        "average_similarity": float(orchestrator.last_reid_stats.get("average_similarity", 0.0)),
    }


def _dump_similarity_matrix(db_path: Path, out_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """
        SELECT id, frame_idx, query_tracker_id, candidate_track_id,
               best_similarity, second_best_similarity, threshold,
               num_candidates, num_anchors, matched, reason
        FROM reid_attempts
        ORDER BY id
        """
    ).fetchall()
    conn.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "frame_idx",
        "query_tracker_id",
        "candidate_track_id",
        "best_similarity",
        "second_best_similarity",
        "threshold",
        "num_candidates",
        "num_anchors",
        "matched",
        "reason",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare WarehouseEye tracking with and without Re-ID.")
    parser.add_argument("--video-path", default="data/warehouse_demo_1.mp4")
    parser.add_argument("--base-dir-no-reid", default="data/reid_eval/no_reid")
    parser.add_argument("--base-dir-reid", default="data/reid_eval/reid")
    parser.add_argument("--output-json", default="data/reid_comparison.json")
    parser.add_argument(
        "--dump-similarity-matrix",
        type=Path,
        default=None,
        help="Also write a CSV of every reid_attempts row for offline plotting.",
    )
    args = parser.parse_args()

    no_reid_metrics = _run_once(args.video_path, Path(args.base_dir_no_reid), enable_reid=False)
    reid_metrics = _run_once(args.video_path, Path(args.base_dir_reid), enable_reid=True)

    report = {
        "video_path": args.video_path,
        "without_reid": no_reid_metrics,
        "with_reid": reid_metrics,
        "delta": {
            "id_switches_reduced_by": int(no_reid_metrics["id_switches"]) - int(reid_metrics["id_switches"]),
            "unique_tracks_reduced_by": int(no_reid_metrics["unique_tracks"]) - int(reid_metrics["unique_tracks"]),
            "reid_matches": int(reid_metrics["reid_matches"]),
            "average_similarity": float(reid_metrics["average_similarity"]),
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Re-ID comparison summary")
    print(f"- Video: {args.video_path}")
    print(
        f"- ID switches (without -> with): {no_reid_metrics['id_switches']} -> {reid_metrics['id_switches']}"
    )
    print(
        f"- Unique tracks (without -> with): {no_reid_metrics['unique_tracks']} -> {reid_metrics['unique_tracks']}"
    )
    print(f"- Re-ID matches: {reid_metrics['reid_matches']}")
    print(f"- Average match similarity: {reid_metrics['average_similarity']:.4f}")
    print(f"- Report: {output_path}")

    if args.dump_similarity_matrix is not None:
        count = _dump_similarity_matrix(Path(reid_metrics["db_path"]), args.dump_similarity_matrix)
        print(f"- Similarity matrix: {args.dump_similarity_matrix} ({count} rows)")


if __name__ == "__main__":
    main()
