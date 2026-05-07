"""Sweep ReID similarity thresholds against IoU pseudo-labels.

For each `reid_attempts` row, pseudo-label as a "true match" when the query
tracker's bbox at `frame_idx` overlaps (IoU >= threshold) the candidate's
last-known bbox up to that point. Then sweep thresholds and report
F1-best, precision>=0.95, and precision>=0.99 cut-offs.

Use the output to set WAREHOUSEEYE_REID_SIMILARITY_THRESHOLD.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = max(area_a + area_b - inter, 1e-6)
    return inter / union


def _load_tracks(conn: sqlite3.Connection) -> dict[int, list[dict]]:
    rows = conn.execute(
        """
        SELECT track_id, frame_idx, timestamp_sec, bbox_x1, bbox_y1, bbox_x2, bbox_y2
        FROM tracks
        ORDER BY track_id, frame_idx
        """
    ).fetchall()
    by_track: dict[int, list[dict]] = defaultdict(list)
    for tid, frame_idx, ts, x1, y1, x2, y2 in rows:
        by_track[int(tid)].append(
            {"frame_idx": int(frame_idx), "ts": float(ts), "bbox": (float(x1), float(y1), float(x2), float(y2))}
        )
    return by_track


def _bbox_at_frame(track_rows: list[dict], frame_idx: int) -> tuple[float, float, float, float] | None:
    # Exact frame.
    for row in track_rows:
        if row["frame_idx"] == frame_idx:
            return row["bbox"]
    # Otherwise nearest prior frame (candidate's last-known position).
    prior = [r for r in track_rows if r["frame_idx"] < frame_idx]
    if prior:
        return prior[-1]["bbox"]
    return None


def _pseudo_labels(
    attempts: list[dict],
    tracks_by_id: dict[int, list[dict]],
    iou_threshold: float,
) -> list[bool]:
    labels: list[bool] = []
    for attempt in attempts:
        query_id = attempt["query_tracker_id"]
        cand_id = attempt["candidate_track_id"]
        frame_idx = attempt["frame_idx"]
        if query_id is None or cand_id is None or frame_idx is None:
            labels.append(False)
            continue
        query_box = _bbox_at_frame(tracks_by_id.get(query_id, []), frame_idx)
        cand_box = _bbox_at_frame(tracks_by_id.get(cand_id, []), frame_idx)
        if query_box is None or cand_box is None:
            labels.append(False)
            continue
        labels.append(_iou(query_box, cand_box) >= iou_threshold)
    return labels


def _metrics(sims: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = sims >= threshold
    tp = int(np.sum(pred & labels))
    fp = int(np.sum(pred & ~labels))
    fn = int(np.sum(~pred & labels))
    tn = int(np.sum(~pred & ~labels))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--out", type=Path, default=Path("data/reid_eval/calibration.json"))
    args = parser.parse_args()

    conn = sqlite3.connect(str(args.db))
    rows = conn.execute(
        """
        SELECT id, frame_idx, query_tracker_id, candidate_track_id, best_similarity, threshold, matched, reason
        FROM reid_attempts
        WHERE candidate_track_id IS NOT NULL AND best_similarity IS NOT NULL
        """
    ).fetchall()
    attempts = [
        {
            "id": r[0],
            "frame_idx": r[1],
            "query_tracker_id": r[2],
            "candidate_track_id": r[3],
            "best_similarity": float(r[4]),
            "threshold": float(r[5]),
            "matched": bool(r[6]),
            "reason": r[7],
        }
        for r in rows
    ]
    if not attempts:
        print(f"No usable reid_attempts in {args.db} (need candidate_track_id + best_similarity)")
        return
    tracks_by_id = _load_tracks(conn)
    conn.close()

    labels = np.array(_pseudo_labels(attempts, tracks_by_id, args.iou_threshold))
    sims = np.array([a["best_similarity"] for a in attempts])
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    print(f"Loaded {len(attempts)} attempts. Pseudo-positives={n_pos} negatives={n_neg} (IoU>={args.iou_threshold})")
    if n_pos == 0:
        print("No pseudo-positive labels; cannot calibrate. Try lowering --iou-threshold.")
        return

    sweep = [_metrics(sims, labels, t) for t in np.arange(0.40, 0.951, 0.01)]
    best_f1 = max(sweep, key=lambda m: m["f1"])
    p95 = next((m for m in sweep if m["precision"] >= 0.95), None)
    p99 = next((m for m in sweep if m["precision"] >= 0.99), None)

    print()
    print(f"F1-best:   threshold={best_f1['threshold']:.2f} P={best_f1['precision']:.3f} R={best_f1['recall']:.3f} F1={best_f1['f1']:.3f}")
    if p95:
        print(f"P>=0.95:   threshold={p95['threshold']:.2f} P={p95['precision']:.3f} R={p95['recall']:.3f}")
    else:
        print("P>=0.95:   not reachable on this data")
    if p99:
        print(f"P>=0.99:   threshold={p99['threshold']:.2f} P={p99['precision']:.3f} R={p99['recall']:.3f}")
    else:
        print("P>=0.99:   not reachable on this data")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "db": str(args.db),
                "iou_threshold": args.iou_threshold,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "f1_best": best_f1,
                "p_ge_0_95": p95,
                "p_ge_0_99": p99,
                "sweep": sweep,
            },
            indent=2,
        )
    )
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
