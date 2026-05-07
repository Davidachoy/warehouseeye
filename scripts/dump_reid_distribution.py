"""Print histogram of reid_attempts.best_similarity and write CSV.

Splits by matched / reason so the user can see where same-person hits cluster
versus the rejected near-misses that are calibration targets.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path


def _bin(value: float, width: float) -> float:
    return (int(value / width)) * width


def _load_attempts(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """
        SELECT id, video_id, frame_idx, timestamp_sec, query_tracker_id,
               candidate_track_id, best_similarity, second_best_similarity,
               num_candidates, num_anchors, threshold, matched, reason
        FROM reid_attempts
        ORDER BY id
        """
    ).fetchall()
    conn.close()
    cols = [
        "id",
        "video_id",
        "frame_idx",
        "timestamp_sec",
        "query_tracker_id",
        "candidate_track_id",
        "best_similarity",
        "second_best_similarity",
        "num_candidates",
        "num_anchors",
        "threshold",
        "matched",
        "reason",
    ]
    return [dict(zip(cols, r, strict=False)) for r in rows]


def _print_histogram(label: str, values: list[float], bin_width: float) -> None:
    if not values:
        print(f"  ({label}: no data)")
        return
    counts: dict[float, int] = defaultdict(int)
    for v in values:
        counts[_bin(v, bin_width)] += 1
    max_count = max(counts.values())
    print(f"  {label} (n={len(values)}, mean={sum(values) / len(values):.4f}):")
    for bin_start in sorted(counts):
        bar = "#" * int(40 * counts[bin_start] / max_count)
        print(f"    [{bin_start:.2f}-{bin_start + bin_width:.2f}) {counts[bin_start]:5d} {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("data/reid_eval/similarity_distribution.csv"))
    parser.add_argument("--top-rejected", type=int, default=20)
    args = parser.parse_args()

    attempts = _load_attempts(args.db)
    if not attempts:
        print(f"No reid_attempts in {args.db}")
        return

    matched_sims = [a["best_similarity"] for a in attempts if a["matched"] and a["best_similarity"] is not None]
    rejected_sims = [
        a["best_similarity"]
        for a in attempts
        if not a["matched"] and a["reason"] == "below_threshold" and a["best_similarity"] is not None
    ]
    no_candidate_count = sum(1 for a in attempts if a["reason"] == "no_candidates")
    blur_count = sum(1 for a in attempts if a["reason"] == "anchor_rejected_blur")

    print(f"Total attempts: {len(attempts)}")
    print(f"  matched={len(matched_sims)}  below_threshold={len(rejected_sims)}  "
          f"no_candidates={no_candidate_count}  anchor_rejected_blur={blur_count}")

    _print_histogram("matched", matched_sims, args.bin_width)
    _print_histogram("below_threshold (rejected)", rejected_sims, args.bin_width)

    rejected_by_sim = sorted(
        [a for a in attempts if not a["matched"] and a["reason"] == "below_threshold" and a["best_similarity"] is not None],
        key=lambda a: a["best_similarity"],
        reverse=True,
    )
    if rejected_by_sim:
        print(f"\nTop {args.top_rejected} rejected (closest near-misses):")
        for a in rejected_by_sim[: args.top_rejected]:
            print(
                f"  sim={a['best_similarity']:.4f} thr={a['threshold']:.3f} "
                f"frame={a['frame_idx']} q={a['query_tracker_id']} cand={a['candidate_track_id']} "
                f"anchors={a['num_anchors']}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(attempts[0].keys()))
        writer.writeheader()
        writer.writerows(attempts)
    print(f"\nCSV written: {args.out}")


if __name__ == "__main__":
    main()
