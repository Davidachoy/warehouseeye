"""Replay ReID decisions against existing crops without re-running detection.

Reads `tracks` rows from a finished pipeline DB, regenerates the embedding /
anchor / reid_attempts tables only, and prints a summary so a user can iterate
on threshold / aggregation / TTA / anchor params in seconds instead of
re-running the full pipeline.

Never touches `tracks`, `frames`, `videos`, or `identities` tables.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from warehouseeye.pipeline import db as pipeline_db
from warehouseeye.tracking.reid import ReIDEngine

logger = logging.getLogger(__name__)


def _build_embedder(backend: str) -> Any:
    backend = backend.strip().lower()
    if backend == "osnet":
        from warehouseeye.gpu import OSNetEmbedder

        return OSNetEmbedder()
    if backend == "qwen":
        from warehouseeye.gpu import EmbeddingClient

        return EmbeddingClient()
    raise ValueError(f"unknown backend: {backend}")


def _reset_reid_tables(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM embedding_anchors")
    conn.execute("DELETE FROM embeddings")
    conn.execute("DELETE FROM reid_attempts")
    conn.commit()


def _load_track_groups(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT track_id, timestamp_sec, frame_idx, crop_path
        FROM tracks
        WHERE crop_path IS NOT NULL
        ORDER BY timestamp_sec ASC, id ASC
        """
    ).fetchall()
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for track_id, ts, frame_idx, crop_path in rows:
        grouped[int(track_id)].append(
            {
                "timestamp_sec": float(ts),
                "frame_idx": int(frame_idx),
                "crop_path": str(crop_path),
            }
        )
    groups = []
    for track_id, items in grouped.items():
        items.sort(key=lambda r: r["timestamp_sec"])
        groups.append(
            {
                "track_id": track_id,
                "first_seen": items[0]["timestamp_sec"],
                "last_seen": items[-1]["timestamp_sec"],
                "items": items,
            }
        )
    groups.sort(key=lambda g: g["first_seen"])
    return groups


def _read_crop(crop_path: Path) -> np.ndarray | None:
    if not crop_path.exists():
        return None
    with Image.open(crop_path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _sample_anchor_indices(num_items: int, anchors_max: int) -> list[int]:
    if num_items <= anchors_max:
        return list(range(num_items))
    # Uniform sample across the track's lifetime including endpoints.
    return [round(i * (num_items - 1) / (anchors_max - 1)) for i in range(anchors_max)]


def replay(
    db_path: Path,
    *,
    backend: str,
    threshold: float,
    aggregation: str,
    topk: int,
    tta_hflip: bool,
    max_anchors: int,
    anchor_min_distance: float,
    max_lost_age_sec: float,
    dry_run: bool,
) -> dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(f"db not found: {db_path}")
    conn = pipeline_db.init_db(db_path)
    groups = _load_track_groups(conn)
    logger.info("loaded %d track groups", len(groups))
    video_id = db_path.parent.name

    if dry_run:
        skipped_missing_crops = 0
        for group in groups:
            crop = _read_crop(Path(group["items"][0]["crop_path"]))
            if crop is None or crop.size == 0:
                skipped_missing_crops += 1
        conn.close()
        return {
            "db_path": str(db_path),
            "total_groups": len(groups),
            "processed_groups": 0,
            "skipped_missing_crops": skipped_missing_crops,
            "dry_run": True,
        }

    _reset_reid_tables(conn)
    embedder = _build_embedder(backend)
    engine = ReIDEngine(
        embedder,
        similarity_threshold=threshold,
        max_lost_track_age_sec=max_lost_age_sec,
        max_anchors_per_track=max_anchors,
        anchor_min_distance=anchor_min_distance,
        aggregation=aggregation,  # type: ignore[arg-type]
        topk=topk,
        tta_hflip=tta_hflip,
    )

    processed: list[dict[str, Any]] = []
    skipped_missing_crops = 0
    for group in groups:
        track_id = group["track_id"]
        first_ts = group["first_seen"]
        # Mark previously-processed tracks as lost before the find_match call,
        # so get_anchors_for_lost_tracks returns them as candidates.
        for prior in processed:
            if prior["last_seen"] <= first_ts:
                pipeline_db.mark_track_lost(conn, prior["track_id"], prior["last_seen"])

        first_item = group["items"][0]
        crop = _read_crop(Path(first_item["crop_path"]))
        if crop is None or crop.size == 0:
            skipped_missing_crops += 1
            continue
        try:
            embedding = engine.compute_embedding(crop)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("embedding_failed track=%s err=%s", track_id, exc)
            continue

        engine.find_match(
                embedding,
                db_conn=conn,
                current_timestamp=first_ts,
                frame_idx=first_item["frame_idx"],
                query_tracker_id=track_id,
            video_id=video_id,
        )
        pipeline_db.save_embedding(conn, track_id, embedding, first_ts)
        pipeline_db.add_anchor(
            conn,
            track_id=track_id,
            vector=embedding,
            timestamp=first_ts,
            max_anchors=engine.max_anchors_per_track,
            min_distance=engine.anchor_min_distance,
        )
        # Enrich the gallery with additional pose anchors sampled across this
        # track's lifetime, gated by the engine's novelty threshold.
        indices = _sample_anchor_indices(len(group["items"]), engine.max_anchors_per_track)
        for idx in indices:
            if idx == 0:
                continue
            extra_item = group["items"][idx]
            extra_crop = _read_crop(Path(extra_item["crop_path"]))
            if extra_crop is None:
                continue
            try:
                extra_embedding = engine.compute_embedding(extra_crop)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("extra_embedding_failed track=%s err=%s", track_id, exc)
                continue
            pipeline_db.add_anchor(
                conn,
                track_id=track_id,
                vector=extra_embedding,
                timestamp=extra_item["timestamp_sec"],
                max_anchors=engine.max_anchors_per_track,
                min_distance=engine.anchor_min_distance,
            )
        pipeline_db.mark_track_active(conn, track_id, group["last_seen"])
        processed.append(group)

    # Aggregate stats from reid_attempts.
    stats = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(matched) AS matched,
            AVG(CASE WHEN matched = 1 THEN best_similarity END) AS mean_match_sim,
            AVG(CASE WHEN matched = 0 AND reason = 'below_threshold' THEN best_similarity END) AS mean_rejected_sim
        FROM reid_attempts
        """
    ).fetchone()
    total, matched, mean_match_sim, mean_rejected_sim = stats
    conn.close()

    summary = {
        "db_path": str(db_path),
        "total_groups": len(groups),
        "processed_groups": len(processed),
        "skipped_missing_crops": skipped_missing_crops,
        "reid_attempts_total": int(total or 0),
        "reid_attempts_matched": int(matched or 0),
        "mean_matched_similarity": float(mean_match_sim) if mean_match_sim is not None else None,
        "mean_rejected_similarity": float(mean_rejected_sim) if mean_rejected_sim is not None else None,
        "config": {
            "backend": backend,
            "threshold": threshold,
            "aggregation": aggregation,
            "topk": topk,
            "tta_hflip": tta_hflip,
            "max_anchors": max_anchors,
            "anchor_min_distance": anchor_min_distance,
            "max_lost_age_sec": max_lost_age_sec,
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--backend", default=os.getenv("REID_BACKEND", "osnet"))
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--aggregation", choices=["max", "mean_topk"], default="max")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--max-anchors", type=int, default=5)
    parser.add_argument("--anchor-min-distance", type=float, default=0.15)
    parser.add_argument("--max-lost-age-sec", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    summary = replay(
        args.db,
        backend=args.backend,
        threshold=args.threshold,
        aggregation=args.aggregation,
        topk=args.topk,
        tta_hflip=args.tta_hflip,
        max_anchors=args.max_anchors,
        anchor_min_distance=args.anchor_min_distance,
        max_lost_age_sec=args.max_lost_age_sec,
        dry_run=args.dry_run,
    )

    print("Re-embed summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
