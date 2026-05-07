"""Persistent Re-ID engine for recovering lost tracker identities."""

from __future__ import annotations

import inspect
import logging
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from warehouseeye.pipeline import db as pipeline_db

logger = logging.getLogger(__name__)

Aggregation = Literal["max", "mean_topk"]


class ReIDEngine:
    """Compute embeddings and match new tracks against lost identities."""

    def __init__(
        self,
        embedding_client: Any,
        similarity_threshold: float = 0.9,
        max_lost_track_age_sec: float = 300.0,
        max_anchors_per_track: int = 5,
        anchor_min_distance: float = 0.15,
        aggregation: Aggregation = "max",
        topk: int = 3,
        tta_hflip: bool = False,
    ) -> None:
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self.max_lost_track_age_sec = max_lost_track_age_sec
        # Multi-anchor gallery: keep up to N pose-diverse embeddings per track,
        # match against any of them. Helps when the tracker creates separate ids
        # for the same person captured in very different poses.
        self.max_anchors_per_track = max(1, int(max_anchors_per_track))
        self.anchor_min_distance = max(0.0, min(0.99, float(anchor_min_distance)))
        if aggregation not in ("max", "mean_topk"):
            raise ValueError(f"unknown reid aggregation: {aggregation}")
        self.aggregation: Aggregation = aggregation
        self.topk = max(1, int(topk))
        self.tta_hflip = bool(tta_hflip)
        self.last_match_similarity: float | None = None
        self.match_similarities: list[float] = []
        # Detect backends that natively accept numpy crops to avoid tempfile IO.
        self._supports_ndarray_input = self._embedder_supports_ndarray(embedding_client)

    @staticmethod
    def _embedder_supports_ndarray(embedding_client: Any) -> bool:
        compute = getattr(embedding_client, "compute_embedding", None)
        if compute is None:
            return False
        try:
            signature = inspect.signature(compute)
        except (TypeError, ValueError):
            return False
        params = list(signature.parameters.values())
        if not params:
            return False
        annotation = params[0].annotation
        if annotation is inspect.Parameter.empty:
            return False
        annotation_text = str(annotation)
        return "ndarray" in annotation_text or "np.ndarray" in annotation_text

    def _embed_one(self, crop_image_array: np.ndarray) -> np.ndarray:
        if self._supports_ndarray_input:
            return self.embedding_client.compute_embedding(crop_image_array)
        image = Image.fromarray(crop_image_array.astype(np.uint8), mode="RGB")
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                temp_path = Path(tmp_file.name)
            image.save(temp_path)
            return self.embedding_client.compute_embedding(temp_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def compute_embedding(self, crop_image_array: np.ndarray) -> np.ndarray:
        """Compute one normalized embedding from an RGB crop array.

        With ``tta_hflip`` enabled, also embeds the horizontal flip and returns
        the L2-renormalized mean of the two vectors.
        """
        base = self._embed_one(crop_image_array)
        if not self.tta_hflip:
            return base
        flipped = np.ascontiguousarray(crop_image_array[:, ::-1])
        mirror = self._embed_one(flipped)
        base = np.asarray(base, dtype=np.float32).reshape(-1)
        mirror = np.asarray(mirror, dtype=np.float32).reshape(-1)
        merged = base + mirror
        norm = float(np.linalg.norm(merged))
        if norm < 1e-9:
            return base
        return merged / norm

    def compute_and_save(
        self,
        track_id: int,
        crop_image_array: np.ndarray,
        db_conn: sqlite3.Connection,
        timestamp: float,
    ) -> np.ndarray:
        """Compute a crop embedding and persist it (primary + multi-anchor)."""
        embedding = self.compute_embedding(crop_image_array)
        pipeline_db.save_embedding(db_conn, track_id, embedding, timestamp)
        pipeline_db.add_anchor(
            db_conn,
            track_id=track_id,
            vector=embedding,
            timestamp=timestamp,
            max_anchors=self.max_anchors_per_track,
            min_distance=self.anchor_min_distance,
        )
        return embedding

    def register_anchor(
        self,
        track_id: int,
        embedding: np.ndarray,
        db_conn: sqlite3.Connection,
        timestamp: float,
    ) -> bool:
        """Register an additional pose anchor for an already-known track."""
        return pipeline_db.add_anchor(
            db_conn,
            track_id=track_id,
            vector=embedding,
            timestamp=timestamp,
            max_anchors=self.max_anchors_per_track,
            min_distance=self.anchor_min_distance,
        )

    def _aggregate_per_track(
        self,
        track_ids: np.ndarray,
        similarities: np.ndarray,
    ) -> dict[int, float]:
        """Reduce per-anchor similarities into one score per candidate track."""
        per_track: dict[int, list[float]] = {}
        for tid, sim in zip(track_ids.tolist(), similarities.tolist(), strict=False):
            per_track.setdefault(int(tid), []).append(float(sim))
        if self.aggregation == "max":
            return {tid: max(vals) for tid, vals in per_track.items()}
        # mean_topk: mean of top-k anchors; falls back to mean of all when < k.
        result: dict[int, float] = {}
        for tid, vals in per_track.items():
            sorted_vals = sorted(vals, reverse=True)
            slice_vals = sorted_vals[: self.topk] if len(sorted_vals) >= self.topk else sorted_vals
            result[tid] = float(np.mean(slice_vals))
        return result

    def find_match(
        self,
        new_embedding: np.ndarray,
        db_conn: sqlite3.Connection,
        current_timestamp: float,
        *,
        frame_idx: int | None = None,
        query_tracker_id: int | None = None,
        video_id: str | None = None,
    ) -> int | None:
        """Return best recovered track_id by configured anchor aggregation."""
        anchors = pipeline_db.get_anchors_for_lost_tracks(
            db_conn,
            max_age_sec=self.max_lost_track_age_sec,
            current_timestamp=current_timestamp,
        )
        if not anchors:
            # Fall back to the single primary embedding for backwards compat.
            anchors = pipeline_db.get_lost_embeddings(
                db_conn,
                max_age_sec=self.max_lost_track_age_sec,
                current_timestamp=current_timestamp,
            )
        if not anchors:
            self.last_match_similarity = None
            pipeline_db.log_reid_attempt(
                db_conn,
                video_id=video_id,
                frame_idx=frame_idx,
                timestamp_sec=current_timestamp,
                query_tracker_id=query_tracker_id,
                candidate_track_id=None,
                best_similarity=None,
                second_best_similarity=None,
                num_candidates=0,
                num_anchors=0,
                threshold=self.similarity_threshold,
                matched=False,
                reason="no_candidates",
            )
            return None

        track_ids = np.array([track_id for track_id, _ in anchors], dtype=np.int64)
        anchor_matrix = np.vstack([vector for _, vector in anchors]).astype(np.float32)
        new_embedding = np.asarray(new_embedding, dtype=np.float32).reshape(-1)

        similarities = np.dot(anchor_matrix, new_embedding) / (
            np.linalg.norm(anchor_matrix, axis=1) * np.linalg.norm(new_embedding)
        )
        scores_per_track = self._aggregate_per_track(track_ids, similarities)
        matched_track_id_value = max(scores_per_track, key=scores_per_track.get)  # type: ignore[arg-type]
        best_similarity = scores_per_track[matched_track_id_value]
        sorted_scores = sorted(scores_per_track.values(), reverse=True)
        second_best = float(sorted_scores[1]) if len(sorted_scores) > 1 else None
        anchors_for_best = int(np.sum(track_ids == matched_track_id_value))

        logger.info(
            "Re-ID candidates=%s anchors=%s best_similarity=%.4f threshold=%.2f aggregation=%s",
            len(scores_per_track),
            len(anchors),
            best_similarity,
            self.similarity_threshold,
            self.aggregation,
        )
        if best_similarity < self.similarity_threshold:
            self.last_match_similarity = None
            pipeline_db.log_reid_attempt(
                db_conn,
                video_id=video_id,
                frame_idx=frame_idx,
                timestamp_sec=current_timestamp,
                query_tracker_id=query_tracker_id,
                candidate_track_id=int(matched_track_id_value),
                best_similarity=float(best_similarity),
                second_best_similarity=second_best,
                num_candidates=len(scores_per_track),
                num_anchors=anchors_for_best,
                threshold=self.similarity_threshold,
                matched=False,
                reason="below_threshold",
            )
            return None

        matched_track_id = int(matched_track_id_value)
        pipeline_db.mark_track_active(db_conn, matched_track_id, current_timestamp)
        pipeline_db.increment_match_count(db_conn, matched_track_id)
        self.last_match_similarity = best_similarity
        self.match_similarities.append(best_similarity)
        pipeline_db.log_reid_attempt(
            db_conn,
            video_id=video_id,
            frame_idx=frame_idx,
            timestamp_sec=current_timestamp,
            query_tracker_id=query_tracker_id,
            candidate_track_id=matched_track_id,
            best_similarity=float(best_similarity),
            second_best_similarity=second_best,
            num_candidates=len(scores_per_track),
            num_anchors=anchors_for_best,
            threshold=self.similarity_threshold,
            matched=True,
            reason="matched",
        )
        logger.info(
            "Re-ID match: track %s recovered with similarity %.2f",
            matched_track_id,
            best_similarity,
        )
        return matched_track_id
