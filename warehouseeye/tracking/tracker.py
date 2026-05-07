"""Persistent person tracking with supervision ByteTrack."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator, Sequence

import cv2
import numpy as np
from PIL import Image
import supervision as sv

from warehouseeye.pipeline import db as pipeline_db
from warehouseeye.tracking.reid import ReIDEngine
from warehouseeye.tracking.types import BoundingBox

logger = logging.getLogger(__name__)


class PersonTracker:
    """Track detections frame by frame using ByteTrack."""

    def __init__(
        self,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        reid_engine: ReIDEngine | None = None,
        min_reid_crop_area: float = 8000.0,
        min_reid_crop_width: float = 36.0,
        min_reid_crop_height: float = 48.0,
        min_reid_crop_aspect_ratio: float = 0.8,
        max_reid_crop_aspect_ratio: float = 4.5,
        reid_crop_expand_ratio: float = 0.15,
        anchor_min_sharpness: float = 0.0,
        video_id: str | None = None,
        active_anchor_refresh_every: int = 0,
    ) -> None:
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        self.reid_engine = reid_engine
        self.min_reid_crop_area = min_reid_crop_area
        self.min_reid_crop_width = min_reid_crop_width
        self.min_reid_crop_height = min_reid_crop_height
        self.min_reid_crop_aspect_ratio = min_reid_crop_aspect_ratio
        self.max_reid_crop_aspect_ratio = max_reid_crop_aspect_ratio
        # Padding around the bbox so the embedding sees more identity context.
        # Positive values expand outward (more background context, useful when
        # the detector outputs tight crops). Negative values inset inward (less
        # background, useful for warehouse/kitchen scenes where the background
        # changes drastically between frames and dominates the embedding).
        self.reid_crop_expand_ratio = float(reid_crop_expand_ratio)
        self.anchor_min_sharpness = max(0.0, float(anchor_min_sharpness))
        self.video_id = video_id
        # When > 0, recompute the embedding for an already-known active track
        # every N frames and try to register it as an additional anchor. The
        # novelty gate inside add_anchor filters near-duplicates so the gallery
        # still stays bounded by max_anchors_per_track.
        self.active_anchor_refresh_every = max(0, int(active_anchor_refresh_every))
        self._known_ids: set[int] = set()
        self._active_ids: set[int] = set()
        self._active_frames_per_id: dict[int, int] = {}
        # Persistent remap so a tracker_id that already resolved to an older
        # identity stays renamed across all future frames, without re-querying ReID.
        self._reid_remap: dict[int, int] = {}

    def _crop_sharpness(self, crop: np.ndarray) -> float:
        """Return Laplacian-variance focus measure for an RGB crop (uint8)."""
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _is_reid_crop_valid(self, bbox: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        if area < self.min_reid_crop_area:
            return False
        if width < self.min_reid_crop_width or height < self.min_reid_crop_height:
            return False
        aspect_ratio = height / max(width, 1e-6)
        if aspect_ratio < self.min_reid_crop_aspect_ratio or aspect_ratio > self.max_reid_crop_aspect_ratio:
            return False
        return True

    @staticmethod
    def _crop_frame(frame_image: Image.Image, bbox: tuple[float, float, float, float]) -> np.ndarray:
        width, height = frame_image.size
        x1, y1, x2, y2 = bbox
        x1_i = max(0, min(width - 1, int(x1)))
        y1_i = max(0, min(height - 1, int(y1)))
        x2_i = max(x1_i + 1, min(width, int(x2)))
        y2_i = max(y1_i + 1, min(height, int(y2)))
        crop = frame_image.crop((x1_i, y1_i, x2_i, y2_i)).convert("RGB")
        return np.asarray(crop, dtype=np.uint8)

    def _expand_bbox_for_reid(
        self,
        bbox: tuple[float, float, float, float],
        frame_size: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        # Step 1: when the person is bent over (wider than tall), reduce the
        # crop to roughly the upper body so the embedder focuses on head + torso
        # instead of background floor/shelves.
        bbox = self._upper_body_bbox(bbox)

        if self.reid_crop_expand_ratio == 0.0:
            return bbox
        frame_width, frame_height = frame_size
        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        pad_x = width * self.reid_crop_expand_ratio
        pad_y = height * self.reid_crop_expand_ratio
        ex1 = x1 - pad_x
        ey1 = y1 - pad_y
        ex2 = x2 + pad_x
        ey2 = y2 + pad_y
        # Clamp to image bounds for outward expansion; for inward inset the
        # values stay within bounds by construction.
        ex1 = max(0.0, ex1)
        ey1 = max(0.0, ey1)
        ex2 = min(float(frame_width), ex2)
        ey2 = min(float(frame_height), ey2)
        # Reject pathological insets that would leave a tiny crop.
        if ex2 - ex1 < max(8.0, width * 0.25) or ey2 - ey1 < max(8.0, height * 0.25):
            return bbox
        return (ex1, ey1, ex2, ey2)

    @staticmethod
    def _upper_body_bbox(
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        width = max(1e-6, x2 - x1)
        height = max(1e-6, y2 - y1)
        # Only trigger upper-body crop on clearly bent poses (h/w < 1.0).
        if height / width >= 1.0:
            return bbox
        # Keep the top ~55% of the bbox, which empirically covers head+torso
        # when the subject is leaning over a workspace.
        return (x1, y1, x2, y1 + height * 0.55)

    def _yield_tracked_rows(
        self,
        tracked: sv.Detections,
        frame_idx: int,
    ) -> Iterator[tuple[int, int, tuple[float, float, float, float], float]]:
        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else np.array([], dtype=int)
        for idx, tracker_id in enumerate(tracker_ids):
            if tracker_id is None:
                continue
            bbox = tuple(float(v) for v in tracked.xyxy[idx].tolist())
            confidence = float(tracked.confidence[idx]) if tracked.confidence is not None else 0.0
            yield (frame_idx, int(tracker_id), bbox, confidence)

    def update(
        self,
        detections_list: Sequence[BoundingBox],
        frame_idx: int,
        frame_image: Image.Image | None = None,
        timestamp_sec: float | None = None,
        db_conn: sqlite3.Connection | None = None,
    ) -> Iterator[tuple[int, int, tuple[float, float, float, float], float]]:
        """Yield tracked tuples (frame_idx, tracker_id, bbox_xyxy, confidence)."""
        if detections_list:
            bboxes = np.array([[b.x1, b.y1, b.x2, b.y2] for b in detections_list], dtype=float)
            confidences = np.array([b.confidence for b in detections_list], dtype=float)
            class_ids = np.zeros(len(detections_list), dtype=int)
        else:
            bboxes = np.empty((0, 4), dtype=float)
            confidences = np.array([], dtype=float)
            class_ids = np.array([], dtype=int)

        detections = sv.Detections(xyxy=bboxes, confidence=confidences, class_id=class_ids)
        tracked = self.tracker.update_with_detections(detections)
        tracker_ids = tracked.tracker_id

        if self.reid_engine is None:
            yield from self._yield_tracked_rows(tracked=tracked, frame_idx=frame_idx)
            return
        if tracker_ids is None or frame_image is None or timestamp_sec is None or db_conn is None:
            logger.info("reid_disabled_for_frame_missing_context", extra={"frame_idx": frame_idx})
            yield from self._yield_tracked_rows(tracked=tracked, frame_idx=frame_idx)
            return

        # Apply previously learned remaps before evaluating brand-new ids,
        # so a tracker_id that ReID already resolved to an older identity
        # never re-enters the ReID flow on subsequent frames.
        for idx, tracker_id in enumerate(tracker_ids):
            if tracker_id is None:
                continue
            remapped = self._reid_remap.get(int(tracker_id))
            if remapped is not None and remapped != int(tracker_id):
                tracked.tracker_id[idx] = remapped

        tracker_ids = tracked.tracker_id

        current_ids = {
            int(tracker_id)
            for tracker_id in tracker_ids
            if tracker_id is not None
        }
        brand_new_ids = current_ids - self._known_ids
        id_to_indices: dict[int, list[int]] = {}
        for idx, tracker_id in enumerate(tracker_ids):
            if tracker_id is None:
                continue
            id_to_indices.setdefault(int(tracker_id), []).append(idx)

        for tracker_id in sorted(brand_new_ids):
            indices = id_to_indices.get(tracker_id, [])
            if not indices:
                continue
            first_idx = indices[0]
            bbox = tuple(float(v) for v in tracked.xyxy[first_idx].tolist())
            if not self._is_reid_crop_valid(bbox):
                logger.info("Skipping ReID for low-quality bbox: tid=%s bbox=%s", tracker_id, bbox)
                # Even when we skip ReID we still want this tracker_id to be
                # treated as known so we don't retry every frame.
                self._known_ids.add(tracker_id)
                continue
            reid_bbox = self._expand_bbox_for_reid(bbox, frame_image.size)
            crop = self._crop_frame(frame_image, reid_bbox)
            embedding = self.reid_engine.compute_embedding(crop)
            matched_id = self.reid_engine.find_match(
                embedding,
                db_conn=db_conn,
                current_timestamp=timestamp_sec,
                frame_idx=frame_idx,
                query_tracker_id=int(tracker_id),
                video_id=self.video_id,
            )
            if matched_id is not None:
                for idx in indices:
                    tracked.tracker_id[idx] = matched_id
                current_ids.discard(tracker_id)
                current_ids.add(matched_id)
                # Memorize the remap so any future frame with this raw tracker_id
                # is rewritten without calling the embedding endpoint again.
                self._reid_remap[tracker_id] = matched_id
                self._known_ids.add(tracker_id)
                self._known_ids.add(matched_id)
                # Grow the multi-anchor gallery of the recovered identity with
                # this new pose, so future matches against it cover more poses.
                # Gate on bbox quality + sharpness so blurry/off crops do not
                # poison the gallery and bias future MAX-similarity matches.
                anchor_ok = self._is_reid_crop_valid(bbox)
                sharpness = 0.0
                if anchor_ok and self.anchor_min_sharpness > 0.0:
                    sharpness = self._crop_sharpness(crop)
                    if sharpness < self.anchor_min_sharpness:
                        anchor_ok = False
                if anchor_ok:
                    try:
                        self.reid_engine.register_anchor(
                            track_id=matched_id,
                            embedding=embedding,
                            db_conn=db_conn,
                            timestamp=timestamp_sec,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.info("anchor_register_failed tid=%s: %s", matched_id, exc)
                else:
                    pipeline_db.log_reid_attempt(
                        db_conn,
                        video_id=self.video_id,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        query_tracker_id=int(tracker_id),
                        candidate_track_id=matched_id,
                        best_similarity=float(self.reid_engine.last_match_similarity or 0.0),
                        second_best_similarity=None,
                        num_candidates=0,
                        num_anchors=0,
                        threshold=self.reid_engine.similarity_threshold,
                        matched=False,
                        reason="anchor_rejected_blur",
                    )
                    logger.info(
                        "anchor_rejected tid=%s sharpness=%.2f min=%.2f",
                        matched_id,
                        sharpness,
                        self.anchor_min_sharpness,
                    )
                logger.info(
                    "ReID hit: tid=%s recovered as old tid=%s, similarity=%.2f",
                    tracker_id,
                    matched_id,
                    float(self.reid_engine.last_match_similarity or 0.0),
                )
                continue

            self.reid_engine.compute_and_save(
                track_id=tracker_id,
                crop_image_array=crop,
                db_conn=db_conn,
                timestamp=timestamp_sec,
            )
            self._known_ids.add(tracker_id)
            logger.info("New track: tid=%s embedding stored", tracker_id)

        if self.active_anchor_refresh_every > 0:
            for tid in current_ids:
                self._active_frames_per_id[tid] = self._active_frames_per_id.get(tid, 0) + 1
                counter = self._active_frames_per_id[tid]
                if counter <= 1 or counter % self.active_anchor_refresh_every != 0:
                    continue
                indices = id_to_indices.get(tid, [])
                if not indices:
                    continue
                bbox = tuple(float(v) for v in tracked.xyxy[indices[0]].tolist())
                if not self._is_reid_crop_valid(bbox):
                    continue
                reid_bbox = self._expand_bbox_for_reid(bbox, frame_image.size)
                crop = self._crop_frame(frame_image, reid_bbox)
                if self.anchor_min_sharpness > 0.0:
                    if self._crop_sharpness(crop) < self.anchor_min_sharpness:
                        continue
                try:
                    embedding = self.reid_engine.compute_embedding(crop)
                    self.reid_engine.register_anchor(
                        track_id=tid,
                        embedding=embedding,
                        db_conn=db_conn,
                        timestamp=timestamp_sec,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.info("active_anchor_refresh_failed tid=%s: %s", tid, exc)

        lost_ids = self._active_ids - current_ids
        for lost_id in sorted(lost_ids):
            pipeline_db.mark_track_lost(db_conn, lost_id, timestamp_sec)

        self._known_ids.update(current_ids)
        self._active_ids = set(current_ids)
        yield from self._yield_tracked_rows(tracked=tracked, frame_idx=frame_idx)


def main() -> None:
    """Simple standalone test entrypoint."""
    tracker = PersonTracker()
    _ = list(tracker.update([], frame_idx=0))


if __name__ == "__main__":
    main()

