"""Persistent person tracking with supervision ByteTrack."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import supervision as sv

from warehouseeye.tracking.types import BoundingBox


class PersonTracker:
    """Track detections frame by frame using ByteTrack."""

    def __init__(
        self,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ) -> None:
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(
        self, detections_list: Sequence[BoundingBox], frame_idx: int
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
        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else np.array([], dtype=int)

        for idx, tracker_id in enumerate(tracker_ids):
            if tracker_id is None:
                continue
            bbox = tuple(float(v) for v in tracked.xyxy[idx].tolist())
            confidence = float(tracked.confidence[idx]) if tracked.confidence is not None else 0.0
            yield (frame_idx, int(tracker_id), bbox, confidence)


def main() -> None:
    """Simple standalone test entrypoint."""
    tracker = PersonTracker()
    _ = list(tracker.update([], frame_idx=0))


if __name__ == "__main__":
    main()

