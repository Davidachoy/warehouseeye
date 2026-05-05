"""End-to-end orchestrator for WarehouseEye Milestone 1."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from warehouseeye.ingestion.audio_extractor import AudioExtractor
from warehouseeye.ingestion.downloader import VideoDownloader
from warehouseeye.ingestion.frame_extractor import FrameExtractor
from warehouseeye.pipeline.db import init_db, insert_track, upsert_identity
from warehouseeye.tracking.color_classifier import classify_color
from warehouseeye.tracking.detector import PersonDetector
from warehouseeye.tracking.tracker import PersonTracker

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinate ingestion, detection, tracking, color tagging, and SQLite writes."""

    def __init__(self, base_dir: str | Path = "data", model_id: str = "PekingU/rtdetr_v2_r50vd") -> None:
        self.base_dir = Path(base_dir)
        self.model_id = model_id
        self.downloader = VideoDownloader()
        self.frame_extractor = FrameExtractor()
        self.audio_extractor = AudioExtractor()

    def _crop_to_path(
        self,
        image: Image.Image,
        bbox: tuple[float, float, float, float],
        track_id: int,
        frame_idx: int,
    ) -> Path:
        crop_dir = self.base_dir / "crops" / f"track_{track_id:04d}"
        crop_dir.mkdir(parents=True, exist_ok=True)
        x1, y1, x2, y2 = [int(max(0, value)) for value in bbox]
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)
        crop = image.crop((x1, y1, x2, y2))
        crop_path = crop_dir / f"frame_{frame_idx:05d}.jpg"
        crop.save(crop_path)
        return crop_path

    def run(self, video_url: str) -> Path:
        """Run full pipeline for a local path or remote video URL."""
        started_at = time.time()
        videos_dir = self.base_dir / "videos"
        frames_dir = self.base_dir / "frames"
        audio_dir = self.base_dir / "audio"
        db_path = self.base_dir / "warehouseeye.sqlite3"
        videos_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        video_path = self.downloader.download(video_url, videos_dir / "input_video.mp4")
        frame_records = self.frame_extractor.extract(video_path, frames_dir)
        self.audio_extractor.extract(video_path, audio_dir / "audio.wav")
        conn = init_db(db_path)

        detector = PersonDetector(model_id=self.model_id)
        tracker = PersonTracker(frame_rate=30.0)
        identity_state: dict[int, dict[str, Any]] = {}

        for frame_idx, (frame_path, timestamp_sec) in enumerate(frame_records):
            detections = detector.detect(frame_path)
            frame_image = Image.open(frame_path).convert("RGB")

            for out_frame_idx, track_id, bbox, confidence in tracker.update(detections, frame_idx=frame_idx):
                crop_path = self._crop_to_path(frame_image, bbox, track_id, out_frame_idx)
                if track_id not in identity_state:
                    color_tag = classify_color(np.asarray(Image.open(crop_path).convert("RGB")))
                    identity_state[track_id] = {
                        "color_tag": color_tag,
                        "first_seen_sec": timestamp_sec,
                        "last_seen_sec": timestamp_sec,
                        "total_frames": 1,
                        "crop_path": str(crop_path),
                    }
                else:
                    identity_state[track_id]["last_seen_sec"] = timestamp_sec
                    identity_state[track_id]["total_frames"] += 1
                    color_tag = identity_state[track_id]["color_tag"]

                insert_track(
                    conn=conn,
                    track_id=track_id,
                    timestamp_sec=timestamp_sec,
                    frame_idx=out_frame_idx,
                    bbox=bbox,
                    confidence=confidence,
                    color_tag=color_tag,
                    crop_path=str(crop_path),
                    activity_json="{}",
                )

        for track_id, state in identity_state.items():
            upsert_identity(
                conn=conn,
                track_id=track_id,
                color_tag=state["color_tag"],
                first_seen_sec=state["first_seen_sec"],
                last_seen_sec=state["last_seen_sec"],
                total_frames=state["total_frames"],
                narrative_summary=f"Track {track_id}: {state['color_tag']} detected.",
            )

        elapsed = time.time() - started_at
        duration = max((ts for _, ts in frame_records), default=0.0)
        logger.info(
            "pipeline_summary",
            extra={
                "people_detected": len(identity_state),
                "frames_processed": len(frame_records),
                "duration_sec": round(duration, 2),
                "processing_sec": round(elapsed, 2),
                "db_path": str(db_path),
            },
        )
        conn.close()
        return db_path


def main() -> None:
    """Simple standalone test entrypoint."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video-url", required=True)
    parser.add_argument("--base-dir", default="data")
    parser.add_argument("--model-id", default="PekingU/rtdetr_v2_r50vd")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    Orchestrator(base_dir=args.base_dir, model_id=args.model_id).run(args.video_url)


if __name__ == "__main__":
    main()

