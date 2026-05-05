"""End-to-end orchestrator for WarehouseEye Milestone 1."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from warehouseeye.gpu import VLLMClient, VisionAnalyzer, WhisperClient
from warehouseeye.ingestion.audio_extractor import AudioExtractor
from warehouseeye.ingestion.downloader import VideoDownloader
from warehouseeye.ingestion.frame_extractor import FrameExtractor
from warehouseeye.pipeline.db import init_db, insert_track, upsert_identity
from warehouseeye.pipeline.timeline_builder import TimelineBuilder
from warehouseeye.tracking.color_classifier import classify_color
from warehouseeye.tracking.detector import PersonDetector
from warehouseeye.tracking.tracker import PersonTracker

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinate ingestion, detection, tracking, color tagging, and SQLite writes."""

    def __init__(
        self,
        base_dir: str | Path = "data",
        model_id: str = "PekingU/rtdetr_v2_r50vd",
        scene_threshold: float = 0.25,
        sample_every_sec: float = 5.0,
        detector_threshold: float = 0.5,
        min_bbox_area: float = 0.0,
        tracker_frame_rate: float = 30.0,
        tracker_activation_threshold: float = 0.25,
        tracker_lost_track_buffer: int = 30,
        tracker_matching_threshold: float = 0.8,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.model_id = model_id
        self.scene_threshold = scene_threshold
        self.sample_every_sec = sample_every_sec
        self.detector_threshold = detector_threshold
        self.min_bbox_area = min_bbox_area
        self.tracker_frame_rate = tracker_frame_rate
        self.tracker_activation_threshold = tracker_activation_threshold
        self.tracker_lost_track_buffer = tracker_lost_track_buffer
        self.tracker_matching_threshold = tracker_matching_threshold
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
        return self.run_with_timeline(
            video_url=video_url,
            enable_semantic_and_timeline=False,
        )

    async def _run_semantic_and_audio_parallel(
        self,
        db_path: Path,
        audio_path: Path,
        amd_profile: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        client = VLLMClient(profile=amd_profile)
        analyzer = VisionAnalyzer(vllm_client=client, base_dir=self.base_dir)
        whisper_client = WhisperClient()

        try:
            semantic_task = asyncio.create_task(
                analyzer.analyze_all_tracks_async(
                    db_path=db_path,
                    dry_run=False,
                )
            )
            whisper_task = asyncio.to_thread(whisper_client.transcribe, audio_path)
            semantic_summary, transcript = await asyncio.gather(semantic_task, whisper_task)
            return semantic_summary, transcript
        finally:
            await client.aclose()

    def run_with_timeline(
        self,
        video_url: str,
        enable_semantic_and_timeline: bool = True,
        amd_profile: str = "dev",
    ) -> Path:
        """Run tracking pipeline, optionally parallelize semantics+ASR, then export timeline."""
        started_at = time.time()
        videos_dir = self.base_dir / "videos"
        frames_dir = self.base_dir / "frames"
        audio_dir = self.base_dir / "audio"
        db_path = self.base_dir / "warehouseeye.sqlite3"
        audio_path = audio_dir / "audio.wav"
        videos_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        video_path = self.downloader.download(video_url, videos_dir / "input_video.mp4")
        frame_records = self.frame_extractor.extract(
            video_path,
            frames_dir,
            scene_threshold=self.scene_threshold,
            sample_every_sec=self.sample_every_sec,
        )
        self.audio_extractor.extract(video_path, audio_path)
        conn = init_db(db_path)

        detector = PersonDetector(model_id=self.model_id, threshold=self.detector_threshold)
        tracker = PersonTracker(
            frame_rate=self.tracker_frame_rate,
            track_activation_threshold=self.tracker_activation_threshold,
            lost_track_buffer=self.tracker_lost_track_buffer,
            minimum_matching_threshold=self.tracker_matching_threshold,
        )
        identity_state: dict[int, dict[str, Any]] = {}

        for frame_idx, (frame_path, timestamp_sec) in enumerate(frame_records):
            detections = detector.detect(frame_path)
            if self.min_bbox_area > 0:
                detections = [
                    box
                    for box in detections
                    if (box.x2 - box.x1) * (box.y2 - box.y1) >= self.min_bbox_area
                ]
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

        if enable_semantic_and_timeline:
            semantic_summary, audio_transcript = asyncio.run(
                self._run_semantic_and_audio_parallel(
                    db_path=db_path,
                    audio_path=audio_path,
                    amd_profile=amd_profile,
                )
            )
            timeline_builder = TimelineBuilder()
            timeline_builder.build(db_path=db_path, audio_transcript=audio_transcript)
            timeline_builder.export_json(self.base_dir / "timeline.json")
            timeline_builder.export_summary_text(self.base_dir / "timeline_summary.txt")
            logger.info(
                "timeline_exported",
                extra={
                    "timeline_path": str(self.base_dir / "timeline.json"),
                    "timeline_summary_path": str(self.base_dir / "timeline_summary.txt"),
                    "audio_words": len(audio_transcript),
                    "audio_track_missing_or_silent": len(audio_transcript) == 0,
                    "semantic_tracks": int(semantic_summary.get("analyzed_tracks", 0)),
                },
            )

        return db_path


def main() -> None:
    """Simple standalone test entrypoint."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video-url", required=True)
    parser.add_argument("--base-dir", default="data")
    parser.add_argument("--model-id", default="PekingU/rtdetr_v2_r50vd")
    parser.add_argument(
        "--enable-semantic-and-timeline",
        action="store_true",
        help="Run VLM semantics and Whisper in parallel, then export timeline artifacts.",
    )
    parser.add_argument(
        "--amd-profile",
        choices=["dev", "prod"],
        default="dev",
        help="Select AMD profile for VLLMClient when semantic/timeline mode is enabled.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    Orchestrator(base_dir=args.base_dir, model_id=args.model_id).run_with_timeline(
        args.video_url,
        enable_semantic_and_timeline=args.enable_semantic_and_timeline,
        amd_profile=args.amd_profile,
    )


if __name__ == "__main__":
    main()

