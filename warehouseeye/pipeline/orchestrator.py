"""End-to-end orchestrator for WarehouseEye Milestone 1."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from warehouseeye.gpu import EmbeddingClient, OSNetEmbedder, VLLMClient, VisionAnalyzer, WhisperClient
from warehouseeye.ingestion.audio_extractor import AudioExtractor
from warehouseeye.ingestion.downloader import VideoDownloader
from warehouseeye.ingestion.frame_extractor import FrameExtractor
from warehouseeye.pipeline.db import init_db, insert_track, upsert_identity
from warehouseeye.pipeline.timeline_builder import TimelineBuilder
from warehouseeye.tracking.color_classifier import classify_color
from warehouseeye.tracking.detector import PersonDetector
from warehouseeye.tracking.reid import ReIDEngine
from warehouseeye.tracking.tracker import PersonTracker
from warehouseeye.tracking.types import BoundingBox

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float, *, min_value: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            logger.warning("invalid_float_env using_default: %s=%s", name, raw)
            value = default
    if min_value is not None and value < min_value:
        return min_value
    return value


def _env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            logger.warning("invalid_int_env using_default: %s=%s", name, raw)
            value = default
    if min_value is not None and value < min_value:
        return min_value
    return value


class Orchestrator:
    """Coordinate ingestion, detection, tracking, color tagging, and SQLite writes."""

    def __init__(
        self,
        base_dir: str | Path = "data",
        model_id: str = "PekingU/rtdetr_v2_r50vd",
        scene_threshold: float = 0.25,
        sample_every_sec: float = 1.0,
        detector_threshold: float = 0.40,
        detector_input_size: int | None = None,
        min_bbox_area: float = 8000.0,
        tracker_frame_rate: float | None = None,
        tracker_activation_threshold: float = 0.25,
        tracker_lost_track_buffer: int = 8,
        tracker_matching_threshold: float = 0.8,
        min_bbox_width: float = 36.0,
        min_bbox_height: float = 48.0,
        min_bbox_aspect_ratio: float = 0.8,
        max_bbox_aspect_ratio: float = 4.5,
        reid_similarity_threshold: float = 0.9,
        reid_crop_expand_ratio: float = 0.15,
        reid_max_anchors: int = 5,
        reid_anchor_min_distance: float = 0.15,
        reid_aggregation: str = "max",
        reid_topk: int = 3,
        reid_tta_hflip: bool = False,
        reid_anchor_min_sharpness: float = 0.0,
        reid_max_lost_age_sec: float = 300.0,
        reid_active_anchor_refresh_every: int = 0,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.model_id = model_id
        self.scene_threshold = scene_threshold
        self.sample_every_sec = _env_float("WAREHOUSEEYE_SAMPLE_EVERY_SEC", sample_every_sec, min_value=0.1)
        self.detector_threshold = _env_float("WAREHOUSEEYE_DETECTOR_THRESHOLD", detector_threshold, min_value=0.01)
        env_input_size = os.getenv("WAREHOUSEEYE_DETECTOR_INPUT_SIZE")
        if env_input_size is not None and env_input_size.strip():
            try:
                detector_input_size = int(env_input_size)
            except ValueError:
                logger.warning("invalid_int_env using_default: %s=%s", "WAREHOUSEEYE_DETECTOR_INPUT_SIZE", env_input_size)
        self.detector_input_size = detector_input_size if detector_input_size and detector_input_size > 0 else None
        self.min_bbox_area = _env_float("WAREHOUSEEYE_MIN_BBOX_AREA", min_bbox_area, min_value=0.0)
        self.tracker_frame_rate = tracker_frame_rate
        self.tracker_activation_threshold = tracker_activation_threshold
        self.tracker_lost_track_buffer = _env_int(
            "WAREHOUSEEYE_TRACKER_LOST_TRACK_BUFFER",
            tracker_lost_track_buffer,
            min_value=1,
        )
        self.tracker_matching_threshold = tracker_matching_threshold
        self.min_bbox_width = _env_float("WAREHOUSEEYE_MIN_BBOX_WIDTH", min_bbox_width, min_value=1.0)
        self.min_bbox_height = _env_float("WAREHOUSEEYE_MIN_BBOX_HEIGHT", min_bbox_height, min_value=1.0)
        self.min_bbox_aspect_ratio = _env_float(
            "WAREHOUSEEYE_MIN_BBOX_ASPECT_RATIO",
            min_bbox_aspect_ratio,
            min_value=0.1,
        )
        self.max_bbox_aspect_ratio = _env_float(
            "WAREHOUSEEYE_MAX_BBOX_ASPECT_RATIO",
            max_bbox_aspect_ratio,
            min_value=self.min_bbox_aspect_ratio,
        )
        self.reid_similarity_threshold = _env_float(
            "WAREHOUSEEYE_REID_SIMILARITY_THRESHOLD",
            reid_similarity_threshold,
            min_value=0.1,
        )
        # Negative values inset (recortan hacia adentro) to remove background;
        # positive values expand. We do not clamp to >=0 anymore.
        self.reid_crop_expand_ratio = _env_float(
            "WAREHOUSEEYE_REID_CROP_EXPAND_RATIO",
            reid_crop_expand_ratio,
        )
        self.reid_max_anchors = _env_int(
            "WAREHOUSEEYE_REID_MAX_ANCHORS", reid_max_anchors, min_value=1
        )
        self.reid_anchor_min_distance = _env_float(
            "WAREHOUSEEYE_REID_ANCHOR_MIN_DISTANCE",
            reid_anchor_min_distance,
            min_value=0.0,
        )
        agg_raw = os.getenv("WAREHOUSEEYE_REID_AGGREGATION", reid_aggregation).strip().lower()
        if agg_raw not in ("max", "mean_topk"):
            logger.warning("invalid_reid_aggregation using_default: %s", agg_raw)
            agg_raw = reid_aggregation
        self.reid_aggregation = agg_raw
        self.reid_topk = _env_int("WAREHOUSEEYE_REID_TOPK", reid_topk, min_value=1)
        self.reid_tta_hflip = os.getenv(
            "WAREHOUSEEYE_REID_TTA_HFLIP", "1" if reid_tta_hflip else "0"
        ).strip() == "1"
        self.reid_anchor_min_sharpness = _env_float(
            "WAREHOUSEEYE_REID_ANCHOR_MIN_SHARPNESS",
            reid_anchor_min_sharpness,
            min_value=0.0,
        )
        self.reid_max_lost_age_sec = _env_float(
            "WAREHOUSEEYE_REID_MAX_LOST_AGE_SEC",
            reid_max_lost_age_sec,
            min_value=1.0,
        )
        self.reid_active_anchor_refresh_every = _env_int(
            "WAREHOUSEEYE_REID_ACTIVE_ANCHOR_REFRESH_EVERY",
            reid_active_anchor_refresh_every,
            min_value=0,
        )
        self.downloader = VideoDownloader()
        self.frame_extractor = FrameExtractor()
        self.audio_extractor = AudioExtractor()
        self.last_reid_stats: dict[str, float] = {"match_count": 0.0, "average_similarity": 0.0}

    def _is_valid_detection_bbox(self, box: BoundingBox) -> bool:
        width = box.x2 - box.x1
        height = box.y2 - box.y1
        area = width * height
        if area < self.min_bbox_area:
            return False
        if width < self.min_bbox_width or height < self.min_bbox_height:
            return False
        aspect_ratio = height / max(width, 1e-6)
        if aspect_ratio < self.min_bbox_aspect_ratio or aspect_ratio > self.max_bbox_aspect_ratio:
            return False
        return True

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

    def run(self, video_url: str, enable_reid: bool = False) -> Path:
        """Run full pipeline for a local path or remote video URL."""
        return self.run_with_timeline(
            video_url=video_url,
            enable_semantic_and_timeline=False,
            enable_reid=enable_reid,
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
        enable_reid: bool = False,
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

        detector = PersonDetector(
            model_id=self.model_id,
            threshold=self.detector_threshold,
            input_size=self.detector_input_size,
        )
        reid_engine = None
        if enable_reid:
            backend = os.getenv("REID_BACKEND", "qwen").strip().lower()
            embedding_client: Any
            if backend == "osnet":
                embedding_client = OSNetEmbedder()
                logger.info("reid_backend_selected", extra={"backend": "osnet"})
            else:
                embedding_client = EmbeddingClient()
                logger.info("reid_backend_selected", extra={"backend": "qwen"})
            reid_engine = ReIDEngine(
                embedding_client,
                similarity_threshold=self.reid_similarity_threshold,
                max_lost_track_age_sec=self.reid_max_lost_age_sec,
                max_anchors_per_track=self.reid_max_anchors,
                anchor_min_distance=self.reid_anchor_min_distance,
                aggregation=self.reid_aggregation,  # type: ignore[arg-type]
                topk=self.reid_topk,
                tta_hflip=self.reid_tta_hflip,
            )
        tracker_frame_rate = (
            self.tracker_frame_rate
            if self.tracker_frame_rate is not None and self.tracker_frame_rate > 0
            else max(1.0, 1.0 / max(self.sample_every_sec, 1e-6))
        )
        tracker = PersonTracker(
            frame_rate=tracker_frame_rate,
            track_activation_threshold=self.tracker_activation_threshold,
            lost_track_buffer=self.tracker_lost_track_buffer,
            minimum_matching_threshold=self.tracker_matching_threshold,
            reid_engine=reid_engine,
            min_reid_crop_area=self.min_bbox_area,
            min_reid_crop_width=self.min_bbox_width,
            min_reid_crop_height=self.min_bbox_height,
            min_reid_crop_aspect_ratio=self.min_bbox_aspect_ratio,
            max_reid_crop_aspect_ratio=self.max_bbox_aspect_ratio,
            reid_crop_expand_ratio=self.reid_crop_expand_ratio,
            anchor_min_sharpness=self.reid_anchor_min_sharpness,
            video_id=self.base_dir.name,
            active_anchor_refresh_every=self.reid_active_anchor_refresh_every,
        )
        identity_state: dict[int, dict[str, Any]] = {}

        for frame_idx, (frame_path, timestamp_sec) in enumerate(frame_records):
            detections = detector.detect(frame_path)
            detections = [box for box in detections if self._is_valid_detection_bbox(box)]
            frame_image = Image.open(frame_path).convert("RGB")

            for out_frame_idx, track_id, bbox, confidence in tracker.update(
                detections,
                frame_idx=frame_idx,
                frame_image=frame_image if enable_reid else None,
                timestamp_sec=timestamp_sec if enable_reid else None,
                db_conn=conn if enable_reid else None,
            ):
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
                    frame_path=str(frame_path),
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
        if reid_engine is not None:
            self.last_reid_stats = {
                "match_count": float(len(reid_engine.match_similarities)),
                "average_similarity": float(np.mean(reid_engine.match_similarities))
                if reid_engine.match_similarities
                else 0.0,
            }
        else:
            self.last_reid_stats = {"match_count": 0.0, "average_similarity": 0.0}
        logger.info(
            "pipeline_summary",
            extra={
                "people_detected": len(identity_state),
                "frames_processed": len(frame_records),
                "duration_sec": round(duration, 2),
                "processing_sec": round(elapsed, 2),
                "db_path": str(db_path),
                "enable_reid": enable_reid,
                "sample_every_sec": self.sample_every_sec,
                "tracker_frame_rate": tracker_frame_rate,
                "reid_match_count": int(self.last_reid_stats["match_count"]),
                "reid_average_similarity": round(self.last_reid_stats["average_similarity"], 4),
                "reid_similarity_threshold": self.reid_similarity_threshold,
                "reid_crop_expand_ratio": self.reid_crop_expand_ratio,
                "reid_max_anchors": self.reid_max_anchors,
                "reid_anchor_min_distance": self.reid_anchor_min_distance,
                "reid_aggregation": self.reid_aggregation,
                "reid_topk": self.reid_topk,
                "reid_tta_hflip": self.reid_tta_hflip,
                "reid_anchor_min_sharpness": self.reid_anchor_min_sharpness,
                "reid_max_lost_age_sec": self.reid_max_lost_age_sec,
                "reid_active_anchor_refresh_every": self.reid_active_anchor_refresh_every,
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
    parser.add_argument(
        "--enable-reid",
        action="store_true",
        help="Enable persistent Re-ID with Qwen3-VL-Embedding-2B on the embedding endpoint.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    enable_reid = args.enable_reid or os.getenv("ENABLE_REID", "0") == "1"
    Orchestrator(base_dir=args.base_dir, model_id=args.model_id).run_with_timeline(
        args.video_url,
        enable_semantic_and_timeline=args.enable_semantic_and_timeline,
        amd_profile=args.amd_profile,
        enable_reid=enable_reid,
    )


if __name__ == "__main__":
    main()

