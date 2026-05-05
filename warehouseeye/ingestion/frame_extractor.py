"""Extract key frames using scene cuts plus sparse interval sampling."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract representative frames with PySceneDetect."""

    def extract(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        scene_threshold: float = 0.25,
        sample_every_sec: float = 5.0,
    ) -> list[tuple[str, float]]:
        """Return tuples of (frame_path, timestamp_sec)."""
        video = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=scene_threshold))
        scene_video = open_video(str(video))
        scene_manager.detect_scenes(scene_video)
        scenes = scene_manager.get_scene_list()

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if total_frames > 0 else 0.0
        if not scenes:
            scenes = [(None, None)]

        timestamps: set[float] = set()
        for start, end in scenes:
            start_sec = start.get_seconds() if start else 0.0
            end_sec = end.get_seconds() if end else duration
            timestamps.add(round(start_sec, 3))
            ts = start_sec + sample_every_sec
            while sample_every_sec > 0 and ts < end_sec:
                timestamps.add(round(ts, 3))
                ts += sample_every_sec

        results: list[tuple[str, float]] = []
        for index, ts in enumerate(sorted(timestamps)):
            capture.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            frame_name = f"frame_{index:05d}_{ts:.3f}.jpg"
            frame_path = out_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            results.append((str(frame_path), ts))

        capture.release()
        logger.info("frames_extracted", extra={"count": len(results), "video": str(video)})
        return results


def main() -> None:
    """Simple standalone test entrypoint."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-threshold", type=float, default=0.25)
    parser.add_argument("--sample-every-sec", type=float, default=5.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    FrameExtractor().extract(
        video_path=args.video,
        output_dir=args.output_dir,
        scene_threshold=args.scene_threshold,
        sample_every_sec=args.sample_every_sec,
    )


if __name__ == "__main__":
    main()

