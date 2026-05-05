"""Tests for key-frame extraction."""

import cv2
import numpy as np

from warehouseeye.ingestion.frame_extractor import FrameExtractor


def test_extract_frames_from_short_video(tmp_path) -> None:
    video_path = tmp_path / "clip.mp4"
    output_dir = tmp_path / "frames"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64))
    for idx in range(15):
        frame = np.full((64, 64, 3), idx * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    rows = FrameExtractor().extract(video_path, output_dir, scene_threshold=5.0, sample_every_sec=1.0)
    assert len(rows) >= 1
    assert output_dir.exists()

