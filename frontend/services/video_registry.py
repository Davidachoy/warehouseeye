"""Video discovery helpers for the Streamlit demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


@dataclass(frozen=True)
class VideoRecord:
    """Represents one video option shown in the sidebar."""

    video_id: str
    video_path: Path
    timeline_path: Path | None = None

    @property
    def has_timeline(self) -> bool:
        return self.timeline_path is not None and self.timeline_path.exists()


def _candidate_paths(data_dir: Path, video_id: str) -> list[Path]:
    roots = [
        data_dir / "videos_processed",
        data_dir / "videos",
        data_dir / video_id,
        data_dir,
    ]
    names = [
        video_id,
        "video",
        "input",
        "source",
    ]
    candidates: list[Path] = []
    for root in roots:
        for name in names:
            for suffix in VIDEO_EXTENSIONS:
                candidates.append(root / f"{name}{suffix}")
    return candidates


def _find_video_path(data_dir: Path, video_id: str) -> Path | None:
    for candidate in _candidate_paths(data_dir, video_id):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def discover_videos(data_dir: Path) -> list[VideoRecord]:
    """Discover processed videos and fallback local videos for demo safety."""
    records: dict[str, VideoRecord] = {}

    for timeline_path in sorted(data_dir.glob("*/timeline.json")):
        video_id = timeline_path.parent.name
        video_path = _find_video_path(data_dir, video_id)
        if video_path is None:
            continue
        records[video_id] = VideoRecord(
            video_id=video_id,
            video_path=video_path,
            timeline_path=timeline_path,
        )

    # Fallback: include local videos even if they are not preprocessed yet.
    for folder in (data_dir / "videos_processed", data_dir / "videos", data_dir):
        if not folder.exists() or not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            video_id = path.stem
            if video_id in records:
                continue
            records[video_id] = VideoRecord(video_id=video_id, video_path=path)

    return sorted(
        records.values(),
        key=lambda item: (not item.has_timeline, item.video_id.lower()),
    )
