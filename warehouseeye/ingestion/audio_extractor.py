"""Extract mono 16kHz WAV audio from video files."""

from __future__ import annotations

import logging
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Audio extraction helper using ffmpeg-python."""

    def extract(self, video_path: str | Path, output_path: str | Path) -> Path:
        """Extract WAV audio (16kHz mono) and return output path."""
        video = Path(video_path)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        (
            ffmpeg.input(str(video))
            .output(str(output), acodec="pcm_s16le", ac=1, ar=16000)
            .overwrite_output()
            .run(quiet=True)
        )
        logger.info("audio_extracted", extra={"video": str(video), "output": str(output)})
        return output


def main() -> None:
    """Simple standalone test entrypoint."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    AudioExtractor().extract(args.video, args.output)


if __name__ == "__main__":
    main()

