"""Download videos from local paths, direct URLs, or YouTube."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import yt_dlp

logger = logging.getLogger(__name__)


class VideoDownloader:
    """Downloader wrapper using yt-dlp in programmatic mode."""

    def download(self, url: str, output_path: str | Path) -> Path:
        """Download a video and return the final local path."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(url)

        if parsed.scheme in ("", "file"):
            source = Path(parsed.path or url)
            if not source.exists():
                raise FileNotFoundError(f"Local video not found: {source}")
            if source.resolve() != output.resolve():
                shutil.copy2(source, output)
            logger.info("video_downloaded", extra={"source": str(source), "output": str(output)})
            return output

        if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
            options = {"outtmpl": str(output), "format": "mp4/best", "quiet": True}
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
            logger.info("youtube_downloaded", extra={"url": url, "output": str(output)})
            return output

        urlretrieve(url, str(output))
        logger.info("direct_downloaded", extra={"url": url, "output": str(output)})
        return output


def main() -> None:
    """Simple standalone test entrypoint."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    VideoDownloader().download(args.url, args.output)


if __name__ == "__main__":
    main()

