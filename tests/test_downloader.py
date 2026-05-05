"""Tests for video downloader local-path handling."""

from warehouseeye.ingestion.downloader import VideoDownloader


def test_download_local_file(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    target = tmp_path / "target.mp4"
    source.write_bytes(b"fake-video")
    result = VideoDownloader().download(str(source), target)
    assert result == target
    assert target.read_bytes() == b"fake-video"

