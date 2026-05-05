"""Tests for ffmpeg audio extraction wrapper."""

from unittest.mock import MagicMock, patch

from warehouseeye.ingestion.audio_extractor import AudioExtractor


def test_audio_extractor_calls_ffmpeg(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    out = tmp_path / "audio.wav"
    video.write_bytes(b"fake-video")

    pipeline = MagicMock()
    pipeline.output.return_value = pipeline
    pipeline.overwrite_output.return_value = pipeline

    with patch("warehouseeye.ingestion.audio_extractor.ffmpeg.input", return_value=pipeline) as ffmpeg_input:
        result = AudioExtractor().extract(video, out)

    ffmpeg_input.assert_called_once()
    pipeline.run.assert_called_once()
    assert result == out

