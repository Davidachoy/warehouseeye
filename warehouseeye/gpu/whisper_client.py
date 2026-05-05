"""Whisper ASR client for local GPU inference.

Whisper and vLLM can share a single MI300X process/node, but mixed workloads may
contend through hipBLASLt when launched as isolated multi-process workers.
Recommended operational setup is a single visible device (for example
``HIP_VISIBLE_DEVICES=0``) and one process coordinating both model clients.

The Whisper large-v3-turbo model card also documents ``torch.compile`` speedups,
but those are not compatible with chunked long-form decoding. This client keeps
chunked decoding enabled (`chunk_length_s=30`) as requested.
"""

from __future__ import annotations

import contextlib
import wave
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperClient:
    """Thin wrapper around the transformers ASR pipeline for Whisper turbo."""

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3-turbo",
        chunk_length_s: int = 30,
        batch_size: int = 16,
        device: str = "cuda",
    ) -> None:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            torch_dtype=torch.float16,
            device=device,
        )

    @staticmethod
    def _looks_silent(audio_path: Path, rms_threshold: float = 120.0) -> bool:
        """Cheap silence/noise gate for mono WAV before expensive model inference."""
        with contextlib.closing(wave.open(str(audio_path), "rb")) as wav_file:
            n_frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()
            if n_frames <= 0:
                return True
            if sample_width not in (1, 2, 4):
                return False

            # Sample a bounded amount of PCM data to avoid loading huge files.
            probe_frames = min(n_frames, 16000 * 20)
            raw = wav_file.readframes(probe_frames)
            if not raw:
                return True

            if sample_width == 1:
                values = [abs(byte - 128) for byte in raw]
            elif sample_width == 2:
                values = [
                    abs(int.from_bytes(raw[idx : idx + 2], "little", signed=True))
                    for idx in range(0, len(raw), 2)
                ]
            else:
                values = [
                    abs(int.from_bytes(raw[idx : idx + 4], "little", signed=True))
                    for idx in range(0, len(raw), 4)
                ]
            if not values:
                return True

            return mean(values) < rms_threshold

    @staticmethod
    def _chunk_to_word_entries(chunk: dict[str, Any]) -> list[dict[str, Any]]:
        text = str(chunk.get("text", "")).strip()
        if not text:
            return []

        timestamp = chunk.get("timestamp")
        if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
            return []

        start, end = timestamp
        if start is None or end is None:
            return []
        start_f = float(start)
        end_f = float(end)
        if end_f < start_f:
            return []

        score = chunk.get("score")
        confidence = float(score) if isinstance(score, (int, float)) else None
        words = text.split()
        if not words:
            return []

        # Some decoders can emit phrase chunks even with return_timestamps="word".
        if len(words) == 1:
            return [{"word": words[0], "start": start_f, "end": end_f, "confidence": confidence}]

        duration = max(0.0, end_f - start_f)
        step = duration / len(words) if duration > 0 else 0.0
        entries: list[dict[str, Any]] = []
        for idx, word in enumerate(words):
            word_start = start_f + idx * step
            word_end = start_f + (idx + 1) * step if idx < len(words) - 1 else end_f
            entries.append(
                {
                    "word": word,
                    "start": word_start,
                    "end": word_end,
                    "confidence": confidence,
                }
            )
        return entries

    def transcribe(self, audio_path: str | Path) -> list[dict[str, Any]]:
        """Transcribe audio and return normalized word-level timestamps."""
        path = Path(audio_path)
        if not path.exists() or not path.is_file():
            return []

        # WAV silence check is fast and helps skip useless ASR calls for CCTV noise.
        if path.suffix.lower() == ".wav":
            with contextlib.suppress(Exception):
                if self._looks_silent(path):
                    return []

        try:
            result = self._pipe(
                str(path),
                return_timestamps="word",
                generate_kwargs={"no_speech_threshold": 0.6},
            )
        except Exception:
            return []

        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        words: list[dict[str, Any]] = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                words.extend(self._chunk_to_word_entries(chunk))

        if words:
            return words

        text = result.get("text", "") if isinstance(result, dict) else ""
        return [] if not str(text).strip() else words
