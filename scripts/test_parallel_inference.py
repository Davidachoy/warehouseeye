"""Concurrency check: Qwen requests + Whisper ASR on one node."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from warehouseeye.gpu import VLLMClient, WhisperClient


def _looks_like_oom(message: str) -> bool:
    text = message.lower()
    return "out of memory" in text or "oom" in text or "hiperroroutofmemory" in text


async def _timed_qwen_request(
    client: VLLMClient,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    request_id: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        await client.describe_image_async(image_path=image_path, prompt=prompt, max_tokens=max_tokens)
        return {
            "request_id": request_id,
            "ok": True,
            "elapsed_sec": time.perf_counter() - started,
            "error": "",
            "oom": False,
        }
    except Exception as exc:
        message = str(exc)
        return {
            "request_id": request_id,
            "ok": False,
            "elapsed_sec": time.perf_counter() - started,
            "error": message,
            "oom": _looks_like_oom(message),
        }


async def _run_qwen_batch(
    client: VLLMClient,
    image_path: Path,
    prompt: str,
    request_count: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    tasks = [
        asyncio.create_task(
            _timed_qwen_request(
                client=client,
                image_path=image_path,
                prompt=prompt,
                max_tokens=max_tokens,
                request_id=index + 1,
            )
        )
        for index in range(request_count)
    ]
    return await asyncio.gather(*tasks)


def _run_whisper(audio_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    whisper = WhisperClient()
    try:
        words = whisper.transcribe(audio_path=audio_path)
        return {
            "ok": True,
            "elapsed_sec": time.perf_counter() - started,
            "word_count": len(words),
            "oom": False,
            "error": "",
        }
    except Exception as exc:
        message = str(exc)
        return {
            "ok": False,
            "elapsed_sec": time.perf_counter() - started,
            "word_count": 0,
            "oom": _looks_like_oom(message),
            "error": message,
        }


async def _run_parallel(
    profile: str,
    image_path: Path,
    audio_path: Path,
    request_count: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    client = VLLMClient(profile=profile)
    started = time.perf_counter()
    try:
        qwen_task = asyncio.create_task(
            _run_qwen_batch(
                client=client,
                image_path=image_path,
                prompt="Describe visible worker activity and anomalies in one sentence.",
                request_count=request_count,
                max_tokens=max_tokens,
            )
        )
        whisper_task = asyncio.create_task(asyncio.to_thread(_run_whisper, audio_path))
        qwen_results, whisper_result = await asyncio.gather(qwen_task, whisper_task)
        return qwen_results, whisper_result, time.perf_counter() - started
    finally:
        await client.aclose()


def _validate(qwen_results: list[dict[str, Any]], whisper_result: dict[str, Any]) -> None:
    qwen_oom = any(result["oom"] for result in qwen_results)
    qwen_failures = [result for result in qwen_results if not result["ok"]]
    whisper_oom = bool(whisper_result["oom"])
    whisper_failed = not bool(whisper_result["ok"])
    if qwen_oom or whisper_oom:
        raise RuntimeError("OOM detected while running parallel Qwen + Whisper inference.")
    if qwen_failures:
        raise RuntimeError(f"{len(qwen_failures)} Qwen requests failed during parallel run.")
    if whisper_failed:
        raise RuntimeError(f"Whisper failed during parallel run: {whisper_result['error']}")

    durations = [float(result["elapsed_sec"]) for result in qwen_results]
    if len(durations) >= 3:
        median_sec = statistics.median(durations)
        max_sec = max(durations)
        if median_sec > 0 and max_sec / median_sec > 4.0:
            raise RuntimeError(
                "Significant Qwen degradation detected (max request latency > 4x median latency)."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Qwen + Whisper single-node inference test.")
    parser.add_argument("--image-path", required=True, help="Image path used for Qwen requests.")
    parser.add_argument("--audio-path", required=True, help="Audio path used for Whisper transcription.")
    parser.add_argument("--qwen-requests", type=int, default=8, help="Concurrent Qwen request count.")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens per Qwen request.")
    parser.add_argument("--profile", choices=["dev", "prod"], default="dev", help="AMD profile.")
    args = parser.parse_args()

    load_dotenv()
    image_path = Path(args.image_path)
    audio_path = Path(args.audio_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    qwen_results, whisper_result, total_elapsed_sec = asyncio.run(
        _run_parallel(
            profile=args.profile,
            image_path=image_path,
            audio_path=audio_path,
            request_count=args.qwen_requests,
            max_tokens=args.max_tokens,
        )
    )
    _validate(qwen_results, whisper_result)

    durations = [float(result["elapsed_sec"]) for result in qwen_results]
    print("Parallel inference completed successfully.")
    print(f"Total wall time: {total_elapsed_sec:.2f}s")
    print(f"Qwen requests: {len(qwen_results)}")
    print(f"Qwen mean latency: {statistics.fmean(durations):.2f}s")
    print(f"Qwen median latency: {statistics.median(durations):.2f}s")
    print(f"Qwen max latency: {max(durations):.2f}s")
    print(f"Whisper latency: {float(whisper_result['elapsed_sec']):.2f}s")
    print(f"Whisper words: {int(whisper_result['word_count'])}")


if __name__ == "__main__":
    main()
