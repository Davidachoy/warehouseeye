"""OpenAI-compatible vLLM client for image semantic descriptions."""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for vision chat/completions against a vLLM OpenAI endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        profile: str | None = None,
        api_key: str | None = None,
        timeout_sec: float | None = None,
        max_attempts: int = 3,
        initial_backoff_sec: float = 0.5,
        temperature: float = 0.2,
    ) -> None:
        self.base_url = (base_url or os.getenv("AMD_URL", "")).rstrip("/")
        if not self.base_url:
            msg = "Missing AMD_URL for vLLM endpoint."
            raise ValueError(msg)

        self.model_name = model_name or self._resolve_model_name(profile=profile)
        self.api_key = api_key or os.getenv("AMD_API_KEY")
        self.timeout_sec = timeout_sec or float(os.getenv("AMD_TIMEOUT_SEC", "45"))
        self.max_attempts = max_attempts
        self.initial_backoff_sec = initial_backoff_sec
        self.temperature = temperature

        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.latencies_ms: list[float] = []

        self._client: httpx.AsyncClient | None = None

    def _resolve_model_name(self, profile: str | None = None) -> str:
        explicit_model = os.getenv("AMD_MODEL")
        if explicit_model:
            return explicit_model

        active_profile = (profile or os.getenv("AMD_PROFILE", "dev")).lower()
        if active_profile == "prod":
            model = os.getenv("AMD_MODEL_PROD")
            if model:
                return model
        model = os.getenv("AMD_MODEL_DEV")
        if model:
            return model
        msg = "Missing model configuration. Set AMD_MODEL or AMD_MODEL_DEV/AMD_MODEL_PROD."
        raise ValueError(msg)

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = httpx.Timeout(self.timeout_sec)
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout)
        return self._client

    @staticmethod
    def _image_data_url(image_path: str | Path) -> str:
        path = Path(image_path)
        raw_bytes = path.read_bytes()
        encoded = base64.b64encode(raw_bytes).decode("ascii")
        detected_type, _ = mimetypes.guess_type(path.name)
        mime_type = detected_type or "image/jpeg"
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _extract_text_content(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Invalid response payload: missing choices.")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_items = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            joined = "\n".join(part for part in text_items if part).strip()
            if joined:
                return joined
        raise RuntimeError("Invalid response payload: missing choices[0].message.content text.")

    async def describe_image_async(
        self, image_path: str | Path, prompt: str, max_tokens: int = 300
    ) -> str:
        """Describe one image using an OpenAI-compatible vLLM endpoint."""
        image_data_url = self._image_data_url(image_path)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }

        client = await self._ensure_client()
        self.request_count += 1
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            started_at = time.perf_counter()
            try:
                response = await client.post("/chat/completions", json=payload)
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                self.latencies_ms.append(elapsed_ms)
                logger.info(
                    "vllm_request_timing",
                    extra={
                        "attempt": attempt,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "image": Path(image_path).name,
                        "status_code": response.status_code,
                    },
                )

                if response.status_code >= 400:
                    if response.status_code in {429, 500, 502, 503, 504}:
                        raise httpx.HTTPStatusError(
                            f"Transient HTTP status {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()

                body = response.json()
                text = self._extract_text_content(body)
                self.success_count += 1
                return text

            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.HTTPStatusError,
                ValueError,
            ) as exc:
                last_error = exc
                is_last_attempt = attempt >= self.max_attempts
                if is_last_attempt:
                    break
                backoff = self.initial_backoff_sec * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)
            except Exception as exc:  # pragma: no cover - hard guard for runtime unknowns.
                last_error = exc
                break

        self.failure_count += 1
        raise RuntimeError(f"Failed to describe image after retries: {image_path}") from last_error

    def describe_image(self, image_path: str | Path, prompt: str, max_tokens: int = 300) -> str:
        """Synchronous wrapper for image description."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.describe_image_async(image_path=image_path, prompt=prompt, max_tokens=max_tokens))
        msg = "describe_image cannot run inside an active event loop; use describe_image_async."
        raise RuntimeError(msg)

    async def aclose(self) -> None:
        """Close underlying async HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def average_latency_ms(self) -> float:
        """Return mean latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)
