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
        self.chat_completion_paths = (
            ["/chat/completions"]
            if self.base_url.endswith("/v1")
            else ["/v1/chat/completions", "/chat/completions"]
        )

        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.latencies_ms: list[float] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

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

        body = await self._chat_completion_request(payload=payload, request_hint=Path(image_path).name)
        return self._extract_text_content(body)

    async def chat_completion_async(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 300,
    ) -> str:
        """Submit generic chat-completions payload and return text content."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        body = await self._chat_completion_request(payload=payload, request_hint="chat")
        return self._extract_text_content(body)

    async def _chat_completion_request(self, payload: dict[str, Any], request_hint: str) -> dict[str, Any]:
        client = await self._ensure_client()
        self.request_count += 1
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._post_chat_with_fallback(
                    client=client,
                    payload=payload,
                    request_hint=request_hint,
                    attempt=attempt,
                )

                body = response.json()
                usage = body.get("usage", {})
                if isinstance(usage, dict):
                    self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                    self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                    reported_total = int(usage.get("total_tokens", 0) or 0)
                    if reported_total > 0:
                        self.total_tokens += reported_total
                    else:
                        self.total_tokens += int(usage.get("prompt_tokens", 0) or 0) + int(
                            usage.get("completion_tokens", 0) or 0
                        )
                self.success_count += 1
                return body
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                backoff = self.initial_backoff_sec * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code in {429, 500, 502, 503, 504} and attempt < self.max_attempts:
                    backoff = self.initial_backoff_sec * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
                    continue
                # Do not retry deterministic failures like 400/401/403/404.
                break
            except Exception as exc:  # pragma: no cover - hard guard for runtime unknowns.
                last_error = exc
                break

        self.failure_count += 1
        raise RuntimeError("Failed chat completion after retries.") from last_error

    async def _post_chat_with_fallback(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        request_hint: str,
        attempt: int,
    ) -> httpx.Response:
        last_http_error: httpx.HTTPStatusError | None = None
        for path in self.chat_completion_paths:
            started_at = time.perf_counter()
            response = await client.post(path, json=payload)
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            self.latencies_ms.append(elapsed_ms)
            logger.info(
                "vllm_request_timing",
                extra={
                    "attempt": attempt,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "request_hint": request_hint,
                    "status_code": response.status_code,
                    "path": path,
                },
            )
            if response.status_code == 404 and path != self.chat_completion_paths[-1]:
                # Some deployments expose /v1/chat/completions instead of /chat/completions.
                continue
            if response.status_code >= 400:
                last_http_error = httpx.HTTPStatusError(
                    f"HTTP status {response.status_code}",
                    request=response.request,
                    response=response,
                )
                raise last_http_error
            return response
        if last_http_error is not None:
            raise last_http_error
        msg = "Failed to call chat completion endpoint."
        raise RuntimeError(msg)

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
